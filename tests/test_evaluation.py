"""Offline evaluation math, padding, and report provenance checks."""

import json
import math
import sys
import tarfile

import numpy as np
import pytest
import torch

from motion_scene_pretrain import evaluate, report
from motion_scene_pretrain.common import sha256


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_gaussian_residual_metrics_are_marginal_and_correct():
    error = np.array([[[0.], [1.], [2.], [float("nan")]]])
    variance = np.array([[[1.], [1.], [1.], [-1.]]])
    valid = np.array([[True, True, True, False]])
    metrics = evaluate.score_prediction(error, variance, valid)
    assert metrics["scalar_elements"] == 3
    assert metrics["mse_normalized"] == pytest.approx(5 / 3)
    assert metrics["gaussian_nll_normalized"] == pytest.approx(.5 * math.log(2 * math.pi) + 5 / 6)
    assert metrics["marginal_coverage_68"] == pytest.approx(2 / 3)
    assert metrics["marginal_coverage_95"] == pytest.approx(2 / 3)
    scaled = evaluate.score_prediction(error, variance * 4, valid)
    assert scaled["marginal_coverage_95"] == 1
    assert scaled["mean_std_normalized"] == 2


@pytest.mark.parametrize("bad_variance", [0., -1., float("nan"), float("inf")])
def test_invalid_selected_variance_is_rejected(bad_variance):
    with pytest.raises(ValueError, match="variance|finite|positive"):
        evaluate.score_prediction(np.zeros((1, 1, 1)),
                                  np.full((1, 1, 1), bad_variance), np.ones((1, 1), bool))


def test_empty_evaluation_mask_is_rejected():
    with pytest.raises(ValueError, match="valid|empty|element"):
        evaluate.score_prediction(np.zeros((1, 1, 1)), np.ones((1, 1, 1)), np.zeros((1, 1), bool))


def test_collect_excludes_padding_from_forecasts_and_generation(monkeypatch):
    rows = [{
        "history": torch.zeros(6, 65), "future": torch.zeros(10, 65),
        "scene": torch.zeros(3, 9), "scene_mask": torch.zeros(3, dtype=torch.bool),
        "goal": torch.zeros(4), "future_mask": torch.tensor([True] * 5 + [False] * 5),
    } for _ in range(2)]
    calls = []

    class Model:
        def __call__(self, history, scene, goal, **kwargs):
            assert "future_mask" in kwargs
            calls.append(kwargs)
            shape = (len(history), kwargs["horizon"], 65)
            return {"forecast_mean": torch.zeros(shape), "logvar": torch.zeros(shape),
                    "scene_attention": torch.zeros(len(history), kwargs["horizon"], scene.shape[1] + 1)}

    counter = 0

    def padded_spread(model, batch, steps):
        nonlocal counter
        counter += 1
        output = torch.zeros_like(batch["future"])
        output[~batch["future_mask"]] = 100 if counter % 2 else -100
        return output

    monkeypatch.setattr(evaluate, "sample_flow", padded_spread)
    checkpoint = {"normalizer": {"mean": [0.] * 65, "std": [1.] * 65},
                  "config": {"mode": "flow"}, "model_config": {"fps": 50.}}
    metrics, arrays = evaluate.collect(Model(), checkpoint, rows, torch.device("cpu"),
                                       batch_size=2, samples=2)
    assert len(calls) == 3
    assert metrics["generation"]["sample_mean_mse_normalized"] == 0
    assert metrics["generation"]["oracle_best_of_k_mse_normalized"] == 0
    assert metrics["generation"]["sample_variance_normalized"] == 0
    assert metrics["uncalibrated"]["scalar_elements"] == 2 * 5 * 65
    assert metrics["scene_interventions"]["empty_scene_response_mse_normalized"] == 0
    assert arrays["valid"].sum() == 10


def calibration_fixture(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_json(data / "manifest.json", {"id": "tiny-test-data"})
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"test checkpoint identity; load is mocked")
    checkpoint = {"dataset_manifest_sha256": sha256(data / "manifest.json")}
    monkeypatch.setattr(evaluate, "load_model", lambda path, device: (object(), checkpoint))
    monkeypatch.setattr(evaluate, "MixedSceneDataset", lambda *args, **kwargs: [1, 2])
    arrays = {"error": np.full((2, 3, 1), 2.), "variance": np.ones((2, 3, 1)),
              "valid": np.ones((2, 3), bool), "prediction": np.full((2, 3, 1), 2.),
              "target": np.zeros((2, 3, 1))}

    def fake_collect(*args, **kwargs):
        return {"uncalibrated": evaluate.score_prediction(arrays["error"], arrays["variance"], arrays["valid"])}, arrays

    monkeypatch.setattr(evaluate, "collect", fake_collect)
    return data, checkpoint_path


def test_calibration_cli_fits_variance_scale_and_applies_to_other_split(tmp_path, monkeypatch):
    data, checkpoint_path = calibration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "calibration"
    common = ["evaluate", "--checkpoint", str(checkpoint_path), "--data", str(data), "--threads", "1"]
    monkeypatch.setattr(sys, "argv", common + ["--output", str(output), "--split", "calibration", "--calibrate"])
    evaluate.main()
    calibration = json.loads((output / "calibration.json").read_text())
    assert calibration["variance_scale"] == 4
    assert calibration["fit_split"] == "calibration"
    assert calibration["checkpoint_sha256"] == sha256(checkpoint_path)
    applied = tmp_path / "test-evaluation"
    monkeypatch.setattr(sys, "argv", common + ["--output", str(applied), "--split", "test",
                                               "--calibration", str(output / "calibration.json")])
    evaluate.main()
    metrics = json.loads((applied / "metrics.json").read_text())
    assert metrics["uncalibrated"]["marginal_coverage_95"] == 0
    assert metrics["calibrated"]["marginal_coverage_95"] == 1
    assert metrics["calibrated"]["mean_std_normalized"] == 2


def test_calibration_rejects_other_checkpoint_and_records_failure(tmp_path, monkeypatch):
    data, checkpoint_path = calibration_fixture(tmp_path, monkeypatch)
    calibration_path = tmp_path / "wrong-calibration.json"
    write_json(calibration_path, {"fit_split": "calibration", "checkpoint_sha256": "other",
                                 "dataset_manifest_sha256": sha256(data / "manifest.json"),
                                 "variance_scale": 4})
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint_path),
                                     "--data", str(data), "--output", str(output), "--split", "test",
                                     "--calibration", str(calibration_path), "--threads", "1"])
    with pytest.raises(ValueError, match="binding"):
        evaluate.main()
    assert json.loads((output / "receipt.json").read_text())["status"] == "failed"


def make_report_inputs(tmp_path):
    run = tmp_path / "run"
    evaluation = tmp_path / "evaluation"
    run.mkdir()
    evaluation.mkdir()
    receipt = {"status": "complete", "checkpoint_sha256": "checkpoint-a",
               "dataset_manifest_sha256": "dataset-a"}
    write_json(run / "receipt.json", receipt)
    write_json(run / "config.json", {"mode": "forecast"})
    write_json(run / "normalizer.json", {"fit_split": "train"})
    (run / "metrics.jsonl").write_text('{"step": 1, "loss": 2.0}\n')
    write_json(evaluation / "receipt.json", receipt)
    write_json(evaluation / "metrics.json", {"checkpoint_sha256": "checkpoint-a",
                                             "dataset_manifest_sha256": "dataset-a", "split": "test"})
    return run, evaluation


@pytest.mark.parametrize("wrong_field", ["checkpoint_sha256", "dataset_manifest_sha256"])
def test_report_rejects_mismatched_evaluation_provenance(tmp_path, monkeypatch, wrong_field):
    run, evaluation = make_report_inputs(tmp_path)
    receipt = json.loads((evaluation / "receipt.json").read_text())
    receipt[wrong_field] = "wrong-identity"
    write_json(evaluation / "receipt.json", receipt)
    monkeypatch.setattr(sys, "argv", ["report", "--run", str(run), "--evaluations", str(evaluation),
                                     "--output", str(tmp_path / "report")])
    with pytest.raises(ValueError, match="checkpoint|dataset|binding"):
        report.main()


def test_report_binds_metrics_to_receipt_and_emits_metadata_only_bundle(tmp_path, monkeypatch):
    run, evaluation = make_report_inputs(tmp_path)
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["report", "--run", str(run), "--evaluations", str(evaluation),
                                     "--output", str(output)])
    report.main()
    with tarfile.open(output / "return-report.tar.gz", "r:gz") as bundle:
        names = bundle.getnames()
    assert "summary.json" in names and "files.sha256.json" in names
    assert "training/metrics.jsonl" in names and "evaluation-0/metrics.json" in names
    assert not any(name.endswith((".pt", ".npz", ".npy")) for name in names)
    metrics = json.loads((evaluation / "metrics.json").read_text())
    metrics["checkpoint_sha256"] = "a-different-checkpoint"
    write_json(evaluation / "metrics.json", metrics)
    monkeypatch.setattr(sys, "argv", ["report", "--run", str(run), "--evaluations", str(evaluation),
                                     "--output", str(tmp_path / "wrong-report")])
    with pytest.raises(ValueError, match="checkpoint|binding|metrics"):
        report.main()
