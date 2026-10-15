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
    assert len(calls) == 1  # Empty inputs cannot support a scene intervention.
    assert metrics["generation"]["sample_mean_mse_normalized"] == 0
    assert metrics["generation"]["oracle_best_of_k_mse_normalized"] == 0
    assert metrics["generation"]["sample_variance_normalized"] == 0
    assert metrics["uncalibrated"]["scalar_elements"] == 2 * 5 * 65
    assert metrics["scene_interventions"]["status"] == "unsupported_no_nonempty_scenes"
    assert metrics["scene_interventions"]["empty_scene_response_mse_normalized"] is None
    assert metrics["scene_interventions"]["shuffled_scene_response_mse_normalized"] is None
    assert metrics["scene_interventions"]["nonempty_windows"] == 0
    assert metrics["attention_example"] is None
    assert metrics["attention_example_status"] == "unsupported_no_nonempty_scenes"
    assert arrays["valid"].sum() == 10


def calibration_fixture(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_json(data / "manifest.json", {"id": "tiny-test-data"})
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"test checkpoint identity; load is mocked")
    checkpoint = {"dataset_manifest_sha256": sha256(data / "manifest.json"), "model_config": {"fps": 1.}}
    monkeypatch.setattr(evaluate, "load_model", lambda path, device: (object(), checkpoint))
    monkeypatch.setattr(evaluate, "MixedSceneDataset", lambda *args, **kwargs: [1, 2])
    arrays = {"error": np.full((2, 3, 1), 2.), "variance": np.ones((2, 3, 1)),
              "valid": np.ones((2, 3), bool), "prediction": np.full((2, 3, 1), 2.),
              "target": np.zeros((2, 3, 1)), "scene_present": np.array([True, False])}

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
    assert set(metrics["calibrated_by_horizon"]) == {"0-1s", "1-2s"}
    assert metrics["calibrated_by_horizon"]["1-2s"]["marginal_coverage_95"] == 1
    assert metrics["calibrated_by_scene_presence"]["archived_scene"]["windows"] == 1
    assert metrics["calibrated_by_scene_presence"]["empty_scene"]["mean_std_normalized"] == 2


class SceneDependentModel:
    def __call__(self, history, scene, goal, **kwargs):
        shape = (len(history), kwargs["horizon"], 65)
        value = scene[..., 0].masked_fill(~kwargs["scene_mask"], 0).sum(1)
        return {"forecast_mean": value[:, None, None].expand(shape), "logvar": torch.zeros(shape),
                "scene_attention": torch.zeros(len(history), kwargs["horizon"], scene.shape[1] + 1)}


def scene_row(value=0, present=False):
    row = {"history": torch.zeros(6, 65), "future": torch.zeros(3, 65),
           "scene": torch.zeros(2, 9), "scene_mask": torch.tensor([present, False]),
           "goal": torch.zeros(4), "future_mask": torch.tensor([True, True, False])}
    row["scene"][0, 0] = value
    return row


def evaluation_checkpoint():
    return {"normalizer": {"mean": [0.] * 65, "std": [1.] * 65},
            "config": {"mode": "forecast"}, "model_config": {"fps": 1.}}


def test_scene_deltas_count_only_nonempty_changed_inputs():
    rows = [scene_row(), scene_row(2, True), scene_row()]
    metrics, _ = evaluate.collect(SceneDependentModel(), evaluation_checkpoint(), rows,
                                  torch.device("cpu"), batch_size=3)
    result = metrics["scene_interventions"]
    assert result["status"] == "supported"
    assert result["nonempty_windows"] == 1
    assert result["empty_scene_response_windows"] == 1
    assert result["shuffled_scene_response_windows"] == 1
    assert result["empty_scene_response_mse_normalized"] == 4  # Not diluted by two empty scenes.
    assert result["shuffled_scene_response_mse_normalized"] == 4
    assert metrics["attention_example"]["scene_mask"] == [True, False]
    assert metrics["attention_example"]["scene"][0][0] == 2
    assert metrics["scene_windows"] == {"nonempty": 1, "empty": 2}


def test_identical_active_scenes_do_not_support_shuffle_diagnostic():
    rows = [scene_row(2, True), scene_row(2, True)]
    rows[1]["scene"][1] = 123  # Padded values are not a changed effective scene.
    metrics, _ = evaluate.collect(SceneDependentModel(), evaluation_checkpoint(), rows,
                                  torch.device("cpu"), batch_size=2)
    result = metrics["scene_interventions"]
    assert result["empty_scene_response_windows"] == 2
    assert result["shuffled_scene_response_windows"] == 0
    assert result["shuffled_scene_response_mse_normalized"] is None
    assert result["shuffle_status"] == "unsupported_no_changed_nonempty_inputs"


def test_point_baselines_use_observed_history_train_mean_and_equal_groups():
    class RecordedRows(list):
        records = [{"group_id": group} for group in ("a", "b", "b")]

    rows = RecordedRows()
    for last in (4, 8, 8):
        row = scene_row()
        row["history"][:] = last
        row["future"][0] = last + 2
        row["future"][1] = last + 4
        row["future"][2] = 1e8  # Invalid targets must not affect a baseline.
        rows.append(row)
    checkpoint = evaluation_checkpoint()
    checkpoint["normalizer"] = {"mean": [2.] * 65, "std": [2.] * 65}
    metrics, _ = evaluate.collect(SceneDependentModel(), checkpoint, rows, torch.device("cpu"), batch_size=3)
    persistence = metrics["baselines"]["last_observed_state"]
    constant = metrics["baselines"]["train_normalizer_mean"]
    assert persistence["mse_normalized"] == pytest.approx(2.5)
    assert persistence["physical_rmse"]["height"]["rmse"] == pytest.approx(math.sqrt(10))
    assert persistence["by_horizon"]["0-1s"]["mse_normalized"] == 1
    assert persistence["by_horizon"]["1-2s"]["mse_normalized"] == 4
    assert constant["mse_normalized"] == pytest.approx((6.5 + 20.5 + 20.5) / 3)
    assert constant["equal_group_mean_mse_normalized"] == pytest.approx((6.5 + 20.5) / 2)
    assert constant["by_group"]["b"]["windows"] == 2
    assert "marginal_coverage_95" not in persistence
    assert "gaussian_nll_normalized" not in constant


@pytest.mark.parametrize("limit,expected_limit,available,full", [(None, None, 2, True), (2, 2, 5, False)])
def test_evaluation_default_uses_full_split_and_records_scope(tmp_path, monkeypatch, limit, expected_limit, available, full):
    data, checkpoint_path = calibration_fixture(tmp_path, monkeypatch)
    requested = []

    class Dataset(list):
        manifest = {"counts": {"val": available}}

    def dataset(*args, **kwargs):
        requested.append(kwargs["max_samples"])
        return Dataset([1, 2])

    monkeypatch.setattr(evaluate, "MixedSceneDataset", dataset)
    output = tmp_path / "evaluation"
    arguments = ["evaluate", "--checkpoint", str(checkpoint_path), "--data", str(data),
                 "--output", str(output), "--threads", "1"]
    if limit is not None:
        arguments += ["--max-samples", str(limit)]
    monkeypatch.setattr(sys, "argv", arguments)
    evaluate.main()
    assert requested == [expected_limit]
    scope = json.loads((output / "metrics.json").read_text())["evaluation_scope"]
    assert scope["available_windows"] == available
    assert scope["evaluated_windows"] == 2
    assert scope["full_split"] is full


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
