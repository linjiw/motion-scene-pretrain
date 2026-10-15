"""Scientific budget, data binding, and completed-stage reuse contracts."""
import json
from pathlib import Path

import pytest

from motion_scene_pretrain.common import sha256
from motion_scene_pretrain.experiments import (
    bind_backup, checked_evaluation, checked_stage, make_plan, require_scene_holdout,
    save_backup_tree, summarize_run, write_comparison,
)


def test_screen_freezes_equal_total_exposure_and_only_declared_controls():
    plan = make_plan()
    assert len(plan["runs"]) == 5
    for run in plan["runs"]:
        configs = [stage["config"] for stage in run["stages"]]
        assert sum(c["steps"] for c in configs) == 2000
        assert sum(c["steps"] * c["batch_size"] for c in configs) == 128000
        assert all(c["drop_last"] and c["max_val_samples"] is None and c["val_batches"] is None for c in configs)
        assert all(c["reconstruction_weight"] == 0 for c in configs)
    configs = {r["arm"]: r["stages"][-1]["config"] for r in plan["runs"]}
    blind = json.loads(json.dumps(configs["blind"]))
    blind["model"]["scene_enabled"] = True
    assert blind == configs["scene"]
    sampling_control = dict(configs["original_sampler"], sampling="scene_group")
    assert sampling_control == configs["scene"]
    assert plan["evaluation_splits"] == ["calibration", "val"]
    assert plan["test_split_used"] is False


def test_confirmation_and_generation_budgets():
    confirmation = make_plan("confirm")
    assert confirmation["seeds"] == [0, 1, 2]
    assert len(confirmation["runs"]) == 9
    assert all(sum(s["config"]["steps"] for s in r["stages"]) == 6000 for r in confirmation["runs"])
    generation = make_plan("generative")
    assert generation["arms"] == ["flow", "masked_flow"]
    assert all(r["stages"][-1]["config"]["mode"] == "flow" for r in generation["runs"])
    assert make_plan("smoke", device="cpu")["steps_per_arm"] == 8


@pytest.mark.parametrize("kwargs", [{"seeds": [0, 0]}, {"arms": ["nonexistent"]},
                                  {"batch_size": 0}, {"device": "cpu"}])
def test_invalid_plans_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        make_plan(**kwargs)


def test_stage_reuse_requires_config_data_code_and_parent_binding(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent fixture")
    cfg = {"mode": "forecast"}
    source_hashes = {"train.py": "code"}
    receipt = {"status": "complete", "config": cfg, "dataset_manifest_sha256": "dataset",
               "initialized_from_sha256": sha256(parent), "code_sha256": source_hashes,
               "checkpoint_sha256": sha256(checkpoint)}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    assert checked_stage(tmp_path, cfg, "dataset", parent, source_hashes) == receipt
    for config, data, previous, code in [({}, "dataset", parent, source_hashes),
                                         (cfg, "new-dataset", parent, source_hashes),
                                         (cfg, "dataset", None, source_hashes),
                                         (cfg, "dataset", parent, {})]:
        with pytest.raises(ValueError, match="different stage"):
            checked_stage(tmp_path, config, data, previous, code)
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        checked_stage(tmp_path, cfg, "dataset", parent, source_hashes)


def test_comparison_includes_pretraining_cost_and_actual_presentations():
    stage = {"config": {"steps": 2}, "elapsed_seconds": 3., "actual_sample_presentations": 8,
             "plan": {"parameters": 100, "nominal_sample_presentations": 10}, "max_cuda_allocated_bytes": 0}
    metrics = {"windows": 5, "ancestry_groups": 2, "by_group": {}, "physical_rmse": {},
               "equal_group_mean_mse_normalized": .5,
               "uncalibrated": {"mse_normalized": .6, "gaussian_nll_normalized": 1., "marginal_coverage_95": .9},
               "calibrated": {"marginal_coverage_95": .96},
               "baselines": {"last_observed_state": {"mse_normalized": .8}}}
    result = summarize_run({"arm": "masked_scene", "seed": 0}, [stage, stage], metrics)
    assert result["total_steps"] == 4 and result["total_training_seconds"] == 6
    assert result["total_presentations"] == 16
    assert result["point_baselines"] == metrics["baselines"]
    assert result["scene_mse"] is None and result["validation_scene_windows"] == 0


def test_backup_binding_precedes_copy_and_rejects_collisions(tmp_path):
    root = tmp_path / "local"
    root.mkdir()
    backup = tmp_path / "backup"
    plan = {"suite": "screen", "dataset_manifest_sha256": "dataset-a"}
    assert bind_backup(backup, root, plan) == backup
    assert json.loads((backup / "suite.json").read_text()) == plan
    bind_backup(backup, root, plan)
    with pytest.raises(ValueError, match="different suite"):
        bind_backup(backup, root, dict(plan, dataset_manifest_sha256="dataset-b"))
    assert json.loads((backup / "suite.json").read_text()) == plan
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "important.txt").write_text("preserve this")
    with pytest.raises(ValueError, match="Nonempty backup"):
        bind_backup(unrelated, root, plan)
    assert (unrelated / "important.txt").read_text() == "preserve this"
    assert not (unrelated / "suite.json").exists()
    for overlapping in (root, root / "backup", tmp_path):
        with pytest.raises(ValueError, match="separate directory"):
            bind_backup(overlapping, root, plan)


def test_completed_backup_cannot_be_replaced_by_different_execution(tmp_path):
    root = tmp_path / "local"
    stage = root / "scene-seed0/01-forecast"
    stage.mkdir(parents=True)
    receipt = stage / "receipt.json"
    receipt.write_text(json.dumps({"status": "complete", "elapsed_seconds": 1.}))
    backup = bind_backup(tmp_path / "backup", root, {"suite": "screen"})
    save_backup_tree(stage, root, backup)
    save_backup_tree(stage, root, backup)  # Identical completed stage is reusable.
    receipt.write_text(json.dumps({"status": "complete", "elapsed_seconds": 2.}))
    with pytest.raises(ValueError, match="Completed backup"):
        save_backup_tree(stage, root, backup)
    saved = backup / stage.relative_to(root) / "receipt.json"
    assert json.loads(saved.read_text())["elapsed_seconds"] == 1.


def test_full_suites_require_registered_holdout_and_forbid_parent_checkpoints():
    groups = {"scene-a", "scene-b"}
    valid = {"scene_holdout_derivation": {"version": 1, "parent_manifest_sha256": "parent",
                                           "parent_checkpoints_allowed": False}}
    require_scene_holdout(valid, groups, "screen")
    require_scene_holdout({}, set(), "smoke")
    for manifest in ({}, {"scene_holdout_derivation": dict(valid["scene_holdout_derivation"],
                                                          parent_checkpoints_allowed=True)},
                     {"scene_holdout_derivation": dict(valid["scene_holdout_derivation"], version=2)}):
        with pytest.raises(ValueError, match="scene_holdout_derivation"):
            require_scene_holdout(manifest, groups, "screen")
    with pytest.raises(ValueError, match="at least two"):
        require_scene_holdout(valid, {"scene-a"}, "confirm")


def evaluation_fixture(tmp_path, split="val"):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    calibration_folder = tmp_path / "calibration"
    calibration_folder.mkdir()
    folder = tmp_path / split
    folder.mkdir(exist_ok=True)
    counts = {"calibration": 7, "val": 9}

    def scope(name):
        return {"available_windows": counts[name], "evaluated_windows": counts[name],
                "full_split": True, "max_samples": 0}

    calibration_path = calibration_folder / "calibration.json"
    calibration = {"fit_split": "calibration", "checkpoint_sha256": sha256(checkpoint),
                   "dataset_manifest_sha256": "dataset", "variance_scale": 2.,
                   "windows": counts["calibration"], "evaluation_scope": scope("calibration")}
    calibration_path.write_text(json.dumps(calibration))
    args = {"split": split, "max_samples": 0, "seed": 0, "samples": 4 if split == "val" else 0,
            "flow_steps": 16, "calibrate": split == "calibration",
            "calibration": str(calibration_path) if split == "val" else None}
    receipt = {"status": "complete", "checkpoint_sha256": sha256(checkpoint),
               "dataset_manifest_sha256": "dataset", "args": args}
    metrics = {"checkpoint_sha256": sha256(checkpoint), "dataset_manifest_sha256": "dataset",
               "split": split, "evaluation_seed": 0, "windows": counts[split],
               "evaluation_scope": scope(split), "uncalibrated": {"marginal_coverage_95": .9}}
    if split == "val":
        metrics.update(calibrated={"marginal_coverage_95": .95}, calibration_variance_scale=2.)
    (folder / "receipt.json").write_text(json.dumps(receipt))
    (folder / "metrics.json").write_text(json.dumps(metrics))
    return folder, args, checkpoint, counts, calibration_path, metrics


@pytest.mark.parametrize("split", ["calibration", "val"])
def test_evaluation_reuse_requires_content_binding_and_matching_protocol(tmp_path, split):
    folder, args, checkpoint, counts, calibration, metrics = evaluation_fixture(tmp_path, split)
    with pytest.raises(ValueError, match="missing evaluation content binding"):
        checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration)
    actual = checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration, create_binding=True)
    assert actual == metrics
    assert checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration) == metrics
    receipt_path = folder / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["args"]["seed"] = 17
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="mismatched evaluation"):
        checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration)


@pytest.mark.parametrize("mutation", ["prefix", "coverage", "checkpoint", "split"])
def test_mismatched_evaluation_scope_and_coverage_are_rejected(tmp_path, mutation):
    folder, args, checkpoint, counts, calibration, metrics = evaluation_fixture(tmp_path)
    if mutation == "prefix":
        metrics["evaluation_scope"].update(evaluated_windows=4, full_split=False, max_samples=4)
        metrics["windows"] = 4
    elif mutation == "coverage":
        metrics["calibrated"]["marginal_coverage_95"] = 1.1
    elif mutation == "checkpoint":
        metrics["checkpoint_sha256"] = "another-model"
    else:
        metrics["split"] = "calibration"
    (folder / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="mismatched evaluation"):
        checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration, create_binding=True)


@pytest.mark.parametrize("mutation", ["checkpoint", "prefix", "scale", "missing"])
def test_wrong_calibration_binding_scope_or_applied_scale_is_rejected(tmp_path, mutation):
    folder, args, checkpoint, counts, calibration_path, _ = evaluation_fixture(tmp_path)
    calibration = json.loads(calibration_path.read_text())
    if mutation == "checkpoint":
        calibration["checkpoint_sha256"] = "another-model"
    elif mutation == "prefix":
        calibration["evaluation_scope"].update(evaluated_windows=3, full_split=False)
    elif mutation == "scale":
        calibration["variance_scale"] = 4.
    calibration_path.write_text(json.dumps(calibration))
    if mutation == "missing":
        calibration_path.unlink()
    with pytest.raises(ValueError, match="calibration"):
        checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration_path, create_binding=True)


@pytest.mark.parametrize("changed_file", ["metrics", "calibration"])
def test_previously_bound_evaluation_detects_changed_bytes(tmp_path, changed_file):
    folder, args, checkpoint, counts, calibration, metrics = evaluation_fixture(tmp_path)
    checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration, create_binding=True)
    if changed_file == "metrics":
        metrics["calibrated"]["marginal_coverage_95"] = .94  # Plausible value, but not the bound result.
        (folder / "metrics.json").write_text(json.dumps(metrics))
    else:
        calibration.write_text(calibration.read_text() + "\n")
    with pytest.raises(ValueError, match="content changed"):
        checked_evaluation(folder, args, checkpoint, "dataset", counts, calibration)


def test_smoke_comparison_does_not_claim_full_validation(tmp_path):
    write_comparison(tmp_path, [], "smoke")
    text = (tmp_path / "comparison.html").read_text()
    assert "at most 16 validation windows" in text
    assert "Full validation" not in text
