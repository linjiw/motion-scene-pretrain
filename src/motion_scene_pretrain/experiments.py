"""Controlled, sequential Colab experiment suites with per-stage Drive handoff."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .common import dump_json, sha256
from .data import verify_dataset

CORE_ARMS = ("blind", "scene", "masked_scene")
SCREEN_ARMS = (*CORE_ARMS, "original_sampler", "medium")


def make_plan(suite="screen", seeds=None, batch_size=64, device="cuda", arms=None):
    """Freeze budgets before results; staged arms include their pretraining cost."""
    if suite not in {"smoke", "screen", "confirm", "generative"}:
        raise ValueError("Unknown suite")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device == "cpu" and suite != "smoke":
        raise ValueError("CPU is supported only for the bounded smoke suite")
    seeds = list(seeds if seeds is not None else ([0, 1, 2] if suite == "confirm" else [0]))
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Provide unique nonnegative seeds")
    allowed = ("flow", "masked_flow") if suite == "generative" else SCREEN_ARMS
    selected = list(arms or (CORE_ARMS if suite in {"smoke", "confirm"} else allowed))
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(allowed):
        raise ValueError(f"Arms must be a unique subset of {allowed}")
    total_steps = {"smoke": 8, "screen": 2000, "confirm": 6000, "generative": 2000}[suite]
    records = []
    for seed in seeds:
        for arm in selected:
            stages = []
            staged = arm.startswith("masked_")
            downstream = "flow" if arm.endswith("flow") else "forecast"
            recipe = [("masked", total_steps // 4), (downstream, total_steps * 3 // 4)] if staged else [(downstream, total_steps)]
            for index, (mode, steps) in enumerate(recipe, 1):
                width, heads, layers = (256, 8, 4) if arm == "medium" else (128, 4, 3)
                cfg = {
                    "mode": mode, "device": device, "seed": seed, "steps": steps,
                    "batch_size": batch_size, "lr": .0003,
                    "sampling": "group_scene" if arm == "original_sampler" else "scene_group",
                    "scene_fraction": .25, "mask_probability": .4,
                    "reconstruction_weight": 0., "forecast_weight": .1,
                    "cpu_threads": 2, "workers": 0, "drop_last": True, "log_every": 100,
                    "eval_every": 500, "val_batches": None, "max_val_samples": None,
                    "model": {"width": width, "heads": heads, "layers": layers,
                              "fps": 50., "scene_enabled": arm != "blind"},
                }
                if suite == "smoke":
                    cfg.update(batch_size=min(batch_size, 4),
                               max_val_samples=16, log_every=2, eval_every=4)
                    cfg["model"].update(width=32, heads=4, layers=1)
                stages.append({"name": f"{index:02d}-{mode}", "config": cfg})
            records.append({"arm": arm, "seed": seed, "name": f"{arm}-seed{seed}", "stages": stages})
    return {
        "suite": suite, "seeds": seeds, "arms": selected, "runs": records,
        "steps_per_arm": total_steps, "checkpoint_selection": "fixed-budget final checkpoint",
        "evaluation_splits": ["calibration", "val"], "test_split_used": False,
        "note": "Equal optimizer budgets do not imply equal FLOPs; report all stages and measured costs. Fresh v2 models only.",
    }


def checked_stage(folder, config, manifest_hash, parent_checkpoint, source_hashes):
    """Only reuse a complete, bound stage; an interrupted optimizer is not resumable."""
    receipt_path = folder / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    checkpoint = folder / "checkpoint.pt"
    parent_hash = sha256(parent_checkpoint) if parent_checkpoint else None
    if (receipt.get("status") != "complete" or receipt.get("config") != config
            or receipt.get("dataset_manifest_sha256") != manifest_hash
            or receipt.get("initialized_from_sha256") != parent_hash
            or receipt.get("code_sha256") != source_hashes
            or not checkpoint.is_file() or receipt.get("checkpoint_sha256") != sha256(checkpoint)):
        raise ValueError(f"Incomplete or different stage: {folder}. Use a new suite output directory; no exact resume is implemented.")
    return receipt


def bind_backup(backup, root, plan):
    """Claim an empty backup or verify its existing plan before writing any files."""
    backup, root = Path(backup).resolve(), Path(root).resolve()
    if backup == root or backup.is_relative_to(root) or root.is_relative_to(backup):
        raise ValueError("Backup and local output must be separate directory trees")
    backup.mkdir(parents=True, exist_ok=True)
    binding = backup / "suite.json"
    if binding.exists():
        if json.loads(binding.read_text()) != plan:
            raise ValueError("Backup belongs to a different suite, code, or dataset; choose a fresh backup directory")
    elif any(backup.iterdir()):
        raise ValueError("Nonempty backup has no suite binding; choose a fresh backup directory")
    else:
        dump_json(binding, plan)
    return backup


def save_backup_tree(folder, root, backup):
    """Do not replace a completed backup stage/evaluation with another execution."""
    target = backup / folder.relative_to(root)
    for receipt_path in folder.rglob("receipt.json"):
        destination = target / receipt_path.relative_to(folder)
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing.get("status") == "complete" and sha256(destination) != sha256(receipt_path):
                raise ValueError(f"Completed backup artifact would be overwritten: {destination}; use a fresh backup directory")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(folder, target, dirs_exist_ok=True)


def require_scene_holdout(manifest, scene_val_groups, suite):
    if suite == "smoke":
        return
    derivation = manifest.get("scene_holdout_derivation", {})
    if (not derivation or derivation.get("version") != 1
            or derivation.get("parent_checkpoints_allowed") is not False
            or not derivation.get("parent_manifest_sha256")):
        raise ValueError("A registered scene_holdout_derivation with parent_checkpoints_allowed=false is required")
    if len(scene_val_groups) < 2:
        raise ValueError("Prepare the versioned scene holdout first; at least two scene-bearing validation groups are required")


def checked_evaluation(folder, expected_args, checkpoint, manifest_hash, split_counts,
                       calibration_path=None, *, create_binding=False):
    """Bind completed evaluations to protocol, scope, metrics, and calibration bytes."""
    folder = Path(folder)
    message = f"Incomplete or mismatched evaluation at {folder}; use a new suite output directory"
    receipt_path, metrics_path = folder / "receipt.json", folder / "metrics.json"
    if not receipt_path.is_file() or not metrics_path.is_file():
        raise ValueError(message)
    receipt = json.loads(receipt_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    checkpoint_hash = sha256(checkpoint)
    split = expected_args["split"]
    maximum = expected_args["max_samples"]

    def scope_for(name):
        available = split_counts[name]
        evaluated = min(maximum, available) if maximum else available
        return {"available_windows": available, "evaluated_windows": evaluated,
                "full_split": evaluated == available, "max_samples": maximum}

    scope = scope_for(split)
    if (receipt.get("status") != "complete" or receipt.get("checkpoint_sha256") != checkpoint_hash
            or receipt.get("dataset_manifest_sha256") != manifest_hash
            or any(receipt.get("args", {}).get(key) != value for key, value in expected_args.items())
            or metrics.get("checkpoint_sha256") != checkpoint_hash
            or metrics.get("dataset_manifest_sha256") != manifest_hash
            or metrics.get("split") != split or metrics.get("evaluation_seed") != expected_args["seed"]
            or metrics.get("evaluation_scope") != scope
            or metrics.get("windows") != scope["evaluated_windows"]):
        raise ValueError(message)
    for name in ("uncalibrated", "calibrated"):
        for key, value in metrics.get(name, {}).items():
            if key.startswith("marginal_coverage_") and (not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(message + ": invalid coverage")

    fitted_path = folder / "calibration.json" if split == "calibration" else None
    used_calibration = fitted_path or calibration_path
    if used_calibration is None or not Path(used_calibration).is_file():
        raise ValueError(message + ": missing calibration")
    calibration = json.loads(Path(used_calibration).read_text())
    scale = calibration.get("variance_scale")
    if (calibration.get("fit_split") != "calibration"
            or calibration.get("checkpoint_sha256") != checkpoint_hash
            or calibration.get("dataset_manifest_sha256") != manifest_hash
            or calibration.get("evaluation_scope") != scope_for("calibration")
            or calibration.get("windows") != scope_for("calibration")["evaluated_windows"]
            or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
        raise ValueError(message + ": calibration binding/scope mismatch")
    if split == "val" and ("calibrated" not in metrics or metrics.get("calibration_variance_scale") != scale):
        raise ValueError(message + ": applied calibration mismatch")
    binding = {"receipt_sha256": sha256(receipt_path), "metrics_sha256": sha256(metrics_path),
               "calibration_sha256": sha256(used_calibration), "checkpoint_sha256": checkpoint_hash,
               "dataset_manifest_sha256": manifest_hash}
    binding_path = folder / "evaluation-binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError(message + ": evaluation/calibration content changed")
    elif create_binding:
        dump_json(binding_path, binding)
    else:
        raise ValueError(message + ": missing evaluation content binding")
    return metrics


def summarize_run(run, receipts, metrics):
    scene = metrics.get("by_scene_presence", {}).get("archived_scene", {})
    baseline = metrics.get("baselines", {})
    return {
        "arm": run["arm"], "seed": run["seed"],
        "total_steps": sum(r["config"]["steps"] for r in receipts),
        "total_training_seconds": sum(r["elapsed_seconds"] for r in receipts),
        "total_presentations": sum(r.get("actual_sample_presentations", r["plan"]["nominal_sample_presentations"]) for r in receipts),
        "parameters": receipts[-1]["plan"]["parameters"],
        "peak_cuda_allocated_gib": max(r["max_cuda_allocated_bytes"] for r in receipts) / 2**30,
        "validation_windows": metrics["windows"], "validation_groups": metrics["ancestry_groups"],
        "validation_scene_windows": scene.get("windows", 0),
        "validation_scope": metrics.get("evaluation_scope", {}),
        "mse": metrics["uncalibrated"]["mse_normalized"],
        "equal_group_mse": metrics["equal_group_mean_mse_normalized"],
        "scene_mse": scene.get("mse_normalized"),
        "nll": metrics["uncalibrated"]["gaussian_nll_normalized"],
        "coverage95_raw": metrics["uncalibrated"]["marginal_coverage_95"],
        "coverage95_calibrated": metrics["calibrated"]["marginal_coverage_95"],
        "point_baselines": baseline,
        "by_group": metrics["by_group"], "physical_rmse": metrics["physical_rmse"],
        "generation": metrics.get("generation"),
    }


def write_comparison(root, results, suite="screen"):
    dump_json(root / "comparison.json", results)
    columns = ["arm", "seed", "total_steps", "total_training_seconds", "total_presentations",
               "parameters", "peak_cuda_allocated_gib", "validation_windows", "validation_groups",
               "validation_scene_windows", "mse", "equal_group_mse", "scene_mse", "nll",
               "coverage95_raw", "coverage95_calibrated"]
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(r[k]))}</td>" for k in columns) + "</tr>" for r in results)
    headers = "".join(f"<th>{c}</th>" for c in columns)
    scope_text = ("Smoke check with at most 16 validation windows; test unused. " if suite == "smoke"
                  else "Fresh scene-holdout v2 models. Full validation; test unused. ")
    (root / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Motion-scene experiment comparison</title>'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}</style>'
        '<h1>Fixed-budget comparison</h1><p>' + scope_text +
        'Scene sensitivity and reference prediction do not establish collision avoidance. '
        'One-seed screens are exploratory. Pretraining cost is included.</p>'
        f'<table><tr>{headers}</tr>{rows}</table>')


def run_suite(args):
    data, root = Path(args.data).resolve(), Path(args.output).resolve()
    verify_dataset(data)
    manifest = json.loads((data / "manifest.json").read_text())
    records = [json.loads(line) for line in (data / "samples.jsonl").read_text().splitlines()]
    scene_val_groups = {r["group_id"] for r in records if r["split"] == "val" and r["scene_id"] != "empty"}
    require_scene_holdout(manifest, scene_val_groups, args.suite)
    plan = make_plan(args.suite, args.seeds, args.batch_size, args.device, args.arms)
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    plan.update(dataset_manifest_sha256=sha256(data / "manifest.json"), code_sha256=source_hashes,
                scene_validation_groups=len(scene_val_groups))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    if (root / "suite.json").exists():
        if json.loads((root / "suite.json").read_text()) != plan:
            raise ValueError("Existing suite has a different plan, code, or dataset. Choose a fresh output directory.")
    elif any(root.iterdir()):
        raise ValueError("Output directory is not an existing matching suite or an empty directory")
    dump_json(root / "suite.json", plan)
    shutil.copyfile(data / "manifest.json", root / "dataset-manifest.json")
    backup = Path(args.backup).resolve() if args.backup else None
    if backup is not None:
        backup = bind_backup(backup, root, plan)

    def save(folder):
        if backup is not None:
            save_backup_tree(folder, root, backup)

    def command(module, *arguments):
        subprocess.run([sys.executable, "-m", "motion_scene_pretrain." + module,
                        *[str(x) for x in arguments]], check=True)

    results = []
    started = time.monotonic()
    for run in plan["runs"]:
        run_root = root / run["name"]
        run_root.mkdir(exist_ok=True)
        previous = None
        receipts = []
        for stage in run["stages"]:
            output = run_root / stage["name"]
            config = run_root / (stage["name"] + ".json")
            dump_json(config, stage["config"])
            if not output.exists():
                init = ["--init", previous] if previous else []
                try:
                    command("train", "--config", config, "--data", data, "--output", output, *init)
                finally:
                    if output.exists():
                        save(output)
            receipt = checked_stage(output, stage["config"], plan["dataset_manifest_sha256"], previous, source_hashes)
            receipts.append(receipt)
            previous = output / "checkpoint.pt"
        for split in ("calibration", "val"):
            destination = run_root / split
            calibration_path = run_root / "calibration/calibration.json" if split == "val" else None
            expected_args = {
                "checkpoint": str(previous), "data": str(data), "device": args.device,
                "batch_size": min(args.batch_size, 32),
                "max_samples": 16 if args.suite == "smoke" else 0,
                "split": split, "seed": 0, "output": str(destination),
                "samples": ((2 if args.suite == "smoke" else 4) if split == "val" else 0),
                "flow_steps": 4 if args.suite == "smoke" else 16,
                "calibrate": split == "calibration",
                "calibration": str(calibration_path) if calibration_path else None,
            }
            evaluation_args = ["--checkpoint", previous, "--data", data, "--device", args.device,
                               "--batch-size", min(args.batch_size, 32), "--max-samples", 16 if args.suite == "smoke" else 0,
                               "--split", split, "--seed", 0, "--output", destination,
                               "--samples", (2 if args.suite == "smoke" else 4) if split == "val" else 0,
                               "--flow-steps", 4 if args.suite == "smoke" else 16]
            if split == "calibration":
                evaluation_args += ["--calibrate"]
            else:
                evaluation_args += ["--calibration", calibration_path]
            fresh_evaluation = not destination.exists()
            if fresh_evaluation:
                command("evaluate", *evaluation_args)
            metrics = checked_evaluation(destination, expected_args, previous,
                                         plan["dataset_manifest_sha256"], manifest["counts"],
                                         calibration_path, create_binding=fresh_evaluation)
            save(destination)
        results.append(summarize_run(run, receipts, metrics))
        write_comparison(root, results, args.suite)
        save(run_root)
        if backup:
            for name in ("suite.json", "dataset-manifest.json", "comparison.json", "comparison.csv", "comparison.html"):
                shutil.copyfile(root / name, backup / name)
        print(f"Completed {run['name']}: group MSE={results[-1]['equal_group_mse']:.4f}", flush=True)
    dump_json(root / "completion.json", {"status": "complete", "runs": len(results),
              "this_invocation_elapsed_seconds": time.monotonic() - started,
              "note": "Invocation includes evaluation, subprocess setup, and backup; stage training times are separate."})
    allowed = {"suite.json", "dataset-manifest.json", "completion.json", "comparison.json", "comparison.csv",
               "comparison.html", "receipt.json", "config.json", "normalizer.json", "metrics.jsonl",
               "metrics.json", "calibration.json", "evaluation-binding.json"}
    archive = root / "a100-return-report.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name in allowed:
                tar.add(path, arcname=path.relative_to(root))
    if backup:
        shutil.copyfile(archive, backup / archive.name)
        shutil.copyfile(root / "completion.json", backup / "completion.json")
    print(f"Return report: {archive}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backup", help="Dedicated Drive folder for this suite, including completed checkpoints")
    p.add_argument("--suite", choices=["smoke", "screen", "confirm", "generative"], default="screen")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    run_suite(p.parse_args())


if __name__ == "__main__":
    main()
