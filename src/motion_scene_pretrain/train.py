"""Bounded non-RL training with train-only normalization and portable receipts."""
from __future__ import annotations
import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import (checkpoint_load, collate, dump_json, environment, fit_normalizer,
                     masked_mean, prepare, sha256)
from .data import MixedSceneDataset
from .model import ModelConfig, MotionSceneModel


def objective(model, batch, mode, mask_probability=.4, reconstruction_weight=.2,
              forecast_weight=.1):
    history, future = batch["history"], batch["future"]
    if mode not in {"masked", "forecast", "flow"}:
        raise ValueError(f"Unknown mode: {mode}")
    if reconstruction_weight < 0 or forecast_weight < 0:
        raise ValueError("Objective weights must be nonnegative")
    kwargs = dict(horizon=future.shape[1], scene_mask=batch["scene_mask"],
                  future_mask=batch["future_mask"])
    rec = history.new_zeros(())
    metrics = {}
    if mode == "masked" or reconstruction_weight > 0:
        if history.shape[1] < 2 or not 0 < mask_probability < 1:
            raise ValueError("Masked pretraining needs history >=2 and mask_probability in (0,1)")
        # Contiguous spans limit the adjacent-position/velocity shortcut; keep the anchor.
        length = max(1, min(history.shape[1] - 1, round(history.shape[1] * mask_probability)))
        starts = torch.randint(0, history.shape[1] - length, (len(history),), device=history.device)
        indices = torch.arange(history.shape[1], device=history.device)[None]
        observed = ~((indices >= starts[:, None]) & (indices < starts[:, None] + length))
        # Forecast/flow context stays fully observed. Masking is a separate pass.
        rec_out = model(history, batch["scene"], batch["goal"],
                        history_observed=observed, **kwargs)
        rec = masked_mean((rec_out["history_reconstruction"] - history)**2, ~observed)
        metrics["masked_mse"] = float(rec.detach())
    if mode == "masked":
        return rec, metrics
    if mode == "flow":
        t = torch.rand(len(future), device=future.device)
        noise = torch.randn_like(future)
        xt = (1 - t[:, None, None]) * future + t[:, None, None] * noise
        flow_out = model(history, batch["scene"], batch["goal"], future_input=xt,
                         flow_time=t, **kwargs)
        fm = masked_mean((flow_out["flow_velocity"] - (noise - future))**2,
                         batch["future_mask"])
    out = model(history, batch["scene"], batch["goal"], **kwargs)
    error2 = (out["forecast_mean"] - future)**2
    nll = masked_mean(.5 * (out["logvar"] + error2 * torch.exp(-out["logvar"])),
                      batch["future_mask"])
    mse = masked_mean(error2, batch["future_mask"])
    # Small MSE stabilizer prevents initial variance inflation from hiding a bad mean.
    forecast = nll + .05 * mse
    total = forecast if mode == "forecast" else fm + forecast_weight * forecast
    total = total + reconstruction_weight * rec
    metrics.update(forecast_nll_no_constant=float(nll.detach()),
                   forecast_mse_normalized=float(mse.detach()))
    if mode == "flow":
        metrics["flow_mse"] = float(fm.detach())
    return total, metrics


def sampling_weights(records, mode="uniform_window", scene_fraction=.25):
    """Normalized row probabilities; scene_group fixes global scene exposure first.

    group_scene preserves the original scheme: equal ancestry mass, then equal
    present/empty mass *within* each ancestry. scene_group instead assigns the
    requested global present/empty mass, then equal ancestry mass inside each
    presence category, then equal row mass inside each ancestry/category pair.
    """
    if not records:
        raise ValueError("Cannot sample an empty training dataset")
    pairs = [(r["group_id"], r["scene_id"] != "empty") for r in records]
    counts = Counter(pairs)
    if mode == "uniform_window":
        weights = [1.] * len(records)
    elif mode == "group_scene":
        kinds = Counter(group for group, _ in counts)
        weights = [1 / (counts[(group, present)] * kinds[group]) for group, present in pairs]
    elif mode == "scene_group":
        if not np.isfinite(scene_fraction) or not 0 <= scene_fraction <= 1:
            raise ValueError("scene_fraction must be finite and in [0,1]")
        groups = Counter(present for _, present in counts)
        mass = {True: float(scene_fraction), False: 1 - float(scene_fraction)}
        if any(mass[present] > 0 and not groups[present] for present in (False, True)):
            raise ValueError("Requested scene_fraction needs unavailable scene/empty training rows")
        weights = [mass[present] / (groups[present] * counts[(group, present)])
                   for group, present in pairs]
    else:
        raise ValueError("sampling must be uniform_window, group_scene, or scene_group")
    total = sum(weights)
    return [weight / total for weight in weights]


def _sampling_summary(records, weights):
    presence, groups = Counter(), Counter()
    for row, weight in zip(records, weights):
        presence["scene" if row["scene_id"] != "empty" else "empty"] += weight
        groups[row["group_id"]] += weight
    return {"expected_scene_probability": {name: float(presence[name]) for name in ("scene", "empty")},
            "expected_group_probability": dict(sorted(groups.items()))}


def _collate_with_metadata(rows):
    batch = collate(rows)
    batch["_metadata"] = [row["metadata"] for row in rows]
    return batch


def _prepared_batch(raw, normalizer, device):
    return prepare({key: value for key, value in raw.items() if key != "_metadata"}, normalizer, device)


@torch.no_grad()
def validate(model, loader, normalizer, device, mode, max_batches=10,
             mask_probability=.4, reconstruction_weight=.2, forecast_weight=.1):
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive or None for full validation")
    was_training = model.training
    model.eval()
    totals, denominators, presence = Counter(), Counter(), Counter()
    windows = future_frames = batches = 0
    groups = set()
    # Fork RNG makes validation reproducible without altering training's random stream.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(1729)
        for i, raw in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _prepared_batch(raw, normalizer, device)
            loss, parts = objective(model, batch, mode, mask_probability,
                                    reconstruction_weight, forecast_weight)
            count = len(batch["history"])
            valid_frames = int(raw["future_mask"].sum())
            hidden_frames = max(1, min(batch["history"].shape[1] - 1,
                                       round(batch["history"].shape[1] * mask_probability)))
            for key, value in parts.items():
                # Each constituent has its own denominator; combining batch means
                # overweights short final batches and partially padded futures.
                elements = ((count * hidden_frames) if key == "masked_mse" else valid_frames) * batch["future"].shape[-1]
                totals[key] += value * elements
                denominators[key] += elements
            windows += count
            future_frames += valid_frames
            batches += 1
            if "_metadata" in raw:
                for row in raw["_metadata"]:
                    groups.add(row["group_id"])
                    presence["scene" if row["scene_id"] != "empty" else "empty"] += 1
            else:
                present = int(raw["scene_mask"].any(-1).sum())
                presence.update(scene=present, empty=count - present)
    model.train(was_training)
    if not windows:
        return {}
    result = {key: totals[key] / max(denominators[key], 1) for key in totals}
    if mode == "masked":
        result["loss"] = result["masked_mse"]
    else:
        forecast = result["forecast_nll_no_constant"] + .05 * result["forecast_mse_normalized"]
        result["loss"] = forecast if mode == "forecast" else result["flow_mse"] + forecast_weight * forecast
        result["loss"] += reconstruction_weight * result.get("masked_mse", 0.)
    result.update(windows=windows, valid_future_frames=future_frames, batches=batches,
                  scene_windows=presence["scene"], empty_windows=presence["empty"])
    if groups:
        result["ancestry_groups"] = len(groups)
    return result


def run(args):
    cfg = json.loads(Path(args.config).read_text())
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if cfg["steps"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("steps and batch_size must be positive")
    if cfg.get("log_every", 100) < 1 or cfg.get("eval_every", 1000) < 1:
        raise ValueError("log_every and eval_every must be positive")
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = MixedSceneDataset(args.data, "train", max_samples=cfg.get("max_train_samples"))
    val = MixedSceneDataset(args.data, "val", max_samples=cfg.get("max_val_samples", 1024))
    sample = data[0]
    model_cfg = ModelConfig(**cfg.get("model", {}))
    expected = {"state_dim": data.manifest["state_dim"], "scene_dim": data.manifest["scene_dim"],
                "goal_dim": data.manifest["goal_dim"], "fps": data.manifest["fps"]}
    if any(getattr(model_cfg, key) != value for key, value in expected.items()):
        raise ValueError("Model dimensions/fps disagree with the dataset manifest")
    model = MotionSceneModel(model_cfg).to(device)
    sampling = cfg.get("sampling", "uniform_window")
    weights = sampling_weights(data.records, sampling, cfg.get("scene_fraction", .25))
    sampling_plan = {"mode": sampling, **_sampling_summary(data.records, weights)}
    if sampling == "scene_group":
        sampling_plan["requested_scene_fraction"] = cfg.get("scene_fraction", .25)
    horizon = len(sample["future"])
    parameter_count = sum(p.numel() for p in model.parameters())
    plan = {"mode": cfg["mode"], "device": str(device), "parameters": parameter_count,
            "train_windows": len(data), "val_windows": len(val),
            "history_frames": len(sample["history"]), "future_frames": horizon,
            "steps": cfg["steps"], "batch_size": cfg["batch_size"],
            "drop_last": bool(cfg.get("drop_last", False)),
            "nominal_sample_presentations": cfg["steps"] * cfg["batch_size"],
            "validation_max_batches": cfg.get("val_batches", 10),
            "sampling": sampling_plan,
            "note": "No RL or simulator. VRAM and throughput must be measured on the target machine."}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(args.data) / "manifest.json"
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    receipt = {"status": "running", "config": cfg, "plan": plan,
               "environment": environment(), "code_sha256": source_hashes,
               "dataset_manifest_sha256": sha256(manifest_path),
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "sampling": sampling_plan,
               "initialization": {"kind": "warm_start" if args.init else "fresh",
                                  "optimizer": "new_adamw", "exact_resume": False,
                                  "sampler_rng": "new_stream_from_config_seed"}}
    dump_json(out / "receipt.json", receipt)
    dump_json(out / "config.json", cfg)
    if args.init:
        ckpt = checkpoint_load(args.init)
        if ckpt["model_config"] != asdict(model_cfg):
            raise ValueError("Warm-start model configuration differs; use matching architecture/settings")
        if ckpt["dataset_manifest_sha256"] != receipt["dataset_manifest_sha256"]:
            raise ValueError("Warm-start dataset differs; use a documented transfer experiment instead")
        model.load_state_dict(ckpt["model"])
        normalizer = ckpt["normalizer"]
        receipt["initialized_from_sha256"] = sha256(args.init)
        receipt["initialization"].update(parent_checkpoint_sha256=receipt["initialized_from_sha256"],
                                         parent_mode=ckpt["config"]["mode"], parent_step=ckpt["step"])
    else:
        normalizer = fit_normalizer(data)
    dump_json(out / "normalizer.json", normalizer)
    sampler = None
    if sampling != "uniform_window":
        sampler = WeightedRandomSampler(weights, len(data), replacement=True,
                                         generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=sampler is None,
                        sampler=sampler,
                        num_workers=cfg.get("workers", 0), collate_fn=_collate_with_metadata,
                        generator=torch.Generator().manual_seed(seed),
                        drop_last=cfg.get("drop_last", False))
    if not len(loader):
        raise ValueError("Training loader is empty; reduce batch_size or disable drop_last")
    val_loader = DataLoader(val, batch_size=cfg["batch_size"], collate_fn=_collate_with_metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                                  weight_decay=cfg.get("weight_decay", .01))
    model.train()
    iterator = iter(loader)
    started = time.monotonic()
    completed_steps = windows_seen = history_frames_seen = future_frames_seen = 0
    scene_counts, group_counts, group_scene_counts = Counter(), Counter(), Counter()

    def exposure():
        return {"windows": windows_seen, "history_frames": history_frames_seen,
                "valid_future_frames": future_frames_seen,
                "scene_counts": {name: scene_counts[name] for name in ("scene", "empty")},
                "group_counts": dict(sorted(group_counts.items())),
                "group_scene_counts": {group: {name: group_scene_counts[(group, name)]
                                                for name in ("scene", "empty")}
                                       for group in sorted(group_counts)}}

    dump_json(out / "receipt.json", receipt)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with (out / "metrics.jsonl").open("w") as log:
            for step in range(1, cfg["steps"] + 1):
                try:
                    raw = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    raw = next(iterator)
                batch = _prepared_batch(raw, normalizer, device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = objective(model, batch, cfg["mode"],
                                        cfg.get("mask_probability", .4),
                                        cfg.get("reconstruction_weight", .2),
                                        cfg.get("forecast_weight", .1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.))
                if not torch.isfinite(grad):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                completed_steps += 1
                windows_seen += len(raw["history"])
                history_frames_seen += raw["history"].shape[0] * raw["history"].shape[1]
                future_frames_seen += int(raw["future_mask"].sum())
                for metadata in raw["_metadata"]:
                    kind = "scene" if metadata["scene_id"] != "empty" else "empty"
                    scene_counts[kind] += 1
                    group_counts[metadata["group_id"]] += 1
                    group_scene_counts[(metadata["group_id"], kind)] += 1
                should_validate = step % cfg.get("eval_every", 1000) == 0 or step == cfg["steps"]
                if step == 1 or step % cfg.get("log_every", 100) == 0 or should_validate:
                    row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad),
                           "sample_presentations": windows_seen,
                           "history_frame_presentations": history_frames_seen,
                           "valid_future_frame_presentations": future_frames_seen,
                           "scene_presentations": scene_counts["scene"],
                           "empty_presentations": scene_counts["empty"], **parts}
                    if should_validate:
                        row["validation"] = validate(model, val_loader, normalizer, device,
                                                      cfg["mode"], cfg.get("val_batches", 10),
                                                      cfg.get("mask_probability", .4),
                                                      cfg.get("reconstruction_weight", .2),
                                                      cfg.get("forecast_weight", .1))
                    row["elapsed_seconds"] = time.monotonic() - started
                    row["windows_per_second"] = windows_seen / max(row["elapsed_seconds"], 1e-9)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
        loop_elapsed = time.monotonic() - started
        # Fixed-budget last checkpoint: no post-hoc selection on test or calibration.
        checkpoint = {"model": model.cpu().state_dict(), "model_config": asdict(model_cfg),
                      "normalizer": normalizer, "config": cfg, "step": cfg["steps"],
                      "dataset_manifest_sha256": receipt["dataset_manifest_sha256"],
                      "history_frames": len(sample["history"]), "horizon": horizon}
        torch.save(checkpoint, out / "checkpoint.pt")
        receipt.update(status="complete", elapsed_seconds=time.monotonic() - started,
                       training_loop_elapsed_seconds=loop_elapsed,
                       completed_optimizer_steps=completed_steps,
                       actual_sample_presentations=windows_seen,
                       presentations=exposure(),
                       windows_per_second=windows_seen / max(loop_elapsed, 1e-9),
                       optimizer_steps_per_second=completed_steps / max(loop_elapsed, 1e-9),
                       throughput_scope="Training loop including validation; excludes normalization and final checkpoint writing",
                       checkpoint_sha256=sha256(out / "checkpoint.pt"),
                       max_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                 if device.type == "cuda" else 0),
                       max_cuda_reserved_bytes=(torch.cuda.max_memory_reserved(device)
                                                if device.type == "cuda" else 0))
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       completed_optimizer_steps=completed_steps, presentations=exposure(),
                       actual_sample_presentations=windows_seen,
                       elapsed_seconds=time.monotonic() - started)
        dump_json(out / "receipt.json", receipt)
        raise
    dump_json(out / "receipt.json", receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--init", help="Warm start; new optimizer and run, not an exact resume")
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--dry-run", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
