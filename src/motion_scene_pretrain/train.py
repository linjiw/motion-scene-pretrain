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
    if history.shape[1] < 2 or not 0 < mask_probability < 1:
        raise ValueError("Masked pretraining needs history >=2 and mask_probability in (0,1)")
    # Contiguous spans limit the trivial adjacent-position/velocity shortcut.
    # Keep the newest observed anchor visible; boundary velocities can still help.
    length = max(1, min(history.shape[1] - 1, round(history.shape[1] * mask_probability)))
    starts = torch.randint(0, history.shape[1] - length, (len(history),), device=history.device)
    indices = torch.arange(history.shape[1], device=history.device)[None]
    observed = ~((indices >= starts[:, None]) & (indices < starts[:, None] + length))
    kwargs = dict(horizon=future.shape[1], scene_mask=batch["scene_mask"],
                  future_mask=batch["future_mask"])
    # Forecast/flow context stays fully observed. Masking is a separate training pass.
    rec_out = model(history, batch["scene"], batch["goal"],
                    history_observed=observed, **kwargs)
    rec = masked_mean((rec_out["history_reconstruction"] - history)**2, ~observed)
    if mode == "masked":
        return rec, {"masked_mse": float(rec.detach())}
    if mode not in {"forecast", "flow"}:
        raise ValueError(f"Unknown mode: {mode}")
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
    metrics = {"masked_mse": float(rec.detach()), "forecast_nll_no_constant": float(nll.detach()),
               "forecast_mse_normalized": float(mse.detach())}
    if mode == "flow":
        metrics["flow_mse"] = float(fm.detach())
    return total, metrics


@torch.no_grad()
def validate(model, loader, normalizer, device, mode, max_batches=10,
             mask_probability=.4, reconstruction_weight=.2, forecast_weight=.1):
    model.eval()
    values = []
    # Fork RNG makes validation reproducible without altering training's random stream.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(1729)
        for i, raw in enumerate(loader):
            if i >= max_batches:
                break
            batch = prepare(raw, normalizer, device)
            loss, parts = objective(model, batch, mode, mask_probability,
                                    reconstruction_weight, forecast_weight)
            values.append({"loss": float(loss), **parts})
    model.train()
    return {k: float(np.mean([v[k] for v in values])) for k in values[0]} if values else {}


def run(args):
    cfg = json.loads(Path(args.config).read_text())
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
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
    horizon = len(sample["future"])
    parameter_count = sum(p.numel() for p in model.parameters())
    plan = {"mode": cfg["mode"], "device": str(device), "parameters": parameter_count,
            "train_windows": len(data), "val_windows": len(val),
            "history_frames": len(sample["history"]), "future_frames": horizon,
            "steps": cfg["steps"], "batch_size": cfg["batch_size"],
            "nominal_sample_presentations": cfg["steps"] * cfg["batch_size"],
            "note": "No RL or simulator. VRAM and throughput must be measured on the target machine."}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    if cfg["steps"] < 1:
        raise ValueError("steps must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(args.data) / "manifest.json"
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    receipt = {"status": "running", "config": cfg, "plan": plan,
               "environment": environment(), "code_sha256": source_hashes,
               "dataset_manifest_sha256": sha256(manifest_path)}
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
    else:
        normalizer = fit_normalizer(data)
    dump_json(out / "normalizer.json", normalizer)
    sampler = None
    sampling = cfg.get("sampling", "uniform_window")
    if sampling == "group_scene":
        pairs = [(r["group_id"], r["scene_id"] != "empty") for r in data.records]
        counts = Counter(pairs)
        kinds = Counter(g for g, _ in counts)
        weights = [1 / (counts[(g, present)] * kinds[g]) for g, present in pairs]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True,
                                         generator=torch.Generator().manual_seed(seed))
    elif sampling != "uniform_window":
        raise ValueError("sampling must be uniform_window or group_scene")
    loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=sampler is None,
                        sampler=sampler,
                        num_workers=cfg.get("workers", 0), collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    val_loader = DataLoader(val, batch_size=cfg["batch_size"], collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                                  weight_decay=cfg.get("weight_decay", .01))
    model.train()
    iterator = iter(loader)
    started = time.monotonic()
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
                batch = prepare(raw, normalizer, device)
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
                if step == 1 or step % cfg.get("log_every", 100) == 0 or step == cfg["steps"]:
                    row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad),
                           "elapsed_seconds": time.monotonic() - started, **parts}
                    if step % cfg.get("eval_every", 1000) == 0 or step == cfg["steps"]:
                        row["validation"] = validate(model, val_loader, normalizer, device,
                                                      cfg["mode"], cfg.get("val_batches", 10),
                                                      cfg.get("mask_probability", .4),
                                                      cfg.get("reconstruction_weight", .2),
                                                      cfg.get("forecast_weight", .1))
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
        # Fixed-budget last checkpoint: no post-hoc selection on test or calibration.
        checkpoint = {"model": model.cpu().state_dict(), "model_config": asdict(model_cfg),
                      "normalizer": normalizer, "config": cfg, "step": cfg["steps"],
                      "dataset_manifest_sha256": receipt["dataset_manifest_sha256"],
                      "history_frames": len(sample["history"]), "horizon": horizon}
        torch.save(checkpoint, out / "checkpoint.pt")
        receipt.update(status="complete", elapsed_seconds=time.monotonic() - started,
                       checkpoint_sha256=sha256(out / "checkpoint.pt"),
                       max_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                 if device.type == "cuda" else 0))
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
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
