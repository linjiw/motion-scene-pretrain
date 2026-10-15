"""Offline future prediction, scene interventions and residual-uncertainty checks."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import (checkpoint_load, collate, dump_json, environment, prepare,
                     sample_flow, sha256)
from .data import MixedSceneDataset
from .model import ModelConfig, MotionSceneModel


def load_model(path, device):
    checkpoint = checkpoint_load(path)
    if checkpoint["config"]["mode"] == "masked":
        raise ValueError("Masked-only checkpoint has no trained forecast head; fine-tune it first")
    model = MotionSceneModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def score_prediction(error, variance, valid):
    """Elementwise marginal diagnostics; not independent trials or joint coverage."""
    select = np.broadcast_to(valid[..., None], error.shape)
    e, v = error[select], variance[select]
    if not len(e) or not np.isfinite(e).all() or not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Scoring needs nonempty finite errors and strictly positive finite variances")
    z = np.abs(e) / np.sqrt(np.maximum(v, 1e-10))
    return {"mse_normalized": float(np.mean(e**2)),
            "gaussian_nll_normalized": float(np.mean(.5 * (np.log(2 * math.pi * v) + e**2 / v))),
            "marginal_coverage_68": float(np.mean(z <= 1.)),
            "marginal_coverage_90": float(np.mean(z <= 1.6448536269514722)),
            "marginal_coverage_95": float(np.mean(z <= 1.959963984540054)),
            "mean_std_normalized": float(np.mean(np.sqrt(v))),
            "scalar_elements": int(len(e))}


@torch.no_grad()
def collect(model, checkpoint, dataset, device, batch_size=32, samples=4,
            flow_steps=16, interventions=True, seed=0):
    torch.manual_seed(seed)
    means, variances, targets, valids, scene_delta, shuffled_delta = [], [], [], [], [], []
    flow_error, flow_oracle_error, flow_diversity = [], [], []
    attention_snapshot = None
    std = np.asarray(checkpoint["normalizer"]["std"])
    for raw in DataLoader(dataset, batch_size=batch_size, collate_fn=collate):
        batch = prepare(raw, checkpoint["normalizer"], device)
        kwargs = dict(horizon=batch["future"].shape[1], scene_mask=batch["scene_mask"],
                      future_mask=batch["future_mask"])
        pred = model(batch["history"], batch["scene"], batch["goal"],
                     return_attention=True, **kwargs)
        means.append(pred["forecast_mean"].cpu().numpy())
        variances.append(pred["logvar"].exp().cpu().numpy())
        targets.append(batch["future"].cpu().numpy())
        valids.append(batch["future_mask"].cpu().numpy())
        if attention_snapshot is None or (not any(attention_snapshot["scene_mask"]) and batch["scene_mask"].any()):
            nonempty = batch["scene_mask"].any(1).nonzero()
            example_index = int(nonempty[0, 0]) if len(nonempty) else 0
            attention_snapshot = {"weights": pred["scene_attention"][example_index].cpu().tolist(),
                                  "scene": batch["scene"][example_index].cpu().tolist(),
                                  "scene_mask": batch["scene_mask"][example_index].cpu().tolist(),
                                  "note": "Last column is null token. Attention is descriptive, not causal importance."}
        if interventions:
            empty = model(batch["history"], batch["scene"], batch["goal"],
                          horizon=kwargs["horizon"], scene_mask=torch.zeros_like(batch["scene_mask"]),
                          future_mask=batch["future_mask"])
            valid = batch["future_mask"][..., None]
            denominator = (valid.sum((1, 2)) * batch["future"].shape[-1]).clamp_min(1)
            scene_delta.append(((empty["forecast_mean"] - pred["forecast_mean"]).square().masked_fill(~valid, 0).sum((1, 2)) / denominator).cpu().numpy())
            # Whole scene geometry+mask moves together. This tests reliance, not valid counterfactual behavior.
            shuffled = model(batch["history"], batch["scene"].roll(1, 0), batch["goal"],
                             horizon=kwargs["horizon"], scene_mask=batch["scene_mask"].roll(1, 0),
                             future_mask=batch["future_mask"])
            if len(batch["history"]) > 1:
                shuffled_delta.append(((shuffled["forecast_mean"] - pred["forecast_mean"]).square().masked_fill(~valid, 0).sum((1, 2)) / denominator).cpu().numpy())
        if checkpoint["config"]["mode"] == "flow" and samples > 0:
            draws = torch.stack([sample_flow(model, batch, flow_steps) for _ in range(samples)], 1)
            valid = batch["future_mask"][:, None, :, None]
            per_draw = ((draws - batch["future"][:, None]).square() * valid).sum((2, 3))
            per_draw /= (valid.sum((2, 3)) * draws.shape[-1]).clamp_min(1)
            flow_error.extend(per_draw.mean(1).cpu().tolist())
            flow_oracle_error.extend(per_draw.min(1).values.cpu().tolist())
            variance_draws = draws.var(1, unbiased=False)
            valid_frames = batch["future_mask"][..., None]
            diversity = variance_draws.masked_fill(~valid_frames, 0).sum((1, 2))
            diversity /= (valid_frames.sum((1, 2)) * draws.shape[-1]).clamp_min(1)
            flow_diversity.extend(diversity.cpu().tolist())
    mean = np.concatenate(means)
    variance = np.concatenate(variances)
    target = np.concatenate(targets)
    valid = np.concatenate(valids)
    error = mean - target
    physical_error = error * std
    physical = {}
    for name, begin, end, unit in [("height", 0, 1, "m"), ("planar_velocity", 4, 6, "m/s"),
                                    ("yaw_rate", 6, 7, "rad/s"), ("joint_position", 7, 36, "rad"),
                                    ("joint_velocity", 36, 65, "rad/s")]:
        e = physical_error[..., begin:end]
        keep = np.broadcast_to(valid[..., None], e.shape)
        physical[name] = {"rmse": float(np.sqrt(np.mean(e[keep]**2))), "unit": unit}
    metrics = {"windows": len(mean), "uncalibrated": score_prediction(error, variance, valid),
               "physical_rmse": physical,
               "attention_example": attention_snapshot,
               "interpretation": "Offline reference prediction only; not physical collision or navigation success."}
    fps = checkpoint["model_config"]["fps"]
    metrics["by_horizon"] = {}
    for start, stop in ((0, int(fps)), (int(fps), int(2 * fps))):
        stop = min(stop, error.shape[1])
        if start < stop and valid[:, start:stop].any():
            metrics["by_horizon"][f"{start/fps:g}-{stop/fps:g}s"] = score_prediction(
                error[:, start:stop], variance[:, start:stop], valid[:, start:stop])
    if hasattr(dataset, "records"):
        metrics["ancestry_groups"] = len({r["group_id"] for r in dataset.records})
        group_ids = np.array([r["group_id"] for r in dataset.records])
        metrics["by_group"] = {}
        for group in sorted(set(group_ids)):
            selection = group_ids == group
            metrics["by_group"][group] = {"windows": int(selection.sum()), **score_prediction(
                error[selection], variance[selection], valid[selection])}
        metrics["equal_group_mean_mse_normalized"] = float(np.mean([
            g["mse_normalized"] for g in metrics["by_group"].values()]))
        present = np.array([r["scene_id"] != "empty" for r in dataset.records])
        metrics["by_scene_presence"] = {}
        for name, selection in (("archived_scene", present), ("empty_scene", ~present)):
            if selection.any():
                metrics["by_scene_presence"][name] = {"windows": int(selection.sum()), **score_prediction(
                    error[selection], variance[selection], valid[selection])}
    if scene_delta:
        metrics["scene_interventions"] = {
            "empty_scene_response_mse_normalized": float(np.mean(np.concatenate(scene_delta))),
            "shuffled_scene_response_mse_normalized": (float(np.mean(np.concatenate(shuffled_delta)))
                                                       if shuffled_delta else None),
            "interpretation": "Sensitivity does not prove relevance, correctness, or causal response to obstacles."}
    if flow_error:
        metrics["generation"] = {"samples_per_window": samples, "integration_steps": flow_steps,
                                 "sample_mean_mse_normalized": float(np.mean(flow_error)),
                                 "oracle_best_of_k_mse_normalized": float(np.mean(flow_oracle_error)),
                                 "sample_variance_normalized": float(np.mean(flow_diversity)),
                                 "interpretation": "Oracle best-of-K uses the reference for selection. Sample spread is not epistemic uncertainty."}
    return metrics, {"error": error, "variance": variance, "valid": valid,
                     "prediction": mean, "target": target}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["val", "calibration", "test"], default="val")
    p.add_argument("--calibrate", action="store_true", help="Fit variance scale on calibration split only")
    p.add_argument("--calibration", help="Saved calibration.json from this checkpoint")
    p.add_argument("--max-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.calibrate and (args.split != "calibration" or args.calibration):
        p.error("--calibrate requires --split calibration and cannot also apply a calibration file")
    if args.calibration and args.split == "calibration":
        p.error("Evaluate calibrated coverage on val or test, not on the calibration-fitting split")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    manifest_hash = sha256(Path(args.data) / "manifest.json")
    if manifest_hash != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("Dataset differs from checkpoint; explicit transfer evaluation is not implemented")
    dataset = MixedSceneDataset(args.data, args.split, max_samples=args.max_samples)
    if not len(dataset):
        raise ValueError("Requested split is empty")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "environment": environment(), "args": vars(args),
               "checkpoint_sha256": sha256(args.checkpoint), "dataset_manifest_sha256": manifest_hash}
    dump_json(out / "receipt.json", receipt)
    try:
        metrics, arrays = collect(model, checkpoint, dataset, device, args.batch_size,
                                  args.samples, args.flow_steps, not args.calibrate, args.seed)
        metrics.update(split=args.split, checkpoint_sha256=receipt["checkpoint_sha256"],
                       dataset_manifest_sha256=manifest_hash,
                       evaluation_seed=args.seed)
        if args.calibrate:
            select = np.broadcast_to(arrays["valid"][..., None], arrays["error"].shape)
            ratio = arrays["error"]**2 / np.maximum(arrays["variance"], 1e-10)
            scale = float(np.clip(np.mean(ratio[select]), .01, 100.))
            dump_json(out / "calibration.json", {
                "variance_scale": scale, "fit_split": "calibration",
                "unclipped_variance_scale": float(np.mean(ratio[select])), "scale_bounds": [.01, 100.],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "dataset_manifest_sha256": manifest_hash, "windows": len(dataset),
                "method": "Global Gaussian residual variance rescaling; no coverage guarantee"})
        if args.calibration:
            calibration = json.loads(Path(args.calibration).read_text())
            if calibration["fit_split"] != "calibration" or calibration["checkpoint_sha256"] != receipt["checkpoint_sha256"] or calibration["dataset_manifest_sha256"] != manifest_hash:
                raise ValueError("Calibration binding/split mismatch")
            if not np.isfinite(calibration["variance_scale"]) or calibration["variance_scale"] <= 0:
                raise ValueError("Calibration variance scale must be finite and positive")
            metrics["calibrated"] = score_prediction(arrays["error"], arrays["variance"] * calibration["variance_scale"], arrays["valid"])
            metrics["calibration_variance_scale"] = calibration["variance_scale"]
        dump_json(out / "metrics.json", metrics)
        np.savez_compressed(out / "examples.npz", **{k: v[:8] for k, v in arrays.items()})
        receipt["status"] = "complete"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        dump_json(out / "receipt.json", receipt)
        raise
    dump_json(out / "receipt.json", receipt)
    print(json.dumps({k: v for k, v in metrics.items() if k != "attention_example"}, indent=2))


if __name__ == "__main__":
    main()
