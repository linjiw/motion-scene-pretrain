"""Shared numerical/data contracts and reproducibility helpers."""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "git_revision": revision,
            "cuda_runtime": torch.version.cuda}


def collate(rows):
    keys = ("history", "future", "scene", "scene_mask", "goal", "future_mask")
    out = {}
    for key in keys:
        if key in rows[0]:
            out[key] = torch.stack([torch.as_tensor(row[key]) for row in rows])
    if "future_mask" not in out:
        out["future_mask"] = torch.ones(out["future"].shape[:2], dtype=torch.bool)
    return out


def fit_normalizer(dataset):
    """Fit exclusively on the supplied training split; stream in float64."""
    if getattr(dataset, "split", None) != "train":
        raise ValueError("Normalization must be fitted on the training split only")
    sums = squares = None
    count = 0
    for row in dataset:
        history = np.asarray(row["history"], dtype=np.float64)
        future = np.asarray(row["future"], dtype=np.float64)
        valid = np.asarray(row.get("future_mask", np.ones(len(future))), dtype=bool)
        x = np.concatenate([history, future[valid]], axis=0)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite motion in training dataset")
        if sums is None:
            sums = np.zeros(x.shape[-1], dtype=np.float64)
            squares = np.zeros_like(sums)
        sums += x.sum(0)
        squares += np.square(x).sum(0)
        count += len(x)
    if not count:
        raise ValueError("Empty training split")
    mean = sums / count
    std = np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), .01)
    return {"mean": mean.tolist(), "std": std.tolist(), "frames": count,
            "fit_split": "train", "min_std": .01}


def prepare(batch, normalizer, device):
    out = {k: v.to(device) for k, v in batch.items()}
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(normalizer["std"], dtype=torch.float32, device=device)
    for k in ("history", "future"):
        out[k] = (out[k].float() - mean) / std
    # Scene coordinates remain metric; yaw remains radians, type remains binary.
    out["scene"] = out["scene"].float()
    out["goal"] = out["goal"].float()
    out["scene_mask"] = out["scene_mask"].bool()
    out["future_mask"] = out["future_mask"].bool()
    return out


def masked_mean(x, mask):
    mask = mask.to(x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x)
    return x.masked_fill(~mask.bool(), 0).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def sample_flow(model, batch, steps=16, noise=None):
    """Data t=0, Gaussian noise t=1; Euler integrate velocity epsilon-y 1→0."""
    if steps < 1:
        raise ValueError("steps must be positive")
    x = torch.randn_like(batch["future"]) if noise is None else noise.clone()
    for i in range(steps):
        t = torch.full((len(x),), 1 - i / steps, device=x.device)
        result = model(batch["history"], batch["scene"], batch["goal"],
                       horizon=x.shape[1], scene_mask=batch["scene_mask"],
                       future_mask=batch["future_mask"], future_input=x, flow_time=t)
        x = x - result["flow_velocity"] / steps
    return x


def checkpoint_load(path, device="cpu"):
    # Only load locally produced/reviewed checkpoints, never unknown pickle objects.
    return torch.load(path, map_location=device, weights_only=True)
