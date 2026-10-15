"""Portable, non-pickle, memory-mapped motion/scene data and integrity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ARRAY_KEYS = ("history", "future", "scene", "scene_mask", "goal", "future_mask")
SPLITS = ("train", "val", "calibration", "test")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


def load_manifest(dataset_path: str | Path) -> dict:
    root = Path(dataset_path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")
    if manifest.get("status") != "complete":
        raise ValueError("Dataset export did not complete")
    assignments = manifest["group_splits"]
    if not assignments or any(v not in SPLITS for v in assignments.values()):
        raise ValueError("Invalid ancestry split assignments")
    return manifest


def verify_dataset(dataset_path: str | Path, *, check_hashes: bool = True) -> dict:
    """Check every numeric shard, metadata lineage, shapes, and content hash."""
    root = Path(dataset_path)
    m = load_manifest(root)
    expected_shapes = {
        "history": (m["history"], m["state_dim"]),
        "future": (m["horizon"], m["state_dim"]),
        "scene": (m["max_obstacles"], m["scene_dim"]),
        "scene_mask": (m["max_obstacles"],),
        "goal": (m["goal_dim"],),
        "future_mask": (m["horizon"],),
    }
    for relative, entry in m["files"].items():
        path = _safe_path(root, relative)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        if check_hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch: {relative}")
    records = [json.loads(s) for s in (root / "samples.jsonl").read_text().splitlines()]
    seen = set()
    shard_counts = {s["id"]: 0 for s in m["shards"]}
    shards = {s["id"]: s for s in m["shards"]}
    counts = {s: 0 for s in SPLITS}
    for row in records:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(row["sample_id"])
        if row["original_split"] != "train":
            raise ValueError("Non-training source lineage in portable corpus")
        if m["group_splits"].get(row["group_id"]) != row["split"]:
            raise ValueError("Ancestry group crosses assigned splits")
        if row["scene_id"].startswith(("witness:", "toolate:")):
            raise ValueError("Colliding counterfactual used as a positive target")
        shard = shards[row["shard_id"]]
        if shard["split"] != row["split"] or row["shard_row"] != shard_counts[row["shard_id"]]:
            raise ValueError("Invalid shard row mapping")
        shard_counts[row["shard_id"]] += 1
        counts[row["split"]] += 1
    for shard in m["shards"]:
        if shard_counts[shard["id"]] != shard["rows"]:
            raise ValueError("Shard row count mismatch")
        for key, expected in expected_shapes.items():
            relative = shard["arrays"][key]
            if relative not in m["files"]:
                raise ValueError("Unhashed array in shard")
            a = np.load(_safe_path(root, relative), mmap_mode="r", allow_pickle=False)
            if a.shape != (shard["rows"], *expected):
                raise ValueError(f"Invalid array shape: {relative}")
            if a.dtype != (np.bool_ if key.endswith("mask") else np.float32):
                raise ValueError(f"Invalid array dtype: {relative}")
            if not np.isfinite(a).all():
                raise ValueError(f"Nonfinite values: {relative}")
    if counts != m["counts"] or len(records) != m["samples"]:
        raise ValueError("Dataset sample count mismatch")
    return {"status": "verified", "samples": len(records), "counts": counts,
            "groups": len(m["group_splits"]), "files": len(m["files"])}


class MixedSceneDataset(Dataset):
    """Raw physical units; normalization must be fitted on ``train`` only.

    Numeric arrays are read-only memory maps. Returned tensors copy just one row,
    so downstream masking/normalization cannot modify the dataset on disk.
    """

    def __init__(self, dataset_path: str | Path, split: str, max_samples: int | None = None):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected {SPLITS}")
        self.root = Path(dataset_path)
        self.manifest = load_manifest(self.root)
        self.split = split
        records = [json.loads(s) for s in (self.root / "samples.jsonl").read_text().splitlines()]
        for row in records:
            if row["original_split"] != "train" or self.manifest["group_splits"].get(row["group_id"]) != row["split"]:
                raise ValueError("Unapproved source or overlapping ancestry")
        self.records = [r for r in records if r["split"] == split]
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            self.records = self.records[:max_samples]
        self._shards = {s["id"]: s for s in self.manifest["shards"]}
        self._maps: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def __getitem__(self, index):
        row = self.records[index]
        sid = row["shard_id"]
        if sid not in self._maps:
            self._maps[sid] = {
                k: np.load(_safe_path(self.root, p), mmap_mode="r", allow_pickle=False)
                for k, p in self._shards[sid]["arrays"].items()
            }
        result = {k: torch.from_numpy(np.array(self._maps[sid][k][row["shard_row"]], copy=True))
                  for k in ARRAY_KEYS}
        # Only fixed-schema scalar fields: PyTorch default_collate remains usable.
        result["metadata"] = {k: row[k] for k in (
            "sample_id", "group_id", "clip_id", "scene_id", "anchor_frame", "paired_empty_available")}
        return result


def causal_history(states: np.ndarray, fps: float) -> np.ndarray:
    """Replace only last observed forward-difference channels with past estimates.

    Planar velocity at t-1 is in heading(t-1), and must be rotated into heading(t).
    This removes the source encoding's dependency on the first unobserved pose.
    """
    h = np.asarray(states, dtype=np.float32).copy()
    if len(h) < 2:
        raise ValueError("Causal history needs at least two frames")
    angle = float(h[-2, 6]) / fps
    c, s = np.cos(angle), np.sin(angle)
    vx, vy = h[-2, 4:6]
    h[-1, 4:6] = (c * vx + s * vy, -s * vx + c * vy)
    h[-1, 6] = h[-2, 6]
    h[-1, 36:65] = (h[-1, 7:36] - h[-2, 7:36]) * fps
    return h
