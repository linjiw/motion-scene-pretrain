"""Create a fresh, metadata-selected scene holdout without changing source data.

This is a new training protocol: checkpoints trained on the parent dataset have
seen newly held-out groups and must not initialize or evaluate this protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .data import ARRAY_KEYS, SPLITS, load_manifest, sha256, verify_dataset


def _group_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def derive_scene_holdout(source, output, scene_groups=2, seed=20261015):
    """Move selected whole training ancestry groups into validation.

    The selection inspects scene-presence metadata only, never model errors.
    Every old validation/calibration/test assignment is preserved. Numerical
    arrays are copied into new shards, with no shared writable source files.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("Output must not be nested inside the immutable source dataset")
    if isinstance(scene_groups, bool) or not isinstance(scene_groups, int) or scene_groups < 1:
        raise ValueError("scene_groups must be a positive integer")
    parent = load_manifest(source)
    if "scene_holdout_derivation" in parent:
        raise ValueError("Repeated scene-holdout derivation is not allowed")
    verify_dataset(source)
    parent_hash = sha256(source / "manifest.json")
    records = [json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines()]
    eligible = sorted({r["group_id"] for r in records
                       if r["split"] == "train" and r["scene_id"] != "empty"},
                      key=lambda group: (_group_key(group, seed), group))
    if len(eligible) <= scene_groups:
        raise ValueError("Keep at least one scene-bearing ancestry group in training")
    moved = eligible[:scene_groups]
    assignments = dict(parent["group_splits"])
    for group in moved:
        if assignments[group] != "train":
            raise ValueError("Only original package training groups may move")
        assignments[group] = "val"
    by_split = {split: [] for split in SPLITS}
    for row in records:
        by_split[assignments[row["group_id"]]].append(row)
    if any(not rows for rows in by_split.values()):
        raise ValueError("Every train/val/calibration/test split must remain nonempty")
    parent_shards = {s["id"]: s for s in parent["shards"]}
    shard_rows = max(s["rows"] for s in parent["shards"])
    maps = {}

    def array(row, key):
        sid = row["shard_id"]
        if sid not in maps:
            maps[sid] = {k: np.load(source / path, mmap_mode="r", allow_pickle=False)
                         for k, path in parent_shards[sid]["arrays"].items()}
        return maps[sid][key][row["shard_row"]]

    output.mkdir(parents=True, exist_ok=False)
    (output / "shards").mkdir()
    new_records, shards = [], []
    for split, rows in by_split.items():
        for start in range(0, len(rows), shard_rows):
            chunk = rows[start:start + shard_rows]
            sid = f"{split}-{start // shard_rows:04d}"
            directory = output / "shards" / sid
            directory.mkdir()
            arrays = {}
            for key in ARRAY_KEYS:
                path = directory / f"{key}.npy"
                np.save(path, np.stack([array(row, key) for row in chunk]), allow_pickle=False)
                arrays[key] = path.relative_to(output).as_posix()
            for index, row in enumerate(chunk):
                record = dict(row)
                record.update(split=split, shard_id=sid, shard_row=index,
                              parent_package_split=row["split"])
                new_records.append(record)
            shards.append({"id": sid, "split": split, "rows": len(chunk), "arrays": arrays})
    (output / "samples.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in new_records))
    if (source / "source_clips.json").exists():
        shutil.copyfile(source / "source_clips.json", output / "source_clips.json")
    files = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = deepcopy(parent)
    # Do not mislabel the parent's creation timestamp as this derivative's time.
    # Omitting a fresh wall-clock timestamp keeps content-addressed output stable.
    manifest.pop("created_utc", None)
    manifest.update(group_splits=assignments, shards=shards, files=files,
                    counts={split: len(rows) for split, rows in by_split.items()},
                    samples=len(new_records))
    manifest["scene_holdout_derivation"] = {
        "version": 1,
        "parent_manifest_sha256": parent_hash,
        "parent_samples_sha256": parent["files"]["samples.jsonl"]["sha256"],
        "parent_created_utc": parent.get("created_utc"),
        "selection_policy": "Scene-bearing TRAIN groups only; sort SHA256(f'{seed}:{group_id}') ascending; move first N whole groups into val",
        "selection_uses": "group ID and scene presence metadata only; no model scores or errors",
        "seed": seed, "scene_groups": scene_groups, "eligible_scene_train_groups": eligible,
        "moved_groups": moved,
        "preserved_assignments": ["all old val", "all old calibration", "all old test"],
        "source_values": "Every source row and every numeric tensor is preserved; only package split and shard mapping change",
        "repartition_code_sha256": sha256(Path(__file__)),
        "training_requirement": "Train from scratch on this derivative; do not warm-start any model pretrained or fine-tuned on the parent train split",
        "parent_checkpoints_allowed": False,
        "normalization_requirement": "Refit on derivative train only; parent normalizer is prohibited",
        "comparison_scope": "New protocol; parent-model metrics are not directly comparable on the changed validation population",
    }
    manifest["semantics"] = {**manifest.get("semantics", {}),
        "split": "Scene-holdout v2: whole metadata-selected parent train groups moved into val; all old heldouts frozen; all data still original-source train",
        "group_assignment": "Use immutable group_splits; parent SHA256 ranking plus declared whole-group scene holdout",
        "warm_start": "No parent/v1-trained weights or parent/v1 normalizer; newly held-out groups were previously training data"}
    summary = manifest.setdefault("summary", {})
    summary.update(counts=manifest["counts"], group_counts=dict(Counter(assignments.values())),
        scene_rows_by_split={split: sum(r["scene_id"] != "empty" for r in rows) for split, rows in by_split.items()},
        scene_groups_by_split={split: len({r["group_id"] for r in rows if r["scene_id"] != "empty"}) for split, rows in by_split.items()})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_dataset(output)
    if sha256(source / "manifest.json") != parent_hash:
        raise RuntimeError("Parent manifest changed during derivation")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scene-groups", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261015)
    args = p.parse_args()
    result = derive_scene_holdout(args.source, args.output, args.scene_groups, args.seed)
    print(json.dumps({"status": "complete", "counts": result["counts"],
        "group_counts": result["summary"]["group_counts"],
        "scene_rows_by_split": result["summary"]["scene_rows_by_split"],
        "scene_groups_by_split": result["summary"]["scene_groups_by_split"],
        "moved_groups": result["scene_holdout_derivation"]["moved_groups"],
        "manifest_sha256": sha256(Path(args.output) / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
