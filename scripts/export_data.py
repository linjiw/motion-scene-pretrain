#!/usr/bin/env python3
"""Export original-training-lineage PASSAGE windows without importing Flow/Isaac.

Longer windows are reconstructed from overlapping source windows. Overlaps must
agree exactly; data are never extrapolated or padded across clip boundaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from motion_scene_pretrain.data import ARRAY_KEYS, SPLITS, causal_history, sha256, verify_dataset


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _key(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_group_splits(groups, seed):
    """Frozen assignment for this source cohort; every group goes to one split."""
    ordered = sorted(set(groups), key=lambda g: _key(g, seed))
    if len(ordered) < 4:
        raise ValueError("Need at least four independent ancestry groups")
    n = len(ordered)
    n_hold = max(1, round(n * .1))
    n_train = n - 3 * n_hold
    if n_train < 1:
        raise ValueError("Not enough groups for disjoint splits")
    labels = ["train"] * n_train + ["val"] * n_hold + ["calibration"] * n_hold + ["test"] * n_hold
    return dict(zip(ordered, labels))


def reconstruct_clip(states, starts, rows):
    """Return clip-relative states and coverage; reject inconsistent duplicates."""
    length = states.shape[1]
    end = int(starts[rows].max()) + length
    sequence = np.zeros((end, states.shape[-1]), dtype=np.float32)
    covered = np.zeros(end, dtype=bool)
    for row in rows:
        start = int(starts[row])
        region = slice(start, start + length)
        old = covered[region]
        if old.any() and not np.array_equal(sequence[region][old], states[row][old]):
            raise ValueError(f"Source overlap disagrees at row {row}")
        sequence[region] = states[row]
        covered[region] = True
    return sequence, covered


def select_bundles(bundles, groups, max_samples, seed):
    """Round-robin ancestry groups, retaining complete same-anchor scene bundles."""
    by_group = defaultdict(list)
    for anchor, rows in bundles.items():
        by_group[groups[anchor[0]]].append((anchor, rows))
    queues = {g: deque(sorted(v, key=lambda item: _key(item[0], seed))) for g, v in by_group.items()}
    order = sorted(queues, key=lambda g: _key(g, seed))
    selected, count = [], 0
    while any(queues.values()):
        progress = False
        for group in order:
            if not queues[group]:
                continue
            anchor, rows = queues[group].popleft()
            if max_samples and count + len(rows) > max_samples:
                continue
            selected.extend(rows)
            count += len(rows)
            progress = True
        if max_samples and count >= max_samples:
            break
        if not progress and not any(queues.values()):
            break
    return selected


def export(args):
    source = Path(args.source).resolve()
    lineage_path = Path(args.lineage).resolve()
    layout = json.loads((source / "layout.json").read_text())
    receipt = json.loads((source / "receipt.json").read_text())
    if receipt.get("status") != "complete":
        raise ValueError("Source receipt is incomplete")
    source_hash = sha256(source / "chunks.npz")
    if receipt["output_sha256"]["chunks.npz"] != source_hash:
        raise ValueError("Source chunks differ from immutable receipt")
    lineage_hash = sha256(lineage_path)
    bound = {v for k, v in receipt["bindings"].items() if k.endswith("manifest.json")}
    if lineage_hash not in bound:
        raise ValueError("Positive lineage manifest is not bound to source receipt")
    lineage = json.loads(lineage_path.read_text())
    splitsets = defaultdict(set)
    for row in lineage:
        splitsets[row["group"]].add(row["split"])
    if any(len(v) != 1 for v in splitsets.values()):
        raise ValueError("Source ancestry crosses official splits")
    allowed = {i: r for i, r in enumerate(lineage)
               if r["split"] == "train" and r["tier"] in ("clean", "repaired")
               and r["cohort"] in ("route", "hindsight-120")}
    with np.load(source / "chunks.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    n = len(data["states"])
    if data["states"].shape[1:] != (layout["history"] + layout["horizon"], 65):
        raise ValueError("Unexpected source state layout")
    ci, starts, scenes = data["clip_index"], data["start_frame"], data["scene"].astype(str)
    candidate = (data["split"] == "train") & np.isin(ci, list(allowed))
    candidate &= ~np.char.startswith(scenes, "witness:") & ~np.char.startswith(scenes, "toolate:")
    candidate &= data["first_collision"] < 0
    sequences, bundles = {}, defaultdict(list)
    fps = float(layout["fps"])
    for clip in sorted(set(ci[candidate])):
        rows = np.flatnonzero((ci == clip) & candidate)
        # One row per start is enough for reconstruction. Scene duplication is
        # checked below against the reconstructed values for every selected row.
        _, first = np.unique(starts[rows], return_index=True)
        seq, coverage = reconstruct_clip(data["states"], starts, rows[first])
        for r in rows:
            s = int(starts[r])
            if not np.array_equal(data["states"][r], seq[s:s + data["states"].shape[1]]):
                raise ValueError("Scene variants have inconsistent motion history")
            anchor = s + int(layout["history"]) - 1
            lo, hi = anchor - args.history + 1, anchor + args.horizon + 1
            if lo >= 0 and hi <= len(seq) and coverage[lo:hi].all():
                bundles[(int(clip), anchor)].append(int(r))
        sequences[int(clip)] = seq
    groups = {int(c): allowed[int(c)]["group"] for c in sequences}
    selected = select_bundles(bundles, groups, args.max_samples, args.seed)
    if not selected:
        raise ValueError("No complete eligible windows")
    selected_clips = sorted(set(int(ci[r]) for r in selected))
    assignments = assign_group_splits([groups[c] for c in selected_clips], args.seed)
    summary = {
        "source_rows": n, "eligible_rows": sum(map(len, bundles.values())),
        "selected_rows": len(selected), "clips": len(selected_clips),
        "groups": len(assignments), "group_counts": dict(Counter(assignments.values())),
        "counts": dict(Counter(assignments[groups[int(ci[r])]] for r in selected)),
        "scene_counts": dict(Counter("empty" if scenes[r] == "empty" else "archived" for r in selected)),
        "history": args.history, "horizon": args.horizon, "fps": fps,
        "source_chunks_sha256": source_hash, "lineage_sha256": lineage_hash,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return summary
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "shards").mkdir()
    # Verify accepted clip hashes without importing their data or exporting them.
    for clip in selected_clips:
        reference = Path(allowed[clip]["path"])
        if not reference.is_absolute():
            reference = source.parent.parent / reference
        if sha256(reference) != allowed[clip]["reference_sha256"]:
            raise ValueError(f"Accepted source reference hash changed: {clip}")
    source_clip_records = [{"source_clip_index": c, "clip_id": allowed[c]["id"],
        "group_id": groups[c], "original_split": "train", "tier": allowed[c]["tier"],
        "cohort": allowed[c]["cohort"], "reference_sha256": allowed[c]["reference_sha256"],
        "source_reference_path": allowed[c]["path"], "source_manifest_sha256": lineage_hash,
        "physical_qualification": "UNTESTED", "source_family": "Kimodo indoor_nav_1k generated motion"}
        for c in selected_clips]
    _write_json(out / "source_clips.json", source_clip_records)
    selected_pairs = {(int(ci[r]), int(starts[r])) for r in selected if scenes[r] == "empty"}
    by_split = {s: [] for s in SPLITS}
    for row in selected:
        by_split[assignments[groups[int(ci[row])]]].append(row)
    records, shards = [], []
    for split in SPLITS:
        rows = sorted(by_split[split], key=lambda r: _key(f"row:{r}", args.seed))
        for offset in range(0, len(rows), args.shard_rows):
            chunk = rows[offset:offset + args.shard_rows]
            sid = f"{split}-{offset // args.shard_rows:04d}"
            directory = out / "shards" / sid
            directory.mkdir()
            arrays = {k: [] for k in ARRAY_KEYS}
            for j, row in enumerate(chunk):
                clip, start = int(ci[row]), int(starts[row])
                anchor = start + int(layout["history"]) - 1
                seq = sequences[clip]
                arrays["history"].append(causal_history(seq[anchor - args.history + 1:anchor + 1], fps))
                arrays["future"].append(seq[anchor + 1:anchor + args.horizon + 1])
                arrays["scene"].append(data["obstacles"][row])
                arrays["scene_mask"].append(data["obstacle_mask"][row])
                arrays["goal"].append(data["goals"][row])
                arrays["future_mask"].append(np.ones(args.horizon, dtype=bool))
                records.append({"sample_id": f"passage-v3-{row:06d}", "split": split,
                    "shard_id": sid, "shard_row": j, "source_row": row,
                    "clip_id": allowed[clip]["id"], "source_clip_index": clip,
                    "group_id": groups[clip], "original_split": "train",
                    "scene_id": str(scenes[row]), "source_start_frame": start,
                    "anchor_frame": anchor, "tier": allowed[clip]["tier"],
                    "reference_sha256": allowed[clip]["reference_sha256"],
                    "paired_empty_available": (clip, start) in selected_pairs,
                    "physical_qualification": "UNTESTED"})
            filenames = {}
            for key, values in arrays.items():
                a = np.stack(values).astype(np.bool_ if key.endswith("mask") else np.float32)
                if not np.isfinite(a).all():
                    raise ValueError(f"Nonfinite exported {key}")
                path = directory / f"{key}.npy"
                np.save(path, a, allow_pickle=False)
                filenames[key] = path.relative_to(out).as_posix()
            shards.append({"id": sid, "split": split, "rows": len(chunk), "arrays": filenames})
    (out / "samples.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    files = {p.relative_to(out).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {
        "schema_version": 1, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(records), "history": args.history, "horizon": args.horizon,
        "fps": fps, "state_dim": 65, "scene_dim": 9, "goal_dim": 4,
        "max_obstacles": int(layout["max_obstacles"]), "counts": {s: len(by_split[s]) for s in SPLITS},
        "group_splits": assignments, "shards": shards, "files": files,
        "source": {"path_at_export": str(source), "chunks_sha256": source_hash,
            "lineage_path_at_export": str(lineage_path), "lineage_sha256": lineage_hash,
            "source_receipt_sha256": sha256(source / "receipt.json"),
            "exporter_sha256": sha256(Path(__file__)), "seed": args.seed,
            "eligible_source_cohorts": ["route", "hindsight-120"],
            "source_split": "train", "source_family": "Kimodo indoor_nav_1k generated motion"},
        "semantics": {"state": "PASSAGE 65D forward-difference targets; final history derivatives causal",
            "goal": "reference pose at anchor+50 source frames; retrospective training condition, not an independently recorded request",
            "scene": "anchor-local boxes/spheres, z up, metres; hindsight compatible, physics UNTESTED",
            "height": "inherited source datum; no SONIC grounding correction applied",
            "license": "inherited source restrictions; private research transfer only, no data license granted",
            "split": "all four package splits carved only from original train; original development/fresh absent",
            "ancestry": "shared-prompt clusters, not independent human subjects",
            "window_sampling": "ancestry round-robin; retain complete same-anchor compatible/empty bundles",
            "group_assignment": "SHA256(seed:group) sorted cohort, 10% groups each val/calibration/test; freeze manifest for comparisons"},
        "summary": summary,
    }
    _write_json(out / "manifest.json", manifest)
    print(json.dumps(verify_dataset(out), indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--lineage", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--history", type=int, default=25)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=20000, help="0 exports all eligible rows")
    p.add_argument("--shard-rows", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.history < 2 or args.horizon < 1 or args.shard_rows < 1 or args.max_samples < 0:
        p.error("Invalid history/horizon/shard/max-samples")
    export(args)


if __name__ == "__main__":
    main()
