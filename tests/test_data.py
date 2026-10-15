import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from motion_scene_pretrain.data import MixedSceneDataset, causal_history, sha256, verify_dataset


_script = Path(__file__).parents[1] / "scripts" / "export_data.py"
_spec = importlib.util.spec_from_file_location("export_data", _script)
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def test_anchor_velocity_is_past_only_and_rotated():
    h = np.zeros((3, 65), dtype=np.float32)
    h[-2, 4:6] = [2, 0]
    h[-2, 6] = np.pi / 2 * 50
    h[-2, 7:36] = 1
    h[-1, 7:36] = 1.1
    h[-1, 4:7] = 1234  # Source forward channels depend on an unseen pose.
    h[-1, 36:] = 999
    fixed = causal_history(h, 50)
    np.testing.assert_allclose(fixed[-1, 4:6], [0, -2], atol=1e-6)
    np.testing.assert_allclose(fixed[-1, 36:], 5, atol=1e-5)
    assert fixed[-1, 6] == h[-2, 6]
    assert h[-1, 36] == 999  # No in-place mutation.
    changed = h.copy()
    changed[-1, 4:7] = -9876
    changed[-1, 36:] = -9876
    np.testing.assert_array_equal(fixed, causal_history(changed, 50))


def test_overlap_reconstruction_refuses_conflicting_states():
    full = np.arange(40 * 65, dtype=np.float32).reshape(40, 65)
    starts = np.array([0, 5, 10])
    states = np.stack([full[s:s + 29] for s in starts])
    sequence, covered = exporter.reconstruct_clip(states, starts, np.arange(3))
    np.testing.assert_array_equal(sequence, full[:39])
    assert covered.all()
    states[1, 2, 3] += 1
    with pytest.raises(ValueError, match="overlap disagrees"):
        exporter.reconstruct_clip(states, starts, np.arange(3))


def _fixture_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    lineage, arrays = [], {k: [] for k in ("states", "obstacles", "obstacle_mask", "goals", "clip_index", "group_index", "split", "tier", "scene", "start_frame", "first_collision")}
    for clip in range(5):
        ref = tmp_path / f"reference-{clip}.npz"
        np.savez(ref, fixture=np.array([clip]))
        split = "train" if clip < 4 else "development"
        lineage.append({"id": str(clip), "cohort": "route", "split": split,
                        "group": f"group-{clip}", "tier": "clean", "path": str(ref),
                        "reference_sha256": sha256(ref)})
        full = np.random.default_rng(clip).normal(size=(40, 65)).astype(np.float32)
        for start in (0, 5, 10):
            for scene in ("empty", "witness:7"):
                values = {"states": full[start:start + 29], "obstacles": np.zeros((8, 9), np.float32),
                          "obstacle_mask": np.zeros(8, bool), "goals": np.zeros(4, np.float32),
                          "clip_index": clip, "group_index": clip, "split": split, "tier": "clean",
                          "scene": scene, "start_frame": start, "first_collision": -1 if scene == "empty" else 9}
                for key, value in values.items():
                    arrays[key].append(value)
    np.savez_compressed(source / "chunks.npz", **{k: np.array(v) for k, v in arrays.items()})
    (source / "layout.json").write_text(json.dumps({"history": 4, "horizon": 25, "fps": 50, "max_obstacles": 8}))
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(lineage))
    (source / "receipt.json").write_text(json.dumps({"status": "complete", "bindings": {"manifest.json": sha256(path)}, "output_sha256": {"chunks.npz": sha256(source / "chunks.npz")}}))
    return source, path


def test_export_excludes_official_eval_and_colliding_targets(tmp_path):
    source, lineage = _fixture_source(tmp_path)
    out = tmp_path / "export"
    args = argparse.Namespace(source=source, lineage=lineage, output=out,
                              history=5, horizon=10, max_samples=0, seed=1,
                              shard_rows=2, dry_run=False)
    manifest = exporter.export(args)
    assert manifest["samples"] == 8
    assert set(manifest["group_splits"]) == {f"group-{i}" for i in range(4)}
    assert verify_dataset(out)["status"] == "verified"
    ds = MixedSceneDataset(out, "train")
    item = ds[0]
    assert item["history"].shape == (5, 65)
    assert item["future"].shape == (10, 65)
    assert item["scene_mask"].dtype.is_floating_point is False
    assert item["metadata"]["scene_id"] == "empty"
    original = ds[0]["history"].clone()
    item["history"].zero_()
    assert bool((ds[0]["history"] == original).all())
    # Detect byte-level corruption even when the array still decodes.
    array = out / manifest["shards"][0]["arrays"]["goal"]
    with array.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x01")
    with pytest.raises(ValueError, match="SHA256"):
        verify_dataset(out)


def test_export_refuses_lineage_not_bound_to_source(tmp_path):
    source, lineage = _fixture_source(tmp_path)
    records = json.loads(lineage.read_text())
    records[-1]["split"] = "train"
    lineage.write_text(json.dumps(records))
    args = argparse.Namespace(source=source, lineage=lineage, output=tmp_path / "export",
                              history=5, horizon=10, max_samples=0, seed=1,
                              shard_rows=2, dry_run=False)
    with pytest.raises(ValueError, match="not bound"):
        exporter.export(args)
