import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from motion_scene_pretrain.data import ARRAY_KEYS, MixedSceneDataset, sha256, verify_dataset
from motion_scene_pretrain.repartition import derive_scene_holdout


def fixture_dataset(root):
    root.mkdir()
    groups = {f"train-{i}": "train" for i in range(4)}
    groups.update({"old-val": "val", "old-cal": "calibration", "old-test": "test"})
    records, shards = [], []
    shapes = {"history": (2, 65), "future": (3, 65), "scene": (8, 9),
              "scene_mask": (8,), "goal": (4,), "future_mask": (3,)}
    for split in ("train", "val", "calibration", "test"):
        selected = [g for g, s in groups.items() if s == split]
        sid = f"{split}-0000"
        directory = root / "shards" / sid
        directory.mkdir(parents=True)
        arrays = {key: [] for key in ARRAY_KEYS}
        for group in selected:
            value = list(groups).index(group) + 1
            for scene in ("empty", "compatible"):
                row_index = len(arrays["history"])
                for key, shape in shapes.items():
                    array = np.full(shape, value, dtype=np.float32)
                    if key.endswith("mask"):
                        array = np.ones(shape, dtype=bool)
                    if key == "scene_mask" and scene == "empty":
                        array[:] = False
                    arrays[key].append(array)
                records.append({"sample_id": f"{group}-{scene}", "group_id": group,
                    "split": split, "original_split": "train", "clip_id": group,
                    "scene_id": scene, "anchor_frame": 2, "paired_empty_available": True,
                    "shard_id": sid, "shard_row": row_index})
        paths = {}
        for key, values in arrays.items():
            path = directory / f"{key}.npy"
            np.save(path, np.stack(values), allow_pickle=False)
            paths[key] = path.relative_to(root).as_posix()
        shards.append({"id": sid, "split": split, "rows": len(arrays["history"]), "arrays": paths})
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "source_clips.json").write_text("[]\n")
    files = {p.relative_to(root).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in root.rglob("*") if p.is_file()}
    manifest = {"schema_version": 1, "status": "complete", "history": 2, "horizon": 3,
        "state_dim": 65, "scene_dim": 9, "goal_dim": 4, "max_obstacles": 8, "fps": 50,
        "group_splits": groups, "shards": shards, "files": files, "samples": len(records),
        "counts": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "calibration", "test")}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    verify_dataset(root)
    return manifest


def _all_rows(path):
    return {row["metadata"]["sample_id"]: row
            for split in ("train", "val", "calibration", "test")
            for row in MixedSceneDataset(path, split)}


def test_derivation_preserves_rows_twins_frozen_holdouts_and_parent(tmp_path):
    source, out = tmp_path / "source", tmp_path / "derived"
    parent = fixture_dataset(source)
    before = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
    result = derive_scene_holdout(source, out, scene_groups=2, seed=20261015)
    expected = sorted([f"train-{i}" for i in range(4)],
                      key=lambda g: hashlib.sha256(f"20261015:{g}".encode()).hexdigest())[:2]
    audit = result["scene_holdout_derivation"]
    assert audit["moved_groups"] == expected
    assert audit["parent_manifest_sha256"] == before["manifest.json"]
    assert audit["parent_samples_sha256"] == before["samples.jsonl"]
    assert audit["parent_checkpoints_allowed"] is False
    assert "scratch" in audit["training_requirement"]
    for group, split in parent["group_splits"].items():
        assert result["group_splits"][group] == ("val" if group in expected else split)
    assert result["counts"] == {"train": 4, "val": 6, "calibration": 2, "test": 2}
    old, new = _all_rows(source), _all_rows(out)
    assert set(old) == set(new)
    for sample in old:
        for key in ARRAY_KEYS:
            np.testing.assert_array_equal(old[sample][key], new[sample][key])
    assert verify_dataset(out)["status"] == "verified"
    assert before == {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
    # Output files are not hardlinks: modifying the derivative cannot corrupt source.
    name = result["shards"][0]["arrays"]["history"]
    assert (out / name).stat().st_ino != (source / parent["shards"][0]["arrays"]["history"]).stat().st_ino


def test_derivation_is_deterministic_and_refuses_repeat_overwrite_or_empty_training(tmp_path):
    source = tmp_path / "source"
    fixture_dataset(source)
    a = derive_scene_holdout(source, tmp_path / "a")
    b = derive_scene_holdout(source, tmp_path / "b")
    assert a == b
    with pytest.raises(FileExistsError):
        derive_scene_holdout(source, tmp_path / "a")
    with pytest.raises(ValueError, match="Repeated"):
        derive_scene_holdout(tmp_path / "a", tmp_path / "repeat")
    with pytest.raises(ValueError, match="at least one"):
        derive_scene_holdout(source, tmp_path / "empty", scene_groups=4)
    with pytest.raises(ValueError, match="nested"):
        derive_scene_holdout(source, source / "nested")


def test_derivation_refuses_source_integrity_failure(tmp_path):
    source = tmp_path / "source"
    manifest = fixture_dataset(source)
    path = source / manifest["shards"][0]["arrays"]["goal"]
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x01")
    with pytest.raises(ValueError, match="SHA256"):
        derive_scene_holdout(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_copied_parent_at_another_machine_path_has_identical_manifest(tmp_path):
    source = tmp_path / "machine-one" / "corpus"
    source.parent.mkdir()
    fixture_dataset(source)
    copied = tmp_path / "machine-two" / "different" / "corpus"
    shutil.copytree(source, copied)
    first = derive_scene_holdout(source, tmp_path / "out-one")
    second = derive_scene_holdout(copied, tmp_path / "out-two")
    assert first == second
    assert sha256(tmp_path / "out-one" / "manifest.json") == sha256(tmp_path / "out-two" / "manifest.json")
    assert "parent_path_at_derivation" not in first["scene_holdout_derivation"]
