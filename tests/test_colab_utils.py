import argparse
import io
import tarfile

import pytest

from motion_scene_pretrain.colab_utils import prepare_dataset
from motion_scene_pretrain.data import sha256, verify_dataset
from test_data import _fixture_source, exporter


@pytest.fixture
def transfer(tmp_path):
    source, lineage = _fixture_source(tmp_path)
    dataset = tmp_path / "fixture-dataset"
    exporter.export(argparse.Namespace(source=source, lineage=lineage, output=dataset,
                                      history=5, horizon=10, max_samples=0, seed=1,
                                      shard_rows=2, dry_run=False))
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(dataset, arcname=dataset.name)
    metadata = {
        "dataset_directory": dataset.name,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "expanded_bytes": sum(p.stat().st_size for p in dataset.rglob("*") if p.is_file()),
    }
    return archive, metadata


def test_prepares_and_fully_reverifies_existing_dataset(tmp_path, transfer):
    archive, metadata = transfer
    root = tmp_path / "runtime"
    prepared = prepare_dataset(archive, root, metadata)
    assert prepared == root / metadata["dataset_directory"]
    assert verify_dataset(prepared)["status"] == "verified"
    archive.unlink()  # A fully verified copy can be reused without its archive.
    assert prepare_dataset(archive, root, metadata) == prepared
    array = next(prepared.rglob("*.npy"))
    with array.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="Size mismatch"):
        prepare_dataset(archive, root, metadata)
    assert array.read_bytes().endswith(b"corruption")  # Never replace a bad destination.


def test_refuses_bad_archive_or_manifest_hash_without_publishing(tmp_path, transfer):
    archive, metadata = transfer
    root = tmp_path / "runtime"
    with pytest.raises(ValueError, match="Archive SHA256"):
        prepare_dataset(archive, root, {**metadata, "archive_sha256": "0" * 64})
    assert not root.exists()
    with pytest.raises(ValueError, match="manifest SHA256"):
        prepare_dataset(archive, root, {**metadata, "dataset_manifest_sha256": "0" * 64})
    assert list(root.iterdir()) == []


def test_refuses_existing_partial_directory(tmp_path, transfer):
    archive, metadata = transfer
    root = tmp_path / "runtime"
    destination = root / metadata["dataset_directory"]
    destination.mkdir(parents=True)
    (destination / "partial.txt").write_text("retain me")
    with pytest.raises(FileNotFoundError):
        prepare_dataset(archive, root, metadata)
    assert (destination / "partial.txt").read_text() == "retain me"


@pytest.mark.parametrize("name,kind,reason", [
    ("../escape", tarfile.REGTYPE, "Unsafe archive path"),
    ("/escape", tarfile.REGTYPE, "Unsafe archive path"),
    ("fixture-dataset/../escape", tarfile.REGTYPE, "Unsafe archive path"),
    ("other-root/file", tarfile.REGTYPE, "Unexpected archive root"),
    ("fixture-dataset/link", tarfile.SYMTYPE, "special entries"),
    ("fixture-dataset/link", tarfile.LNKTYPE, "special entries"),
    ("fixture-dataset/device", tarfile.CHRTYPE, "special entries"),
])
def test_rejects_unsafe_tar_before_extracting(tmp_path, name, kind, reason):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
            member.linkname = "../../outside"
        tar.addfile(member)
    metadata = {"dataset_directory": "fixture-dataset", "archive_sha256": sha256(archive),
                "dataset_manifest_sha256": "0" * 64, "expanded_bytes": 0}
    root = tmp_path / "runtime"
    with pytest.raises(ValueError, match=reason):
        prepare_dataset(archive, root, metadata)
    assert list(root.iterdir()) == []
    assert not (tmp_path / "escape").exists()


def test_rejects_expansion_above_pinned_budget(tmp_path):
    archive = tmp_path / "oversized.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("fixture-dataset/oversized")
        member.size = 1024 * 1024 + 1
        tar.addfile(member, io.BytesIO(b"\0" * member.size))
    metadata = {"dataset_directory": "fixture-dataset", "archive_sha256": sha256(archive),
                "dataset_manifest_sha256": "0" * 64, "expanded_bytes": 0}
    root = tmp_path / "runtime"
    with pytest.raises(ValueError, match="expanded-byte limit"):
        prepare_dataset(archive, root, metadata)
    assert list(root.iterdir()) == []
