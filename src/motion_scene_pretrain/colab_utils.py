"""Verified local dataset preparation for an ephemeral notebook runtime."""
from __future__ import annotations

import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .data import sha256, verify_dataset


def _verified_dataset(path: Path, manifest_hash: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Dataset destination must be a regular directory: {path}")
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError(f"Dataset contains a symbolic link: {path}")
    if sha256(path / "manifest.json") != manifest_hash:
        raise ValueError("Dataset manifest SHA256 differs from the transfer metadata")
    verify_dataset(path)


def prepare_dataset(archive_path: str | Path, data_root: str | Path,
                    transfer_metadata: dict) -> Path:
    """Verify, safely unpack, and atomically publish one approved data archive.

    Metadata is the repository's ``manifests/transfer_package.json``. An existing
    destination is reused only after its pinned manifest and every dataset file
    pass verification; an incomplete/corrupt destination is never overwritten.
    Extraction stays in a temporary sibling until all checks pass. The archive
    need not remain available when reusing an already verified destination.
    """
    name = transfer_metadata["dataset_directory"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("dataset_directory must be a single safe directory name")
    archive_hash = transfer_metadata["archive_sha256"]
    manifest_hash = transfer_metadata["dataset_manifest_sha256"]
    if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
           for h in (archive_hash, manifest_hash)):
        raise ValueError("Transfer metadata must contain lowercase SHA256 digests")
    expanded_bytes = transfer_metadata["expanded_bytes"]
    if type(expanded_bytes) is not int or expanded_bytes < 0:
        raise ValueError("expanded_bytes must be a nonnegative integer")
    root = Path(data_root).expanduser().resolve()
    target = root / name
    if target.exists() or target.is_symlink():
        _verified_dataset(target, manifest_hash)
        return target

    archive_path = Path(archive_path).expanduser()
    if "archive_bytes" in transfer_metadata:
        if archive_path.stat().st_size != transfer_metadata["archive_bytes"]:
            raise ValueError("Archive size differs from the transfer metadata")
    if sha256(archive_path) != archive_hash:
        raise ValueError("Archive SHA256 differs from the transfer metadata")
    root.mkdir(parents=True, exist_ok=True)
    # The published expanded-byte count can exclude manifest.json. Allow only a
    # bounded 1 MiB overhead, rather than trusting size claims inside the tar.
    byte_limit = expanded_bytes + 1024 * 1024
    with tarfile.open(archive_path, "r:*") as archive:
        members, seen, total_bytes = [], set(), 0
        for member in archive:
            if len(members) >= 100_000:
                raise ValueError("Archive contains too many entries")
            raw = member.name.rstrip("/") if member.isdir() else member.name
            relative = PurePosixPath(raw)
            if (not raw or relative.is_absolute() or "\\" in raw
                    or any(part in {"", ".", ".."} for part in raw.split("/"))):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if relative.parts[0] != name:
                raise ValueError(f"Unexpected archive root: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive links and special entries are forbidden: {member.name}")
            if len(relative.parts) == 1 and not member.isdir():
                raise ValueError("Archive root must be a directory")
            if relative in seen:
                raise ValueError(f"Duplicate archive path: {member.name}")
            seen.add(relative)
            if member.size < 0:
                raise ValueError("Archive has a negative entry size")
            total_bytes += member.size
            if total_bytes > byte_limit:
                raise ValueError("Archive exceeds the approved expanded-byte limit")
            members.append((member, relative))
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
            stage = Path(temporary)
            for member, relative in members:
                destination = stage.joinpath(*relative.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            staged_dataset = stage / name
            _verified_dataset(staged_dataset, manifest_hash)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Dataset destination appeared during extraction: {target}")
            staged_dataset.rename(target)
    return target
