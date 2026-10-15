#!/usr/bin/env python3
"""Package one verified dataset for private machine-to-machine transport."""
import argparse
import json
import tarfile
from pathlib import Path
from motion_scene_pretrain.data import sha256, verify_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
source, output = Path(args.data), Path(args.output)
if output.exists():
    raise FileExistsError(output)
result = verify_dataset(source)
output.parent.mkdir(parents=True, exist_ok=True)
manifest = json.loads((source / "manifest.json").read_text())
if str(output).endswith(".tar.xz"):
    mode, options = "w:xz", {"preset": 6}
elif str(output).endswith(".tar.gz"):
    mode, options = "w:gz", {"compresslevel": 6}
else:
    raise ValueError("Output must end in .tar.xz or .tar.gz")
with tarfile.open(output, mode, **options) as archive:
    for name in ["manifest.json", *sorted(manifest["files"])]:
        archive.add(source / name, arcname=f"{source.name}/{name}", recursive=False)
digest = sha256(output)
output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
print(json.dumps({**result, "archive": str(output), "bytes": output.stat().st_size, "sha256": digest}, indent=2))
