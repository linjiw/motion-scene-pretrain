"""Export context-only representations for offline linear probes or retrieval."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import checkpoint_load, collate, dump_json, prepare, sha256
from .data import MixedSceneDataset
from .model import ModelConfig, MotionSceneModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["train", "val", "calibration", "test"], default="val")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = checkpoint_load(args.checkpoint)
    if sha256(Path(args.data) / "manifest.json") != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint/dataset mismatch")
    device = torch.device(args.device)
    model = MotionSceneModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = MixedSceneDataset(args.data, args.split)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    vectors = []
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate):
            batch = prepare(raw, checkpoint["normalizer"], device)
            # Future tensors only determine no input: horizon=1, no future_input supplied.
            prediction = model(batch["history"], batch["scene"], batch["goal"],
                               horizon=1, scene_mask=batch["scene_mask"])
            vectors.append(prediction["representation"].cpu().numpy())
    np.save(out / "representations.npy", np.concatenate(vectors), allow_pickle=False)
    with (out / "records.jsonl").open("w") as stream:
        for record in dataset.records:
            stream.write(json.dumps(record) + "\n")
    dump_json(out / "receipt.json", {"status": "complete", "split": args.split,
              "windows": len(dataset), "checkpoint_sha256": sha256(args.checkpoint),
              "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
              "representation_sha256": sha256(out / "representations.npy"),
              "condition": "observed history, supplied scene and supplied goal; no target future"})


if __name__ == "__main__":
    main()
