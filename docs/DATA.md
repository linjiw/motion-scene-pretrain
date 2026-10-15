# Portable motion–scene corpus

The initial corpus contains **20,000 windows from 868 Kimodo-generated clips and 65 recorded shared-prompt ancestry groups**. Each window provides 0.5 seconds of observed motion and 2 seconds of future motion at 50 Hz. This is a reference-motion dataset with hindsight scene geometry. It is not a collection of executed controller actions, contact outcomes, or independently captured human responses to obstacles.

Only accepted `clean` or `repaired` clips in the immutable source manifest's original `train` split are eligible. Source `development`, `fresh`, and rejected clips are excluded. No AMASS, LAFAN1, raw Bones-SEED corpus, sealed benchmark, or later mixed-lineage arena derivative is added. The Kimodo generator's own upstream training ancestry is not established by this export; do not interpret the source groups as independent people or claim an out-of-distribution human dataset.

## What was exported

| Item | Count |
|---|---:|
| Source `passage-data-v3` windows | 189,124 |
| Accepted original-training clips | 868 |
| Original-training ancestry groups | 65 |
| Complete compatible/empty windows supporting H25/F100 | 47,020 |
| Packaged rows | 20,000 |
| Empty-scene rows | 16,552 |
| Archived-scene rows | 3,448 |
| Train rows / groups | 18,845 / 47 |
| Validation rows / groups | 235 / 6 |
| Calibration rows / groups | 585 / 6 |
| Test rows / groups | 335 / 6 |

The imbalance comes from the source's very large shared-prompt cluster. We retain its ancestry rather than divide correlated windows across splits. Report group-level results and uncertainty; six groups in each holdout support a pilot, not a broad generalization claim. All 868 eligible clips survive sampling. Source lineages are 773 route clips and 95 hindsight-120 clips, both from the Kimodo indoor_nav_1k collection.

The four package splits are **new splits within the original training cohort**. The package test set is not the Motion2Scene project's official test set. Split assignment is frozen in `manifest.json`; use that file, not a new random window split, for every comparison. All overlapping windows, scene variants, and derivatives of a clip remain with its group. Freeze the manifest when expanding the dataset: the exporter's ranked group assignment depends on the input group cohort and must not silently redefine an established split.

## Tensor contract

| Key | Per-sample shape | Meaning |
|---|---|---|
| `history` | `(25, 65)` float32 | Observed state history, including the anchor |
| `future` | `(100, 65)` float32 | Frames anchor+1 through anchor+100 |
| `scene` | `(8, 9)` float32 | Anchor-local obstacle tokens |
| `scene_mask` | `(8,)` bool | True means a real obstacle; padded tokens are invalid |
| `goal` | `(4,)` float32 | Local x/y destination and cosine/sine of heading difference |
| `future_mask` | `(100,)` bool | True means a valid future frame; all are valid in v1 |

State channels are root height `[0]`, gravity in root frame `[1:4]`, planar velocity in heading frame `[4:6]`, yaw rate `[6]`, 29 joint positions `[7:36]`, and 29 joint velocities `[36:65]`. Positions use metres, angles radians, derivatives per second. Global xy and yaw are deliberately absent; scene geometry is already transformed into the source row's anchor frame. Obstacle channels are `cx, cy, cz, yaw, hx, hy, hz, radius, sphere_flag`; box half extents are in metres. Empty scenes contain eight masked zero tokens.

The exact ordered joint names and channel conventions are recorded in [`manifests/state_schema.json`](../manifests/state_schema.json), bound to the source clean-reference receipt hash. Preserve that joint order when transferring representations back to the Flow planner.

The source goal is the reference pose **50 frames (1 second) ahead of the anchor**, clipped at the source clip end. It is retrospective conditioning, not an independently recorded navigation instruction. Preserve that meaning in reports; measure goal-only and map-shuffled baselines because this condition can explain much of the future. It is neither a 2-second target nor the planner's deployment route goal.

Source targets use forward finite differences. The source anchor's forward velocity depends on the first future pose, so the exporter replaces the **last observed history frame's** velocity channels with past estimates. Previous planar velocity is rotated from the previous heading into the current heading; yaw rate and joint velocity are estimated from the preceding observed interval. Earlier history derivatives only use poses still within the observed history. The future retains the source forward-difference convention. Do not compare velocity errors across these conventions without accounting for their time alignment.

## Reconstruction and provenance

The source archive contains 4-history/25-future windows with stride 5. Their overlapping state values are checked for exact equality and reconstructed into each clip's covered timeline. New H25/F100 windows are cut only where all 125 frames are available. No extrapolation, padded future, clip concatenation, new pose interpolation, or scene re-anchoring is used. The new anchor equals the original scene row's anchor. The original scene geometry and goal are copied unchanged.

`manifest.json` binds the source archive, clean-reference manifest, source receipt, exporter, group assignment, every array, and row metadata by SHA256. `source_clips.json` records clip/group IDs, original split, source reference SHA256, original path, cohort, and preprocessing tier. Every source reference hash is verified before export. `samples.jsonl` records each source row, source start frame, anchor frame, output shard/row, and same-anchor empty-twin availability. The source clean-reference receipt binds the lineage manifest; a caller cannot relabel official evaluation clips as training without breaking this check.

Sampling uses a deterministic round-robin over ancestry groups and keeps complete same-anchor bundles of compatible archived scenes plus empty scene. Every archived scene in the v1 export retains its empty twin; this is a **same-motion scene comparison**, not evidence that the motion changes appropriately with the scene. Witness and too-late rows pair a colliding motion with an obstacle and are excluded from positive generative supervision. A later separately typed diagnostic export could retain them as incompatibility examples, with explicit geometry-only labels and the same ancestry split.

Compatible archived scenes were selected from hindsight reference geometry. That selection does not establish necessity, causal attention, physical safety, or successful SONIC execution. Repaired joint-limit clips can change geometry, and the inherited source height datum is not a SONIC grounding correction. Do not send this package directly to a controller and treat training losses as a qualification result.

## Layout and usage

```text
data/kimodo-passage-h25-f100-v1/
  manifest.json             # Schema, source hashes, splits, every-file hashes
  source_clips.json         # Positive source ancestry
  samples.jsonl             # One auditable provenance record per example
  shards/train-0000/
    history.npy
    future.npy
    scene.npy
    scene_mask.npy
    goal.npy
    future_mask.npy
  shards/val-0000/ ...      # The same numeric layout for each split
```

Files contain only numeric NumPy arrays and JSON; pickle is never used. Uncompressed `.npy` arrays permit read-only memory mapping without repeatedly decompressing a shard for shuffled batches. A compressed transfer archive reduces transport size. The loader copies each selected row before returning a tensor, so training cannot modify the on-disk data. It carries no Flow, Isaac, robot asset, ONNX, or MuJoCo dependency.

```python
from motion_scene_pretrain.data import MixedSceneDataset, verify_dataset

print(verify_dataset("data/kimodo-passage-h25-f100-v1"))
train = MixedSceneDataset("data/kimodo-passage-h25-f100-v1", "train")
sample = train[0]
```

Fit normalizers using only package `train`. Select models using `val`; fit any post-hoc uncertainty calibration using `calibration`; report `test` once the protocol is fixed. Avoid estimating normalization, augmentation distributions, stopping decisions, or hyperparameters from calibration/test. Calibration with only six groups is exploratory, and correlated windows do not provide independent calibration guarantees.

## Reproduce or expand on the source machine

With the package installed, run this CPU-only dry run before export:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python scripts/export_data.py \
  --source /home/linjiw/flow/artifacts/passage-data-v3 \
  --lineage /home/linjiw/flow/artifacts/clean-reference-v2/manifest.json \
  --output data/kimodo-passage-h25-f100-v1 \
  --history 25 --horizon 100 --max-samples 20000 --dry-run
```

Remove `--dry-run` to create a fresh directory. Existing directories are refused. Use `--max-samples 0` and a new output path to export all 47,020 eligible rows from this source. This preserves the same input ancestry cohort and therefore the same group assignments. Do not add broad artifact globs: new sources must receive positive ancestry manifests and explicit split qualification first.

Data stay outside Git. The repository grants no new rights to upstream data. The prepared bundle is for the user's own authorized research machine; no data are published to GitHub or uploaded by this exporter. Preserve the manifest and inherited source restrictions during transfer. Remote researchers should return code changes, configuration, checkpoint metadata, metrics grouped by source ancestry, calibration plots, and a compact report—not duplicate source assets in a pull request.
