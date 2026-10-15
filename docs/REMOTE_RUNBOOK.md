# Remote machine handoff

Target: one 16–24 GB GPU. The initial package is CPU-smoke-tested; GPU throughput and memory are not yet measured. Python 3.11, PyTorch 2.7.0 and NumPy 1.26.4 were used locally. Install a CUDA-compatible PyTorch build using its official installation selector before installing this package. Neither the dataset loader nor the model imports the original project or a simulator.

## Transfer

Clone the public repository using HTTPS. Transfer `kimodo-passage-h25-f100-v1.tar.xz` and its `.sha256` sidecar using your usual private `scp`/`rsync` route. These two local files are the data handoff; the Git repository contains code, schema, and documentation. For Google Colab, use the [notebook instructions](COLAB.md) instead of installing a separate CUDA environment. The repository is not a dataset hosting service.

```bash
# Run from the directory containing the received archive and checksum.
sha256sum -c kimodo-passage-h25-f100-v1.tar.xz.sha256
mkdir -p data
tar -xJf kimodo-passage-h25-f100-v1.tar.xz -C data
python scripts/verify_data.py data/kimodo-passage-h25-f100-v1
```

`manifest.json` must hash to `1d65ecce47b37dbb3ffce6b651484f79f4567a5a15f929f80c79c7bb56c8659a`. The loader memory-maps numeric shards. Allow at least 2 GB for code, the compressed archive, expanded data, and small-model checkpoints, in addition to the Python/PyTorch/CUDA environment; retain more storage for multiple experiments. The source archive is not needed on the remote machine.

Keep every output directory fresh. The commands refuse existing output directories to protect receipts. A failure creates a failed receipt when possible; only `status: complete` is a finished run.

## Smoke check and measure the target GPU

```bash
OMP_NUM_THREADS=2 pytest -q
python -m motion_scene_pretrain.train \
  --config configs/smoke.json --data data/kimodo-passage-h25-f100-v1 \
  --output runs/cpu-smoke

python -m motion_scene_pretrain.train \
  --config configs/forecast_small.json --data data/kimodo-passage-h25-f100-v1 \
  --output runs/gpu-plan --dry-run
python -m motion_scene_pretrain.train \
  --config configs/forecast_small.json --steps 200 \
  --data data/kimodo-passage-h25-f100-v1 --output runs/gpu-trial
```

Check the trial's elapsed time and `max_cuda_allocated_bytes` in its receipt. The implementation uses float32 and does not silently fall back from CUDA to CPU. Reduce batch size in a copied, versioned configuration if necessary. Do not use a speed trial's checkpoint as a selected starting point for a comparison: start each registered arm from its stated seed or common initialization.

## Train and compare

Use the README's scene-aware, scene-disabled, and masked-initialized forecast commands. `group_scene` sampling gives each ancestry group equal total probability, then balances empty/archived scene categories where both exist. It does not fabricate motion alternatives, and its sampling distribution differs from the raw window distribution. Use `uniform_window` in a copied config to measure the sampling intervention separately.

For the generative comparison:

```bash
python -m motion_scene_pretrain.train \
  --config configs/flow_small.json --seed 0 \
  --data data/kimodo-passage-h25-f100-v1 --output runs/flow-s0
python -m motion_scene_pretrain.train \
  --config configs/flow_small.json --steps 10000 --seed 0 \
  --init runs/masked-s0/checkpoint.pt \
  --data data/kimodo-passage-h25-f100-v1 --output runs/pretrained-flow-s0
```

The second arm includes the earlier 5k masked steps. `configs/flow_medium.json` changes width, depth, and batch size and must be trained as a separate capacity/cost comparison. Its shape differs from small checkpoints. Do not warm-start it with `--init` from the small model.

## Validate, calibrate, then freeze the final evaluation

Replace the checkpoint path consistently if using another arm. Validation scene removal/shuffling measures sensitivity; it is not evidence of appropriate response. Reports separate 0–1 s and 1–2 s, and empty versus archived scenes. Attention weights are last-layer head averages; the last token is null. They are not causal explanations.

```bash
python -m motion_scene_pretrain.evaluate \
  --checkpoint runs/forecast-s0/checkpoint.pt \
  --data data/kimodo-passage-h25-f100-v1 --split val \
  --output runs/forecast-s0-val --device cuda

python -m motion_scene_pretrain.evaluate \
  --checkpoint runs/forecast-s0/checkpoint.pt \
  --data data/kimodo-passage-h25-f100-v1 --split calibration --calibrate \
  --output runs/forecast-s0-calibration --device cuda

# Run after configuration/checkpoint selection is frozen.
python -m motion_scene_pretrain.evaluate \
  --checkpoint runs/forecast-s0/checkpoint.pt \
  --data data/kimodo-passage-h25-f100-v1 --split test \
  --calibration runs/forecast-s0-calibration/calibration.json \
  --output runs/forecast-s0-test --device cuda
```

The Gaussian head is trained separately from noisy-future flow calls. Calibration rescales its variance on an ancestry-disjoint calibration split; it provides no finite-sample coverage guarantee. Coverage is marginal over correlated state elements, not joint trajectory coverage, collision probability, or independent sample count. Compare NLL, errors, coverage and interval size together. Flow sample spread and best-of-K reference error describe different quantities; oracle best-of-K cannot be deployed without a selector.

The internal holdouts are small (six groups each). Keep primary group-level analysis and larger independent-scene replication in the research plan. A low MSE alone can favor an averaged, unexecutable motion. Planner integration still requires physical and motion-quality qualification.

## Send the result back

```bash
python -m motion_scene_pretrain.report \
  --run runs/forecast-s0 \
  --evaluations runs/forecast-s0-val runs/forecast-s0-calibration runs/forecast-s0-test \
  --output runs/forecast-s0-report
```

Send `runs/forecast-s0-report/return-report.tar.gz` back to the original machine. It contains training/configuration/provenance records, evaluation metrics, and a standalone HTML report; it excludes motion arrays and model weights. Report negative findings and failed runs alongside improvements. Include the GPU model, wall time, any configuration or code changes, and whether comparisons used equal data, presentations, and compute. The original research chat can inspect the returned archive and decide which checkpoints warrant integration.

## Data and research extensions

The current data permit learning motion regularities and scene compatibility. To learn *when to change*, collect qualified same-history alternatives for clear, narrow, overhead, and irrelevant obstacles; include approach, passage and recovery. Do not label attention or uncertainty from proximity alone. Add near-obstacle and transition-focused evaluation only after defining trustworthy geometry/context labels. Add ensemble disagreement as a separate mechanism, and use executed state/action/outcome records before training physical risk prediction.
