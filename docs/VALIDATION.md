# Initial package validation — 2026-10-15

This validates runnable software and transfer integrity. It is not a trained-model performance result.

- **37 CPU tests passed**, covering data ancestry/hash checks, overlap reconstruction, causal anchor velocities, masks, scene permutation invariance, no-scene/empty scenes, future-target isolation, gradients, flow direction, train-only normalization, variance calibration, and report provenance.
- The **full 20,000-row real dataset** passed schema, finite-value, source-lineage, split, and every-file SHA256 verification.
- Eight-step CPU smoke runs completed for masked reconstruction, forecasting, and flow matching. The flow run warm-started the masked checkpoint, exercising the stage handoff. Tiny model: width32, one layer,55,140 parameters. These runs establish pipeline operation only.
- Calibration on16 calibration windows and application on16 validation windows completed, with both receipts bound to the same checkpoint and data manifest. The package test split was not evaluated for these smoke checks.
- Context-only embedding export on235 validation windows completed.
- A self-contained HTML report and compact return archive were produced from the smoke run.
- Flow sampling/evaluation and standalone Python wheel packaging completed successfully.
- The prepared dataset transfer archive is bound by [`transfer_package.json`](../manifests/transfer_package.json).

Environment: Python3.11, PyTorch2.7.0+cu128, NumPy1.26.4. All checks ran with CUDA hidden and bounded CPU threads. No simulator, physical evaluation, long training run, or target-GPU memory benchmark was performed.

On the remote GPU, run the included dry-run and200-step throughput trial before setting longer budgets. Report full source/data/config/checkpoint hashes, actual peak memory and wall time; do not extrapolate quality from the smoke losses.
