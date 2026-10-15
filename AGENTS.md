# Package working agreement

This is an independent non-RL pretraining side project. Keep the original Motion2Scene planner and running experiments unchanged.

- Use only positive source manifests. Never mix original development/fresh/sealed data into this package's training lineage.
- Split by recorded ancestry group, keeping every overlapping window and scene variant together. Freeze the manifest for comparisons.
- Data and checkpoints stay outside Git. Preserve upstream provenance and use the user's authorized private machine transport for dataset handoffs.
- Distinguish generated reference motion, kinematic compatibility, and physical execution. Do not invent action, collision, success, causal-attention, or uncertainty labels.
- Conditioning uses observed history, supplied scene, and supplied goal. Future targets are labels; only explicitly noised future tensors enter the flow decoder.
- Flow convention: data at time0, noise at time1, velocity noise-minus-data, integrate1→0.
- Normalize on train only. Use validation for choices, calibration only for fitting uncertainty calibration, and freeze choices before test reporting.
- New run directories only. Log complete/failed receipts with hashes and costs. Warm start is not exact resume.
- Run the CPU tests and a bounded real-data smoke check after changing contracts. Do not launch long GPU research runs merely to prepare a plan or handoff.
- Return code changes, configs, receipts and compact reports. Offline losses do not establish navigation or collision-avoidance gains.
