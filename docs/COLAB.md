# Run in Google Colab

[Open the notebook](https://colab.research.google.com/github/linjiw/motion-scene-pretrain/blob/main/notebooks/MotionScene_Pretrain_Colab.ipynb).

1. Select **Runtime → Change runtime type → GPU**.
2. Put the separately supplied `kimodo-passage-h25-f100-v1.tar.xz` archive in `My Drive/motion-scene-pretrain/`. Alternatively select `DATA_SOURCE = upload` in the first cell and choose the file when prompted. The archive is approximately 435 MiB and is not hosted in the public code repository.
3. Leave `PROFILE = pilot` for the first run and use **Runtime → Run all**. Authorize Drive if requested. Change `SAVE_TO_DRIVE` to false if you prefer downloading completed results manually.

The notebook keeps Colab's installed PyTorch/CUDA build, installs the package without replacing its dependencies, checks the GPU, copies the archive to VM-local storage, verifies its pinned SHA256, safely extracts it, verifies all dataset files, and runs an eight-step CPU smoke check before the selected training recipe.

| Setting | Behavior |
|---|---|
| `smoke` | Tiny model, 64 training windows, 8 steps per stage; CPU permitted |
| `pilot` | Small model, 700 total optimizer steps; 200 masked + 500 downstream, or 700 direct |
| `full` | Small model, 15,000 total optimizer steps; 5,000 masked + 10,000 downstream, or 15,000 direct |
| `masked_then_forecast` | Default; learn masked motion structure, then future prediction and residual variance |
| `masked_then_flow` | Learn masked motion structure, then conditional motion generation |
| `forecast` / `flow` | Direct-learning comparisons |

Batch 16 is the starting point. GPU memory and runtime depend on the assigned device; inspect the recorded peak memory and elapsed time before scaling. Lower the batch size if needed, create a fresh experiment folder, and account for sample presentations when comparing runs. The notebook writes unique output folders and never overwrites incomplete training runs.

The data are unpacked into local `/content` storage for training. With Drive saving enabled, each **completed** stage is copied into `My Drive/motion-scene-pretrain/results/<run-id>/`. Checkpoints are written only at stage completion. A disconnected/deleted runtime can lose an unfinished stage; the notebook does not claim exact optimizer/RNG resume. Re-executing the training cell can reuse already completed stages in the same runtime after checking configuration and hashes.

Calibration uses the calibration split; evaluation uses validation. The package test split stays unused by default. Reports contain attention diagnostics, training curves, residual calibration, physical-unit prediction errors, and generative metrics when applicable. These remain offline results, not executed collision-avoidance measurements.

The final report cell downloads **`colab-return-report.tar.gz`**. Send this file back to the research chat. It includes the normal report, notebook settings, and every stage's config, receipt, metrics and total cost. Checkpoints are saved separately to Drive and can optionally be downloaded in the last cell. Neither source motion data nor model weights are placed in the report archive.

The notebook source is `notebooks/MotionScene_Pretrain_Colab.ipynb`; `python scripts/build_colab_notebook.py` regenerates it. Committed notebooks contain no executed outputs. The public GitHub code can be cloned without credentials; publishing the repository does not publish or change access to the dataset.

Colab storage and runtime behavior follow its [official FAQ](https://research.google.com/colaboratory/faq.html), which recommends transferring archives to VM storage before extracting and notes that runtime resources/lifetimes vary.

Validation on October 15, 2026: 51 CPU tests passed, the notebook passed Jupyter schema validation, and its preview/training/calibration/report cells completed a real-data CPU smoke run for masked-to-flow learning. Drive copies and browser downloads were simulated locally. A hosted Colab GPU session has not been executed as part of this check.
