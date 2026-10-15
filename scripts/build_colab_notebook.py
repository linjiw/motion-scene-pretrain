"""Rebuild the checked-in Colab notebook without requiring Jupyter tooling."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source, name):
    source = textwrap.dedent(source).strip() + "\n"
    item = {"cell_type": kind, "id": name, "metadata": {"id": name},
            "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
        item["metadata"]["cellView"] = "form"
    cells.append(item)


cell("markdown", """
# Motion–Scene Pretraining · Google Colab

Learn motion representations and two-second futures **without RL**. This notebook installs the public package, verifies your dataset, runs a short pilot, and creates a report to send back.

**Before Run all**
1. Select **Runtime → Change runtime type → GPU**.
2. Put `kimodo-passage-h25-f100-v1.tar.xz` in your Google Drive folder `motion-scene-pretrain`, or select **upload** below and choose the archive when prompted. The archive is about **435 MiB**. It is supplied separately from the public code repository.
3. Leave **pilot** selected for the first run. Authorize Drive when prompted if using it.

The data contain generated reference motions with hindsight scenes, not physical collision labels. A lower training loss does not establish navigation performance.

[Repository](https://github.com/linjiw/motion-scene-pretrain) · [Research design](https://github.com/linjiw/motion-scene-pretrain/blob/main/docs/RESEARCH.md) · [Colab storage/runtime guidance](https://research.google.com/colaboratory/faq.html)
""", "intro")

cell("code", '''
#@title 1. Settings
DATA_SOURCE = "drive" #@param ["drive", "upload"]
DRIVE_ARCHIVE_PATH = "/content/drive/MyDrive/motion-scene-pretrain/kimodo-passage-h25-f100-v1.tar.xz" #@param {type:"string"}
SAVE_TO_DRIVE = True #@param {type:"boolean"}
DRIVE_RESULTS_FOLDER = "/content/drive/MyDrive/motion-scene-pretrain/results" #@param {type:"string"}
PROFILE = "pilot" #@param ["smoke", "pilot", "full"]
PIPELINE = "masked_then_forecast" #@param ["forecast", "masked_then_forecast", "flow", "masked_then_flow"]
BATCH_SIZE = 16 #@param {type:"integer"}
SEED = 0 #@param {type:"integer"}
DEVICE = "cuda" #@param ["cuda", "cpu"]

# smoke: 8 steps/stage, tiny model, small data subset, CPU allowed.
# pilot: 700 total steps (200 masked + 500 forecast/flow, or 700 direct).
# full: 15,000 total steps (5,000 masked + 10,000 forecast/flow, or 15,000 direct).
# Each stage saves a checkpoint at completion. An interrupted stage must restart;
# this notebook does not implement exact optimizer/RNG resume.
assert PROFILE in {"smoke", "pilot", "full"}
assert PIPELINE in {"forecast", "masked_then_forecast", "flow", "masked_then_flow"}
assert isinstance(BATCH_SIZE, int) and 1 <= BATCH_SIZE <= 128
if DEVICE == "cpu" and PROFILE != "smoke":
    raise ValueError("Choose PROFILE='smoke' for CPU, or select a GPU runtime for training.")
''', "settings")

cell("code", '''
#@title 2. Install the public package and check the runtime
import os, sys, json, shutil, subprocess, uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/linjiw/motion-scene-pretrain.git"
WORK_ROOT = Path("/content")
REPO = WORK_ROOT / "motion-scene-pretrain"
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
else:
    origin = subprocess.check_output(["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True).strip()
    if origin.rstrip("/") != REPO_URL:
        raise RuntimeError(f"Existing folder belongs to another repository: {REPO}")
    print("Reusing existing checkout; no local files will be reset.")

# Preserve Colab's preinstalled CUDA-enabled PyTorch and NumPy.
import torch
import numpy as np
if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 4):
    raise RuntimeError("PyTorch >=2.4 is required; use a current Colab runtime.")
if tuple(int(x) for x in np.__version__.split(".")[:2]) < (1, 26):
    raise RuntimeError("NumPy >=1.26 is required; use a current Colab runtime.")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(REPO)], check=True)
# Make the editable package visible immediately, without a kernel restart.
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
os.environ["OMP_NUM_THREADS"] = "2"
os.chdir(REPO)
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("No GPU is attached. Choose Runtime → Change runtime type → GPU, then rerun; or use CPU + smoke.")
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print({"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "code_commit": revision})
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}; VRAM: {props.total_memory / 2**30:.1f} GiB")
print(f"Local disk free: {shutil.disk_usage('/content').free / 2**30:.1f} GiB")
''', "setup")

cell("code", '''
#@title 3. Load the dataset archive
from google.colab import drive, files
import tempfile
from motion_scene_pretrain.data import sha256

if DATA_SOURCE == "drive" or SAVE_TO_DRIVE:
    drive.mount("/content/drive")
transfer = json.loads((REPO / "manifests/transfer_package.json").read_text())
CACHE = Path("/content/motion-scene-cache")
CACHE.mkdir(exist_ok=True)
ARCHIVE = CACHE / transfer["archive"]

if ARCHIVE.exists() and sha256(ARCHIVE) == transfer["archive_sha256"]:
    print("Reusing the already verified local archive.")
elif DATA_SOURCE == "drive":
    source = Path(DRIVE_ARCHIVE_PATH)
    if not source.is_file():
        raise FileNotFoundError(f"Upload the supplied dataset archive to {source}, or choose DATA_SOURCE='upload'.")
    shutil.copyfile(source, ARCHIVE)
elif DATA_SOURCE == "upload":
    print(f"Choose {transfer['archive']} in the file picker.")
    # An empty upload folder avoids Colab renaming the file on retries.
    with tempfile.TemporaryDirectory(prefix="motion-scene-upload-") as folder:
        previous = Path.cwd()
        os.chdir(folder)
        try:
            uploaded = files.upload()
        finally:
            os.chdir(previous)
        if set(uploaded) != {transfer["archive"]}:
            raise ValueError(f"Choose only the archive named {transfer['archive']}")
        shutil.copyfile(Path(folder)/transfer["archive"], ARCHIVE)
        del uploaded  # Release the browser upload's in-memory bytes.
else:
    raise ValueError("DATA_SOURCE must be drive or upload")
print(f"Local archive: {ARCHIVE} ({ARCHIVE.stat().st_size / 2**20:.1f} MiB)")
''', "load-data")

cell("code", '''
#@title 4. Verify and unpack onto Colab's local disk
from motion_scene_pretrain.colab_utils import prepare_dataset
from motion_scene_pretrain.data import MixedSceneDataset

DATA = prepare_dataset(ARCHIVE, Path("/content/motion-scene-data"), transfer)
manifest = json.loads((DATA / "manifest.json").read_text())
print("Dataset verified:", DATA)
print(json.dumps({"windows": manifest["samples"], "splits": manifest["counts"],
                  "history_frames": manifest["history"], "future_frames": manifest["horizon"],
                  "fps": manifest["fps"]}, indent=2))
''', "verify-data")

cell("code", '''
#@title 5. Inspect one training pair
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D

dataset = MixedSceneDataset(DATA, "train")
example_index = next((i for i,r in enumerate(dataset.records) if r["scene_id"] != "empty"), 0)
example = dataset[example_index]
past = example["history"].numpy()
future = example["future"].numpy()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(np.arange(1-len(past), 1)/manifest["fps"], past[:, 0], label="Observed history")
ax[0].plot(np.arange(1,len(future)+1)/manifest["fps"], future[:, 0], label="Reference future")
ax[0].axvline(0, color="gray", linestyle="--")
ax[0].set(xlabel="Time relative to anchor (seconds)", ylabel="Root height (m)", title="What the model predicts")
ax[0].legend()
for row in example["scene"][example["scene_mask"]].numpy():
    cx,cy,cz,yaw,hx,hy,hz,r,sphere = row
    if sphere > .5:
        shape = Circle((cx,cy), r, color="tab:orange", alpha=.4)
    else:
        shape = Rectangle((cx-hx,cy-hy),2*hx,2*hy,color="tab:orange",alpha=.4)
        shape.set_transform(Affine2D().rotate_around(cx,cy,yaw) + ax[1].transData)
    ax[1].add_patch(shape)
ax[1].scatter([0], [0], label="Motion anchor", color="tab:blue")
ax[1].scatter([example["goal"][0]], [example["goal"][1]], marker="*", s=100, label="Supplied goal")
ax[1].autoscale_view()
ax[1].set_aspect("equal", adjustable="datalim")
ax[1].set(xlabel="Local x (m)", ylabel="Local y (m)", title="Obstacle footprints · heights omitted")
ax[1].legend()
plt.tight_layout()
plt.show()
''', "preview")

cell("code", '''
#@title 6. Create this experiment's output folder and run a CPU smoke check
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
RUN_ROOT = WORK_ROOT / "motion-scene-runs" / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=False)
DRIVE_RUN = Path(DRIVE_RESULTS_FOLDER) / RUN_ID if SAVE_TO_DRIVE else None

def run_command(arguments):
    # Stream progress into the notebook and fail immediately on an unsuccessful command.
    subprocess.run([str(x) for x in arguments], cwd=REPO, check=True)

def save_to_drive(folder):
    if DRIVE_RUN is not None:
        DRIVE_RUN.mkdir(parents=True, exist_ok=True)
        destination = DRIVE_RUN / Path(folder).resolve().relative_to(RUN_ROOT.resolve())
        shutil.copytree(folder, destination, dirs_exist_ok=True)
        print("Saved to Drive:", destination)

(RUN_ROOT / "notebook-settings.json").write_text(json.dumps({
    "profile": PROFILE, "pipeline": PIPELINE, "seed": SEED, "batch_size": BATCH_SIZE,
    "device": DEVICE, "commit": revision, "dataset_manifest_sha256": transfer["dataset_manifest_sha256"]
}, indent=2))
if DRIVE_RUN is not None:
    DRIVE_RUN.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(RUN_ROOT/"notebook-settings.json",DRIVE_RUN/"notebook-settings.json")
run_command([sys.executable, "-m", "motion_scene_pretrain.train", "--config", REPO/"configs/smoke.json",
             "--data", DATA, "--output", RUN_ROOT/"cpu-smoke"])
save_to_drive(RUN_ROOT/"cpu-smoke")
print("Experiment folder:", RUN_ROOT)
''', "smoke")

cell("code", '''
#@title 7. Prepare the selected training recipe
pretrain = PIPELINE.startswith("masked_then_")
final_mode = "flow" if PIPELINE.endswith("flow") else "forecast"
if PROFILE == "smoke":
    pretrain_steps, final_steps = 8, 8
elif PROFILE == "pilot":
    pretrain_steps, final_steps = 200, (500 if pretrain else 700)
else:
    pretrain_steps, final_steps = 5000, (10000 if pretrain else 15000)

CONFIG_DIR = RUN_ROOT / "configs"
CONFIG_DIR.mkdir(exist_ok=True)
stages = []
for index, (mode, steps) in enumerate(([('masked',pretrain_steps)] if pretrain else []) + [(final_mode,final_steps)], 1):
    filename = f"{mode}_small.json"
    cfg = json.loads((REPO/"configs"/filename).read_text())
    cfg.update(device=DEVICE, seed=SEED, steps=steps, batch_size=BATCH_SIZE,
               workers=0, cpu_threads=2, log_every=25 if PROFILE != "full" else 100,
               eval_every=steps, val_batches=2 if PROFILE != "full" else 8)
    if PROFILE == "smoke":
        cfg.update(model={"width":32,"heads":4,"layers":1,"fps":50.0},
                   max_train_samples=64, max_val_samples=16, batch_size=min(BATCH_SIZE,4))
    config_path = CONFIG_DIR / f"{index:02d}-{mode}.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    stages.append({"mode":mode, "config":config_path, "output":RUN_ROOT/f"{index:02d}-{mode}"})
    run_command([sys.executable,"-m","motion_scene_pretrain.train","--config",config_path,
                 "--data",DATA,"--output",stages[-1]["output"],"--dry-run"])
save_to_drive(CONFIG_DIR)
print("Total training steps:", sum(json.loads(s["config"].read_text())["steps"] for s in stages))
''', "plan")

cell("markdown", """
### Train

The next cell starts the selected recipe. **Pilot** checks the workflow; it is not a full research comparison. Completed stages are copied to Drive when enabled. The trainer writes its checkpoint at the end of each stage, so interruption before that point requires restarting that stage. Masked → forecast/flow is a warm start with a new optimizer.

If CUDA runs out of memory, lower `BATCH_SIZE` in Settings and rerun from the experiment-folder cell to create a fresh run. Keep sample counts and budgets matched when comparing models.
""", "train-intro")

cell("code", '''
#@title 8. Start training
previous_checkpoint = None
for stage in stages:
    output = stage["output"]
    command = [sys.executable,"-m","motion_scene_pretrain.train","--config",stage["config"],
               "--data",DATA,"--output",output]
    if previous_checkpoint is not None:
        command += ["--init",previous_checkpoint]
    if output.exists():
        receipt_path = output/"receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        requested = json.loads(stage["config"].read_text())
        if (receipt.get("status") != "complete" or receipt.get("config") != requested
            or receipt.get("dataset_manifest_sha256") != transfer["dataset_manifest_sha256"]
            or receipt.get("initialized_from_sha256") != (sha256(previous_checkpoint) if previous_checkpoint is not None else None)
            or not (output/"checkpoint.pt").is_file()
            or receipt.get("checkpoint_sha256") != sha256(output/"checkpoint.pt")):
            raise RuntimeError(f"Incomplete or different run at {output}; create a fresh experiment folder.")
        print("Reusing this completed stage:", output)
    else:
        try:
            run_command(command)
        finally:
            if output.exists():
                save_to_drive(output)
    previous_checkpoint = output/"checkpoint.pt"
FINAL_RUN = stages[-1]["output"]
CHECKPOINT = previous_checkpoint
print("Trained checkpoint:", CHECKPOINT)
''', "train")

cell("code", '''
#@title 9. Inspect training progress
fig, axes = plt.subplots(1,len(stages),figsize=(6*len(stages),3.5),squeeze=False)
for axis,stage in zip(axes[0],stages):
    rows = [json.loads(line) for line in (stage["output"]/"metrics.jsonl").read_text().splitlines()]
    axis.plot([r["step"] for r in rows],[r["loss"] for r in rows],marker=".")
    axis.set(xlabel="Optimizer step",ylabel="Training objective",title=stage["mode"])
plt.tight_layout()
plt.show()
receipt = json.loads((FINAL_RUN/"receipt.json").read_text())
print({"elapsed_seconds": receipt["elapsed_seconds"],
       "peak_cuda_allocated_GiB": receipt["max_cuda_allocated_bytes"]/2**30})
''', "curves")

cell("code", '''
#@title 10. Calibrate residual uncertainty and evaluate validation data
EVAL_ROOT = RUN_ROOT / ("evaluation-" + uuid.uuid4().hex[:6])
EVAL_ROOT.mkdir()
CAL = EVAL_ROOT/"calibration"
VAL = EVAL_ROOT/"validation"
eval_limit = 16 if PROFILE == "smoke" else (128 if PROFILE == "pilot" else 1024)
draws, flow_steps = (2,4) if PROFILE == "smoke" else ((2,8) if PROFILE == "pilot" else (4,16))
shared = ["--checkpoint",CHECKPOINT,"--data",DATA,"--device",DEVICE,
          "--batch-size",min(BATCH_SIZE,16),"--max-samples",eval_limit,"--seed",SEED]
run_command([sys.executable,"-m","motion_scene_pretrain.evaluate",*shared,
             "--split","calibration","--calibrate","--samples",0,"--output",CAL])
save_to_drive(CAL)
run_command([sys.executable,"-m","motion_scene_pretrain.evaluate",*shared,
             "--split","val","--calibration",CAL/"calibration.json",
             "--samples",draws,"--flow-steps",flow_steps,"--output",VAL])
save_to_drive(VAL)
metrics = json.loads((VAL/"metrics.json").read_text())
print("Validation MSE:",metrics["uncalibrated"]["mse_normalized"])
print("95% marginal coverage, before/after:",metrics["uncalibrated"]["marginal_coverage_95"],metrics["calibrated"]["marginal_coverage_95"])
print("Calibration applies to forecast residuals; it is not collision-risk calibration.")
''', "evaluate")

cell("code", '''
#@title 11. Create, display, and download the return report
from IPython.display import HTML, display
import tarfile
REPORT = EVAL_ROOT/"report"
run_command([sys.executable,"-m","motion_scene_pretrain.report","--run",FINAL_RUN,
             "--evaluations",CAL,VAL,"--output",REPORT])
pipeline_receipts = [json.loads((s["output"]/"receipt.json").read_text()) for s in stages]
pipeline_summary = {"pipeline":PIPELINE,"profile":PROFILE,"stages":pipeline_receipts,
                    "total_optimizer_steps":sum(r["config"]["steps"] for r in pipeline_receipts),
                    "total_training_elapsed_seconds":sum(r["elapsed_seconds"] for r in pipeline_receipts),
                    "note":"Warm-start stages use new optimizers; include all stages when comparing costs."}
(REPORT/"pipeline.json").write_text(json.dumps(pipeline_summary,indent=2))
RETURN_BUNDLE = REPORT/"colab-return-report.tar.gz"
with tarfile.open(RETURN_BUNDLE,"w:gz") as bundle:
    for name in ("return-report.tar.gz","pipeline.json","report.html","summary.json"):
        bundle.add(REPORT/name,arcname=name)
    bundle.add(RUN_ROOT/"notebook-settings.json",arcname="notebook-settings.json")
    for index,stage in enumerate(stages,1):
        for name in ("receipt.json","config.json","normalizer.json","metrics.jsonl"):
            bundle.add(stage["output"]/name,arcname=f"stage-{index}/{name}")
save_to_drive(REPORT)
if DRIVE_RUN is not None:
    shutil.copyfile(RUN_ROOT/"notebook-settings.json",DRIVE_RUN/"notebook-settings.json")
display(HTML((REPORT/"report.html").read_text()))
files.download(str(RETURN_BUNDLE))
print("Send colab-return-report.tar.gz back to the research chat; it includes every training stage's cost.")
print("Checkpoint saved separately:",CHECKPOINT)
if DRIVE_RUN is not None:
    print("Persistent outputs:",DRIVE_RUN)
''', "report")

cell("code", '''
#@title 12. Optional checkpoint download
DOWNLOAD_CHECKPOINT = False #@param {type:"boolean"}
if DOWNLOAD_CHECKPOINT:
    files.download(str(CHECKPOINT))
else:
    print("Checkpoint download skipped. Completed stages are already saved to Drive when enabled.")
''', "checkpoint-download")

cell("markdown", """
### Next experiments

Compare `forecast` with `masked_then_forecast` using the same total training budget, then compare `flow` with `masked_then_flow`. Repeat with independent seeds. Use the repository's `forecast_blind.json` for a no-scene baseline. Keep the dataset manifest fixed.

This notebook deliberately evaluates **validation** data; it leaves the package test split untouched while you choose settings. The runbook has the separate test command for a frozen experiment. The current six-group holdouts support exploratory results. Generated sample diversity, predicted residual variance, and model disagreement are different quantities.

Colab's resources and runtime lifetime vary. The notebook copies the archive onto local VM storage before extraction and saves completed stages back to Drive, following [Colab's guidance](https://research.google.com/colaboratory/faq.html). No training result or GPU runtime is promised by the notebook.
""", "next")

notebook = {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"name": "MotionScene_Pretrain_Colab.ipynb", "provenance": [], "toc_visible": True},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks/MotionScene_Pretrain_Colab.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
