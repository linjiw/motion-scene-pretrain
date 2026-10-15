"""Generate an A100 experiment notebook using the same verified archive setup."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / "notebooks/MotionScene_Pretrain_Colab.ipynb").read_text())
base_cells = {c["id"]: c for c in base["cells"]}
cells = []


def cell(kind, source, name):
    item = {"cell_type": kind, "id": name, "metadata": {"id": name},
            "source": (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
        item["metadata"]["cellView"] = "form"
    cells.append(item)


cell("markdown", '''
# Motion–Scene · A100 controlled experiments

The first pilot trained quickly, but its original validation split contained **no obstacle scenes**. This notebook derives a **new, fixed scene holdout** from your existing Drive archive, trains fresh models, and compares them fairly. Your archive needs no re-upload.

**Start:** select an A100/GPU runtime, leave `SUITE = screen`, and run cells in order. Five arms × 2,000 total optimizer steps; batch 64. Pretraining steps count toward the same budget. All 585 validation and 585 calibration windows are evaluated; test remains unused.

| Arm | What it tests |
|---|---|
| blind | History + goal, without scene information |
| scene | Same model with scene information |
| masked_scene | 500 masked-history steps + 1,500 forecast steps |
| original_sampler | Scene model with the old sampler instead of 25% scene draws |
| medium | Larger 9.79M-parameter model, compared with 1.89M |

A new checkout keeps this notebook separate from the pilot checkout. **Never initialize from the old pilot checkpoint**: it has seen the groups now held out. Existing pilot results stay in Drive. The data contain compatible scenes and identical empty-scene twins, so neither attention response nor offline error establishes collision avoidance.

[Protocol and pilot review](https://github.com/linjiw/motion-scene-pretrain/blob/main/docs/A100_EXPERIMENTS.md)
''', "intro")

cell("code", '''
#@title 1. A100 experiment settings
SUITE = "screen" #@param ["screen", "confirm", "generative", "smoke"]
BATCH_SIZE = 64 #@param {type:"integer"}
SEEDS = "" #@param {type:"string"}
# Empty uses seed 0 for screen/generative; 0,1,2 for confirm.
ARMS = "" #@param {type:"string"}
# Empty runs the suite defaults. Optional comma-separated subset, e.g. scene,masked_scene.
DEVICE = "cuda" #@param ["cuda", "cpu"]
DATA_SOURCE = "drive"
DRIVE_ARCHIVE_PATH = "/content/drive/MyDrive/motion2scene/kimodo-passage-h25-f100-v1.tar.xz" #@param {type:"string"}
SAVE_TO_DRIVE = True #@param {type:"boolean"}
DRIVE_RESULTS_FOLDER = "/content/drive/MyDrive/motion-scene-pretrain/a100-experiments" #@param {type:"string"}
if DEVICE == "cpu" and SUITE != "smoke":
    raise ValueError("CPU is only for the bounded smoke suite.")
if BATCH_SIZE < 1:
    raise ValueError("BATCH_SIZE must be positive.")
''', "settings")

setup = "".join(base_cells["setup"]["source"])
setup = setup.replace('"MOTION_SCENE_PRETRAIN_CHECKOUT"', '"MOTION_SCENE_A100_CHECKOUT"')
setup = setup.replace('"motion-scene-code-"', '"motion-scene-a100-code-"')
cell("code", setup, "setup")
cell("code", "".join(base_cells["load-data"]["source"]), "load-data")
cell("code", '''
#@title 4. Verify the original archive and derive the fixed scene holdout
from motion_scene_pretrain.colab_utils import prepare_dataset
from motion_scene_pretrain.data import verify_dataset

V1_DATA = prepare_dataset(ARCHIVE, WORK_ROOT / "motion-scene-data", transfer)
protocol = json.loads((REPO / "manifests/scene_holdout_protocol.json").read_text())
if sha256(V1_DATA / "manifest.json") != protocol["parent_manifest_sha256"]:
    raise ValueError("Unexpected parent dataset; this comparison needs the frozen original archive.")
DATA = WORK_ROOT / "motion-scene-data" / protocol["name"]
if not DATA.exists():
    subprocess.run([sys.executable, "-m", "motion_scene_pretrain.repartition",
                    "--source", str(V1_DATA), "--output", str(DATA),
                    "--scene-groups", str(protocol["scene_groups"]), "--seed", str(protocol["seed"])], check=True)
if sha256(DATA / "manifest.json") != protocol["dataset_manifest_sha256"]:
    raise ValueError("Existing derivative differs from the frozen protocol; use a fresh derivative directory.")
verify_dataset(DATA)
print("Verified fresh-training protocol:", DATA)
print("All split rows:", protocol["counts"])
print("Obstacle rows:", protocol["scene_rows"])
print("Validation: 8 ancestry groups, including 2 with obstacles. The original test and calibration groups remain fixed.")
''', "verify-data")

cell("code", '''
#@title 5. Freeze and inspect the experiment plan
from motion_scene_pretrain.experiments import make_plan
SEED_LIST = [int(x.strip()) for x in SEEDS.split(",") if x.strip()] or None
ARM_LIST = [x.strip() for x in ARMS.split(",") if x.strip()] or None
plan = make_plan(SUITE, SEED_LIST, BATCH_SIZE, DEVICE, ARM_LIST)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
RUN_ROOT = WORK_ROOT / "motion-scene-a100-runs" / RUN_ID
DRIVE_RUN = Path(DRIVE_RESULTS_FOLDER) / RUN_ID if SAVE_TO_DRIVE else None
for run in plan["runs"]:
    stages = run["stages"]
    print(run["name"], " | ".join(f"{s['config']['mode']}: {s['config']['steps']} steps" for s in stages),
          "| presentations:", sum(s["config"]["steps"] * s["config"]["batch_size"] for s in stages))
print("Total optimizer steps:", sum(s["config"]["steps"] for r in plan["runs"] for s in r["stages"]))
print("Local output:", RUN_ROOT)
print("Drive output:", DRIVE_RUN)
print("Wall time and memory will be measured; capacity changes are not equal-FLOP comparisons.")
''', "plan")

cell("code", '''
#@title 6. Run the suite (completed stages and checkpoints save to Drive)
command = [sys.executable, "-m", "motion_scene_pretrain.experiments", "--data", str(DATA),
           "--output", str(RUN_ROOT), "--suite", SUITE, "--device", DEVICE, "--batch-size", str(BATCH_SIZE)]
if SEED_LIST:
    command += ["--seeds", *map(str, SEED_LIST)]
if ARM_LIST:
    command += ["--arms", *ARM_LIST]
if DRIVE_RUN is not None:
    command += ["--backup", str(DRIVE_RUN)]
subprocess.run(command, cwd=REPO, check=True)
print("Completed:", RUN_ROOT)
''', "train")

cell("code", '''
#@title 7. Compare full-validation errors, coverage, and measured cost
import matplotlib.pyplot as plt
from IPython.display import HTML, display

results = json.loads((RUN_ROOT / "comparison.json").read_text())
display(HTML((RUN_ROOT / "comparison.html").read_text()))
labels = [f"{r['arm']} / s{r['seed']}" for r in results]
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
x = np.arange(len(results))
if all(r.get("generation") for r in results):
    axes[0].bar(x-.18, [r["generation"]["sample_mean_mse_normalized"] for r in results], width=.36, label="Average generated sample")
    axes[0].bar(x+.18, [r["generation"]["oracle_best_of_k_mse_normalized"] for r in results], width=.36, label="Oracle best-of-4")
    axes[0].set_title("Generated motion error · lower is better")
    print("Generated sample variance:", {labels[i]:r["generation"]["sample_variance_normalized"] for i,r in enumerate(results)})
    print("Coverage below belongs to the separate auxiliary forecast head. Oracle best-of-K is not a deployed selector.")
else:
    axes[0].bar(x-.18, [r["equal_group_mse"] for r in results], width=.36, label="Equal-group MSE")
    axes[0].bar(x+.18, [r["scene_mse"] if r["scene_mse"] is not None else np.nan for r in results], width=.36, label="Obstacle examples MSE")
    axes[0].set_title("Forecast error · lower is better")
axes[0].set_ylabel("Normalized MSE")
axes[0].legend()
axes[1].bar(x-.18, [r["coverage95_raw"]*100 for r in results], width=.36, label="Raw")
axes[1].bar(x+.18, [r["coverage95_calibrated"]*100 for r in results], width=.36, label="Calibrated")
axes[1].axhline(95, color="black", linestyle="--", label="Nominal 95%")
axes[1].set(title="Marginal coverage · aim near 95%", ylabel="Coverage (%)", ylim=(0,100))
axes[1].legend()
axes[2].bar(x, [r["total_training_seconds"]/60 for r in results])
axes[2].set(title="Measured training cost · all stages", ylabel="Minutes")
for axis in axes:
    axis.set_xticks(x, labels, rotation=40, ha="right")
fig.tight_layout()
fig.savefig(RUN_ROOT / "comparison.png", dpi=160)
plt.show()
if DRIVE_RUN is not None:
    shutil.copyfile(RUN_ROOT / "comparison.png", DRIVE_RUN / "comparison.png")
print("Compare the two obstacle ancestry groups individually, and check the last-observed-state baseline in comparison.json.")
print("Do not select a model using calibration error or test results. A one-seed screen is exploratory.")
''', "compare")

cell("code", '''
#@title 8. Download the compact report to send back
RETURN_REPORT = RUN_ROOT / "a100-return-report.tar.gz"
files.download(str(RETURN_REPORT))
print("Send back:", RETURN_REPORT.name)
print("Contains every stage's metrics/config/receipt and evaluation summaries with their scope; no raw motion or checkpoints.")
if DRIVE_RUN is not None:
    print("Report and completed checkpoints are also saved in:", DRIVE_RUN)
''', "report")

cell("markdown", '''
## After the screen

Review the screen before increasing the budget. `confirm` runs the three core arms at 6,000 total steps each and seeds 0,1,2; `generative` compares direct flow with masked→flow at 2,000 total steps each. Keep batch size and the frozen dataset fixed within a comparison. Use `ARMS` to choose an explicit subset after reviewing the screen.

Every execution of cell 5 creates a fresh suite folder. Re-executing cell 6 can reuse verified completed stages in that folder; an interrupted stage requires a fresh suite folder. Checkpoints save at stage completion, not every optimizer step. This is not exact optimizer resume. An out-of-memory failure calls for a smaller batch size and a new matched comparison across all selected arms.

The next data priority is **more independent motion/scene families and valid contrasting futures**, including approach, adaptation, recovery, and clear-space controls. Current scene/empty twins share the same future. More overlapping windows or a bigger network cannot create missing supervision for obstacle-dependent control.
''', "next")

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "MotionScene_A100_Experiments.ipynb", "provenance": [], "toc_visible": True}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "notebooks/MotionScene_A100_Experiments.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
