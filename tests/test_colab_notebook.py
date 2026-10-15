"""Structural checks for the real notebook, including executable Python cells."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
import uuid

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/MotionScene_Pretrain_Colab.ipynb"
A100_NOTEBOOK = NOTEBOOK.with_name("MotionScene_A100_Experiments.ipynb")


@pytest.mark.parametrize("path", [NOTEBOOK, A100_NOTEBOOK])
def test_notebook_is_clean_and_all_code_compiles(path):
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    identifiers = []
    for cell in notebook["cells"]:
        identifiers.append(cell["id"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=cell["id"])
    assert len(set(identifiers)) == len(identifiers)


def _a100_settings(overrides=None):
    cells = {c["id"]: "".join(c["source"]) for c in json.loads(A100_NOTEBOOK.read_text())["cells"]}
    tree = ast.parse(cells["settings"])
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            name = statement.targets[0].id
            if name in (overrides or {}):
                statement.value = ast.Constant(overrides[name])
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), "a100-settings", "exec"), namespace)
    return cells, namespace


def test_a100_settings_default_to_screen_and_existing_drive_archive():
    _, settings = _a100_settings()
    assert settings["SUITE"] == "screen"
    assert settings["BATCH_SIZE"] == 64
    assert settings["DATA_SOURCE"] == "drive"
    assert settings["DRIVE_ARCHIVE_PATH"] == "/content/drive/MyDrive/motion2scene/kimodo-passage-h25-f100-v1.tar.xz"
    assert settings["SEEDS"] == "" and settings["ARMS"] == ""
    with pytest.raises(ValueError, match="CPU.*smoke"):
        _a100_settings({"DEVICE": "cpu"})


@pytest.mark.parametrize("suite,device,arms,steps,batch", [
    ("screen", "cuda", 5, 2000, 64),
    ("smoke", "cpu", 3, 8, 4),
])
def test_a100_plan_cell_executes_real_planner_with_matched_budgets(
        tmp_path, suite, device, arms, steps, batch):
    cells, settings = _a100_settings({"SUITE": suite, "DEVICE": device, "SAVE_TO_DRIVE": False})
    settings.update(WORK_ROOT=tmp_path, Path=Path, datetime=datetime, timezone=timezone, uuid=uuid)
    # The pilot's globals may remain in an existing runtime; the new plan must
    # not incorporate its weights or normalizer.
    settings["CHECKPOINT"] = tmp_path / "old-pilot/checkpoint.pt"
    exec(compile(cells["plan"], "a100-plan", "exec"), settings)
    plan = settings["plan"]
    assert len(plan["runs"]) == arms
    assert plan["test_split_used"] is False
    assert plan["evaluation_splits"] == ["calibration", "val"]
    assert settings["DRIVE_RUN"] is None
    assert not settings["RUN_ROOT"].exists()  # Planning does not launch training.
    for run in plan["runs"]:
        configs = [stage["config"] for stage in run["stages"]]
        assert sum(config["steps"] for config in configs) == steps
        assert all(config["batch_size"] == batch for config in configs)
        assert all(config["device"] == device for config in configs)
        assert all("init" not in config and "checkpoint" not in config for config in configs)
    calls = []
    settings.update(subprocess=SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
                    sys=sys, DATA=tmp_path / "v2-sceneholdout", REPO=tmp_path / "new-repo")
    exec(compile(cells["train"], "a100-train", "exec"), settings)
    command = calls[0][0][0]
    assert "--init" not in command
    assert str(settings["CHECKPOINT"]) not in command
    assert command[command.index("--data") + 1] == str(settings["DATA"])
    assert calls[0][1]["check"] is True


def test_default_is_bounded_pilot_and_evaluation_does_not_open_test():
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}
    settings = {}
    exec(compile(cells["settings"], "settings", "exec"), settings)
    assert settings["PROFILE"] == "pilot"
    assert settings["BATCH_SIZE"] == 16
    assert settings["SAVE_TO_DRIVE"] is True
    tree = ast.parse(cells["evaluate"])
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "calibration" in strings and "val" in strings
    assert "test" not in strings


def test_drive_saves_keep_unique_evaluation_directories(tmp_path):
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}
    tree = ast.parse(cells["smoke"])
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "save_to_drive")
    import shutil
    namespace = {"Path": Path, "shutil": shutil, "RUN_ROOT": tmp_path/"local", "DRIVE_RUN": tmp_path/"drive"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "save", "exec"), namespace)
    for identifier in ("evaluation-a", "evaluation-b"):
        folder = namespace["RUN_ROOT"]/identifier/"validation"
        folder.mkdir(parents=True)
        (folder/"metrics.json").write_text(identifier)
        namespace["save_to_drive"](folder)
    assert (namespace["DRIVE_RUN"]/"evaluation-a/validation/metrics.json").read_text() == "evaluation-a"
    assert (namespace["DRIVE_RUN"]/"evaluation-b/validation/metrics.json").read_text() == "evaluation-b"


def _load_data_cell_fixture(tmp_path, monkeypatch, uploaded_files=None):
    """Run the checked-in cell with browser/Drive operations replaced locally."""
    code = next("".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]
                if c["id"] == "load-data")
    work = tmp_path / "runtime"
    repo = work / "repo"
    (repo / "manifests").mkdir(parents=True)
    archive_name = "fixture.tar.xz"
    payload = b"dataset archive fixture; only transfer is exercised here"
    (repo / "manifests/transfer_package.json").write_text(json.dumps({
        "archive": archive_name, "archive_sha256": hashlib.sha256(payload).hexdigest(),
    }))
    uploads = {archive_name: payload} if uploaded_files is None else uploaded_files
    calls = {"mounts": [], "upload_directories": []}

    def upload():
        calls["upload_directories"].append(Path.cwd())
        for name, data in uploads.items():
            Path(name).write_bytes(data)
        return uploads.copy()

    google = ModuleType("google")
    google.__path__ = []
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=calls["mounts"].append)
    colab.files = SimpleNamespace(upload=upload)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.chdir(repo)
    namespace = {
        "WORK_ROOT": work, "REPO": repo, "DATA_SOURCE": "drive", "SAVE_TO_DRIVE": True,
        "DRIVE_ARCHIVE_PATH": str(work / "drive" / archive_name),
        "Path": Path, "json": json, "os": os, "shutil": shutil,
    }
    return compile(code, "load-data", "exec"), namespace, calls, payload


def test_missing_drive_archive_opens_upload_picker(tmp_path, monkeypatch, capsys):
    code, namespace, calls, payload = _load_data_cell_fixture(tmp_path, monkeypatch)
    previous = Path.cwd()
    exec(code, namespace)
    assert calls["mounts"] == ["/content/drive"]
    assert len(calls["upload_directories"]) == 1
    assert namespace["ARCHIVE"].read_bytes() == payload
    assert Path.cwd() == previous
    assert not calls["upload_directories"][0].exists()
    printed = capsys.readouterr().out
    assert "not found at" in printed
    assert "Opening the upload picker instead" in printed
    assert "cell 4 verifies the checksum" in printed


def test_found_drive_archive_does_not_prompt_for_upload(tmp_path, monkeypatch):
    code, namespace, calls, payload = _load_data_cell_fixture(tmp_path, monkeypatch)
    source = Path(namespace["DRIVE_ARCHIVE_PATH"])
    source.parent.mkdir()
    source.write_bytes(payload)
    exec(code, namespace)
    assert not calls["upload_directories"]
    assert namespace["ARCHIVE"].read_bytes() == payload
    assert source.read_bytes() == payload


def test_verified_cached_archive_does_not_prompt_for_upload(tmp_path, monkeypatch, capsys):
    code, namespace, calls, payload = _load_data_cell_fixture(tmp_path, monkeypatch)
    cache = namespace["WORK_ROOT"] / "motion-scene-cache"
    cache.mkdir()
    (cache / "fixture.tar.xz").write_bytes(payload)
    assert not Path(namespace["DRIVE_ARCHIVE_PATH"]).exists()
    exec(code, namespace)
    assert not calls["upload_directories"]
    assert namespace["ARCHIVE"].read_bytes() == payload
    assert "Reusing the already verified local archive" in capsys.readouterr().out


@pytest.mark.parametrize("uploaded_files,error,message", [
    ({}, RuntimeError, "No file selected"),
    ({"wrong-name.tar.xz": b"wrong file"}, ValueError, "Choose only the archive named fixture.tar.xz"),
])
def test_upload_recovery_errors_restore_working_directory(
        tmp_path, monkeypatch, uploaded_files, error, message):
    code, namespace, calls, _ = _load_data_cell_fixture(tmp_path, monkeypatch, uploaded_files)
    previous = Path.cwd()
    with pytest.raises(error, match=message) as caught:
        exec(code, namespace)
    assert "rerun cell 3" in str(caught.value)
    assert Path.cwd() == previous
    assert len(calls["upload_directories"]) == 1
    assert not calls["upload_directories"][0].exists()
    assert not namespace["ARCHIVE"].exists()
