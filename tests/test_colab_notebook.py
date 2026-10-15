"""Structural checks for the real notebook, including executable Python cells."""
import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/MotionScene_Pretrain_Colab.ipynb"


def test_notebook_is_clean_and_all_code_compiles():
    notebook = json.loads(NOTEBOOK.read_text())
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
