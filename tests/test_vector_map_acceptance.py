"""S2.8 standalone acceptance: repeatability and end-to-end mask/edge evidence. STABL-ntgnbxci.

Spec 10 repeatability row and 11.2 exit evidence. Real vtracer==0.6.15, real resvg-py==0.5.0,
and the real st-canny-map command. Mocks are not conversion proof.

Set ST_VECTOR_MAP_BIN to a directory with installed st-vector-map and st-canny-map console
scripts to run the command checks against an installed package. Default: source scripts.
Set VECTOR_MAP_ACCEPTANCE_EVIDENCE to a JSON path to save per-case evidence for the report.
No CAD acceptance gate. OpenSCAD checks live in test_vector_map_openscad.py.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vtracer", reason="vtracer==0.6.15 is not installed. Install the vector extra.")
pytest.importorskip("resvg_py", reason="resvg-py==0.5.0 is not installed. Install the vector extra.")
pytest.importorskip("cv2", reason="opencv is not installed. Install the vector extra.")

from tests.fixtures.vector_map import make_fixtures  # noqa: E402
from tests.fixtures.vector_map.corpus import metrics  # noqa: E402
from tests.test_vector_map_corpus import FAILING_AT_EVERY_VALUE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
FIXTURES = ROOT / "tests" / "fixtures" / "vector_map"
CORPUS = FIXTURES / "corpus"

import vector_map  # noqa: E402
import vector_map_config as config  # noqa: E402
import vector_map_preview as preview  # noqa: E402

TOPOLOGY = sorted(make_fixtures.EXPECTED)
CORPUS_CASES = sorted(path.name for path in CORPUS.iterdir() if (path / "mask.png").is_file())
MASK_CASES = [("topology", name) for name in TOPOLOGY] + [("corpus", name) for name in CORPUS_CASES]
EDGE_SETTINGS = ["--input-kind", "edges", "--width-mm", "100", "--line-width-mm", "0.8"]
MASK_SETTINGS = ["--input-kind", "mask", "--width-mm", "100"]
RUNS = 3
EVIDENCE = []


@pytest.fixture(scope="module", autouse=True)
def _save_evidence():
    yield
    target = os.environ.get("VECTOR_MAP_ACCEPTANCE_EVIDENCE")
    if target:
        Path(target).write_text(json.dumps({
            "versions": _versions(), "python": sys.version.split()[0], "platform": sys.platform,
            "commands": _commands(), "cases": EVIDENCE,
        }, indent=2, sort_keys=True) + "\n")


def _versions():
    names = ("st-controlnet-helpers", "vtracer", "resvg-py", "Pillow", "numpy", "opencv-python-headless", "opencv-python")
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except Exception:
            found[name] = None
    return found


def _commands():
    bin_dir = os.environ.get("ST_VECTOR_MAP_BIN")
    if bin_dir:
        return {"vector": [str(Path(bin_dir) / "st-vector-map")], "canny": [str(Path(bin_dir) / "st-canny-map")]}
    return {"vector": [sys.executable, str(SCRIPTS / "vector_map.py")],
            "canny": [sys.executable, str(SCRIPTS / "canny_map.py")]}


def _record(**values):
    EVIDENCE.append(values)
    return values


# --- Helpers ----------------------------------------------------------------


def _in_process(capsys, monkeypatch, *argv):
    """Run the console-script main(). Spy which publication primitive moves files."""
    calls = {"link": 0, "replace": 0}
    link, replace = os.link, os.replace

    def spy_link(*args, **kwargs):
        calls["link"] += 1
        return link(*args, **kwargs)

    def spy_replace(*args, **kwargs):
        calls["replace"] += 1
        return replace(*args, **kwargs)

    monkeypatch.setattr(os, "link", spy_link)
    monkeypatch.setattr(os, "replace", spy_replace)
    start = time.perf_counter()
    code = vector_map.main([*map(str, argv), "--json"])
    elapsed = time.perf_counter() - start
    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "replace", replace)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    return code, json.loads(lines[0]), captured.err, calls, elapsed


def _bundle_files(destination):
    """Every bundle member on disk, by name relative to the destination directory."""
    parent = destination.parent
    names = [destination, destination.with_suffix(".vector.json"), destination.with_suffix(".preview.png")]
    debug = destination.with_suffix(".debug")
    names += [debug / "mask.png", debug / "recipe.json"]
    return {path.relative_to(parent).as_posix(): path.read_bytes() for path in names if path.exists()}


def _verify_manifest(manifest_path):
    """Independent of the publisher: hash every listed file with hashlib here."""
    manifest = json.loads(manifest_path.read_bytes())
    for member in manifest["artifacts"]:
        data = (manifest_path.parent / member["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == member["sha256"], member["path"]
        assert len(data) == member["bytes"], member["path"]
    return manifest


def _canny(source, destination):
    command = _commands()["canny"]
    start = time.perf_counter()
    result = subprocess.run([*command, str(source), str(destination)], capture_output=True, text=True, cwd=ROOT)
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, result.stderr
    assert destination.is_file()
    return elapsed


def _geometry(manifest, destination):
    """Crisp render of the published SVG against the prepared material from the debug bundle."""
    canvas = config.Canvas(**manifest["canvas"])
    material = np.asarray(Image.open(destination.with_suffix(".debug") / "mask.png").convert("L")) >= 128
    svg = destination.read_text()
    rendered = preview.render_material(svg, canvas)
    return metrics.evaluate(material, rendered, metrics.polygon_area(svg)), material, rendered


def _repeat(capsys, monkeypatch, tmp_path, source, settings):
    """Three runs at one destination. Save bytes after each run, before the next starts."""
    destination = tmp_path / "out" / "result.svg"
    snapshots, runs = [], []
    for index in range(RUNS):
        extra = ["--overwrite"] if index else []
        code, result, error, calls, elapsed = _in_process(
            capsys, monkeypatch, source, destination, *settings, "--preview", "--debug-bundle", *extra,
        )
        assert code == 0, error
        manifest = _verify_manifest(destination.with_suffix(".vector.json"))
        snapshots.append(_bundle_files(destination))
        runs.append({"exit": code, "seconds": round(elapsed, 4), "publication": calls})
        if index == 0:
            assert calls["link"] > 0 and calls["replace"] == 0, calls
        else:
            assert calls["replace"] > 0 and calls["link"] == 0, calls
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert set(snapshots[0]) == {
        "result.svg", "result.vector.json", "result.preview.png", "result.debug/mask.png", "result.debug/recipe.json",
    }
    return destination, manifest, result, runs, snapshots[0]


# --- Steps 1, 4-6, 10: mask mode repeatability --------------------------------


@pytest.mark.parametrize("group, name", MASK_CASES, ids=[f"{g}-{n}" for g, n in MASK_CASES])
def test_mask_conversion_repeats_and_records_evidence(capsys, monkeypatch, tmp_path, group, name):
    source = FIXTURES / f"{name}.png" if group == "topology" else CORPUS / name / "mask.png"
    destination, manifest, result, runs, files = _repeat(capsys, monkeypatch, tmp_path, source, MASK_SETTINGS)
    geometry, material, rendered = _geometry(manifest, destination)
    if group == "topology":
        assert list(metrics.topology(rendered)) == list(make_fixtures.EXPECTED[name])
    else:
        assert geometry["failures"] == FAILING_AT_EVERY_VALUE.get(name, []), geometry
    _record(
        mode="mask", group=group, case=name, runs=runs, identical_runs=RUNS,
        canvas=manifest["canvas"], processed_size=manifest["preparation"]["processed_size"],
        counts=manifest["counts"], warnings=[w["code"] for w in manifest["warnings"]],
        hashes={path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
        geometry={"failures": geometry["failures"], "xor_px": geometry.get("xor_px"),
                  "topology_mask": geometry["topology_mask"]},
        paths=result["counts"]["paths"],
    )


# --- Steps 2-6, 10: actual st-canny-map output in edge mode -------------------


@pytest.mark.parametrize("name", CORPUS_CASES)
def test_real_canny_output_converts_in_edge_mode_and_repeats(capsys, monkeypatch, tmp_path, name):
    edges = tmp_path / "canny" / f"{name}.png"
    canny_seconds = _canny(CORPUS / name / "source.png", edges)
    destination, manifest, result, runs, files = _repeat(capsys, monkeypatch, tmp_path, edges, EDGE_SETTINGS)
    geometry, material, rendered = _geometry(manifest, destination)
    assert manifest["preparation"]["input_kind"] == "edges"
    assert manifest["preparation"]["expansion"]["kernel_size_px"] >= 1
    _record(
        mode="edges", group="canny", case=name, canny_seconds=round(canny_seconds, 4), runs=runs,
        identical_runs=RUNS, canvas=manifest["canvas"], expansion=manifest["preparation"]["expansion"],
        counts=manifest["counts"], warnings=[w["code"] for w in manifest["warnings"]],
        hashes={path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
        geometry={"failures": geometry["failures"], "xor_px": geometry.get("xor_px"),
                  "topology_mask": geometry["topology_mask"]},
        paths=result["counts"]["paths"],
    )


@pytest.mark.parametrize("name", ["donut", "asymmetric"])
def test_real_command_matches_in_process_bytes(capsys, monkeypatch, tmp_path, name):
    """The subprocess command, source or installed, publishes the same bundle bytes."""
    source = FIXTURES / f"{name}.png"
    command = _commands()["vector"]
    by_command = tmp_path / "command" / "result.svg"
    result = subprocess.run(
        [*command, str(source), str(by_command), *MASK_SETTINGS, "--preview", "--debug-bundle", "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "converted"
    _verify_manifest(by_command.with_suffix(".vector.json"))
    in_process = tmp_path / "in_process" / "result.svg"
    code, *_ = _in_process(capsys, monkeypatch, source, in_process, *MASK_SETTINGS, "--preview", "--debug-bundle")
    assert code == 0
    assert _bundle_files(by_command) == _bundle_files(in_process)
    _record(mode="mask", group="command", case=name, command=command[-1], exit=result.returncode)
