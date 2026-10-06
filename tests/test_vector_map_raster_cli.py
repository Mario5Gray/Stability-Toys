"""S2.3 CLI preparation integration. STABL-vjpnctjh."""

import json
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def save(path, pixels):
    Image.fromarray(np.asarray(pixels, np.uint8)).save(path)
    return path


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "vector_map.py"), *map(str, args)],
                          capture_output=True, text=True)


def line(tmp_path):
    pixels = np.zeros((20, 20), np.uint8)
    pixels[4:16, 10] = 255
    return save(tmp_path / "line.png", pixels)


def payload(result, expected=0):
    assert result.returncode == expected, result.stderr
    value = json.loads(result.stdout)
    assert set(value) == {"schema_version", "status", "artifacts", "counts", "diagnostics"}
    return value


def test_real_edges_keep_both_warnings_in_one_json_result(tmp_path):
    dest = tmp_path / "out.svg"
    result = run(line(tmp_path), dest, "--input-kind", "edges", "--width-mm", 20,
                 "--line-width-mm", 3, "--json")
    value = payload(result)
    codes = {item["code"] for item in value["diagnostics"]}
    assert {"requested_width_below_four_pixels", "thin_material"} <= codes
    assert "fewer than four" in result.stderr
    assert "4x4 square" in result.stderr
    assert dest.is_file()


def test_report_requested_and_achieved_width_mm(tmp_path):
    result = run(line(tmp_path), tmp_path / "out.svg", "--input-kind", "edges", "--width-mm", 20,
                 "--line-width-mm", 3.1, "--json")
    payload(result)
    assert "requested 3.1 mm" in result.stderr
    assert "achieved 5 mm" in result.stderr


def test_awkward_resize_produces_matching_svg_aspect_and_origin(tmp_path):
    pixels = np.zeros((70, 100), np.uint8)
    pixels[10:60, 20:80] = 255
    source = save(tmp_path / "source.png", pixels)
    dest = tmp_path / "out.svg"
    recipe = tmp_path / "scale.json"
    recipe.write_text(json.dumps({"schema_version": 1, "vtracer": {"filter_speckle": 0}}))
    result = run(source, dest, "--input-kind", "mask", "--width-mm", 14, "--max-res", 7,
                 "--recipe", recipe, "--json")
    payload(result)
    root = ET.parse(dest).getroot()
    assert root.get("width") == "14mm"
    assert root.get("height") == "10mm"
    assert root.get("viewBox") == "0 0 7 5"
    assert "original 100x70" in result.stderr
    assert "processed 7x5" in result.stderr


def test_recipe_relative_constraints_and_cli_override(tmp_path):
    source = line(tmp_path)
    include = np.zeros((20, 20), np.uint8)
    include[3:17, 7:14] = 255
    save(tmp_path / "include.png", include)
    save(tmp_path / "exclude.png", np.zeros((20, 20)))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "input": source.name, "input_kind": "edges",
                                 "width_mm": 20, "line_width_mm": 3,
                                 "include_mask": "missing.png", "exclude_mask": "exclude.png"}))
    result = run(tmp_path / "out.svg", "--recipe", recipe, "--include-mask", tmp_path / "include.png", "--json")
    assert payload(result)["status"] == "converted"


@pytest.mark.parametrize("flag", ["--include-mask", "--exclude-mask"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_constraint_destination_alias_refused_before_decode(tmp_path, flag, alias):
    source = line(tmp_path)
    mask = tmp_path / "mask.png"
    original = b"Do not overwrite this input."
    mask.write_bytes(original)
    dest = mask if alias == "direct" else tmp_path / "out.svg"
    if alias == "symlink":
        dest.symlink_to(mask)
    elif alias == "hardlink":
        dest.hardlink_to(mask)
    result = run(source, dest, "--input-kind", "mask", "--width-mm", 20, flag, mask, "--overwrite", "--json")
    value = payload(result, 2)
    assert "same file" in value["diagnostics"][0]["message"]
    assert mask.read_bytes() == original


def test_alpha_warning_survives_success(tmp_path):
    pixels = np.zeros((20, 20, 4), np.uint8)
    pixels[4:16, 4:16, :3] = 255
    source = save(tmp_path / "source.png", pixels)
    result = run(source, tmp_path / "out.svg", "--input-kind", "mask", "--width-mm", 20, "--json")
    assert "alpha_ignored" in {item["code"] for item in payload(result)["diagnostics"]}
    assert "luminance used" in result.stderr


def test_alpha_warning_survives_empty_trace_failure(tmp_path):
    source = save(tmp_path / "source.png", np.zeros((20, 20, 4)))
    dest = tmp_path / "out.svg"
    result = run(source, dest, "--input-kind", "mask", "--width-mm", 20, "--json")
    value = payload(result, 1)
    assert any(item.get("code") == "alpha_ignored" for item in value["diagnostics"])
    assert value["diagnostics"][0]["level"] == "error"
    assert not dest.exists()


@pytest.mark.parametrize("case, expected", [("missing-alpha", 2), ("mismatch", 2), ("bad-bytes", 1), ("unsupported", 2)])
def test_failures_emit_one_result_and_no_artifact(tmp_path, case, expected):
    source = line(tmp_path)
    extra = []
    if case == "missing-alpha":
        extra = ["--alpha"]
    elif case == "mismatch":
        mask = save(tmp_path / "mask.png", np.ones((10, 10)))
        extra = ["--exclude-mask", mask]
    elif case == "bad-bytes":
        source.write_bytes(b"broken image")
    else:
        Image.new("L", (20, 20)).save(source, format="GIF")
    dest = tmp_path / "out.svg"
    result = run(source, dest, "--input-kind", "mask", "--width-mm", 20, *extra, "--json")
    value = payload(result, expected)
    assert value["status"] == ("invalid" if expected == 2 else "failed")
    assert not dest.exists()


def test_diagnostic_module_ships_in_package():
    metadata = tomllib.loads((SCRIPTS / "pyproject.toml").read_text())
    assert "vector_map_diagnostics" in metadata["tool"]["setuptools"]["py-modules"]


@pytest.mark.parametrize("role", ["source", "include-mask", "exclude-mask"])
def test_16_bit_source_and_constraints_exit_2_without_output(tmp_path, role):
    image_path = tmp_path / "sixteen.png"
    Image.fromarray(np.full((20, 20), 200, dtype=np.uint16)).save(image_path)
    original = image_path.read_bytes()
    source = image_path if role == "source" else line(tmp_path)
    extra = [] if role == "source" else [f"--{role}", image_path]
    dest = tmp_path / "out.svg"
    result = run(source, dest, "--input-kind", "mask", "--width-mm", 20, *extra, "--json")
    value = payload(result, 2)
    assert value["status"] == "invalid"
    assert value["artifacts"]["svg"] is None
    message = value["diagnostics"][0]["message"]
    assert "I;16" in message and "8-bit" in message and "Convert" in message
    assert "Traceback" not in result.stderr
    assert image_path.read_bytes() == original
    assert not dest.exists()
