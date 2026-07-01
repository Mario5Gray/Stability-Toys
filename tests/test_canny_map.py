from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PYPROJECT = ROOT / "scripts" / "pyproject.toml"


def test_pyproject_exposes_canny_install_surface():
    data = tomllib.loads(SCRIPTS_PYPROJECT.read_text())
    project = data["project"]

    assert project["optional-dependencies"]["canny"] == [
        "opencv-python-headless>=4.5",
    ]
    assert project["optional-dependencies"]["all"] == [
        "st-controlnet-helpers[depth,pose,canny]",
    ]
    assert project["scripts"]["st-canny-map"] == "canny_map:main"
    assert "canny_map" in data["tool"]["setuptools"]["py-modules"]


import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw


SCRIPT = ROOT / "scripts" / "canny_map.py"


def _run_script(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _write_fixture(path: Path, size: tuple[int, int] = (96, 64)) -> None:
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)
    draw.rectangle((12, 12, size[0] - 12, size[1] - 12), outline="white", width=4)
    draw.line((0, size[1] // 2, size[0], size[1] // 2), fill="white", width=2)
    img.save(path)


def test_missing_source_path_reports_error(tmp_path: Path):
    missing = tmp_path / "missing.png"
    dest = tmp_path / "edges.png"

    result = _run_script(missing, dest)

    assert result.returncode != 0
    assert f"error: source not found: {missing}" in result.stderr


def test_generates_l_mode_edge_map(tmp_path: Path):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source)

    result = _run_script(source, dest)

    assert result.returncode == 0, result.stderr
    out = Image.open(dest)
    assert out.mode == "L"
    arr = np.array(out)
    assert arr.max() == 255
    assert arr.min() == 0


def test_invert_flips_edge_polarity(tmp_path: Path):
    source = tmp_path / "source.png"
    base_dest = tmp_path / "edges.png"
    inv_dest = tmp_path / "edges_invert.png"
    _write_fixture(source)

    base = _run_script(source, base_dest)
    inverted = _run_script(source, inv_dest, "--invert")

    assert base.returncode == 0, base.stderr
    assert inverted.returncode == 0, inverted.stderr
    base_arr = np.array(Image.open(base_dest))
    inv_arr = np.array(Image.open(inv_dest))
    assert np.array_equal(inv_arr, 255 - base_arr)


def test_max_res_caps_longest_edge(tmp_path: Path):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source, size=(400, 200))

    result = _run_script(source, dest, "--max-res", "100")

    assert result.returncode == 0, result.stderr
    out = Image.open(dest)
    assert out.size == (100, 50)


def test_valid_blur_value_exercises_blur_path(tmp_path: Path):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source)

    result = _run_script(source, dest, "--blur", "5")

    assert result.returncode == 0, result.stderr
    assert dest.exists()


@pytest.mark.parametrize("blur", ["2", "-1"])
def test_invalid_blur_values_fail_fast(tmp_path: Path, blur: str):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source)

    result = _run_script(source, dest, "--blur", blur)

    assert result.returncode != 0
    assert "--blur must be 0 or a positive odd integer" in result.stderr
