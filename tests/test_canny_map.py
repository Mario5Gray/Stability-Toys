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
