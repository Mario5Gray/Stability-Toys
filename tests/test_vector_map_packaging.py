"""S2.1 packaging contract for the vector extra (STABL-fbvazitj).

Q1 (STABL-orcwoxml): pin vtracer==0.6.15.
Q3 (STABL-stntbgim): torch moves from the global list into the depth and pose extras.
all stays depth, pose and canny. vector is a separate extra.
S2.2 (STABL-ascsgqha) adds the st-vector-map entry point with its modules in one commit.
S2.7 (STABL-kfrksmnp) pins the Q2 preview renderer resvg-py==0.5.0 in the vector extra.
"""

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROJECT = tomllib.loads((SCRIPTS / "pyproject.toml").read_text())["project"]
SETUPTOOLS = tomllib.loads((SCRIPTS / "pyproject.toml").read_text())["tool"]["setuptools"]
EXTRAS = PROJECT["optional-dependencies"]

# Modules that a torch-free vector or canny install must never import.
HEAVY = ("torch", "transformers", "diffusers", "huggingface_hub", "controlnet_aux", "mediapipe", "server", "backends")


def _names(requirements):
    return {req.split(">")[0].split("=")[0].split("<")[0].strip().lower() for req in requirements}


def test_global_dependencies_exclude_torch():
    assert "torch" not in _names(PROJECT["dependencies"])
    assert {"pillow", "numpy"} <= _names(PROJECT["dependencies"])


@pytest.mark.parametrize("extra", ["depth", "pose"])
def test_depth_and_pose_extras_require_torch(extra):
    assert "torch>=2.1" in EXTRAS[extra]


def test_vector_extra_pins_vtracer_opencv_and_renderer():
    assert EXTRAS["vector"] == ["vtracer==0.6.15", "opencv-python-headless>=4.5", "resvg-py==0.5.0"]


def test_preview_module_is_packaged():
    assert "vector_map_preview" in SETUPTOOLS["py-modules"]


def test_test_container_installs_exact_vector_pins():
    """Native container render checks must run, not skip on a missing renderer."""
    lines = {line.strip() for line in (ROOT / "requirements-test.txt").read_text().splitlines()}
    assert {"vtracer==0.6.15", "resvg-py==0.5.0"} <= lines


def test_all_extra_excludes_vector():
    assert EXTRAS["all"] == ["st-controlnet-helpers[depth,pose,canny]"]


def test_vector_adapter_is_packaged():
    assert "vector_map_vtracer" in SETUPTOOLS["py-modules"]


VECTOR_CLI_MODULES = {"vector_map", "vector_map_config", "vector_map_raster", "vector_map_svg"}


def test_vector_cli_entry_point_ships_with_its_modules():
    assert PROJECT["scripts"]["st-vector-map"] == "vector_map:main"
    assert VECTOR_CLI_MODULES <= set(SETUPTOOLS["py-modules"])


def test_every_packaged_module_and_entry_point_has_a_file():
    """setuptools skips a missing py-module silently. Its console script then fails."""
    for module in SETUPTOOLS["py-modules"]:
        assert (SCRIPTS / f"{module}.py").is_file(), module
    for script, target in PROJECT["scripts"].items():
        module = target.split(":")[0]
        assert module in SETUPTOOLS["py-modules"], script


# --- Import isolation in a fresh interpreter --------------------------------

_BLOCKER = textwrap.dedent(
    """
    import sys

    class Block:
        def __init__(self, names):
            self.names = names

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.names:
                raise ModuleNotFoundError(f"blocked import: {name}", name=name)
            return None

    sys.meta_path.insert(0, Block(set(sys.argv[1].split(","))))
    sys.path.insert(0, sys.argv[2])
    """
)


def _run_blocked(blocked, body, *args):
    code = _BLOCKER + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code, ",".join(blocked), str(SCRIPTS), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_vector_conversion_runs_without_torch_server_or_model_modules():
    pytest.importorskip("vtracer")
    result = _run_blocked(
        HEAVY,
        """
        import numpy as np
        import vector_map_vtracer

        layer = np.zeros((32, 32), bool)
        layer[8:24, 8:24] = True
        svg = vector_map_vtracer.trace_layer(layer).svg
        assert "<path" in svg
        loaded = sorted({m.split(".")[0] for m in sys.modules} & set(sys.argv[1].split(",")))
        print("HEAVY_LOADED", loaded)
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "HEAVY_LOADED []" in result.stdout


def test_vector_cli_runs_without_torch_server_or_model_modules(tmp_path):
    pytest.importorskip("vtracer")
    out = tmp_path / "out.svg"
    result = _run_blocked(
        HEAVY,
        """
        import vector_map

        code = vector_map.main([sys.argv[3], sys.argv[4], "--input-kind", "mask", "--width-mm", "10"])
        loaded = sorted({m.split(".")[0] for m in sys.modules} & set(sys.argv[1].split(",")))
        print("EXIT", code, "HEAVY_LOADED", loaded)
        """,
        ROOT / "tests" / "fixtures" / "vector_map" / "donut.png",
        out,
    )
    assert result.returncode == 0, result.stderr
    assert "EXIT 0 HEAVY_LOADED []" in result.stdout
    assert out.is_file()


def test_missing_vtracer_gives_install_hint():
    result = _run_blocked(
        ["vtracer"],
        """
        import numpy as np
        import vector_map_vtracer

        vector_map_vtracer.trace_layer(np.ones((4, 4), bool))
        """,
    )
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert "vtracer==0.6.15" in result.stderr
    assert "vector extra" in result.stderr


def test_canny_sibling_runs_without_torch(tmp_path):
    pytest.importorskip("cv2")
    from PIL import Image, ImageDraw

    source = tmp_path / "in.png"
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle([16, 16, 47, 47], fill="white")
    image.save(source)
    result = _run_blocked(
        HEAVY,
        """
        import canny_map

        sys.argv = ["st-canny-map", sys.argv[3], sys.argv[4]]
        canny_map.main()
        """,
        source,
        tmp_path / "out.png",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out.png").is_file()


@pytest.mark.parametrize("module, extra", [("depth_map", "depth"), ("pose_map", "pose")])
def test_torch_sibling_without_torch_gives_install_hint(module, extra):
    """torch is no longer global. A canny- or vector-only install must fail clearly."""
    result = _run_blocked(
        ["torch"],
        f"""
        sys.argv = ["st-{module}", "--help"]
        import {module}
        """,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "torch" in result.stderr
    assert f"./scripts[{extra}]" in result.stderr


@pytest.mark.parametrize("module", ["depth_map", "pose_map"])
def test_broken_torch_dependency_is_not_reported_as_missing_torch(module, tmp_path):
    """A torch that fails on its own missing dependency must surface that error unchanged."""
    (tmp_path / "torch.py").write_text("import torch_missing_dependency\n")
    code = f"import sys; sys.path[:0] = [{str(tmp_path)!r}, {str(SCRIPTS)!r}]; import {module}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    assert "ModuleNotFoundError: No module named 'torch_missing_dependency'" in result.stderr
    assert "Install the" not in result.stderr


def test_conversion_without_preview_never_imports_renderer(tmp_path):
    pytest.importorskip("vtracer")
    out = tmp_path / "out.svg"
    result = _run_blocked(
        ["resvg_py"],
        """
        import vector_map

        code = vector_map.main([sys.argv[3], sys.argv[4], "--input-kind", "mask", "--width-mm", "10"])
        print("EXIT", code, "RENDERER", "resvg_py" in sys.modules)
        """,
        ROOT / "tests" / "fixtures" / "vector_map" / "donut.png",
        out,
    )
    assert result.returncode == 0, result.stderr
    assert "EXIT 0 RENDERER False" in result.stdout


def test_missing_renderer_fails_preview_before_processing(tmp_path):
    pytest.importorskip("vtracer")
    out = tmp_path / "out.svg"
    result = _run_blocked(
        ["resvg_py"],
        """
        import vector_map

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared before renderer check")

        vector_map.raster.prepare = forbidden
        code = vector_map.main([sys.argv[3], sys.argv[4], "--input-kind", "mask", "--width-mm", "10", "--preview"])
        print("EXIT", code)
        """,
        ROOT / "tests" / "fixtures" / "vector_map" / "donut.png",
        out,
    )
    assert "EXIT 1" in result.stdout, result.stderr
    assert "resvg-py==0.5.0" in result.stderr
    assert "vector extra" in result.stderr
    assert "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []
