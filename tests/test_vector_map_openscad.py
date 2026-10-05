"""S1.3 relief proof: real OpenSCAD import and render of traced silhouettes (STABL-nygbbrrn).

Spec 6.5 and the spec 10 CAD row: no claim from SCAD syntax alone. Each test runs
an actual OpenSCAD render and checks the exported mesh.

OpenSCAD is often not on PATH (macOS app bundle, Linux Flatpak). Set OPENSCAD to
the command, for example OPENSCAD="flatpak run org.openscad.OpenSCAD".
The Flatpak sandbox reads $HOME only. With a Flatpak command, the render files
move to a temporary directory under $HOME automatically.
"""

import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.vector_map import make_fixtures
from tests.test_vector_map_topology import load_mask

pytest.importorskip("vtracer", reason="vtracer==0.6.15 is not installed. Install the vector extra.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map_vtracer as adapter  # noqa: E402

SCAD = Path(__file__).parent / "fixtures" / "vector_map" / "relief_proof.scad"
MM_PER_PX = 25.4 / 96  # The SVG header below states mm, so the import dpi does not apply.
THICKNESS = 2.0


def _openscad():
    if os.environ.get("OPENSCAD"):
        return shlex.split(os.environ["OPENSCAD"])
    found = shutil.which("openscad") or next(iter(sorted(glob.glob("/Applications/OpenSCAD*.app/Contents/MacOS/OpenSCAD"))), None)
    return [found] if found else None


OPENSCAD = _openscad()
needs_openscad = pytest.mark.skipif(OPENSCAD is None, reason="OpenSCAD is absent. Set OPENSCAD to its command.")


def _is_flatpak(command):
    return bool(command) and Path(command[0]).name == "flatpak"


def _under_home(path):
    return Path(path).resolve().is_relative_to(Path.home().resolve())


def choose_work_dir(tmp_path, command):
    """Return (directory, created). Flatpak OpenSCAD reads $HOME only, and pytest tmp_path is outside it."""
    if not _is_flatpak(command) or _under_home(tmp_path):
        return tmp_path, False
    return Path(tempfile.mkdtemp(prefix=".st-vector-map-openscad-", dir=Path.home())), True


def sandbox_error(command, paths):
    """Return a diagnostic when a Flatpak OpenSCAD would get a path it cannot read."""
    if not _is_flatpak(command):
        return None
    outside = [str(path) for path in paths if not _under_home(path)]
    if not outside:
        return None
    return (
        f"Flatpak OpenSCAD can read $HOME only (filesystems=home). It cannot read: {', '.join(outside)}. "
        "Use a directory under $HOME."
    )


@pytest.fixture
def render_dir(tmp_path):
    work, created = choose_work_dir(tmp_path, OPENSCAD)
    yield work
    if created:
        shutil.rmtree(work, ignore_errors=True)


def physical_svg(svg, width, height):
    """Test probe only: state the canvas in mm with a px viewBox. S2.5 owns production normalization."""
    raw = f'width="{width}" height="{height}"'
    assert svg.count(raw) == 1
    return svg.replace(
        raw,
        f'width="{width * MM_PER_PX:.6f}mm" height="{height * MM_PER_PX:.6f}mm" viewBox="0 0 {width} {height}"',
    )


def render(svg, work_dir, thickness=THICKNESS, command=None):
    """Run a real OpenSCAD render of the hand-written proof SCAD. Return (vertices, triangles, log)."""
    command = command or OPENSCAD
    assert command is not None
    svg_path = work_dir / "layer.svg"
    stl_path = work_dir / "layer.stl"
    if error := sandbox_error(command, [svg_path, stl_path, SCAD]):
        raise RuntimeError(error)
    svg_path.write_text(svg)
    result = subprocess.run(
        [*command, "-o", str(stl_path), "-D", f'svg="{svg_path}"', "-D", f"thickness={thickness}", str(SCAD)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    corners = np.array(
        [[float(v) for v in m] for m in re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", stl_path.read_text())]
    )
    vertices, index = np.unique(corners.round(6), axis=0, return_inverse=True)
    return vertices, index.reshape(-1, 3), result.stderr


def mesh_components(vertices, triangles):
    """Return (bounds_min, bounds_max, genus) for each connected closed surface."""
    parent = list(range(len(vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in triangles:
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)
    groups = {}
    for tri in triangles:
        groups.setdefault(find(tri[0]), []).append(tri)
    components = []
    for tris in groups.values():
        tris = np.array(tris)
        used = np.unique(tris)
        edges = {tuple(sorted(e)) for t in tris for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))}
        euler = len(used) - len(edges) + len(tris)
        components.append((vertices[used].min(0), vertices[used].max(0), (2 - euler) // 2))
    return components


def expected_bounds(material):
    """Material bbox in mm, OpenSCAD Y-up: SVG row y maps to height - y."""
    height = material.shape[0]
    ys, xs = np.nonzero(material)
    return (
        np.array([xs.min(), height - 1 - ys.max(), 0.0]) * [MM_PER_PX, MM_PER_PX, 1],
        np.array([xs.max() + 1, height - ys.min(), 0.0]) * [MM_PER_PX, MM_PER_PX, 1],
    )


FIXTURES = sorted(make_fixtures.EXPECTED)


@needs_openscad
@pytest.mark.parametrize("name", FIXTURES)
def test_render_has_one_solid_per_component_and_one_genus_per_hole(name, render_dir):
    components, holes = make_fixtures.EXPECTED[name]
    material = load_mask(name)
    svg = physical_svg(adapter.trace_layer(material).svg, *material.shape[::-1])
    parts = mesh_components(*render(svg, render_dir)[:2])
    assert len(parts) == components
    assert sum(genus for *_, genus in parts) == holes


@needs_openscad
@pytest.mark.parametrize("name", FIXTURES)
def test_render_bounds_match_the_mask_in_mm(name, render_dir):
    material = load_mask(name)
    svg = physical_svg(adapter.trace_layer(material).svg, *material.shape[::-1])
    vertices, _, _ = render(svg, render_dir)
    low, high = expected_bounds(material)
    # Polygon fitting may move a curved boundary. Rectilinear fixtures stay exact.
    tolerance = 1e-4 if name in ("asymmetric", "border_touching", "separate_components") else MM_PER_PX
    assert np.allclose(vertices.min(0)[:2], low[:2], atol=tolerance)
    assert np.allclose(vertices.max(0)[:2], high[:2], atol=tolerance)
    assert vertices[:, 2].min() == pytest.approx(0.0)
    assert vertices[:, 2].max() == pytest.approx(THICKNESS)


@needs_openscad
def test_asymmetric_marker_stays_top_right(render_dir):
    """SVG is y-down and OpenSCAD is y-up. The marker must land at high x and high y."""
    material = load_mask("asymmetric")
    svg = physical_svg(adapter.trace_layer(material).svg, *material.shape[::-1])
    parts = mesh_components(*render(svg, render_dir)[:2])
    marker = min(parts, key=lambda part: np.prod(part[1][:2] - part[0][:2]))
    assert np.allclose(marker[0][:2], np.array([96, 128 - 32]) * MM_PER_PX, atol=1e-4)
    assert np.allclose(marker[1][:2], np.array([112, 128 - 16]) * MM_PER_PX, atol=1e-4)


@needs_openscad
def test_raw_vtracer_header_is_misplaced_by_openscad(render_dir):
    """Pin the OpenSCAD 2021.01 quirk that S2.5 normalization must remove.

    Without a viewBox, path coordinates stay in px while the Y flip uses the height at 72 dpi.
    """
    material = load_mask("asymmetric")
    vertices, _, _ = render(adapter.trace_layer(material).svg, render_dir)
    assert vertices[:, 0].min() == pytest.approx(16.0)
    assert vertices[:, 1].max() == pytest.approx(128 * 25.4 / 72 - 16, abs=1e-3)


# --- Flatpak sandbox: these run without OpenSCAD -----------------------------

FLATPAK = ["flatpak", "run", "org.openscad.OpenSCAD"]
# Synthetic paths: these tests must not depend on where pytest puts tmp_path (--basetemp).
OUTSIDE_HOME = Path(Path.home().anchor) / "st-vector-map-outside-home"
INSIDE_HOME = Path.home() / ".st-vector-map-openscad-probe"


def test_synthetic_paths_sit_on_each_side_of_home():
    assert not _under_home(OUTSIDE_HOME)
    assert _under_home(INSIDE_HOME)


def test_flatpak_work_dir_moves_under_home():
    work, created = choose_work_dir(OUTSIDE_HOME, FLATPAK)
    try:
        assert created
        assert work.resolve().is_relative_to(Path.home().resolve())
    finally:
        shutil.rmtree(work)


def test_flatpak_work_dir_under_home_stays_unchanged():
    assert choose_work_dir(INSIDE_HOME, FLATPAK) == (INSIDE_HOME, False)


def test_native_work_dir_stays_unchanged():
    assert choose_work_dir(OUTSIDE_HOME, ["/usr/bin/openscad"]) == (OUTSIDE_HOME, False)


def test_sandbox_error_names_the_unreadable_path():
    message = sandbox_error(FLATPAK, [Path("/tmp/layer.svg")])
    assert "/tmp/layer.svg" in message and "$HOME" in message
    assert sandbox_error(FLATPAK, [Path.home() / "layer.svg"]) is None
    assert sandbox_error(["/usr/bin/openscad"], [Path("/tmp/layer.svg")]) is None


def test_flatpak_render_outside_home_fails_before_openscad_runs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("OpenSCAD must not start")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match=r"\$HOME"):
        render("<svg/>", OUTSIDE_HOME, command=FLATPAK)
