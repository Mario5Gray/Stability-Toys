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
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.vector_map import make_fixtures
from tests.test_vector_map_topology import load_mask

pytest.importorskip("vtracer", reason="vtracer==0.6.15 is not installed. Install the vector extra.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map_vtracer as adapter  # noqa: E402
from vector_map_config import Canvas  # noqa: E402
from vector_map_svg import inspect_svg, normalize_svg, size_svg  # noqa: E402

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
    """Production normalizer (S2.5, STABL-npoznayt) at MM_PER_PX. Replaces the S1.3 string-rewrite probe."""
    return normalize_svg(svg, Canvas(width, height, width * MM_PER_PX, height * MM_PER_PX, MM_PER_PX)).svg


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


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def edge_triangles(triangles):
    """Return a map of edge to triangle rows. An edge is a sorted vertex pair."""
    found = {}
    for row, (a, b, c) in enumerate(triangles):
        for first, second in ((a, b), (b, c), (a, c)):
            found.setdefault((min(first, second), max(first, second)), []).append(row)
    return found


def nonmanifold_edge_count(triangles):
    """Count edges that are not shared by exactly two triangles.

    A closed surface uses every edge twice. Any other count means the exported
    mesh is not a valid 2-manifold, whatever the tool's exit code or warning.
    """
    return sum(1 for rows in edge_triangles(triangles).values() if len(rows) != 2)


def nonmanifold_errors(triangles):
    """Return errors for a triangle set whose edges are not each used exactly twice."""
    count = nonmanifold_edge_count(triangles)
    return [] if count == 0 else [f"{count} mesh edges are not shared by exactly two triangles."]


def nonmanifold_stl_errors(stl_path):
    """Return errors for an exported STL whose edges are not each used exactly twice.

    Read the mesh, not the tool's prose. OpenSCAD 2026.10.11 can exit zero, print
    "Top level object is a 3D object (manifold)", and export a pinch
    (STABL-jbjtmecm).
    """
    triples = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", Path(stl_path).read_text())
    if not triples:
        return [f"OpenSCAD emitted no STL triangles: {Path(stl_path).name}."]
    corners = np.array([[float(value) for value in row] for row in triples])
    _, index = np.unique(corners.round(6), axis=0, return_inverse=True)
    return nonmanifold_errors(index.reshape(-1, 3))


def _genus(triangles):
    """Euler genus of one closed surface. Call it on a manifold sheet only."""
    edges = {tuple(sorted(e)) for t in triangles for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))}
    euler = len(np.unique(triangles)) - len(edges) + len(triangles)
    return (2 - euler) // 2


def mesh_components(vertices, triangles):
    """Return (bounds_min, bounds_max, genus) for each connected closed surface.

    Shells come from shared vertices, so the solid count matches the mask. The
    genus comes from the shell's manifold sheets, because OpenSCAD 2026 exports
    sheets that share edges. A single Euler count over the glued complex then
    reports a negative genus for a surface with holes (STABL-eeieoujm).
    """
    shell_parent = list(range(len(vertices)))
    for a, b, c in triangles:
        shell_parent[_find(shell_parent, b)] = _find(shell_parent, a)
        shell_parent[_find(shell_parent, c)] = _find(shell_parent, a)

    sheet_parent = list(range(len(triangles)))
    for rows in edge_triangles(triangles).values():
        if len(rows) == 2:
            sheet_parent[_find(sheet_parent, rows[1])] = _find(sheet_parent, rows[0])

    shells = {}
    for row, tri in enumerate(triangles):
        shells.setdefault(_find(shell_parent, tri[0]), []).append(row)

    components = []
    for rows in shells.values():
        tris = np.array([triangles[row] for row in rows])
        used = np.unique(tris)
        sheets = {}
        for row in rows:
            sheets.setdefault(_find(sheet_parent, row), []).append(triangles[row])
        genus = sum(_genus(np.array(sheet)) for sheet in sheets.values())
        components.append((vertices[used].min(0), vertices[used].max(0), genus))
    return components


def expected_bounds(material):
    """Material bbox in mm, OpenSCAD Y-up: SVG row y maps to height - y."""
    height = material.shape[0]
    ys, xs = np.nonzero(material)
    return (
        np.array([xs.min(), height - 1 - ys.max(), 0.0]) * [MM_PER_PX, MM_PER_PX, 1],
        np.array([xs.max() + 1, height - ys.min(), 0.0]) * [MM_PER_PX, MM_PER_PX, 1],
    )


def cube_mesh(origin=(0.0, 0.0, 0.0)):
    """A unit cube as 8 vertices and 12 triangles. Each quad splits from its first corner.

    The split is fixed so two cubes placed edge to edge share one edge in four triangles.
    """
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    quads = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    triangles = []
    for a, b, c, d in quads:
        triangles += [[a, b, c], [a, c, d]]
    return corners + np.asarray(origin, dtype=float), np.array(triangles)


def pinched_pair():
    """Two unit cubes that share the vertical edge at x=1, y=1 (STABL-eeieoujm)."""
    left_vertices, left_triangles = cube_mesh()
    right_vertices, right_triangles = cube_mesh((1.0, 1.0, 0.0))
    vertices = np.vstack([left_vertices, right_vertices])
    triangles = np.vstack([left_triangles, right_triangles + len(left_vertices)])
    # Weld the two shared corners, as the STL reader does after rounding.
    welded, index = np.unique(vertices.round(6), axis=0, return_inverse=True)
    return welded, index[triangles]


def test_a_clean_cube_is_one_solid_of_genus_zero():
    parts = mesh_components(*cube_mesh())
    assert [part[2] for part in parts] == [0]
    assert nonmanifold_edge_count(cube_mesh()[1]) == 0


def test_nonmanifold_edge_count_finds_an_edge_shared_by_four_triangles():
    """The pinched pair shares one edge. A tool that exits zero still exported it."""
    assert nonmanifold_edge_count(pinched_pair()[1]) == 1


@needs_openscad
def test_nonmanifold_stl_errors_refuses_a_pinch_the_default_backend_hides(tmp_path):
    """Two cubes share one edge, and the tool exits zero.

    OpenSCAD 2026.10.11 on the Manifold backend prints "Top level object is a 3D
    object (manifold):" and "Genus: -1" for this mesh, with no 2-manifold
    warning. The gates look for "2-manifold", so they pass. The mesh is the proof
    on every backend (STABL-jbjtmecm).
    """
    work, _ = choose_work_dir(tmp_path, OPENSCAD)
    stl = work / "pinched.stl"
    source = work / "pinched.scad"
    source.write_text("cube(1);\ntranslate([1, 1, 0]) cube(1);\n")
    run = subprocess.run([*OPENSCAD, "-o", str(stl), str(source)],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    errors = nonmanifold_stl_errors(stl)
    assert errors and "not shared by exactly two triangles" in errors[0]

    clean, _ = choose_work_dir(tmp_path / "clean", OPENSCAD)
    clean.mkdir(exist_ok=True)
    clean_source = clean / "cube.scad"
    clean_source.write_text("cube(1);\n")
    clean_stl = clean / "cube.stl"
    subprocess.run([*OPENSCAD, "-o", str(clean_stl), str(clean_source)],
                   capture_output=True, text=True, timeout=300, check=True)
    assert nonmanifold_stl_errors(clean_stl) == []


def test_a_pinched_pair_is_two_closed_surfaces_not_one_negative_genus():
    """A single Euler count over the glued complex reports -1. The sheets report 0.

    OpenSCAD 2026 exports such pins, so the genus gate must read each sheet
    (STABL-eeieoujm).
    """
    parts = mesh_components(*pinched_pair())
    assert sum(genus for *_, genus in parts) == 0


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
def test_production_sized_svg_renders_at_the_requested_width(render_dir):
    """S2.2 (STABL-ascsgqha): size_svg output imports at the requested physical width."""
    material = load_mask("asymmetric")
    height, width = material.shape
    canvas = Canvas(width, height, 100.0, 100.0 * height / width, 100.0 / width)
    vertices, _, _ = render(size_svg(adapter.trace_layer(material).svg, canvas), render_dir)
    low, high = expected_bounds(material)
    scale = canvas.mm_per_px / MM_PER_PX
    assert np.allclose(vertices.min(0)[:2], low[:2] * scale, atol=1e-4)
    assert np.allclose(vertices.max(0)[:2], high[:2] * scale, atol=1e-4)


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


# --- S2.5 degenerate subpaths (STABL-npoznayt) --------------------------------


def without_degenerate_subpaths(svg):
    """Polygon-only control: drop closed subpaths with fewer than 3 distinct points. Test-only."""
    root = ET.fromstring(svg)
    for element in root.iter("{http://www.w3.org/2000/svg}path"):
        kept, current = [], []
        for token in element.get("d").split():
            current.append(token)
            if token == "Z":
                points = {item[1:] for item in current if item[0] in "ML"}
                if len(points) >= 3:
                    kept.extend(current)
                current = []
        element.set("d", " ".join(kept) + " ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def noise(seed=0, size=64):
    return np.random.default_rng(seed).random((size, size)) > 0.5


@needs_openscad
def test_degenerate_subpaths_import_like_the_polygon_only_control(render_dir):
    """Required acceptance: OpenSCAD imports M p Z inside a valid path without changing the mesh."""
    material = noise()
    raw = adapter.trace_layer(material).svg
    height, width = material.shape
    degenerate = inspect_svg(raw, width_px=width, height_px=height).metrics.degenerate_subpaths
    assert degenerate > 0
    one_point = [
        (first, second)
        for element in ET.fromstring(raw).iter("{http://www.w3.org/2000/svg}path")
        for first, second in zip(element.get("d").split(), element.get("d").split()[1:])
        if first[0] == "M" and second == "Z"
    ]
    assert one_point, "the noise trace must contain an M p Z subpath"
    with_points = physical_svg(raw, width, height)
    control = without_degenerate_subpaths(with_points)
    assert control != with_points
    vertices, triangles, log = render(with_points, render_dir)
    control_vertices, control_triangles, _ = render(control, render_dir)
    assert np.array_equal(vertices, control_vertices)
    assert len(triangles) == len(control_triangles)
    summary = sorted((tuple(low), tuple(high), genus) for low, high, genus in mesh_components(vertices, triangles))
    control_summary = sorted(
        (tuple(low), tuple(high), genus) for low, high, genus in mesh_components(control_vertices, control_triangles)
    )
    assert summary == control_summary
    assert vertices[:, 2].min() == pytest.approx(0.0)
    assert vertices[:, 2].max() == pytest.approx(THICKNESS)
