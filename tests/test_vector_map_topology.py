"""S1.2 polarity and topology fixtures for VTracer (STABL-snyaxjef).

These tests run the real upstream converter. Mocks are not tracing proof (spec 10).
Fixtures use the wrapper convention: white (luma >= 128) is material (spec 5).
VTracer 0.6.15 binary mode traces DARK pixels as foreground (Q1, STABL-orcwoxml).
"""

import re
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from tests.fixtures.vector_map import make_fixtures

vtracer = pytest.importorskip(
    "vtracer", reason="vtracer==0.6.15 is not installed. S2.1 adds the vector extra."
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vector_map"
RSVG = shutil.which("rsvg-convert")

# Every option is explicit. The 0.6.15 stub defaults are not reliable (Q1).
TRACE_OPTIONS = dict(
    colormode="binary",
    hierarchical="stacked",
    mode="polygon",
    filter_speckle=4,
    color_precision=6,
    layer_difference=16,
    corner_threshold=60,
    length_threshold=4.0,
    max_iterations=10,
    splice_threshold=45,
    path_precision=8,
)

PATH_RE = re.compile(r'<path d="([^"]*)" fill="([^"]*)" transform="translate\(([-\d.]+),([-\d.]+)\)"/>')


def load_mask(name):
    """Return the committed fixture as a boolean material array."""
    return np.asarray(Image.open(FIXTURE_DIR / f"{name}.png").convert("L")) >= 128


def png_bytes(array):
    out = BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(out, "PNG")
    return out.getvalue()


def call_vtracer(gray):
    return vtracer.convert_raw_image_to_svg(png_bytes(gray), img_format="png", **TRACE_OPTIONS)


def trace_material(material):
    """Trace a wrapper-convention material mask. Invert: VTracer material is dark."""
    return call_vtracer(np.where(material, 0, 255))


def raster_topology(material):
    """Count material components and enclosed holes in a boolean raster."""
    _, components = ndimage.label(material)
    background, count = ndimage.label(~material)
    edge = set(np.unique(np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]])))
    holes = len(set(range(1, count + 1)) - edge)
    return components, holes


def svg_paths(svg):
    return [
        (d, fill, float(tx), float(ty)) for d, fill, tx, ty in PATH_RE.findall(svg)
    ]


def subpaths(d):
    """Split polygon-mode path data into per-subpath (x, y) point lists, before translate."""
    return [
        [tuple(float(v) for v in point.split(",")) for point in re.findall(r"-?[\d.]+,-?[\d.]+", part)]
        for part in d.split("Z")
        if part.strip()
    ]


def covers_canvas(svg, width, height):
    """True when any subpath spans the complete canvas (spec 6.2 reject case)."""
    for d, _, tx, ty in svg_paths(svg):
        for points in subpaths(d):
            xs = [x + tx for x, _ in points]
            ys = [y + ty for _, y in points]
            if (min(xs), min(ys), max(xs), max(ys)) == (0, 0, width, height):
                return True
    return False


def render(svg, width, height):
    """Rasterize with rsvg-convert. Return the boolean rendered-material array."""
    assert RSVG is not None
    result = subprocess.run(
        [RSVG, "--width", str(width), "--height", str(height), "--background-color", "white"],
        input=svg.encode(),
        capture_output=True,
        check=True,
    )
    return np.asarray(Image.open(BytesIO(result.stdout)).convert("L")) < 128


FIXTURES = sorted(make_fixtures.EXPECTED)


@pytest.mark.parametrize("name", FIXTURES)
def test_committed_fixture_matches_generator(name):
    assert np.array_equal(load_mask(name), make_fixtures.build(name) >= 128)


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_raster_has_expected_topology(name):
    material = load_mask(name)
    assert raster_topology(material) == make_fixtures.EXPECTED[name]
    # No diagonal-only contacts, so 4- and 8-connectivity agree.
    assert ndimage.label(material, structure=np.ones((3, 3)))[1] == make_fixtures.EXPECTED[name][0]


def test_untransformed_wrapper_mask_traces_full_canvas_background():
    """Pin the upstream trap. White material passed as-is becomes a full-canvas path."""
    material = load_mask("donut")
    svg = call_vtracer(np.where(material, 255, 0))
    assert covers_canvas(svg, *material.shape[::-1])


@pytest.mark.parametrize("name", FIXTURES)
def test_trace_has_no_full_canvas_background(name):
    material = load_mask(name)
    assert not covers_canvas(trace_material(material), *material.shape[::-1])


@pytest.mark.parametrize("name", FIXTURES)
def test_trace_path_structure_matches_topology(name):
    components, holes = make_fixtures.EXPECTED[name]
    paths = svg_paths(trace_material(load_mask(name)))
    assert len(paths) == components
    assert sum(len(subpaths(d)) for d, *_ in paths) == components + holes
    assert {fill for _, fill, *_ in paths} == {"#000000"}


@pytest.mark.skipif(RSVG is None, reason="rsvg-convert is absent. Q2 selects the renderer.")
@pytest.mark.parametrize("name", FIXTURES)
def test_rendered_trace_has_mask_topology(name):
    material = load_mask(name)
    rendered = render(trace_material(material), *material.shape[::-1])
    assert raster_topology(rendered) == make_fixtures.EXPECTED[name]


@pytest.mark.skipif(RSVG is None, reason="rsvg-convert is absent. Q2 selects the renderer.")
@pytest.mark.parametrize("name", ["asymmetric", "border_touching"])
def test_rendered_trace_keeps_orientation(name):
    """Topology counts are mirror-invariant. The render must match the unflipped mask best."""
    material = load_mask(name)
    rendered = render(trace_material(material), *material.shape[::-1])
    error = np.count_nonzero(rendered ^ material)
    for flipped in (np.fliplr(material), np.flipud(material), material.T):
        assert error < np.count_nonzero(rendered ^ flipped)
