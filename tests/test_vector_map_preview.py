"""S2.7 preview rendering and composition. STABL-kfrksmnp, spec 6.4.

Real vtracer==0.6.15 traces and real resvg-py==0.5.0 renders. Mocks are not render proof.
The locked corpus renderer (metrics.render) renders the raw pixel trace. It is the reference.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vtracer", reason="vtracer==0.6.15 is not installed. Install the vector extra.")
pytest.importorskip("resvg_py", reason="resvg-py==0.5.0 is not installed. Install the vector extra.")

from tests.fixtures.vector_map.corpus import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures" / "vector_map"

import vector_map_config as config  # noqa: E402
import vector_map_preview as preview  # noqa: E402
import vector_map_svg as svg_io  # noqa: E402
import vector_map_vtracer as adapter  # noqa: E402

# 25.4/96 and 1/3 broke an exact-dpi mm-root render. Keep them.
SCALES = [0.1, 1 / 3, 0.37, 0.5, 25.4 / 96, 2.54, 7.0]


def _mask(path, rows=None):
    material = np.asarray(Image.open(path).convert("L")) >= 128
    return material[:rows] if rows else material


MATERIALS = {
    "asymmetric": lambda: _mask(FIXTURES / "asymmetric.png"),
    # S2.5: gravel at a non-exact dpi rendered 475 to 6631 px wrong.
    "gravel": lambda: _mask(FIXTURES / "corpus" / "gravel" / "mask.png"),
    "gravel_wide": lambda: _mask(FIXTURES / "corpus" / "gravel" / "mask.png", rows=200),
}


def _canvas(material, mm_per_px):
    height, width = material.shape
    return config.Canvas(width, height, width * mm_per_px, height * mm_per_px, mm_per_px)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name", sorted(MATERIALS))
def test_crisp_render_of_normalized_svg_equals_raw_pixel_trace(name, scale):
    material = MATERIALS[name]()
    traced = adapter.trace_layer(material)
    canvas = _canvas(material, scale)
    normalized = svg_io.normalize_svg(traced.svg, canvas)
    rendered = preview.render_material(normalized.svg, canvas)
    height, width = material.shape
    assert rendered.shape == (height, width)
    assert np.array_equal(rendered, metrics.render(traced.svg, width, height))


@pytest.mark.parametrize("dimension", ["width_mm", "height_mm"])
def test_render_uses_physical_canvas_from_either_dimension(dimension):
    material = MATERIALS["gravel_wide"]()
    height, width = material.shape
    settings = config.resolve(config.DEFAULTS, {
        "input": FIXTURES / "asymmetric.png", "input_kind": "mask", dimension: 123.4,
    })
    canvas = config.physical_canvas(settings, width, height)
    traced = adapter.trace_layer(material)
    rendered = preview.render_material(svg_io.normalize_svg(traced.svg, canvas).svg, canvas)
    assert np.array_equal(rendered, metrics.render(traced.svg, width, height))


def test_exact_dpi_mm_root_render_is_wrong_at_non_binary_scales():
    """Pin the trap. resvg-py 0.5.0 converts mm to px in float32.

    At 25.4/96 mm/px, 67.7333 mm becomes 255.99998 px, and the render moves edges.
    S2.5 measured exact dpi only at 0.5 mm/px, where float32 is exact.
    """
    import resvg_py

    material = MATERIALS["gravel_wide"]()
    height, width = material.shape
    traced = adapter.trace_layer(material)
    canvas = _canvas(material, 25.4 / 96)
    png = resvg_py.svg_to_bytes(
        svg_string=svg_io.normalize_svg(traced.svg, canvas).svg, width=width, height=height,
        dpi=25.4 / canvas.mm_per_px, background="#ffffff", shape_rendering="crisp_edges",
    )
    rendered = np.asarray(Image.open(__import__("io").BytesIO(bytes(png))).convert("L")) < 128
    assert np.count_nonzero(rendered ^ metrics.render(traced.svg, width, height)) > 0


def test_pixel_root_changes_only_root_size():
    material = MATERIALS["asymmetric"]()
    canvas = _canvas(material, 0.37)
    normalized = svg_io.normalize_svg(adapter.trace_layer(material).svg, canvas).svg
    pixel = preview.pixel_root(normalized, canvas)
    root = svg_io.ET.fromstring(pixel)
    assert (root.get("width"), root.get("height"), root.get("viewBox")) == ("128", "128", "0 0 128 128")
    original = svg_io.ET.fromstring(normalized)
    assert [(e.tag, e.attrib) for e in root.iter()][1:] == [(e.tag, e.attrib) for e in original.iter()][1:]
