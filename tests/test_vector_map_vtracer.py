"""S2.4 VTracer adapter contract (STABL-lohtpfiy).

Real vtracer==0.6.15 conversions. Mocks are not tracing proof (spec 10).
Spies below wrap the real upstream call. They only observe the arguments.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.vector_map import make_fixtures
from tests.test_vector_map_topology import (
    RSVG,
    TRACE_OPTIONS,
    covers_canvas,
    load_mask,
    raster_topology,
    render,
    subpaths,
    svg_paths,
)

vtracer = pytest.importorskip(
    "vtracer", reason="vtracer==0.6.15 is not installed. S2.1 adds the vector extra."
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map_vtracer as adapter  # noqa: E402

TOPOLOGY_FIXTURES = ["asymmetric", "border_touching", "donut", "nested_island", "separate_components"]
UPSTREAM_PARAMS = set(inspect.signature(vtracer.convert_raw_image_to_svg).parameters) - {"img_bytes"}


@pytest.fixture
def upstream_calls(monkeypatch):
    """Record each real convert_raw_image_to_svg call. Fail on any other interface."""
    calls = []
    real = vtracer.convert_raw_image_to_svg

    def spy(img_bytes, **kwargs):
        calls.append(kwargs)
        return real(img_bytes, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the adapter must use convert_raw_image_to_svg only")

    monkeypatch.setattr(vtracer, "convert_raw_image_to_svg", spy)
    monkeypatch.setattr(vtracer, "convert_image_to_svg_py", forbidden)
    monkeypatch.setattr(vtracer, "convert_pixels_to_svg", forbidden)
    return calls


# --- Input layer checks: rejected before the upstream call -----------------


@pytest.mark.parametrize(
    "layer",
    [
        np.zeros((8, 8, 4), np.uint8),  # RGBA
        np.zeros((8, 8, 2), np.uint8),  # LA
        np.zeros((8, 8, 3), np.uint8),  # RGB
    ],
    ids=["rgba", "la", "rgb"],
)
def test_rejects_multichannel_layer(layer, upstream_calls):
    with pytest.raises(ValueError, match="2-D"):
        adapter.trace_layer(layer)
    assert upstream_calls == []


@pytest.mark.parametrize(
    "layer",
    [
        np.array([[0, 128], [255, 0]], np.uint8),
        np.array([[0, 1], [1, 0]], np.uint8),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0, 255], [255, 0]], np.int64),
    ],
    ids=["gray-128", "uint8-0-1", "float", "int64"],
)
def test_rejects_nonbinary_layer(layer, upstream_calls):
    with pytest.raises(ValueError, match="binary"):
        adapter.trace_layer(layer)
    assert upstream_calls == []


@pytest.mark.parametrize("layer", [np.zeros((0, 8), bool), [[0, 255]], "mask.png"], ids=["empty", "list", "path"])
def test_rejects_non_array_or_empty_layer(layer, upstream_calls):
    with pytest.raises((TypeError, ValueError)):
        adapter.trace_layer(layer)
    assert upstream_calls == []


def test_bool_and_uint8_layers_trace_identically():
    material = load_mask("donut")
    as_bool = adapter.trace_layer(material).svg
    as_uint8 = adapter.trace_layer(np.where(material, 255, 0).astype(np.uint8)).svg
    assert as_bool == as_uint8


# --- Option checks: names, enums and ranges before the upstream call -------


@pytest.mark.parametrize(
    "options, message",
    [
        ({"colormode": "binary"}, "unknown"),
        ({"bogus": 1}, "unknown"),
        ({"corner_threshold": 60}, "polygon"),
        ({"length_threshold": 4.0}, "polygon"),
        ({"splice_threshold": 45}, "polygon"),
        ({"path_precision": 8}, "polygon"),
        ({"max_iterations": 10}, "polygon"),
        ({"mode": "spline"}, "blocked"),
        ({"mode": "none"}, "mode"),
        ({"mode": "bogus"}, "mode"),
        ({"mode": "POLYGON"}, "mode"),
        ({"filter_speckle": -1}, "filter_speckle"),
        ({"filter_speckle": 129}, "filter_speckle"),
        ({"filter_speckle": 4.5}, "filter_speckle"),
        ({"filter_speckle": "4"}, "filter_speckle"),
        ({"filter_speckle": True}, "filter_speckle"),
    ],
)
def test_rejects_unchecked_options(options, message, upstream_calls):
    with pytest.raises(ValueError, match=message):
        adapter.trace_layer(load_mask("donut"), options)
    assert upstream_calls == []


@pytest.mark.parametrize(
    "options",
    [[], "", False, 0, [("mode", "polygon")], "mode", ("mode",), 4],
    ids=["empty-list", "empty-str", "false", "zero", "pair-list", "str", "tuple", "int"],
)
def test_rejects_non_mapping_options(options, upstream_calls):
    with pytest.raises(TypeError, match="mapping"):
        adapter.trace_layer(load_mask("donut"), options)
    assert upstream_calls == []


def test_accepts_none_and_empty_mapping_as_defaults():
    assert adapter.trace_layer(load_mask("donut"), None).upstream_args == (
        adapter.trace_layer(load_mask("donut"), {}).upstream_args
    )


@pytest.mark.parametrize("speckle", [0, 128])
def test_accepts_filter_speckle_bounds(speckle):
    result = adapter.trace_layer(load_mask("donut"), {"mode": "polygon", "filter_speckle": speckle})
    assert result.upstream_args["filter_speckle"] == speckle


# --- The upstream call -----------------------------------------------------


def test_passes_every_upstream_argument_explicitly(upstream_calls):
    adapter.trace_layer(load_mask("donut"))
    assert len(upstream_calls) == 1
    assert set(upstream_calls[0]) == UPSTREAM_PARAMS
    assert all(value is not None for value in upstream_calls[0].values())
    assert upstream_calls[0]["img_format"] == "png"
    assert upstream_calls[0]["colormode"] == "binary"
    assert upstream_calls[0]["mode"] == "polygon"


def test_resolved_values_match_the_upstream_call(upstream_calls):
    result = adapter.trace_layer(load_mask("donut"), {"filter_speckle": 6})
    assert result.upstream_args == upstream_calls[0]
    assert result.upstream_args["filter_speckle"] == 6
    assert result.vtracer_version == "0.6.15"


def test_resolve_options_is_public_and_needs_no_upstream_call(upstream_calls):
    """S2.2 config checks resolved values through this function before any trace."""
    assert adapter.resolve_options({"filter_speckle": 8}) == {"mode": "polygon", "filter_speckle": 8}
    with pytest.raises(ValueError, match="filter_speckle"):
        adapter.resolve_options({"filter_speckle": 999})
    assert upstream_calls == []


def test_resolved_values_cannot_change_a_later_call():
    first = adapter.trace_layer(load_mask("donut"))
    first.upstream_args["filter_speckle"] = 99
    assert adapter.trace_layer(load_mask("donut")).upstream_args["filter_speckle"] == 4


# --- Polarity and preserved SVG structure ----------------------------------


@pytest.mark.parametrize("name", TOPOLOGY_FIXTURES)
def test_white_material_becomes_traced_material(name):
    components, holes = make_fixtures.EXPECTED[name]
    material = load_mask(name)
    svg = adapter.trace_layer(material).svg
    paths = svg_paths(svg)
    assert not covers_canvas(svg, *material.shape[::-1])
    assert len(paths) == components
    assert sum(len(subpaths(d)) for d, *_ in paths) == components + holes


@pytest.mark.parametrize("name", TOPOLOGY_FIXTURES)
def test_returns_upstream_svg_unchanged(name):
    """Paths, compound paths, winding and transforms stay as upstream wrote them."""
    material = load_mask(name)
    upstream = vtracer.convert_raw_image_to_svg(
        _png(np.where(material, 0, 255)), img_format="png", **TRACE_OPTIONS
    )
    assert adapter.trace_layer(material).svg == upstream


@pytest.mark.skipif(RSVG is None, reason="rsvg-convert is absent. Q2 selects the renderer.")
@pytest.mark.parametrize("name", TOPOLOGY_FIXTURES)
def test_rendered_adapter_output_matches_mask(name):
    material = load_mask(name)
    rendered = render(adapter.trace_layer(material).svg, *material.shape[::-1])
    assert raster_topology(rendered) == make_fixtures.EXPECTED[name]
    error = np.count_nonzero(rendered ^ material)
    for flipped in (np.fliplr(material), np.flipud(material), material.T):
        if np.array_equal(flipped, material):
            continue  # A symmetric fixture cannot show orientation under this flip.
        assert error < np.count_nonzero(rendered ^ flipped)


def test_empty_layer_returns_zero_paths():
    result = adapter.trace_layer(np.zeros((32, 32), bool))
    assert svg_paths(result.svg) == []


def test_full_material_layer_is_an_intended_full_canvas():
    svg = adapter.trace_layer(np.ones((32, 32), bool)).svg
    assert covers_canvas(svg, 32, 32)


# --- Full-canvas rejection -------------------------------------------------


def test_check_rejects_unintended_full_canvas():
    material = load_mask("donut")
    naive = vtracer.convert_raw_image_to_svg(
        _png(np.where(material, 255, 0)), img_format="png", **TRACE_OPTIONS
    )
    with pytest.raises(adapter.UnintendedBackgroundError):
        adapter.check_full_canvas(naive, material)


@pytest.mark.parametrize("name", TOPOLOGY_FIXTURES)
def test_check_accepts_correct_trace(name):
    material = load_mask(name)
    adapter.check_full_canvas(adapter.trace_layer(material).svg, material)


def _cross(size=32, arm=(14, 18)):
    """Material that touches all four sides but covers far less than the canvas."""
    material = np.zeros((size, size), bool)
    material[arm[0]:arm[1], :] = True
    material[:, arm[0]:arm[1]] = True
    return material


def test_cross_touching_every_side_is_accepted():
    material = _cross()
    assert material.sum() == 240
    result = adapter.trace_layer(material)
    assert len(svg_paths(result.svg)) == 1


def test_check_accepts_cross_trace():
    material = _cross()
    svg = vtracer.convert_raw_image_to_svg(_png(np.where(material, 0, 255)), img_format="png", **TRACE_OPTIONS)
    adapter.check_full_canvas(svg, material)


def test_check_accepts_full_canvas_when_material_covers_the_border():
    material = np.ones((32, 32), bool)
    material[10:20, 10:20] = False
    adapter.check_full_canvas(adapter.trace_layer(material).svg, material)


def _png(gray):
    from io import BytesIO

    from PIL import Image

    out = BytesIO()
    Image.fromarray(gray.astype(np.uint8)).save(out, "PNG")
    return out.getvalue()


def test_speckle_filtered_border_notch_is_accepted():
    """filter_speckle removes a 1 px border notch, so upstream emits the exact canvas rectangle."""
    material = np.ones((64, 64), bool)
    material[0, 30] = False
    svg = adapter.trace_layer(material).svg
    assert covers_canvas(svg, 64, 64)


def _border_material(size, count):
    """A size x size layer whose first `count` unique border pixels are material."""
    border = [(0, x) for x in range(size)] + [(size - 1, x) for x in range(size)]
    border += [(y, 0) for y in range(1, size - 1)] + [(y, size - 1) for y in range(1, size - 1)]
    assert len(set(border)) == len(border) == 4 * size - 4
    material = np.zeros((size, size), bool)
    for y, x in border[:count]:
        material[y, x] = True
    return material


def _canvas_rectangle_svg(size):
    svg = adapter.trace_layer(np.ones((size, size), bool)).svg
    assert covers_canvas(svg, size, size)
    return svg


def test_check_rejects_canvas_rectangle_at_a_border_tie():
    material = _border_material(32, 62)  # 62 of 124 unique border pixels
    with pytest.raises(adapter.UnintendedBackgroundError):
        adapter.check_full_canvas(_canvas_rectangle_svg(32), material)


def test_check_accepts_canvas_rectangle_with_border_majority():
    material = _border_material(32, 63)  # 63 of 124 unique border pixels
    adapter.check_full_canvas(_canvas_rectangle_svg(32), material)
