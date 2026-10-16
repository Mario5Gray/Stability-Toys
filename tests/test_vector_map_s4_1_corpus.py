"""S4.1 corpus probe contracts (STABL-etpzsxvn).

Plan: docs/superpowers/plans/2026-10-15-vector-map-s4-1.md.
The blocking corpus run is tests/vector_map_s4_1_corpus_probe.py, run explicitly.
These collected tests use hand-built fixtures. Native OpenSCAD tests skip when it is absent.
"""

import importlib.util

import numpy as np
import pytest

from tests import vector_map_s4_1_corpus_probe as probe

needs_resvg = pytest.mark.skipif(
    importlib.util.find_spec("resvg_py") is None, reason="resvg-py==0.5.0 is not installed. Install the vector extra."
)
needs_vtracer = pytest.mark.skipif(
    importlib.util.find_spec("vtracer") is None, reason="vtracer==0.6.15 is not installed. Install the vector extra."
)


def mm_svg(paths, width, height, mm_per_px=0.5, viewbox=True):
    """Build a published-style SVG: millimetre root, pixel viewBox, polygon paths."""
    box = f' viewBox="0 0 {width} {height}"' if viewbox else ""
    body = "".join(f'<path d="{d}" fill="#000000" transform="translate({x},{y})" />' for d, x, y in paths)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width * mm_per_px}mm" '
            f'height="{height * mm_per_px}mm"{box}>{body}</svg>')


RECT = ("M0,0 L5,0 L5,6 L0,6 Z", 2, 3)  # Covers columns 2..6 and rows 3..8.
# An L shape: a 6 x 2 bar on top, a 2 x 6 stem on the left. No flip or rotation maps it to itself.
ELL = ("M0,0 L6,0 L6,2 L2,2 L2,6 L0,6 Z", 3, 2)


def ell_mask(width=12, height=10):
    mask = np.zeros((height, width), bool)
    mask[2:4, 3:9] = True
    mask[2:8, 3:5] = True
    return mask


@needs_resvg
def test_a_millimetre_root_renders_at_one_pixel_per_viewbox_unit():
    rendered = probe.render_published(mm_svg([RECT], 20, 12))
    expected = np.zeros((12, 20), bool)
    expected[3:9, 2:7] = True
    assert np.array_equal(rendered, expected)


@needs_resvg
@needs_vtracer
def test_published_coins_silhouette_renders_exactly_like_the_locked_raw_trace(tmp_path):
    """Coins is 256 x 202. The rejected DPI method moved 11 pixels on it."""
    import contextlib
    import io

    import vector_map_vtracer as adapter
    from PIL import Image

    case = probe.CORPUS / "coins"
    argv = [str(case / "source.png"), str(tmp_path / "out.svg"), "--input-kind", "image",
            "--mask", str(case / "mask.png"), "--width-mm", "100", "--layers", "silhouette"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert probe.vector_map.main(argv) == 0
    published = (tmp_path / "out.layers/silhouette.svg").read_text()
    assert 'width="100mm"' in published
    mask = np.asarray(Image.open(case / "mask.png").convert("L")) == 255
    assert mask.shape == (202, 256)
    locked = probe.metrics.render(adapter.trace_layer(mask).svg, 256, 202)
    assert np.array_equal(probe.render_published(published), locked)


@needs_resvg
def test_a_matching_trace_passes_every_fidelity_check():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10), ell_mask())
    assert result["failures"] == [] and result["passed"]


@needs_resvg
def test_a_flipped_trace_fails_orientation():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10), np.fliplr(ell_mask()))
    assert "orientation" in result["failures"]


@needs_resvg
def test_a_lost_hole_fails_topology():
    mask = np.zeros((12, 12), bool)
    mask[2:10, 2:10] = True
    mask[5:7, 5:7] = False
    result = probe.measure_fidelity(mm_svg([("M0,0 L8,0 L8,8 L0,8 Z", 2, 2)], 12, 12), mask)
    assert result["topology_mask"] == [1, 1]
    assert result["topology_render"] == [1, 0]
    assert "topology" in result["failures"]


@needs_resvg
def test_the_area_check_reads_the_published_polygon_area_not_the_raster():
    """A duplicated path renders the same pixels but doubles the shoelace area."""
    mask = np.zeros((12, 20), bool)
    mask[3:9, 2:7] = True
    result = probe.measure_fidelity(mm_svg([RECT, RECT], 20, 12), mask)
    assert result["xor_px"] == 0
    assert result["polygon_area_px"] == 60.0
    assert result["failures"] == ["area"]


@needs_resvg
def test_a_viewbox_that_differs_from_the_mask_fails_scale_without_raising():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10), np.zeros((10, 14), bool))
    assert "scale" in result["failures"] and not result["passed"]


def test_an_svg_without_a_viewbox_is_a_failed_fidelity_row():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10, viewbox=False), ell_mask())
    assert result["passed"] is False
    assert result["failures"] == ["render"]
    assert "viewBox" in result["error"]
