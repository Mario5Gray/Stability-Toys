"""S1.4 acceptance corpus v1, locked tolerances, and fitting defaults (STABL-cbwzjyky).

Spec 10 and 11.1. Pure metric and inventory tests do not need resvg.
Render tests skip until S2.7 installs resvg-py in package environments.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.fixtures.vector_map.corpus import make_corpus, metrics

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(make_corpus.__file__).parent
CASES = sorted(make_corpus.CASES)

needs_resvg = pytest.mark.skipif(
    importlib.util.find_spec("resvg_py") is None,
    reason="resvg-py==0.5.0 is absent. S2.7 installs it in the vector extra.",
)


def inventory():
    return json.loads((CORPUS / "corpus.json").read_text())


def load(case, kind):
    return np.asarray(Image.open(CORPUS / case / f"{kind}.png"))


def square(rows, cols, top, left, size):
    out = np.zeros((rows, cols), bool)
    out[top : top + size, left : left + size] = True
    return out


# --- Locked phase-one rules ---------------------------------------------------


def test_tolerances_are_locked():
    """Phase one locks these values. Do not relax them to make a candidate pass (spec 10)."""
    assert metrics.TOLERANCES == {
        "topology": "exact",
        "bounds_shift_px": 1.0,
        "bbox_size_delta_px": 1.0,
        "centroid_shift_px": 0.5,
        "orientation": "strictly_lowest_xor",
        "xor_per_perimeter_px": 0.5,
        "boundary_max_distance_px": 1.5,
        "area_dev_abs_per_perimeter_px": 0.5,
        "mesh_volume_rel": 1e-4,
        "mesh_z_abs_mm": 1e-6,
    }


def test_speckle_grid_is_locked():
    assert metrics.SPECKLE_GRID == (0, 1, 2, 4, 8, 16, 32, 64, 128)


def test_render_settings_are_locked():
    assert metrics.RENDER == {
        "renderer": "resvg-py",
        "version": "0.5.0",
        "shape_rendering": "crisp_edges",
        "background": "#ffffff",
        "material": "luma < 128",
        "resolution": "processing size, 1 SVG unit = 1 px",
    }


def test_timing_protocol_is_locked():
    assert metrics.TIMING == {"warmup_calls": 1, "timed_calls": 5, "statistic": "median"}


# --- Topology and boundary algorithms ----------------------------------------


def test_topology_counts_4_connected_components_and_enclosed_holes():
    ring = square(9, 9, 1, 1, 7)
    ring[3:6, 3:6] = False
    assert metrics.topology(ring) == (1, 1)
    two = square(8, 12, 1, 1, 3) | square(8, 12, 1, 7, 3)
    assert metrics.topology(two) == (2, 0)


def test_topology_ignores_background_that_touches_the_border():
    notch = square(6, 6, 0, 0, 6)
    notch[0:3, 2:4] = False
    assert metrics.topology(notch) == (1, 0)


def test_diagonal_contact_breaks_4_8_agreement():
    checker = np.array([[1, 0], [0, 1]], bool)
    assert not metrics.connectivity_agrees(checker)
    assert metrics.connectivity_agrees(square(5, 5, 1, 1, 3))


def test_perimeter_counts_adjacent_unequal_pairs_inside_the_canvas():
    assert metrics.perimeter(square(6, 6, 1, 1, 3)) == 12
    # The canvas edge is not a boundary.
    assert metrics.perimeter(square(4, 4, 0, 0, 4)) == 0
    assert metrics.perimeter(square(4, 4, 0, 0, 2)) == 4


def test_boundary_distance_is_center_distance_to_opposite_class_minus_half():
    mask = square(9, 9, 0, 0, 9)
    mask[:, 5:] = False
    distance = metrics.boundary_distance(mask)
    assert distance[4, 4] == pytest.approx(0.5)
    assert distance[4, 5] == pytest.approx(0.5)
    assert distance[4, 2] == pytest.approx(2.5)
    assert distance[4, 7] == pytest.approx(2.5)
    assert distance[4, 8] == pytest.approx(3.5)


def test_boundary_distance_is_infinite_without_an_opposite_class():
    assert np.isinf(metrics.boundary_distance(np.ones((3, 3), bool))).all()


def test_bounds_and_centroid_use_pixel_edges_and_centers():
    mask = square(10, 10, 2, 3, 4)
    assert metrics.bounds(mask) == (3, 2, 7, 6)
    assert metrics.centroid(mask) == pytest.approx((5.0, 4.0))
    assert metrics.bounds(np.zeros((3, 3), bool)) is None


def test_orientation_variants_keep_shape_and_skip_identical_arrays():
    tall = square(6, 4, 0, 0, 2)
    names = [name for name, _ in metrics.orientation_variants(tall)]
    assert names == ["fliplr", "flipud", "rot180"]
    # Square: transpose variants join. A corner block equals its transpose, so that one is skipped.
    sq = square(6, 6, 0, 0, 2)
    names = [name for name, _ in metrics.orientation_variants(sq)]
    assert names == ["fliplr", "flipud", "rot180", "antitranspose", "rot90", "rot270"]
    centered = square(6, 6, 2, 2, 2)
    assert [name for name, _ in metrics.orientation_variants(centered)] == []


# --- SVG parsing --------------------------------------------------------------

SVG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="20" height="10">\n'
    '<path d="M0,0 L8,0 L8,8 L0,8 Z M2,2 L2,6 L6,6 L6,2 Z " fill="#000000" transform="translate(1,1)"/>\n'
    '<path d="M0,0 L3,0 L3,3 L0,3 Z " fill="#000000" transform="translate(12,2)"/>\n'
    "</svg>"
)


def test_svg_complexity_counts_xml_paths_moves_and_commands():
    assert metrics.svg_complexity(SVG) == {
        "paths": 2,
        "subpaths": 3,
        "commands": 15,
        "svg_bytes": len(SVG.encode("utf-8")),
    }


def test_svg_complexity_rejects_a_non_polygon_command():
    with pytest.raises(ValueError, match="C1,2"):
        metrics.svg_complexity(SVG.replace("L3,0", "C1,2"))


def test_polygon_area_subtracts_opposite_winding_holes():
    assert metrics.polygon_area(SVG) == pytest.approx(64 - 16 + 9)


def test_svg_canvas_reads_the_root_size():
    assert metrics.svg_canvas(SVG) == (20, 10)


# --- Per-case evaluation ------------------------------------------------------


def evaluate(mask, render, area=None):
    return metrics.evaluate(mask, render, polygon_area=float(mask.sum()) if area is None else area)


def test_identical_render_passes_every_rule():
    mask = square(20, 30, 3, 4, 9) | square(20, 30, 4, 18, 5)
    result = evaluate(mask, mask.copy())
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["xor_px"] == 0


def test_one_pixel_shift_passes_bounds_but_fails_two():
    mask = square(20, 20, 5, 5, 6)
    assert evaluate(mask, np.roll(mask, 1, axis=1))["checks"]["bounds"] is True
    result = evaluate(mask, np.roll(mask, 2, axis=1))
    assert result["checks"]["bounds"] is False
    assert "bounds" in result["failures"]


def test_lost_hole_fails_topology():
    mask = square(12, 12, 1, 1, 10)
    mask[5:7, 5:7] = False
    result = evaluate(mask, square(12, 12, 1, 1, 10))
    assert result["topology_mask"] == [1, 1]
    assert result["topology_render"] == [1, 0]
    assert result["checks"]["topology"] is False


def test_mirrored_render_fails_orientation():
    mask = square(16, 16, 2, 2, 10)
    mask[2:6, 6:12] = False  # an asymmetric notch
    mask[8:12, 2:5] = False
    result = evaluate(mask, np.fliplr(mask))
    assert result["checks"]["orientation"] is False


def test_area_deviation_records_signed_and_absolute_values():
    mask = square(10, 10, 2, 2, 4)  # area 16, perimeter 16
    result = evaluate(mask, mask, area=20.0)
    assert result["area_dev_signed_per_perimeter_px"] == pytest.approx(0.25)
    assert result["area_dev_abs_per_perimeter_px"] == pytest.approx(0.25)
    result = evaluate(mask, mask, area=4.0)
    assert result["area_dev_signed_per_perimeter_px"] == pytest.approx(-0.75)
    assert result["checks"]["area"] is False


def test_empty_render_fails_without_raising():
    mask = square(10, 10, 2, 2, 4)
    result = evaluate(mask, np.zeros_like(mask), area=0.0)
    assert result["passed"] is False
    assert {"topology", "bounds", "centroid"} <= set(result["failures"])


def test_shape_mismatch_fails_scale():
    mask = square(10, 10, 2, 2, 4)
    result = evaluate(mask, np.zeros((10, 11), bool))
    assert result["checks"]["scale"] is False
    assert result["passed"] is False


# --- Mesh volume rule ---------------------------------------------------------


def test_mesh_volume_of_a_unit_cube_is_one():
    corners = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 1], [1, 1, 1]], [[0, 0, 1], [1, 1, 1], [0, 1, 1]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1]], [[0, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 1]], [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 1]],
        ],
        float,
    )
    assert metrics.mesh_volume(corners) == pytest.approx(1.0)


def test_extrusion_mapping_is_locked():
    assert metrics.EXTRUSION == {
        "mm_per_px": 25.4 / 96,
        "thickness_mm": 2.0,
        "scad": "tests/fixtures/vector_map/relief_proof.scad",
    }
    assert metrics.expected_volume(96 * 96) == pytest.approx(25.4 * 25.4 * 2.0)


def test_mesh_volume_rule_is_relative():
    assert metrics.mesh_volume_ok(volume=100.00999, expected=100.0) is True
    assert metrics.mesh_volume_ok(volume=100.02, expected=100.0) is False


# --- Default selection rule ---------------------------------------------------


def row(passed, xpp, commands):
    return {"passed": passed, "xor_per_perimeter_px": xpp, "commands": commands}


def test_selection_takes_the_unique_highest_pass_count():
    results = {
        2: {"a": row(True, 0.2, 10), "b": row(True, 0.2, 10)},
        4: {"a": row(True, 0.1, 5), "b": row(False, 0.9, 5)},
    }
    assert metrics.select_default(results)[0] == 2


def test_selection_takes_a_unique_pareto_dominator_on_a_tie():
    results = {
        2: {"a": row(True, 0.2, 10), "b": row(True, 0.3, 12)},
        4: {"a": row(True, 0.2, 9), "b": row(True, 0.3, 12)},
        8: {"a": row(True, 0.25, 9), "b": row(True, 0.3, 11)},
    }
    # 4 beats 2 but not 8 (case b, 12 > 11). No unique dominator, so 4 stays.
    assert metrics.select_default(results)[0] == 4
    results[8]["a"] = row(True, 0.2, 8)
    assert metrics.select_default(results)[0] == 8


def test_selection_keeps_4_when_no_candidate_dominates():
    results = {
        2: {"a": row(True, 0.1, 12), "b": row(True, 0.3, 12)},
        4: {"a": row(True, 0.2, 10), "b": row(True, 0.3, 10)},
        8: {"a": row(True, 0.3, 8), "b": row(True, 0.3, 8)},
    }
    assert metrics.select_default(results)[0] == 4


def test_selection_needs_a_strict_improvement():
    results = {2: {"a": row(True, 0.2, 10)}, 4: {"a": row(True, 0.2, 10)}, 8: {"a": row(True, 0.2, 10)}}
    assert metrics.select_default(results)[0] == 4


def test_selection_defers_to_review_when_4_is_not_tied():
    results = {
        2: {"a": row(True, 0.1, 12), "b": row(True, 0.3, 12)},
        4: {"a": row(False, 0.2, 10), "b": row(True, 0.3, 10)},
        8: {"a": row(True, 0.3, 8), "b": row(True, 0.3, 8)},
    }
    assert metrics.select_default(results)[0] is None


# --- Corpus inventory and fixture rules ---------------------------------------


def test_corpus_has_two_cases_per_required_class():
    classes = [make_corpus.CASES[case]["class"] for case in CASES]
    assert sorted(set(classes)) == sorted(make_corpus.CLASSES)
    assert all(classes.count(name) == 2 for name in make_corpus.CLASSES)
    assert len(CASES) == 8


def test_corpus_covers_every_required_property():
    covered = {prop for case in CASES for prop in inventory()["cases"][case]["properties"]}
    assert set(make_corpus.PROPERTIES) <= covered


def test_inventory_lists_every_case_with_provenance():
    data = inventory()
    assert sorted(data["cases"]) == CASES
    for case in CASES:
        entry = data["cases"][case]
        for key in ("class", "properties", "source", "license", "processing_size", "mask_provenance", "topology"):
            assert entry[key], (case, key)
        assert entry["source"]["origin"] in {"skimage", "original"}
        assert max(entry["processing_size"]) <= make_corpus.MAX_SIDE


@pytest.mark.parametrize("case", CASES)
def test_committed_source_matches_the_generator(case):
    assert np.array_equal(load(case, "source"), make_corpus.build_source(case))


@pytest.mark.parametrize("case", CASES)
def test_committed_mask_is_reproduced_from_the_committed_source(case):
    assert np.array_equal(load(case, "mask"), make_corpus.derive_mask(case, load(case, "source")))


@pytest.mark.parametrize("case", CASES)
def test_mask_is_binary_white_material_at_processing_size(case):
    mask = load(case, "mask")
    assert mask.dtype == np.uint8 and mask.ndim == 2
    assert set(np.unique(mask)) <= {0, 255}
    assert list(mask.shape[::-1]) == inventory()["cases"][case]["processing_size"]
    assert load(case, "source").shape == mask.shape


@pytest.mark.parametrize("case", CASES)
def test_mask_topology_matches_inventory_and_connectivities_agree(case):
    material = load(case, "mask") == 255
    assert metrics.connectivity_agrees(material)
    assert list(metrics.topology(material)) == inventory()["cases"][case]["topology"]


@pytest.mark.parametrize("case", CASES)
def test_declared_properties_match_measured_values(case):
    entry = inventory()["cases"][case]
    components, holes = entry["topology"]
    props = set(entry["properties"])
    assert ("holes" in props) == (holes > 0)
    assert ("disconnected" in props) == (components > 1)
    assert props & {"sparse", "dense"}
    assert props & {"high_contrast", "low_contrast", "medium_contrast"}
    assert make_corpus.contrast_label(entry["contrast"]) in props
    assert make_corpus.density_label(entry["edge_density"]) in props


def test_every_mask_is_reviewed():
    for case in CASES:
        assert inventory()["cases"][case]["review"]["status"] == "reviewed", case


# --- Render-dependent checks (resvg) ------------------------------------------


@needs_resvg
def test_crisp_render_is_two_level_and_has_mask_shape():
    material = metrics.render(SVG, 20, 10)
    assert material.shape == (10, 20)
    assert material.dtype == bool
    assert material.sum() == 64 - 16 + 9


@needs_resvg
def test_resvg_version_matches_the_locked_renderer():
    from importlib.metadata import version

    assert version("resvg-py") == metrics.RENDER["version"]


# --- Phase two: selected default and per-case regression ------------------------
# Selection from the sweep in sweep-results.json (931b49f). These pin each case at the
# selected default, so an upstream, adapter or metric change shows up per case.

SWEEP = CORPUS / "sweep-results.json"
SELECTED_DEFAULT = 4
# Cases that fail at every grid value. Root cause: mask features 2 px wide or less.
FAILING_AT_EVERY_VALUE = {
    "bracket": ["centroid"],
    "brick": ["centroid", "topology"],
    "gravel": ["boundary", "topology"],
    "horse": ["centroid", "topology"],
}

needs_vtracer = pytest.mark.skipif(
    importlib.util.find_spec("vtracer") is None, reason="vtracer==0.6.15 is not installed. Install the vector extra."
)


def sweep():
    return json.loads(SWEEP.read_text())


def test_sweep_was_measured_with_the_locked_metrics():
    import hashlib

    data = sweep()
    assert data["phase_one_commit"] == "b95c718"
    assert data["metrics_sha256"] == hashlib.sha256((CORPUS / "metrics.py").read_bytes()).hexdigest()
    assert data["grid"] == list(metrics.SPECKLE_GRID)
    assert data["tolerances"] == metrics.TOLERANCES
    assert data["environment"]["vtracer"] == "0.6.15"
    assert data["environment"]["resvg-py"] == metrics.RENDER["version"]


def test_locked_rule_on_recorded_results_selects_the_adapter_default():
    sys.path.insert(0, str(ROOT / "scripts"))
    import vector_map_vtracer as adapter

    rows = {
        int(value): {
            case: {key: row.get(key, float("inf")) for key in ("xor_per_perimeter_px", "commands")} | {"passed": row["passed"]}
            for case, row in cases.items()
        }
        for value, cases in sweep()["results"].items()
    }
    assert metrics.select_default(rows)[0] == SELECTED_DEFAULT
    assert sweep()["selection"]["default"] == SELECTED_DEFAULT
    assert adapter.DEFAULT_OPTIONS == {"mode": "polygon", "filter_speckle": SELECTED_DEFAULT}


def test_recorded_failures_match_the_pinned_cases():
    data = sweep()
    assert data["selection"]["overrides"] == {}
    assert sorted(data["selection"]["failing_every_value"]) == sorted(FAILING_AT_EVERY_VALUE)
    for case, failures in FAILING_AT_EVERY_VALUE.items():
        assert data["results"][str(SELECTED_DEFAULT)][case]["failures"] == failures


@needs_vtracer
@needs_resvg
@pytest.mark.parametrize("case", CASES)
def test_case_result_at_the_selected_default(case):
    sys.path.insert(0, str(ROOT / "scripts"))
    import vector_map_vtracer as adapter

    material = load(case, "mask") == 255
    trace = adapter.trace_layer(material)
    assert trace.upstream_args["filter_speckle"] == SELECTED_DEFAULT
    height, width = material.shape
    assert metrics.svg_canvas(trace.svg) == (width, height)
    result = metrics.evaluate(material, metrics.render(trace.svg, width, height), metrics.polygon_area(trace.svg))
    assert result["failures"] == FAILING_AT_EVERY_VALUE.get(case, []), result
    recorded = sweep()["results"][str(SELECTED_DEFAULT)][case]
    assert result["xor_px"] == recorded["xor_px"]
    assert metrics.svg_complexity(trace.svg)["commands"] == recorded["commands"]


# --- S2.5 normalization compatibility (STABL-npoznayt) ----------------------


NORMALIZED_MM_PER_PX = 0.5  # 25.4 / 0.5 = 50.8 dpi maps one viewBox unit to exactly one pixel.


def render_normalized(svg, width, height):
    """Render a normalized (mm root) SVG with the locked resvg settings.

    resvg-py 0.5.0 sizes a mm root at dpi 0 and fails with "SVG has an invalid size".
    It also rounds the intrinsic size before scaling, so only a dpi that makes the
    intrinsic size equal the processing size reproduces the raw render exactly.
    """
    from io import BytesIO

    import resvg_py

    png = bytes(
        resvg_py.svg_to_bytes(
            svg_string=svg,
            width=width,
            height=height,
            dpi=25.4 / NORMALIZED_MM_PER_PX,
            background=metrics.RENDER["background"],
            shape_rendering=metrics.RENDER["shape_rendering"],
        )
    )
    return np.asarray(Image.open(BytesIO(png)).convert("L")) < 128


def _normalized(trace_svg, material, mm_per_px=NORMALIZED_MM_PER_PX):
    sys.path.insert(0, str(ROOT / "scripts"))
    from vector_map_config import Canvas
    from vector_map_svg import normalize_svg

    height, width = material.shape
    return normalize_svg(trace_svg, Canvas(width, height, width * mm_per_px, height * mm_per_px, mm_per_px))


@needs_vtracer
@needs_resvg
@pytest.mark.parametrize("case", CASES)
def test_normalized_corpus_case_renders_like_the_raw_trace(case):
    """Normalization changes only the root size. The locked result must not move."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import vector_map_vtracer as adapter

    material = load(case, "mask") == 255
    trace = adapter.trace_layer(material)
    height, width = material.shape
    normalized = _normalized(trace.svg, material)
    raw = metrics.render(trace.svg, width, height)
    rendered = render_normalized(normalized.svg, width, height)
    assert np.array_equal(rendered, raw)
    result = metrics.evaluate(material, rendered, metrics.polygon_area(trace.svg))
    assert result["failures"] == FAILING_AT_EVERY_VALUE.get(case, []), result
    assert normalized.metrics.commands == metrics.svg_complexity(trace.svg)["commands"]


@needs_vtracer
@needs_resvg
@pytest.mark.parametrize("name", ["asymmetric", "border_touching", "donut", "nested_island", "separate_components"])
def test_normalized_topology_fixture_renders_like_the_raw_trace(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    import vector_map_vtracer as adapter
    from tests.fixtures.vector_map import make_fixtures
    from tests.test_vector_map_topology import load_mask, raster_topology

    material = load_mask(name)
    trace = adapter.trace_layer(material)
    height, width = material.shape
    rendered = render_normalized(_normalized(trace.svg, material).svg, width, height)
    assert np.array_equal(rendered, metrics.render(trace.svg, width, height))
    assert raster_topology(rendered) == make_fixtures.EXPECTED[name]


@needs_resvg
def test_locked_renderer_cannot_size_a_mm_root_without_a_dpi():
    """Pin for S2.7 previews: the locked metrics.render rejects normalized SVG."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="5mm" viewBox="0 0 20 10"/>'
    with pytest.raises(ValueError, match="invalid size"):
        metrics.render(svg, 20, 10)
