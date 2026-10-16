"""CAD compatibility step for published layer SVGs (STABL-ygtexxzr).

OpenSCAD exits 0 on a layer whose rings touch at one point, then reports a non-closed mesh.
The step closes each such contact with a small, explicit width before the layer is published.
These tests never start OpenSCAD. The native render gates live in the section at the end.
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map_cad as cad  # noqa: E402
from vector_map_config import Canvas  # noqa: E402
from vector_map_svg import normalize_svg  # noqa: E402

SVG_NS = "http://www.w3.org/2000/svg"


def canvas(mm_per_px=0.4, size=32):
    return Canvas(size, size, size * mm_per_px, size * mm_per_px, mm_per_px)


def layer(*paths, size=32):
    """A raw VTracer-shaped layer. Each path is (subpath text, translate or None)."""
    body = ""
    for data, shift in paths:
        move = f' transform="translate({shift[0]},{shift[1]})"' if shift else ""
        body += f'<path d="{data}" fill="#000000"{move}/>'
    return (f'<svg xmlns="{SVG_NS}" version="1.1" width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}">{body}</svg>')


def normalized(svg, mm_per_px=0.4, size=32):
    return normalize_svg(svg, canvas(mm_per_px, size))


def rings_of(svg_text):
    """Every subpath of every path as absolute points, via an XML parse."""
    out = []
    for path in ET.fromstring(svg_text).iter(f"{{{SVG_NS}}}path"):
        shift = (0.0, 0.0)
        if transform := path.get("transform"):
            shift = tuple(float(v) for v in transform[len("translate("):-1].split(","))
        for sub in (path.get("d") or "").split("Z"):
            pairs = [token[1:].split(",") for token in sub.split() if token]
            if pairs:
                out.append([(float(x) + shift[0], float(y) + shift[1]) for x, y in pairs])
    return out


def signed_area(ring):
    return sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
               for i in range(len(ring))) / 2


def material_area(svg_text):
    """Outer rings count positive and holes negative. Degenerate rings add zero."""
    return sum(signed_area(ring) for ring in rings_of(svg_text))


OUTER = "M0,0 L20,0 L20,20 L0,20 Z"
HOLE_A = "M2,2 L2,9 L9,9 L9,2 Z"
HOLE_B_TOUCHING = "M9,9 L9,16 L16,16 L16,9 Z"  # shares the vertex 9,9 with HOLE_A
HOLE_B_APART = "M11,11 L11,16 L16,16 L16,11 Z"

# A reflex notch tip at 10,10 lies on the lower edge of the hole. The wall between them is zero wide.
NOTCHED_OUTER = "M0,0 L20,0 L20,20 L12,20 L10,10 L8,20 L0,20 Z"
HOLE_ON_TIP = "M6,4 L6,10 L14,10 L14,4 Z"


def test_two_holes_sharing_a_vertex_are_reported_as_one_pinch():
    pinches = cad.find_pinches(layer((f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} ", None)))
    assert len(pinches) == 1


def test_a_hole_edge_through_another_rings_vertex_is_reported():
    pinches = cad.find_pinches(layer((f"{NOTCHED_OUTER} {HOLE_ON_TIP} ", None)))
    assert len(pinches) == 1


HOLE_ON_OUTER_EDGE = "M10,0 L6,6 L14,6 Z"  # its vertex 10,0 lies inside the edge 0,0 to 20,0 of OUTER


def test_a_hole_vertex_on_an_outer_edge_is_reported_and_closed():
    before = normalized(layer((f"{OUTER} {HOLE_ON_OUTER_EDGE} ", None)))
    assert len(cad.find_pinches(before.svg)) == 1
    result = cad.close_pinches(before, canvas())
    assert result.closed == 1 and result.unrepaired == 0
    assert cad.find_pinches(result.layer.svg) == []
    assert [len(ring) for ring in rings_of(result.layer.svg)] == [4, 3]  # a vertex moved, none added


def test_holes_that_stay_apart_have_no_pinch():
    assert cad.find_pinches(layer((f"{OUTER} {HOLE_A} {HOLE_B_APART} ", None))) == []


def test_degenerate_rings_are_not_pinches():
    line = "M9,9 L12,12 Z"  # two distinct points, no area
    assert cad.find_pinches(layer((f"{OUTER} {HOLE_A} {line} ", None))) == []


def test_contacts_are_found_after_translate_is_applied():
    svg = layer((f"{OUTER} {HOLE_A} ", (4, 4)), ("M0,0 L2,0 L2,2 L0,2 Z ", (13, 13)))
    # The hole corner 9,9 plus 4 is 13,13, where the second path begins. Without translate they are apart.
    pinches = cad.find_pinches(svg)
    assert len(pinches) == 1


def test_pinch_width_is_capped_in_millimetres():
    assert cad.pinch_width_px(canvas(mm_per_px=0.4)) == pytest.approx(0.01 / 0.4)


def test_pinch_width_never_exceeds_a_quarter_pixel_on_fine_scales():
    assert cad.pinch_width_px(canvas(mm_per_px=0.001)) == 0.25


def test_clean_layer_is_returned_byte_identical():
    clean = normalized(layer((f"{OUTER} {HOLE_A} {HOLE_B_APART} ", None)))
    result = cad.close_pinches(clean, canvas())
    assert result.layer.svg == clean.svg
    assert result.closed == 0 and result.unrepaired == 0


@pytest.mark.parametrize("data", [f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} ", f"{NOTCHED_OUTER} {HOLE_ON_TIP} "])
def test_close_pinches_leaves_no_contact(data):
    result = cad.close_pinches(normalized(layer((data, None))), canvas())
    assert result.closed == 1 and result.unrepaired == 0
    assert cad.find_pinches(result.layer.svg) == []


@pytest.mark.parametrize("data", [f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} ", f"{NOTCHED_OUTER} {HOLE_ON_TIP} "])
def test_close_pinches_adds_material_and_only_a_little(data):
    before = normalized(layer((data, None)))
    result = cad.close_pinches(before, canvas())
    gain = material_area(result.layer.svg) - material_area(before.svg)
    width = cad.pinch_width_px(canvas())
    assert gain > 0
    assert gain <= 2 * width * 24  # at most the width times twice a hole perimeter here


def test_close_pinches_keeps_the_holes_separate_and_the_outer_ring_unchanged():
    before = normalized(layer((f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} ", None)))
    after = cad.close_pinches(before, canvas()).layer.svg
    old, new = rings_of(before.svg), rings_of(after)
    assert len(new) == len(old) == 3
    assert new[0] == old[0]


def test_every_moved_point_moves_by_at_most_the_pinch_width():
    before = normalized(layer((f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} ", None)))
    after = cad.close_pinches(before, canvas()).layer.svg
    width = cad.pinch_width_px(canvas())
    for old, new in zip(rings_of(before.svg), rings_of(after)):
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert math.dist(a, b) <= width + 1e-6  # published coordinates round to 1e-6 px


def test_a_hole_edge_contact_inserts_one_vertex_into_the_hole():
    before = normalized(layer((f"{NOTCHED_OUTER} {HOLE_ON_TIP} ", None)))
    after = cad.close_pinches(before, canvas()).layer.svg
    assert [len(r) for r in rings_of(before.svg)] == [7, 4]
    assert [len(r) for r in rings_of(after)] == [7, 5]


def test_close_pinches_is_idempotent():
    once = cad.close_pinches(normalized(layer((f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} ", None))), canvas())
    twice = cad.close_pinches(once.layer, canvas())
    assert twice.layer.svg == once.layer.svg and twice.closed == 0


def test_translate_stays_on_the_path_and_the_repair_is_in_local_pixels():
    data = f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} "
    before = normalized(layer((data, (3, 5))))
    result = cad.close_pinches(before, canvas())
    assert 'transform="translate(3,5)"' in result.layer.svg
    assert result.closed == 1


def test_two_islands_meeting_at_a_point_are_joined_by_growing_the_smaller():
    # Two outer rings of different paths meet at 5,5. No hole can give way, so the smaller island grows.
    big, small = "M0,0 L5,0 L5,5 L0,5 Z ", "M5,5 L9,5 L9,9 L5,9 Z "
    before = normalized(layer((big, None), (small, None)))
    result = cad.close_pinches(before, canvas())
    assert result.closed == 1 and result.unrepaired == 0
    assert cad.find_pinches(result.layer.svg) == []
    old, new = rings_of(before.svg), rings_of(result.layer.svg)
    assert new[0] == old[0]  # the larger island is untouched
    assert math.dist(old[1][0], new[1][0]) == pytest.approx(cad.pinch_width_px(canvas()), abs=1e-6)
    assert material_area(result.layer.svg) > material_area(before.svg)


def test_normalized_bytes_metric_follows_the_repaired_text():
    result = cad.close_pinches(normalized(layer((f"{OUTER} {HOLE_A} {HOLE_B_TOUCHING} ", None))), canvas())
    assert result.layer.metrics.normalized_bytes == len(result.layer.svg.encode("utf-8"))


# --- Structure and detail contact (defect B) ----------------------------------

STRUCTURE = "M4,4 L12,4 L12,12 L4,12 Z "
DETAIL_TOUCHING = "M12,12 L18,12 L18,18 L12,18 Z "  # shares the vertex 12,12
DETAIL_APART = "M14,14 L18,14 L18,18 L14,18 Z "
DETAIL_CROSSING = "M8,8 L16,8 L16,16 L8,16 Z "  # edges cross, no vertex lies on the other ring
DETAIL_ON_EDGE = "M8,12 L16,16 L16,8 Z "  # vertex 8,12 lies on the structure edge, wedge to both sides


def test_detail_touching_structure_at_a_vertex_needs_growth():
    assert cad.structure_detail_contact(layer((STRUCTURE, None)), layer((DETAIL_TOUCHING, None)))


def test_detail_vertex_on_a_structure_edge_needs_growth():
    assert cad.structure_detail_contact(layer((STRUCTURE, None)), layer((DETAIL_ON_EDGE, None)))


def test_detail_apart_from_structure_needs_no_growth():
    assert not cad.structure_detail_contact(layer((STRUCTURE, None)), layer((DETAIL_APART, None)))


def test_detail_that_only_crosses_structure_needs_no_growth():
    assert not cad.structure_detail_contact(layer((STRUCTURE, None)), layer((DETAIL_CROSSING, None)))


def test_detail_growth_is_at_most_two_micrometres_and_a_quarter_pixel():
    assert cad.detail_growth_mm(canvas(mm_per_px=0.4)) == pytest.approx(0.002)
    assert cad.detail_growth_mm(canvas(mm_per_px=0.001)) == pytest.approx(0.25 * 0.001)


# --- Engraved recess and silhouette outline ------------------------------------
#
# The final recess is detail inside the silhouette and outside structure. Where it meets the
# outline at one isolated point, cutting it leaves two material wedges that touch at that point.

SILHOUETTE = "M0,0 L20,0 L20,20 L0,20 Z "
RECESS_TIP_ON_OUTLINE = "M10,0 L6,6 L14,6 Z "  # vertex 10,0 lies inside the outline edge
RECESS_SHARED_RUN = "M4,0 L16,0 L16,6 L4,6 Z "  # runs along the outline from 4,0 to 16,0
RECESS_APART = "M6,4 L14,4 L14,10 L6,10 Z "
STRUCTURE_LEAVING_A_TIP = "M4,0 L10,0 L4,6 Z M10,0 L16,0 L16,6 Z "  # final recess: 10,0 16,6 4,6
STRUCTURE_OVER_THE_TIP = "M5,0 L15,0 L15,7 L5,7 Z "


def recess_contacts(detail, structure=None, silhouette=SILHOUETTE):
    return cad.recess_outline_contacts(layer((silhouette, None)), layer((detail, None)),
                                       layer((structure, None)) if structure else None)


def test_recess_tip_on_the_outline_is_an_isolated_contact():
    assert recess_contacts(RECESS_TIP_ON_OUTLINE) == [(10.0, 0.0)]


def test_recess_vertex_on_an_outline_vertex_is_an_isolated_contact():
    notched = "M0,0 L20,0 L20,20 L12,20 L10,10 L8,20 L0,20 Z "  # notch tip at 10,10
    assert recess_contacts("M10,10 L6,4 L14,4 Z ", silhouette=notched) == [(10.0, 10.0)]


def test_recess_sharing_a_boundary_run_with_the_outline_is_no_contact():
    assert recess_contacts(RECESS_SHARED_RUN) == []


def test_recess_apart_from_the_outline_is_no_contact():
    assert recess_contacts(RECESS_APART) == []


def test_structure_that_cuts_the_recess_to_a_tip_makes_a_contact():
    assert recess_contacts(RECESS_SHARED_RUN, STRUCTURE_LEAVING_A_TIP) == [(10.0, 0.0)]


def test_structure_that_covers_the_tip_removes_the_contact():
    assert recess_contacts(RECESS_TIP_ON_OUTLINE, STRUCTURE_OVER_THE_TIP) == []


def test_detail_outside_the_silhouette_is_not_recess():
    # The detail tip lies outside, so the final recess follows the outline: a shared run.
    assert recess_contacts("M10,-2 L4,6 L16,6 Z ") == []


def test_recess_contacts_apply_translate():
    shifted = cad.recess_outline_contacts(layer((SILHOUETTE, None)), layer(("M0,0 L-4,6 L4,6 Z ", (10, 0))), None)
    assert shifted == [(10.0, 0.0)]


def test_recess_shrink_is_at_most_two_micrometres_and_a_quarter_pixel():
    assert cad.recess_shrink_mm(canvas(mm_per_px=0.4)) == pytest.approx(0.002)
    assert cad.recess_shrink_mm(canvas(mm_per_px=0.001)) == pytest.approx(0.25 * 0.001)


class _Settings:
    export_scad, detail_mode = True, "engraved"


def test_engraved_contact_reports_cad_recess_shrink():
    import vector_map
    from types import SimpleNamespace
    layers = {"silhouette": SimpleNamespace(svg=layer((SILHOUETTE, None))),
              "detail": SimpleNamespace(svg=layer((RECESS_TIP_ON_OUTLINE, None)))}
    diagnostics = []
    shrink = vector_map._recess_shrink(_Settings(), layers, canvas(), diagnostics)
    assert shrink == pytest.approx(0.002)
    assert [d["code"] for d in diagnostics] == ["cad_recess_shrink"]
    assert diagnostics[0]["contacts"] == 1


@pytest.mark.parametrize("mode, detail", (("engraved", RECESS_SHARED_RUN), ("raised", RECESS_TIP_ON_OUTLINE)))
def test_no_recess_shrink_without_an_engraved_isolated_contact(mode, detail):
    import vector_map
    from types import SimpleNamespace
    settings = _Settings()
    settings.detail_mode = mode
    layers = {"silhouette": SimpleNamespace(svg=layer((SILHOUETTE, None))),
              "detail": SimpleNamespace(svg=layer((detail, None)))}
    diagnostics = []
    assert vector_map._recess_shrink(settings, layers, canvas(), diagnostics) == 0.0
    assert diagnostics == []


# --- SCAD source ---------------------------------------------------------------

import vector_map_scad as scad  # noqa: E402

SCAD_LAYERS = {"silhouette": "out.layers/silhouette.svg", "structure": "out.layers/structure.svg",
               "detail": "out.layers/detail.svg"}
SCAD_HEIGHTS: dict[str, Any] = {"silhouette_thickness_mm": 1.2, "structure_height_mm": 0.6, "detail_height_mm": 0.2}


def test_scad_without_growth_is_unchanged_and_names_no_growth():
    source = scad.build_scad(canvas(), SCAD_LAYERS, **SCAD_HEIGHTS)
    assert "DETAIL_GROW" not in source and "offset(" not in source


def test_scad_with_growth_offsets_the_detail_import_and_subtracts_a_shrunk_structure():
    source = scad.build_scad(canvas(), SCAD_LAYERS, **SCAD_HEIGHTS, detail_growth_mm=0.01)
    assert "DETAIL_GROW = 0.01;" in source
    assert 'offset(delta = DETAIL_GROW) import("out.layers/detail.svg", center=false);' in source
    assert source.count("offset(") == 2
    # Detail edges that run along structure edges differ by rounding noise. The shrunk subtrahend
    # moves them inside the taller structure, so no hairline gap closes to a point.
    assert "    offset(delta = -DETAIL_GROW) structure_2d();\n  }\n}" in source
    assert "H = 0.6;" in source and "D = 0.2;" in source  # heights untouched


def test_scad_growth_is_ignored_for_engraved_detail():
    source = scad.build_scad(canvas(), SCAD_LAYERS, **SCAD_HEIGHTS, detail_mode="engraved", detail_growth_mm=0.01)
    assert "DETAIL_GROW" not in source


def test_scad_engraved_recess_shrink_offsets_the_final_recess_inward():
    source = scad.build_scad(canvas(), SCAD_LAYERS, **SCAD_HEIGHTS, detail_mode="engraved", recess_shrink_mm=0.002)
    assert "RECESS_SHRINK = 0.002;" in source
    assert "module detail_2d() {\n  offset(delta = -RECESS_SHRINK) difference() {" in source
    assert source.count("offset(") == 1 and "DETAIL_GROW" not in source
    assert "H = 0.6;" in source and "D = 0.2;" in source  # heights untouched


def test_scad_engraved_recess_shrink_without_structure_offsets_the_clip():
    layers = {k: v for k, v in SCAD_LAYERS.items() if k != "structure"}
    source = scad.build_scad(canvas(), layers, silhouette_thickness_mm=1.2, detail_height_mm=0.2,
                             detail_mode="engraved", recess_shrink_mm=0.002)
    assert "module detail_2d() {\n  offset(delta = -RECESS_SHRINK) intersection() {" in source


def test_scad_recess_shrink_is_ignored_for_raised_detail():
    source = scad.build_scad(canvas(), SCAD_LAYERS, **SCAD_HEIGHTS, recess_shrink_mm=0.002)
    assert "RECESS_SHRINK" not in source and "offset(" not in source


def test_scad_growth_is_ignored_without_a_detail_layer():
    layers = {k: v for k, v in SCAD_LAYERS.items() if k != "detail"}
    source = scad.build_scad(canvas(), layers, silhouette_thickness_mm=1.2, structure_height_mm=0.6,
                             detail_growth_mm=0.01)
    assert "DETAIL_GROW" not in source


# --- Probe solid-count rule ----------------------------------------------------
#
# The S3.6 probe expects one solid per 4-connected mask component of at least filter_speckle²
# pixels. These tests hold that rule against the real tracer at the threshold.

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import vector_map_vtracer as vt  # noqa: E402


def _shapes(n):
    """Four 4-connected shapes of exactly n pixels."""
    side = math.isqrt(n - 1) + 1
    block = np.zeros((side, side), bool)
    block.flat[:n] = True
    line = np.ones((1, n), bool)
    ell = np.zeros((n // 2 + 1, n - n // 2), bool)
    ell[:, 0] = ell[-1, :] = True
    stair = np.zeros((n, n), bool)
    y = x = 0
    for i in range(n):
        stair[y, x] = True
        y, x = (y, x + 1) if i % 2 == 0 else (y + 1, x)
    return {"block": block, "line": line, "L": ell, "stair": stair}


def _padded(material):
    out = np.zeros((material.shape[0] + 8, material.shape[1] + 8), bool)
    out[4:-4, 4:-4] = material
    return out


def _traced_islands(material, speckle):
    svg = vt.trace_layer(material, {"filter_speckle": speckle}).svg
    return sum(1 for ring in cad._rings(cad._parse(svg))[1] if ring.solid and not ring.hole)


def _expected_solids(material, speckle, tmp_path):
    from tests import vector_map_s3_6_cad_probe as probe
    path = tmp_path / "mask.png"
    Image.fromarray(np.where(material, 255, 0).astype(np.uint8), mode="L").save(path)
    return probe._mask_expectation(path, {"mm_per_px": 0.4}, 0, speckle)["solids"]


@pytest.mark.parametrize("speckle", (3, 4, 5, 6))
@pytest.mark.parametrize("offset", (-1, 0, 1))
@pytest.mark.parametrize("shape", ("block", "line", "L", "stair"))
def test_probe_solid_rule_matches_the_tracer_at_the_speckle_threshold(speckle, offset, shape, tmp_path):
    material = _padded(_shapes(speckle * speckle + offset)[shape])
    assert material.sum() == speckle * speckle + offset
    assert _expected_solids(material, speckle, tmp_path) == _traced_islands(material, speckle)


@pytest.mark.parametrize("side, solids", ((3, 0), (4, 2)))
def test_probe_solid_rule_counts_corner_touching_blocks_apart(side, solids, tmp_path):
    material = np.zeros((2 * side, 2 * side), bool)
    material[:side, :side] = material[side:, side:] = True
    material = _padded(material)
    assert _traced_islands(material, 4) == solids
    assert _expected_solids(material, 4, tmp_path) == solids


@pytest.mark.parametrize("speckle", (0, 1, 2))
def test_probe_refuses_a_solid_count_below_speckle_three(speckle, tmp_path):
    # At these values components of up to 4 px can trace to degenerate rings and give no solid.
    # A 1 x 4 line gives none and a 2 x 2 block gives one, so pixel count cannot predict it.
    with pytest.raises(ValueError, match="filter_speckle"):
        _expected_solids(_padded(np.ones((2, 2), bool)), speckle, tmp_path)


# --- Native OpenSCAD gates ----------------------------------------------------
#
# Defect A is gated per layer: each published layer must extrude to a closed mesh by itself.
# Defect B is gated on the assembled source: the real SCAD at the specified heights must render
# without a manifold or closed-mesh warning, with the S3.6 probe's volume and Z checks passing.
# The probe is the single source for what a passing CAD row means.

import json  # noqa: E402
import subprocess  # noqa: E402

from tests import test_vector_map_openscad as open_scad  # noqa: E402

AFFECTED_ROWS = ("corpus_coins", "corpus_gravel", "corpus_pcb", "corpus_brick", "corpus_truchet")
ENGRAVED_ROWS = ("corpus_brick", "corpus_gravel", "corpus_truchet")  # recess met the outline at a point
CAD_WARNINGS = ("not closed", "2-manifold")


def _probe_row(case, tmp_path, *extra):
    from tests import vector_map_s3_6_cad_probe as probe

    work, _ = open_scad.choose_work_dir(tmp_path, open_scad.OPENSCAD)
    evidence = work / "evidence.json"
    probe.main(["--work-dir", str(work / "rows"), "--case", case, "--evidence-json", str(evidence), *extra])
    row = json.loads(evidence.read_text())["cases"][0]
    return row, work / "rows" / case


@open_scad.needs_openscad
@pytest.mark.parametrize("case", AFFECTED_ROWS)
def test_native_assembled_mesh_is_closed_at_the_specified_heights(case, tmp_path):
    row, _ = _probe_row(case, tmp_path)
    assert row["passed"], row["errors"]


@open_scad.needs_openscad
@pytest.mark.parametrize("case", AFFECTED_ROWS)
def test_native_each_published_layer_extrudes_to_a_closed_mesh(case, tmp_path):
    _, directory = _probe_row(case, tmp_path)
    assert open_scad.OPENSCAD
    for svg in sorted((directory / "out.layers").glob("*.svg")):
        source = directory / f"extrude-{svg.stem}.scad"
        # The tiny cube forces the conversion that finds a non-closed mesh. A bare extrude does not.
        source.write_text(f'union() {{ linear_extrude(1) import("out.layers/{svg.name}");'
                          " translate([0, 0, 0.5]) cube(0.1); }\n")
        run = subprocess.run([*open_scad.OPENSCAD, "-o", str(directory / f"extrude-{svg.stem}.stl"), str(source)],
                             capture_output=True, text=True, timeout=300, cwd=directory)
        assert run.returncode == 0, run.stderr
        assert not [line for line in run.stderr.splitlines() if any(w in line for w in CAD_WARNINGS)], svg.name


@open_scad.needs_openscad
@pytest.mark.parametrize("case", ENGRAVED_ROWS)
def test_native_engraved_corpus_mesh_is_closed_with_the_recess_shrink(case, tmp_path):
    row, directory = _probe_row(case, tmp_path, "--detail-mode", "engraved")
    assert row["passed"], row["errors"]
    assert row["relief"]["detail_mode"] == "engraved"
    assert "RECESS_SHRINK = 0.002;" in (directory / "out.scad").read_text()


# --- Engraved recess: known open cases (STABL-ygtexxzr) ------------------------
#
# Both tests are strict xfail. They record defects of the current engraved rule and will fail
# loudly when a repair makes them pass. The repair choice is open.

SLIVER = ROOT / "tests/fixtures/vector_map/cad/shared_run_sliver"


def _render_engraved(tmp_path, layers, canvas_):
    """Write layers, build engraved SCAD with the production recess rule, render it. Return (stderr, stl)."""
    from types import SimpleNamespace
    import vector_map

    work, _ = open_scad.choose_work_dir(tmp_path, open_scad.OPENSCAD)
    assert open_scad.OPENSCAD
    (work / "out.layers").mkdir(parents=True, exist_ok=True)
    for role, text in layers.items():
        (work / "out.layers" / f"{role}.svg").write_text(text)
    settings = SimpleNamespace(export_scad=True, detail_mode="engraved")
    normalized = {role: SimpleNamespace(svg=text) for role, text in layers.items()}
    shrink = vector_map._recess_shrink(settings, normalized, canvas_, [])
    source = scad.build_scad(canvas_, {role: f"out.layers/{role}.svg" for role in layers}, detail_mode="engraved",
                             silhouette_thickness_mm=1.2, detail_height_mm=0.2, recess_shrink_mm=shrink)
    (work / "out.scad").write_text(source)
    run = subprocess.run([*open_scad.OPENSCAD, "-o", str(work / "out.stl"), str(work / "out.scad")],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    return run.stderr, work / "out.stl"


def _cad_warnings(stderr):
    return [line for line in stderr.splitlines() if any(w in line for w in CAD_WARNINGS)]


@open_scad.needs_openscad
@pytest.mark.xfail(strict=True, reason="Shared-run sliver: the isolated-point trigger does not fire. Repair open.")
def test_native_engraved_shared_run_sliver_renders_closed(tmp_path):
    # Two rings cropped from corpus gravel at px (180, 51). The recess shares a boundary run with the
    # outline, so the exact geometry has no isolated contact. The OpenSCAD intersection rounds the next
    # recess vertex off the outline and leaves a hairline sliver that closes to a point.
    layers = {role: (SLIVER / f"{role}.svg").read_text() for role in ("silhouette", "detail")}
    assert cad.recess_outline_contacts(layers["silhouette"], layers["detail"]) == []
    stderr, _ = _render_engraved(tmp_path, layers, Canvas(256, 256, 100.0, 100.0, 0.390625))
    assert not _cad_warnings(stderr)


OPENING_SILHOUETTE = "M2,2 L38,2 L38,20 L2,20 Z "
GROOVE_TO_EDGE = "M10,12 L14,12 L14,20 L10,20 Z "  # opens through the wall at px y = 20
TIP_ON_TOP_EDGE = "M30,2 L26,8 L34,8 Z "  # isolated contact at px (30, 2)


@open_scad.needs_openscad
@pytest.mark.xfail(strict=True, reason="The whole-recess shrink seals edge openings with a skin. Repair open.")
def test_native_engraved_edge_opening_survives_a_recess_repair(tmp_path):
    from tests import vector_map_s3_6_cad_probe as probe

    canvas_ = Canvas(40, 30, 16.0, 12.0, 0.4)
    layers = {"silhouette": layer((OPENING_SILHOUETTE, None), size=40),
              "detail": layer((GROOVE_TO_EDGE + TIP_ON_TOP_EDGE, None), size=40)}
    stderr, stl = _render_engraved(tmp_path, layers, canvas_)
    assert not _cad_warnings(stderr)
    vertices, _ = probe._vertices(stl)
    wall_y, floor_z = 12.0 - 20 * 0.4, 1.2 - 0.2  # mesh y of px row 20, engraving floor
    groove = vertices[(vertices[:, 0] > 10 * 0.4) & (vertices[:, 0] < 14 * 0.4)]
    assert any(abs(y - wall_y) < 1e-6 and abs(z - floor_z) < 1e-6 for _, y, z in groove), "groove floor stops short of the wall"
