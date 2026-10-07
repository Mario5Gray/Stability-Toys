"""S2.5 SVG inspection and normalization contract (STABL-npoznayt, spec 6.3).

Pure XML strings. These tests do not import VTracer.
The accepted dialect is the pinned vtracer==0.6.15 polygon output. Everything else fails.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import xml.etree.ElementTree as ET  # noqa: E402

from vector_map_config import Canvas  # noqa: E402
from vector_map_svg import (  # noqa: E402
    SvgInspectionError,
    SvgLimits,
    inspect_svg,
    normalize_svg,
    size_svg,
)

NS = "http://www.w3.org/2000/svg"
TRIANGLE = "M0,0 L4,0 L0,3 Z "
PATH = f'<path d="{TRIANGLE}" fill="#000000" transform="translate(2,1)"/>'


def doc(body=PATH, *, width="20", height="10", extra="", prolog=""):
    """VTracer 0.6.15 document shape: declaration, Generator comment, svg root, path leaves."""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n{prolog}<!-- Generator: visioncortex VTracer 0.6.12 -->\n'
        f'<svg version="1.1" xmlns="{NS}" width="{width}" height="{height}"{extra}>\n{body}\n</svg>\n'
    )


def path(d=TRIANGLE, attrs=""):
    return f'<path d="{d}"{attrs}/>'


def inspect(svg, **kwargs):
    return inspect_svg(svg, width_px=20, height_px=10, **kwargs)


# --- Accepted documents -------------------------------------------------------


def test_vtracer_document_reports_counts():
    result = inspect(doc(), limits=SvgLimits(max_paths=1))
    assert result.metrics.paths == 1
    assert result.metrics.commands == 4
    assert result.metrics.subpaths == 1
    assert result.metrics.degenerate_subpaths == 0
    assert result.metrics.raw_bytes == len(doc().encode("utf-8"))
    assert result.metrics.normalized_bytes is None


def test_translated_subpaths_use_accumulated_translations():
    body = f'<g transform="translate(1,2)"><path d="{TRIANGLE}" transform="translate(3)"/></g>'
    result = inspect(doc(body))
    assert result.subpaths == (((4.0, 2.0), (8.0, 2.0), (4.0, 5.0)),)


def test_subpaths_keep_document_order():
    body = path("M0,0 L4,0 L0,3 Z M5,5 L6,5 L5,6 Z ") + path("M9,9 L10,9 L9,8 Z ")
    result = inspect(doc(body))
    assert [points[0] for points in result.subpaths] == [(0.0, 0.0), (5.0, 5.0), (9.0, 9.0)]


@pytest.mark.parametrize("fill", ["black", "BLACK", "#000", "#000000", "#000000".upper()])
def test_black_fill_spellings_are_accepted(fill):
    assert inspect(doc(path(attrs=f' fill="{fill}"'))).metrics.paths == 1


def test_group_fill_is_inherited():
    assert inspect(doc(f'<g fill="#000">{path()}</g>')).metrics.paths == 1


def test_nonzero_fill_rule_is_accepted():
    assert inspect(doc(path(attrs=' fill-rule="nonzero"'))).metrics.paths == 1


@pytest.mark.parametrize(
    "extra, width, height",
    [("", "20px", "10px"), (' viewBox="0 0 20 10"', "20", "10"), (' viewBox="0,0,20,10"', "20.0", "10")],
)
def test_pixel_canvas_spellings_are_accepted(extra, width, height):
    assert inspect(doc(extra=extra, width=width, height=height)).metrics.paths == 1


def test_comments_are_ignored():
    assert inspect(doc(f"<!-- a -->{path()}<!-- b -->")).metrics.paths == 1


def test_version_may_be_absent():
    assert inspect(doc().replace(' version="1.1"', "")).metrics.paths == 1


@pytest.mark.parametrize("number", ["+1", "-1", "1.5", ".5", "5.", "1e1", "1E-1", "1e+1"])
def test_decimal_number_forms_are_accepted(number):
    assert inspect(doc(path(f"M{number},0 L4,0 L0,3 Z"))).metrics.paths == 1


# --- Degenerate subpaths ------------------------------------------------------


def test_degenerate_subpaths_next_to_a_polygon_are_kept_and_counted():
    result = inspect(doc(path("M0,0 L4,0 L0,3 Z M1,1 Z M2,2 L3,2 Z ")))
    assert result.metrics.subpaths == 3
    assert result.metrics.degenerate_subpaths == 2
    assert result.metrics.commands == 9


def test_repeated_vertices_count_as_one_distinct_vertex():
    result = inspect(doc(path("M0,0 L4,0 L0,3 Z M1,1 L1,1 L1,1 Z ")))
    assert result.metrics.degenerate_subpaths == 1


def test_degenerate_only_path_fails_with_the_approved_diagnostic():
    body = path() + path("M1,1 Z ") + path("M2,2 L3,2 Z ")
    with pytest.raises(SvgInspectionError) as raised:
        inspect(doc(body))
    message = str(raised.value)
    assert "VTracer returned 2 paths containing only points or lines." in message
    assert "--max-res" in message and "only when processing resolution can increase" in message
    assert "filter_speckle" in message and "on purpose" in message


def test_a_polygon_in_another_path_does_not_rescue_a_degenerate_only_path():
    with pytest.raises(SvgInspectionError, match="only points or lines"):
        inspect(doc(path() + path("M1,1 Z ")))


# --- Rejected documents -------------------------------------------------------

REJECTED = {
    # XML structure
    "malformed": (doc().replace("</svg>", ""), "malformed"),
    "wrong namespace": ('<svg xmlns="urn:other" width="20" height="10"/>', "root"),
    "wrong root": (f'<g xmlns="{NS}"/>', "root"),
    "nested svg": (doc(f'<svg width="20" height="10">{path()}</svg>'), "svg"),
    "dtd": (doc(prolog='<!DOCTYPE svg [<!ENTITY e "1">]>\n'), "DTD"),
    "dtd entity use": (doc(path().replace("M0,0", "M&e;,0"), prolog='<!DOCTYPE svg [<!ENTITY e "0">]>\n'), "DTD"),
    "processing instruction": (doc(prolog="<?foo bar?>\n"), "processing instruction"),
    "root text": (doc("junk" + path()), "text"),
    "element tail": (doc(path() + "junk"), "text"),
    "group text": (doc(f"<g>junk{path()}</g>"), "text"),
    # Elements
    **{
        f"element {name}": (doc(path() + f"<{name}/>"), f"unsupported element {name}")
        for name in ["image", "use", "defs", "script", "mask", "clipPath", "style", "rect", "text", "title"]
    },
    "foreign element": (doc(path() + '<x:y xmlns:x="urn:x"/>'), "unsupported element"),
    "path child": (doc(f'<path d="{TRIANGLE}">{path()}</path>'), "leaf"),
    # Attributes
    **{
        f"attribute {name}": (doc(path(attrs=f' {name}="1"')), "attribute")
        for name in ["style", "opacity", "stroke", "onclick", "clip-path", "mask", "class", "data-x"]
    },
    "href": (doc(path(attrs=' href="#a"')), "attribute"),
    "xlink href": (doc(path(attrs=' xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#a"')), "attribute"),
    "root attribute": (doc(extra=' preserveAspectRatio="none"'), "attribute"),
    "root fill": (doc(extra=' fill="#000"'), "attribute"),
    "version": (doc().replace('version="1.1"', 'version="2.0"'), "version"),
    # Paint
    **{f"fill {fill}": (doc(path(attrs=f' fill="{fill}"')), "fill") for fill in ["none", "red", "#111", "url(#g)", ""]},
    "group fill none": (doc(f'<g fill="none">{path()}</g>'), "fill"),
    "fill rule": (doc(path(attrs=' fill-rule="evenodd"')), "fill-rule"),
    # Path data
    "empty d": (doc(path("")), "empty"),
    "blank d": (doc(path("   ")), "empty"),
    "missing d": (doc('<path fill="#000"/>'), "empty"),
    "only M Z": (doc(path("M0,0 Z ")), "only points or lines"),
    "collinear": (doc(path("M0,0 L1,0 L2,0 Z ")), "collinear"),
    "missing Z": (doc(path("M0,0 L4,0 L0,3")), "closed"),
    "M before Z": (doc(path("M0,0 L4,0 L0,3 M5,5 L6,5 L5,6 Z")), "closed"),
    "unmatched Z": (doc(path("Z M0,0 L4,0 L0,3 Z")), "Z"),
    "L before M": (doc(path("L4,0 L0,3 Z")), "before M"),
    "incomplete pair": (doc(path("M0, L4,0 L0,3 Z")), "coordinate"),
    "missing comma": (doc(path("M0 L4,0 L0,3 Z")), "coordinate"),
    "extra field": (doc(path("M0,0,1 L4,0 L0,3 Z")), "coordinate"),
    "space pair": (doc(path("M0 0 L4,0 L0,3 Z")), "coordinate"),
    "relative": (doc(path("m0,0 l4,0 l0,3 z")), "unsupported path"),
    "curve": (doc(path("M0,0 C1,1 2,2 0,3 Z")), "unsupported path"),
    "horizontal": (doc(path("M0,0 H4 L0,3 Z")), "unsupported path"),
    "trailing junk": (doc(path("M0,0 L4,0 L0,3 Z x")), "unsupported path"),
    "Z suffix": (doc(path("M0,0 L4,0 L0,3 Z1")), "Z"),
    "nbsp separator": (doc(path("M0,0\u00a0L4,0 L0,3 Z")), "coordinate"),
    "tab reference": (doc(path("M0,0&#9;L4,0 L0,3 Z")), "coordinate"),
    # Numbers
    **{
        f"number {number}": (doc(path(f"M{number},0 L4,0 L0,3 Z")), "number")
        for number in ["NaN", "nan", "inf", "Infinity", "1e999", "1_0", "1e", "1e+", "٣", "0x1", ""]
    },
    # Transforms
    **{
        f"transform {value}": (doc(path(attrs=f' transform="{value}"')), "transform")
        for value in [
            "rotate(10)", "translate(1,2) translate(1,1)", "translate(1,2,3)", "matrix(1,0,0,1,0,0)",
            "translate(NaN,0)", "translate()", "translate(1,2", "scale(2)",
        ]
    },
    "group transform": (doc(f'<g transform="scale(2)">{path()}</g>'), "transform"),
    "translated overflow": (doc(path("M1e308,0 L4,0 L0,3 Z", ' transform="translate(1e308,0)"')), "finite"),
    "accumulated overflow": (
        doc(f'<g transform="translate(1e308)"><path d="{TRIANGLE}" transform="translate(1e308)"/></g>'),
        "finite",
    ),
    # Canvas
    "width mismatch": (doc(width="21"), "width"),
    "height mismatch": (doc(height="11"), "height"),
    "height mm": (doc(height="10mm"), "height"),
    "negative width": (doc(width="-20"), "width"),
    "missing width": (doc().replace(' width="20"', ""), "width"),
    "viewBox origin": (doc(extra=' viewBox="1 0 20 10"'), "viewBox"),
    "viewBox extent": (doc(extra=' viewBox="0 0 40 20"'), "viewBox"),
    "viewBox short": (doc(extra=' viewBox="0 0 20"'), "viewBox"),
    "viewBox junk": (doc(extra=' viewBox="0 0 20 ten"'), "viewBox"),
    # Empty output
    "no paths": (doc(""), "empty: it contains no paths"),
}


@pytest.mark.parametrize("svg, match", REJECTED.values(), ids=REJECTED.keys())
def test_incompatible_output_is_rejected(svg, match):
    with pytest.raises(SvgInspectionError, match=match):
        inspect(svg)


def test_inspection_error_is_a_value_error():
    assert issubclass(SvgInspectionError, ValueError)


# --- Empty policy -------------------------------------------------------------


def test_zero_paths_pass_only_when_empty_output_is_allowed():
    result = inspect(doc(""), allow_empty=True)
    assert result.metrics.paths == 0
    assert result.subpaths == ()


def test_empty_path_element_fails_even_when_empty_output_is_allowed():
    with pytest.raises(SvgInspectionError, match="empty"):
        inspect(doc(path("")), allow_empty=True)


# --- Limits -------------------------------------------------------------------


def test_default_limits():
    assert SvgLimits() == SvgLimits(max_svg_bytes=20_971_520, max_paths=10_000, max_path_commands=1_000_000)


@pytest.mark.parametrize("field", ["max_svg_bytes", "max_paths", "max_path_commands"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, "5"])
def test_limits_accept_positive_integers_only(field, value):
    with pytest.raises(ValueError, match=field):
        SvgLimits(**{field: value})


def test_limits_and_metrics_are_immutable():
    result = inspect(doc())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.metrics.paths = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        SvgLimits().max_paths = 2


def test_byte_limit_passes_at_the_limit_and_fails_one_over():
    svg = doc()
    size = len(svg.encode("utf-8"))
    assert inspect(svg, limits=SvgLimits(max_svg_bytes=size)).metrics.raw_bytes == size
    with pytest.raises(SvgInspectionError, match=f"{size} bytes exceeds limit {size - 1}"):
        inspect(svg, limits=SvgLimits(max_svg_bytes=size - 1))


def test_byte_limit_counts_utf8_bytes():
    svg = doc(f"<!-- é -->{path()}")
    assert len(svg.encode("utf-8")) == len(svg) + 1
    with pytest.raises(SvgInspectionError, match="bytes exceeds"):
        inspect(svg, limits=SvgLimits(max_svg_bytes=len(svg)))


def test_byte_limit_is_checked_before_parsing():
    malformed = "<svg" + " " * 100
    with pytest.raises(SvgInspectionError, match="bytes exceeds") as raised:
        inspect(malformed, limits=SvgLimits(max_svg_bytes=10))
    assert "malformed" not in str(raised.value)


def test_path_limit_passes_at_the_limit_and_reports_a_lower_bound():
    body = path() * 3
    assert inspect(doc(body), limits=SvgLimits(max_paths=3)).metrics.paths == 3
    with pytest.raises(SvgInspectionError, match="path count at least 3 exceeds limit 2"):
        inspect(doc(body), limits=SvgLimits(max_paths=2))


def test_paths_share_one_command_budget():
    body = path() * 2
    assert inspect(doc(body), limits=SvgLimits(max_path_commands=8)).metrics.commands == 8
    with pytest.raises(SvgInspectionError, match="command count at least 8 exceeds limit 7"):
        inspect(doc(body), limits=SvgLimits(max_path_commands=7))


def test_degenerate_commands_consume_the_command_budget():
    body = path("M0,0 L4,0 L0,3 Z M1,1 Z ")
    assert inspect(doc(body), limits=SvgLimits(max_path_commands=6)).metrics.commands == 6
    with pytest.raises(SvgInspectionError, match="command count"):
        inspect(doc(body), limits=SvgLimits(max_path_commands=5))


@pytest.mark.parametrize("limits", [SvgLimits(max_paths=1), SvgLimits(max_path_commands=1)])
def test_limit_failures_suggest_upstream_settings_without_changing_them(limits):
    with pytest.raises(SvgInspectionError) as raised:
        inspect(doc(path() * 2), limits=limits)
    message = str(raised.value)
    assert "--max-res" in message and "filter_speckle" in message and "0..128" in message
    assert "can change the geometry" in message


# --- Normalization ------------------------------------------------------------

CANVAS = Canvas(width_px=20, height_px=10, width_mm=10.0, height_mm=5.0, mm_per_px=0.5)
NESTED = (
    f'<g transform="translate(3,1)" id="outer"><path d="{TRIANGLE}" transform="translate(2,5)"/>'
    f'<g fill="#000"><path d="M1,1 L5,1 L1,4 Z M2,2 Z " fill-rule="nonzero"/></g></g>'
    f'<path d="M9,9 L10,9 L9,8 Z " fill="black"/>'
)


def canvas(width_px=20, height_px=10, mm_per_px=0.5, width_mm=None, height_mm=None):
    return Canvas(
        width_px,
        height_px,
        width_px * mm_per_px if width_mm is None else width_mm,
        height_px * mm_per_px if height_mm is None else height_mm,
        mm_per_px,
    )


def elements(svg):
    """Every element below the root, in document order, with its attributes."""
    root = ET.fromstring(svg)
    return [(element.tag, dict(element.attrib)) for element in root.iter() if element is not root]


def test_normalized_root_states_mm_and_the_pixel_view_box():
    root = ET.fromstring(normalize_svg(doc(), CANVAS).svg)
    assert (root.get("width"), root.get("height"), root.get("viewBox")) == ("10mm", "5mm", "0 0 20 10")


def test_normalization_changes_no_geometry_paint_transform_or_order():
    raw = doc(NESTED)
    assert elements(normalize_svg(raw, CANVAS).svg) == elements(raw)


def test_nested_translations_survive_with_asymmetric_origins():
    out = ET.fromstring(normalize_svg(doc(NESTED), CANVAS).svg)
    transforms = [element.get("transform") for element in out.iter() if element.get("transform")]
    assert transforms == ["translate(3,1)", "translate(2,5)"]


def test_raw_text_is_not_modified():
    raw = doc(NESTED)
    copy = str(raw)
    normalize_svg(raw, CANVAS)
    assert raw == copy


def test_matching_raw_view_box_is_accepted_and_kept_in_pixels():
    out = normalize_svg(doc(extra=' viewBox="0 0 20 10"'), CANVAS).svg
    assert ET.fromstring(out).get("viewBox") == "0 0 20 10"


def test_mismatched_raw_view_box_is_rejected_not_replaced():
    with pytest.raises(SvgInspectionError, match="viewBox"):
        normalize_svg(doc(extra=' viewBox="0 0 40 20"'), CANVAS)


@pytest.mark.parametrize("width, height", [("21", "10"), ("20", "11"), ("20mm", "10"), ("20", "10in")])
def test_raw_size_must_be_the_canvas_in_pixels(width, height):
    with pytest.raises(SvgInspectionError, match="canvas|width|height"):
        normalize_svg(doc(width=width, height=height), CANVAS)


def test_output_is_well_formed_with_the_default_namespace():
    out = normalize_svg(doc(), CANVAS).svg
    assert out.startswith("<?xml")
    assert "ns0:" not in out
    assert ET.fromstring(out).tag == f"{{{NS}}}svg"


@pytest.mark.parametrize(
    "mm_per_px, width, height",
    [
        (5.0, "100mm", "50mm"),
        (0.25, "5mm", "2.5mm"),
        (1e-8, "0.0000002mm", "0.0000001mm"),
        (0.1, "2mm", "1mm"),
        (1e18, "20000000000000000000mm", "10000000000000000000mm"),
    ],
)
def test_mm_lengths_are_plain_decimals(mm_per_px, width, height):
    root = ET.fromstring(normalize_svg(doc(), canvas(mm_per_px=mm_per_px)).svg)
    assert (root.get("width"), root.get("height")) == (width, height)


def test_small_positive_size_never_becomes_zero():
    root = ET.fromstring(normalize_svg(doc(), canvas(mm_per_px=1e-9)).svg)
    assert root.get("width") != "0mm"
    assert float(root.get("width")[:-2]) > 0


def test_mm_lengths_keep_the_full_float_value():
    value = 1 / 3
    root = ET.fromstring(normalize_svg(doc(), canvas(mm_per_px=value)).svg)
    assert float(root.get("width")[:-2]) == 20 * value
    assert "e" not in root.get("width").lower()


@pytest.mark.parametrize(
    "bad",
    [
        canvas(mm_per_px=float("nan")),
        canvas(mm_per_px=float("inf")),
        canvas(mm_per_px=0.0),
        canvas(mm_per_px=-0.5),
        canvas(width_mm=float("inf")),
        canvas(height_mm=0.0),
    ],
    ids=["nan scale", "inf scale", "zero scale", "negative scale", "inf width", "zero height"],
)
def test_canvas_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match="positive and finite"):
        normalize_svg(doc(), bad)


@pytest.mark.parametrize("bad", [canvas(width_mm=10.001), canvas(height_mm=4.999)], ids=["width", "height"])
def test_canvas_scale_must_be_uniform(bad):
    with pytest.raises(ValueError, match="uniform"):
        normalize_svg(doc(), bad)


def test_normalized_byte_limit_is_checked_after_serialization():
    raw = doc().replace("<!-- Generator: visioncortex VTracer 0.6.12 -->\n", "")
    wide = canvas(mm_per_px=0.123456789012345)
    normalized = len(normalize_svg(raw, wide).svg.encode("utf-8"))
    assert normalized > len(raw.encode("utf-8"))
    with pytest.raises(SvgInspectionError, match=f"Normalized SVG size {normalized} bytes exceeds limit {normalized - 1}"):
        normalize_svg(raw, wide, limits=SvgLimits(max_svg_bytes=normalized - 1))


def test_normalized_metrics_report_both_sizes_and_counts():
    raw = doc(NESTED)
    result = normalize_svg(raw, CANVAS)
    assert result.metrics.raw_bytes == len(raw.encode("utf-8"))
    assert result.metrics.normalized_bytes == len(result.svg.encode("utf-8"))
    assert (result.metrics.paths, result.metrics.commands, result.metrics.subpaths) == (3, 14, 4)
    assert result.metrics.degenerate_subpaths == 1


def test_degenerate_subpaths_survive_normalization_unchanged():
    raw = doc(path("M0,0 L4,0 L0,3 Z M1,1 Z M2,2 L3,2 Z "))
    out = normalize_svg(raw, CANVAS).svg
    assert [d for _, attrib in elements(out) if (d := attrib.get("d"))] == ["M0,0 L4,0 L0,3 Z M1,1 Z M2,2 L3,2 Z "]


def test_empty_output_is_rejected():
    with pytest.raises(SvgInspectionError, match="empty"):
        normalize_svg(doc(""), CANVAS)


def test_normalization_limits_apply_to_the_raw_inspection():
    with pytest.raises(SvgInspectionError, match="path count at least 2"):
        normalize_svg(doc(path() * 2), CANVAS, limits=SvgLimits(max_paths=1))


def test_size_svg_is_the_normalizer():
    assert size_svg(doc(NESTED), CANVAS) == normalize_svg(doc(NESTED), CANVAS).svg
