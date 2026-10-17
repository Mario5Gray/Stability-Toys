"""Near-limit warnings use accepted per-layer SVG measurements. STABL-vpodjirg."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map_diagnostics as diagnostics  # noqa: E402
from vector_map_svg import SvgInspectionError, SvgLimits, SvgMetrics, inspect_svg  # noqa: E402


def metrics(*, paths=1, commands=4, raw_bytes=100, normalized_bytes=100):
    return SvgMetrics(raw_bytes=raw_bytes, normalized_bytes=normalized_bytes,
                      paths=paths, commands=commands, vertices=3,
                      subpaths=1, degenerate_subpaths=0)


@pytest.mark.parametrize("paths, warned", [(899, False), (900, True), (1000, True), (1001, False)])
def test_path_warning_starts_at_exactly_ninety_percent(paths, warned):
    warnings = diagnostics.limit_near_miss(
        "silhouette", metrics(paths=paths), SvgLimits(max_paths=1000))
    assert [item["metric"] for item in warnings] == (["paths"] if warned else [])
    if warned:
        assert warnings[0] == {
            "level": "warning", "code": "limit_near_miss", "role": "silhouette",
            "metric": "paths", "value": paths, "limit": 1000,
            "message": (f"silhouette: measured paths = {paths}. This is near the per-layer limit 1000 (--max-paths). "
                        "Reduce layer complexity or raise the limit."),
        }


def test_threshold_uses_integer_comparison_for_large_limits():
    limit = 10**18 + 1
    first_warning = (9 * limit + 9) // 10
    policy = SvgLimits(max_paths=limit)
    assert diagnostics.limit_near_miss("detail", metrics(paths=first_warning - 1), policy) == ()
    assert diagnostics.limit_near_miss("detail", metrics(paths=first_warning), policy)[0]["value"] == first_warning


@pytest.mark.parametrize("raw_bytes, normalized_bytes", [(900, 200), (200, 900)])
def test_byte_warning_uses_larger_raw_or_published_size(raw_bytes, normalized_bytes):
    warnings = diagnostics.limit_near_miss(
        "structure", metrics(raw_bytes=raw_bytes, normalized_bytes=normalized_bytes),
        SvgLimits(max_svg_bytes=1000))
    assert [(item["metric"], item["value"], item["limit"]) for item in warnings] == [
        ("svg_bytes", 900, 1000)]


def test_one_warning_per_metric_in_fixed_order():
    warnings = diagnostics.limit_near_miss(
        "detail", metrics(paths=9, commands=9, raw_bytes=9, normalized_bytes=9),
        SvgLimits(max_paths=10, max_path_commands=10, max_svg_bytes=10))
    assert [item["metric"] for item in warnings] == ["paths", "commands", "svg_bytes"]
    assert {item["role"] for item in warnings} == {"detail"}
    assert [flag in item["message"] for item, flag in zip(
        warnings, ("--max-paths", "--max-path-commands", "--max-svg-bytes"))] == [True, True, True]


def test_svg_inspector_rejects_an_over_limit_layer():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">'
           '<path d="M0,0 L4,0 L0,3 Z"/><path d="M5,5 L9,5 L5,8 Z"/></svg>')
    with pytest.raises(SvgInspectionError, match="path count at least 2 exceeds limit 1"):
        inspect_svg(svg, width_px=20, height_px=10, limits=SvgLimits(max_paths=1))
