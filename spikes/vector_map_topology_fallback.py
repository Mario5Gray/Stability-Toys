"""STABL-wuzmvgxm research spike: topology-preserving fallback for minimum-feature-size masks.

Throwaway research code. Nothing in scripts/ imports it. No production change, no dependency change.
Uses the locked S1.4 rules in tests/fixtures/vector_map/corpus/metrics.py unchanged.
Run with the dedicated Python 3.12 interpreter (VTracer 0.6.15 crashes on 3.14):
    conda activate stability-toys && python spikes/vector_map_topology_fallback.py
Writes docs/superpowers/reports/2026-10-17-vector-map-topology-fallback/results.json.

Methods, each scored on all eight reviewed corpus masks at their processing size:
  vtracer        Current production trace, filter_speckle 4 (S1.4 default).
  up{k}_s{s}     Nearest-neighbour upscale by k, VTracer at speckle s, vectors scaled back by viewBox.
  pixel_edge     Exact pixel-boundary (crack) tracer. Topology-exact by construction.
  router_width   vtracer when both class minimum widths are >= 4 px, otherwise pixel_edge.
  router_verify  vtracer when it passes S1.4 against its own mask, otherwise pixel_edge.
"""

import hashlib
import json
import platform
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

import vector_map_vtracer as adapter  # noqa: E402
from tests.fixtures.vector_map.corpus import make_corpus, metrics  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "vector_map" / "corpus"
OUT = ROOT / "docs" / "superpowers" / "reports" / "2026-10-17-vector-map-topology-fallback" / "results.json"
LOCKED_METRICS_SHA256 = "7050d8cc4938a2fc75103c735c64430b275a32d79b74488861e4c8f1ac672898"
HARD = ("bracket", "brick", "gravel", "horse")
UPSCALES = ((2, 4), (2, 16), (4, 4), (4, 64))  # (k, speckle). Speckle k*k*4 keeps the same area in source pixels.
MAX_WIDTH = 16


def _timed(call):
    """One untimed warm-up, then the locked number of timed calls. Return (last value, median ms)."""
    for _ in range(metrics.TIMING["warmup_calls"]):
        call()
    times = []
    for _ in range(metrics.TIMING["timed_calls"]):
        start = time.perf_counter()
        value = call()
        times.append((time.perf_counter() - start) * 1000)
    return value, statistics.median(times)


# --- Local width ------------------------------------------------------------------


def local_width(cls, outside):
    """Per-pixel side of the largest axis-aligned square of the class that covers the pixel, capped.

    Generalizes the production 4x4 square-coverage diagnostic. outside is the value of
    off-canvas pixels: False for material, True for background (thin exterior margins are not gaps).
    """
    pad = MAX_WIDTH
    padded = np.pad(cls, pad, constant_values=outside)
    width = np.zeros(padded.shape, np.int32)
    for k in range(1, MAX_WIDTH + 1):
        covered = ndimage.binary_dilation(ndimage.binary_erosion(padded, np.ones((k, k), bool)), np.ones((k, k), bool))
        width[covered & padded] = k
    return width[pad:-pad, pad:-pad]


def _check_local_width():
    bar = np.zeros((12, 12), bool)
    bar[2:10, 4:7] = True
    width = local_width(bar, False)
    assert width[bar].min() == 3 and width[bar].max() == 3, width[bar]


def skeleton_width(cls):
    """Minimum inscribed diameter on the interior medial axis (skeleton pixels with two or more neighbours)."""
    from skimage.morphology import skeletonize

    skeleton = skeletonize(cls)
    neighbours = ndimage.convolve(skeleton.astype(int), np.ones((3, 3), int), mode="constant") - 1
    interior = skeleton & (neighbours >= 2)
    if not interior.any():
        return None
    return float(2 * ndimage.distance_transform_edt(np.pad(cls, 1))[1:-1, 1:-1][interior].min())


def widths(material):
    fg, bg = local_width(material, False), local_width(~material, True)
    return {
        "material_min_px": int(fg[material].min()),
        "background_min_px": int(bg[~material].min()),
        "material_below_4_px": int(np.count_nonzero(material & (fg < 4))),
        "background_below_4_px": int(np.count_nonzero(~material & (bg < 4))),
        "material_skeleton_min_px": skeleton_width(material),
        "background_skeleton_min_px": skeleton_width(~material),
    }


# --- Tracers ----------------------------------------------------------------------


def trace_vtracer(material, speckle=metrics.CURRENT_DEFAULT):
    return adapter.trace_layer(material, {"filter_speckle": speckle}).svg


def trace_upscaled(material, k, speckle):
    """Trace a k-times nearest-neighbour upscale. The viewBox maps it back onto the source pixel grid."""
    height, width = material.shape
    svg = adapter.trace_layer(np.kron(material, np.ones((k, k), bool)), {"filter_speckle": speckle}).svg
    root = f'width="{width * k}" height="{height * k}"'
    assert root in svg, svg[:300]
    return svg.replace(root, f'width="{width}" height="{height}" viewBox="0 0 {width * k} {height * k}"', 1)


# Directed boundary edges, clockwise on screen (y down) around each material pixel at (row, col).
_EDGES = (  # (neighbour drow, dcol), start corner (dx, dy), end corner (dx, dy)
    ((-1, 0), (0, 0), (1, 0)),  # top
    ((0, 1), (1, 0), (1, 1)),  # right
    ((1, 0), (1, 1), (0, 1)),  # bottom
    ((0, -1), (0, 1), (0, 0)),  # left
)


def trace_pixel_edge(material):
    """Trace the exact pixel boundary. One path. Holes wind opposite to their outline, so nonzero fill is exact."""
    height, width = material.shape
    padded = np.pad(material, 1)
    outgoing = {}
    for (dr, dc), (sx, sy), (ex, ey) in _EDGES:
        neighbour = padded[1 + dr:1 + dr + height, 1 + dc:1 + dc + width]
        for row, col in zip(*np.nonzero(material & ~neighbour)):
            outgoing.setdefault((col + sx, row + sy), []).append((col + ex, row + ey))
    loops = []
    while outgoing:
        start = min(outgoing)
        loop, point, heading = [start], start, None
        while True:
            options = outgoing[point]
            if heading is not None and len(options) > 1:
                # Saddle: turn right (screen), which keeps diagonal material pixels in separate loops.
                options.sort(key=lambda q: _turn(heading, (q[0] - point[0], q[1] - point[1])))
            nxt = options.pop(0)
            if not options:
                del outgoing[point]
            heading, point = (nxt[0] - point[0], nxt[1] - point[1]), nxt
            if point == start:
                break
            loop.append(point)
        loops.append(_corners(loop))
    d = " ".join(" ".join([f"M{loop[0][0]},{loop[0][1]}", *(f"L{x},{y}" for x, y in loop[1:]), "Z"]) for loop in loops)
    return (f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<path d="{d}" fill="#000000"/></svg>')


def _turn(heading, step):
    """0 for a right turn, 1 straight, 2 left (screen coordinates, y down)."""
    cross = heading[0] * step[1] - heading[1] * step[0]
    return 0 if cross > 0 else (1 if cross == 0 else 2)


def _corners(loop):
    """Drop vertices where the boundary does not change direction."""
    n = len(loop)
    keep = []
    for i in range(n):
        (ax, ay), (bx, by), (cx, cy) = loop[i - 1], loop[i], loop[(i + 1) % n]
        if (bx - ax) * (cy - by) - (by - ay) * (cx - bx) != 0:
            keep.append(loop[i])
    return keep


# --- Scoring ----------------------------------------------------------------------


def score(material, trace):
    height, width = material.shape
    svg, trace_ms = _timed(lambda: trace(material))
    rendered = metrics.render(svg, width, height)
    area = metrics.polygon_area(svg)
    scale = _viewbox_scale(svg, width)
    result = metrics.evaluate(material, rendered, area / scale ** 2)
    result.update(metrics.svg_complexity(svg))
    result.update(trace_median_ms=round(trace_ms, 3), svg_sha256=hashlib.sha256(svg.encode()).hexdigest())
    return result


def _viewbox_scale(svg, width):
    marker = 'viewBox="0 0 '
    if marker not in svg:
        return 1
    return int(svg.split(marker, 1)[1].split()[0]) // width


SUMMARY = ("passed", "failures", "topology_mask", "topology_render", "xor_per_perimeter_px",
           "boundary_max_distance_px", "centroid_shift_px", "area_dev_abs_per_perimeter_px",
           "paths", "commands", "svg_bytes", "trace_median_ms")


def main():
    actual = hashlib.sha256((CORPUS / "metrics.py").read_bytes()).hexdigest()
    if actual != LOCKED_METRICS_SHA256:
        raise SystemExit(f"metrics.py changed since S1.4 ({actual}). Refuse to measure.")
    _check_local_width()
    masks = {case: np.asarray(Image.open(CORPUS / case / "mask.png")) == 255 for case in sorted(make_corpus.CASES)}
    methods = {"vtracer": trace_vtracer, "pixel_edge": trace_pixel_edge}
    for k, speckle in UPSCALES:
        methods[f"up{k}_s{speckle}"] = lambda m, k=k, s=speckle: trace_upscaled(m, k, s)
    cases = {}
    for case, material in masks.items():
        row = {"shape": list(material.shape), "hard": case in HARD, "widths": widths(material), "methods": {}}
        for name, trace in methods.items():
            measured = score(material, trace)
            row["methods"][name] = {key: measured[key] for key in SUMMARY}
            print(f"{case:<8} {name:<11} {'PASS' if measured['passed'] else 'FAIL ' + ','.join(measured['failures'])}"
                  f"  commands={measured['commands']} bytes={measured['svg_bytes']} ms={measured['trace_median_ms']}",
                  flush=True)
        w = row["widths"]
        thin = w["material_min_px"] < 4 or w["background_min_px"] < 4
        verify = row["methods"]["vtracer"]["passed"]
        row["router_width"] = "pixel_edge" if thin else "vtracer"
        row["router_verify"] = "vtracer" if verify else "pixel_edge"
        cases[case] = row
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "issue": "STABL-wuzmvgxm",
        "metrics_sha256": actual,
        "tolerances": metrics.TOLERANCES,
        "timing": metrics.TIMING,
        "speckle_default": metrics.CURRENT_DEFAULT,
        "upscales": [list(u) for u in UPSCALES],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            **{name: version(name) for name in ("vtracer", "resvg-py", "numpy", "scipy", "scikit-image", "pillow")},
        },
        "cases": cases,
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
