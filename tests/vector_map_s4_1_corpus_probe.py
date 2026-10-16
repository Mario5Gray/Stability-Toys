"""Blocking S4.1 corpus acceptance (STABL-etpzsxvn). Run explicitly with a real OpenSCAD executable.

Plan: docs/superpowers/plans/2026-10-15-vector-map-s4-1.md.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import vector_map  # noqa: E402
from tests.fixtures.vector_map.corpus import metrics  # noqa: E402

CORPUS = ROOT / "tests/fixtures/vector_map/corpus"

ET.register_namespace("", "http://www.w3.org/2000/svg")


def pixel_root(svg: str) -> tuple[str, int, int]:
    """Return an in-memory copy whose root width and height are the viewBox pixel size.

    The published SVG keeps a millimetre root, which metrics.render() rejects. Only the root
    width and height change. The viewBox, paths, groups and transforms stay as published.
    """
    root = ET.fromstring(svg)
    box = root.get("viewBox")
    if not box:
        raise ValueError("Published SVG has no viewBox. The pixel size is unknown.")
    x, y, width, height = (float(value) for value in box.replace(",", " ").split())
    if x or y or width != int(width) or height != int(height):
        raise ValueError(f"Published SVG viewBox is not an integer pixel canvas at the origin: {box!r}")
    root.set("width", str(int(width)))
    root.set("height", str(int(height)))
    return ET.tostring(root, encoding="unicode"), int(width), int(height)


def render_published(svg: str) -> np.ndarray:
    """Rasterize a published SVG with the unchanged locked renderer, one viewBox unit per pixel."""
    copy, width, height = pixel_root(svg)
    return metrics.render(copy, width, height)


def measure_fidelity(svg: str, mask: np.ndarray) -> dict:
    """Run the locked S1.4 evaluation on a published silhouette against its reviewed mask."""
    try:
        rendered = render_published(svg)
    except Exception as exc:
        return {"passed": False, "failures": ["render"], "checks": {"render": False},
                "error": f"{type(exc).__name__}: {exc}"}
    return metrics.evaluate(mask, rendered, metrics.polygon_area(svg))
