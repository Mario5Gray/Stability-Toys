"""SVG sizing for st-vector-map (spec 6.3; STABL-ascsgqha).

Minimal seam: root size in millimetres, the matching pixel viewBox, and the path count.
OpenSCAD imports this form exactly and independent of dpi (S1.3, STABL-nygbbrrn).
S2.5 (STABL-npoznayt) adds full inspection, supported elements, and complexity limits.
Parse with an XML parser only. Never use regular expressions.
"""

import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
_PATH = f"{{{SVG_NS}}}path"

ET.register_namespace("", SVG_NS)


def size_svg(svg, canvas):
    """State the canvas in mm with a px viewBox. Keep every path and transform unchanged."""
    root = ET.fromstring(svg)
    traced = (root.get("width"), root.get("height"))
    if traced != (str(canvas.width_px), str(canvas.height_px)):
        raise ValueError(
            f"The traced SVG is {traced[0]}x{traced[1]} px, "
            f"but the canvas is {canvas.width_px}x{canvas.height_px} px."
        )
    root.set("width", _mm(canvas.width_mm))
    root.set("height", _mm(canvas.height_mm))
    root.set("viewBox", f"0 0 {canvas.width_px} {canvas.height_px}")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def count_paths(svg):
    """Count <path> elements at any depth."""
    return sum(1 for _ in ET.fromstring(svg).iter(_PATH))


def _mm(value):
    """Fixed-point mm with at most 6 decimals, as in the S1.3 OpenSCAD proof."""
    return f"{value:.6f}".rstrip("0").rstrip(".") + "mm"
