"""Operator preview for st-vector-map bundles. STABL-kfrksmnp, spec 6.4.

Renderer: resvg-py==0.5.0 (Q2, STABL-dheskftn). Import it only when a preview is requested.

resvg-py 0.5.0 cannot size a millimetre root without a dpi. With dpi = 25.4 / mm_per_px it
converts mm to px in float32. At 25.4/96 mm/px, 67.7333 mm becomes 255.99998 px, and edges
move (S2.7 measurement). S2.5 measured that dpi only at 0.5 mm/px, where float32 is exact.
Render the published paths under a pixel root instead: width and height equal the viewBox.
"""

import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
from PIL import Image

BACKGROUND = "#ffffff"
# Geometry checks only. Operator previews keep the renderer default anti-aliasing.
CRISP = "crisp_edges"
MATERIAL_LUMINANCE = 128

ET.register_namespace("", "http://www.w3.org/2000/svg")


def pixel_root(svg, canvas):
    """Restate normalized SVG with a pixel root size. Keep viewBox, paths, paint and order."""
    root = ET.fromstring(svg)
    root.set("width", str(canvas.width_px))
    root.set("height", str(canvas.height_px))
    return ET.tostring(root, encoding="unicode")


def render_luminance(svg, canvas, *, shape_rendering=None):
    """Render normalized SVG at processing size. Return 8-bit luminance, black material on white.

    One viewBox unit is one processing pixel. No dpi takes part.
    """
    import resvg_py

    options = {"shape_rendering": shape_rendering} if shape_rendering is not None else {}
    png = bytes(resvg_py.svg_to_bytes(
        svg_string=pixel_root(svg, canvas),
        width=canvas.width_px,
        height=canvas.height_px,
        background=BACKGROUND,
        skip_system_fonts=True,
        **options,
    ))
    with Image.open(BytesIO(png)) as image:
        return np.asarray(image.convert("L")).copy()


def render_material(svg, canvas):
    """Crisp geometry render. True where the vector output has material."""
    return render_luminance(svg, canvas, shape_rendering=CRISP) < MATERIAL_LUMINANCE
