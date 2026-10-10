"""Operator preview for st-vector-map bundles. STABL-kfrksmnp, spec 6.4.

Renderer: resvg-py==0.5.0 (Q2, STABL-dheskftn). Import it only when a preview is requested.

resvg-py 0.5.0 cannot size a millimetre root without a dpi. With dpi = 25.4 / mm_per_px it
converts mm to px in float32. At 25.4/96 mm/px, 67.7333 mm becomes 255.99998 px, and edges
move (S2.7 measurement). S2.5 measured that dpi only at 0.5 mm/px, where float32 is exact.
Render the published paths under a pixel root instead: width and height equal the viewBox.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

BACKGROUND = "#ffffff"
# Geometry checks only. Operator previews keep the renderer default anti-aliasing.
CRISP = "crisp_edges"
MATERIAL_LUMINANCE = 128

ET.register_namespace("", "http://www.w3.org/2000/svg")

# Fixed text, layout and colours. Labels use the Pillow embedded default font only.
LABELS = {
    "mask": "Prepared material mask",
    "vector": "Rendered vector output",
    "overlay": "Vector overlay on source",
}
# Okabe-Ito colours. Overlap and difference stay distinct for colour-blind operators.
LEGEND = {
    "both": (0, 158, 115),
    "vector_only": (213, 94, 0),
    "mask_only": (0, 114, 178),
}
LEGEND_TEXT = {"both": "Mask and vector", "vector_only": "Vector only", "mask_only": "Mask only"}
OVERLAY_ALPHA = 0.65
LABEL_SIZE = 12
LABEL_HEIGHT = 18
LEGEND_HEIGHT = 18
MARGIN = 8
SWATCH = 12
PAGE = (232, 232, 232)
TEXT = (0, 0, 0)


@dataclass(frozen=True)
class Preview:
    png: bytes
    panels: dict
    provenance: dict


def require_renderer():
    """Fail before processing when --preview cannot render."""
    try:
        import resvg_py  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("--preview needs resvg-py==0.5.0. Install the vector extra.") from exc


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


def label_font():
    """Embedded Aileron at LABEL_SIZE with FreeType on Pillow >= 10.1. Else embedded bitmap font.

    A size argument forces FreeType loading, so pass it only when FreeType exists.
    Pillow 10.0 load_default() takes no size. The pyproject floor is Pillow>=10.0.
    """
    from PIL import features

    if features.check("freetype2"):
        try:
            return ImageFont.load_default(LABEL_SIZE)
        except TypeError:
            pass
    return ImageFont.load_default()


def blend_base(source):
    """Lighten source luminance by half, so overlay colours stay visible."""
    return (np.asarray(source, np.uint16) + 255) // 2


def _source(data, prepared):
    """Orient like preparation, then resize to the processing canvas like the material."""
    with Image.open(BytesIO(data)) as image:
        with ImageOps.exif_transpose(image) as oriented:
            if oriented.size != tuple(prepared.oriented_size):
                raise ValueError(f"Preview source size {oriented.size} differs from prepared {prepared.oriented_size}.")
            with oriented.convert("L") as gray:
                with gray.resize(tuple(prepared.processed_size), resample=Image.Resampling.NEAREST) as resized:
                    return np.asarray(resized).copy()


def _overlay(source, material, luminance):
    coverage = (255 - luminance.astype(float)) / 255
    mask = material.astype(float)
    weights = {
        "both": mask * coverage,
        "vector_only": (1 - mask) * coverage,
        "mask_only": mask * (1 - coverage),
    }
    total = sum(weights.values())
    out = np.repeat(blend_base(source).astype(float)[..., None] * (1 - OVERLAY_ALPHA * total)[..., None], 3, axis=2)
    for name, weight in weights.items():
        out += OVERLAY_ALPHA * weight[..., None] * np.array(LEGEND[name], float)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _gray_rgb(values):
    return np.repeat(values[..., None], 3, axis=2)


def compose(prepared, svg, source_bytes):
    """Draw mask, anti-aliased vector render and source overlay at processing resolution.

    The vector panel renders the published SVG. A mask-only preview is not vector proof.
    """
    width, height = prepared.processed_size
    luminance = render_luminance(svg, prepared.canvas)
    panels = {
        "mask": _gray_rgb(np.where(prepared.material, 0, 255).astype(np.uint8)),
        "vector": _gray_rgb(luminance),
        "overlay": _overlay(_source(source_bytes, prepared), prepared.material, luminance),
    }
    font = label_font()
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    column = max(width, *(math.ceil(measure.textlength(text, font=font)) for text in LABELS.values()))
    entries = [
        (name, SWATCH + 4 + math.ceil(measure.textlength(LEGEND_TEXT[name], font=font)) + 12) for name in LEGEND
    ]
    legend_width = sum(size for _, size in entries)
    page_width = max(MARGIN + 3 * (column + MARGIN), legend_width + 2 * MARGIN)
    top = MARGIN + LABEL_HEIGHT
    legend_top = top + height + MARGIN
    page = Image.new("RGB", (page_width, legend_top + LEGEND_HEIGHT + MARGIN), PAGE)
    draw = ImageDraw.Draw(page)
    boxes = {}
    for index, (name, pixels) in enumerate(panels.items()):
        column_left = MARGIN + index * (column + MARGIN)
        left = column_left + (column - width) // 2
        page.paste(Image.fromarray(pixels), (left, top))
        boxes[name] = (left, top, left + width, top + height)
        draw.text((column_left, MARGIN), LABELS[name], font=font, fill=TEXT)
    x = MARGIN
    for name, size in entries:
        swatch_top = legend_top + (LEGEND_HEIGHT - SWATCH) // 2
        draw.rectangle((x, swatch_top, x + SWATCH - 1, swatch_top + SWATCH - 1), fill=LEGEND[name])
        draw.text((x + SWATCH + 4, legend_top + 2), LEGEND_TEXT[name], font=font, fill=TEXT)
        x += size
    boxes["legend"] = (MARGIN, legend_top, MARGIN + legend_width, legend_top + LEGEND_HEIGHT)
    buffer = BytesIO()
    page.save(buffer, format="PNG")
    return Preview(buffer.getvalue(), boxes, provenance(prepared, font))


# S3.2 candidate masks (STABL-uifsadne). Prepared masks only. S3.3 adds rendered SVG panels.
CANDIDATE_LABELS = {
    "silhouette": "Silhouette mask",
    "structure": "Structure candidate mask",
    "detail": "Detail candidate mask",
}
CANDIDATE_NOTE = "Prepared masks only. No vector render."


def compose_candidates(prepared):
    """Draw each selected prepared mask in role order. No panel claims vector render proof."""
    width, height = prepared.silhouette.processed_size
    masks = {"silhouette": prepared.silhouette.material}
    masks.update({role: getattr(prepared, role).material for role in prepared.roles[1:]})
    labels = {role: CANDIDATE_LABELS[role] + ("" if role == "silhouette" or material.any() else " (empty)")
              for role, material in masks.items()}
    font = label_font()
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    column = max(width, *(math.ceil(measure.textlength(text, font=font)) for text in labels.values()))
    note_width = math.ceil(measure.textlength(CANDIDATE_NOTE, font=font))
    page_width = max(MARGIN + len(masks) * (column + MARGIN), note_width + 2 * MARGIN)
    top = MARGIN + LABEL_HEIGHT
    note_top = top + height + MARGIN
    page = Image.new("RGB", (page_width, note_top + LABEL_HEIGHT + MARGIN), PAGE)
    draw = ImageDraw.Draw(page)
    boxes = {}
    for index, (role, material) in enumerate(masks.items()):
        column_left = MARGIN + index * (column + MARGIN)
        left = column_left + (column - width) // 2
        page.paste(Image.fromarray(_gray_rgb(np.where(material, 0, 255).astype(np.uint8))), (left, top))
        boxes[role] = (left, top, left + width, top + height)
        draw.text((column_left, MARGIN), labels[role], font=font, fill=TEXT)
    draw.text((MARGIN, note_top + 2), CANDIDATE_NOTE, font=font, fill=TEXT)
    boxes["note"] = (MARGIN, note_top, MARGIN + note_width, note_top + LABEL_HEIGHT)
    buffer = BytesIO()
    page.save(buffer, format="PNG")
    return Preview(buffer.getvalue(), boxes, candidate_provenance(prepared.silhouette, font, list(labels.values())))


def candidate_provenance(silhouette, font, labels):
    """Values that decide candidate preview bytes. No renderer takes part."""
    import PIL
    from PIL import features

    return {
        "vector_render": False,
        "resolution": list(silhouette.processed_size),
        "pillow": PIL.__version__,
        "font": type(font).__name__,
        "freetype2": features.version("freetype2"),
        "label_size": LABEL_SIZE,
        "labels": labels,
    }


def provenance(prepared, font):
    """Values that decide preview bytes. Same versions and build give same bytes."""
    import PIL
    from PIL import features

    return {
        "renderer": "resvg-py",
        "renderer_version": version("resvg-py"),
        "shape_rendering": "default",
        "root": "pixel",
        "resolution": list(prepared.processed_size),
        "pillow": PIL.__version__,
        "font": type(font).__name__,
        "freetype2": features.version("freetype2"),
        "label_size": LABEL_SIZE,
        "labels": list(LABELS.values()),
    }
