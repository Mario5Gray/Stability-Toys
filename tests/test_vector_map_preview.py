"""S2.7 preview rendering and composition. STABL-kfrksmnp, spec 6.4.

Real vtracer==0.6.15 traces and real resvg-py==0.5.0 renders. Mocks are not render proof.
The locked corpus renderer (metrics.render) renders the raw pixel trace. It is the reference.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vtracer", reason="vtracer==0.6.15 is not installed. Install the vector extra.")
pytest.importorskip("resvg_py", reason="resvg-py==0.5.0 is not installed. Install the vector extra.")

from tests.fixtures.vector_map.corpus import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures" / "vector_map"

import vector_map_config as config  # noqa: E402
import vector_map_preview as preview  # noqa: E402
import vector_map_svg as svg_io  # noqa: E402
import vector_map_vtracer as adapter  # noqa: E402

# 25.4/96 and 1/3 broke an exact-dpi mm-root render. Keep them.
SCALES = [0.1, 1 / 3, 0.37, 0.5, 25.4 / 96, 2.54, 7.0]


def _mask(path, rows=None):
    material = np.asarray(Image.open(path).convert("L")) >= 128
    return material[:rows] if rows else material


MATERIALS = {
    "asymmetric": lambda: _mask(FIXTURES / "asymmetric.png"),
    # S2.5: gravel at a non-exact dpi rendered 475 to 6631 px wrong.
    "gravel": lambda: _mask(FIXTURES / "corpus" / "gravel" / "mask.png"),
    "gravel_wide": lambda: _mask(FIXTURES / "corpus" / "gravel" / "mask.png", rows=200),
}


def _canvas(material, mm_per_px):
    height, width = material.shape
    return config.Canvas(width, height, width * mm_per_px, height * mm_per_px, mm_per_px)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name", sorted(MATERIALS))
def test_crisp_render_of_normalized_svg_equals_raw_pixel_trace(name, scale):
    material = MATERIALS[name]()
    traced = adapter.trace_layer(material)
    canvas = _canvas(material, scale)
    normalized = svg_io.normalize_svg(traced.svg, canvas)
    rendered = preview.render_material(normalized.svg, canvas)
    height, width = material.shape
    assert rendered.shape == (height, width)
    assert np.array_equal(rendered, metrics.render(traced.svg, width, height))


@pytest.mark.parametrize("dimension", ["width_mm", "height_mm"])
def test_render_uses_physical_canvas_from_either_dimension(dimension):
    material = MATERIALS["gravel_wide"]()
    height, width = material.shape
    settings = config.resolve(config.DEFAULTS, {
        "input": FIXTURES / "asymmetric.png", "input_kind": "mask", dimension: 123.4,
    })
    canvas = config.physical_canvas(settings, width, height)
    traced = adapter.trace_layer(material)
    rendered = preview.render_material(svg_io.normalize_svg(traced.svg, canvas).svg, canvas)
    assert np.array_equal(rendered, metrics.render(traced.svg, width, height))


def test_exact_dpi_mm_root_render_is_wrong_at_non_binary_scales():
    """Pin the trap. resvg-py 0.5.0 converts mm to px in float32.

    At 25.4/96 mm/px, 67.7333 mm becomes 255.99998 px, and the render moves edges.
    S2.5 measured exact dpi only at 0.5 mm/px, where float32 is exact.
    """
    import resvg_py

    material = MATERIALS["gravel_wide"]()
    height, width = material.shape
    traced = adapter.trace_layer(material)
    canvas = _canvas(material, 25.4 / 96)
    png = resvg_py.svg_to_bytes(
        svg_string=svg_io.normalize_svg(traced.svg, canvas).svg, width=width, height=height,
        dpi=25.4 / canvas.mm_per_px, background="#ffffff", shape_rendering="crisp_edges",
    )
    rendered = np.asarray(Image.open(__import__("io").BytesIO(bytes(png))).convert("L")) < 128
    assert np.count_nonzero(rendered ^ metrics.render(traced.svg, width, height)) > 0


def test_pixel_root_changes_only_root_size():
    material = MATERIALS["asymmetric"]()
    canvas = _canvas(material, 0.37)
    normalized = svg_io.normalize_svg(adapter.trace_layer(material).svg, canvas).svg
    pixel = preview.pixel_root(normalized, canvas)
    root = svg_io.ET.fromstring(pixel)
    assert (root.get("width"), root.get("height"), root.get("viewBox")) == ("128", "128", "0 0 128 128")
    original = svg_io.ET.fromstring(normalized)
    assert [(e.tag, e.attrib) for e in root.iter()][1:] == [(e.tag, e.attrib) for e in original.iter()][1:]


# --- Composition ------------------------------------------------------------


import vector_map_raster as raster  # noqa: E402

PANELS = ("mask", "vector", "overlay")


def _build(source, *, max_res=None, **values):
    """Prepare, trace, normalize and compose like the CLI. Return parts for assertions."""
    settings = config.resolve(config.DEFAULTS, {
        "input": source, "input_kind": "mask", "width_mm": 50, "max_res": max_res, **values,
    })
    data = Path(source).read_bytes()
    prepared = raster.prepare(settings, input_bytes={Path(source): data})
    normalized = svg_io.normalize_svg(adapter.trace_layer(prepared.material, settings.vtracer).svg, prepared.canvas)
    result = preview.compose(prepared, normalized.svg, data)
    image = np.asarray(Image.open(__import__("io").BytesIO(result.png)).convert("RGB"))
    return result, image, prepared, normalized


def _crop(image, box):
    left, top, right, bottom = box
    return image[top:bottom, left:right]


def _speckled(tmp_path):
    """Isolated 1 px dots. filter_speckle removes them, so vector differs from mask."""
    pixels = np.asarray(Image.open(FIXTURES / "asymmetric.png").convert("L")).copy()
    pixels[2:120:9, 2] = 255
    path = tmp_path / "speckled.png"
    Image.fromarray(pixels).save(path)
    return path


def test_preview_is_png_with_three_processing_size_panels():
    result, image, prepared, _ = _build(FIXTURES / "asymmetric.png")
    width, height = prepared.processed_size
    boxes = [result.panels[name] for name in PANELS]
    for left, top, right, bottom in boxes:
        assert (right - left, bottom - top) == (width, height)
        assert 0 <= left and right <= image.shape[1] and 0 <= top and bottom <= image.shape[0]
    for index, a in enumerate(boxes):
        for b in boxes[index + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    assert result.png.startswith(b"\x89PNG")


def test_mask_panel_shows_prepared_material_exactly():
    result, image, prepared, _ = _build(FIXTURES / "asymmetric.png", max_res=100)
    panel = _crop(image, result.panels["mask"])
    expected = np.where(prepared.material, 0, 255).astype(np.uint8)
    for channel in range(3):
        assert np.array_equal(panel[..., channel], expected)


def test_vector_panel_is_anti_aliased_render_of_published_svg():
    result, image, prepared, normalized = _build(FIXTURES / "corpus" / "gravel" / "mask.png")
    panel = _crop(image, result.panels["vector"])
    expected = preview.render_luminance(normalized.svg, prepared.canvas)
    assert np.array_equal(panel[..., 0], expected)
    assert np.count_nonzero((expected > 0) & (expected < 255)) > 0  # default anti-aliasing


def test_vector_panel_shows_fitting_change_not_mask(tmp_path):
    result, image, prepared, _ = _build(_speckled(tmp_path))
    mask_panel = _crop(image, result.panels["mask"])[..., 0]
    vector_panel = _crop(image, result.panels["vector"])[..., 0]
    assert not np.array_equal(vector_panel < 128, mask_panel < 128)
    assert np.all(vector_panel[2:120:9, 2] >= 128)  # removed dots stay absent in the vector panel


def test_overlay_marks_mask_only_material_with_mask_only_colour(tmp_path):
    result, image, *_ = _build(_speckled(tmp_path))
    overlay = _crop(image, result.panels["overlay"]).astype(float)
    names = ("both", "vector_only", "mask_only")
    alpha = preview.OVERLAY_ALPHA
    # Speckle dots are white in the source.
    blended = np.array([preview.blend_base(255) * (1 - alpha) + np.array(preview.LEGEND[n]) * alpha for n in names])
    for y in range(2, 120, 9):
        assert names[np.argmin(np.abs(blended - overlay[y, 2]).sum(1))] == "mask_only", (y, overlay[y, 2])


def _rotated_source(tmp_path):
    """Stored sideways with EXIF orientation 6. Oriented pixels equal the gravel mask."""
    oriented = Image.open(FIXTURES / "corpus" / "gravel" / "mask.png").convert("L")
    stored = oriented.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.png"
    stored.save(path, exif=exif)
    return path, np.asarray(oriented)


@pytest.mark.parametrize("max_res", [None, 100])
def test_overlay_aligns_oriented_source_with_processing_canvas(tmp_path, max_res):
    path, oriented = _rotated_source(tmp_path)
    result, image, prepared, normalized = _build(path, max_res=max_res)
    width, height = prepared.processed_size
    source = np.asarray(Image.fromarray(oriented).resize((width, height), Image.Resampling.NEAREST))
    overlay = _crop(image, result.panels["overlay"])
    coverage = preview.render_luminance(normalized.svg, prepared.canvas) < 255
    neither = ~prepared.material & ~coverage
    assert np.count_nonzero(neither) > 0
    for channel in range(3):
        assert np.array_equal(overlay[..., channel][neither], preview.blend_base(source)[neither])


def test_legend_shows_every_overlay_colour():
    result, image, *_ = _build(FIXTURES / "asymmetric.png")
    legend = _crop(image, result.panels["legend"]).reshape(-1, 3)
    present = {tuple(pixel) for pixel in legend}
    for colour in preview.LEGEND.values():
        assert tuple(colour) in present


def test_labels_use_embedded_default_font_only(monkeypatch):
    from PIL import ImageFont

    calls = []
    original = ImageFont.truetype

    def spy(font=None, *args, **kwargs):
        calls.append(font)
        return original(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", spy)
    result, image, *_ = _build(FIXTURES / "asymmetric.png")
    assert not [font for font in calls if isinstance(font, (str, Path))]
    for name in PANELS:
        left, top, right, _ = result.panels[name]
        strip = image[top - preview.LABEL_HEIGHT:top, left:right]
        assert np.count_nonzero(strip.min(axis=2) < 128) > 0, name


def test_provenance_records_renderer_pillow_and_font():
    from importlib.metadata import version

    import PIL
    from PIL import ImageFont, features

    result, *_ = _build(FIXTURES / "asymmetric.png")
    assert result.provenance == {
        "renderer": "resvg-py",
        "renderer_version": version("resvg-py"),
        "shape_rendering": "default",
        "root": "pixel",
        "resolution": [128, 128],
        "pillow": PIL.__version__,
        "font": type(ImageFont.load_default(preview.LABEL_SIZE)).__name__,
        "freetype2": features.version("freetype2"),
        "label_size": preview.LABEL_SIZE,
        "labels": list(preview.LABELS.values()),
    }


def test_provenance_records_null_freetype_when_absent(monkeypatch):
    from PIL import features

    real = features.version
    monkeypatch.setattr(features, "version", lambda name: None if name == "freetype2" else real(name))
    result, *_ = _build(FIXTURES / "asymmetric.png")
    assert result.provenance["freetype2"] is None


def test_compose_is_deterministic():
    first, *_ = _build(FIXTURES / "asymmetric.png")
    second, *_ = _build(FIXTURES / "asymmetric.png")
    assert first.png == second.png
