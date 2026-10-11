"""S2.7 preview rendering and composition. STABL-kfrksmnp, spec 6.4.

Real vtracer==0.6.15 traces and real resvg-py==0.5.0 renders. Mocks are not render proof.
The locked corpus renderer (metrics.render) renders the raw pixel trace. It is the reference.
"""

import json
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


def test_pillow_without_size_argument_uses_embedded_bitmap_font(monkeypatch):
    """Pillow 10.0 load_default() takes no size. The pyproject floor is Pillow>=10.0."""
    from PIL import ImageFont

    bitmap = ImageFont.load_default_imagefont

    def load_default_10_0():
        return bitmap()

    monkeypatch.setattr(ImageFont, "load_default", load_default_10_0)
    result, image, *_ = _build(FIXTURES / "asymmetric.png")
    assert result.provenance["font"] == "ImageFont"
    left, top, right, _ = result.panels["mask"]
    assert np.count_nonzero(image[top - preview.LABEL_HEIGHT:top, left:right].min(axis=2) < 128) > 0


_NO_FREETYPE = """
import json, sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "PIL._imagingft":
            raise ModuleNotFoundError("blocked FreeType", name=name)
        return None

sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import vector_map
code = vector_map.main([sys.argv[2], sys.argv[3], "--input-kind", "mask", "--width-mm", "10",
                        "--preview", "--debug-bundle", "--json"])
print("EXIT", code)
"""


def test_preview_without_freetype_uses_embedded_bitmap_font(tmp_path):
    """Real absent FreeType: Pillow cannot import PIL._imagingft in a fresh interpreter."""
    import json
    import subprocess

    out = tmp_path / "out.svg"
    result = subprocess.run(
        [sys.executable, "-c", _NO_FREETYPE, str(ROOT / "scripts"), str(FIXTURES / "donut.png"), str(out)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert "EXIT 0" in result.stdout, result.stderr
    payload = json.loads(result.stdout.splitlines()[0])
    assert payload["artifacts"]["preview"] == str(tmp_path / "out.preview.png")
    manifest = json.loads((tmp_path / "out.vector.json").read_bytes())
    assert manifest["preview"]["freetype2"] is None
    assert manifest["preview"]["font"] == "ImageFont"


def test_compose_is_deterministic():
    first, *_ = _build(FIXTURES / "asymmetric.png")
    second, *_ = _build(FIXTURES / "asymmetric.png")
    assert first.png == second.png


# --- S3.2 candidate mask preview (STABL-uifsadne) ------------------------------

import vector_map_layers as layers  # noqa: E402


def _candidates(tmp_path, *, detail_empty=False, mm_per_px=1.0):
    """Silhouette from a mask, structure and detail from supplied maps. One mm per pixel by default."""
    shape = (24, 32)
    source = tmp_path / "source.png"
    Image.fromarray(np.full((*shape, 3), 90, np.uint8)).save(source)
    silhouette = np.zeros(shape, bool)
    silhouette[2:22, 2:30] = True
    structure = np.zeros(shape, bool)
    structure[8, :] = True
    detail = np.zeros(shape, bool)
    if not detail_empty:
        detail[14:17, 5:25] = True
    paths = {}
    for name, material in (("mask", silhouette), ("structure", structure), ("detail", detail)):
        paths[name] = tmp_path / f"{name}.png"
        Image.fromarray(np.where(material, 255, 0).astype(np.uint8)).save(paths[name])
    layer = {**dict.fromkeys(config.FIELDS), "mask": paths["mask"]}
    settings = config.resolve(config.DEFAULTS, {"input": source, "input_kind": "image",
                                                "width_mm": 32 * mm_per_px}, layer)
    requests = [layers.RoleRequest("structure", source="map", path=paths["structure"]),
                layers.RoleRequest("detail", source="map", path=paths["detail"])]
    return layers.prepare_layers(settings, requests)


def _image(result):
    return np.asarray(Image.open(__import__("io").BytesIO(result.png)).convert("RGB"))


def test_candidate_preview_shows_each_selected_mask_in_role_order(tmp_path):
    prepared = _candidates(tmp_path)
    result = preview.compose_candidates(prepared)
    image = _image(result)
    assert [name for name in result.panels if name in prepared.roles] == list(prepared.roles)
    lefts = [result.panels[role][0] for role in prepared.roles]
    assert lefts == sorted(lefts)
    for role in prepared.roles:
        material = prepared.silhouette.material if role == "silhouette" else getattr(prepared, role).material
        panel = _crop(image, result.panels[role])
        for channel in range(3):
            assert np.array_equal(panel[..., channel], np.where(material, 0, 255).astype(np.uint8)), role


def test_candidate_preview_marks_an_empty_optional_panel(tmp_path):
    prepared = _candidates(tmp_path, detail_empty=True)
    result = preview.compose_candidates(prepared)
    assert result.provenance["labels"][-1] == "Detail candidate mask (empty)"
    assert (_crop(_image(result), result.panels["detail"]) == 255).all()


def test_candidate_preview_labels_masks_not_rendered_vectors(tmp_path):
    prepared = _candidates(tmp_path)
    result = preview.compose_candidates(prepared)
    image = _image(result)
    assert result.provenance["labels"] == ["Silhouette mask", "Structure candidate mask", "Detail candidate mask"]
    assert result.provenance["vector_render"] is False
    assert "renderer" not in result.provenance
    for role in prepared.roles:
        left, top, right, _ = result.panels[role]
        assert np.count_nonzero(image[top - preview.LABEL_HEIGHT:top, left:right].min(axis=2) < 128) > 0, role
    left, top, right, bottom = result.panels["note"]
    assert np.count_nonzero(image[top:bottom, left:right].min(axis=2) < 128) > 0


def test_candidate_preview_for_a_silhouette_only_run_has_one_panel(tmp_path):
    prepared = _candidates(tmp_path)
    only = layers.PreparedLayers(prepared.silhouette)
    result = preview.compose_candidates(only)
    assert [name for name in result.panels if name != "note"] == ["silhouette"]


def test_candidate_preview_is_deterministic(tmp_path):
    prepared = _candidates(tmp_path)
    assert preview.compose_candidates(prepared).png == preview.compose_candidates(prepared).png


# --- S3.3 layered preview (STABL-qlagdbmh) -------------------------------------


def _published(prepared):
    """Trace and normalize each nonempty selected role like the CLI. Return role -> normalized SVG."""
    svgs = {}
    for role in prepared.roles:
        material = _material(prepared, role)
        if material.any():
            svgs[role] = svg_io.normalize_svg(adapter.trace_layer(material).svg, prepared.canvas).svg
    return svgs


def _material(prepared, role):
    return prepared.silhouette.material if role == "silhouette" else getattr(prepared, role).material


def _compose(tmp_path, prepared, svgs=None):
    svgs = _published(prepared) if svgs is None else svgs
    return preview.compose_layers(prepared, svgs, (tmp_path / "source.png").read_bytes())


def test_layered_preview_has_a_mask_and_a_vector_panel_per_selected_role(tmp_path):
    prepared = _candidates(tmp_path)
    result = _compose(tmp_path, prepared)
    image = _image(result)
    width, height = prepared.silhouette.processed_size
    names = [f"{role}_{kind}" for kind in ("mask", "vector") for role in prepared.roles] + ["overlay"]
    assert [name for name in result.panels if name != "legend"] == names
    boxes = [result.panels[name] for name in names]
    for left, top, right, bottom in boxes:
        assert (right - left, bottom - top) == (width, height)
        assert 0 <= left and right <= image.shape[1] and 0 <= top and bottom <= image.shape[0]
    for index, a in enumerate(boxes):
        for b in boxes[index + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    for role in prepared.roles:
        assert result.panels[f"{role}_mask"][3] <= result.panels[f"{role}_vector"][1]
        assert result.panels[f"{role}_mask"][0] == result.panels[f"{role}_vector"][0]
    assert result.png.startswith(bytes([0x89]) + b"PNG")


def test_layered_mask_panels_show_prepared_masks_exactly(tmp_path):
    prepared = _candidates(tmp_path)
    result = _compose(tmp_path, prepared)
    image = _image(result)
    for role in prepared.roles:
        panel = _crop(image, result.panels[f"{role}_mask"])
        expected = np.where(_material(prepared, role), 0, 255).astype(np.uint8)
        for channel in range(3):
            assert np.array_equal(panel[..., channel], expected), role


@pytest.mark.parametrize("mm_per_px", [1.0, 0.37, 25.4 / 96])
def test_layered_vector_panels_render_the_published_svgs_under_a_pixel_root(tmp_path, mm_per_px):
    prepared = _candidates(tmp_path, mm_per_px=mm_per_px)
    svgs = _published(prepared)
    result = _compose(tmp_path, prepared, svgs)
    image = _image(result)
    for role, svg in svgs.items():
        expected = preview.render_luminance(svg, prepared.canvas)
        panel = _crop(image, result.panels[f"{role}_vector"])
        for channel in range(3):
            assert np.array_equal(panel[..., channel], expected), role
        assert np.array_equal(preview.render_material(svg, prepared.canvas), _material(prepared, role)), role


def test_layered_preview_marks_an_empty_role_and_renders_no_svg_for_it(tmp_path, monkeypatch):
    prepared = _candidates(tmp_path, detail_empty=True)
    svgs = _published(prepared)
    assert list(svgs) == ["silhouette", "structure"]
    rendered = []
    render = preview.render_luminance

    def record(svg, canvas, **kwargs):
        rendered.append(svg)
        return render(svg, canvas, **kwargs)

    monkeypatch.setattr(preview, "render_luminance", record)
    result = _compose(tmp_path, prepared, svgs)
    assert sorted(rendered) == sorted(svgs.values())
    labels = result.provenance["labels"]
    assert "Detail mask (empty)" in labels
    assert "Detail vector: empty, no SVG" in labels
    panel = _crop(_image(result), result.panels["detail_vector"])
    assert (panel == np.array(preview.EMPTY_PANEL, np.uint8)).all()


def test_layered_overlay_tints_source_with_each_role_in_order(tmp_path):
    prepared = _candidates(tmp_path)
    source_bytes = (tmp_path / "source.png").read_bytes()
    result = _compose(tmp_path, prepared)
    overlay = _crop(_image(result), result.panels["overlay"]).astype(int)
    base = preview.blend_base(preview._source(source_bytes, prepared.silhouette)).astype(float)

    def tint(value, roles):
        out = np.array([value] * 3, float)
        for role in roles:
            out = out * (1 - preview.OVERLAY_ALPHA) + preview.OVERLAY_ALPHA * np.array(preview.ROLE_COLOURS[role])
        return np.rint(out).astype(int)

    assert (overlay[0, 0] == tint(base[0, 0], [])).all()
    assert (overlay[4, 4] == tint(base[4, 4], ["silhouette"])).all()
    assert (overlay[8, 10] == tint(base[8, 10], ["silhouette", "structure"])).all()
    assert (overlay[15, 10] == tint(base[15, 10], ["silhouette", "detail"])).all()


def test_layered_legend_shows_each_selected_role_colour(tmp_path):
    prepared = _candidates(tmp_path)
    result = _compose(tmp_path, prepared)
    assert result.provenance["legend"] == ["silhouette", "structure", "detail"]
    legend = _crop(_image(result), result.panels["legend"]).reshape(-1, 3)
    colours = {tuple(int(value) for value in pixel) for pixel in legend}
    assert all(tuple(preview.ROLE_COLOURS[role]) in colours for role in prepared.roles)


def test_silhouette_only_layered_preview_has_one_column(tmp_path):
    prepared = layers.PreparedLayers(_candidates(tmp_path).silhouette)
    result = _compose(tmp_path, prepared)
    assert [name for name in result.panels if name != "legend"] == ["silhouette_mask", "silhouette_vector", "overlay"]
    assert result.provenance["legend"] == ["silhouette"]


def test_layered_preview_provenance_records_renderer_and_font(tmp_path):
    from importlib.metadata import version

    prepared = _candidates(tmp_path)
    provenance = _compose(tmp_path, prepared).provenance
    assert (provenance["renderer"], provenance["root"]) == ("resvg-py", "pixel")
    assert provenance["renderer_version"] == version("resvg-py")
    assert provenance["resolution"] == list(prepared.silhouette.processed_size)
    assert provenance["layers"] == list(prepared.roles)
    assert provenance["font"] in ("FreeTypeFont", "ImageFont")


def test_layered_preview_without_freetype_uses_embedded_bitmap_font(tmp_path):
    """Real absent FreeType in a fresh interpreter. Pillow >= 10.1 load_default() otherwise returns FreeType."""
    import subprocess

    _candidates(tmp_path)
    script = _NO_FREETYPE.split("import vector_map")[0] + (
        "import vector_map\ncode = vector_map.main(json.loads(sys.argv[2]))\nprint('EXIT', code)\n")
    argv = [str(value) for value in (
        tmp_path / "source.png", tmp_path / "out.svg", "--input-kind", "image", "--mask", tmp_path / "mask.png",
        "--width-mm", "32", "--layers", "silhouette,structure,detail", "--structure-map", tmp_path / "structure.png",
        "--detail-map", tmp_path / "detail.png", "--preview", "--json")]
    result = subprocess.run([sys.executable, "-c", script, str(ROOT / "scripts"), json.dumps(argv)],
                            capture_output=True, text=True, cwd=ROOT)
    assert "EXIT 0" in result.stdout, result.stderr
    manifest = json.loads((tmp_path / "out.vector.json").read_bytes())
    assert manifest["preview"]["freetype2"] is None
    assert manifest["preview"]["font"] == "ImageFont"


def test_layered_preview_is_deterministic(tmp_path):
    prepared = _candidates(tmp_path)
    svgs = _published(prepared)
    assert _compose(tmp_path, prepared, svgs).png == _compose(tmp_path, prepared, svgs).png


def test_layered_cli_preview_renders_the_published_layer_files(tmp_path, capsys):
    import vector_map

    prepared = _candidates(tmp_path)
    code = vector_map.main([str(value) for value in (
        tmp_path / "source.png", tmp_path / "out.svg", "--input-kind", "image", "--mask", tmp_path / "mask.png",
        "--width-mm", "32", "--layers", "silhouette,structure,detail", "--structure-map", tmp_path / "structure.png",
        "--detail-map", tmp_path / "detail.png", "--preview", "--json")])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    result = json.loads(captured.out)
    assert result["artifacts"]["preview"] == str(tmp_path / "out.preview.png")
    published = {role: (tmp_path / "out.layers" / f"{role}.svg").read_text() for role in prepared.roles}
    assert (tmp_path / "out.preview.png").read_bytes() == _compose(tmp_path, prepared, published).png
    manifest = json.loads((tmp_path / "out.vector.json").read_bytes())
    assert "out.preview.png" in {item["path"] for item in manifest["artifacts"]}
    assert manifest["preview"]["layers"] == ["silhouette", "structure", "detail"]
