"""S2.3 raster contracts. STABL-vjpnctjh."""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vector_map_config as config
import vector_map_raster as raster


def save(path, pixels, **kwargs):
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path, **kwargs)
    return path


def settings(path, **kwargs):
    return config.resolve(config.DEFAULTS, {"input": path, "input_kind": "mask", "width_mm": 20.0}, kwargs)


def prepare(path, **kwargs):
    return raster.prepare(settings(path, **kwargs))


def diagnostics(material):
    return importlib.import_module("vector_map_diagnostics").feature_diagnostics(material)


def codes(result):
    return {item["code"] for item in result.diagnostics}


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_exif_orientations_use_expected_pixels(tmp_path, orientation):
    pixels = np.zeros((6, 10), np.uint8)
    pixels[1:4, 2:5] = 255
    pixels[4, 8] = 255
    exif = Image.Exif()
    exif[274] = orientation
    source = save(tmp_path / "source.png", pixels, exif=exif)
    expected = {
        1: pixels, 2: np.fliplr(pixels), 3: np.rot90(pixels, 2), 4: np.flipud(pixels),
        5: pixels.T, 6: np.rot90(pixels, -1), 7: np.rot90(pixels.T, 2), 8: np.rot90(pixels),
    }[orientation]
    result = prepare(source)
    np.testing.assert_array_equal(result.material, expected >= 128)
    assert result.original_size == (10, 6)
    assert result.oriented_size == expected.shape[::-1]
    assert result.processed_size == expected.shape[::-1]


def test_luminance_threshold_and_inversion(tmp_path):
    path = save(tmp_path / "source.png", [[0, 127, 128, 255]])
    assert prepare(path).material.tolist() == [[False, False, True, True]]
    assert prepare(path, invert=True).material.tolist() == [[True, True, False, False]]


def test_alpha_selection_and_ignored_alpha_warning(tmp_path):
    pixels = np.full((4, 4, 4), 255, np.uint8)
    pixels[:, :, 3] = [0, 127, 128, 255]
    path = save(tmp_path / "source.png", pixels)
    alpha = prepare(path, alpha=True)
    assert alpha.material[0].tolist() == [False, False, True, True]
    assert "alpha_ignored" not in codes(alpha)
    np.testing.assert_array_equal(prepare(path, alpha=True, invert=True).material, ~alpha.material)
    default = prepare(path)
    assert default.material.all()
    warning = next(item for item in default.diagnostics if item["code"] == "alpha_ignored")
    assert "--alpha" in warning["message"] and "luminance" in warning["message"]


def test_palette_alpha_selection(tmp_path):
    image = Image.fromarray(np.tile(np.arange(4, dtype=np.uint8), (4, 1))).convert("P")
    image.save(tmp_path / "source.png", transparency=bytes([0, 127, 128, 255]))
    assert prepare(tmp_path / "source.png", alpha=True).material[0].tolist() == [False, False, True, True]


def test_alpha_missing_is_configuration_error(tmp_path):
    path = save(tmp_path / "source.png", np.full((8, 8), 255))
    with pytest.raises(config.ConfigError, match="alpha"):
        prepare(path, alpha=True)


def test_jpeg_accepted_but_gif_rejected_by_decoded_format(tmp_path):
    jpeg = save(tmp_path / "source.jpg", np.full((8, 12), 255))
    assert prepare(jpeg).material.all()
    gif = save(tmp_path / "not-a-gif.png", np.ones((8, 8)), format="GIF")
    with pytest.raises(config.ConfigError, match="PNG|JPEG"):
        prepare(gif)


def test_bad_bytes_and_pixel_limit_remain_processing_errors(tmp_path, monkeypatch):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        prepare(path)
    path = save(path, np.zeros((8, 8)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="pixel limit"):
        prepare(path)


@pytest.mark.parametrize("axis", ["width_mm", "height_mm"])
def test_resize_preserves_uniform_scale_and_selected_axis(tmp_path, axis):
    path = save(tmp_path / "source.png", np.full((7, 10), 255))
    result = prepare(path, max_res=7, **{axis: 14.0})
    assert result.processed_size == (7, 5)
    assert getattr(result.canvas, axis) == 14.0
    assert result.canvas.width_mm / 7 == result.canvas.height_mm / 5
    assert result.canvas.mm_per_px == 14 / (7 if axis == "width_mm" else 5)
    assert result.original_size == result.oriented_size == (10, 7)


def test_resize_thresholds_before_nearest_sampling_and_never_upscales(tmp_path):
    pixels = np.array([[0, 255, 0, 255], [255, 0, 255, 0]], np.uint8)
    path = save(tmp_path / "source.png", pixels)
    assert prepare(path, max_res=2).material.tolist() == [[False, False]]
    np.testing.assert_array_equal(prepare(path, max_res=100).material, pixels >= 128)
    assert prepare(path, max_res=1).processed_size == (1, 1)


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_invalid_resolution_rejected(value):
    with pytest.raises(config.ConfigError, match="max_res"):
        settings(Path("unused"), max_res=value)


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_invalid_edge_width_rejected(value):
    with pytest.raises(config.ConfigError, match="line_width_mm"):
        settings(Path("unused"), input_kind="edges", line_width_mm=value)


def test_mask_width_and_reserved_silhouette_mask_rejected():
    with pytest.raises(config.ConfigError, match="edges"):
        settings(Path("unused"), line_width_mm=1)
    with pytest.raises(config.ConfigError, match="S3.1"):
        settings(Path("unused"), mask=Path("mask.png"))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_constraints_orient_independently_before_dimension_check(tmp_path, orientation):
    source = save(tmp_path / "source.png", np.full((6, 10), 255))
    expected = np.zeros((6, 10), np.uint8)
    expected[1:5, 2:6] = 255
    inverse = {
        1: expected, 2: np.fliplr(expected), 3: np.rot90(expected, 2), 4: np.flipud(expected),
        5: expected.T, 6: np.rot90(expected), 7: np.rot90(expected.T, 2), 8: np.rot90(expected, -1),
    }[orientation]
    exif = Image.Exif()
    exif[274] = orientation
    include = save(tmp_path / "include.png", inverse, exif=exif)
    result = prepare(source, include_mask=include)
    np.testing.assert_array_equal(result.material, expected >= 128)


def test_constraint_mismatch_rejected_before_resize(tmp_path):
    source = save(tmp_path / "source.png", np.ones((8, 12)))
    include = save(tmp_path / "include.png", np.ones((4, 6)))
    with pytest.raises(config.ConfigError, match="dimensions"):
        prepare(source, include_mask=include, max_res=6)


def test_constraints_precede_and_follow_expansion(tmp_path):
    pixels = np.zeros((16, 20), np.uint8)
    pixels[5:11, 9] = 255
    pixels[5:11, 5] = 255
    include = np.zeros_like(pixels)
    include[:, 7:13] = 255
    exclude = np.zeros_like(pixels)
    exclude[8, :] = 255
    source = save(tmp_path / "source.png", pixels)
    kwargs = dict(input_kind="edges", width_mm=20, line_width_mm=5,
                  include_mask=save(tmp_path / "include.png", include),
                  exclude_mask=save(tmp_path / "exclude.png", exclude))
    result = prepare(source, **kwargs)
    expected = np.zeros_like(pixels, bool)
    expected[3:13, 7:12] = True
    expected[8] = False
    np.testing.assert_array_equal(result.material, expected)


def test_constraints_keep_polarity_when_source_inverts(tmp_path):
    path = save(tmp_path / "source.png", np.zeros((8, 8)))
    include = np.zeros((8, 8), np.uint8)
    include[2:6, 2:6] = 255
    inc = save(tmp_path / "include.png", include)
    np.testing.assert_array_equal(prepare(path, invert=True, include_mask=inc).material, include >= 128)


@pytest.mark.parametrize("requested, side", [(0.1, 1), (1, 1), (1.01, 3), (3, 3), (3.1, 5), (4, 5), (5, 5)])
def test_edge_rounding_reports_achieved_width(tmp_path, requested, side):
    pixels = np.zeros((20, 20), np.uint8)
    pixels[5:15, 10] = 255
    path = save(tmp_path / "source.png", pixels)
    result = prepare(path, input_kind="edges", line_width_mm=requested)
    assert result.material[10].sum() == side
    assert result.expansion.kernel_size_px == side
    assert result.expansion.radius_px == (side - 1) // 2
    assert result.expansion.achieved_width_mm == side
    assert result.expansion.expansion_mm == (side - 1) / 2
    assert result.expansion.requested_width_mm == requested
    assert ("requested_width_below_four_pixels" in codes(result)) == (requested < 4)


def test_existing_wide_bands_never_shrink(tmp_path):
    pixels = np.zeros((20, 20), np.uint8)
    pixels[3:17, 5:12] = 255
    path = save(tmp_path / "source.png", pixels)
    plain = prepare(path, input_kind="edges")
    assert plain.expansion is None
    np.testing.assert_array_equal(plain.material, pixels >= 128)
    assert prepare(path, input_kind="edges", line_width_mm=3).material[10].sum() == 9


def test_huge_width_rejected_without_allocation(tmp_path):
    path = save(tmp_path / "source.png", np.ones((8, 8)))
    with pytest.raises(config.ConfigError, match="width|kernel"):
        prepare(path, input_kind="edges", line_width_mm=1e300)


def test_underflow_scale_rejected(tmp_path):
    path = save(tmp_path / "source.png", np.ones((8, 8)))
    with pytest.raises(config.ConfigError, match="scale"):
        prepare(path, width_mm=5e-324)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("background", [False, True])
def test_feature_threshold_for_bands_and_gaps(width, background):
    material = np.zeros((20, 20), bool)
    material[:, 8:8 + width] = True
    if background:
        material = ~material
    before = material.copy()
    found = diagnostics(material)
    target = "thin_background" if background else "thin_material"
    assert (target in {item["code"] for item in found}) == (width < 4)
    np.testing.assert_array_equal(material, before)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_canvas_edge_material_warns_but_exterior_margin_does_not(width):
    material = np.zeros((16, 16), bool)
    material[:, :width] = True
    found = diagnostics(material)
    assert ("thin_material" in {item["code"] for item in found}) == (width < 4)
    inverse = diagnostics(~material)
    assert "thin_background" not in {item["code"] for item in inverse}


def test_deep_border_channel_and_enclosed_hole_warn():
    material = np.ones((20, 20), bool)
    material[:12, 8:10] = False
    material[14:16, 14:16] = False
    found = diagnostics(material)
    warning = next(item for item in found if item["code"] == "thin_background")
    assert warning["pixels"] >= 20


def test_bridge_warns_and_wide_clipped_material_does_not():
    material = np.zeros((20, 30), bool)
    material[4:16, :10] = True
    material[4:16, 20:] = True
    assert not diagnostics(material)
    material[8:10, 10:20] = True
    assert "thin_material" in {item["code"] for item in diagnostics(material)}


def test_even_kernel_does_not_shift_four_pixel_square():
    material = np.zeros((12, 12), bool)
    material[3:7, 5:9] = True
    assert not diagnostics(material)


def test_uniform_and_narrow_canvases():
    assert not diagnostics(np.zeros((2, 2), bool))
    assert not diagnostics(np.ones((8, 8), bool))
    assert "thin_material" in {item["code"] for item in diagnostics(np.ones((2, 8), bool))}


def test_diagonal_diagnostic_is_conservative_and_preserves_pixels():
    material = np.eye(12, dtype=bool)
    before = material.tobytes()
    found = diagnostics(material)
    assert any(item["code"] == "thin_material" for item in found)
    assert any("false positives" in item["message"] for item in found)
    assert material.tobytes() == before


def test_width_and_material_warnings_both_remain(tmp_path):
    pixels = np.zeros((20, 20), np.uint8)
    pixels[5:15, 10] = 255
    path = save(tmp_path / "source.png", pixels)
    result = prepare(path, input_kind="edges", line_width_mm=3)
    assert {"requested_width_below_four_pixels", "thin_material"} <= codes(result)


@pytest.mark.parametrize("constraint", ["include_mask", "exclude_mask"])
def test_removed_source_cannot_seed_dilation(tmp_path, constraint):
    pixels = np.zeros((20, 20), np.uint8)
    pixels[10, 9] = 255
    constraint_pixels = np.zeros_like(pixels)
    constraint_pixels[:, 10:] = 255
    if constraint == "exclude_mask":
        constraint_pixels = 255 - constraint_pixels
    path = save(tmp_path / "source.png", pixels)
    gate = save(tmp_path / "gate.png", constraint_pixels)
    result = prepare(path, input_kind="edges", line_width_mm=5, **{constraint: gate})
    assert not result.material.any()


def test_background_diagnostics_match_square_placement_oracle():
    rng = np.random.default_rng(8)
    for material in (rng.random((7, 8)) > 0.3, rng.random((7, 8)) > 0.7):
        actual = {item["code"]: item["pixels"] for item in diagnostics(material)}
        for background in (False, True):
            selected = ~material if background else material
            covered = np.zeros_like(selected)
            for y in range(-3, selected.shape[0]):
                for x in range(-3, selected.shape[1]):
                    positions = [(j, i) for j in range(y, y + 4) for i in range(x, x + 4)]
                    def present(j, i):
                        return selected[j, i] if 0 <= j < selected.shape[0] and 0 <= i < selected.shape[1] else background
                    if all(present(j, i) for j, i in positions):
                        for j, i in positions:
                            if 0 <= j < selected.shape[0] and 0 <= i < selected.shape[1]:
                                covered[j, i] = True
            count = int((selected & ~covered).sum())
            assert actual.get("thin_background" if background else "thin_material", 0) == count


@pytest.mark.parametrize("case", ["badge", "bracket", "brick", "coins", "gravel", "horse", "pcb", "truchet"])
def test_reviewed_corpus_masks_remain_unchanged(case):
    path = Path(__file__).parent / "fixtures" / "vector_map" / "corpus" / case / "mask.png"
    before = path.read_bytes()
    with Image.open(path) as image:
        expected = np.asarray(image) >= 128
    result = prepare(path)
    np.testing.assert_array_equal(result.material, expected)
    assert path.read_bytes() == before
    assert any(item["code"].startswith("thin_") for item in result.diagnostics)


def test_constraint_alpha_warning_does_not_suggest_source_selector(tmp_path):
    source = save(tmp_path / "source.png", np.full((8, 8), 255))
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[2:6, 2:6, :3] = 255
    include = save(tmp_path / "include.png", rgba)
    result = prepare(source, include_mask=include)
    warning = next(item for item in result.diagnostics if item["code"] == "alpha_ignored")
    assert "constraint" in warning["message"]
    assert "pass --alpha" not in warning["message"]
    assert result.material.sum() == 16
