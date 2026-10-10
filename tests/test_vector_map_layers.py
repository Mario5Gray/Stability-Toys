"""S3.2 structure and detail candidate masks. STABL-uifsadne.

Plan: docs/superpowers/plans/2026-10-09-vector-map-s3-2.md. Spec 6.1.
- Optional roles come from Canny or supplied maps on the S3.1 silhouette canvas.
- Constraints apply before and after gap closing and band expansion. Exclusion wins.
- Gap closing is opt-in. gap_close_mm is the maximum gap to close, not a kernel width.
- Roles are clipped to the silhouette. Final structure is removed from final detail.
"""

import math
import sys
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import canny_map  # noqa: E402
import vector_map_config as config  # noqa: E402
import vector_map_layers as layers  # noqa: E402
import vector_map_raster as raster  # noqa: E402
from vector_map_layers import CannySpec, RoleRequest  # noqa: E402


def save(path, pixels, **kwargs):
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path, **kwargs)
    return path


def settings(source, mask, *, width_mm=None, max_res=None, invert=None, include=None, exclude=None):
    """Image mode with a mask silhouette. Default width gives one millimetre per pixel."""
    with Image.open(source) as image:
        width = image.size[0]
    layer = {**dict.fromkeys(config.FIELDS), "mask": mask, "max_res": max_res, "invert": invert,
             "include_mask": include, "exclude_mask": exclude}
    base = {"input": source, "input_kind": "image", "width_mm": float(width if width_mm is None else width_mm)}
    return config.resolve(config.DEFAULTS, base, layer)


def rectangle_source(tmp_path, shape=(32, 40), box=(8, 24, 10, 30), name="source.png"):
    """Dark field with one light rectangle. Canny finds its border."""
    pixels = np.full((*shape, 3), 30, np.uint8)
    top, bottom, left, right = box
    pixels[top:bottom, left:right] = 220
    return save(tmp_path / name, pixels)


def full_mask(tmp_path, shape=(32, 40), name="mask.png"):
    return save(tmp_path / name, np.full(shape, 255, np.uint8))


def binary(tmp_path, material, name):
    return save(tmp_path / name, np.where(material, 255, 0))


def direct_canny(rgb, spec):
    edges = canny_map.canny_edges(Image.fromarray(rgb, "RGB"), low_threshold=spec.low_threshold,
                                  high_threshold=spec.high_threshold, blur=spec.blur, invert=False)
    return np.asarray(edges) >= 128


EXPLICIT = CannySpec(low_threshold=50, high_threshold=150, blur=0)


def codes(diagnostics):
    return [item["code"] for item in diagnostics]


# --- Request validation -------------------------------------------------------


def test_duplicate_roles_are_rejected(tmp_path):
    source = rectangle_source(tmp_path)
    with pytest.raises(config.ConfigError, match="duplicate"):
        layers.prepare_layers(settings(source, full_mask(tmp_path)),
                              [RoleRequest("structure", canny=EXPLICIT), RoleRequest("structure", canny=EXPLICIT)])


def test_unknown_role_and_source_are_rejected(tmp_path):
    source = rectangle_source(tmp_path)
    prepared = settings(source, full_mask(tmp_path))
    with pytest.raises(config.ConfigError, match="role"):
        layers.prepare_layers(prepared, [RoleRequest("silhouette")])
    with pytest.raises(config.ConfigError, match="source"):
        layers.prepare_layers(prepared, [RoleRequest("detail", source="vector")])


def test_map_request_requires_path(tmp_path):
    source = rectangle_source(tmp_path)
    with pytest.raises(config.ConfigError, match="path"):
        layers.prepare_layers(settings(source, full_mask(tmp_path)), [RoleRequest("structure", source="map")])


def test_map_request_rejects_canny_values(tmp_path):
    source = rectangle_source(tmp_path)
    request = RoleRequest("structure", source="map", path=full_mask(tmp_path, name="map.png"), canny=EXPLICIT)
    with pytest.raises(config.ConfigError, match="canny"):
        layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])


def test_canny_request_rejects_path(tmp_path):
    source = rectangle_source(tmp_path)
    request = RoleRequest("detail", path=full_mask(tmp_path, name="map.png"))
    with pytest.raises(config.ConfigError, match="path"):
        layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])


@pytest.mark.parametrize("blur", [-1, 2, 4, 1.0, True])
def test_invalid_blur_is_rejected(tmp_path, blur):
    source = rectangle_source(tmp_path)
    request = RoleRequest("detail", canny=CannySpec(50, 150, blur))
    with pytest.raises(config.ConfigError, match="blur"):
        layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])


@pytest.mark.parametrize("low, high", [(150, 150), (200, 100), (-1, 100), (True, 100), (50.0, 100), (50, "x")])
def test_thresholds_must_be_ordered_integers(tmp_path, low, high):
    source = rectangle_source(tmp_path)
    request = RoleRequest("structure", canny=CannySpec(low, high, 0))
    with pytest.raises(config.ConfigError, match="threshold"):
        layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])


@pytest.mark.parametrize("field", ["width_mm", "gap_close_mm"])
@pytest.mark.parametrize("value", [0, -1.0, math.nan, math.inf, True, "2"])
def test_width_and_gap_must_be_positive_finite_numbers(tmp_path, field, value):
    source = rectangle_source(tmp_path)
    request = RoleRequest("structure", canny=EXPLICIT, **{field: value})
    with pytest.raises(config.ConfigError, match=field):
        layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])


def test_selected_roles_normalize_to_fixed_order(tmp_path):
    source = rectangle_source(tmp_path)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)),
                                     [RoleRequest("detail", canny=EXPLICIT), RoleRequest("structure", canny=EXPLICIT)])
    assert prepared.roles == ("silhouette", "structure", "detail")
    only_detail = layers.prepare_layers(settings(source, full_mask(tmp_path)), [RoleRequest("detail", canny=EXPLICIT)])
    assert only_detail.roles == ("silhouette", "detail")
    assert only_detail.structure is None


def test_standalone_modes_are_rejected(tmp_path):
    mask = full_mask(tmp_path)
    standalone = config.resolve(config.DEFAULTS, {"input": mask, "input_kind": "mask", "width_mm": 40.0})
    with pytest.raises(config.ConfigError, match="image mode"):
        layers.prepare_layers(standalone, [])


# --- Source RGB preparation and Canny parity ----------------------------------


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_rgb_path_orients_the_source_first(tmp_path, orientation):
    rng = np.random.default_rng(orientation)
    pixels = rng.integers(0, 256, (12, 20, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = orientation
    source = save(tmp_path / "source.png", pixels, exif=exif)
    oriented = {3: np.rot90(pixels, 2), 6: np.rot90(pixels, -1), 8: np.rot90(pixels)}[orientation]
    size = oriented.shape[1::-1]
    assert np.array_equal(layers.source_rgb(source, size, size), oriented)


def test_rgb_path_composites_transparency_onto_black(tmp_path):
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[:, 6:, 3] = 255
    rgba[:, :3, 3] = 128
    source = tmp_path / "source.png"
    Image.fromarray(rgba, "RGBA").save(source)
    rgb = layers.source_rgb(source, (12, 10), (12, 10))
    assert rgb.shape == (10, 12, 3)
    assert (rgb[:, 6:] == 255).all()
    assert (rgb[:, 3:6] == 0).all()
    assert (rgb[:, :3] == 128).all()


def test_rgb_path_resizes_with_lanczos_to_processing_size(tmp_path):
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (30, 50, 3), dtype=np.uint8)
    source = save(tmp_path / "source.png", pixels)
    rgb = layers.source_rgb(source, (50, 30), (25, 15))
    with Image.fromarray(pixels, "RGB") as image:
        expected = np.asarray(image.resize((25, 15), resample=Image.Resampling.LANCZOS))
    assert np.array_equal(rgb, expected)


def test_rgb_path_rejects_a_size_that_differs_from_the_oriented_source(tmp_path):
    source = rectangle_source(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        layers.source_rgb(source, (32, 40), (32, 40))


@pytest.mark.parametrize("max_res", [None, 20])
def test_canny_candidate_matches_direct_canny_on_prepared_pixels(tmp_path, max_res):
    source = rectangle_source(tmp_path)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path), max_res=max_res),
                                     [RoleRequest("structure", canny=EXPLICIT)])
    silhouette = prepared.silhouette
    rgb = layers.source_rgb(source, silhouette.oriented_size, silhouette.processed_size)
    expected = direct_canny(rgb, EXPLICIT) & silhouette.material
    assert prepared.structure.material.shape == silhouette.material.shape
    assert np.array_equal(prepared.structure.material, expected)
    assert prepared.structure.material.any()


def test_candidates_share_the_silhouette_canvas(tmp_path):
    source = rectangle_source(tmp_path)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path), max_res=20),
                                     [RoleRequest("structure", canny=EXPLICIT), RoleRequest("detail", canny=EXPLICIT)])
    assert prepared.canvas == prepared.silhouette.canvas
    for candidate in (prepared.structure, prepared.detail):
        assert candidate.material.shape == prepared.silhouette.material.shape


def test_global_invert_changes_only_the_silhouette(tmp_path):
    source = rectangle_source(tmp_path)
    mask = np.zeros((32, 40), bool)
    mask[:, :20] = True
    mask_path = binary(tmp_path, mask, "mask.png")
    edge_map = np.zeros((32, 40), bool)
    edge_map[5, :] = True
    map_path = binary(tmp_path, edge_map, "map.png")
    requests = [RoleRequest("structure", canny=EXPLICIT), RoleRequest("detail", source="map", path=map_path)]
    plain = layers.prepare_layers(settings(source, mask_path), requests)
    inverted = layers.prepare_layers(settings(source, mask_path, invert=True), requests)
    assert np.array_equal(inverted.silhouette.material, ~plain.silhouette.material)
    rgb = layers.source_rgb(source, (40, 32), (40, 32))
    assert np.array_equal(inverted.structure.material, direct_canny(rgb, EXPLICIT) & ~mask)
    assert np.array_equal(inverted.detail.material, edge_map & ~mask & ~inverted.structure.material)


# --- External maps -------------------------------------------------------------


def test_map_must_match_oriented_source_dimensions(tmp_path):
    source = rectangle_source(tmp_path)
    map_path = binary(tmp_path, np.ones((32, 41), bool), "map.png")
    with pytest.raises(config.ConfigError, match="differ"):
        layers.prepare_layers(settings(source, full_mask(tmp_path)),
                              [RoleRequest("structure", source="map", path=map_path)])


def test_map_is_oriented_before_the_dimension_check(tmp_path):
    source = rectangle_source(tmp_path)
    upright = np.zeros((32, 40), bool)
    upright[:4, :] = True
    exif = Image.Exif()
    exif[274] = 6
    map_path = save(tmp_path / "map.png", np.where(np.rot90(upright), 255, 0), exif=exif)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)),
                                     [RoleRequest("structure", source="map", path=map_path)])
    assert np.array_equal(prepared.structure.material, upright)


def test_map_resizes_with_nearest_neighbour_to_processing_size(tmp_path):
    source = rectangle_source(tmp_path)
    rng = np.random.default_rng(5)
    edge_map = rng.random((32, 40)) < 0.3
    map_path = binary(tmp_path, edge_map, "map.png")
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path), max_res=20),
                                     [RoleRequest("structure", source="map", path=map_path)])
    assert np.array_equal(prepared.structure.material, raster._resize(edge_map, (20, 16)))


def test_map_alpha_is_ignored_with_a_warning(tmp_path):
    source = rectangle_source(tmp_path)
    rgba = np.zeros((32, 40, 4), np.uint8)
    rgba[:, :20, :3] = 255
    rgba[..., 3] = 0
    map_path = tmp_path / "map.png"
    Image.fromarray(rgba, "RGBA").save(map_path)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)),
                                     [RoleRequest("detail", source="map", path=map_path)])
    expected = np.zeros((32, 40), bool)
    expected[:, :20] = True
    assert np.array_equal(prepared.detail.material, expected)
    notices = [item for item in prepared.detail.diagnostics if item["code"] == "alpha_ignored"]
    assert len(notices) == 1 and str(map_path) in notices[0]["message"]


def test_constraint_alpha_is_ignored_with_a_warning(tmp_path):
    source = rectangle_source(tmp_path)
    rgba = np.full((32, 40, 4), 255, np.uint8)
    include = tmp_path / "include.png"
    Image.fromarray(rgba, "RGBA").save(include)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)),
                                     [RoleRequest("structure", canny=EXPLICIT, include_mask=include)])
    assert "alpha_ignored" in codes(prepared.structure.diagnostics)


def test_captured_input_bytes_replace_files_on_disk(tmp_path):
    source = rectangle_source(tmp_path)
    mask = full_mask(tmp_path)
    edge_map = np.zeros((32, 40), bool)
    edge_map[3:6, :] = True
    map_path = binary(tmp_path, edge_map, "map.png")
    exclude = binary(tmp_path, np.zeros((32, 40), bool), "exclude.png")
    captured = {path: Path(path).read_bytes() for path in (source, mask, map_path, exclude)}
    for path in (map_path, exclude):
        binary(tmp_path, np.ones((32, 40), bool), Path(path).name)
    rectangle_source(tmp_path, box=(0, 0, 0, 0))
    requests = [RoleRequest("structure", canny=EXPLICIT),
                RoleRequest("detail", source="map", path=map_path, exclude_mask=exclude)]
    prepared = layers.prepare_layers(settings(source, mask), requests, input_bytes=captured)
    assert prepared.structure.material.any()
    assert np.array_equal(prepared.detail.material, edge_map & ~prepared.structure.material)


# --- Constraints, width, and composition --------------------------------------


def test_exclusion_wins_over_inclusion(tmp_path):
    source = rectangle_source(tmp_path)
    edge_map = np.ones((32, 40), bool)
    include = np.zeros((32, 40), bool)
    include[:, :30] = True
    exclude = np.zeros((32, 40), bool)
    exclude[:, 20:] = True
    request = RoleRequest("detail", source="map", path=binary(tmp_path, edge_map, "map.png"),
                          include_mask=binary(tmp_path, include, "in.png"),
                          exclude_mask=binary(tmp_path, exclude, "ex.png"))
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])
    assert np.array_equal(prepared.detail.material, include & ~exclude)


def test_width_absent_keeps_the_input_band(tmp_path):
    source = rectangle_source(tmp_path)
    edge_map = np.zeros((32, 40), bool)
    edge_map[10, 5:35] = True
    edge_map[20:23, 5:35] = True
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)),
                                     [RoleRequest("structure", source="map", path=binary(tmp_path, edge_map, "map.png"))])
    assert np.array_equal(prepared.structure.material, edge_map)
    assert prepared.structure.expansion is None


def test_role_widths_are_independent_and_report_effective_expansion(tmp_path):
    source = rectangle_source(tmp_path)
    line = np.zeros((32, 40), bool)
    line[16, :] = True
    map_path = binary(tmp_path, line, "map.png")
    requests = [RoleRequest("structure", source="map", path=map_path, width_mm=3.0),
                RoleRequest("detail", source="map", path=map_path, width_mm=7.0)]
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), requests)
    assert prepared.structure.expansion == raster.BandExpansion(3.0, 1, 3, 3.0, 1.0)
    assert prepared.detail.expansion == raster.BandExpansion(7.0, 3, 7, 7.0, 3.0)
    assert prepared.structure.material.sum(axis=0).tolist() == [3] * 40
    # Detail loses the three structure rows. Two rows remain on each side.
    assert prepared.detail.material.sum(axis=0).tolist() == [4] * 40


def test_width_below_four_pixels_warns_with_the_role_name(tmp_path):
    source = rectangle_source(tmp_path)
    line = np.zeros((32, 40), bool)
    line[16, :] = True
    map_path = binary(tmp_path, line, "map.png")
    requests = [RoleRequest("structure", source="map", path=map_path, width_mm=3.0),
                RoleRequest("detail", source="map", path=map_path, width_mm=4.0)]
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), requests)
    warning = [item for item in prepared.structure.diagnostics if item["code"] == "requested_width_below_four_pixels"]
    assert len(warning) == 1 and warning[0]["role"] == "structure" and "structure" in warning[0]["message"]
    assert "requested_width_below_four_pixels" not in codes(prepared.detail.diagnostics)


def test_constraints_apply_again_after_expansion(tmp_path):
    source = rectangle_source(tmp_path)
    line = np.zeros((32, 40), bool)
    line[16, :] = True
    exclude = np.zeros((32, 40), bool)
    exclude[17:, :] = True
    request = RoleRequest("structure", source="map", path=binary(tmp_path, line, "map.png"), width_mm=5.0,
                          exclude_mask=binary(tmp_path, exclude, "ex.png"))
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])
    expected = np.zeros((32, 40), bool)
    expected[14:17, :] = True
    assert np.array_equal(prepared.structure.material, expected)


def test_roles_are_clipped_to_the_silhouette(tmp_path):
    source = rectangle_source(tmp_path)
    silhouette = np.zeros((32, 40), bool)
    silhouette[4:28, 4:36] = True
    request = RoleRequest("detail", source="map", path=binary(tmp_path, np.ones((32, 40), bool), "map.png"),
                          width_mm=5.0)
    prepared = layers.prepare_layers(settings(source, binary(tmp_path, silhouette, "mask.png")), [request])
    assert np.array_equal(prepared.detail.material, silhouette)


def test_final_structure_is_removed_from_final_detail(tmp_path):
    source = rectangle_source(tmp_path)
    structure = np.zeros((32, 40), bool)
    structure[16, :] = True
    detail = np.zeros((32, 40), bool)
    detail[12:21, :] = True
    requests = [RoleRequest("structure", source="map", path=binary(tmp_path, structure, "s.png"), width_mm=3.0),
                RoleRequest("detail", source="map", path=binary(tmp_path, detail, "d.png"))]
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), requests)
    expected_structure = np.zeros((32, 40), bool)
    expected_structure[15:18, :] = True
    assert np.array_equal(prepared.structure.material, expected_structure)
    assert np.array_equal(prepared.detail.material, detail & ~expected_structure)


def test_empty_optional_role_stays_selected_and_is_reported(tmp_path):
    source = rectangle_source(tmp_path)
    request = RoleRequest("detail", source="map", path=binary(tmp_path, np.zeros((32, 40), bool), "map.png"))
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])
    assert prepared.roles == ("silhouette", "detail")
    assert not prepared.detail.material.any()
    empty = [item for item in prepared.detail.diagnostics if item["code"] == "role_empty"]
    assert len(empty) == 1 and empty[0]["role"] == "detail"


def test_final_masks_are_read_only(tmp_path):
    source = rectangle_source(tmp_path)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)),
                                     [RoleRequest("structure", canny=EXPLICIT), RoleRequest("detail", canny=EXPLICIT)])
    for material in (prepared.silhouette.material, prepared.structure.material, prepared.detail.material):
        assert not material.flags.writeable
        with pytest.raises(ValueError):
            material[0, 0] = True


def test_omitted_canny_values_use_separate_role_defaults(tmp_path):
    defaults = layers.DEFAULT_CANNY
    assert set(defaults) == {"structure", "detail"}
    assert defaults["structure"] != defaults["detail"]
    for spec in defaults.values():
        assert None not in (spec.low_threshold, spec.high_threshold, spec.blur)
    source = rectangle_source(tmp_path)
    requests = [RoleRequest("structure"), RoleRequest("detail", canny=CannySpec(blur=0))]
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), requests)
    assert prepared.structure.canny == defaults["structure"]
    assert prepared.detail.canny == CannySpec(defaults["detail"].low_threshold, defaults["detail"].high_threshold, 0)


# --- Gap closing --------------------------------------------------------------


def gap_band(gap, *, height=9, left=10, right=10):
    material = np.zeros((height, left + gap + right), bool)
    material[:, :left] = True
    material[:, left + gap:] = True
    return material


def closes(material, gap, request_px, left=10):
    closed, _ = layers.close_gaps(material, float(request_px), 1.0)
    return bool(closed[material.shape[0] // 2, left:left + gap].all())


def test_gap_closing_is_disabled_by_default(tmp_path):
    source = rectangle_source(tmp_path)
    edge_map = np.zeros((32, 40), bool)
    edge_map[16, :19] = True
    edge_map[16, 20:] = True
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)),
                                     [RoleRequest("structure", source="map", path=binary(tmp_path, edge_map, "m.png"))])
    assert np.array_equal(prepared.structure.material, edge_map)
    assert prepared.structure.gap_closing is None


def test_gap_closing_closes_a_gap_when_selected(tmp_path):
    source = rectangle_source(tmp_path)
    edge_map = np.zeros((32, 40), bool)
    edge_map[14:19, :19] = True
    edge_map[14:19, 20:] = True
    request = RoleRequest("structure", source="map", path=binary(tmp_path, edge_map, "m.png"), gap_close_mm=1.0)
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])
    assert prepared.structure.material[16, 19]
    assert prepared.structure.gap_closing == layers.GapClosing(1.0, 1.0, 1, 3, 2.0)
    info = [item for item in prepared.structure.diagnostics if item["code"] == "gap_closing"]
    assert len(info) == 1
    assert {key: info[0][key] for key in ("role", "requested_gap_mm", "gap_px", "radius_px", "kernel_size_px",
                                          "achieved_gap_mm")} == {
        "role": "structure", "requested_gap_mm": 1.0, "gap_px": 1.0, "radius_px": 1, "kernel_size_px": 3,
        "achieved_gap_mm": 2.0}


@pytest.mark.parametrize("request_px, radius", [(0.3, 1), (1, 1), (2, 1), (3, 2), (4, 2)])
def test_requested_gap_maps_to_radius(request_px, radius):
    _, record = layers.close_gaps(gap_band(1), float(request_px), 1.0)
    assert record.radius_px == radius
    assert record.kernel_size_px == 2 * radius + 1
    assert record.achieved_gap_mm == 2 * radius


@pytest.mark.parametrize("request_px", [0.3, 1, 2])
def test_radius_one_closes_two_pixels_and_leaves_three_open(request_px):
    assert closes(gap_band(1), 1, request_px)
    assert closes(gap_band(2), 2, request_px)
    assert not closes(gap_band(3), 3, request_px)


@pytest.mark.parametrize("request_px", [3, 4])
def test_radius_two_closes_four_pixels_and_leaves_five_open(request_px):
    assert closes(gap_band(4), 4, request_px)
    assert not closes(gap_band(5), 5, request_px)


def test_gap_closing_keeps_material_at_the_canvas_edge():
    material = np.zeros((12, 12), bool)
    material[:, :2] = True
    material[0, :] = True
    material[5, 9:] = True
    for request_px in (1, 4, 30):
        closed, _ = layers.close_gaps(material, float(request_px), 1.0)
        assert (closed | ~material).all()


def square_closing(material, radius):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.morphologyEx(material.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def test_bounded_closing_equals_square_closing():
    rng = np.random.default_rng(11)
    for shape in [(9, 13), (17, 5), (1, 20), (20, 1), (16, 16)]:
        for radius in range(1, max(shape) + 4):
            for density in (0.05, 0.3):
                material = rng.random(shape) < density
                closed, record = layers.close_gaps(material, 2.0 * radius, 1.0)
                assert record.radius_px == radius
                assert np.array_equal(closed, square_closing(material, radius)), (shape, radius)


def test_a_million_pixel_gap_request_uses_bounded_kernels():
    material = np.zeros((32, 32), bool)
    material[4:8, 4:8] = True
    material[20:24, 20:28] = True
    tracemalloc.start()
    try:
        closed, record = layers.close_gaps(material, 1_000_000.0, 1.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert record.radius_px == 500_000
    assert record.kernel_size_px == 1_000_001
    assert closed.all()
    assert peak < 1_000_000


def test_nonfinite_gap_pixels_are_rejected():
    with pytest.raises(config.ConfigError, match="gap"):
        layers.close_gaps(gap_band(1), 1e300, 1e-10)


def test_second_exclusion_removes_a_gap_bridge(tmp_path):
    source = rectangle_source(tmp_path)
    edge_map = np.zeros((32, 40), bool)
    edge_map[14:19, :18] = True
    edge_map[14:19, 21:] = True
    exclude = np.zeros((32, 40), bool)
    exclude[:, 18:21] = True
    request = RoleRequest("structure", source="map", path=binary(tmp_path, edge_map, "m.png"), gap_close_mm=3.0,
                          exclude_mask=binary(tmp_path, exclude, "ex.png"))
    prepared = layers.prepare_layers(settings(source, full_mask(tmp_path)), [request])
    assert not prepared.structure.material[:, 18:21].any()
    assert np.array_equal(prepared.structure.material, edge_map)
