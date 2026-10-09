"""S3.1 image-mode silhouette methods. STABL-memwrtos.

Plan: docs/superpowers/plans/2026-10-08-vector-map-sprint-3.md, decision 1 and S3.1.
- Image mode requires exactly one method: --mask, --alpha, or --threshold.
- mask, alpha, and threshold form one precedence group. A later source replaces the method.
- --invert reverses every image method. Selection precedes the common binary resize.
- Mask and edge modes reject --mask and --threshold with exit 2.
- An empty silhouette after constraints exits 1 before VTracer.
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vector_map_config as config  # noqa: E402
import vector_map_raster as raster  # noqa: E402

RESULT_KEYS = {"schema_version", "status", "artifacts", "counts", "diagnostics"}


def save(path, pixels, **kwargs):
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path, **kwargs)
    return path


def cli_layer(**values):
    """An explicit-CLI layer: every field present, None where the user gave nothing."""
    return {**dict.fromkeys(config.FIELDS), **values}


def image_settings(*layers):
    base = {"input": Path("a.png"), "input_kind": "image", "width_mm": 20.0}
    return config.resolve(config.DEFAULTS, base, *layers)


def prepare(source, **values):
    return raster.prepare(config.resolve(
        config.DEFAULTS, {"input": source, "input_kind": "image", "width_mm": 20.0}, cli_layer(**values),
    ))


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "vector_map.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


def payload(result, expected=0):
    assert result.returncode == expected, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, result.stdout
    value = json.loads(lines[0])
    assert set(value) == RESULT_KEYS
    return value


def codes(items):
    return {item.get("code") for item in items}


def square_on_white(tmp_path, name="photo.png"):
    """Dark square on a light background, RGB. Luminance: background 230, square 20."""
    pixels = np.full((20, 24, 3), 230, np.uint8)
    pixels[5:15, 6:18] = 20
    return save(tmp_path / name, pixels)


def square_mask():
    expected = np.zeros((20, 24), bool)
    expected[5:15, 6:18] = True
    return expected


# --- Method selection and precedence ----------------------------------------


def test_image_mode_without_method_exits_2_with_instruction():
    with pytest.raises(config.ConfigError) as error:
        image_settings(cli_layer())
    message = str(error.value)
    for name in ("--mask", "--alpha", "--threshold"):
        assert name in message


def test_cli_threshold_replaces_recipe_alpha():
    settings = image_settings({"alpha": True}, cli_layer(threshold=200))
    assert settings.threshold == 200
    assert settings.alpha is False
    assert settings.mask is None


def test_cli_mask_replaces_recipe_threshold():
    settings = image_settings({"threshold": 90}, cli_layer(mask=Path("m.png")))
    assert settings.mask == Path("m.png")
    assert settings.threshold is None


def test_no_alpha_clears_inherited_alpha_and_leaves_no_method():
    with pytest.raises(config.ConfigError, match="--threshold"):
        image_settings({"alpha": True}, cli_layer(alpha=False))


@pytest.mark.parametrize("recipe, field, value", [
    ({"mask": Path("m.png")}, "mask", Path("m.png")),
    ({"threshold": 40}, "threshold", 40),
])
def test_no_alpha_keeps_recipe_mask_or_threshold(recipe, field, value):
    settings = image_settings(recipe, cli_layer(alpha=False))
    assert getattr(settings, field) == value
    assert settings.alpha is False


@pytest.mark.parametrize("layer", [
    {"alpha": True, "threshold": 10},
    {"mask": Path("m.png"), "alpha": True},
    {"mask": Path("m.png"), "threshold": 10},
])
def test_two_methods_from_one_source_exit_2(layer):
    with pytest.raises(config.ConfigError, match="one silhouette method"):
        image_settings(cli_layer(**layer))


@pytest.mark.parametrize("value", [-1, 256, True, 12.5])
def test_threshold_outside_integer_range_rejected(value):
    with pytest.raises(config.ConfigError, match="threshold must be an integer from 0 to 255"):
        image_settings(cli_layer(threshold=value))


def test_recipe_threshold_must_be_an_integer(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "threshold": "128"}))
    with pytest.raises(config.ConfigError, match="Recipe field threshold must be an integer"):
        config.load_recipe(recipe)


def test_recipe_threshold_decodes(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "threshold": 128}))
    assert config.load_recipe(recipe)["threshold"] == 128


@pytest.mark.parametrize("kind", ["mask", "edges"])
@pytest.mark.parametrize("field, value", [("mask", Path("m.png")), ("threshold", 128)])
def test_standalone_modes_reject_mask_and_threshold_settings(kind, field, value):
    base = {"input": Path("a.png"), "input_kind": kind, "width_mm": 20.0}
    with pytest.raises(config.ConfigError, match="input_kind image"):
        config.resolve(config.DEFAULTS, base, cli_layer(**{field: value}))


# --- Standalone CLI rejection: exit 2, one JSON result, no output ------------


@pytest.mark.parametrize("kind", ["mask", "edges"])
@pytest.mark.parametrize("source", ["cli", "recipe"])
@pytest.mark.parametrize("field", ["mask", "threshold"])
def test_standalone_cli_rejects_mask_and_threshold(tmp_path, kind, source, field):
    pixels = np.zeros((20, 20), np.uint8)
    pixels[5:15, 5:15] = 255
    image = save(tmp_path / "in.png", pixels)
    mask = save(tmp_path / "m.png", pixels)
    dest = tmp_path / "out.svg"
    value = {"mask": "m.png", "threshold": 128}[field]
    extra = []
    if source == "cli":
        extra = [f"--{field}", mask if field == "mask" else value]
    else:
        recipe = tmp_path / "recipe.json"
        recipe.write_text(json.dumps({"schema_version": 1, field: value}))
        extra = ["--recipe", recipe]
    result = run(image, dest, "--input-kind", kind, "--width-mm", 20, *extra, "--json")
    value = payload(result, 2)
    assert value["status"] == "invalid"
    assert "input_kind image" in value["diagnostics"][0]["message"]
    assert not dest.exists()
    assert not dest.with_suffix(".vector.json").exists()


# --- Raster selection --------------------------------------------------------


def test_threshold_selects_luminance_at_or_above_value(tmp_path):
    source = save(tmp_path / "steps.png", [[0, 99, 100, 255]])
    assert prepare(source, threshold=100).material.tolist() == [[False, False, True, True]]


def test_threshold_polarity_with_invert(tmp_path):
    source = square_on_white(tmp_path)
    np.testing.assert_array_equal(prepare(source, threshold=128).material, ~square_mask())
    np.testing.assert_array_equal(prepare(source, threshold=128, invert=True).material, square_mask())


def test_threshold_runs_before_the_common_resize(tmp_path):
    # Row luminance 100 and 160 at full size. A resize before threshold would blend columns.
    pixels = np.zeros((4, 8), np.uint8)
    pixels[:, 0::2] = 100
    pixels[:, 1::2] = 160
    source = save(tmp_path / "stripes.png", pixels)
    result = prepare(source, threshold=150, max_res=4)
    assert result.processed_size == (4, 2)
    expected = np.asarray(Image.fromarray(pixels >= 150).resize((4, 2), Image.Resampling.NEAREST))
    np.testing.assert_array_equal(result.material, expected)


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_threshold_uses_oriented_source(tmp_path, orientation):
    pixels = np.zeros((6, 10), np.uint8)
    pixels[1:4, 2:5] = 255
    exif = Image.Exif()
    exif[274] = orientation
    source = save(tmp_path / "source.png", pixels, exif=exif)
    oriented = {3: np.rot90(pixels, 2), 6: np.rot90(pixels, -1), 8: np.rot90(pixels)}[orientation]
    result = prepare(source, threshold=128)
    np.testing.assert_array_equal(result.material, oriented >= 128)
    assert result.original_size == (10, 6)
    assert result.oriented_size == oriented.shape[::-1]


def test_alpha_method_selects_source_alpha(tmp_path):
    pixels = np.zeros((20, 24, 4), np.uint8)
    pixels[..., :3] = 200
    pixels[5:15, 6:18, 3] = 255
    source = save(tmp_path / "rgba.png", pixels)
    np.testing.assert_array_equal(prepare(source, alpha=True).material, square_mask())
    np.testing.assert_array_equal(prepare(source, alpha=True, invert=True).material, ~square_mask())


def test_alpha_method_without_alpha_channel_exits_2(tmp_path):
    with pytest.raises(config.ConfigError, match="alpha"):
        prepare(square_on_white(tmp_path), alpha=True)


def test_mask_method_selects_white_mask_luminance(tmp_path):
    source = square_on_white(tmp_path)
    mask = save(tmp_path / "mask.png", square_mask().astype(np.uint8) * 255)
    result = prepare(source, mask=mask)
    np.testing.assert_array_equal(result.material, square_mask())
    assert "alpha_ignored" not in codes(result.diagnostics)
    np.testing.assert_array_equal(prepare(source, mask=mask, invert=True).material, ~square_mask())


def test_mask_method_orients_mask_independently(tmp_path):
    source = square_on_white(tmp_path)
    exif = Image.Exif()
    exif[274] = 3
    rotated = np.rot90(square_mask().astype(np.uint8) * 255, 2)
    mask = save(tmp_path / "mask.png", rotated, exif=exif)
    np.testing.assert_array_equal(prepare(source, mask=mask).material, square_mask())


def test_mask_dimension_mismatch_exits_2(tmp_path):
    source = square_on_white(tmp_path)
    mask = save(tmp_path / "mask.png", np.full((10, 12), 255))
    with pytest.raises(config.ConfigError, match="dimensions"):
        prepare(source, mask=mask)


def test_mask_file_alpha_ignored_with_warning(tmp_path):
    source = square_on_white(tmp_path)
    pixels = np.zeros((20, 24, 4), np.uint8)
    pixels[5:15, 6:18, :3] = 255
    pixels[..., 3] = 255
    pixels[0:3, 0:3, 3] = 0
    mask = save(tmp_path / "mask.png", pixels)
    result = prepare(source, mask=mask)
    np.testing.assert_array_equal(result.material, square_mask())
    warnings = [item for item in result.diagnostics if item.get("code") == "alpha_ignored"]
    assert len(warnings) == 1
    assert str(mask) in warnings[0]["message"]


def test_source_alpha_does_not_warn_with_mask_method(tmp_path):
    pixels = np.full((20, 24, 4), 255, np.uint8)
    source = save(tmp_path / "rgba.png", pixels)
    mask = save(tmp_path / "mask.png", square_mask().astype(np.uint8) * 255)
    assert "alpha_ignored" not in codes(prepare(source, mask=mask).diagnostics)


def test_source_alpha_warns_with_threshold_method(tmp_path):
    pixels = np.full((20, 24, 4), 255, np.uint8)
    pixels[5:15, 6:18, :3] = 0
    source = save(tmp_path / "rgba.png", pixels)
    result = prepare(source, threshold=128, invert=True)
    np.testing.assert_array_equal(result.material, square_mask())
    assert "alpha_ignored" in codes(result.diagnostics)


def test_constraints_apply_after_selection(tmp_path):
    source = square_on_white(tmp_path)
    exclude = np.zeros((20, 24), np.uint8)
    exclude[:, :12] = 255
    exclude_path = save(tmp_path / "exclude.png", exclude)
    result = prepare(source, threshold=128, invert=True, exclude_mask=exclude_path)
    expected = square_mask()
    expected[:, :12] = False
    np.testing.assert_array_equal(result.material, expected)


def test_empty_silhouette_after_constraints_fails_before_tracing(tmp_path):
    source = square_on_white(tmp_path)
    exclude = save(tmp_path / "exclude.png", np.full((20, 24), 255))
    with pytest.raises(ValueError, match="silhouette is empty"):
        prepare(source, threshold=128, invert=True, exclude_mask=exclude)


# --- End-to-end CLI through real VTracer -------------------------------------


@pytest.mark.parametrize("method", ["threshold", "mask", "alpha"])
def test_image_conversion_through_vtracer(tmp_path, method):
    if method == "alpha":
        pixels = np.full((20, 24, 4), 200, np.uint8)
        pixels[..., 3] = 0
        pixels[5:15, 6:18, 3] = 255
        source = save(tmp_path / "photo.png", pixels)
        extra = ["--alpha"]
    else:
        source = square_on_white(tmp_path)
        if method == "mask":
            mask = save(tmp_path / "mask.png", square_mask().astype(np.uint8) * 255)
            extra = ["--mask", mask]
        else:
            extra = ["--threshold", 128, "--invert"]
    dest = tmp_path / "out.svg"
    result = run(source, dest, "--input-kind", "image", "--width-mm", 24, *extra, "--json")
    value = payload(result)
    assert value["status"] == "converted"
    assert value["counts"]["paths"] >= 1
    root = ET.parse(dest).getroot()
    assert root.get("width") == "24mm"
    assert root.get("height") == "20mm"
    assert root.get("viewBox") == "0 0 24 20"
    manifest = json.loads(dest.with_suffix(".vector.json").read_text())
    assert manifest["preparation"]["input_kind"] == "image"
    assert manifest["preparation"]["silhouette"]["method"] == method
    assert manifest["layers"] == [{"id": "silhouette", "height_mm": None}]
    roles = {item["role"] for item in manifest["inputs"]}
    assert ("mask" in roles) is (method == "mask")


def test_cli_threshold_replaces_recipe_alpha_end_to_end(tmp_path):
    source = square_on_white(tmp_path)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "alpha": True}))
    dest = tmp_path / "out.svg"
    result = run(source, dest, "--input-kind", "image", "--width-mm", 24, "--recipe", recipe,
                 "--threshold", 128, "--invert", "--json")
    payload(result)
    manifest = json.loads(dest.with_suffix(".vector.json").read_text())
    assert manifest["preparation"]["silhouette"] == {"method": "threshold", "mask": None, "threshold": 128}


def test_cli_without_method_exits_2_without_output(tmp_path):
    dest = tmp_path / "out.svg"
    result = run(square_on_white(tmp_path), dest, "--input-kind", "image", "--width-mm", 24, "--json")
    value = payload(result, 2)
    assert "--threshold" in value["diagnostics"][0]["message"]
    assert not dest.exists()


def test_cli_empty_silhouette_exits_1_before_vtracer(tmp_path):
    dest = tmp_path / "out.svg"
    source = save(tmp_path / "white.png", np.full((20, 24), 255))
    result = run(source, dest, "--input-kind", "image", "--width-mm", 24, "--threshold", 128, "--invert", "--json")
    value = payload(result, 1)
    message = value["diagnostics"][0]["message"]
    assert "silhouette is empty" in message
    assert "VTracer" not in message
    assert "tracing" not in result.stderr
    assert not dest.exists()


def test_mask_destination_alias_refused(tmp_path):
    source = square_on_white(tmp_path)
    mask = save(tmp_path / "mask.png", square_mask().astype(np.uint8) * 255)
    dest = tmp_path / "out.svg"
    alias = dest.with_suffix(".preview.png")
    alias.hardlink_to(mask)
    result = run(source, dest, "--input-kind", "image", "--width-mm", 24, "--mask", mask,
                 "--preview", "--overwrite", "--json")
    value = payload(result, 2)
    assert "mask" in value["diagnostics"][0]["message"]
    assert not dest.exists()
