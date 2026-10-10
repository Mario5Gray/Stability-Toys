"""S2.2 CLI facade and configuration model (STABL-ascsgqha).

Walking skeleton, design A with the 2026-10-06 review corrections:
- One real mask conversion: luminance >= 128, optional --invert, real VTracer.
- The SVG root states the physical size in mm with the matching pixel viewBox.
- Precedence: defaults, preset, recipe, explicit CLI. None means "not supplied".
- Deferred preparation fails with exit 2 and names the owning task.
- --json emits exactly one result object, also for argparse and processing failures.
"""

import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "tests" / "fixtures" / "vector_map"
sys.path.insert(0, str(SCRIPTS))

import vector_map_config as config  # noqa: E402
import vector_map_raster as raster  # noqa: E402
import vector_map_svg as svg_io  # noqa: E402

SVG_NS = "{http://www.w3.org/2000/svg}"
RESULT_KEYS = {"schema_version", "status", "artifacts", "counts", "diagnostics"}
S23 = "S2.3 (STABL-vjpnctjh)"


def resolve(*layers):
    return config.resolve(config.DEFAULTS, *layers)


def cli_layer(**values):
    """An explicit-CLI layer: every field present, None where the user gave nothing."""
    return {**dict.fromkeys(config.FIELDS), **values}


# --- Precedence --------------------------------------------------------------


def test_absent_cli_values_keep_recipe_values():
    settings = resolve({}, {"input": Path("a.png"), "input_kind": "mask", "width_mm": 80.0, "invert": True}, cli_layer())
    assert settings.invert is True
    assert settings.width_mm == 80.0
    assert settings.input == Path("a.png")


def test_explicit_cli_false_overrides_recipe_true():
    settings = resolve({}, {"input": Path("a.png"), "input_kind": "mask", "width_mm": 80.0, "invert": True}, cli_layer(invert=False))
    assert settings.invert is False


def test_later_layers_win_in_order():
    base = {"input": Path("a.png"), "input_kind": "mask"}
    preset = {"width_mm": 10.0, "invert": True}
    recipe = {"width_mm": 20.0}
    settings = resolve(preset, {**base, **recipe}, cli_layer())
    assert (settings.width_mm, settings.invert) == (20.0, True)
    settings = resolve(preset, {**base, **recipe}, cli_layer(width_mm=30.0))
    assert settings.width_mm == 30.0


def test_higher_layer_width_clears_lower_layer_height():
    recipe = {"input": Path("a.png"), "input_kind": "mask", "height_mm": 50.0}
    settings = resolve({}, recipe, cli_layer(width_mm=100.0))
    assert (settings.width_mm, settings.height_mm) == (100.0, None)


def test_higher_layer_height_clears_lower_layer_width():
    recipe = {"input": Path("a.png"), "input_kind": "mask", "width_mm": 100.0}
    settings = resolve({}, recipe, cli_layer(height_mm=50.0))
    assert (settings.width_mm, settings.height_mm) == (None, 50.0)


def test_both_dimensions_in_one_layer_are_rejected():
    with pytest.raises(config.ConfigError, match="exactly one"):
        resolve({}, {}, cli_layer(input=Path("a.png"), input_kind="mask", width_mm=1.0, height_mm=1.0))


def test_missing_dimension_is_rejected():
    with pytest.raises(config.ConfigError, match="exactly one"):
        resolve({}, {}, cli_layer(input=Path("a.png"), input_kind="mask"))


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_dimension_must_be_positive_and_finite(value):
    with pytest.raises(config.ConfigError, match="width_mm"):
        resolve({}, {}, cli_layer(input=Path("a.png"), input_kind="mask", width_mm=value))


def test_missing_input_kind_is_rejected():
    with pytest.raises(config.ConfigError, match="input_kind"):
        resolve({}, {}, cli_layer(input=Path("a.png"), width_mm=1.0))


def test_missing_source_is_rejected():
    with pytest.raises(config.ConfigError, match="source"):
        resolve({}, {}, cli_layer(input_kind="mask", width_mm=1.0))


def test_vtracer_controls_merge_by_name():
    settings = resolve({}, {"input": Path("a.png"), "input_kind": "mask", "width_mm": 1.0, "vtracer": {"filter_speckle": 8}}, cli_layer())
    assert settings.vtracer == {"mode": "polygon", "filter_speckle": 8}


def test_vtracer_controls_from_lower_layers_survive_a_higher_layer():
    base = {"input": Path("a.png"), "input_kind": "mask", "width_mm": 1.0}
    settings = resolve({"vtracer": {"filter_speckle": 8}}, {**base, "vtracer": {"mode": "polygon"}}, cli_layer())
    assert settings.vtracer == {"mode": "polygon", "filter_speckle": 8}


def test_vtracer_values_are_checked_after_precedence():
    recipe = {"input": Path("a.png"), "input_kind": "mask", "width_mm": 1.0, "vtracer": {"filter_speckle": 999}}
    with pytest.raises(config.ConfigError, match="filter_speckle"):
        resolve({}, recipe, cli_layer())


def test_settings_are_frozen():
    settings = resolve({}, {}, cli_layer(input=Path("a.png"), input_kind="mask", width_mm=1.0))
    with pytest.raises(AttributeError):
        settings.width_mm = 2.0


# --- Silhouette mask outside image mode: exit 2 (S3.1, STABL-memwrtos) -------


def test_silhouette_mask_requires_image_mode():
    layer = cli_layer(input=Path("a.png"), input_kind="mask", width_mm=1.0, mask=Path("m.png"))
    with pytest.raises(config.ConfigError, match="input_kind image"):
        resolve({}, {}, layer)


def test_explicit_no_alpha_is_accepted():
    settings = resolve({}, {}, cli_layer(input=Path("a.png"), input_kind="mask", width_mm=1.0, alpha=False))
    assert settings.alpha is False


# --- Recipe decoding ---------------------------------------------------------


def write_recipe(directory, payload):
    path = directory / "recipe.json"
    path.write_text(json.dumps(payload))
    return path


def test_recipe_paths_resolve_relative_to_the_recipe_directory(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    recipe = config.load_recipe(write_recipe(sub, {"schema_version": 1, "input": "masks/a.png", "mask": "m.png"}))
    assert recipe["input"] == sub / "masks" / "a.png"
    assert recipe["mask"] == sub / "m.png"


def test_recipe_absolute_paths_stay_unchanged(tmp_path):
    recipe = config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, "input": "/abs/a.png"}))
    assert recipe["input"] == Path("/abs/a.png")


def test_recipe_omits_the_schema_version_from_settings_values(tmp_path):
    recipe = config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, "width_mm": 10}))
    assert recipe == {"width_mm": 10.0}


@pytest.mark.parametrize("version", [None, 2, "1", True, 1.0])
def test_recipe_requires_schema_version_1(tmp_path, version):
    payload = {} if version is None else {"schema_version": version}
    with pytest.raises(config.ConfigError, match="schema_version"):
        config.load_recipe(write_recipe(tmp_path, payload))


def test_recipe_rejects_unknown_fields(tmp_path):
    with pytest.raises(config.ConfigError, match="simplify_mm"):
        config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, "simplify_mm": 0.1}))


@pytest.mark.parametrize("name", ["corner_threshold", "path_precision", "colormode", "bogus"])
def test_recipe_rejects_unsupported_vtracer_control_names(tmp_path, name):
    with pytest.raises(config.ConfigError, match=name):
        config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, "vtracer": {name: 1}}))


def test_recipe_accepts_out_of_range_vtracer_values_until_resolution(tmp_path):
    recipe = config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, "vtracer": {"filter_speckle": 999}}))
    assert recipe["vtracer"] == {"filter_speckle": 999}


@pytest.mark.parametrize(
    "field, value",
    [
        ("width_mm", "100"),
        ("width_mm", True),
        ("invert", "yes"),
        ("input", 3),
        ("input_kind", 1),
        ("max_res", 1.5),
        ("vtracer", []),
    ],
)
def test_recipe_rejects_wrong_value_types(tmp_path, field, value):
    with pytest.raises(config.ConfigError, match=field):
        config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, field: value}))


def test_recipe_must_be_a_json_object(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text("[1]")
    with pytest.raises(config.ConfigError, match="object"):
        config.load_recipe(path)


def test_recipe_rejects_invalid_json(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text("{")
    with pytest.raises(config.ConfigError, match="JSON"):
        config.load_recipe(path)


def test_missing_recipe_is_a_configuration_error(tmp_path):
    with pytest.raises(config.ConfigError, match="recipe"):
        config.load_recipe(tmp_path / "absent.json")


# --- Physical canvas and SVG sizing ------------------------------------------


def test_width_sets_the_scale():
    canvas = config.physical_canvas(resolve({}, {}, cli_layer(input=Path("a"), input_kind="mask", width_mm=100.0)), 128, 64)
    assert (canvas.width_px, canvas.height_px) == (128, 64)
    assert (canvas.width_mm, canvas.height_mm) == (100.0, 50.0)
    assert canvas.mm_per_px == 100.0 / 128


def test_height_sets_the_scale():
    canvas = config.physical_canvas(resolve({}, {}, cli_layer(input=Path("a"), input_kind="mask", height_mm=32.0)), 128, 64)
    assert (canvas.width_mm, canvas.height_mm) == (64.0, 32.0)


def raw_svg(width=12, height=8):
    material = np.zeros((height, width), bool)
    material[2:6, 3:9] = True
    import vector_map_vtracer

    return vector_map_vtracer.trace_layer(material).svg


def test_sized_svg_states_mm_and_the_pixel_view_box():
    pytest.importorskip("vtracer")
    canvas = config.Canvas(width_px=12, height_px=8, width_mm=60.0, height_mm=40.0, mm_per_px=5.0)
    root = ET.fromstring(svg_io.size_svg(raw_svg(), canvas))
    assert root.get("width") == "60mm"
    assert root.get("height") == "40mm"
    assert root.get("viewBox") == "0 0 12 8"


def test_sized_svg_keeps_paths_and_transforms_unchanged():
    pytest.importorskip("vtracer")
    raw = raw_svg()
    canvas = config.Canvas(width_px=12, height_px=8, width_mm=60.0, height_mm=40.0, mm_per_px=5.0)
    before = [p.attrib for p in ET.fromstring(raw).iter(f"{SVG_NS}path")]
    after = [p.attrib for p in ET.fromstring(svg_io.size_svg(raw, canvas)).iter(f"{SVG_NS}path")]
    assert after == before
    assert any("transform" in attrib for attrib in after)


def test_sized_svg_keeps_the_default_svg_namespace():
    pytest.importorskip("vtracer")
    canvas = config.Canvas(width_px=12, height_px=8, width_mm=60.0, height_mm=40.0, mm_per_px=5.0)
    text = svg_io.size_svg(raw_svg(), canvas)
    assert "ns0:" not in text
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in text or 'xmlns="http://www.w3.org/2000/svg"' in text


def test_sized_svg_rejects_a_canvas_that_does_not_match_the_trace():
    pytest.importorskip("vtracer")
    canvas = config.Canvas(width_px=13, height_px=8, width_mm=65.0, height_mm=40.0, mm_per_px=5.0)
    with pytest.raises(ValueError, match="canvas"):
        svg_io.size_svg(raw_svg(), canvas)


def test_count_paths_uses_xml_path_elements():
    text = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2">'
        '<path d="M0,0 L1,0 L1,1 Z"/><g><path d="M0,0 L1,1 Z"/></g></svg>'
    )
    assert svg_io.count_paths(text) == 2


# --- Raster skeleton ---------------------------------------------------------


def write_gray(path, array, **save):
    Image.fromarray(np.asarray(array, np.uint8), mode="L").save(path, **save)
    return path


def test_mask_foreground_is_luminance_at_least_128(tmp_path):
    path = write_gray(tmp_path / "m.png", [[0, 127, 128, 255]])
    assert raster.prepare_mask(path, invert=False).tolist() == [[False, False, True, True]]


def test_invert_flips_the_foreground(tmp_path):
    path = write_gray(tmp_path / "m.png", [[0, 127, 128, 255]])
    assert raster.prepare_mask(path, invert=True).tolist() == [[True, True, False, False]]


def test_rgb_input_uses_luminance(tmp_path):
    path = tmp_path / "m.png"
    Image.fromarray(np.array([[[255, 255, 255], [0, 0, 0]]], np.uint8), mode="RGB").save(path)
    assert raster.prepare_mask(path, invert=False).tolist() == [[True, False]]


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_alpha_bearing_input_warns_when_luminance_used(tmp_path, mode):
    path = tmp_path / "m.png"
    Image.new(mode, (4, 4)).save(path)
    with pytest.warns(UserWarning, match="luminance used"):
        raster.prepare_mask(path, invert=False)


def test_palette_transparency_warns_when_luminance_used(tmp_path):
    path = tmp_path / "m.png"
    image = Image.new("P", (4, 4))
    image.info["transparency"] = 0
    image.save(path, transparency=0)
    with pytest.warns(UserWarning, match="luminance used"):
        raster.prepare_mask(path, invert=False)


def exif_jpeg(path, orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.new("L", (8, 4), 255).save(path, exif=exif)
    return path


def test_nonidentity_exif_orientation_is_applied(tmp_path):
    assert raster.prepare_mask(exif_jpeg(tmp_path / "m.jpg", 6), invert=False).shape == (8, 4)


def test_identity_exif_orientation_is_accepted(tmp_path):
    material = raster.prepare_mask(exif_jpeg(tmp_path / "m.jpg", 1), invert=False)
    assert material.shape == (4, 8)


def test_jpeg_without_exif_is_accepted(tmp_path):
    path = write_gray(tmp_path / "m.jpg", np.full((4, 8), 255))
    assert raster.prepare_mask(path, invert=False).all()


# --- CLI: real subprocess runs -----------------------------------------------


def run_cli(*args, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "vector_map.py"), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        env=env,
    )


def one_result(result):
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, result.stdout
    payload = json.loads(lines[0])
    assert set(payload) == RESULT_KEYS
    assert payload["schema_version"] == 1
    return payload


def test_help_exits_zero():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--input-kind" in result.stdout


def test_mask_conversion_writes_a_physical_svg(tmp_path):
    pytest.importorskip("vtracer")
    out = tmp_path / "donut.svg"
    result = run_cli(FIXTURES / "donut.png", out, "--input-kind", "mask", "--width-mm", 100)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr
    root = ET.parse(out).getroot()
    assert (root.get("width"), root.get("height"), root.get("viewBox")) == ("100mm", "100mm", "0 0 128 128")
    assert svg_io.count_paths(out.read_text()) >= 1


def test_height_mm_sizes_the_written_svg(tmp_path):
    pytest.importorskip("vtracer")
    out = tmp_path / "donut.svg"
    result = run_cli(FIXTURES / "donut.png", out, "--input-kind", "mask", "--height-mm", 32)
    assert result.returncode == 0, result.stderr
    root = ET.parse(out).getroot()
    assert (root.get("width"), root.get("height")) == ("32mm", "32mm")


def test_json_success_reports_the_published_svg_and_measured_paths(tmp_path):
    pytest.importorskip("vtracer")
    out = tmp_path / "donut.svg"
    result = run_cli(FIXTURES / "donut.png", out, "--input-kind", "mask", "--width-mm", 100, "--json")
    assert result.returncode == 0, result.stderr
    payload = one_result(result)
    assert payload["status"] == "converted"
    assert payload["artifacts"] == {"svg": str(out), "manifest": str(out.with_suffix(".vector.json"))}
    assert payload["counts"] == {"layers": 1, "paths": svg_io.count_paths(out.read_text())}
    assert all(item["level"] == "warning" for item in payload["diagnostics"])


def test_invert_changes_the_traced_material(tmp_path):
    pytest.importorskip("vtracer")
    plain, inverted = tmp_path / "plain.svg", tmp_path / "inverted.svg"
    assert run_cli(FIXTURES / "donut.png", plain, "--input-kind", "mask", "--width-mm", 10).returncode == 0
    assert run_cli(FIXTURES / "donut.png", inverted, "--input-kind", "mask", "--width-mm", 10, "--invert").returncode == 0
    assert plain.read_text() != inverted.read_text()


def test_json_on_argparse_failure_is_one_invalid_result(tmp_path):
    result = run_cli(FIXTURES / "donut.png", tmp_path / "o.svg", "--bogus", "--json")
    assert result.returncode == 2
    payload = one_result(result)
    assert payload["status"] == "invalid"
    assert payload["artifacts"] == {"svg": None}
    assert payload["counts"] == {"layers": 0, "paths": 0}
    assert payload["diagnostics"][0]["level"] == "error"
    assert "--bogus" in payload["diagnostics"][0]["message"]
    assert "--bogus" in result.stderr


def test_argparse_failure_without_json_keeps_stdout_empty(tmp_path):
    result = run_cli(FIXTURES / "donut.png", tmp_path / "o.svg", "--bogus")
    assert result.returncode == 2
    assert result.stdout == ""


def test_json_on_processing_failure_is_one_failed_result(tmp_path):
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image")
    out = tmp_path / "o.svg"
    result = run_cli(source, out, "--input-kind", "mask", "--width-mm", 10, "--json")
    assert result.returncode == 1
    payload = one_result(result)
    assert payload["status"] == "failed"
    assert payload["artifacts"] == {"svg": None}
    assert payload["counts"] == {"layers": 0, "paths": 0}
    assert not out.exists()


def test_missing_source_is_a_processing_failure(tmp_path):
    result = run_cli(tmp_path / "absent.png", tmp_path / "o.svg", "--input-kind", "mask", "--width-mm", 10)
    assert result.returncode == 1
    assert "absent.png" in result.stderr


def test_empty_mask_result_is_rejected(tmp_path):
    pytest.importorskip("vtracer")
    source = write_gray(tmp_path / "empty.png", np.zeros((16, 16)))
    out = tmp_path / "o.svg"
    result = run_cli(source, out, "--input-kind", "mask", "--width-mm", 10, "--json")
    assert result.returncode == 1
    payload = one_result(result)
    assert payload["status"] == "failed"
    assert "empty" in payload["diagnostics"][0]["message"]
    assert not out.exists()


def test_missing_vtracer_is_a_processing_failure_with_install_hint(tmp_path):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "vtracer.py").write_text("raise ImportError('blocked for test')\n")
    env = {**__import__("os").environ, "PYTHONPATH": str(shadow)}
    out = tmp_path / "o.svg"
    result = run_cli(FIXTURES / "donut.png", out, "--input-kind", "mask", "--width-mm", 10, env=env)
    assert result.returncode == 1
    assert "vtracer==0.6.15" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_mask_flag_outside_image_mode_exits_2(tmp_path):
    out = tmp_path / "o.svg"
    result = run_cli(FIXTURES / "donut.png", out, "--width-mm", 10, "--input-kind", "mask",
                     "--mask", "m.png", "--json")
    assert result.returncode == 2, result.stderr
    payload = one_result(result)
    assert payload["status"] == "invalid"
    assert "input_kind image" in payload["diagnostics"][0]["message"]
    assert not out.exists()


def test_alpha_bearing_source_converts_with_warning(tmp_path):
    source = tmp_path / "rgba.png"
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(source)
    result = run_cli(source, tmp_path / "o.svg", "--input-kind", "mask", "--width-mm", 10)
    assert result.returncode == 0
    assert "luminance used" in result.stderr


@pytest.mark.parametrize("dims", [[], ["--width-mm", "10", "--height-mm", "10"]])
def test_dimension_rule_exits_2(tmp_path, dims):
    result = run_cli(FIXTURES / "donut.png", tmp_path / "o.svg", "--input-kind", "mask", *dims)
    assert result.returncode == 2
    assert "exactly one" in result.stderr


def test_existing_destination_is_refused_without_overwrite(tmp_path):
    out = tmp_path / "o.svg"
    out.write_text("keep")
    result = run_cli(FIXTURES / "donut.png", out, "--input-kind", "mask", "--width-mm", 10)
    assert result.returncode == 2
    assert "--overwrite" in result.stderr
    assert out.read_text() == "keep"


def test_overwrite_replaces_the_destination(tmp_path):
    pytest.importorskip("vtracer")
    out = tmp_path / "o.svg"
    out.write_text("old")
    result = run_cli(FIXTURES / "donut.png", out, "--input-kind", "mask", "--width-mm", 10, "--overwrite")
    assert result.returncode == 0, result.stderr
    assert ET.parse(out).getroot().get("viewBox") == "0 0 128 128"


# --- CLI: positional grammar and recipes -------------------------------------


def test_recipe_supplies_source_and_settings(tmp_path):
    pytest.importorskip("vtracer")
    recipe_dir = tmp_path / "r"
    recipe_dir.mkdir()
    (recipe_dir / "mask.png").write_bytes((FIXTURES / "donut.png").read_bytes())
    recipe = write_recipe(recipe_dir, {"schema_version": 1, "input": "mask.png", "input_kind": "mask", "width_mm": 40})
    out = tmp_path / "o.svg"
    result = run_cli(out, "--recipe", recipe, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert ET.parse(out).getroot().get("width") == "40mm"


def test_cli_source_overrides_recipe_input(tmp_path):
    pytest.importorskip("vtracer")
    recipe = write_recipe(tmp_path, {"schema_version": 1, "input": "absent.png", "input_kind": "mask", "width_mm": 40})
    out = tmp_path / "o.svg"
    result = run_cli(FIXTURES / "donut.png", out, "--recipe", recipe)
    assert result.returncode == 0, result.stderr


def test_cli_width_clears_recipe_height(tmp_path):
    pytest.importorskip("vtracer")
    recipe = write_recipe(tmp_path, {"schema_version": 1, "input_kind": "mask", "height_mm": 40})
    out = tmp_path / "o.svg"
    result = run_cli(FIXTURES / "donut.png", out, "--recipe", recipe, "--width-mm", 64)
    assert result.returncode == 0, result.stderr
    assert ET.parse(out).getroot().get("width") == "64mm"


def test_cli_no_invert_overrides_recipe_invert(tmp_path):
    pytest.importorskip("vtracer")
    recipe = write_recipe(tmp_path, {"schema_version": 1, "input_kind": "mask", "width_mm": 10, "invert": True})
    via_recipe, via_cli = tmp_path / "a.svg", tmp_path / "b.svg"
    plain = tmp_path / "plain.svg"
    assert run_cli(FIXTURES / "donut.png", via_recipe, "--recipe", recipe).returncode == 0
    assert run_cli(FIXTURES / "donut.png", via_cli, "--recipe", recipe, "--no-invert").returncode == 0
    assert run_cli(FIXTURES / "donut.png", plain, "--input-kind", "mask", "--width-mm", 10).returncode == 0
    assert via_cli.read_text() == plain.read_text()
    assert via_recipe.read_text() != plain.read_text()


def test_one_path_without_recipe_exits_2(tmp_path):
    result = run_cli(tmp_path / "o.svg", "--input-kind", "mask", "--width-mm", 10)
    assert result.returncode == 2
    assert "source" in result.stderr


def test_one_path_with_recipe_but_no_recipe_input_exits_2(tmp_path):
    recipe = write_recipe(tmp_path, {"schema_version": 1, "input_kind": "mask", "width_mm": 10})
    out = tmp_path / "o.svg"
    result = run_cli(out, "--recipe", recipe)
    assert result.returncode == 2
    assert "source" in result.stderr
    assert not out.exists()


def test_three_paths_exit_2(tmp_path):
    result = run_cli("a.png", "b.png", tmp_path / "o.svg", "--input-kind", "mask", "--width-mm", 10)
    assert result.returncode == 2
    assert "at most two paths" in result.stderr


def test_recipe_vtracer_controls_reach_the_trace(tmp_path):
    """A 6x6 island survives filter_speckle 4 and is filtered at 10."""
    pytest.importorskip("vtracer")
    material = np.zeros((64, 64), np.uint8)
    material[16:48, 16:48] = 255
    material[2:8, 2:8] = 255
    source = write_gray(tmp_path / "island.png", material)
    counts = {}
    for speckle in (4, 10):
        recipe_dir = tmp_path / f"r{speckle}"
        recipe_dir.mkdir()
        recipe = write_recipe(recipe_dir, {"schema_version": 1, "input_kind": "mask", "width_mm": 10, "vtracer": {"filter_speckle": speckle}})
        result = run_cli(source, tmp_path / f"o{speckle}.svg", "--recipe", recipe, "--json")
        assert result.returncode == 0, result.stderr
        counts[speckle] = one_result(result)["counts"]["paths"]
    assert counts == {4: 2, 10: 1}


def test_invalid_recipe_exits_2_with_json(tmp_path):
    recipe = write_recipe(tmp_path, {"schema_version": 1, "corner_threshold": 60})
    result = run_cli(FIXTURES / "donut.png", tmp_path / "o.svg", "--recipe", recipe, "--json")
    assert result.returncode == 2
    payload = one_result(result)
    assert payload["status"] == "invalid"
    assert "corner_threshold" in payload["diagnostics"][0]["message"]


# --- Review fixes (e5d6d32 review) -------------------------------------------


def _alias(kind, target, link):
    if kind == "direct":
        return target
    if kind == "symlink":
        link.symlink_to(target)
    else:
        import os

        os.link(target, link)
    return link


@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
def test_destination_aliasing_the_source_is_refused_even_with_overwrite(tmp_path, kind):
    source = tmp_path / "mask.png"
    source.write_bytes((FIXTURES / "donut.png").read_bytes())
    before = source.read_bytes()
    destination = _alias(kind, source, tmp_path / "out.svg")
    result = run_cli(source, destination, "--input-kind", "mask", "--width-mm", 10, "--overwrite", "--json")
    assert result.returncode == 2, result.stderr
    payload = one_result(result)
    assert payload["status"] == "invalid"
    assert "source" in payload["diagnostics"][0]["message"]
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
def test_destination_aliasing_the_recipe_is_refused_even_with_overwrite(tmp_path, kind):
    recipe = write_recipe(tmp_path, {"schema_version": 1, "input_kind": "mask", "width_mm": 10})
    before = recipe.read_bytes()
    destination = _alias(kind, recipe, tmp_path / "out.svg")
    result = run_cli(FIXTURES / "donut.png", destination, "--recipe", recipe, "--overwrite", "--json")
    assert result.returncode == 2, result.stderr
    payload = one_result(result)
    assert payload["status"] == "invalid"
    assert "recipe" in payload["diagnostics"][0]["message"]
    assert recipe.read_bytes() == before


def test_recipe_with_invalid_utf8_is_a_configuration_error(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_bytes(b'{"schema_version": 1, "input": "\xff.png"}')
    with pytest.raises(config.ConfigError, match="UTF-8"):
        config.load_recipe(path)


def test_recipe_with_invalid_utf8_exits_2_with_json(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_bytes(b'{"schema_version": 1, "input": "\xff.png"}')
    result = run_cli(FIXTURES / "donut.png", tmp_path / "o.svg", "--recipe", path, "--json")
    assert result.returncode == 2
    assert one_result(result)["status"] == "invalid"


MALFORMED_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"><path d="M0,0 Z"'


@pytest.mark.parametrize("where", ["upstream", "adapter"])
def test_malformed_svg_is_a_processing_failure_with_one_result(tmp_path, monkeypatch, capsys, where):
    """upstream: VTracer output breaks the adapter check. adapter: the facade's own parse breaks."""
    pytest.importorskip("vtracer")
    import vector_map
    import vector_map_vtracer
    import vtracer

    if where == "upstream":
        monkeypatch.setattr(vtracer, "convert_raw_image_to_svg", lambda *_a, **_k: MALFORMED_SVG)
    else:
        monkeypatch.setattr(
            vector_map.adapter,
            "trace_layer",
            lambda *_a, **_k: vector_map_vtracer.TraceResult(MALFORMED_SVG, {}, "0.6.15"),
        )
    out = tmp_path / "o.svg"
    code = vector_map.main([str(FIXTURES / "donut.png"), str(out), "--input-kind", "mask", "--width-mm", "10", "--json"])
    captured = capsys.readouterr()
    assert code == 1
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "failed"
    assert "malformed SVG" in payload["diagnostics"][0]["message"]
    assert not out.exists()


def test_decompression_bomb_is_a_processing_failure_with_one_result(tmp_path, monkeypatch, capsys):
    """Pillow raises DecompressionBombError above 2 x MAX_IMAGE_PIXELS. It subclasses Exception only."""
    import vector_map

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    out = tmp_path / "o.svg"
    code = vector_map.main([str(FIXTURES / "donut.png"), str(out), "--input-kind", "mask", "--width-mm", "10", "--json"])
    captured = capsys.readouterr()
    assert code == 1
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "failed"
    assert payload["artifacts"] == {"svg": None}
    assert "pixel limit" in payload["diagnostics"][0]["message"]
    assert "Traceback" not in captured.err
    assert not out.exists()


# --- S2.5 SVG limits and inspection through the CLI (STABL-npoznayt) --------

from vector_map_svg import SvgLimits  # noqa: E402

LIMITS = ("max_svg_bytes", "max_paths", "max_path_commands")
BASE = {"input": Path("a.png"), "input_kind": "mask", "width_mm": 10.0}


def test_svg_limits_default_to_the_wrapper_policy():
    assert resolve(BASE, cli_layer()).svg_limits == SvgLimits()


@pytest.mark.parametrize("field", LIMITS)
def test_svg_limits_follow_precedence(field):
    assert getattr(resolve({**BASE, field: 5}, cli_layer()).svg_limits, field) == 5
    assert getattr(resolve({**BASE, field: 5}, cli_layer(**{field: 7})).svg_limits, field) == 7


@pytest.mark.parametrize("field", LIMITS)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_svg_limits_are_configuration_errors(field, value):
    with pytest.raises(config.ConfigError, match=field):
        resolve({**BASE, field: value}, cli_layer())


@pytest.mark.parametrize("field", LIMITS)
@pytest.mark.parametrize("value", [None, 1.5, True, "5"])
def test_recipe_svg_limits_must_be_integers(tmp_path, field, value):
    with pytest.raises(config.ConfigError, match=field):
        config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, field: value}))


def test_recipe_rejects_an_unknown_limit_name(tmp_path):
    with pytest.raises(config.ConfigError, match="max_svg_size"):
        config.load_recipe(write_recipe(tmp_path, {"schema_version": 1, "max_svg_size": 10}))


def test_svg_limits_never_become_vtracer_options():
    settings = resolve({**BASE, "max_paths": 5}, cli_layer())
    assert not set(LIMITS) & set(settings.vtracer)


@pytest.mark.parametrize("flag, field", [("--max-svg-bytes", "max_svg_bytes"), ("--max-paths", "max_paths"),
                                         ("--max-path-commands", "max_path_commands")])
@pytest.mark.parametrize("value", ["0", "-3"])
def test_invalid_limit_flag_is_one_invalid_result(tmp_path, flag, field, value):
    result = run_cli(FIXTURES / "donut.png", tmp_path / "o.svg", "--input-kind", "mask", "--width-mm", 10,
                     flag, value, "--json")
    assert result.returncode == 2
    payload = one_result(result)
    assert payload["status"] == "invalid"
    assert field in payload["diagnostics"][0]["message"]


def test_non_integer_limit_flag_is_a_usage_error(tmp_path):
    result = run_cli(FIXTURES / "donut.png", tmp_path / "o.svg", "--input-kind", "mask", "--width-mm", 10,
                     "--max-paths", "1.5", "--json")
    assert result.returncode == 2
    assert one_result(result)["status"] == "invalid"


def injected(body, size=128, prolog=""):
    return f'{prolog}<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">{body}</svg>'


TRIANGLE_PATH = '<path d="M0,0 L4,0 L0,3 Z"/>'


@pytest.fixture
def run_injected(tmp_path, monkeypatch, capsys):
    """In-process CLI run with the upstream SVG replaced. Records every upstream call."""
    pytest.importorskip("vtracer")
    import vector_map
    import vtracer

    def run(svg, *extra, json_mode=True):
        calls = []

        def upstream(img_bytes, **kwargs):
            calls.append(kwargs)
            return svg

        monkeypatch.setattr(vtracer, "convert_raw_image_to_svg", upstream)
        out = tmp_path / "o.svg"
        args = [str(FIXTURES / "donut.png"), str(out), "--input-kind", "mask", "--width-mm", "10", *extra]
        code = vector_map.main(args + (["--json"] if json_mode else []))
        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        payload = json.loads(lines[0]) if json_mode else None
        if json_mode:
            assert len(lines) == 1
        return code, payload, captured.err, out, calls

    return run


FAILURES = {
    "malformed": (injected(TRIANGLE_PATH)[:-3], (), "malformed SVG"),
    "viewBox repeated comma": (injected(TRIANGLE_PATH).replace('<svg ', '<svg viewBox="0,,0,128,128" '), (), "viewBox"),
    "deep groups": (injected('<g>' * 1200 + TRIANGLE_PATH + '</g>' * 1200), (),
                    "group nesting exceeds supported depth"),
    "dtd": (injected(TRIANGLE_PATH, prolog="<!DOCTYPE svg>"), (), "DTD"),
    "image": (injected(TRIANGLE_PATH + '<image href="x.png"/>'), (), "unsupported element image"),
    "curve": (injected('<path d="M0,0 C1,1 2,2 0,3 Z"/>'), (), "unsupported path"),
    "open": (injected('<path d="M0,0 L4,0 L0,3"/>'), (), "closed"),
    "degenerate only": (injected(TRIANGLE_PATH + '<path d="M1,1 Z"/>'), (),
                        "VTracer returned 1 paths containing only points or lines"),
    "empty": (injected(""), (), "empty"),
    "bytes": (injected(TRIANGLE_PATH), ("--max-svg-bytes", "50"), "bytes exceeds limit 50"),
    "paths": (injected(TRIANGLE_PATH * 3), ("--max-paths", "2"), "path count at least 3 exceeds limit 2"),
    "commands": (injected(TRIANGLE_PATH * 2), ("--max-path-commands", "7"),
                 "path command count at least 8 exceeds limit 7"),
    "normalized bytes": (injected(TRIANGLE_PATH), ("--max-svg-bytes", str(len(injected(TRIANGLE_PATH)))),
                         "Normalized SVG size"),
}


@pytest.mark.parametrize("svg, extra, match", FAILURES.values(), ids=FAILURES.keys())
def test_incompatible_output_fails_without_publishing_or_retrying(run_injected, svg, extra, match):
    code, payload, stderr, out, calls = run_injected(svg, *extra, "--overwrite")
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["artifacts"] == {"svg": None}
    assert payload["counts"] == {"layers": 0, "paths": 0}
    assert match in payload["diagnostics"][0]["message"]
    assert match in stderr
    assert len(calls) == 1
    assert calls[0]["mode"] == "polygon" and calls[0]["filter_speckle"] == 4
    assert not set(LIMITS) & set(calls[0])
    assert not out.exists()


def test_failure_keeps_an_existing_destination_under_overwrite(run_injected, tmp_path):
    (tmp_path / "o.svg").write_text("previous")
    code, _, _, out, _ = run_injected(injected(TRIANGLE_PATH * 3), "--max-paths", "2", "--overwrite")
    assert code == 1
    assert out.read_text() == "previous"


def test_limit_failure_suggests_upstream_settings(run_injected):
    _, payload, _, _, _ = run_injected(injected(TRIANGLE_PATH * 3), "--max-paths", "2")
    message = payload["diagnostics"][0]["message"]
    assert "--max-res" in message and "filter_speckle" in message and "0..128" in message


def degenerate_warnings(stderr):
    """Warning lines only. pytest tmp paths repeat the test name, which contains the word."""
    return [line for line in stderr.splitlines() if line.startswith("warning:") and "degenerate" in line]


MIXED = injected('<path d="M0,0 L4,0 L0,3 Z M1,1 Z M2,2 L3,2 Z"/>')


def test_degenerate_subpaths_publish_with_one_counted_warning(run_injected):
    code, payload, stderr, out, _ = run_injected(MIXED)
    assert code == 0
    assert payload["status"] == "converted"
    assert payload["counts"] == {"layers": 1, "paths": 1}
    warnings = [item for item in payload["diagnostics"] if item.get("code") == "degenerate_subpaths"]
    assert len(warnings) == 1
    assert warnings[0]["level"] == "warning"
    assert warnings[0]["subpaths"] == 2
    assert "2" in warnings[0]["message"]
    assert [p.get("d") for p in ET.parse(out).getroot().iter(f"{SVG_NS}path")] == ["M0,0 L4,0 L0,3 Z M1,1 Z M2,2 L3,2 Z"]


def test_degenerate_warning_appears_once_in_text_mode(run_injected):
    code, _, stderr, _, _ = run_injected(MIXED, json_mode=False)
    assert code == 0
    assert len(degenerate_warnings(stderr)) == 1


def test_no_degenerate_warning_without_degenerate_subpaths(run_injected):
    code, payload, stderr, _, _ = run_injected(injected(TRIANGLE_PATH))
    assert code == 0
    assert not [item for item in payload["diagnostics"] if item.get("code") == "degenerate_subpaths"]
    assert degenerate_warnings(stderr) == []


def test_published_svg_is_the_normalizer_output(run_injected):
    from vector_map_svg import normalize_svg

    code, payload, _, out, _ = run_injected(injected(TRIANGLE_PATH))
    assert code == 0
    canvas = config.Canvas(width_px=128, height_px=128, width_mm=10.0, height_mm=10.0, mm_per_px=10 / 128)
    assert out.read_text() == normalize_svg(injected(TRIANGLE_PATH), canvas).svg
