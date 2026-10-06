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


# --- Deferred preparation: exit 2 and the owning task ------------------------


@pytest.mark.parametrize(
    "values, owner",
    [
        ({"input_kind": "edges"}, S23),
        ({"input_kind": "image"}, "S3.1 (STABL-memwrtos)"),
        ({"max_res": 64}, S23),
        ({"line_width_mm": 0.5}, S23),
        ({"mask": Path("m.png")}, S23),
        ({"alpha": True}, S23),
    ],
)
def test_deferred_settings_name_the_owning_task(values, owner):
    layer = cli_layer(input=Path("a.png"), input_kind="mask", width_mm=1.0)
    with pytest.raises(config.ConfigError, match=owner.replace("(", r"\(").replace(")", r"\)")):
        resolve({}, {}, {**layer, **values})


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
def test_alpha_bearing_input_is_deferred(tmp_path, mode):
    path = tmp_path / "m.png"
    Image.new(mode, (4, 4)).save(path)
    with pytest.raises(config.ConfigError, match=r"S2\.3 \(STABL-vjpnctjh\)"):
        raster.prepare_mask(path, invert=False)


def test_palette_transparency_is_deferred(tmp_path):
    path = tmp_path / "m.png"
    image = Image.new("P", (4, 4))
    image.info["transparency"] = 0
    image.save(path, transparency=0)
    with pytest.raises(config.ConfigError, match=r"S2\.3 \(STABL-vjpnctjh\)"):
        raster.prepare_mask(path, invert=False)


def exif_jpeg(path, orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.new("L", (8, 4), 255).save(path, exif=exif)
    return path


def test_nonidentity_exif_orientation_is_deferred(tmp_path):
    with pytest.raises(config.ConfigError, match=r"S2\.3 \(STABL-vjpnctjh\)"):
        raster.prepare_mask(exif_jpeg(tmp_path / "m.jpg", 6), invert=False)


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
    assert payload["artifacts"] == {"svg": str(out)}
    assert payload["counts"] == {"layers": 1, "paths": svg_io.count_paths(out.read_text())}
    assert payload["diagnostics"] == []


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


@pytest.mark.parametrize(
    "extra, owner",
    [
        (["--input-kind", "edges"], "STABL-vjpnctjh"),
        (["--input-kind", "image"], "STABL-memwrtos"),
        (["--max-res", "64"], "STABL-vjpnctjh"),
        (["--line-width-mm", "0.5"], "STABL-vjpnctjh"),
        (["--mask", "m.png"], "STABL-vjpnctjh"),
        (["--alpha"], "STABL-vjpnctjh"),
        (["--preview"], "STABL-kfrksmnp"),
    ],
)
def test_deferred_flags_exit_2_and_name_the_owner(tmp_path, extra, owner):
    out = tmp_path / "o.svg"
    args = [FIXTURES / "donut.png", out, "--width-mm", 10]
    if "--input-kind" not in extra:
        args += ["--input-kind", "mask"]
    result = run_cli(*args, *extra, "--json")
    assert result.returncode == 2, result.stderr
    payload = one_result(result)
    assert payload["status"] == "invalid"
    assert owner in payload["diagnostics"][0]["message"]
    assert owner in result.stderr
    assert not out.exists()


def test_alpha_bearing_source_exits_2(tmp_path):
    source = tmp_path / "rgba.png"
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(source)
    result = run_cli(source, tmp_path / "o.svg", "--input-kind", "mask", "--width-mm", 10)
    assert result.returncode == 2
    assert "STABL-vjpnctjh" in result.stderr


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
