"""S3.4 OpenSCAD source assembly (STABL-gorlibrt, spec 6.5, 7).

Plan: docs/superpowers/plans/2026-10-11-vector-map-s3-4.md.
Admission: SCAD export needs explicit layers with silhouette. Relief settings are dormant without
export. The one exception is an explicit CLI --no-export-scad over a recipe export request.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map  # noqa: E402
import vector_map_config as config  # noqa: E402
import vector_map_scad as scad  # noqa: E402
from tests import test_vector_map_openscad as open_scad  # noqa: E402

SHAPE = (24, 32)
CANVAS = config.Canvas(32, 24, 32.0, 24.0, 1.0)
PUBLISHED = {
    "silhouette": "out.layers/silhouette.svg",
    "structure": "out.layers/structure.svg",
    "detail": "out.layers/detail.svg",
}
RELIEF_DEFAULTS = {
    "export_scad": False,
    "center": False,
    "detail_mode": "raised",
    "backing_mm": 0.0,
    "silhouette_thickness_mm": None,
    "structure_height_mm": None,
    "detail_height_mm": None,
}


def cli_layer(**values):
    return {**dict.fromkeys(config.FIELDS), **values}


def resolve_image(tmp_path, **values):
    """Resolve image-mode settings with a silhouette mask. values form the explicit CLI layer."""
    recipe = values.pop("recipe", {})
    base = {"input": tmp_path / "source.png", "input_kind": "image", "width_mm": 32.0, "mask": tmp_path / "mask.png"}
    return config.resolve(config.DEFAULTS, {}, {**base, **recipe}, cli_layer(**values))


def resolve_mask(tmp_path, **values):
    base = {"input": tmp_path / "source.png", "input_kind": "mask", "width_mm": 32.0}
    return config.resolve(config.DEFAULTS, {}, base, cli_layer(**values))


def write_recipe(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, **fields}))
    return path


def relief(settings):
    return {name: getattr(settings, name) for name in RELIEF_DEFAULTS}


# --- Configuration and admission ---------------------------------------------


def test_relief_defaults_are_off_and_raised(tmp_path):
    assert relief(resolve_image(tmp_path, layers="silhouette")) == RELIEF_DEFAULTS


def test_export_scad_requires_explicit_layers(tmp_path):
    with pytest.raises(config.ConfigError, match="layers"):
        resolve_image(tmp_path, export_scad=True, silhouette_thickness_mm=2)


def test_relief_setting_requires_explicit_layers(tmp_path):
    with pytest.raises(config.ConfigError, match="layers"):
        resolve_image(tmp_path, center=True)


def test_export_with_explicit_silhouette_is_admitted(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette", export_scad=True, silhouette_thickness_mm=2)
    assert settings.export_scad is True
    assert settings.layers == ("silhouette",)
    assert settings.silhouette_thickness_mm == 2.0


def test_export_requires_a_silhouette_thickness(tmp_path):
    with pytest.raises(config.ConfigError, match="silhouette_thickness_mm"):
        resolve_image(tmp_path, layers="silhouette", export_scad=True)


def test_export_admits_omitted_optional_heights(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette,structure", export_scad=True,
                             silhouette_thickness_mm=2)
    assert settings.structure_height_mm is None


@pytest.mark.parametrize("field, kept, absent", [
    ("structure_height_mm", "detail", "structure"),
    ("detail_height_mm", "structure", "detail"),
])
def test_height_for_an_unselected_role_is_rejected(tmp_path, field, kept, absent):
    """S3.3 rejects a role setting outside the selection. A role height is one."""
    with pytest.raises(config.ConfigError, match=field) as error:
        resolve_image(tmp_path, layers=f"silhouette,{kept}", export_scad=True,
                      silhouette_thickness_mm=2, **{field: 0.5})
    assert f"{absent} is not selected" in str(error.value)
    assert "layers" in str(error.value)


@pytest.mark.parametrize("field, absent", [("structure_height_mm", "structure"),
                                           ("detail_height_mm", "detail")])
def test_recipe_height_for_an_unselected_role_is_rejected(tmp_path, field, absent):
    recipe = write_recipe(tmp_path / "recipe.json", export_scad=True, layers=["silhouette"],
                          silhouette_thickness_mm=2, **{field: 1})
    with pytest.raises(config.ConfigError, match=field) as error:
        resolve_image(tmp_path, recipe=config.load_recipe(recipe))
    assert f"{absent} is not selected" in str(error.value)


def test_unselected_height_is_rejected_before_the_dormancy_rule(tmp_path):
    """The absent role is the root cause. It fails even when export is on and the value is valid."""
    with pytest.raises(config.ConfigError, match="detail_height_mm") as error:
        resolve_image(tmp_path, layers="silhouette", export_scad=True,
                      silhouette_thickness_mm=2, detail_height_mm=0.5)
    assert "export" not in str(error.value)


def test_dormant_relief_setting_without_export_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError, match="export"):
        resolve_image(tmp_path, layers="silhouette,structure", center=True, structure_height_mm=1)


def test_dormant_height_without_export_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError, match="export"):
        resolve_image(tmp_path, layers="silhouette,structure", structure_height_mm=1)


@pytest.mark.parametrize("name, value", [("center", False), ("detail_mode", "raised"), ("backing_mm", 0.0)])
def test_default_relief_values_are_admitted_without_export(tmp_path, name, value):
    settings = resolve_image(tmp_path, layers="silhouette,structure", **{name: value})
    assert getattr(settings, name) == value


def test_no_export_scad_overrides_a_recipe_export_request(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", export_scad=True, layers=["silhouette", "structure"],
                          silhouette_thickness_mm=2, structure_height_mm=1)
    settings = resolve_image(tmp_path, recipe=config.load_recipe(recipe), export_scad=False)
    assert settings.export_scad is False
    assert settings.silhouette_thickness_mm == 2.0
    assert settings.structure_height_mm == 1.0


def test_no_export_scad_without_a_request_is_rejected(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", layers=["silhouette", "structure"],
                          structure_height_mm=1)
    with pytest.raises(config.ConfigError, match="export"):
        resolve_image(tmp_path, recipe=config.load_recipe(recipe), export_scad=False)


def test_no_export_scad_override_still_validates_supplied_values(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", export_scad=True, layers=["silhouette"],
                          silhouette_thickness_mm=-2)
    with pytest.raises(config.ConfigError, match="silhouette_thickness_mm"):
        resolve_image(tmp_path, recipe=config.load_recipe(recipe), export_scad=False)


def test_recipe_export_request_without_the_override_keeps_export(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", export_scad=True, layers=["silhouette"],
                          silhouette_thickness_mm=2)
    assert resolve_image(tmp_path, recipe=config.load_recipe(recipe)).export_scad is True


def test_standalone_mode_rejects_relief_settings(tmp_path):
    with pytest.raises(config.ConfigError, match="input_kind image"):
        resolve_mask(tmp_path, export_scad=True, silhouette_thickness_mm=2)


def test_standalone_mode_rejects_a_relief_number(tmp_path):
    with pytest.raises(config.ConfigError, match="input_kind image"):
        resolve_mask(tmp_path, backing_mm=3)


def test_unknown_detail_mode_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError, match="detail_mode"):
        resolve_image(tmp_path, layers="silhouette", export_scad=True, silhouette_thickness_mm=2,
                      detail_mode="flush")


def test_negative_backing_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError, match="backing_mm"):
        resolve_image(tmp_path, layers="silhouette", export_scad=True, silhouette_thickness_mm=2,
                      backing_mm=-1)


def test_zero_backing_is_admitted(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette", export_scad=True,
                             silhouette_thickness_mm=2, backing_mm=0)
    assert settings.backing_mm == 0.0


@pytest.mark.parametrize("name", ["backing_mm", "silhouette_thickness_mm", "structure_height_mm"])
def test_bool_and_nonfinite_relief_numbers_are_rejected(tmp_path, name):
    for bad in (True, float("inf"), float("nan")):
        values = {"silhouette_thickness_mm": 2, name: bad}
        with pytest.raises(config.ConfigError, match=name):
            resolve_image(tmp_path, layers="silhouette,structure", export_scad=True, **values)


@pytest.mark.parametrize("name", ["silhouette_thickness_mm", "structure_height_mm"])
def test_nonpositive_relief_numbers_are_rejected(tmp_path, name):
    values = {"silhouette_thickness_mm": 2, name: 0}
    with pytest.raises(config.ConfigError, match=name):
        resolve_image(tmp_path, layers="silhouette,structure", export_scad=True, **values)


def test_cli_relief_values_override_the_recipe(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", export_scad=True, layers=["silhouette"],
                          silhouette_thickness_mm=2, center=False, detail_mode="raised")
    settings = resolve_image(tmp_path, recipe=config.load_recipe(recipe),
                             silhouette_thickness_mm=3.5, center=True)
    assert settings.silhouette_thickness_mm == 3.5
    assert settings.center is True


def test_recipe_decodes_relief_fields(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", export_scad=True, layers=["silhouette"],
                          silhouette_thickness_mm=2, backing_mm=1.5, detail_mode="engraved", center=True)
    settings = resolve_image(tmp_path, recipe=config.load_recipe(recipe))
    assert settings.backing_mm == 1.5
    assert settings.detail_mode == "engraved"
    assert settings.center is True


def test_recipe_rejects_an_unknown_relief_field(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", export_scad=True, depth_mm=2)
    with pytest.raises(config.ConfigError, match="unknown fields"):
        config.load_recipe(recipe)


# --- Source geometry ---------------------------------------------------------


def build(**values):
    """Build source for a raised three-role relief. values override the builder arguments."""
    arguments = {
        "layers": dict(PUBLISHED), "silhouette_thickness_mm": 2.0,
        "structure_height_mm": 1.0, "detail_height_mm": 0.5,
    }
    arguments.update(values)
    return scad.build_scad(CANVAS, **arguments)


def test_source_imports_each_published_layer_relative_and_uncentered():
    text = build()
    for role in PUBLISHED:
        assert f'import("out.layers/{role}.svg", center=false);' in text


def test_source_never_imports_the_combined_svg():
    assert "out.svg" not in build()


def test_source_is_deterministic():
    assert build() == build()


def test_source_omits_an_unpublished_role():
    text = build(layers={"silhouette": PUBLISHED["silhouette"]})
    assert text.count("import(") == 1
    assert "module structure_2d" not in text
    assert "module detail_2d" not in text


def test_source_requires_the_silhouette_role():
    with pytest.raises(config.ConfigError, match="silhouette"):
        build(layers={"structure": PUBLISHED["structure"]})


@pytest.mark.parametrize("name", ["structure_height_mm", "detail_height_mm"])
def test_source_requires_a_height_for_each_published_role(name):
    with pytest.raises(config.ConfigError, match=name):
        build(**{name: None})


def test_source_requires_the_silhouette_thickness():
    with pytest.raises(config.ConfigError, match="silhouette_thickness_mm"):
        build(silhouette_thickness_mm=None)


def test_source_intersects_structure_and_detail_with_the_silhouette():
    text = build()
    assert """module structure_2d() {
  intersection() {
    import("out.layers/structure.svg", center=false);
    silhouette_2d();
  }
}""" in text
    assert """module detail_2d() {
  difference() {
    intersection() {
      import("out.layers/detail.svg", center=false);
      silhouette_2d();
    }
    structure_2d();
  }
}""" in text


def test_source_keeps_detail_union_only_without_a_structure_role():
    text = build(layers={name: PUBLISHED[name] for name in ("silhouette", "detail")})
    assert """module detail_2d() {
  intersection() {
    import("out.layers/detail.svg", center=false);
    silhouette_2d();
  }
}""" in text


def test_raised_solids_sit_on_the_silhouette_top():
    text = build()
    assert "linear_extrude(height = H + min(B + S, OVERLAP)) structure_2d();" in text
    assert "linear_extrude(height = D + min(B + S, OVERLAP)) detail_2d();" in text


def test_engraving_removes_detail_from_the_silhouette_only():
    text = build(detail_mode="engraved")
    assert """  difference() {
    translate([0, 0, max(0, B - OVERLAP)])
      linear_extrude(height = S + min(B, OVERLAP)) silhouette_2d();
    translate([0, 0, B + S - D])
      linear_extrude(height = D + OVERLAP) engrave_2d();
  }
  translate([0, 0, max(0, B + S - OVERLAP)])
    linear_extrude(height = H + min(B + S, OVERLAP)) structure_2d();""" in text


def test_engraving_keeps_a_floor_at_the_detail_depth():
    assert "translate([0, 0, B + S - D])" in build(detail_mode="engraved")


def test_engraved_detail_must_fit_inside_the_silhouette_thickness():
    with pytest.raises(config.ConfigError, match="detail_height_mm"):
        build(detail_mode="engraved", detail_height_mm=2.0)


@pytest.mark.parametrize("mode", ["flush", "carved", ""])
def test_unknown_detail_mode_is_rejected_by_the_builder(mode):
    with pytest.raises(config.ConfigError, match="detail_mode"):
        build(detail_mode=mode)


def test_source_negative_backing_is_rejected():
    with pytest.raises(config.ConfigError, match="backing_mm"):
        build(backing_mm=-1)


def test_backing_spans_the_canvas_behind_every_other_solid():
    text = build(backing_mm=1.5)
    assert "B = 1.5; //" in text
    assert """  if (B > 0) {
    linear_extrude(height = B) square([32, 24]);
  }""" in text


def test_support_overlap_keeps_the_top_heights_and_the_zero_floor():
    text = build()
    assert "linear_extrude(height = S + min(B, OVERLAP)) silhouette_2d();" in text
    assert "translate([0, 0, max(0, B - OVERLAP)])" in text
    assert "translate([0, 0, max(0, B + S - OVERLAP)])" in text


def test_centering_applies_one_outer_transform():
    text = build(center=True)
    assert text.count("translate([-16, -12, 0])") == 1
    assert text.rstrip().endswith("translate([-16, -12, 0]) relief();")
    assert all("translate" not in line for line in text.splitlines() if "import(" in line)


def test_uncentered_source_places_the_assembly_at_the_origin():
    text = build(center=False)
    assert "translate([-" not in text
    assert text.rstrip().endswith("relief();")


def test_source_escapes_quotes_and_backslashes_in_a_layer_path():
    text = build(layers={"silhouette": 'stem "odd"\\dir/silhouette.svg'})
    assert 'import("stem \\"odd\\"\\\\dir/silhouette.svg", center=false);' in text


def test_source_keeps_a_non_ascii_layer_path():
    text = build(layers={"silhouette": "café.layers/silhouette.svg"})
    assert 'import("café.layers/silhouette.svg", center=false);' in text


def test_source_uses_the_canvas_millimetres():
    text = build()
    assert "square([32, 24])" in text


def test_source_generation_needs_no_executable(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert build(center=True).rstrip().endswith("relief();")


# --- Layered run, manifest, and diagnostics ----------------------------------


def write_mask(path, material):
    Image.fromarray(np.where(material, 255, 0).astype(np.uint8)).save(path)
    return path


def scene_files(directory):
    """Source, silhouette mask, and two role maps on a 32 x 24 px canvas. One mm per pixel."""
    source = directory / "source.png"
    Image.fromarray(np.full((*SHAPE, 3), 90, np.uint8)).save(source)
    silhouette = np.zeros(SHAPE, bool)
    silhouette[2:22, 2:30] = True
    structure = np.zeros(SHAPE, bool)
    structure[7:10, 4:28] = True
    detail = np.zeros(SHAPE, bool)
    detail[14:17, 5:25] = True
    return {
        "dir": directory,
        "source": source,
        "mask": write_mask(directory / "mask.png", silhouette),
        "structure": write_mask(directory / "structure.png", structure),
        "detail": write_mask(directory / "detail.png", detail),
    }


@pytest.fixture
def scene(tmp_path):
    return scene_files(tmp_path)


def layered_args(scene, *extra, destination="out.svg", layers="silhouette,structure,detail"):
    return (scene["source"], scene["dir"] / destination, "--input-kind", "image", "--mask", scene["mask"],
            "--width-mm", "32", "--layers", layers,
            "--structure-map", scene["structure"], "--detail-map", scene["detail"], *extra)


def export_args(scene, *extra):
    return layered_args(scene, "--export-scad", "--silhouette-thickness-mm", "2",
                        "--structure-height-mm", "1", "--detail-height-mm", "0.5", *extra)


def run(capsys, *argv):
    code = vector_map.main([*map(str, argv), "--json"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured
    return code, json.loads(lines[0]), captured.err


def manifest(scene, name="out.vector.json"):
    return json.loads((scene["dir"] / name).read_bytes())


def entries(scene):
    return {item["id"]: item for item in manifest(scene)["layers"]}


def test_layered_run_without_export_writes_no_scad(scene, capsys):
    code, result, error = run(capsys, *layered_args(scene))
    assert code == 0, error
    assert "scad" not in result["artifacts"]
    assert not (scene["dir"] / "out.scad").exists()
    assert manifest(scene)["relief"] == {
        "export_scad": False, "center": False, "backing_mm": 0.0, "silhouette_thickness_mm": None,
        "structure_height_mm": None, "detail_height_mm": None, "detail_mode": "raised", "preset": None,
    }


def test_export_writes_source_and_records_its_hash(scene, capsys):
    code, result, error = run(capsys, *export_args(scene))
    assert code == 0, error
    path = scene["dir"] / "out.scad"
    assert result["artifacts"]["scad"] == str(path)
    assert 'import("out.layers/silhouette.svg", center=false);' in path.read_text()
    published = {item["path"]: item for item in manifest(scene)["artifacts"]}
    assert published["out.scad"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert published["out.scad"]["bytes"] == len(path.read_bytes())


def test_export_generates_source_without_any_executable(scene, capsys, monkeypatch):
    """Source generation never starts OpenSCAD. An empty PATH and a forbidden process API prove it."""
    def forbidden(*args, **kwargs):
        raise AssertionError("SCAD export must not start a process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setenv("PATH", "")
    code, result, error = run(capsys, *export_args(scene))
    assert code == 0, error
    assert result["artifacts"]["scad"] == str(scene["dir"] / "out.scad")
    assert (scene["dir"] / "out.scad").exists()


def test_export_records_every_relief_value(scene, capsys):
    code, _, error = run(capsys, *export_args(scene, "--center", "--backing-mm", "1.5",
                                              "--detail-mode", "engraved"))
    assert code == 0, error
    assert manifest(scene)["relief"] == {
        "export_scad": True, "center": True, "backing_mm": 1.5, "silhouette_thickness_mm": 2.0,
        "structure_height_mm": 1.0, "detail_height_mm": 0.5, "detail_mode": "engraved", "preset": None,
    }


def test_manifest_records_published_layer_heights_and_modes(scene, capsys):
    code, _, error = run(capsys, *export_args(scene))
    assert code == 0, error
    assert (entries(scene)["silhouette"]["height_mm"], entries(scene)["silhouette"]["mode"]) == (2.0, "base")
    assert (entries(scene)["structure"]["height_mm"], entries(scene)["structure"]["mode"]) == (1.0, "raised")
    assert (entries(scene)["detail"]["height_mm"], entries(scene)["detail"]["mode"]) == (0.5, "raised")


def test_engraved_detail_is_recorded_as_a_mode(scene, capsys):
    code, _, error = run(capsys, *export_args(scene, "--detail-mode", "engraved"))
    assert code == 0, error
    assert entries(scene)["detail"]["mode"] == "engraved"


def test_export_stops_before_publication_when_a_published_role_has_no_height(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--export-scad", "--silhouette-thickness-mm", "2",
                                               "--structure-height-mm", "1"))
    assert code == 2
    assert "detail_height_mm" in error
    assert not (scene["dir"] / "out.scad").exists()
    assert not (scene["dir"] / "out.svg").exists()
    assert not (scene["dir"] / "out.vector.json").exists()


def test_empty_optional_role_has_no_import_and_a_null_height(scene, capsys):
    write_mask(scene["detail"], np.zeros(SHAPE, bool))
    code, _, error = run(capsys, *export_args(scene))
    assert code == 0, error
    text = (scene["dir"] / "out.scad").read_text()
    assert "detail.svg" not in text
    assert "D = " not in text
    assert (entries(scene)["detail"]["height_mm"], entries(scene)["detail"]["mode"]) == (None, None)
    assert (entries(scene)["structure"]["height_mm"], entries(scene)["structure"]["mode"]) == (1.0, "raised")


def test_empty_optional_role_still_validates_a_supplied_height(scene, capsys):
    write_mask(scene["detail"], np.zeros(SHAPE, bool))
    code, _, error = run(capsys, *export_args(scene, "--detail-height-mm", "-1"))
    assert code == 2
    assert "detail_height_mm" in error


@pytest.mark.parametrize("flag, role", [("--structure-height-mm", "structure"),
                                        ("--detail-height-mm", "detail")])
def test_cli_height_for_an_unselected_role_exits_2(scene, capsys, flag, role):
    """A height for a role outside the selection is ignored geometry. It exits 2 before processing."""
    code, _, error = run(capsys, scene["source"], scene["dir"] / "out.svg", "--input-kind", "image",
                         "--mask", scene["mask"], "--width-mm", "32", "--layers", "silhouette",
                         "--export-scad", "--silhouette-thickness-mm", "2", flag, "0.5")
    assert code == 2
    assert f"{role} is not selected" in error
    assert not (scene["dir"] / "out.scad").exists()
    assert not (scene["dir"] / "out.svg").exists()
    assert not (scene["dir"] / "out.vector.json").exists()


def test_centered_export_translates_the_whole_assembly(scene, capsys):
    code, _, error = run(capsys, *export_args(scene, "--center"))
    assert code == 0, error
    assert (scene["dir"] / "out.scad").read_text().rstrip().endswith("translate([-16, -12, 0]) relief();")


def test_default_image_run_rejects_export(scene, capsys):
    code, _, error = run(capsys, scene["source"], scene["dir"] / "out.svg", "--input-kind", "image",
                         "--mask", scene["mask"], "--width-mm", "32",
                         "--export-scad", "--silhouette-thickness-mm", "2")
    assert code == 2
    assert "layers" in error


def test_recipe_export_with_a_cli_opt_out_writes_no_scad_and_keeps_the_heights(scene, capsys):
    recipe = write_recipe(scene["dir"] / "recipe.json", layers=["silhouette", "structure", "detail"],
                          export_scad=True, silhouette_thickness_mm=2, structure_height_mm=1,
                          detail_height_mm=0.5, structure={"source": "map", "path": "structure.png"},
                          detail={"source": "map", "path": "detail.png"})
    code, _, error = run(capsys, scene["source"], scene["dir"] / "out.svg", "--input-kind", "image",
                         "--mask", scene["mask"], "--width-mm", "32", "--recipe", recipe,
                         "--no-export-scad")
    assert code == 0, error
    assert not (scene["dir"] / "out.scad").exists()
    assert manifest(scene)["relief"]["export_scad"] is False
    assert manifest(scene)["relief"]["structure_height_mm"] == 1.0
    assert (entries(scene)["structure"]["height_mm"], entries(scene)["structure"]["mode"]) == (None, None)


def two_island_mask():
    islands = np.zeros(SHAPE, bool)
    islands[2:11, 2:12] = True
    islands[13:22, 18:30] = True
    return islands


def silhouette_export_args(scene, *extra):
    return (scene["source"], scene["dir"] / "out.svg", "--input-kind", "image", "--mask", scene["mask"],
            "--width-mm", "32", "--layers", "silhouette",
            "--export-scad", "--silhouette-thickness-mm", "2", *extra)


def test_unbacked_silhouette_with_two_islands_warns_about_separate_solids(scene, capsys):
    write_mask(scene["mask"], two_island_mask())
    code, result, error = run(capsys, *silhouette_export_args(scene))
    assert code == 0, error
    (warning,) = [item for item in result["diagnostics"] if item.get("code") == "multiple_solids"]
    assert warning["level"] == "warning"
    assert "2" in warning["message"]
    assert "backing" in warning["message"]
    assert "S3.6" in warning["message"]


def test_backing_suppresses_the_separate_solids_warning(scene, capsys):
    write_mask(scene["mask"], two_island_mask())
    code, result, error = run(capsys, *silhouette_export_args(scene, "--backing-mm", "1"))
    assert code == 0, error
    assert "multiple_solids" not in {item.get("code") for item in result["diagnostics"]}


def test_one_island_needs_no_separate_solids_warning(scene, capsys):
    code, result, error = run(capsys, *silhouette_export_args(scene))
    assert code == 0, error
    assert "multiple_solids" not in {item.get("code") for item in result["diagnostics"]}


def test_module_is_listed_for_installation():
    metadata = tomllib.loads((ROOT / "scripts" / "pyproject.toml").read_text())
    assert "vector_map_scad" in metadata["tool"]["setuptools"]["py-modules"]


# --- Native OpenSCAD smoke render ---------------------------------------------


def test_s36_probe_blocks_when_openscad_is_missing(monkeypatch, tmp_path, capsys):
    from tests import vector_map_s3_6_cad_probe as probe

    monkeypatch.setattr(probe.open_scad, "_openscad", lambda: None)
    code = probe.main(["--work-dir", str(tmp_path / "work")])
    assert code != 0
    assert "OpenSCAD" in capsys.readouterr().err


def test_s36_area_conversion_uses_manifest_canvas_scale():
    from tests import vector_map_s3_6_cad_probe as probe

    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="64mm" height="48mm" viewBox="0 0 32 24"><path d="M0,0 L28,0 L28,20 L0,20 Z" fill="black"/></svg>'
    canvas = {"width_px": 32, "height_px": 24, "width_mm": 64, "height_mm": 48, "mm_per_px": 2}
    assert probe.role_area_mm2(svg, canvas) == 2240
#
# The sections above never start OpenSCAD. These tests render generated source in a real OpenSCAD
# 2021.01, so a mesher checks the Z intervals, the canvas, and the relative imports.


def render_scad(scad_path, stl_path, *, cwd=None):
    """Render generated source in a real OpenSCAD. Return (vertices, triangles, stderr)."""
    command = open_scad.OPENSCAD
    assert command is not None
    if error := open_scad.sandbox_error(command, [scad_path, stl_path]):
        raise RuntimeError(error)
    result = subprocess.run([*command, "-o", str(stl_path), str(scad_path)], capture_output=True,
                            text=True, timeout=300, cwd=None if cwd is None else str(cwd))
    assert result.returncode == 0, result.stderr
    corners = np.array([[float(value) for value in found]
                        for found in re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", stl_path.read_text())])
    vertices, index = np.unique(corners.round(6), axis=0, return_inverse=True)
    return vertices, index.reshape(-1, 3), result.stderr


def solids(vertices, triangles):
    """(bounds_min, bounds_max, genus) for each connected solid."""
    return open_scad.mesh_components(vertices, triangles)


def planes(vertices):
    """The distinct Z values in the mesh. Every support plane and every top is one of them."""
    return sorted({round(float(value), 4) for value in vertices[:, 2]})


@pytest.fixture
def render_dir(tmp_path):
    """A work directory a Flatpak OpenSCAD can read. pytest tmp_path is outside $HOME on Linux."""
    work, created = open_scad.choose_work_dir(tmp_path, open_scad.OPENSCAD)
    yield work
    if created:
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture
def render_scene(render_dir):
    """The layered scene inside the render work directory, so every input sits beside the source."""
    return scene_files(render_dir)


@open_scad.needs_openscad
def test_native_raised_render_matches_the_canvas_and_every_height(render_scene, render_dir, capsys):
    """Raises on a backing: floor 0, backing top B, silhouette top B + S, detail top B + S + D."""
    code, _, error = run(capsys, *export_args(render_scene, "--backing-mm", "1"))
    assert code == 0, error
    vertices, triangles, log = render_scad(render_scene["dir"] / "out.scad", render_dir / "raised.stl")
    assert np.allclose(vertices.min(0)[:2], [0, 0], atol=1e-4), log
    assert np.allclose(vertices.max(0)[:2], [32, 24], atol=1e-4), log
    assert planes(vertices) == [0.0, 1.0, 3.0, 3.5, 4.0]
    parts = solids(vertices, triangles)
    assert len(parts) == 1
    assert sum(genus for *_, genus in parts) == 0


@open_scad.needs_openscad
def test_native_engraved_render_keeps_the_floor_and_an_uncut_structure(render_scene, render_dir, capsys):
    """Engraving cuts to B + S - D only. The structure still reaches B + S + H and the top is flat."""
    code, _, error = run(capsys, *export_args(render_scene, "--detail-mode", "engraved"))
    assert code == 0, error
    vertices, triangles, log = render_scad(render_scene["dir"] / "out.scad", render_dir / "engraved.stl")
    assert np.allclose(vertices.min(0)[:2], [2, 2], atol=1e-4), log
    assert np.allclose(vertices.max(0)[:2], [30, 22], atol=1e-4), log
    assert planes(vertices) == [0.0, 1.5, 2.0, 3.0]
    parts = solids(vertices, triangles)
    assert len(parts) == 1
    assert sum(genus for *_, genus in parts) == 0


@open_scad.needs_openscad
def test_native_render_resolves_a_layer_import_beside_the_source(render_scene, render_dir, capsys):
    """Relative imports resolve against the SCAD file, not the working directory."""
    code, _, error = run(capsys, *export_args(render_scene, "--backing-mm", "1"))
    assert code == 0, error
    here = render_scad(render_scene["dir"] / "out.scad", render_dir / "here.stl", cwd=render_dir)[:2]
    elsewhere = render_dir / "elsewhere"
    elsewhere.mkdir()
    away = render_scad(render_scene["dir"] / "out.scad", elsewhere / "away.stl", cwd=elsewhere)[:2]
    assert planes(here[0]) == [0.0, 1.0, 3.0, 3.5, 4.0]
    assert np.array_equal(here[0], away[0])


@open_scad.needs_openscad
def test_native_render_shows_two_islands_and_the_backing_join(render_scene, render_dir, capsys):
    """The separate_solids warning is exact: two solids without a backing, one with it."""
    write_mask(render_scene["mask"], two_island_mask())
    code, _, error = run(capsys, *silhouette_export_args(render_scene))
    assert code == 0, error
    loose = render_scad(render_scene["dir"] / "out.scad", render_dir / "loose.stl")[:2]
    assert len(solids(*loose)) == 2
    code, _, error = run(capsys, *silhouette_export_args(render_scene, "--backing-mm", "1", "--overwrite"))
    assert code == 0, error
    joined = render_scad(render_scene["dir"] / "out.scad", render_dir / "joined.stl")[:2]
    assert len(solids(*joined)) == 1
    assert np.allclose(joined[0].max(0)[:2], [32, 24], atol=1e-4)


@open_scad.needs_openscad
def test_native_donut_backing_fills_the_through_hole(render_scene, render_dir, capsys):
    from tests import vector_map_s3_6_cad_probe as probe

    ring = np.zeros(SHAPE, bool)
    ring[2:22, 2:30] = True
    ring[8:16, 11:21] = False
    write_mask(render_scene["mask"], ring)
    code, _, error = run(capsys, *silhouette_export_args(render_scene))
    assert code == 0, error
    bare = render_scad(render_scene["dir"] / "out.scad", render_dir / "donut-open.stl")[:2]
    assert len(solids(*bare)) == 1
    assert solids(*bare)[0][2] == 1
    open_preview = render_dir / "donut-open-preview.png"
    probe._preview(*bare, open_preview)
    code, _, error = run(capsys, *silhouette_export_args(render_scene, "--backing-mm", "1",
                                                        "--overwrite"))
    assert code == 0, error
    backed = render_scad(render_scene["dir"] / "out.scad", render_dir / "donut-filled.stl")[:2]
    assert len(solids(*backed)) == 1
    assert solids(*backed)[0][2] == 0
    filled_preview = render_dir / "donut-filled-preview.png"
    probe._preview(*backed, filled_preview)

    def sample(preview, mesh, x, y):
        low, high = mesh[0][:, :2].min(0), mesh[0][:, :2].max(0)
        scale = min(560 / (high[0] - low[0]), 360 / (high[1] - low[1]))
        px = round(40 + (x - low[0]) * scale)
        py = round(40 + (high[1] - y) * scale)
        return Image.open(preview).convert("RGB").getpixel((px, py))

    assert sample(open_preview, bare, 15.5, 12.5) == (255, 255, 255)
    assert sample(filled_preview, backed, 15.5, 12.5) == sample(
        filled_preview, backed, 1.5, 1.5)
    assert sample(filled_preview, backed, 5.5, 5.5) != sample(
        filled_preview, backed, 15.5, 12.5)


@open_scad.needs_openscad
def test_native_preset_reaches_every_raised_height(render_scene, render_dir, capsys):
    code, _, error = run(capsys, *layered_args(render_scene, "--preset", "relief-0.4",
                                               "--export-scad"))
    assert code == 0, error
    vertices, triangles, _ = render_scad(render_scene["dir"] / "out.scad",
                                         render_dir / "preset.stl")
    assert planes(vertices) == [0.0, 1.2, 1.4, 1.8]
    assert len(solids(vertices, triangles)) == 1


@open_scad.needs_openscad
@pytest.mark.parametrize("case", ("badge", "bracket", "brick", "coins", "gravel", "horse",
                                  "pcb", "truchet"))
def test_native_corpus_layered_preset_renders(case, render_dir, capsys):
    """Every reviewed corpus mask gets a real layered image-to-mesh path."""
    from shutil import copyfile

    corpus = ROOT / "tests/fixtures/vector_map/corpus" / case
    directory = render_dir / f"cad-{case}"
    directory.mkdir()
    for name in ("source.png", "mask.png"):
        copyfile(corpus / name, directory / name)
    code, _, error = run(capsys, directory / "source.png", directory / "out.svg",
                         "--input-kind", "image", "--mask", directory / "mask.png",
                         "--width-mm", "100", "--layers", "silhouette,structure,detail",
                         "--preset", "relief-0.4", "--export-scad")
    assert code == 0, error
    vertices, triangles, _ = render_scad(directory / "out.scad", directory / "out.stl")
    assert len(vertices) and len(triangles)
    assert vertices[:, 2].min() == pytest.approx(0)
    assert vertices[:, 2].max() <= 1.8 + 1e-6
