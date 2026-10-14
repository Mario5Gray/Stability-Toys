"""S3.5 relief-0.4 fabrication preset (STABL-mknlfcui, spec 7).

Plan: docs/superpowers/plans/2026-10-14-vector-map-s3-5.md.
A preset is never implicit. It fills the preset precedence slot: defaults, preset, recipe, CLI.
It supplies fabrication values only: S, both nominal band widths, H, D, and disabled backing.
It never decides role selection. A preset requires an effective export, with the S3.4
--no-export-scad opt-out preserved. The replay recipe keeps the preset and the resolved role
widths, and a prepared role bypasses a second expansion.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map  # noqa: E402
import vector_map_config as config  # noqa: E402
from tests.test_vector_map_scad import scene_files  # noqa: E402

PRESET = "relief-0.4"
PRESET_RELIEF = {
    "export_scad": True, "center": False, "backing_mm": 0.0, "silhouette_thickness_mm": 1.2,
    "structure_height_mm": 0.6, "detail_height_mm": 0.2, "detail_mode": "raised", "preset": PRESET,
}


def cli_layer(**values):
    return {**dict.fromkeys(config.FIELDS), **values}


def resolve_image(tmp_path, *, preset=None, recipe=None, **values):
    """Resolve an image-mode run whose preset layer is built like the entry point builds it."""
    recipe = recipe or {}
    base = {"input": tmp_path / "source.png", "input_kind": "image", "width_mm": 32.0,
            "mask": tmp_path / "mask.png"}
    layer = cli_layer(**{"preset": preset, **values})
    name = preset if preset is not None else recipe.get("preset")
    return config.resolve(config.DEFAULTS, config.preset_layer(name, recipe, layer),
                          {**base, **recipe}, layer)


def resolve_mask(tmp_path, *, preset=None, **values):
    base = {"input": tmp_path / "source.png", "input_kind": "mask", "width_mm": 32.0}
    layer = cli_layer(**{"preset": preset, **values})
    return config.resolve(config.DEFAULTS, config.preset_layer(preset, base, layer), base, layer)


def widths(settings):
    return {request.role: request.width_mm for request in settings.role_requests}


@pytest.fixture
def scene(tmp_path):
    return scene_files(tmp_path)


def run(capsys, *argv):
    code = vector_map.main([*map(str, argv), "--json"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured
    return code, json.loads(lines[0]), captured.err


def layered_args(scene, *extra, layers="silhouette,structure,detail", width_mm="32", directory=None,
                 maps=True):
    """Layered CLI arguments. directory gives a run its own folder, so two runs share one output stem."""
    parent = scene["dir"]
    if directory is not None:
        parent = scene["dir"] / directory
        parent.mkdir(parents=True, exist_ok=True)
    role_maps = ("--structure-map", scene["structure"], "--detail-map", scene["detail"]) if maps else ()
    return (scene["source"], parent / "out.svg", "--input-kind", "image", "--mask", scene["mask"],
            "--width-mm", width_mm, "--layers", layers, *role_maps, *extra)


def manifest(directory, name="out.vector.json"):
    return json.loads((directory / name).read_bytes())


def entries(directory):
    return {item["id"]: item for item in manifest(directory)["layers"]}


# --- Preset selection and precedence ------------------------------------------


def test_no_preset_is_selected_by_default(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette", export_scad=True, silhouette_thickness_mm=2)
    assert settings.preset is None
    assert settings.silhouette_thickness_mm == 2.0


def test_preset_layer_is_empty_without_a_selection(tmp_path):
    layer = {"input": tmp_path / "source.png", "input_kind": "image", "layers": "silhouette"}
    assert config.preset_layer(None, layer) == {}


def test_an_export_without_a_preset_still_needs_a_silhouette_thickness(tmp_path):
    """No implicit preset: the missing thickness still exits 2."""
    with pytest.raises(config.ConfigError, match="silhouette_thickness_mm"):
        resolve_image(tmp_path, layers="silhouette", export_scad=True)


def test_a_cli_preset_resolves_every_fabrication_value(tmp_path):
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette,structure,detail",
                             export_scad=True)
    assert settings.preset == PRESET
    assert settings.silhouette_thickness_mm == 1.2
    assert settings.structure_height_mm == 0.6
    assert settings.detail_height_mm == 0.2
    assert settings.backing_mm == 0.0
    assert settings.center is False
    assert settings.detail_mode == "raised"
    assert widths(settings) == {"structure": 0.8, "detail": 0.5}


def test_a_recipe_preset_resolves_the_same_values(tmp_path):
    settings = resolve_image(tmp_path, recipe={"preset": PRESET}, layers="silhouette,structure",
                             export_scad=True)
    assert settings.preset == PRESET
    assert settings.silhouette_thickness_mm == 1.2
    assert settings.structure_height_mm == 0.6
    assert widths(settings) == {"structure": 0.8}


def test_an_unknown_preset_name_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError, match=PRESET) as error:
        resolve_image(tmp_path, preset="relief-0.2", layers="silhouette", export_scad=True)
    assert "relief-0.2" in str(error.value)


def test_a_preset_requires_an_export(tmp_path):
    with pytest.raises(config.ConfigError, match="export") as error:
        resolve_image(tmp_path, preset=PRESET, layers="silhouette,structure,detail")
    assert PRESET in str(error.value)


def test_an_explicit_no_export_over_a_recipe_export_keeps_the_preset_values(tmp_path):
    """S3.4's opt-out still validates every value and writes no source."""
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette,structure",
                             recipe={"export_scad": True}, export_scad=False)
    assert settings.export_scad is False
    assert settings.preset == PRESET
    assert settings.silhouette_thickness_mm == 1.2
    assert settings.structure_height_mm == 0.6


def test_a_recorded_relief_recipe_is_admitted_without_an_export(tmp_path):
    """The opt-out path's replay recipe records what the run validated, and needs no export."""
    settings = resolve_image(tmp_path, layers="silhouette,structure,detail",
                             recipe={"preset": PRESET, "export_scad": False, "relief_recorded": True,
                                     "silhouette_thickness_mm": 1.2, "structure_height_mm": 0.6,
                                     "detail_height_mm": 0.2})
    assert settings.export_scad is False
    assert settings.preset == PRESET
    assert (settings.silhouette_thickness_mm, settings.structure_height_mm) == (1.2, 0.6)


def test_a_recorded_relief_recipe_rejects_a_requested_export(tmp_path):
    """A recipe that asks for a source records no relief. The two claims contradict."""
    with pytest.raises(config.ConfigError, match="relief_recorded") as error:
        resolve_image(tmp_path, layers="silhouette,structure",
                      recipe={"preset": PRESET, "export_scad": True, "relief_recorded": True,
                              "silhouette_thickness_mm": 1.2, "structure_height_mm": 0.6})
    assert "export_scad" in str(error.value)


def test_a_recorded_relief_recipe_still_needs_its_heights_to_be_valid(tmp_path):
    """The marker relaxes the export rule only. Every other admission rule still holds."""
    with pytest.raises(config.ConfigError, match="structure_height_mm"):
        resolve_image(tmp_path, layers="silhouette",
                      recipe={"export_scad": False, "relief_recorded": True, "structure_height_mm": 0.6})


def test_a_preset_is_rejected_in_standalone_mask_mode(tmp_path):
    with pytest.raises(config.ConfigError, match="image") as error:
        resolve_mask(tmp_path, preset=PRESET)
    assert "preset" in str(error.value)


def test_a_preset_requires_an_explicit_selection(tmp_path):
    with pytest.raises(config.ConfigError, match="layers"):
        resolve_image(tmp_path, preset=PRESET, export_scad=True)


@pytest.mark.parametrize("field", ["structure_height_mm", "detail_height_mm"])
def test_preset_heights_are_dropped_for_unselected_roles(tmp_path, field):
    """The preset never decides role selection, so a silhouette-only run stays valid."""
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette", export_scad=True)
    assert settings.layers == ("silhouette",)
    assert getattr(settings, field) is None
    assert settings.role_requests == ()
    assert widths(settings) == {}


def test_preset_widths_are_dropped_for_unselected_roles(tmp_path):
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette,detail", export_scad=True)
    assert widths(settings) == {"detail": 0.5}


def test_an_explicit_height_for_an_absent_role_still_exits_2_under_a_preset(tmp_path):
    """S3.4's rule is about operator values. A preset does not relax it."""
    with pytest.raises(config.ConfigError, match="structure_height_mm") as error:
        resolve_image(tmp_path, preset=PRESET, layers="silhouette", export_scad=True,
                      structure_height_mm=1)
    assert "structure is not selected" in str(error.value)


@pytest.mark.parametrize("field, value, expected", [
    ("silhouette_thickness_mm", 1.5, 1.5),
    ("structure_height_mm", 1.0, 1.0),
    ("detail_height_mm", 0.1, 0.1),
    ("backing_mm", 2.0, 2.0),
])
def test_every_preset_height_stays_overridable(tmp_path, field, value, expected):
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette,structure,detail",
                             export_scad=True, **{field: value})
    assert getattr(settings, field) == expected


def test_both_preset_widths_stay_overridable(tmp_path):
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette,structure,detail",
                             export_scad=True, structure={"width_mm": 1.1})
    assert widths(settings) == {"structure": 1.1, "detail": 0.5}


def test_a_preset_keeps_the_raised_mode_and_uncentered_assembly(tmp_path):
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette,structure,detail",
                             export_scad=True)
    assert (settings.detail_mode, settings.center) == ("raised", False)


def test_a_preset_admits_engraved_detail(tmp_path):
    """0.2 mm stays above zero and below the preset's 1.2 mm silhouette thickness."""
    settings = resolve_image(tmp_path, preset=PRESET, layers="silhouette,detail", export_scad=True,
                             detail_mode="engraved")
    assert (settings.detail_mode, settings.detail_height_mm) == ("engraved", 0.2)


def test_a_preset_leaves_the_vtracer_defaults_alone(tmp_path):
    plain = resolve_image(tmp_path, layers="silhouette,structure", export_scad=True,
                          silhouette_thickness_mm=2, structure_height_mm=1)
    selected = resolve_image(tmp_path, preset=PRESET, layers="silhouette,structure", export_scad=True)
    assert dict(selected.vtracer) == dict(plain.vtracer)


def test_the_preset_name_travels_in_the_recipe_and_cli_layers():
    recipe = {"preset": PRESET, "layers": "silhouette,structure", "export_scad": True}
    layer = cli_layer()
    assert config.preset_layer(PRESET, recipe, layer)["structure"] == {"width_mm": 0.8}
    assert config.preset_layer(PRESET, {"layers": "silhouette", "export_scad": True}, layer) == {
        "silhouette_thickness_mm": 1.2, "backing_mm": 0.0}


# --- Layered runs, manifest, and replay ---------------------------------------


def test_a_preset_run_records_the_name_and_every_resolved_value(scene, capsys):
    code, result, error = run(capsys, *layered_args(scene, "--preset", PRESET, "--export-scad"))
    assert code == 0, error
    assert result["artifacts"]["scad"] == str(scene["dir"] / "out.scad")
    assert manifest(scene["dir"])["relief"] == PRESET_RELIEF
    published = entries(scene["dir"])
    assert (published["silhouette"]["height_mm"], published["silhouette"]["mode"]) == (1.2, "base")
    assert (published["structure"]["height_mm"], published["structure"]["mode"]) == (0.6, "raised")
    assert (published["detail"]["height_mm"], published["detail"]["mode"]) == (0.2, "raised")


def test_a_recipe_preset_run_records_the_same_manifest(scene, capsys):
    recipe = scene["dir"] / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "preset": PRESET, "export_scad": True}))
    code, _, error = run(capsys, *layered_args(scene, "--recipe", recipe))
    assert code == 0, error
    assert manifest(scene["dir"])["relief"] == PRESET_RELIEF


def test_a_cli_preset_replaces_a_recipe_preset(scene, capsys):
    recipe = scene["dir"] / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "preset": PRESET, "export_scad": True}))
    code, _, error = run(capsys, *layered_args(scene, "--recipe", recipe, "--preset", PRESET))
    assert code == 0, error
    assert manifest(scene["dir"])["relief"] == PRESET_RELIEF


def test_a_run_without_a_preset_keeps_the_null_preset_field(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--export-scad", "--silhouette-thickness-mm", "2",
                                               "--structure-height-mm", "1", "--detail-height-mm", "0.5"))
    assert code == 0, error
    assert manifest(scene["dir"])["relief"]["preset"] is None


def test_a_preset_without_export_exits_2_and_publishes_nothing(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--preset", PRESET))
    assert code == 2
    assert "unrecognized" not in error
    assert PRESET in error and "export" in error
    assert not (scene["dir"] / "out.svg").exists()
    assert not (scene["dir"] / "out.vector.json").exists()


def test_an_unknown_preset_exits_2_without_publishing(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--preset", "relief-0.2", "--export-scad"))
    assert code == 2
    assert PRESET in error
    assert not (scene["dir"] / "out.vector.json").exists()


def test_a_preset_is_rejected_in_standalone_edge_mode(scene, capsys):
    code, _, error = run(capsys, scene["source"], scene["dir"] / "out.svg", "--input-kind", "edges",
                         "--width-mm", "32", "--preset", PRESET)
    assert code == 2
    assert "unrecognized" not in error
    assert "preset" in error


def test_a_preset_silhouette_only_run_publishes_that_layer_alone(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--preset", PRESET, "--export-scad",
                                               layers="silhouette", maps=False))
    assert code == 0, error
    assert manifest(scene["dir"])["relief"] == {
        **PRESET_RELIEF, "structure_height_mm": None, "detail_height_mm": None}
    assert sorted(entries(scene["dir"])) == ["silhouette"]


def test_a_preset_source_matches_the_equivalent_explicit_flags(scene, capsys):
    """The preset is a value bundle and nothing more."""
    explicit = ["--silhouette-thickness-mm", "1.2", "--structure-height-mm", "0.6",
                "--detail-height-mm", "0.2", "--structure-width-mm", "0.8", "--detail-width-mm", "0.5"]
    code, _, error = run(capsys, *layered_args(scene, "--export-scad", *explicit, directory="explicit"))
    assert code == 0, error
    code, _, error = run(capsys, *layered_args(scene, "--preset", PRESET, "--export-scad",
                                               directory="preset"))
    assert code == 0, error
    assert (scene["dir"] / "preset/out.scad").read_bytes() == (scene["dir"] / "explicit/out.scad").read_bytes()
    assert manifest(scene["dir"], "preset/out.vector.json")["relief"] == manifest(
        scene["dir"], "explicit/out.vector.json")["relief"] | {"preset": PRESET}


def test_a_preset_replay_recipe_records_the_preset_widths_and_prepared_roles(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--preset", PRESET, "--export-scad",
                                               "--debug-bundle", width_mm="3.2"))
    assert code == 0, error
    recipe = json.loads((scene["dir"] / "out.debug/recipe.json").read_bytes())
    assert recipe["preset"] == PRESET
    assert recipe["silhouette_thickness_mm"] == 1.2
    assert recipe["structure_height_mm"] == 0.6
    assert recipe["detail_height_mm"] == 0.2
    assert recipe["backing_mm"] == 0.0
    assert recipe["detail_mode"] == "raised"
    assert recipe["export_scad"] is True
    assert recipe["structure"] == {"source": "map", "path": "structure.png", "width_mm": 0.8,
                                   "prepared": True}
    assert recipe["detail"] == {"source": "map", "path": "detail.png", "width_mm": 0.5,
                                "prepared": True}


def test_a_replayed_preset_bundle_matches_the_original_layers(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--preset", PRESET, "--export-scad",
                                               "--debug-bundle", width_mm="3.2"))
    assert code == 0, error
    original = {name: (scene["dir"] / f"out.layers/{name}.svg").read_bytes()
                for name in ("silhouette", "structure", "detail")}
    original_masks = {name: (scene["dir"] / f"out.debug/{name}.png").read_bytes()
                      for name in ("silhouette", "structure", "detail")}
    original_relief = manifest(scene["dir"])["relief"]

    destination = scene["dir"] / "replay.svg"
    code, _, error = run(capsys, destination, "--recipe", scene["dir"] / "out.debug/recipe.json",
                         "--debug-bundle")
    assert code == 0, error
    for name, expected in original.items():
        assert (scene["dir"] / f"replay.layers/{name}.svg").read_bytes() == expected, name
    for name, expected in original_masks.items():
        assert (scene["dir"] / f"replay.debug/{name}.png").read_bytes() == expected, name
    assert manifest(scene["dir"], "replay.vector.json")["relief"] == original_relief
    roles = manifest(scene["dir"], "replay.vector.json")["roles"]
    assert roles["structure"]["width_mm"] == 0.8
    assert roles["structure"]["expansion"] is None
    assert roles["detail"]["expansion"] is None


def test_a_suppressed_preset_replay_matches_the_original_layers(scene, capsys):
    """S3.4's opt-out still records the preset and heights, so its replay must record them too."""
    recipe = scene["dir"] / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "preset": PRESET, "export_scad": True}))
    code, result, error = run(capsys, *layered_args(scene, "--recipe", recipe, "--no-export-scad",
                                                    "--debug-bundle", width_mm="3.2"))
    assert code == 0, error
    assert "scad" not in result["artifacts"]
    original_relief = manifest(scene["dir"])["relief"]
    assert (original_relief["preset"], original_relief["export_scad"]) == (PRESET, False)
    assert original_relief["structure_height_mm"] == 0.6
    original = {name: (scene["dir"] / f"out.layers/{name}.svg").read_bytes()
                for name in ("silhouette", "structure", "detail")}
    original_masks = {name: (scene["dir"] / f"out.debug/{name}.png").read_bytes()
                      for name in ("silhouette", "structure", "detail")}

    replay_recipe = json.loads((scene["dir"] / "out.debug/recipe.json").read_bytes())
    assert replay_recipe["relief_recorded"] is True
    assert replay_recipe["export_scad"] is False
    assert replay_recipe["preset"] == PRESET
    assert replay_recipe["silhouette_thickness_mm"] == 1.2
    assert replay_recipe["structure_height_mm"] == 0.6
    assert replay_recipe["detail_height_mm"] == 0.2

    target = scene["dir"] / "replay"
    target.mkdir()
    code, _, error = run(capsys, target / "out.svg", "--recipe", scene["dir"] / "out.debug/recipe.json",
                         "--debug-bundle")
    assert code == 0, error
    for name, expected in original.items():
        assert (target / f"out.layers/{name}.svg").read_bytes() == expected, name
    for name, expected in original_masks.items():
        assert (target / f"out.debug/{name}.png").read_bytes() == expected, name
    assert manifest(scene["dir"], "replay/out.vector.json")["relief"] == original_relief
    assert not (target / "out.scad").exists()


def test_a_layered_run_without_relief_records_none_in_its_recipe(scene, capsys):
    """Only a run that records a relief value carries the block. A plain replay recipe stays as it was."""
    code, _, error = run(capsys, *layered_args(scene, "--debug-bundle"))
    assert code == 0, error
    recipe = json.loads((scene["dir"] / "out.debug/recipe.json").read_bytes())
    assert "relief_recorded" not in recipe
    assert "preset" not in recipe
    assert "silhouette_thickness_mm" not in recipe


def test_a_replay_keeps_the_published_scad_identical(scene, capsys):
    code, _, error = run(capsys, *layered_args(scene, "--preset", PRESET, "--export-scad",
                                               "--debug-bundle", width_mm="3.2"))
    assert code == 0, error
    original = (scene["dir"] / "out.scad").read_bytes()
    target = scene["dir"] / "replay"
    target.mkdir()
    code, _, error = run(capsys, target / "out.svg", "--recipe", scene["dir"] / "out.debug/recipe.json",
                         "--export-scad")
    assert code == 0, error
    assert (target / "out.scad").read_bytes() == original
    assert manifest(scene["dir"], "replay/out.vector.json")["relief"] == PRESET_RELIEF
