"""S3.3 layered aligned SVG export (STABL-qlagdbmh, spec 6.4, 7, 8).

Plan: docs/superpowers/plans/2026-10-09-vector-map-s3-3.md.
Public selection: --layers and recipe layers. Role settings: four CLI flags and recipe role objects.
A role setting outside the resolved selection is exit 2, also for the default silhouette-only run.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vector_map  # noqa: E402
import vector_map_artifacts as artifacts  # noqa: E402
import vector_map_config as config  # noqa: E402
import vector_map_layers as layers  # noqa: E402

SHAPE = (24, 32)


def cli_layer(**values):
    return {**dict.fromkeys(config.FIELDS), **values}


def resolve_image(tmp_path, **values):
    """Resolve image-mode settings with a silhouette mask. values form the explicit CLI layer."""
    recipe = values.pop("recipe", {})
    base = {"input": tmp_path / "source.png", "input_kind": "image", "width_mm": 32.0, "mask": tmp_path / "mask.png"}
    return config.resolve(config.DEFAULTS, {}, {**base, **recipe}, cli_layer(**values))


def write_mask(path, material):
    Image.fromarray(np.where(material, 255, 0).astype(np.uint8)).save(path)
    return path


@pytest.fixture
def scene(tmp_path):
    """Source, silhouette mask, and two role maps on a 32 x 24 px canvas. One mm per pixel."""
    source = tmp_path / "source.png"
    Image.fromarray(np.full((*SHAPE, 3), 90, np.uint8)).save(source)
    silhouette = np.zeros(SHAPE, bool)
    silhouette[2:22, 2:30] = True
    structure = np.zeros(SHAPE, bool)
    structure[7:10, 4:28] = True
    detail = np.zeros(SHAPE, bool)
    detail[14:17, 5:25] = True
    return {
        "dir": tmp_path,
        "source": source,
        "mask": write_mask(tmp_path / "mask.png", silhouette),
        "structure": write_mask(tmp_path / "structure.png", structure),
        "detail": write_mask(tmp_path / "detail.png", detail),
    }


def run(capsys, *argv):
    code = vector_map.main([*map(str, argv), "--json"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured
    return code, json.loads(lines[0]), captured.err


def image_args(scene, destination="out.svg", *extra):
    return (scene["source"], scene["dir"] / destination, "--input-kind", "image", "--mask", scene["mask"],
            "--width-mm", "32", *extra)


def write_recipe(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, **fields}))
    return path


# --- Selection ---------------------------------------------------------------


def test_image_mode_without_layers_keeps_single_svg_output(scene, capsys):
    code, result, error = run(capsys, *image_args(scene))
    assert code == 0, error
    assert "layers" not in result["artifacts"]
    assert not (scene["dir"] / "out.layers").exists()
    manifest = json.loads((scene["dir"] / "out.vector.json").read_bytes())
    assert manifest["layers"] == [{"id": "silhouette", "height_mm": None}]


def test_default_selection_is_unset(tmp_path):
    assert resolve_image(tmp_path).layers is None
    assert resolve_image(tmp_path).role_requests == ()


@pytest.mark.parametrize("given, expected", [
    ("silhouette", ("silhouette",)),
    ("detail,silhouette", ("silhouette", "detail")),
    ("detail,structure,silhouette", ("silhouette", "structure", "detail")),
])
def test_layers_normalize_to_fixed_role_order(tmp_path, given, expected):
    assert resolve_image(tmp_path, layers=given).layers == expected


def test_recipe_layers_list_selects_roles(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", layers=["structure", "silhouette"])
    decoded = config.load_recipe(recipe)
    assert resolve_image(tmp_path, recipe=decoded).layers == ("silhouette", "structure")


def test_cli_layers_replace_recipe_layers(tmp_path):
    settings = resolve_image(tmp_path, recipe={"layers": ("silhouette", "structure")}, layers="silhouette,detail")
    assert settings.layers == ("silhouette", "detail")


@pytest.mark.parametrize("given, message", [
    ("", "at least"),
    ("silhouette,", "empty"),
    ("silhouette,,detail", "empty"),
    ("silhouette,silhouette", "once"),
    ("silhouette,structure,structure", "once"),
    ("structure", "silhouette"),
    ("structure,detail", "silhouette"),
    ("silhouette,edges", "edges"),
    ("Silhouette", "Silhouette"),
])
def test_invalid_layers_selection_is_a_config_error(tmp_path, given, message):
    with pytest.raises(config.ConfigError, match=message):
        resolve_image(tmp_path, layers=given)


@pytest.mark.parametrize("value", ["silhouette", ["silhouette", 3], {"silhouette": True}, [True]])
def test_recipe_layers_must_be_a_list_of_strings(tmp_path, value):
    recipe = write_recipe(tmp_path / "recipe.json", layers=value)
    with pytest.raises(config.ConfigError, match="Recipe field layers must be a list of role names"):
        config.load_recipe(recipe)


def test_recipe_empty_layers_list_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError, match="at least"):
        resolve_image(tmp_path, recipe={"layers": ()})


@pytest.mark.parametrize("kind", ["mask", "edges"])
@pytest.mark.parametrize("extra", [
    ("--layers", "silhouette"),
    ("--structure-width-mm", "1"),
    ("--detail-width-mm", "1"),
    ("--structure-map", "MAP"),
    ("--detail-map", "MAP"),
])
def test_standalone_modes_reject_layer_selection_and_role_flags(scene, capsys, kind, extra):
    extra = tuple(str(scene["structure"]) if value == "MAP" else value for value in extra)
    code, result, error = run(capsys, scene["mask"], scene["dir"] / "out.svg", "--input-kind", kind,
                              "--width-mm", "32", *extra)
    assert code == 2
    assert result["status"] == "invalid"
    assert "require input_kind image" in error
    assert not (scene["dir"] / "out.svg").exists()


def test_standalone_recipe_role_object_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError, match="require input_kind image"):
        config.resolve(config.DEFAULTS, {}, {"input": tmp_path / "m.png", "input_kind": "mask", "width_mm": 1.0,
                                             "structure": {"width_mm": 1.0}}, cli_layer())


# --- Role settings outside the selection (Mario, 2026-10-09) -----------------


@pytest.mark.parametrize("extra", [
    ("--structure-width-mm", "1"),
    ("--detail-width-mm", "1"),
    ("--structure-map", "MAP"),
    ("--layers", "silhouette,structure", "--detail-map", "MAP"),
    ("--layers", "silhouette,detail", "--structure-width-mm", "1"),
])
def test_role_flag_for_an_unselected_role_is_exit_2(scene, capsys, extra):
    extra = tuple(str(scene["structure"]) if value == "MAP" else value for value in extra)
    code, _, error = run(capsys, *image_args(scene, "out.svg", *extra))
    assert code == 2
    assert "is not selected" in error and "--layers" in error
    assert not (scene["dir"] / "out.svg").exists()


@pytest.mark.parametrize("selection", [None, ("silhouette",), ("silhouette", "structure")])
def test_recipe_role_object_for_an_unselected_role_is_rejected(tmp_path, selection):
    recipe = {"detail": {"width_mm": 1.0}}
    if selection is not None:
        recipe["layers"] = selection
    with pytest.raises(config.ConfigError, match="detail is not selected"):
        resolve_image(tmp_path, recipe=recipe)


def test_empty_recipe_role_object_still_names_its_role(tmp_path):
    with pytest.raises(config.ConfigError, match="structure is not selected"):
        resolve_image(tmp_path, recipe={"structure": {}})


# --- Role recipe objects -----------------------------------------------------


def test_recipe_role_paths_resolve_relative_to_recipe_directory(tmp_path):
    recipe = write_recipe(tmp_path / "nested" / "recipe.json", layers=["silhouette", "structure"],
                          structure={"source": "map", "path": "maps/s.png",
                                     "include_mask": "c/in.png", "exclude_mask": "/abs/out.png"})
    settings = resolve_image(tmp_path, recipe=config.load_recipe(recipe))
    (request,) = settings.role_requests
    assert request.path == tmp_path / "nested" / "maps" / "s.png"
    assert request.include_mask == tmp_path / "nested" / "c" / "in.png"
    assert request.exclude_mask == Path("/abs/out.png")


def test_recipe_role_object_resolves_to_role_request(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", layers=["silhouette", "structure", "detail"],
                          structure={"canny": {"low_threshold": 120, "high_threshold": 240, "blur": 5},
                                     "width_mm": 0.8, "gap_close_mm": 0.6},
                          detail={"source": "canny", "width_mm": 0.5})
    settings = resolve_image(tmp_path, recipe=config.load_recipe(recipe))
    assert settings.role_requests == (
        layers.RoleRequest("structure", canny=layers.CannySpec(120, 240, 5), width_mm=0.8, gap_close_mm=0.6),
        layers.RoleRequest("detail", width_mm=0.5),
    )


def test_selected_role_without_settings_uses_canny_defaults(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette,structure,detail")
    assert settings.role_requests == (layers.RoleRequest("structure"), layers.RoleRequest("detail"))


def test_detail_canny_keeps_omitted_blur_for_the_shared_default(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette,detail",
                             recipe={"detail": {"canny": {"low_threshold": 40, "high_threshold": 90}}})
    (request,) = settings.role_requests
    assert request.canny == layers.CannySpec(40, 90, None)
    assert layers._resolve_canny(request).blur == layers.DEFAULT_CANNY["structure"].blur


@pytest.mark.parametrize("role", ["structure", "detail"])
@pytest.mark.parametrize("fields, message", [
    ({"source": "map"}, "path"),
    ({"source": "map", "path": "m.png", "canny": {"blur": 3}}, "canny"),
    ({"source": "canny", "path": "m.png"}, "path"),
    ({"path": "m.png"}, "path"),
    ({"source": "learned"}, "source"),
])
def test_role_source_and_path_rules(tmp_path, role, fields, message):
    recipe = write_recipe(tmp_path / "recipe.json", layers=["silhouette", role], **{role: fields})
    with pytest.raises(config.ConfigError, match=message):
        resolve_image(tmp_path, recipe=config.load_recipe(recipe))


@pytest.mark.parametrize("fields, message", [
    ({"height_mm": 1}, "height_mm"),
    ({"canny": {"sigma": 1}}, "sigma"),
    ({"canny": {"low_threshold": True}}, "low_threshold"),
    ({"canny": {"low_threshold": 1.5}}, "low_threshold"),
    ({"canny": {"blur": "3"}}, "blur"),
    ({"canny": [100, 200]}, "canny"),
    ({"width_mm": True}, "width_mm"),
    ({"width_mm": "1"}, "width_mm"),
    ({"gap_close_mm": False}, "gap_close_mm"),
    ({"source": 1}, "source"),
    ({"path": 3, "source": "map"}, "path"),
    ({"include_mask": 1}, "include_mask"),
])
def test_recipe_role_object_rejects_bad_fields_and_types(tmp_path, fields, message):
    recipe = write_recipe(tmp_path / "recipe.json", layers=["silhouette", "structure"], structure=fields)
    with pytest.raises(config.ConfigError, match=message):
        config.load_recipe(recipe)


def test_recipe_role_value_must_be_an_object(tmp_path):
    recipe = write_recipe(tmp_path / "recipe.json", layers=["silhouette", "structure"], structure=[1])
    with pytest.raises(config.ConfigError, match="Recipe field structure must be an object"):
        config.load_recipe(recipe)


@pytest.mark.parametrize("fields, message", [
    ({"canny": {"low_threshold": 200, "high_threshold": 100}}, "low_threshold < high_threshold"),
    ({"canny": {"low_threshold": -1}}, "low_threshold < high_threshold"),
    ({"canny": {"blur": 4}}, "blur"),
    ({"canny": {"blur": -1}}, "blur"),
    ({"width_mm": 0}, "width_mm"),
    ({"width_mm": -0.5}, "width_mm"),
    ({"width_mm": math.inf}, "width_mm"),
    ({"gap_close_mm": math.nan}, "gap_close_mm"),
    ({"gap_close_mm": 0}, "gap_close_mm"),
])
def test_invalid_role_values_fail_at_resolution(tmp_path, fields, message):
    with pytest.raises(config.ConfigError, match=message):
        resolve_image(tmp_path, layers="silhouette,structure", recipe={"structure": fields})


def test_recipe_nonfinite_json_length_is_rejected(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"schema_version": 1, "layers": ["silhouette", "detail"], "detail": {"width_mm": Infinity}}')
    with pytest.raises(config.ConfigError, match="width_mm"):
        resolve_image(tmp_path, recipe=config.load_recipe(recipe))


# --- CLI role flags and precedence -------------------------------------------


def test_cli_map_replaces_recipe_canny_configuration(tmp_path):
    settings = resolve_image(
        tmp_path, layers="silhouette,structure", structure={"source": "map", "path": tmp_path / "s.png"},
        recipe={"structure": {"canny": {"low_threshold": 10, "high_threshold": 20}, "width_mm": 0.8,
                              "gap_close_mm": 0.4}},
    )
    assert settings.role_requests == (
        layers.RoleRequest("structure", source="map", path=tmp_path / "s.png", width_mm=0.8, gap_close_mm=0.4),
    )


def test_cli_map_replaces_recipe_map_path(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette,detail", detail={"source": "map", "path": tmp_path / "b.png"},
                             recipe={"detail": {"source": "map", "path": tmp_path / "a.png"}})
    assert settings.role_requests[0].path == tmp_path / "b.png"


def test_cli_width_replaces_recipe_width_and_keeps_other_fields(tmp_path):
    settings = resolve_image(tmp_path, layers="silhouette,detail", detail={"width_mm": 0.9},
                             recipe={"detail": {"width_mm": 0.5, "gap_close_mm": 0.3,
                                                "canny": {"low_threshold": 30, "high_threshold": 60}}})
    assert settings.role_requests == (
        layers.RoleRequest("detail", canny=layers.CannySpec(30, 60, None), width_mm=0.9, gap_close_mm=0.3),
    )


def test_cli_parser_maps_role_flags_to_role_layers():
    args = vector_map.build_parser().parse_args([
        "s.png", "o.svg", "--layers", "silhouette,structure,detail", "--structure-map", "s.png",
        "--detail-map", "d.png", "--structure-width-mm", "0.8", "--detail-width-mm", "0.5",
    ])
    assert args.layers == "silhouette,structure,detail"
    assert (args.structure_map, args.detail_map) == (Path("s.png"), Path("d.png"))
    assert (args.structure_width_mm, args.detail_width_mm) == (0.8, 0.5)


@pytest.mark.parametrize("flag", ["--gap-close-mm", "--structure-gap-close-mm", "--detail-gap-close-mm"])
def test_gap_close_mm_has_no_cli_flag(scene, capsys, flag):
    """Guard: gap_close_mm stays recipe-only. It passes before S3.3 and must keep passing."""
    code, result, _ = run(capsys, *image_args(scene, "out.svg", "--layers", "silhouette,structure", flag, "1"))
    assert code == 2
    assert result["status"] == "invalid"


# --- Inputs: role maps and constraints are captured inputs -------------------


def layered_settings(scene, **values):
    return config.resolve(config.DEFAULTS, {}, {
        "input": scene["source"], "input_kind": "image", "width_mm": 32.0, "mask": scene["mask"],
    }, cli_layer(**values))


def test_input_paths_name_every_role_map_and_constraint(scene, tmp_path):
    settings = layered_settings(
        scene, layers="silhouette,structure,detail",
        structure={"source": "map", "path": scene["structure"], "include_mask": tmp_path / "si.png",
                   "exclude_mask": tmp_path / "se.png"},
        detail={"source": "map", "path": scene["detail"], "exclude_mask": tmp_path / "de.png"},
    )
    assert artifacts.input_paths(settings, None) == [
        ("source", scene["source"]), ("mask", scene["mask"]),
        ("structure-map", scene["structure"]), ("structure-include-mask", tmp_path / "si.png"),
        ("structure-exclude-mask", tmp_path / "se.png"),
        ("detail-map", scene["detail"]), ("detail-exclude-mask", tmp_path / "de.png"),
    ]


def test_shared_input_path_is_read_once_and_kept_for_each_role(scene):
    settings = layered_settings(scene, layers="silhouette,structure,detail",
                                structure={"source": "map", "path": scene["structure"]},
                                detail={"source": "map", "path": scene["structure"]})
    snapshots = {}
    inputs = artifacts.snapshot_inputs(settings, None, snapshots)
    assert list(snapshots) == [scene["source"], scene["mask"], scene["structure"]]
    roles = {entry["role"]: entry for entry in inputs}
    assert set(roles) == {"source", "mask", "structure-map", "detail-map"}
    assert roles["structure-map"]["sha256"] == roles["detail-map"]["sha256"]


def test_role_maps_and_constraints_decode_from_snapshots(scene, tmp_path):
    exclude = np.zeros(SHAPE, bool)
    exclude[:, :8] = True
    settings = layered_settings(scene, layers="silhouette,structure,detail",
                                structure={"source": "map", "path": scene["structure"],
                                           "exclude_mask": write_mask(tmp_path / "ex.png", exclude)},
                                detail={"source": "map", "path": scene["detail"]})
    snapshots = {}
    artifacts.snapshot_inputs(settings, None, snapshots)
    expected = layers.prepare_layers(settings, settings.role_requests, input_bytes=snapshots)
    for name in ("structure", "detail"):
        write_mask(scene[name], np.ones(SHAPE, bool))
    write_mask(tmp_path / "ex.png", np.zeros(SHAPE, bool))
    replayed = layers.prepare_layers(settings, settings.role_requests, input_bytes=snapshots)
    for role in ("structure", "detail"):
        assert np.array_equal(getattr(replayed, role).material, getattr(expected, role).material)
    assert not expected.structure.material[:, :8].any()
