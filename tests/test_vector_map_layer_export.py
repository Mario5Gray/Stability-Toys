"""S3.3 layered aligned SVG export (STABL-qlagdbmh, spec 6.4, 7, 8).

Plan: docs/superpowers/plans/2026-10-09-vector-map-s3-3.md.
Public selection: --layers and recipe layers. Role settings: four CLI flags and recipe role objects.
A role setting outside the resolved selection is exit 2, also for the default silhouette-only run.
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
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
import vector_map_preview as preview  # noqa: E402
import vector_map_svg as svg_io  # noqa: E402
import vector_map_vtracer as adapter  # noqa: E402

SHAPE = (24, 32)
SVG = "{http://www.w3.org/2000/svg}"


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


# --- Combined SVG assembly ---------------------------------------------------

GEOMETRY = (30, 40)


def geometry_masks():
    """Asymmetric silhouette touching two borders, with a donut hole and a nested island.

    Structure has separate components and an L. Detail is a donut. All shapes trace exactly.
    """
    silhouette = np.zeros(GEOMETRY, bool)
    silhouette[0:28, 0:35] = True
    silhouette[10:18, 20:29] = False
    silhouette[12:16, 23:27] = True
    structure = np.zeros(GEOMETRY, bool)
    structure[3:7, 3:15] = True
    structure[3:7, 18:31] = True
    structure[3:20, 3:7] = True
    detail = np.zeros(GEOMETRY, bool)
    detail[20:27, 8:18] = True
    detail[22:25, 11:15] = False
    return {"silhouette": silhouette, "structure": structure, "detail": detail}


def canvas_at(mm_per_px, shape=GEOMETRY):
    height, width = shape
    return config.Canvas(width, height, width * mm_per_px, height * mm_per_px, mm_per_px)


def normalized(material, canvas):
    return svg_io.normalize_svg(adapter.trace_layer(material).svg, canvas).svg


def test_combine_puts_each_layer_in_a_named_group_on_the_common_root():
    canvas = canvas_at(0.37)
    parts = [(role, normalized(material, canvas)) for role, material in geometry_masks().items()]
    combined = ET.fromstring(svg_io.combine_layers(parts))
    assert combined.tag == f"{SVG}svg"
    first = ET.fromstring(parts[0][1])
    assert {name: combined.get(name) for name in ("width", "height", "viewBox")} == {
        name: first.get(name) for name in ("width", "height", "viewBox")}
    assert [group.get("id") for group in combined] == ["silhouette", "structure", "detail"]
    for group, (_, text) in zip(combined, parts):
        assert group.tag == f"{SVG}g"
        assert set(group.attrib) == {"id"}
        assert [(child.tag, child.attrib) for child in group] == [
            (child.tag, child.attrib) for child in ET.fromstring(text)]


def test_combine_keeps_the_given_group_order():
    canvas = canvas_at(1.0)
    masks = geometry_masks()
    parts = [(role, normalized(masks[role], canvas)) for role in ("detail", "silhouette")]
    combined = ET.fromstring(svg_io.combine_layers(parts))
    assert [group.get("id") for group in combined] == ["detail", "silhouette"]


@pytest.mark.parametrize("change", [
    {"width": "41mm"}, {"height": "29mm"}, {"viewBox": "0 0 40 31"}, {"viewBox": None},
])
def test_combine_rejects_mismatched_roots(change):
    canvas = canvas_at(1.0)
    masks = geometry_masks()
    other = ET.fromstring(normalized(masks["structure"], canvas))
    for name, value in change.items():
        if value is None:
            del other.attrib[name]
        else:
            other.set(name, value)
    parts = [("silhouette", normalized(masks["silhouette"], canvas)),
             ("structure", ET.tostring(other, encoding="unicode"))]
    with pytest.raises(svg_io.SvgInspectionError, match="root"):
        svg_io.combine_layers(parts)


@pytest.mark.parametrize("mutate, message", [
    (lambda root: ET.SubElement(root, f"{SVG}circle", r="3"), "circle"),
    (lambda root: root[0].set("style", "fill:red"), "style"),
    (lambda root: root[0].set("fill", "red"), "fill"),
    (lambda root: root[0].set("id", "structure"), "id"),
])
def test_combine_rejects_unsupported_layer_content(mutate, message):
    canvas = canvas_at(1.0)
    masks = geometry_masks()
    root = ET.fromstring(normalized(masks["silhouette"], canvas))
    mutate(root)
    parts = [("silhouette", ET.tostring(root, encoding="unicode")),
             ("structure", normalized(masks["structure"], canvas))]
    with pytest.raises(svg_io.SvgInspectionError, match=message):
        svg_io.combine_layers(parts)


def test_combine_rejects_a_dtd():
    canvas = canvas_at(1.0)
    text = normalized(geometry_masks()["silhouette"], canvas)
    text = text.replace("<svg", '<!DOCTYPE svg [<!ENTITY x "y">]>\n<svg', 1)
    with pytest.raises(svg_io.SvgInspectionError, match="DTD"):
        svg_io.combine_layers([("silhouette", text)])


@pytest.mark.parametrize("parts, message", [
    ([], "at least one"),
    ([("silhouette", None), ("silhouette", None)], "once"),
    ([("bad id", None)], "id"),
])
def test_combine_rejects_bad_group_lists(parts, message):
    text = normalized(geometry_masks()["silhouette"], canvas_at(1.0))
    with pytest.raises(svg_io.SvgInspectionError, match=message):
        svg_io.combine_layers([(role, text) for role, _ in parts])


# --- Layered CLI geometry ----------------------------------------------------


@pytest.fixture
def geometry_scene(tmp_path):
    masks = geometry_masks()
    source = tmp_path / "source.png"
    Image.fromarray(np.dstack([np.where(masks["silhouette"], 200, 30).astype(np.uint8)] * 3)).save(source)
    paths = {role: write_mask(tmp_path / f"{role}.png", material) for role, material in masks.items()}
    expected = {
        "silhouette": masks["silhouette"],
        "structure": masks["structure"] & masks["silhouette"],
        "detail": masks["detail"] & masks["silhouette"] & ~(masks["structure"] & masks["silhouette"]),
    }
    return {"dir": tmp_path, "source": source, "paths": paths, "expected": expected}


def layered_run(capsys, scene, mm_per_px, *extra, destination="out.svg", layers_arg="silhouette,structure,detail"):
    width_mm = GEOMETRY[1] * mm_per_px
    return run(capsys, scene["source"], scene["dir"] / destination, "--input-kind", "image",
               "--mask", scene["paths"]["silhouette"], "--width-mm", repr(width_mm), "--layers", layers_arg,
               "--structure-map", scene["paths"]["structure"], "--detail-map", scene["paths"]["detail"], *extra)


def keep_group(text, role):
    root = ET.fromstring(text)
    for group in list(root):
        if group.get("id") != role:
            root.remove(group)
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("mm_per_px", [0.37, 25.4 / 96])
def test_layered_run_publishes_aligned_separate_and_combined_svgs(geometry_scene, capsys, mm_per_px):
    code, result, error = layered_run(capsys, geometry_scene, mm_per_px)
    assert code == 0, error
    out = geometry_scene["dir"] / "out.svg"
    roles = ("silhouette", "structure", "detail")
    separate = {role: geometry_scene["dir"] / "out.layers" / f"{role}.svg" for role in roles}
    assert result["artifacts"]["svg"] == str(out)
    assert result["artifacts"]["layers"] == {role: str(path) for role, path in separate.items()}
    texts = {role: path.read_text() for role, path in separate.items()}
    assert result["counts"] == {"layers": 3, "paths": sum(svg_io.count_paths(text) for text in texts.values())}

    canvas = canvas_at(mm_per_px)
    root_attributes = {"width": svg_io._mm(canvas.width_mm), "height": svg_io._mm(canvas.height_mm),
                       "viewBox": "0 0 40 30"}
    combined = out.read_text()
    for text in (combined, *texts.values()):
        root = ET.fromstring(text)
        assert {name: root.get(name) for name in root_attributes} == root_attributes

    union = np.zeros(GEOMETRY, bool)
    for role in roles:
        expected = geometry_scene["expected"][role]
        assert np.array_equal(preview.render_material(texts[role], canvas), expected), role
        assert np.array_equal(preview.render_material(keep_group(combined, role), canvas), expected), role
        union |= expected
    rendered = preview.render_material(combined, canvas)
    assert np.array_equal(rendered, union)
    assert not rendered[10:12, 20:29].any(), "donut hole lost"
    assert rendered[12:16, 23:27].all(), "nested island lost"


def test_combined_groups_copy_each_separate_layer_unchanged(geometry_scene, capsys):
    code, _, error = layered_run(capsys, geometry_scene, 1.0)
    assert code == 0, error
    combined = ET.fromstring((geometry_scene["dir"] / "out.svg").read_text())
    assert [group.get("id") for group in combined] == ["silhouette", "structure", "detail"]
    for group in combined:
        layer = ET.fromstring((geometry_scene["dir"] / "out.layers" / f"{group.get('id')}.svg").read_text())
        assert [(child.tag, child.attrib) for child in group] == [(child.tag, child.attrib) for child in layer]


@pytest.mark.parametrize("given", ["silhouette", "silhouette,structure", "detail,silhouette"])
def test_layered_selection_publishes_only_selected_roles(geometry_scene, capsys, given):
    code, result, error = run(
        capsys, geometry_scene["source"], geometry_scene["dir"] / "out.svg", "--input-kind", "image",
        "--mask", geometry_scene["paths"]["silhouette"], "--width-mm", "40", "--layers", given,
        *(["--structure-map", geometry_scene["paths"]["structure"]] if "structure" in given else []),
        *(["--detail-map", geometry_scene["paths"]["detail"]] if "detail" in given else []),
    )
    assert code == 0, error
    selected = [role for role in ("silhouette", "structure", "detail") if role in given.split(",")]
    assert list(result["artifacts"]["layers"]) == selected
    assert sorted(path.name for path in (geometry_scene["dir"] / "out.layers").iterdir()) == sorted(
        f"{role}.svg" for role in selected)
    combined = ET.fromstring((geometry_scene["dir"] / "out.svg").read_text())
    assert [group.get("id") for group in combined] == selected
    assert result["counts"]["layers"] == len(selected)


def test_empty_optional_role_is_reported_and_not_exported(geometry_scene, capsys):
    write_mask(geometry_scene["paths"]["detail"], np.zeros(GEOMETRY, bool))
    code, result, error = layered_run(capsys, geometry_scene, 1.0)
    assert code == 0, error
    assert list(result["artifacts"]["layers"]) == ["silhouette", "structure"]
    assert not (geometry_scene["dir"] / "out.layers" / "detail.svg").exists()
    assert result["counts"]["layers"] == 2
    empty = [item for item in result["diagnostics"] if item.get("code") == "role_empty"]
    assert [(item["level"], item["role"]) for item in empty] == [("warning", "detail")]
    assert "detail" in error
    combined = ET.fromstring((geometry_scene["dir"] / "out.svg").read_text())
    assert [group.get("id") for group in combined] == ["silhouette", "structure"]


def test_every_selected_role_is_published_or_reported_empty(geometry_scene, capsys):
    write_mask(geometry_scene["paths"]["structure"], np.zeros(GEOMETRY, bool))
    code, result, error = layered_run(capsys, geometry_scene, 1.0)
    assert code == 0, error
    reported = {item["role"] for item in result["diagnostics"] if item.get("code") == "role_empty"}
    assert set(result["artifacts"]["layers"]) | reported == {"silhouette", "structure", "detail"}
    assert set(result["artifacts"]["layers"]) & reported == set()


def test_recipe_gap_closing_keeps_the_s32_radius_and_kernel_cap(geometry_scene, capsys):
    recipe = write_recipe(geometry_scene["dir"] / "recipe.json", structure={"gap_close_mm": 1000.0})
    code, result, error = layered_run(capsys, geometry_scene, 1.0, "--recipe", recipe)
    assert code == 0, error
    (closing,) = [item for item in result["diagnostics"] if item.get("code") == "gap_closing"]
    assert closing["role"] == "structure"
    assert (closing["gap_px"], closing["radius_px"], closing["kernel_size_px"]) == (1000.0, 500, 1001)


@pytest.mark.parametrize("gap_mm, radius", [(1.0, 1), (2.0, 1), (2.5, 2), (4.0, 2)])
def test_recipe_gap_close_mm_is_the_widest_gap_not_a_kernel_width(geometry_scene, capsys, gap_mm, radius):
    recipe = write_recipe(geometry_scene["dir"] / "recipe.json", detail={"gap_close_mm": gap_mm})
    code, result, error = layered_run(capsys, geometry_scene, 1.0, "--recipe", recipe)
    assert code == 0, error
    (closing,) = [item for item in result["diagnostics"] if item.get("code") == "gap_closing"]
    assert (closing["role"], closing["radius_px"]) == ("detail", radius)
    assert closing["radius_px"] == max(1, math.ceil(gap_mm / 2))
