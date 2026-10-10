"""S2.6 provenance and publication contracts. STABL-fmjwbrzw."""

import hashlib
import errno
import os
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURE = ROOT / "tests/fixtures/vector_map/donut.png"

import vector_map_config as config
import vector_map_raster as raster
import vector_map_svg as svg
import vector_map_vtracer as adapter


def settings(source=FIXTURE, **values):
    return config.resolve(config.DEFAULTS, {
        "input": source, "input_kind": "mask", "width_mm": 100, **values,
    })


def test_manifest_records_consumed_bytes_and_measured_artifacts(tmp_path):
    import vector_map_artifacts as artifacts

    recipe = tmp_path / "input.json"
    recipe.write_text('{"schema_version": 1}')
    captured = {}
    config.load_recipe(recipe, snapshots=captured)
    recipe_before = recipe.read_bytes()
    recipe.write_text("changed after parse")
    opts = settings(include_mask=FIXTURE)
    inputs = artifacts.snapshot_inputs(opts, recipe, captured)
    prepared = raster.prepare(opts, input_bytes=captured)
    traced = adapter.trace_layer(prepared.material, opts.vtracer)
    normalized = svg.normalize_svg(traced.svg, prepared.canvas)
    files = {Path("out.svg"): normalized.svg.encode()}
    raw = artifacts.manifest_bytes(opts, prepared, normalized, inputs, files, list(prepared.diagnostics))
    result = json.loads(raw)
    assert result["schema_version"] == 1
    assert result["versions"] == {"wrapper": version("st-controlnet-helpers"), "vtracer": version("vtracer")}
    assert result["layers"] == [{"id": "standalone", "height_mm": None}]
    assert result["canvas"] == asdict(prepared.canvas)
    assert result["counts"] == asdict(normalized.metrics)
    assert result["preparation"]["input_kind"] == "mask"
    assert result["preparation"]["original_size"] == [128, 128]
    assert result["vtracer"] == dict(opts.vtracer)
    assert result["svg_limits"] == asdict(opts.svg_limits)
    by_role = {item["role"]: item for item in result["inputs"]}
    for role in ("source", "include-mask"):
        assert by_role[role]["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert by_role["recipe"]["sha256"] == hashlib.sha256(recipe_before).hexdigest()
    assert result["artifacts"] == [{
        "path": "out.svg", "sha256": hashlib.sha256(files[Path("out.svg")]).hexdigest(),
        "bytes": len(files[Path("out.svg")]),
    }]
    assert result["warnings"] == list(prepared.diagnostics)
    assert not ({"timestamp", "elapsed", "staging", "sha256"} & result.keys())
    assert raw == artifacts.manifest_bytes(opts, prepared, normalized, inputs, files, list(prepared.diagnostics))


def test_raster_decodes_snapshot_after_source_and_constraint_change(tmp_path):
    import vector_map_artifacts as artifacts

    source = tmp_path / "source.png"
    constraint = tmp_path / "constraint.png"
    source.write_bytes(FIXTURE.read_bytes())
    constraint.write_bytes(FIXTURE.read_bytes())
    opts = settings(source, include_mask=constraint, exclude_mask=None)
    captured = {}
    inputs = artifacts.snapshot_inputs(opts, None, captured)
    expected = raster.prepare(opts)
    source.write_bytes(b"replaced source")
    constraint.write_bytes(b"replaced constraint")
    actual = raster.prepare(opts, input_bytes=captured)
    assert (actual.material == expected.material).all()
    assert all(item["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest() for item in inputs)


def test_manifest_refuses_nonfinite_json():
    import vector_map_artifacts as artifacts

    with pytest.raises(ValueError):
        artifacts.json_bytes({"bad": float("nan")})


def test_artifact_module_is_packaged():
    import tomllib

    metadata = tomllib.loads((ROOT / "scripts/pyproject.toml").read_text())
    assert "vector_map_artifacts" in metadata["tool"]["setuptools"]["py-modules"]


def bundle_at(tmp_path, **kwargs):
    import vector_map_artifacts as artifacts

    return artifacts.Bundle(tmp_path / "out.svg", **kwargs)


@pytest.mark.parametrize("name", ["out.svg", "out.vector.json", "out.debug/mask.png", "out.debug/recipe.json"])
def test_preflight_refuses_each_existing_member(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"old")
    with pytest.raises(config.ConfigError, match="--overwrite"):
        bundle_at(tmp_path, debug=True).check()
    assert path.read_bytes() == b"old"


@pytest.mark.parametrize("role", ["source", "recipe", "include-mask", "exclude-mask"])
@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("name", ["out.svg", "out.vector.json", "out.debug/mask.png", "out.debug/recipe.json"])
def test_each_output_refuses_input_aliases(tmp_path, role, kind, name):
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    source = path if kind == "direct" else tmp_path / "input"
    source.write_bytes(b"input bytes")
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, path)
    with pytest.raises(config.ConfigError, match=role):
        bundle_at(tmp_path, debug=True, overwrite=True, inputs=[(role, source)]).check()
    assert source.read_bytes() == b"input bytes"


def test_missing_input_output_identity_is_refused(tmp_path):
    with pytest.raises(config.ConfigError, match="source"):
        bundle_at(tmp_path, inputs=[("source", tmp_path / "out.svg")]).check()


def test_debug_symlink_directory_is_refused(tmp_path):
    real = tmp_path / "unrelated"
    real.mkdir()
    (tmp_path / "out.debug").symlink_to(real, target_is_directory=True)
    with pytest.raises(config.ConfigError, match="symlink"):
        bundle_at(tmp_path, debug=True, overwrite=True).check()


@pytest.mark.parametrize("name", ["out.svg", "out.vector.json", "out.debug/mask.png", "out.debug/recipe.json"])
def test_output_directory_is_refused(tmp_path, name):
    (tmp_path / name).mkdir(parents=True)
    with pytest.raises(config.ConfigError, match="directory"):
        bundle_at(tmp_path, debug=True, overwrite=True).check()


def test_publication_order_and_unrelated_files(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path, debug=True, overwrite=True)
    directory = tmp_path / "out.debug"
    directory.mkdir()
    unrelated = directory / "notes.txt"
    unrelated.write_bytes(b"keep")
    manifest = tmp_path / "out.vector.json"
    manifest.write_bytes(b"old completion")
    files = {Path("out.svg"): b"svg", Path("out.debug/mask.png"): b"mask", Path("out.debug/recipe.json"): b"recipe"}
    events = []
    original_replace = os.replace

    def observe(source, target):
        target = Path(target)
        assert not manifest.exists()
        if target == manifest:
            assert all((tmp_path / name).read_bytes() == data for name, data in files.items())
        events.append(target.name)
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", observe)
    bundle.publish(files, b"new completion")
    assert events[-1] == "out.vector.json"
    assert manifest.read_bytes() == b"new completion"
    assert unrelated.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("failure", ["stage", "invalidate", "first", "later", "manifest"])
def test_failures_preserve_old_bundle_or_remove_completion(tmp_path, monkeypatch, failure):
    bundle = bundle_at(tmp_path, debug=True, overwrite=True)
    manifest = tmp_path / "out.vector.json"
    output = tmp_path / "out.svg"
    manifest.write_bytes(b"old completion")
    output.write_bytes(b"old svg")
    write = Path.write_bytes
    unlink = Path.unlink
    replace = os.replace
    calls = 0

    def fail_write(path, data):
        if failure == "stage":
            raise OSError("injected staging failure")
        return write(path, data)

    def fail_unlink(path, *args, **kwargs):
        if path == manifest and failure == "invalidate":
            raise OSError("injected invalidation failure")
        return unlink(path, *args, **kwargs)

    def fail_replace(source, target):
        nonlocal calls
        calls += 1
        if ((failure == "first" and calls == 1) or (failure == "later" and calls == 2)
                or (failure == "manifest" and Path(target) == manifest)):
            raise OSError("injected replacement failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        bundle.publish({Path("out.svg"): b"new svg", Path("out.debug/mask.png"): b"mask"}, b"new completion")
    if failure in ("stage", "invalidate"):
        assert manifest.read_bytes() == b"old completion"
        assert output.read_bytes() == b"old svg"
    else:
        assert not manifest.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("target_name", ["out.svg", "out.vector.json"])
def test_competing_file_after_preflight_survives(tmp_path, monkeypatch, target_name):
    bundle = bundle_at(tmp_path)
    bundle.check()
    link = os.link
    target = tmp_path / target_name

    def compete(staged, final):
        if Path(final) == target:
            target.write_bytes(b"competitor")
        return link(staged, final)

    monkeypatch.setattr(os, "link", compete)
    with pytest.raises(FileExistsError):
        bundle.publish({Path("out.svg"): b"our svg"}, b"our manifest")
    assert target.read_bytes() == b"competitor"
    if target_name == "out.svg":
        assert not (tmp_path / "out.vector.json").exists()


@pytest.mark.parametrize("error", [errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV])
def test_unsupported_links_explain_explicit_overwrite_retry(tmp_path, monkeypatch, error):
    def unsupported(*args):
        raise OSError(error, "hard links unavailable")

    monkeypatch.setattr(os, "link", unsupported)
    with pytest.raises(OSError, match="hard links.*--overwrite"):
        bundle_at(tmp_path).publish({Path("out.svg"): b"svg"}, b"manifest")
    assert not (tmp_path / "out.svg").exists()
    assert not (tmp_path / "out.vector.json").exists()
    bundle_at(tmp_path, overwrite=True).publish({Path("out.svg"): b"svg"}, b"manifest")
    assert (tmp_path / "out.svg").read_bytes() == b"svg"


@pytest.mark.parametrize("name", ["../escape", "unrelated.txt", "/tmp/escape"])
def test_publisher_rejects_unowned_members(tmp_path, name):
    with pytest.raises(ValueError, match="owned"):
        bundle_at(tmp_path).publish({Path(name): b"bad"}, b"manifest")


def cli(capsys, destination, *extra, source=FIXTURE):
    import vector_map

    code = vector_map.main([
        str(source), str(destination), "--input-kind", "mask", "--width-mm", "100", "--json", *map(str, extra),
    ])
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    return code, json.loads(captured.out), captured.err


def verify_manifest(path):
    data = json.loads(path.read_bytes())
    for member in data["artifacts"]:
        content = (path.parent / member["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == member["sha256"]
        assert len(content) == member["bytes"]
    return data


def test_cli_publishes_manifest_and_all_upstream_arguments(tmp_path, capsys):
    output = tmp_path / "out.svg"
    code, result, error = cli(capsys, output)
    assert code == 0, error
    assert result["artifacts"] == {"svg": str(output), "manifest": str(tmp_path / "out.vector.json")}
    manifest = verify_manifest(tmp_path / "out.vector.json")
    assert manifest["upstream"]["img_format"] == "png"
    assert manifest["upstream"]["colormode"] == "binary"
    assert manifest["upstream"]["mode"] == "polygon"
    assert manifest["counts"]["normalized_bytes"] == len(output.read_bytes())


@pytest.mark.parametrize("name", ["out.vector.json", "out.debug/mask.png", "out.debug/recipe.json"])
def test_cli_checks_all_collisions_before_prepare(tmp_path, capsys, monkeypatch, name):
    import vector_map

    collision = tmp_path / name
    collision.parent.mkdir(exist_ok=True)
    collision.write_bytes(b"old")
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation ran before collision refusal")
    monkeypatch.setattr(vector_map.raster, "prepare", forbidden)
    code, result, _ = cli(capsys, tmp_path / "out.svg", "--debug-bundle")
    assert code == 2
    assert "--overwrite" in result["diagnostics"][0]["message"]
    assert collision.read_bytes() == b"old"


@pytest.mark.parametrize("kind", ["mask", "edges"])
def test_debug_recipe_replays_prepared_material_without_repreparation(tmp_path, capsys, kind):
    import vector_map

    output = tmp_path / "out.svg"
    extra = ["--input-kind", kind, "--max-res", "64", "--invert"]
    if kind == "edges":
        extra += ["--line-width-mm", "4"]
    code, result, error = cli(capsys, output, "--debug-bundle", *extra)
    assert code == 0, error
    debug = tmp_path / "out.debug"
    assert result["artifacts"]["debug"] == str(debug)
    manifest = verify_manifest(tmp_path / "out.vector.json")
    assert {item["path"] for item in manifest["artifacts"]} == {"out.svg", "out.debug/mask.png", "out.debug/recipe.json"}
    recipe = json.loads((debug / "recipe.json").read_bytes())
    assert recipe["input"] == "mask.png"
    assert recipe["input_kind"] == "mask"
    assert not recipe.get("invert", False)
    assert "line_width_mm" not in recipe
    replay = tmp_path / "replay.svg"
    code = vector_map.main([str(replay), "--recipe", str(debug / "recipe.json"), "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert replay.read_bytes() == output.read_bytes()


def test_tracing_failure_retains_debug_and_retry_needs_overwrite(tmp_path, capsys, monkeypatch):
    import vector_map

    trace = vector_map.adapter.trace_layer
    def failure(*args, **kwargs):
        raise RuntimeError("native tracing failed")
    monkeypatch.setattr(vector_map.adapter, "trace_layer", failure)
    code, result, error = cli(capsys, tmp_path / "out.svg", "--debug-bundle")
    assert code == 1
    assert result["status"] == "failed"
    assert result["artifacts"] == {"svg": None, "debug": str(tmp_path / "out.debug")}
    assert "native tracing failed" in error
    assert "0.6.15" in error and "standalone" in error
    assert "Rerun with --overwrite" in error
    assert (tmp_path / "out.debug/mask.png").is_file()
    assert (tmp_path / "out.debug/recipe.json").is_file()
    assert not (tmp_path / "out.vector.json").exists()
    assert not (tmp_path / "out.svg").exists()
    monkeypatch.setattr(vector_map.adapter, "trace_layer", trace)
    assert cli(capsys, tmp_path / "out.svg", "--debug-bundle")[0] == 2
    unrelated = tmp_path / "out.debug/keep.txt"
    unrelated.write_text("keep")
    code, _, error = cli(capsys, tmp_path / "out.svg", "--debug-bundle", "--overwrite")
    assert code == 0, error
    assert unrelated.read_text() == "keep"
    verify_manifest(tmp_path / "out.vector.json")


def test_failed_debug_run_invalidates_old_manifest_before_debug_replacement(tmp_path, capsys, monkeypatch):
    import vector_map

    output = tmp_path / "out.svg"
    assert cli(capsys, output, "--debug-bundle")[0] == 0
    old_svg = output.read_bytes()
    manifest = tmp_path / "out.vector.json"
    def failure(*args, **kwargs):
        raise ValueError("bad upstream geometry")
    monkeypatch.setattr(vector_map.adapter, "trace_layer", failure)
    replace = os.replace
    seen = []
    def observe(source, final):
        assert not manifest.exists()
        seen.append(Path(final).name)
        replace(source, final)
    monkeypatch.setattr(os, "replace", observe)
    code, result, _ = cli(capsys, output, "--debug-bundle", "--overwrite")
    assert code == 1
    assert result["artifacts"]["debug"] == str(tmp_path / "out.debug")
    assert set(seen) == {"mask.png", "recipe.json"}
    assert output.read_bytes() == old_svg
    assert not manifest.exists()


def test_debug_publication_failure_preserves_original_error(tmp_path, capsys, monkeypatch):
    import vector_map
    import vector_map_artifacts as artifacts

    def failure(*args, **kwargs):
        raise RuntimeError("original native error")
    def publication_failure(*args, **kwargs):
        raise OSError("debug disk error")
    monkeypatch.setattr(vector_map.adapter, "trace_layer", failure)
    monkeypatch.setattr(artifacts.Bundle, "publish", publication_failure)
    code, result, error = cli(capsys, tmp_path / "out.svg", "--debug-bundle")
    assert code == 1
    assert "original native error" in error and "debug disk error" in error
    assert "debug" not in result["artifacts"]
    assert "Debug files retained" not in error


def test_three_runs_save_bytes_before_overwrite(tmp_path, capsys):
    output = tmp_path / "out.svg"
    snapshots = []
    for index in range(3):
        flags = ["--debug-bundle"] + (["--overwrite"] if index else [])
        code, _, error = cli(capsys, output, *flags)
        assert code == 0, error
        manifest = verify_manifest(tmp_path / "out.vector.json")
        paths = [tmp_path / item["path"] for item in manifest["artifacts"]] + [tmp_path / "out.vector.json"]
        snapshots.append({str(path): path.read_bytes() for path in paths})
    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_cleanup_failure_after_manifest_link_invalidates_completion(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path)
    manifest = tmp_path / "out.vector.json"
    unlink = Path.unlink
    injected = False

    def fail_once(path, *args, **kwargs):
        nonlocal injected
        if manifest.exists() and path.name == "1" and not injected:
            injected = True
            raise OSError("injected cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(OSError, match="cleanup failure"):
        bundle.publish({Path("out.svg"): b"svg"}, b"completion")
    assert injected
    assert not manifest.exists()


def test_output_alias_to_other_output_is_refused(tmp_path):
    output = tmp_path / "out.svg"
    output.write_bytes(b"old")
    (tmp_path / "out.vector.json").symlink_to(output)
    with pytest.raises(config.ConfigError, match="another output"):
        bundle_at(tmp_path, overwrite=True).check()


# --- S2.7 preview in the bundle (STABL-kfrksmnp) ------------------------------


def test_manifest_preview_is_null_without_preview(tmp_path, capsys):
    code, result, error = cli(capsys, tmp_path / "out.svg")
    assert code == 0, error
    assert "preview" not in result["artifacts"]
    assert verify_manifest(tmp_path / "out.vector.json")["preview"] is None
    assert not (tmp_path / "out.preview.png").exists()


@pytest.mark.parametrize("extra", [[], ["--input-kind", "edges", "--line-width-mm", "2"]])
def test_cli_publishes_hashed_preview_with_provenance(tmp_path, capsys, extra):
    pytest.importorskip("resvg_py")
    output = tmp_path / "out.svg"
    code, result, error = cli(capsys, output, "--preview", *extra)
    assert code == 0, error
    preview_path = tmp_path / "out.preview.png"
    assert result["artifacts"]["preview"] == str(preview_path)
    manifest = verify_manifest(tmp_path / "out.vector.json")
    assert "out.preview.png" in {item["path"] for item in manifest["artifacts"]}
    assert manifest["preview"]["renderer_version"] == version("resvg-py")
    assert manifest["preview"]["root"] == "pixel"
    assert manifest["preview"]["resolution"] == manifest["preparation"]["processed_size"]
    assert preview_path.read_bytes().startswith(b"\x89PNG")


def test_existing_preview_collides_before_prepare(tmp_path, capsys, monkeypatch):
    import vector_map

    collision = tmp_path / "out.preview.png"
    collision.write_bytes(b"old")
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation ran before collision refusal")
    monkeypatch.setattr(vector_map.raster, "prepare", forbidden)
    code, result, _ = cli(capsys, tmp_path / "out.svg", "--preview")
    assert code == 2
    assert "--overwrite" in result["diagnostics"][0]["message"]
    assert collision.read_bytes() == b"old"


def test_unrelated_preview_file_is_kept_without_preview(tmp_path, capsys):
    unrelated = tmp_path / "out.preview.png"
    unrelated.write_bytes(b"keep")
    code, _, error = cli(capsys, tmp_path / "out.svg", "--overwrite")
    assert code == 0, error
    assert unrelated.read_bytes() == b"keep"


def test_renderer_failure_publishes_nothing(tmp_path, capsys, monkeypatch):
    import vector_map

    def failure(*args, **kwargs):
        raise ValueError("SVG has an invalid size")
    monkeypatch.setattr(vector_map.preview, "render_luminance", failure)
    code, result, error = cli(capsys, tmp_path / "out.svg", "--preview")
    assert code == 1
    assert result["status"] == "failed"
    assert "resvg-py" in error and "invalid size" in error
    assert "Traceback" not in error
    assert list(tmp_path.iterdir()) == []


def test_renderer_failure_retains_debug_bundle(tmp_path, capsys, monkeypatch):
    import vector_map

    def failure(*args, **kwargs):
        raise ValueError("render failed")
    monkeypatch.setattr(vector_map.preview, "render_luminance", failure)
    code, result, error = cli(capsys, tmp_path / "out.svg", "--preview", "--debug-bundle")
    assert code == 1
    assert result["artifacts"] == {"svg": None, "debug": str(tmp_path / "out.debug")}
    assert (tmp_path / "out.debug/mask.png").is_file()
    assert not (tmp_path / "out.svg").exists()
    assert not (tmp_path / "out.preview.png").exists()
    assert not (tmp_path / "out.vector.json").exists()


def test_preview_runs_repeat_bytes(tmp_path, capsys):
    pytest.importorskip("resvg_py")
    output = tmp_path / "out.svg"
    snapshots = []
    for index in range(2):
        code, _, error = cli(capsys, output, "--preview", *(["--overwrite"] if index else []))
        assert code == 0, error
        snapshots.append({name: (tmp_path / name).read_bytes() for name in ("out.svg", "out.preview.png", "out.vector.json")})
    assert snapshots[0] == snapshots[1]


def test_preview_uses_recipe_source_snapshot(tmp_path, capsys):
    import vector_map

    pytest.importorskip("resvg_py")
    source = tmp_path / "in.png"
    source.write_bytes(FIXTURE.read_bytes())
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "input": "in.png", "input_kind": "mask", "width_mm": 40}))
    code = vector_map.main([str(tmp_path / "out.svg"), "--recipe", str(recipe), "--preview", "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    manifest = verify_manifest(tmp_path / "out.vector.json")
    assert "out.preview.png" in {item["path"] for item in manifest["artifacts"]}


# --- S3.3 layered bundle (STABL-qlagdbmh) -------------------------------------

LAYER_NAMES = ["out.layers/silhouette.svg", "out.layers/structure.svg", "out.layers/detail.svg"]
LAYER_SHAPE = (24, 32)


def layered_scene(tmp_path):
    """Source, silhouette mask, and structure and detail maps. One mm per pixel."""
    import numpy as np
    from PIL import Image

    def save(name, material):
        path = tmp_path / name
        Image.fromarray(np.where(material, 255, 0).astype(np.uint8)).save(path)
        return path

    silhouette = np.zeros(LAYER_SHAPE, bool)
    silhouette[2:22, 2:30] = True
    structure = np.zeros(LAYER_SHAPE, bool)
    structure[6:10, 4:28] = True
    detail = np.zeros(LAYER_SHAPE, bool)
    detail[14:18, 5:25] = True
    source = tmp_path / "source.png"
    Image.fromarray(np.dstack([np.where(silhouette, 220, 20).astype(np.uint8)] * 3)).save(source)
    return {"source": source, "mask": save("mask.png", silhouette),
            "structure": save("structure.png", structure), "detail": save("detail.png", detail)}


def layered_cli(capsys, tmp_path, scene, *extra, layers="silhouette,structure,detail", destination="out.svg"):
    import vector_map

    maps = []
    if "structure" in layers:
        maps += ["--structure-map", scene["structure"]]
    if "detail" in layers:
        maps += ["--detail-map", scene["detail"]]
    code = vector_map.main([str(value) for value in (
        scene["source"], tmp_path / destination, "--input-kind", "image", "--mask", scene["mask"],
        "--width-mm", "32", "--layers", layers, *maps, "--json", *extra)])
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    return code, json.loads(captured.out), captured.err


@pytest.mark.parametrize("name", LAYER_NAMES)
def test_layered_preflight_refuses_each_owned_layer_name(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"old")
    with pytest.raises(config.ConfigError, match="--overwrite"):
        bundle_at(tmp_path, layered=True).check()
    assert path.read_bytes() == b"old"


def test_standalone_bundle_does_not_own_layer_names(tmp_path):
    (tmp_path / "out.layers").mkdir()
    (tmp_path / "out.layers/structure.svg").write_bytes(b"old")
    bundle_at(tmp_path).check()


@pytest.mark.parametrize("name", ["out.layers/structure.svg", "out.layers/detail.svg"])
def test_silhouette_only_run_refuses_unselected_owned_names_before_prepare(tmp_path, capsys, monkeypatch, name):
    import vector_map

    scene = layered_scene(tmp_path)
    collision = tmp_path / name
    collision.parent.mkdir(exist_ok=True)
    collision.write_bytes(b"old")
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation ran before collision refusal")
    monkeypatch.setattr(vector_map.layer_prep, "prepare_layers", forbidden)
    code, result, _ = layered_cli(capsys, tmp_path, scene, layers="silhouette")
    assert code == 2
    assert "--overwrite" in result["diagnostics"][0]["message"]
    assert collision.read_bytes() == b"old"


def test_layer_symlink_directory_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "out.layers").symlink_to(target, target_is_directory=True)
    with pytest.raises(config.ConfigError, match="symlink"):
        bundle_at(tmp_path, layered=True, overwrite=True).check()


def test_layer_path_that_is_not_a_directory_is_refused(tmp_path):
    (tmp_path / "out.layers").write_bytes(b"file")
    with pytest.raises(config.ConfigError, match="directory"):
        bundle_at(tmp_path, layered=True, overwrite=True).check()


@pytest.mark.parametrize("role", ["source", "mask", "structure-map", "detail-map", "structure-include-mask",
                                  "detail-exclude-mask"])
@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("name", ["out.svg", *LAYER_NAMES])
def test_each_layered_output_refuses_input_aliases(tmp_path, role, kind, name):
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    source = path if kind == "direct" else tmp_path / "input"
    source.write_bytes(b"input bytes")
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, path)
    with pytest.raises(config.ConfigError, match=role):
        bundle_at(tmp_path, layered=True, overwrite=True, inputs=[(role, source)]).check()
    assert source.read_bytes() == b"input bytes"


def test_cli_refuses_a_role_map_that_is_an_owned_output(tmp_path, capsys):
    scene = layered_scene(tmp_path)
    assert layered_cli(capsys, tmp_path, scene)[0] == 0
    owned = tmp_path / "out.layers/detail.svg"
    before = owned.read_bytes()
    scene["detail"] = owned
    code, _, error = layered_cli(capsys, tmp_path, scene, "--overwrite")
    assert code == 2
    assert "detail-map" in error
    assert owned.read_bytes() == before


def test_publisher_rejects_unowned_layer_names(tmp_path):
    with pytest.raises(ValueError, match="owned"):
        bundle_at(tmp_path, layered=True).publish({Path("out.layers/other.svg"): b"bad"}, b"manifest")


def test_overwrite_removes_unselected_owned_layers_and_keeps_unrelated_files(tmp_path, capsys):
    scene = layered_scene(tmp_path)
    assert layered_cli(capsys, tmp_path, scene)[0] == 0
    unrelated = [tmp_path / "out.layers/notes.txt", tmp_path / "other.svg"]
    for path in unrelated:
        path.write_bytes(b"keep")
    code, result, error = layered_cli(capsys, tmp_path, scene, "--overwrite", layers="silhouette,detail")
    assert code == 0, error
    assert not (tmp_path / "out.layers/structure.svg").exists()
    assert (tmp_path / "out.layers").is_dir()
    assert all(path.read_bytes() == b"keep" for path in unrelated)
    manifest = verify_manifest(tmp_path / "out.vector.json")
    assert {item["path"] for item in manifest["artifacts"]} == {
        "out.svg", "out.layers/silhouette.svg", "out.layers/detail.svg"}
    assert result["artifacts"]["layers"] == {
        role: str(tmp_path / f"out.layers/{role}.svg") for role in ("silhouette", "detail")}


def test_overwrite_removes_a_layer_that_became_empty(tmp_path, capsys):
    import numpy as np
    from PIL import Image

    scene = layered_scene(tmp_path)
    assert layered_cli(capsys, tmp_path, scene)[0] == 0
    Image.fromarray(np.zeros(LAYER_SHAPE, np.uint8)).save(scene["detail"])
    code, _, error = layered_cli(capsys, tmp_path, scene, "--overwrite")
    assert code == 0, error
    assert not (tmp_path / "out.layers/detail.svg").exists()
    manifest = verify_manifest(tmp_path / "out.vector.json")
    assert "out.layers/detail.svg" not in {item["path"] for item in manifest["artifacts"]}
    assert [(layer["id"], layer["svg"]) for layer in manifest["layers"]] == [
        ("silhouette", "out.layers/silhouette.svg"), ("structure", "out.layers/structure.svg"), ("detail", None)]


def test_stale_layer_removal_follows_invalidation_and_precedes_completion(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path, layered=True, overwrite=True)
    (tmp_path / "out.layers").mkdir()
    manifest = tmp_path / "out.vector.json"
    stale = tmp_path / "out.layers/structure.svg"
    stale.write_bytes(b"old structure")
    manifest.write_bytes(b"old completion")
    events = []
    unlink, replace = Path.unlink, os.replace

    def observe_unlink(path, *args, **kwargs):
        if path in (manifest, stale):
            events.append(("unlink", path.name, manifest.exists()))
        return unlink(path, *args, **kwargs)

    def observe_replace(source, target):
        events.append(("replace", Path(target).name, stale.exists()))
        return replace(source, target)

    monkeypatch.setattr(Path, "unlink", observe_unlink)
    monkeypatch.setattr(os, "replace", observe_replace)
    bundle.publish({Path("out.svg"): b"svg", Path("out.layers/silhouette.svg"): b"silhouette"}, b"completion")
    assert events[0] == ("unlink", "out.vector.json", True)
    assert ("unlink", "structure.svg", False) in events
    assert events[-1] == ("replace", "out.vector.json", False)
    assert not stale.exists()
    assert manifest.read_bytes() == b"completion"


@pytest.mark.parametrize("failure", ["stage", "invalidate", "layer", "combined", "stale", "manifest"])
def test_layered_failures_preserve_old_bundle_or_remove_completion(tmp_path, monkeypatch, failure):
    bundle = bundle_at(tmp_path, layered=True, overwrite=True)
    (tmp_path / "out.layers").mkdir()
    manifest = tmp_path / "out.vector.json"
    old = {tmp_path / "out.svg": b"old svg", tmp_path / "out.layers/silhouette.svg": b"old silhouette",
           tmp_path / "out.layers/structure.svg": b"old structure", manifest: b"old completion"}
    for path, data in old.items():
        path.write_bytes(data)
    write, unlink, replace = Path.write_bytes, Path.unlink, os.replace

    def fail_write(path, data):
        if failure == "stage":
            raise OSError("injected staging failure")
        return write(path, data)

    def fail_unlink(path, *args, **kwargs):
        if (failure == "invalidate" and path == manifest) or (failure == "stale" and path.name == "structure.svg"):
            raise OSError("injected unlink failure")
        return unlink(path, *args, **kwargs)

    def fail_replace(source, target):
        name = Path(target).name
        if ((failure == "layer" and name == "silhouette.svg") or (failure == "combined" and name == "out.svg")
                or (failure == "manifest" and name == "out.vector.json")):
            raise OSError("injected replacement failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        bundle.publish({Path("out.svg"): b"new svg", Path("out.layers/silhouette.svg"): b"new silhouette"},
                       b"new completion")
    if failure in ("stage", "invalidate"):
        assert all(path.read_bytes() == data for path, data in old.items())
    else:
        assert not manifest.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_layered_runs_repeat_bytes(tmp_path, capsys):
    scene = layered_scene(tmp_path)
    snapshots = []
    for index in range(2):
        code, _, error = layered_cli(capsys, tmp_path, scene, *(["--overwrite"] if index else []))
        assert code == 0, error
        manifest = verify_manifest(tmp_path / "out.vector.json")
        paths = [tmp_path / item["path"] for item in manifest["artifacts"]] + [tmp_path / "out.vector.json"]
        snapshots.append({str(path): path.read_bytes() for path in paths})
    assert snapshots[0] == snapshots[1]


def test_layered_manifest_records_roles_inputs_and_counts(tmp_path, capsys):
    scene = layered_scene(tmp_path)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "structure": {"width_mm": 2.0, "gap_close_mm": 1.0}}))
    code, result, error = layered_cli(capsys, tmp_path, scene, "--recipe", recipe)
    assert code == 0, error
    manifest = verify_manifest(tmp_path / "out.vector.json")
    assert manifest["schema_version"] == 1
    assert manifest["preparation"]["layers"] == ["silhouette", "structure", "detail"]
    assert [entry["role"] for entry in manifest["inputs"]] == [
        "source", "recipe", "mask", "structure-map", "detail-map"]
    assert {item["path"] for item in manifest["artifacts"]} == {"out.svg", *LAYER_NAMES}
    layers = manifest["layers"]
    assert [(layer["id"], layer["height_mm"], layer["mode"]) for layer in layers] == [
        ("silhouette", None, None), ("structure", None, None), ("detail", None, None)]
    assert all(layer["counts"]["paths"] >= 1 for layer in layers)
    structure = manifest["roles"]["structure"]
    assert (structure["source"], structure["path"]) == ("map", str(scene["structure"]))
    assert (structure["width_mm"], structure["gap_close_mm"]) == (2.0, 1.0)
    assert structure["gap_closing"]["radius_px"] == 1
    assert structure["expansion"]["kernel_size_px"] == 3
    assert manifest["counts"]["layers"] == 3
    assert manifest["counts"]["paths"] == sum(layer["counts"]["paths"] for layer in layers) == result["counts"]["paths"]
    assert manifest["counts"]["combined_bytes"] == len((tmp_path / "out.svg").read_bytes())
    assert "time" not in json.dumps(manifest).lower()


def test_layered_manifest_records_resolved_canny_with_shared_blur(tmp_path, capsys):
    import vector_map

    scene = layered_scene(tmp_path)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "detail": {"canny": {"low_threshold": 40,
                                                                             "high_threshold": 90}}}))
    code = vector_map.main([str(value) for value in (
        scene["source"], tmp_path / "out.svg", "--input-kind", "image", "--mask", scene["mask"], "--width-mm", "32",
        "--layers", "silhouette,structure,detail", "--recipe", recipe, "--json")])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    roles = verify_manifest(tmp_path / "out.vector.json")["roles"]
    assert roles["structure"]["canny"] == {"low_threshold": 100, "high_threshold": 200, "blur": 3}
    assert roles["detail"]["canny"] == {"low_threshold": 40, "high_threshold": 90, "blur": 3}
    assert roles["detail"]["source"] == "canny" and roles["detail"]["path"] is None
