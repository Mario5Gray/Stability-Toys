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
