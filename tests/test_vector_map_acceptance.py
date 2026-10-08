"""S2.8 standalone acceptance: repeatability and end-to-end mask/edge evidence. STABL-ntgnbxci.

Spec 10 repeatability row and 11.2 exit evidence. Real vtracer==0.6.15, real resvg-py==0.5.0,
and the real st-canny-map command. Mocks are not conversion proof.

Set ST_VECTOR_MAP_BIN to a directory with installed st-vector-map and st-canny-map console
scripts to run the command checks against an installed package. Default: source scripts.
Set VECTOR_MAP_ACCEPTANCE_EVIDENCE to a JSON path to save per-case evidence for the report.
No CAD acceptance gate. OpenSCAD checks live in test_vector_map_openscad.py.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vtracer", reason="vtracer==0.6.15 is not installed. Install the vector extra.")
pytest.importorskip("resvg_py", reason="resvg-py==0.5.0 is not installed. Install the vector extra.")
pytest.importorskip("cv2", reason="opencv is not installed. Install the vector extra.")

from tests.fixtures.vector_map import make_fixtures  # noqa: E402
from tests.fixtures.vector_map.corpus import metrics  # noqa: E402
from tests.test_vector_map_corpus import FAILING_AT_EVERY_VALUE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
FIXTURES = ROOT / "tests" / "fixtures" / "vector_map"
CORPUS = FIXTURES / "corpus"

import vector_map  # noqa: E402
import vector_map_config as config  # noqa: E402
import vector_map_preview as preview  # noqa: E402

TOPOLOGY = sorted(make_fixtures.EXPECTED)
CORPUS_CASES = sorted(path.name for path in CORPUS.iterdir() if (path / "mask.png").is_file())
MASK_CASES = [("topology", name) for name in TOPOLOGY] + [("corpus", name) for name in CORPUS_CASES]
EDGE_SETTINGS = ["--input-kind", "edges", "--width-mm", "100", "--line-width-mm", "0.8"]
MASK_SETTINGS = ["--input-kind", "mask", "--width-mm", "100"]
RUNS = 3
EVIDENCE = []


@pytest.fixture(scope="module", autouse=True)
def _save_evidence():
    yield
    target = os.environ.get("VECTOR_MAP_ACCEPTANCE_EVIDENCE")
    if target:
        Path(target).write_text(json.dumps({
            "versions": _versions(), "python": sys.version.split()[0], "platform": sys.platform,
            "commands": _commands(), "cases": EVIDENCE,
        }, indent=2, sort_keys=True, default=_plain) + "\n")


def _plain(value):
    """numpy scalars from measurements become plain JSON numbers."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _versions():
    names = ("st-controlnet-helpers", "vtracer", "resvg-py", "Pillow", "numpy", "opencv-python-headless", "opencv-python")
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except Exception:
            found[name] = None
    return found


def _commands():
    bin_dir = os.environ.get("ST_VECTOR_MAP_BIN")
    if bin_dir:
        return {"vector": [str(Path(bin_dir) / "st-vector-map")], "canny": [str(Path(bin_dir) / "st-canny-map")]}
    return {"vector": [sys.executable, str(SCRIPTS / "vector_map.py")],
            "canny": [sys.executable, str(SCRIPTS / "canny_map.py")]}


def _record(**values):
    EVIDENCE.append(values)
    return values


# --- Helpers ----------------------------------------------------------------


def _in_process(capsys, monkeypatch, *argv):
    """Run the console-script main(). Spy which publication primitive moves files."""
    calls = {"link": 0, "replace": 0}
    link, replace = os.link, os.replace

    def spy_link(*args, **kwargs):
        calls["link"] += 1
        return link(*args, **kwargs)

    def spy_replace(*args, **kwargs):
        calls["replace"] += 1
        return replace(*args, **kwargs)

    monkeypatch.setattr(os, "link", spy_link)
    monkeypatch.setattr(os, "replace", spy_replace)
    start = time.perf_counter()
    code = vector_map.main([*map(str, argv), "--json"])
    elapsed = time.perf_counter() - start
    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "replace", replace)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    return code, json.loads(lines[0]), captured.err, calls, elapsed


def _bundle_files(destination):
    """Every bundle member on disk, by name relative to the destination directory."""
    parent = destination.parent
    names = [destination, destination.with_suffix(".vector.json"), destination.with_suffix(".preview.png")]
    debug = destination.with_suffix(".debug")
    names += [debug / "mask.png", debug / "recipe.json"]
    return {path.relative_to(parent).as_posix(): path.read_bytes() for path in names if path.exists()}


def _verify_manifest(manifest_path):
    """Independent of the publisher: hash every listed file with hashlib here."""
    manifest = json.loads(manifest_path.read_bytes())
    for member in manifest["artifacts"]:
        data = (manifest_path.parent / member["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == member["sha256"], member["path"]
        assert len(data) == member["bytes"], member["path"]
    return manifest


def _canny(source, destination):
    command = _commands()["canny"]
    start = time.perf_counter()
    result = subprocess.run([*command, str(source), str(destination)], capture_output=True, text=True, cwd=ROOT)
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, result.stderr
    assert destination.is_file()
    return elapsed


def _geometry(manifest, destination):
    """Crisp render of the published SVG against the prepared material from the debug bundle."""
    canvas = config.Canvas(**manifest["canvas"])
    material = np.asarray(Image.open(destination.with_suffix(".debug") / "mask.png").convert("L")) >= 128
    svg = destination.read_text()
    rendered = preview.render_material(svg, canvas)
    return metrics.evaluate(material, rendered, metrics.polygon_area(svg)), material, rendered


def _repeat(capsys, monkeypatch, tmp_path, source, settings):
    """Three runs at one destination. Save bytes after each run, before the next starts."""
    destination = tmp_path / "out" / "result.svg"
    snapshots, runs = [], []
    for index in range(RUNS):
        extra = ["--overwrite"] if index else []
        code, result, error, calls, elapsed = _in_process(
            capsys, monkeypatch, source, destination, *settings, "--preview", "--debug-bundle", *extra,
        )
        assert code == 0, error
        manifest = _verify_manifest(destination.with_suffix(".vector.json"))
        snapshots.append(_bundle_files(destination))
        runs.append({"exit": code, "seconds": round(elapsed, 4), "publication": calls})
        if index == 0:
            assert calls["link"] > 0 and calls["replace"] == 0, calls
        else:
            assert calls["replace"] > 0 and calls["link"] == 0, calls
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert set(snapshots[0]) == {
        "result.svg", "result.vector.json", "result.preview.png", "result.debug/mask.png", "result.debug/recipe.json",
    }
    return destination, manifest, result, runs, snapshots[0]


# --- Steps 1, 4-6, 10: mask mode repeatability --------------------------------


@pytest.mark.parametrize("group, name", MASK_CASES, ids=[f"{g}-{n}" for g, n in MASK_CASES])
def test_mask_conversion_repeats_and_records_evidence(capsys, monkeypatch, tmp_path, group, name):
    source = FIXTURES / f"{name}.png" if group == "topology" else CORPUS / name / "mask.png"
    destination, manifest, result, runs, files = _repeat(capsys, monkeypatch, tmp_path, source, MASK_SETTINGS)
    geometry, material, rendered = _geometry(manifest, destination)
    if group == "topology":
        assert list(metrics.topology(rendered)) == list(make_fixtures.EXPECTED[name])
    else:
        assert geometry["failures"] == FAILING_AT_EVERY_VALUE.get(name, []), geometry
    _record(
        mode="mask", group=group, case=name, runs=runs, identical_runs=RUNS,
        canvas=manifest["canvas"], processed_size=manifest["preparation"]["processed_size"],
        counts=manifest["counts"], warnings=[w["code"] for w in manifest["warnings"]],
        hashes={path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
        geometry={"failures": geometry["failures"], "xor_px": geometry.get("xor_px"),
                  "topology_mask": geometry["topology_mask"]},
        paths=result["counts"]["paths"],
    )


# --- Steps 2-6, 10: actual st-canny-map output in edge mode -------------------


@pytest.mark.parametrize("name", CORPUS_CASES)
def test_real_canny_output_converts_in_edge_mode_and_repeats(capsys, monkeypatch, tmp_path, name):
    edges = tmp_path / "canny" / f"{name}.png"
    canny_seconds = _canny(CORPUS / name / "source.png", edges)
    destination, manifest, result, runs, files = _repeat(capsys, monkeypatch, tmp_path, edges, EDGE_SETTINGS)
    geometry, material, rendered = _geometry(manifest, destination)
    assert manifest["preparation"]["input_kind"] == "edges"
    assert manifest["preparation"]["expansion"]["kernel_size_px"] >= 1
    _record(
        mode="edges", group="canny", case=name, canny_seconds=round(canny_seconds, 4), runs=runs,
        identical_runs=RUNS, canvas=manifest["canvas"], expansion=manifest["preparation"]["expansion"],
        counts=manifest["counts"], warnings=[w["code"] for w in manifest["warnings"]],
        hashes={path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
        geometry={"failures": geometry["failures"], "xor_px": geometry.get("xor_px"),
                  "topology_mask": geometry["topology_mask"]},
        paths=result["counts"]["paths"],
    )


@pytest.mark.parametrize("name", ["donut", "asymmetric"])
def test_real_command_matches_in_process_bytes(capsys, monkeypatch, tmp_path, name):
    """The subprocess command, source or installed, publishes the same bundle bytes."""
    source = FIXTURES / f"{name}.png"
    command = _commands()["vector"]
    by_command = tmp_path / "command" / "result.svg"
    result = subprocess.run(
        [*command, str(source), str(by_command), *MASK_SETTINGS, "--preview", "--debug-bundle", "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "converted"
    _verify_manifest(by_command.with_suffix(".vector.json"))
    in_process = tmp_path / "in_process" / "result.svg"
    code, *_ = _in_process(capsys, monkeypatch, source, in_process, *MASK_SETTINGS, "--preview", "--debug-bundle")
    assert code == 0
    assert _bundle_files(by_command) == _bundle_files(in_process)
    _record(mode="mask", group="command", case=name, command=command[-1], exit=result.returncode)


# --- Step 7: coordinates ------------------------------------------------------


def _save(path, pixels, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, np.uint8)).save(path, **kwargs)
    return path


def _convert(capsys, monkeypatch, source, destination, *extra):
    code, result, error, _, elapsed = _in_process(capsys, monkeypatch, source, destination, *extra)
    return code, result, error, elapsed


def _published(destination):
    manifest = _verify_manifest(destination.with_suffix(".vector.json"))
    canvas = config.Canvas(**manifest["canvas"])
    rendered = preview.render_material(destination.read_text(), canvas)
    material = np.asarray(Image.open(destination.with_suffix(".debug") / "mask.png").convert("L")) >= 128
    return manifest, canvas, rendered, material


def _asymmetric(rows=120):
    return np.asarray(Image.open(FIXTURES / "asymmetric.png").convert("L"))[:rows]


def test_exif_orientation_applies_before_tracing(capsys, monkeypatch, tmp_path):
    oriented = _asymmetric()
    stored = np.asarray(Image.fromarray(oriented).transpose(Image.Transpose.ROTATE_90))
    exif = Image.Exif()
    exif[274] = 6
    source = _save(tmp_path / "rotated.png", stored, exif=exif)
    destination = tmp_path / "out.svg"
    code, _, error, elapsed = _convert(capsys, monkeypatch, source, destination, *MASK_SETTINGS, "--debug-bundle")
    assert code == 0, error
    manifest, _, rendered, material = _published(destination)
    assert manifest["preparation"]["original_size"] == [120, 128]
    assert manifest["preparation"]["oriented_size"] == [128, 120]
    assert np.array_equal(material, oriented >= 128)
    error_px = np.count_nonzero(rendered ^ material)
    for flipped in (np.fliplr(material), np.flipud(material), np.rot90(material, 2)):
        assert error_px < np.count_nonzero(rendered ^ flipped)
    _record(mode="mask", group="coordinates", case="exif_orientation_6", exit=code, seconds=round(elapsed, 4),
            original_size=[120, 128], oriented_size=[128, 120], xor_px=error_px)


def test_constraint_mask_with_other_dimensions_is_rejected(capsys, monkeypatch, tmp_path):
    include = _save(tmp_path / "include.png", np.full((64, 64), 255))
    destination = tmp_path / "out" / "out.svg"
    code, result, error, _ = _convert(capsys, monkeypatch, FIXTURES / "asymmetric.png", destination,
                                      *MASK_SETTINGS, "--include-mask", include)
    assert code == 2
    assert "oriented dimensions" in error
    assert not destination.parent.exists() or not any(destination.parent.iterdir())
    _record(mode="mask", group="coordinates", case="constraint_size_mismatch", exit=code,
            message=result["diagnostics"][0]["message"])


def test_common_resize_applies_one_scale_to_source_and_constraints(capsys, monkeypatch, tmp_path):
    allowed = np.zeros((128, 128), np.uint8)
    allowed[:, :64] = 255
    include = _save(tmp_path / "left.png", allowed)
    destination = tmp_path / "out.svg"
    code, _, error, _ = _convert(capsys, monkeypatch, FIXTURES / "asymmetric.png", destination, *MASK_SETTINGS,
                                 "--max-res", "64", "--include-mask", include, "--debug-bundle")
    assert code == 0, error
    manifest, canvas, rendered, material = _published(destination)
    assert manifest["preparation"]["processed_size"] == [64, 64]
    assert (canvas.width_px, canvas.height_px, canvas.width_mm, canvas.mm_per_px) == (64, 64, 100, 100 / 64)
    assert material.shape == (64, 64)
    assert not material[:, 32:].any() and material[:, :32].any()
    assert not rendered[:, 32:].any()
    _record(mode="mask", group="coordinates", case="common_resize_max_res_64", exit=code,
            processed_size=[64, 64], mm_per_px=canvas.mm_per_px)


def test_rectangle_has_known_physical_size(capsys, monkeypatch, tmp_path):
    pixels = np.zeros((100, 200), np.uint8)
    pixels[20:70, 40:140] = 255
    source = _save(tmp_path / "rectangle.png", pixels)
    destination = tmp_path / "out.svg"
    code, _, error, _ = _convert(capsys, monkeypatch, source, destination,
                                 "--input-kind", "mask", "--width-mm", "50", "--debug-bundle")
    assert code == 0, error
    manifest, canvas, rendered, _ = _published(destination)
    svg = destination.read_text()
    assert 'width="50mm"' in svg and 'height="25mm"' in svg and 'viewBox="0 0 200 100"' in svg
    rows, cols = np.nonzero(rendered)
    size_mm = ((cols.max() + 1 - cols.min()) * canvas.mm_per_px, (rows.max() + 1 - rows.min()) * canvas.mm_per_px)
    origin_mm = (cols.min() * canvas.mm_per_px, rows.min() * canvas.mm_per_px)
    assert size_mm == (25.0, 12.5) and origin_mm == (10.0, 5.0)
    assert np.count_nonzero(rendered) == 100 * 50
    _record(mode="mask", group="coordinates", case="physical_rectangle", exit=code,
            canvas_mm=[canvas.width_mm, canvas.height_mm], rectangle_mm=list(size_mm), origin_mm=list(origin_mm))


def test_asymmetric_output_keeps_orientation(capsys, monkeypatch, tmp_path):
    destination = tmp_path / "out.svg"
    code, _, error, _ = _convert(capsys, monkeypatch, FIXTURES / "asymmetric.png", destination,
                                 *MASK_SETTINGS, "--debug-bundle")
    assert code == 0, error
    _, _, rendered, material = _published(destination)
    error_px = np.count_nonzero(rendered ^ material)
    flips = {name: np.count_nonzero(rendered ^ array) for name, array in
             {"lr": np.fliplr(material), "ud": np.flipud(material), "transpose": material.T}.items()}
    assert all(error_px < value for value in flips.values()), (error_px, flips)
    _record(mode="mask", group="coordinates", case="asymmetric_alignment", exit=code, xor_px=error_px, flipped_xor_px=flips)


# --- Step 8: composition --------------------------------------------------------


def test_exclusion_wins_over_inclusion(capsys, monkeypatch, tmp_path):
    include = _save(tmp_path / "include.png", np.full((128, 128), 255))
    removed = np.zeros((128, 128), np.uint8)
    removed[60:100, 10:60] = 255
    exclude = _save(tmp_path / "exclude.png", removed)
    destination = tmp_path / "out.svg"
    code, _, error, _ = _convert(capsys, monkeypatch, FIXTURES / "asymmetric.png", destination, *MASK_SETTINGS,
                                 "--include-mask", include, "--exclude-mask", exclude, "--debug-bundle")
    assert code == 0, error
    _, _, rendered, material = _published(destination)
    source = np.asarray(Image.open(FIXTURES / "asymmetric.png").convert("L")) >= 128
    assert (source & (removed > 0)).any()
    assert not (material & (removed > 0)).any()
    assert not (rendered & (removed > 0)).any()
    _record(mode="mask", group="composition", case="exclude_wins", exit=code,
            excluded_source_px=int((source & (removed > 0)).sum()))


def test_edge_expansion_is_clipped_by_constraints(capsys, monkeypatch, tmp_path):
    lines = np.zeros((128, 128), np.uint8)
    lines[64, :] = 255
    lines[:, 64] = 255
    source = _save(tmp_path / "lines.png", lines)
    allowed = np.zeros((128, 128), np.uint8)
    allowed[32:96, 32:96] = 255
    include = _save(tmp_path / "include.png", allowed)
    destination = tmp_path / "out.svg"
    code, _, error, _ = _convert(capsys, monkeypatch, source, destination, "--input-kind", "edges",
                                 "--width-mm", "100", "--line-width-mm", "6", "--include-mask", include, "--debug-bundle")
    assert code == 0, error
    manifest, _, rendered, material = _published(destination)
    outside = allowed == 0
    assert manifest["preparation"]["expansion"]["radius_px"] >= 3
    assert material.any() and not (material & outside).any()
    assert not (rendered & outside).any()
    _record(mode="edges", group="composition", case="clip_after_expansion", exit=code,
            expansion=manifest["preparation"]["expansion"])


# --- Step 9: failures ------------------------------------------------------------


def _no_success(destination):
    for path in (destination, destination.with_suffix(".vector.json"), destination.with_suffix(".preview.png")):
        assert not path.exists(), path


@pytest.mark.parametrize("case", ["black", "speckle_only"])
def test_empty_standalone_material_exits_1_without_bundle(capsys, monkeypatch, tmp_path, case):
    pixels = np.zeros((64, 64), np.uint8)
    if case == "speckle_only":
        pixels[10:60:8, 10:60:8] = 255  # filter_speckle 4 removes every single-pixel island
    source = _save(tmp_path / f"{case}.png", pixels)
    destination = tmp_path / "out" / "out.svg"
    code, result, error, _ = _convert(capsys, monkeypatch, source, destination, *MASK_SETTINGS,
                                      "--preview", "--debug-bundle")
    assert code == 1
    assert result["status"] == "failed"
    _no_success(destination)
    _record(mode="mask", group="failure", case=f"empty_material_{case}", exit=code,
            message=result["diagnostics"][0]["message"], debug_retained=(destination.with_suffix(".debug") / "mask.png").is_file())


def test_malformed_upstream_svg_exits_1_without_bundle(capsys, monkeypatch, tmp_path):
    import vtracer

    def malformed(*args, **kwargs):
        return '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"><script>x</script></svg>'

    monkeypatch.setattr(vtracer, "convert_raw_image_to_svg", malformed)
    destination = tmp_path / "out" / "out.svg"
    code, result, error, _ = _convert(capsys, monkeypatch, FIXTURES / "donut.png", destination, *MASK_SETTINGS)
    assert code == 1
    assert "0.6.15" in error
    _no_success(destination)
    _record(mode="mask", group="failure", case="malformed_upstream_svg", exit=code,
            message=result["diagnostics"][0]["message"])


@pytest.mark.parametrize("flag, value, unit", [("--max-paths", "2", "path"), ("--max-svg-bytes", "200", "bytes"),
                                                ("--max-path-commands", "10", "command")])
def test_exceeded_limits_exit_1_without_bundle(capsys, monkeypatch, tmp_path, flag, value, unit):
    destination = tmp_path / "out" / "out.svg"
    code, result, error, _ = _convert(capsys, monkeypatch, FIXTURES / "separate_components.png", destination,
                                      *MASK_SETTINGS, flag, value)
    assert code == 1
    assert f"limit {value}" in error, error
    _no_success(destination)
    _record(mode="mask", group="failure", case=f"limit{flag.replace('--max', '')}", exit=code,
            message=result["diagnostics"][0]["message"])


@pytest.mark.parametrize("existing", ["out.svg", "out.vector.json", "out.preview.png", "out.debug/mask.png"])
def test_existing_bundle_member_is_refused_without_overwrite(capsys, monkeypatch, tmp_path, existing):
    collision = tmp_path / existing
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_bytes(b"old")
    destination = tmp_path / "out.svg"
    code, result, error, _ = _convert(capsys, monkeypatch, FIXTURES / "donut.png", destination, *MASK_SETTINGS,
                                      "--preview", "--debug-bundle")
    assert code == 2
    assert "--overwrite" in error
    assert collision.read_bytes() == b"old"
    assert not destination.with_suffix(".vector.json").exists() or existing == "out.vector.json"
    _record(mode="mask", group="failure", case=f"overwrite_refused_{existing}", exit=code)


def test_interrupted_overwrite_leaves_no_completion_manifest(capsys, monkeypatch, tmp_path):
    destination = tmp_path / "out.svg"
    code, *_ = _convert(capsys, monkeypatch, FIXTURES / "donut.png", destination, *MASK_SETTINGS, "--preview")
    assert code == 0
    old_svg = destination.read_bytes()
    replace = os.replace

    def interrupted(source, target, *args, **kwargs):
        if Path(target).name == "out.preview.png":
            raise OSError("injected disk failure")
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(vector_map.artifacts.os, "replace", interrupted)
    code = vector_map.main([str(FIXTURES / "asymmetric.png"), str(destination), *MASK_SETTINGS,
                            "--preview", "--overwrite", "--json"])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert code == 1
    assert result["status"] == "failed"
    assert "injected disk failure" in captured.err
    assert not destination.with_suffix(".vector.json").exists()
    # Spec 8: renames are not atomic. The new SVG landed before the failure. No manifest marks it complete.
    assert destination.read_bytes() != old_svg
    _record(mode="mask", group="failure", case="interrupted_overwrite", exit=code,
            manifest_present=False, svg_replaced_before_failure=True)
