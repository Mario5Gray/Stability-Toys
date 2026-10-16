"""S4.1 corpus probe contracts (STABL-etpzsxvn).

Plan: docs/superpowers/plans/2026-10-15-vector-map-s4-1.md.
The blocking corpus run is tests/vector_map_s4_1_corpus_probe.py, run explicitly.
These collected tests use hand-built fixtures. Native OpenSCAD tests skip when it is absent.
"""

import hashlib
import importlib.util
import json

import numpy as np
import pytest

from tests import vector_map_s4_1_corpus_probe as probe

needs_resvg = pytest.mark.skipif(
    importlib.util.find_spec("resvg_py") is None, reason="resvg-py==0.5.0 is not installed. Install the vector extra."
)
needs_vtracer = pytest.mark.skipif(
    importlib.util.find_spec("vtracer") is None, reason="vtracer==0.6.15 is not installed. Install the vector extra."
)


def mm_svg(paths, width, height, mm_per_px=0.5, viewbox=True):
    """Build a published-style SVG: millimetre root, pixel viewBox, polygon paths."""
    box = f' viewBox="0 0 {width} {height}"' if viewbox else ""
    body = "".join(f'<path d="{d}" fill="#000000" transform="translate({x},{y})" />' for d, x, y in paths)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width * mm_per_px}mm" '
            f'height="{height * mm_per_px}mm"{box}>{body}</svg>')


RECT = ("M0,0 L5,0 L5,6 L0,6 Z", 2, 3)  # Covers columns 2..6 and rows 3..8.
# An L shape: a 6 x 2 bar on top, a 2 x 6 stem on the left. No flip or rotation maps it to itself.
ELL = ("M0,0 L6,0 L6,2 L2,2 L2,6 L0,6 Z", 3, 2)


def ell_mask(width=12, height=10):
    mask = np.zeros((height, width), bool)
    mask[2:4, 3:9] = True
    mask[2:8, 3:5] = True
    return mask


@needs_resvg
def test_a_millimetre_root_renders_at_one_pixel_per_viewbox_unit():
    rendered = probe.render_published(mm_svg([RECT], 20, 12))
    expected = np.zeros((12, 20), bool)
    expected[3:9, 2:7] = True
    assert np.array_equal(rendered, expected)


@needs_resvg
@needs_vtracer
def test_published_coins_silhouette_renders_exactly_like_the_locked_raw_trace(tmp_path):
    """Coins is 256 x 202. The rejected DPI method moved 11 pixels on it."""
    import contextlib
    import io

    import vector_map_vtracer as adapter
    from PIL import Image

    case = probe.CORPUS / "coins"
    argv = [str(case / "source.png"), str(tmp_path / "out.svg"), "--input-kind", "image",
            "--mask", str(case / "mask.png"), "--width-mm", "100", "--layers", "silhouette"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert probe.vector_map.main(argv) == 0
    published = (tmp_path / "out.layers/silhouette.svg").read_text()
    assert 'width="100mm"' in published
    mask = np.asarray(Image.open(case / "mask.png").convert("L")) == 255
    assert mask.shape == (202, 256)
    locked = probe.metrics.render(adapter.trace_layer(mask).svg, 256, 202)
    assert np.array_equal(probe.render_published(published), locked)


@needs_resvg
def test_a_matching_trace_passes_every_fidelity_check():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10), ell_mask())
    assert result["failures"] == [] and result["passed"]


@needs_resvg
def test_a_flipped_trace_fails_orientation():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10), np.fliplr(ell_mask()))
    assert "orientation" in result["failures"]


@needs_resvg
def test_a_lost_hole_fails_topology():
    mask = np.zeros((12, 12), bool)
    mask[2:10, 2:10] = True
    mask[5:7, 5:7] = False
    result = probe.measure_fidelity(mm_svg([("M0,0 L8,0 L8,8 L0,8 Z", 2, 2)], 12, 12), mask)
    assert result["topology_mask"] == [1, 1]
    assert result["topology_render"] == [1, 0]
    assert "topology" in result["failures"]


@needs_resvg
def test_the_area_check_reads_the_published_polygon_area_not_the_raster():
    """A duplicated path renders the same pixels but doubles the shoelace area."""
    mask = np.zeros((12, 20), bool)
    mask[3:9, 2:7] = True
    result = probe.measure_fidelity(mm_svg([RECT, RECT], 20, 12), mask)
    assert result["xor_px"] == 0
    assert result["polygon_area_px"] == 60.0
    assert result["failures"] == ["area"]


@needs_resvg
def test_a_viewbox_that_differs_from_the_mask_fails_scale_without_raising():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10), np.zeros((10, 14), bool))
    assert "scale" in result["failures"] and not result["passed"]


def test_an_svg_without_a_viewbox_is_a_failed_fidelity_row():
    result = probe.measure_fidelity(mm_svg([ELL], 12, 10, viewbox=False), ell_mask())
    assert result["passed"] is False
    assert result["failures"] == ["render"]
    assert "viewBox" in result["error"]


# --- Evidence schema and the pinned failure signature (plan step 3) ---------------

PIN = {"alpha": [], "beta": ["centroid"]}


def row(case, mode, failures, cad=True, missing=()):
    return {"case": case, "mode": mode, "fidelity": {"failures": list(failures)},
            "cad_passed": cad, "evidence_missing": list(missing)}


def full(changes=None, pin=PIN):
    """One row per case and mode. changes maps (case, mode) to replacement rows."""
    rows = [row(case, mode, failures) for case, failures in pin.items() for mode in probe.MODES]
    return [(changes or {}).get((r["case"], r["mode"]), r) for r in rows]


CLEAN_HASHES = {"checked": 10, "mismatches": []}


def test_the_pinned_signature_comes_from_the_s1_4_sweep_at_speckle_4():
    assert probe.pinned_signature() == {
        "badge": [], "bracket": ["centroid"], "brick": ["centroid", "topology"], "coins": [],
        "gravel": ["boundary", "topology"], "horse": ["centroid", "topology"], "pcb": [], "truchet": [],
    }


def test_the_exact_pinned_signature_exits_zero_and_keeps_the_failure_visible():
    index = probe.build_index(full(), PIN, CLEAN_HASHES)
    assert index["exit_code"] == 0
    assert index["evidence_complete"] and index["fidelity_matches_pin"]
    beta = [r for r in index["rows"] if r["case"] == "beta"]
    assert [r["fidelity_passed"] for r in beta] == [False, False]
    assert index["fidelity_summary"]["beta"] == {"failures": ["centroid"], "passed": False}
    assert "release_accepted" not in index


def test_every_case_has_one_row_per_mode_even_when_rows_are_absent():
    index = probe.build_index([row("alpha", "raised", [])], PIN, CLEAN_HASHES)
    assert [(r["case"], r["mode"]) for r in index["rows"]] == [
        ("alpha", "raised"), ("alpha", "engraved"), ("beta", "raised"), ("beta", "engraved")]
    assert [r["evidence_missing"] for r in index["rows"]] == [[], ["row"], ["row"], ["row"]]
    assert not index["evidence_complete"] and index["exit_code"] == 1


def test_a_new_failure_changes_the_signature_and_exits_nonzero():
    index = probe.build_index(full({("alpha", "raised"): row("alpha", "raised", ["xor"])}), PIN, CLEAN_HASHES)
    assert not index["fidelity_matches_pin"] and index["exit_code"] == 1


def test_a_pinned_failure_that_passes_changes_the_signature_and_exits_nonzero():
    rows = full({("beta", "raised"): row("beta", "raised", []), ("beta", "engraved"): row("beta", "engraved", [])})
    index = probe.build_index(rows, PIN, CLEAN_HASHES)
    assert index["fidelity_summary"]["beta"]["passed"]
    assert not index["fidelity_matches_pin"] and index["exit_code"] == 1


def test_modes_that_disagree_on_fidelity_exit_nonzero():
    index = probe.build_index(full({("beta", "engraved"): row("beta", "engraved", [])}), PIN, CLEAN_HASHES)
    assert not index["fidelity_matches_pin"] and index["exit_code"] == 1


def test_a_cad_failure_exits_nonzero():
    index = probe.build_index(full({("alpha", "engraved"): row("alpha", "engraved", [], cad=False)}), PIN, CLEAN_HASHES)
    assert index["fidelity_matches_pin"] and index["exit_code"] == 1


def test_a_manifest_hash_mismatch_exits_nonzero():
    index = probe.build_index(full(), PIN, {"checked": 10, "mismatches": ["alpha/raised/out.svg"]})
    assert index["exit_code"] == 1


def test_a_missing_artifact_is_missing_evidence(tmp_path):
    for name in probe.REQUIRED_EVIDENCE:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    assert probe.missing_evidence(tmp_path) == []
    (tmp_path / "out.scad").unlink()
    assert probe.missing_evidence(tmp_path) == ["out.scad"]
    index = probe.build_index(full({("alpha", "raised"): row("alpha", "raised", [], missing=["out.scad"])}),
                              PIN, CLEAN_HASHES)
    assert not index["evidence_complete"] and index["exit_code"] == 1


BASE_RECIPE = {"preset": "relief-0.4", "vtracer": {"filter_speckle": 4, "mode": "polygon"}}


def test_an_override_records_the_fields_it_actually_changed():
    changed = {"preset": "relief-0.4", "vtracer": {"filter_speckle": 8, "mode": "polygon"}}
    record = probe.override_record("brick", BASE_RECIPE, ["centroid", "topology"], changed,
                                   reason="Larger speckle", override_failures=["topology"])
    assert record == {
        "case": "brick", "baseline_recipe": BASE_RECIPE, "override_recipe": changed,
        "changed_fields": ["vtracer.filter_speckle"], "reason": "Larger speckle",
        "baseline_checks": ["centroid", "topology"], "override_checks": ["topology"],
    }


def test_a_failed_case_without_an_override_records_null_and_stays_failed():
    record = probe.override_record("brick", BASE_RECIPE, ["centroid", "topology"])
    assert record["override_recipe"] is None and record["changed_fields"] == []
    assert record["baseline_checks"] == ["centroid", "topology"] and record["override_checks"] is None


def test_an_override_needs_a_reason_and_a_real_change():
    changed = {"preset": "relief-0.4", "vtracer": {"filter_speckle": 8, "mode": "polygon"}}
    with pytest.raises(ValueError, match="reason"):
        probe.override_record("brick", BASE_RECIPE, ["topology"], changed, override_failures=[])
    with pytest.raises(ValueError, match="changes no field"):
        probe.override_record("brick", BASE_RECIPE, ["topology"], dict(BASE_RECIPE), reason="x", override_failures=[])


# --- Workflow, native render and CLI (plan step 4) -------------------------------

needs_openscad = pytest.mark.skipif(probe.open_scad.OPENSCAD is None, reason="OpenSCAD is not installed.")


@pytest.fixture
def work(tmp_path):
    """A directory the OpenSCAD sandbox can read. A Flatpak directory under $HOME is removed after."""
    import shutil

    directory, created = probe.open_scad.choose_work_dir(tmp_path, probe.open_scad.OPENSCAD)
    yield directory
    if created:
        shutil.rmtree(directory)


@needs_openscad
def test_a_non_manifold_mesh_that_exits_zero_is_a_cad_error(work):
    """Two cubes that share one edge: OpenSCAD 2021.01 warns and still exits 0."""
    (work / "out.scad").write_text("cube(1);\ntranslate([1,1,0]) cube(1);\n")
    render = probe.render_scad(work, probe.open_scad.OPENSCAD)
    assert render["exit_code"] == 0
    assert any("2-manifold" in error for error in render["errors"])


def test_the_probe_blocks_when_openscad_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(probe.open_scad, "_openscad", lambda: None)
    assert probe.main(["--work-dir", str(tmp_path / "work")]) == 2
    assert "OpenSCAD is absent" in capsys.readouterr().err
    assert not (tmp_path / "work").exists()


def test_cad_gate_names_each_geometry_mismatch():
    expected = {"bounds_xy": [[0.0, 0.0], [10.0, 8.0]], "solids": 2, "genus_sum": 1, "volume_mm3": 100.0}
    measured = {"bounds": [[0.0, 0.0, 0.0], [10.0, 8.0, 1.8]], "solids": 2, "genus_sum": 1, "volume_mm3": 100.0}
    assert probe.cad_gate(measured, expected, z_top=1.8, xy_tol=0.39) == []
    wrong = dict(measured, solids=3, genus_sum=0, volume_mm3=100.02,
                 bounds=[[0.0, 0.0, 0.0], [11.0, 8.0, 1.6]])
    errors = probe.cad_gate(wrong, expected, z_top=1.8, xy_tol=0.39)
    assert [e.split()[0] for e in errors] == ["XY", "Z", "Solid", "Genus", "Volume"]


@needs_openscad
@needs_vtracer
@needs_resvg
def test_one_corpus_row_runs_the_complete_workflow(work):
    row = probe.run_case("badge", "raised", work, probe.open_scad.OPENSCAD)
    assert row["evidence_missing"] == [], row.get("errors")
    assert row["fidelity"]["failures"] == [] and row["cad_passed"], row.get("errors")
    assert row["resolved_recipe"]["vtracer"] == {"filter_speckle": 4, "mode": "polygon"}
    assert row["mask_sha256"] == probe.CORPUS_INDEX["cases"]["badge"]["sha256"]["mask"]
    assert set(row["complexity"]) == {"silhouette", "structure", "combined"}  # Badge has no detail.
    assert set(row["timings"]) >= {"conversion_s", "trace_median_s", "fidelity_median_s", "openscad_render_s"}
    assert row["override"]["override_recipe"] is None
    manifest = json.loads((work / "badge/raised/out.vector.json").read_text())
    assert [entry["path"] for entry in manifest["inputs"]] == [
        "tests/fixtures/vector_map/corpus/badge/source.png", "tests/fixtures/vector_map/corpus/badge/mask.png"]


# --- Replay, regeneration and committed bundles (plan steps 5-6) ---------------------


def bundle(directory, scad=b"scad", preview=b"preview", relief=None):
    files = {"out.svg": b"svg", "out.layers/silhouette.svg": b"sil", "out.debug/silhouette.png": b"mask",
             "out.scad": scad, "out.preview.png": preview}
    for name, data in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
    (directory / "out.vector.json").write_text(json.dumps({"relief": relief or {"backing_mm": 0.0}}))
    return directory


def test_replay_compare_accepts_identical_outputs_and_names_the_preview_exclusion(tmp_path):
    result = probe.compare_replay(bundle(tmp_path / "a"), bundle(tmp_path / "b", preview=b"other"))
    assert result["passed"]
    assert "out.preview.png" in result["excluded"]
    assert set(result["compared"]) == {"out.svg", "out.layers/silhouette.svg", "out.debug/silhouette.png", "out.scad"}


def test_replay_compare_fails_on_a_changed_scad_or_relief(tmp_path):
    original = bundle(tmp_path / "a")
    assert probe.compare_replay(original, bundle(tmp_path / "b", scad=b"new"))["differs"] == ["out.scad"]
    changed = probe.compare_replay(original, bundle(tmp_path / "c", relief={"backing_mm": 1.0}))
    assert not changed["passed"] and changed["differs"] == ["relief"]


def regen_row(svg="h1", verdict=("centroid",), cad=True, recipe=None):
    return {"case": "beta", "mode": "raised", "fidelity": {"failures": list(verdict)}, "cad_passed": cad,
            "resolved_recipe": recipe or {"preset": "relief-0.4"},
            "artifact_hashes": {"out.svg": {"sha256": svg}, "out.stl": {"sha256": "stl-a"},
                                "out.debug/silhouette.png": {"sha256": "m1"}}}


def test_regeneration_compare_ignores_the_stl_and_names_each_difference():
    same = probe.compare_runs({"rows": [regen_row()]}, {"rows": [dict(regen_row(), artifact_hashes={
        **regen_row()["artifact_hashes"], "out.stl": {"sha256": "stl-b"}})]})
    assert same["passed"], same
    differs = probe.compare_runs({"rows": [regen_row()]},
                                 {"rows": [regen_row(svg="h2", verdict=(), recipe={"preset": None})]})
    assert not differs["passed"]
    assert differs["rows"]["beta/raised"] == ["out.svg", "resolved_recipe", "fidelity"]


def test_regeneration_compare_fails_when_a_row_is_missing():
    result = probe.compare_runs({"rows": [regen_row()]}, {"rows": []})
    assert result["rows"]["beta/raised"] == ["row"] and not result["passed"]


def test_an_exported_bundle_has_no_stl_and_still_matches_its_manifest(tmp_path):
    work = tmp_path / "work" / "beta" / "raised"
    work.mkdir(parents=True)
    artifacts = []
    for name, data in (("out.svg", b"svg"), ("out.scad", b"scad")):
        (work / name).write_bytes(data)
        artifacts.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (work / "out.vector.json").write_text(json.dumps({"artifacts": artifacts}))
    for name in ("out.stl", "area-silhouette.scad", "area-silhouette.svg"):
        (work / name).write_bytes(b"x")
    exported = probe.export_bundles(tmp_path / "work", tmp_path / "bundle", {"rows": [regen_row()]})
    target = tmp_path / "bundle" / "beta" / "raised"
    assert sorted(p.name for p in target.iterdir()) == ["out.scad", "out.svg", "out.vector.json", "result.json"]
    assert probe.verify_manifest_hashes(tmp_path / "bundle" / "beta") == {"checked": 2, "mismatches": []}
    assert json.loads((target / "result.json").read_text())["case"] == "beta"
    assert exported["excluded"] == ["beta/raised/area-silhouette.scad", "beta/raised/area-silhouette.svg",
                                    "beta/raised/out.stl"]


# --- Host paths stay out of committed evidence --------------------------------------


def test_scrub_replaces_the_work_directory_repository_and_home(tmp_path):
    from pathlib import Path

    work = tmp_path / "work"
    record = {"argv": [str(probe.ROOT / "tests/fixtures/a.png"), str(work / "beta/raised/out.svg")],
              "nested": {"stderr": f"scad {work}/beta/raised/out.scad\nsaved {Path.home()}/x.svg"},
              "count": 3, "root": str(probe.ROOT)}
    assert probe.scrub(record, work) == {
        "argv": ["tests/fixtures/a.png", "<work>/beta/raised/out.svg"],
        "nested": {"stderr": "scad <work>/beta/raised/out.scad\nsaved ~/x.svg"},
        "count": 3, "root": ".",
    }


def test_export_refuses_a_committed_file_that_holds_a_host_path(tmp_path):
    work = tmp_path / "work" / "beta" / "raised"
    work.mkdir(parents=True)
    leaked = json.dumps({"artifacts": [], "inputs": [{"path": str(probe.ROOT / "tests/fixtures/a.png")}]})
    (work / "out.vector.json").write_text(leaked)
    with pytest.raises(ValueError, match="beta/raised/out.vector.json"):
        probe.export_bundles(tmp_path / "work", tmp_path / "bundle", {"rows": [regen_row()]})


def test_export_refuses_a_result_row_that_was_not_scrubbed(tmp_path):
    (tmp_path / "work" / "beta" / "raised").mkdir(parents=True)
    row = dict(regen_row(), work_dir=str(tmp_path / "work" / "beta" / "raised"))
    with pytest.raises(ValueError, match="result.json"):
        probe.export_bundles(tmp_path / "work", tmp_path / "bundle", {"rows": [row]})


@pytest.mark.parametrize("leak", ["/private/tmp/claude-501/run/s41-runA.json", "/var/folders/ab/T/x.json",
                                  "/home/someone/x.json", "/Users/someone/x.json"])
def test_the_guard_flags_an_absolute_host_path_it_was_not_told_about(tmp_path, leak):
    with pytest.raises(ValueError, match="index.json"):
        probe.host_path_check(json.dumps({"against": leak}), tmp_path / "work", "index.json")


def test_the_guard_flags_the_login_name(tmp_path):
    import getpass

    with pytest.raises(ValueError, match="report.md"):
        probe.host_path_check(f"slug -Users-{getpass.getuser()}-workspace", tmp_path / "work", "report.md")


def test_the_guard_accepts_placeholders_and_repo_relative_paths(tmp_path):
    text = "python -m tests.vector_map_s4_1_corpus_probe --work-dir <work> --compare-index <earlier-index> " \
           "tests/fixtures/vector_map/corpus/badge/mask.png"
    probe.host_path_check(text, tmp_path / "work", "index.json")


def test_scrub_names_the_earlier_index_and_the_interpreter(tmp_path):
    import sys

    earlier = tmp_path / "elsewhere" / "s41-runA.json"
    record = {"invocation": [sys.executable, "-m", "x", "--compare-index", str(earlier)], "against": str(earlier)}
    assert probe.scrub(record, tmp_path / "work", {earlier: "<earlier-index>"}) == {
        "invocation": ["python", "-m", "x", "--compare-index", "<earlier-index>"], "against": "<earlier-index>"}
