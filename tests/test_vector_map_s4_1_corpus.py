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
S4_1_REPORT = probe.ROOT / "docs/superpowers/reports/2026-10-15-vector-map-s4-1-corpus"


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


# --- PR #107 review: coverage gate, output paths, preset description -----------------


def test_output_paths_outside_the_repository_get_their_own_placeholders(tmp_path):
    import argparse

    work = tmp_path / "run"
    options = argparse.Namespace(index_json=tmp_path / "run.json", compare_index=tmp_path / "a.json",
                                 bundle_dir=probe.ROOT / "docs/bundle", report=tmp_path / "r.md")
    extra = probe.output_placeholders(options)
    invocation = ["--work-dir", str(work), "--index-json", str(tmp_path / "run.json"), "--compare-index",
                  str(tmp_path / "a.json"), "--bundle-dir", str(probe.ROOT / "docs/bundle"), "--report", str(tmp_path / "r.md")]
    scrubbed = probe.scrub(invocation, work, extra)
    assert scrubbed == ["--work-dir", "<work>", "--index-json", "<index-json>", "--compare-index", "<earlier-index>",
                        "--bundle-dir", "docs/bundle", "--report", "<report>"]
    probe.host_path_check(json.dumps(scrubbed), work, "index.json")


def minimal_index(rows):
    index = probe.build_index(rows, probe.pinned_signature(), CLEAN_HASHES)
    index.update(host={"os": "os", "machine": "m", "processor": "", "python": "3", "vtracer": "v", "resvg-py": "r",
                       "openscad": "OpenSCAD version 2021.01"}, openscad_command=["openscad"],
                 invocation=["python", "-m", "tests.vector_map_s4_1_corpus_probe"], work_dir="<work>")
    return index


def test_a_partial_run_reports_its_coverage_and_no_release_decision():
    rows = [row("badge", mode, []) for mode in probe.MODES]
    index = minimal_index(rows)
    assert len(index["rows"]) == 16 and not index["evidence_complete"] and index["exit_code"] == 1
    text = probe.report(index)
    assert "| Evidence complete | False (2/16 mode rows) |" in text
    assert "`release_accepted`: true" not in text
    assert "Not recorded. Release acceptance needs a complete run of all 16 mode rows." in text


def test_a_complete_run_reports_sixteen_rows_and_the_release_decision():
    pin = probe.pinned_signature()
    index = minimal_index([row(case, mode, names) for case, names in pin.items() for mode in probe.MODES])
    text = probe.report(index)
    assert "| Evidence complete | True (16/16 mode rows) |" in text
    assert "`release_accepted`: true for the corpus-fidelity limitation only." in text


def test_the_defaults_assessment_lists_every_relief_preset_value():
    pin = probe.pinned_signature()
    text = probe.report(minimal_index([row(case, mode, names) for case, names in pin.items() for mode in probe.MODES]))
    assert ("`relief-0.4` sets silhouette thickness 1.2 mm, structure height 0.6 mm, detail height 0.2 mm, "
            "backing 0 mm, structure width 0.8 mm and detail width 0.5 mm.") in text
    assert "heights only" not in text


@needs_openscad
@needs_vtracer
@needs_resvg
def test_a_one_case_probe_run_cannot_exit_zero(work):
    index_path = work / "index.json"
    assert probe.main(["--work-dir", str(work / "run"), "--case", "badge", "--index-json", str(index_path)]) == 1
    index = json.loads(index_path.read_text())
    assert len(index["rows"]) == 16 and not index["evidence_complete"]
    assert sum(r["evidence_missing"] == ["row"] for r in index["rows"]) == 14
    assert index["invocation"][index["invocation"].index("--index-json") + 1] == "<index-json>"


def test_placeholders_cover_the_typed_and_the_resolved_form_of_a_symlinked_path(tmp_path):
    """macOS /tmp is a symlink to /private/tmp. The invocation keeps the typed form."""
    import argparse

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    options = argparse.Namespace(work_dir=link / "run", index_json=link / "run.json", compare_index=None,
                                 bundle_dir=None, report=None)
    extra = probe.output_placeholders(options)
    record = [str(link / "run"), str(link / "run.json"), str(real / "run"), str(real / "run.json")]
    assert probe.scrub(record, (link / "run").resolve(), extra) == ["<work>", "<index-json>", "<work>", "<index-json>"]


# --- S4.2 Task 5: cross-check committed complexity against the published SVGs (STABL-zpbyzqpk) ---------


def audit_corpus_row(directory):
    """Recount one row's published SVGs with the S4.2 inspection. Return each mismatched field name.

    complexity[role] came from the independent metrics.svg_complexity counter. Layer paths and commands
    must sum to combined. Combined svg_bytes is the out.svg file size, not the sum of the layer sizes.
    """
    import vector_map_config as config
    import vector_map_svg as svg_io

    recorded = json.loads((directory / "result.json").read_text())["complexity"]
    manifest = json.loads((directory / "out.vector.json").read_text())
    canvas = config.Canvas(**manifest["canvas"])
    limits = svg_io.SvgLimits(**manifest["svg_limits"])
    layers = [entry for entry in manifest["layers"] if entry["svg"]]
    if sorted(entry["id"] for entry in layers) != sorted(role for role in recorded if role != "combined"):
        return ["roles"]
    mismatches, totals = [], {"paths": 0, "commands": 0}
    for entry in layers:
        data = (directory / entry["svg"]).read_bytes()
        found = svg_io.inspect_published_svg(data.decode("utf-8"), canvas, limits).metrics
        measured = {"paths": found.paths, "commands": found.commands, "svg_bytes": len(data)}
        mismatches += [f"{entry['id']}.{key}" for key, value in measured.items() if recorded[entry["id"]][key] != value]
        totals["paths"] += found.paths
        totals["commands"] += found.commands
    combined = {**totals, "svg_bytes": len((directory / "out.svg").read_bytes())}
    mismatches += [f"combined.{key}" for key, value in combined.items() if recorded["combined"][key] != value]
    return mismatches


def copied_row(tmp_path, case="badge", mode="raised"):
    """Copy one committed row so a test can corrupt its record without touching the report."""
    import shutil

    target = tmp_path / case / mode
    shutil.copytree(S4_1_REPORT / case / mode, target)
    return target


def corrupt(directory, edit):
    path = directory / "result.json"
    record = json.loads(path.read_text())
    edit(record["complexity"])
    path.write_text(json.dumps(record))


def test_a_layer_count_that_disagrees_with_its_published_svg_is_named(tmp_path):
    directory = copied_row(tmp_path)
    corrupt(directory, lambda c: c["silhouette"].update(commands=c["silhouette"]["commands"] + 1))
    assert audit_corpus_row(directory) == ["silhouette.commands"]


def test_combined_paths_and_commands_are_the_sum_of_the_published_layers(tmp_path):
    """badge/raised: silhouette 6 + structure 8 = 14 paths, 156 + 318 = 474 commands."""
    directory = copied_row(tmp_path)
    corrupt(directory, lambda c: c["combined"].update(paths=13, commands=475))
    assert audit_corpus_row(directory) == ["combined.paths", "combined.commands"]


def test_a_layer_byte_count_is_the_utf8_size_of_the_published_file(tmp_path):
    directory = copied_row(tmp_path)
    corrupt(directory, lambda c: c["structure"].update(svg_bytes=2808))  # badge/raised structure.svg is 2809 bytes
    assert audit_corpus_row(directory) == ["structure.svg_bytes"]


def test_combined_bytes_are_the_combined_file_not_the_layer_sum(tmp_path):
    """badge/raised: silhouette 1590 + structure 2809 = 4399 bytes. out.svg is 4291 bytes."""
    directory = copied_row(tmp_path)
    corrupt(directory, lambda c: c["combined"].update(svg_bytes=4399))
    assert audit_corpus_row(directory) == ["combined.svg_bytes"]


def test_a_recorded_role_without_a_published_layer_is_named(tmp_path):
    directory = copied_row(tmp_path)
    corrupt(directory, lambda c: c.update(detail={"paths": 0, "subpaths": 0, "commands": 0, "svg_bytes": 0}))
    assert audit_corpus_row(directory) == ["roles"]


def test_the_committed_report_has_one_row_per_case_and_mode():
    rows = sorted(str(p.parent.relative_to(S4_1_REPORT)) for p in S4_1_REPORT.glob("*/*/result.json"))
    assert rows == [f"{case}/{mode}" for case in ("badge", "bracket", "brick", "coins", "gravel", "horse", "pcb",
                                                    "truchet") for mode in ("engraved", "raised")]


@pytest.mark.parametrize("row_dir", sorted(p.parent for p in S4_1_REPORT.glob("*/*/result.json")),
                         ids=lambda p: f"{p.parent.name}-{p.name}")
def test_every_committed_row_matches_its_published_svgs(row_dir):
    """Historical evidence from PR #107. A mismatch is a finding. Do not re-baseline the row."""
    assert audit_corpus_row(row_dir) == []
