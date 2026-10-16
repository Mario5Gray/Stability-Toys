"""Blocking S4.1 corpus acceptance (STABL-etpzsxvn). Run explicitly with a real OpenSCAD executable.

Plan: docs/superpowers/plans/2026-10-15-vector-map-s4-1.md.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import io
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import vector_map  # noqa: E402
import vector_map_vtracer as adapter  # noqa: E402
from tests.fixtures.vector_map.corpus import metrics  # noqa: E402
from tests import test_vector_map_openscad as open_scad  # noqa: E402
# Harness helpers from the S3.6 probe. _run_case is tied to S3.6 scenes and is not reused.
from tests.vector_map_s3_6_cad_probe import (  # noqa: E402
    _hashes, _mask_expectation, _preview, _vertices, expected_volume, verify_manifest_hashes,
)

CORPUS = ROOT / "tests/fixtures/vector_map/corpus"
CORPUS_INDEX = json.loads((CORPUS / "corpus.json").read_text())
CASE_NAMES = tuple(sorted(CORPUS_INDEX["cases"]))
SWEEP = CORPUS / "sweep-results.json"
# The S1.4 pins are measured at the shipped default. A run at another speckle value has no pin.
PINNED_SPECKLE = metrics.CURRENT_DEFAULT
MODES = ("raised", "engraved")
# Files a mode bundle must hold before its row can count. Empty structure or detail layers are
# valid and publish no SVG, so those layer files are not required.
REQUIRED_EVIDENCE = (
    "out.svg", "out.vector.json", "out.scad", "out.stl", "out.preview.png",
    "out.layers/silhouette.svg", "out.debug/recipe.json", "out.debug/silhouette.png",
    "fidelity.png", "mesh.png",
)

ET.register_namespace("", "http://www.w3.org/2000/svg")


def pixel_root(svg: str) -> tuple[str, int, int]:
    """Return an in-memory copy whose root width and height are the viewBox pixel size.

    The published SVG keeps a millimetre root, which metrics.render() rejects. Only the root
    width and height change. The viewBox, paths, groups and transforms stay as published.
    """
    root = ET.fromstring(svg)
    box = root.get("viewBox")
    if not box:
        raise ValueError("Published SVG has no viewBox. The pixel size is unknown.")
    x, y, width, height = (float(value) for value in box.replace(",", " ").split())
    if x or y or width != int(width) or height != int(height):
        raise ValueError(f"Published SVG viewBox is not an integer pixel canvas at the origin: {box!r}")
    root.set("width", str(int(width)))
    root.set("height", str(int(height)))
    return ET.tostring(root, encoding="unicode"), int(width), int(height)


def render_published(svg: str) -> np.ndarray:
    """Rasterize a published SVG with the unchanged locked renderer, one viewBox unit per pixel."""
    copy, width, height = pixel_root(svg)
    return metrics.render(copy, width, height)


def measure_fidelity(svg: str, mask: np.ndarray) -> dict:
    """Run the locked S1.4 evaluation on a published silhouette against its reviewed mask."""
    try:
        rendered = render_published(svg)
    except Exception as exc:
        return {"passed": False, "failures": ["render"], "checks": {"render": False},
                "error": f"{type(exc).__name__}: {exc}"}
    return metrics.evaluate(mask, rendered, metrics.polygon_area(svg))


# --- Evidence schema ---------------------------------------------------------------


def pinned_signature(sweep: Path = SWEEP) -> dict[str, list[str]]:
    """Read the S1.4 failure names per case at the shipped default.

    tests/test_vector_map_corpus.py checks this file against its own pins, so the probe holds
    no third copy.
    """
    results = json.loads(sweep.read_text())["results"][str(PINNED_SPECKLE)]
    return {case: list(results[case]["failures"]) for case in sorted(results)}


def missing_evidence(directory: Path) -> list[str]:
    return [name for name in REQUIRED_EVIDENCE if not (directory / name).is_file()]


def build_index(rows: list[dict], pinned: dict[str, list[str]], manifest_check: dict) -> dict:
    """Order one row per case and mode, then derive the run verdict.

    A case with a failed fidelity check stays FAIL even when it matches the pin. The exit is 0
    only when the evidence is complete, every CAD row passes, every manifest hash matches, and
    the fidelity failure names equal the pin exactly. The probe never decides release.
    """
    found = {(r["case"], r["mode"]): r for r in rows}
    ordered = []
    for case in pinned:
        for mode in MODES:
            r = found.get((case, mode)) or {"case": case, "mode": mode, "fidelity": {"failures": []},
                                             "cad_passed": False, "evidence_missing": ["row"]}
            r = dict(r, fidelity_passed=not r["fidelity"]["failures"])
            ordered.append(r)
    summary, signature, modes_agree = {}, {}, True
    for case in pinned:
        per_mode = [sorted(r["fidelity"]["failures"]) for r in ordered
                    if r["case"] == case and r["evidence_missing"] != ["row"]]
        failures = per_mode[0] if per_mode else ["row"]
        modes_agree &= all(other == failures for other in per_mode)
        signature[case] = failures
        summary[case] = {"failures": failures, "passed": not failures}
    evidence_complete = all(not r["evidence_missing"] for r in ordered)
    matches = modes_agree and signature == {case: sorted(names) for case, names in pinned.items()}
    cad = all(r["cad_passed"] for r in ordered)
    hashes_ok = not manifest_check["mismatches"]
    return {
        "rows": ordered, "fidelity_summary": summary, "fidelity_signature": signature,
        "pinned_signature": pinned, "fidelity_modes_agree": modes_agree,
        "fidelity_matches_pin": matches, "evidence_complete": evidence_complete,
        "cad_passed": cad, "manifest_hash_check": manifest_check,
        "exit_code": 0 if evidence_complete and matches and cad and hashes_ok else 1,
    }


def _changed_fields(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        names = []
        for key in sorted(set(before) | set(after)):
            names += _changed_fields(before.get(key), after.get(key), f"{prefix}{key}.")
        return names
    return [] if before == after else [prefix.rstrip(".")]


def override_record(case: str, baseline_recipe: dict, baseline_failures: list[str],
                    override_recipe: dict | None = None, reason: str | None = None,
                    override_failures: list[str] | None = None) -> dict:
    """Record a per-case recipe override. With no override the case keeps its baseline result."""
    changed = []
    if override_recipe is not None:
        changed = _changed_fields(baseline_recipe, override_recipe)
        if not changed:
            raise ValueError(f"{case}: the override recipe changes no field.")
        if not reason:
            raise ValueError(f"{case}: an override needs a reason.")
    return {
        "case": case, "baseline_recipe": baseline_recipe, "override_recipe": override_recipe,
        "changed_fields": changed, "reason": reason, "baseline_checks": list(baseline_failures),
        "override_checks": None if override_recipe is None else list(override_failures or []),
    }


# --- Workflow and native render ------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median_seconds(call) -> float:
    """One untimed warm-up, then the median of the timed calls (metrics.TIMING)."""
    call()
    samples = []
    for _ in range(metrics.TIMING["timed_calls"]):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def render_scad(directory: Path, command: list[str]) -> dict:
    """Render out.scad to out.stl. OpenSCAD can exit 0 with a non-manifold mesh, so read stderr."""
    argv = [*command, "-o", str(directory / "out.stl"), str(directory / "out.scad")]
    record: dict[str, Any] = {"argv": argv}
    if issue := open_scad.sandbox_error(command, [directory / "out.scad", directory / "out.stl"]):
        return dict(record, exit_code=None, errors=[issue])
    started = time.perf_counter()
    run = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    errors = [line for line in run.stderr.splitlines()
              if line.startswith("ERROR:") or "2-manifold" in line]
    if run.returncode:
        errors.insert(0, f"OpenSCAD exited {run.returncode}.")
    return dict(record, exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr,
                seconds=time.perf_counter() - started, errors=errors)


def cad_gate(measured: dict, expected: dict, z_top: float, xy_tol: float) -> list[str]:
    """The S3.6 CAD gates: XY and Z bounds, solid count, genus, and volume at 1e-4 relative."""
    errors = []
    low, high = measured["bounds"]
    if not (np.allclose(low[:2], expected["bounds_xy"][0], atol=xy_tol, rtol=0)
            and np.allclose(high[:2], expected["bounds_xy"][1], atol=xy_tol, rtol=0)):
        errors.append(f"XY bounds differ beyond {xy_tol:g} mm: {[low[:2], high[:2]]} versus {expected['bounds_xy']}.")
    if abs(low[2]) > metrics.TOLERANCES["mesh_z_abs_mm"] or abs(high[2] - z_top) > metrics.TOLERANCES["mesh_z_abs_mm"]:
        errors.append(f"Z bounds differ: [{low[2]}, {high[2]}] versus [0, {z_top}].")
    if measured["solids"] != expected["solids"]:
        errors.append(f"Solid count {measured['solids']} versus {expected['solids']}.")
    if measured["genus_sum"] != expected["genus_sum"]:
        errors.append(f"Genus sum {measured['genus_sum']} versus {expected['genus_sum']}.")
    if not metrics.mesh_volume_ok(measured["volume_mm3"], expected["volume_mm3"]):
        errors.append(f"Volume {measured['volume_mm3']} versus {expected['volume_mm3']} mm³ exceeds 1e-4 relative.")
    return errors


def _fidelity_preview(mask: np.ndarray, rendered: np.ndarray, path: Path) -> None:
    """Mask, rendered published silhouette, and their XOR, side by side."""
    height, width = mask.shape
    canvas = np.full((height, width * 3 + 8, 3), 255, np.uint8)
    canvas[:, :width][mask] = (30, 36, 54)
    if rendered.shape == mask.shape:
        canvas[:, width + 4:2 * width + 4][rendered] = (30, 36, 54)
        xor = canvas[:, 2 * width + 8:]
        xor[mask & rendered] = (200, 204, 214)
        xor[mask & ~rendered] = (214, 39, 40)
        xor[~mask & rendered] = (31, 119, 180)
    Image.fromarray(canvas, "RGB").save(path)


def _argv(case: str, mode: str, directory: Path) -> list[str]:
    """Inputs are repo-relative, so the manifest records no host path. _convert runs from ROOT."""
    source = CORPUS.relative_to(ROOT) / case
    return [str(source / "source.png"), str(directory / "out.svg"), "--input-kind", "image",
            "--mask", str(source / "mask.png"), "--width-mm", "100",
            "--layers", "silhouette,structure,detail", "--preset", "relief-0.4",
            *(["--detail-mode", "engraved"] if mode == "engraved" else []),
            "--export-scad", "--debug-bundle", "--preview", "--json"]


def _convert(argv: list[str]) -> tuple[int, str, str, float]:
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with contextlib.chdir(ROOT), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = vector_map.main(argv)
    return code, stdout.getvalue(), stderr.getvalue(), time.perf_counter() - started


def run_case(case: str, mode: str, root: Path, command: list[str]) -> dict:
    """Run one corpus case in one detail mode into root/case/mode and measure every gate."""
    directory = root / case / mode
    directory.mkdir(parents=True)
    source, mask_path = CORPUS / case / "source.png", CORPUS / case / "mask.png"
    pinned_hashes = CORPUS_INDEX["cases"][case]["sha256"]
    row: dict[str, Any] = {"case": case, "mode": mode, "work_dir": str(directory),
                           "source_sha256": _sha256(source), "mask_sha256": _sha256(mask_path), "errors": []}
    for kind, digest in (("source", row["source_sha256"]), ("mask", row["mask_sha256"])):
        if digest != pinned_hashes[kind]:
            row["errors"].append(f"Reviewed {kind} hash differs from corpus.json.")
    argv = _argv(case, mode, directory)
    code, stdout, stderr, seconds = _convert(argv)
    row.update(argv=["python", "scripts/vector_map.py", *argv], exit_code=code, stderr=stderr,
               timings={"conversion_s": seconds}, fidelity={"failures": ["conversion"]}, cad_passed=False)
    if code != 0:
        row["errors"].append(f"Conversion exited {code}.")
        row["evidence_missing"] = missing_evidence(directory)
        return row
    row["result"] = json.loads(stdout)
    manifest = json.loads((directory / "out.vector.json").read_text())
    recipe = json.loads((directory / "out.debug/recipe.json").read_text())
    row.update(resolved_recipe=recipe, manifest_sha256=_sha256(directory / "out.vector.json"),
               relief=manifest["relief"], canvas=manifest["canvas"], vtracer=manifest["vtracer"])
    if manifest["vtracer"]["filter_speckle"] != PINNED_SPECKLE:
        row["errors"].append(f"filter_speckle {manifest['vtracer']['filter_speckle']} has no S1.4 pin.")

    reviewed = np.asarray(Image.open(mask_path).convert("L")) == 255
    prepared = np.asarray(Image.open(directory / "out.debug/silhouette.png").convert("L")) == 255
    row["prepared_mask_matches_review"] = bool(np.array_equal(reviewed, prepared))
    if not row["prepared_mask_matches_review"]:
        row["errors"].append("The prepared silhouette differs from the reviewed mask.")
    silhouette = (directory / "out.layers/silhouette.svg").read_text()
    row["fidelity"] = measure_fidelity(silhouette, reviewed)
    timings = row["timings"]
    timings["trace_median_s"] = _median_seconds(lambda: adapter.trace_layer(prepared))
    timings["fidelity_median_s"] = _median_seconds(lambda: measure_fidelity(silhouette, reviewed))
    row["complexity"] = {entry["id"]: metrics.svg_complexity((directory / entry["svg"]).read_text())
                         for entry in manifest["layers"] if entry["svg"]}
    row["complexity"]["combined"] = metrics.svg_complexity((directory / "out.svg").read_text())

    started = time.perf_counter()
    rendered = render_published(silhouette) if "render" not in row["fidelity"]["failures"] else np.zeros_like(reviewed)
    _fidelity_preview(reviewed, rendered, directory / "fidelity.png")
    timings["fidelity_preview_s"] = time.perf_counter() - started

    render = render_scad(directory, command)
    timings["openscad_render_s"] = render.get("seconds")
    cad: dict[str, Any] = {"render": render, "errors": list(render["errors"])}
    row["cad"] = cad
    if render["exit_code"] == 0:
        vertices, triangles = _vertices(directory / "out.stl")
        parts = open_scad.mesh_components(vertices, triangles)
        measured = {"bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                    "solids": len(parts), "genus_per_solid": [int(part[2]) for part in parts],
                    "genus_sum": int(sum(part[2] for part in parts)),
                    "z_planes": sorted({round(float(z), 6) for z in vertices[:, 2]}),
                    "volume_mm3": metrics.mesh_volume(vertices[triangles])}
        relief = manifest["relief"]
        started = time.perf_counter()
        volume, areas = expected_volume(manifest, directory, command)
        timings["effective_area_export_s"] = time.perf_counter() - started
        published_volume, published_areas = expected_volume(manifest, directory)
        expected = _mask_expectation(directory / "out.debug/silhouette.png", manifest["canvas"],
                                     relief["backing_mm"], manifest["vtracer"]["filter_speckle"])
        z_top = relief["backing_mm"] + (relief["silhouette_thickness_mm"] or 0) + max(
            relief["structure_height_mm"] or 0,
            0 if relief["detail_mode"] == "engraved" else relief["detail_height_mm"] or 0)
        expected.update(volume_mm3=volume, areas_mm2=areas, z_max=z_top, volume_basis="effective_openscad_2d",
                        published_volume_mm3=published_volume, published_areas_mm2=published_areas,
                        published_volume_excess=(published_volume - volume) / volume if volume else 0.0)
        cad.update(measured=measured, expected=expected)
        cad["errors"] += cad_gate(measured, expected, z_top, manifest["canvas"]["mm_per_px"])
        started = time.perf_counter()
        _preview(vertices, triangles, directory / "mesh.png")
        timings["mesh_preview_s"] = time.perf_counter() - started
    row["cad_passed"] = render["exit_code"] == 0 and not cad["errors"]
    row["override"] = override_record(case, recipe, row["fidelity"]["failures"])
    row["artifact_hashes"] = _hashes(directory)
    row["evidence_missing"] = missing_evidence(directory)
    if row["errors"] and not row["evidence_missing"]:
        row["evidence_missing"] = ["provenance"]
    return row


# --- Host paths ---------------------------------------------------------------------------
# Committed evidence must not hold local paths. Records name the work directory as <work>,
# repository files by repo-relative path, and the home directory as ~.


# Absolute host locations that never belong in committed evidence, wherever they come from.
HOST_PATH = re.compile(r"(?<![\w<>.~-])/(?:Users|home|private|var/folders|tmp)/")


def _host_forms(work_root: Path, extra: dict | None = None) -> list[tuple[str, str]]:
    forms = []
    for path, placeholder in {**(extra or {}), work_root: "<work>"}.items():
        forms += [(form, placeholder) for form in {str(path), os.path.realpath(path)}]
    forms.sort(key=lambda item: len(item[0]), reverse=True)
    return [*forms, (sys.executable, "python"), (f"{ROOT}/", ""), (str(ROOT), "."), (str(Path.home()), "~")]


def scrub(value: Any, work_root: Path, extra: dict | None = None) -> Any:
    """Return a copy of a JSON value with host paths replaced.

    extra maps other host paths, such as the earlier index, to placeholders.
    """
    if isinstance(value, dict):
        return {key: scrub(item, work_root, extra) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item, work_root, extra) for item in value]
    if isinstance(value, str):
        for form, replacement in _host_forms(work_root, extra):
            value = value.replace(form, replacement)
    return value


def host_path_check(text: str, work_root: Path, name: str) -> None:
    """Raise when committed text holds a known host path, any absolute host location, or the login name."""
    found = [form.rstrip("/") for form, _ in _host_forms(work_root) if form.rstrip("/") in text]
    found += HOST_PATH.findall(text)
    user = getpass.getuser()
    if len(user) >= 3 and user in text:
        found.append(f"login name {user!r}")
    if found:
        raise ValueError(f"Host path in committed file {name}: {found[0]}")


# --- Replay, regeneration and committed bundles ---------------------------------------

REPLAY_EXCLUDED = {"out.preview.png": "The preview overlays the source. A recipe replay reads the prepared mask as its source."}
# OpenSCAD outputs. The STL is not committed and need not be byte-identical across hosts.
# mesh.png is drawn from the STL.
NOT_COMMITTED = ("out.stl", "area-*")
NOT_REPRODUCED = ("out.stl", "area-*", "mesh.png")
TEXT_SUFFIXES = {".json", ".svg", ".scad", ".md", ".txt"}


def _replay_files(directory: Path) -> list[str]:
    names = ["out.svg", *sorted(f"out.layers/{p.name}" for p in (directory / "out.layers").glob("*.svg")),
             *sorted(f"out.debug/{p.name}" for p in (directory / "out.debug").glob("*.png")), "out.scad"]
    return [name for name in names if (directory / name).is_file()]


def compare_replay(original: Path, replay: Path) -> dict:
    """Compare a debug-recipe replay with its original bundle.

    The recipe reads prepared masks (source "map", prepared true), so this proves replay from
    prepared masks only. A fresh run from the source PNG proves preparation repeatability.
    """
    compared = _replay_files(original)
    differs = [name for name in compared
               if not (replay / name).is_file() or _sha256(original / name) != _sha256(replay / name)]
    relief = [json.loads((d / "out.vector.json").read_text())["relief"] for d in (original, replay)]
    if relief[0] != relief[1]:
        differs.append("relief")
    return {"compared": compared, "excluded": REPLAY_EXCLUDED, "differs": differs, "passed": not differs}


def replay_case(case: str, mode: str, root: Path) -> dict:
    original = root / case / mode
    target = root / "_replay" / case / mode
    target.mkdir(parents=True)
    argv = ["--recipe", str(original / "out.debug/recipe.json"), str(target / "out.svg"), "--debug-bundle", "--json"]
    code, _, stderr, seconds = _convert(argv)
    result = {"case": case, "mode": mode, "argv": ["python", "scripts/vector_map.py", *argv],
              "exit_code": code, "seconds": seconds}
    if code != 0:
        return dict(result, stderr=stderr, differs=["conversion"], passed=False)
    return dict(result, **compare_replay(original, target))


def _matches(name: str, patterns) -> bool:
    return any(Path(name).match(pattern) for pattern in patterns)


def compare_runs(first: dict, second: dict) -> dict:
    """Compare two independent runs from the source PNGs: artifact hashes, settings and verdicts."""
    rows_a = {f"{r['case']}/{r['mode']}": r for r in first["rows"]}
    rows_b = {f"{r['case']}/{r['mode']}": r for r in second["rows"]}
    report = {}
    for key in sorted(set(rows_a) | set(rows_b)):
        a, b = rows_a.get(key), rows_b.get(key)
        if a is None or b is None:
            report[key] = ["row"]
            continue
        hashes_a, hashes_b = a.get("artifact_hashes", {}), b.get("artifact_hashes", {})
        names = sorted(name for name in set(hashes_a) | set(hashes_b) if not _matches(name, NOT_REPRODUCED))
        differs = [name for name in names
                   if (hashes_a.get(name) or {}).get("sha256") != (hashes_b.get(name) or {}).get("sha256")]
        if a.get("resolved_recipe") != b.get("resolved_recipe"):
            differs.append("resolved_recipe")
        if sorted(a["fidelity"]["failures"]) != sorted(b["fidelity"]["failures"]):
            differs.append("fidelity")
        if a["cad_passed"] != b["cad_passed"]:
            differs.append("cad_passed")
        report[key] = differs
    return {"rows": report, "excluded": list(NOT_REPRODUCED), "passed": not any(report.values())}


def export_bundles(root: Path, destination: Path, index: dict) -> dict:
    """Copy each intact mode bundle without OpenSCAD outputs, plus its result JSON."""
    excluded = []
    for row in index["rows"]:
        source = root / row["case"] / row["mode"]
        target = destination / row["case"] / row["mode"]
        if not source.is_dir():
            continue
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            name = path.relative_to(source).as_posix()
            if _matches(name, NOT_COMMITTED):
                excluded.append(f"{row['case']}/{row['mode']}/{name}")
                continue
            if path.suffix in TEXT_SUFFIXES:
                host_path_check(path.read_text(), root, f"{row['case']}/{row['mode']}/{name}")
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target / name)
        text = json.dumps(row, indent=2, sort_keys=True) + "\n"
        host_path_check(text, root, f"{row['case']}/{row['mode']}/result.json")
        target.mkdir(parents=True, exist_ok=True)
        (target / "result.json").write_text(text)
    return {"destination": str(destination), "excluded": sorted(excluded)}


# --- Report -----------------------------------------------------------------------------

LIMIT_RULING = ("Mario accepted bracket, brick, gravel and horse as documented minimum-feature limitations "
                "on 2026-10-15 (`STABL-etpzsxvn`).")


def _cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "none"
    return str(value)


def _table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(_cell(v) for v in r) + " |" for r in rows), ""]


def report(index: dict, links: Path | None = None) -> str:
    """Write the S4.1 report from the index. Links point into the committed bundle directory."""
    rows = {(r["case"], r["mode"]): r for r in index["rows"]}
    cases = list(index["pinned_signature"])
    sweep = json.loads(SWEEP.read_text())["results"][str(PINNED_SPECKLE)]
    scale = {r["canvas"]["mm_per_px"] for r in index["rows"] if "canvas" in r}

    def link(case, mode, name):
        return f"{links.as_posix()}/{case}/{mode}/{name}" if links else f"{case}/{mode}/{name}"

    def cad(case, mode):
        r = rows[(case, mode)]
        return "PASS" if r["cad_passed"] else "FAIL"

    host = index["host"]
    lines = ["# S4.1 reviewed corpus bundles", "",
             "Issue: `STABL-etpzsxvn`. Plan: [S4.1 plan](../plans/2026-10-15-vector-map-s4-1.md). "
             "Spec: [design section 10](../specs/2026-09-06-vector-map-relief-design.md).",
             "This report is generated by `tests/vector_map_s4_1_corpus_probe.py`. Do not edit it by hand.", "",
             "## Run", ""]
    lines += _table(["Item", "Value"], [
        ["OS", host["os"]], ["Machine", f"{host['machine']} {host['processor']}".strip()],
        ["Python", host["python"]], ["VTracer", host["vtracer"]], ["resvg-py", host["resvg-py"]],
        ["OpenSCAD", host["openscad"]], ["OpenSCAD command", " ".join(index["openscad_command"])],
        ["Work directory", index["work_dir"]]])
    lines += [f"Invocation: `{' '.join(index['invocation'])}`.", "",
              "Regenerate the evidence with the same invocation and an empty work directory.",
              "Each case JSON holds the exact conversion argv and OpenSCAD argv.",
              "The STL files are not committed. Rebuild one from its committed mode bundle:",
              "`cd <case>/<mode> && openscad -o out.stl out.scad`. The SCAD imports `out.layers/*.svg` relative to itself.", "",
              "## Verdict", ""]
    replay = index.get("replay", [])
    regeneration = index.get("regeneration")
    lines += _table(["Check", "Result"], [
        ["Evidence complete (16 mode rows)", index["evidence_complete"]],
        ["Fidelity failure names equal the S1.4 pin", index["fidelity_matches_pin"]],
        ["Raised and engraved fidelity agree", index["fidelity_modes_agree"]],
        ["CAD rows passed", f"{sum(r['cad_passed'] for r in index['rows'])}/{len(index['rows'])}"],
        ["Manifest artifact hashes", f"{index['manifest_hash_check']['checked']} checked, "
                                     f"{len(index['manifest_hash_check']['mismatches'])} mismatches"],
        ["Recipe replay", ", ".join(f"{r['case']} {r['mode']} {'PASS' if r['passed'] else 'FAIL'}" for r in replay) or "not run"],
        ["Independent regeneration", ("PASS" if regeneration["passed"] else "FAIL") if regeneration else "not run"],
        ["Probe exit", index["exit_code"]]])
    passing = [c for c in cases if index["fidelity_summary"][c]["passed"]]
    lines += [f"Fidelity: {len(passing)}/{len(cases)} cases pass the locked S1.4 checks. "
              "Probe exit 0 means the expected evidence was collected. It does not mean that every case passes.", "",
              "## Release decision", "",
              "`release_accepted`: true for the corpus-fidelity limitation only.",
              LIMIT_RULING,
              "The ruling takes effect when this report and `scripts/USAGE.md` disclose the four cases and the minimum-feature risk.",
              "Review confirms that disclosure. The probe does not compute this value.",
              "The four cases stay FAIL. The other Sprint 4 gates still apply.", "",
              "## Case results", ""]
    lines += _table(["Case", "Class", "Fidelity", "Failed checks", "S1.4 pin", "CAD raised", "CAD engraved"], [
        [c, CORPUS_INDEX["cases"][c]["class"], "PASS" if index["fidelity_summary"][c]["passed"] else "FAIL",
         index["fidelity_summary"][c]["failures"], index["pinned_signature"][c] or ["none"],
         cad(c, "raised"), cad(c, "engraved")] for c in cases])
    millimetres = ", ".join(f"{1 * mm:g}–{2 * mm:g} mm at {mm:g} mm/px" for mm in sorted(scale))
    lines += ["## Known minimum-feature limitations", "",
              "These four cases fail the locked S1.4 fidelity checks. They are documented limitations, not passes.",
              "Their reviewed masks hold foreground strokes, isthmuses, gaps or channels 1–2 processing pixels wide.",
              f"In these bundles 1–2 processing pixels are {millimetres}.",
              "VTracer can lose or change features near this size. It changes topology in brick, gravel and horse.",
              "It moves the centroid in bracket, brick and horse. It moves the gravel boundary beyond 1.5 px.",
              "No `filter_speckle` value in the S1.4 grid fixes these cases (`tests/test_vector_map_corpus.py`).",
              "The CLI warns about possible features below four pixels. Later research: `STABL-wuzmvgxm`.", ""]
    limited = [c for c in cases if not index["fidelity_summary"][c]["passed"]]
    lines += _table(["Case", "Failed checks", "Topology mask", "Topology render", "Centroid shift px", "Boundary max px"], [
        [c, rows[(c, "raised")]["fidelity"]["failures"], rows[(c, "raised")]["fidelity"].get("topology_mask"),
         rows[(c, "raised")]["fidelity"].get("topology_render"), rows[(c, "raised")]["fidelity"].get("centroid_shift_px"),
         rows[(c, "raised")]["fidelity"].get("boundary_max_distance_px")] for c in limited])
    vtracer = sorted({json.dumps(r.get("vtracer"), sort_keys=True) for r in index["rows"] if "vtracer" in r})
    published_matches = [c for c in cases if "fidelity" in rows[(c, "raised")]
                         and rows[(c, "raised")]["fidelity"].get("xor_per_perimeter_px") == sweep[c]["xor_per_perimeter_px"]
                         and rows[(c, "raised")].get("complexity", {}).get("silhouette", {}).get("commands") == sweep[c]["commands"]]
    overrides = [r["override"] for r in index["rows"] if r.get("override", {}).get("override_recipe") is not None]
    lines += ["## Defaults assessment", "",
              "The shipped fitting default is VTracer polygon mode with `filter_speckle` 4, selected by the S1.4 sweep.",
              "`relief-0.4` is an opt-in fabrication preset. It sets relief heights only.",
              f"Resolved VTracer settings in every bundle: {', '.join(f'`{v}`' for v in vtracer)}.",
              f"The published silhouette reproduces the S1.4 raw-trace XOR and command count in {len(published_matches)}/{len(cases)} cases.", ""]
    lines += _table(["Case", "S1.4 XOR/perimeter px", "Published XOR/perimeter px", "S1.4 commands", "Published commands"], [
        [c, sweep[c]["xor_per_perimeter_px"], rows[(c, "raised")].get("fidelity", {}).get("xor_per_perimeter_px"),
         sweep[c]["commands"], rows[(c, "raised")].get("complexity", {}).get("silhouette", {}).get("commands")] for c in cases])
    lines += [f"Recipe overrides: {len(overrides)}. Every row records `override_recipe: null`.",
              "No override can repair the four limitations under the locked tolerances, because no tested speckle value passes them.",
              "The defaults stay unchanged. No tolerance is relaxed.", "",
              "## Complexity and timings", "",
              "Complexity counts are paths, subpaths, commands and UTF-8 bytes. Timings are diagnostic and host-specific.",
              "Trace and fidelity times are the median of five calls after one warm-up. Conversion and render are one call.", ""]
    table = []
    for c in cases:
        for mode in MODES:
            r = rows[(c, mode)]
            cx, t = r.get("complexity", {}), r.get("timings", {})
            table.append([c, mode, *(f"{cx[k]['commands']} / {cx[k]['svg_bytes']}" if k in cx else "no layer"
                                     for k in ("silhouette", "structure", "detail", "combined")),
                          t.get("conversion_s"), t.get("trace_median_s"), t.get("fidelity_median_s"), t.get("openscad_render_s")])
    lines += _table(["Case", "Mode", "Silhouette cmd / B", "Structure cmd / B", "Detail cmd / B", "Combined cmd / B",
                     "Convert s", "Trace s", "Fidelity s", "OpenSCAD s"], table)
    lines += ["## Replay and regeneration", "",
              "Recipe replay reads the prepared masks (`source: map`, `prepared: true`).",
              "It proves replay from prepared masks only. It does not prove source-image preparation.", ""]
    lines += _table(["Case", "Mode", "Compared", "Excluded", "Differs", "Result"], [
        [r["case"], r["mode"], r.get("compared", []), list(r.get("excluded", {})), r["differs"] or ["none"],
         "PASS" if r["passed"] else "FAIL"] for r in replay])
    if regeneration:
        changed = {k: v for k, v in regeneration["rows"].items() if v}
        lines += ["An independent run from the source PNGs and reviewed masks proves preparation repeatability.",
                  f"Compared against `{regeneration['against']}`. Rows compared: {len(regeneration['rows'])}. "
                  f"Rows that differ: {len(changed)}.",
                  f"Excluded from the hash comparison: {', '.join(f'`{p}`' for p in regeneration['excluded'])}.", ""]
        for key, value in changed.items():
            lines.append(f"- {key}: {', '.join(value)}")
        lines.append("")
    lines += ["## Per-case evidence", "",
              "`out.preview.png` is the CLI preview. `fidelity.png` shows the reviewed mask, the rendered published "
              "silhouette, and their XOR: gray is in both, red is in the mask only, blue is in the trace only.",
              "`mesh.png` is a top view of the OpenSCAD mesh. Color shows the top surface height.", ""]
    for c in cases:
        meta = CORPUS_INDEX["cases"][c]
        raised = rows[(c, "raised")]
        lines += [f"### {c}", "", f"Class: {meta['class']}. Properties: {', '.join(meta['properties'])}.",
                  f"Reviewed source: `tests/fixtures/vector_map/corpus/{c}/source.png`, sha256 `{meta['sha256']['source']}`.",
                  f"Reviewed mask: `tests/fixtures/vector_map/corpus/{c}/mask.png`, sha256 `{meta['sha256']['mask']}`.",
                  f"Fidelity: {'PASS' if index['fidelity_summary'][c]['passed'] else 'FAIL'} "
                  f"({_cell(index['fidelity_summary'][c]['failures'])} failed).", ""]
        fidelity = raised.get("fidelity", {})
        lines += _table(["Metric", "Value", "Tolerance"], [
            ["Topology mask / render", f"{fidelity.get('topology_mask')} / {fidelity.get('topology_render')}", "exact"],
            ["Bounds shift px", fidelity.get("bounds_shift_px"), metrics.TOLERANCES["bounds_shift_px"]],
            ["Box size delta px", fidelity.get("bbox_size_delta_px"), metrics.TOLERANCES["bbox_size_delta_px"]],
            ["Centroid shift px", fidelity.get("centroid_shift_px"), metrics.TOLERANCES["centroid_shift_px"]],
            ["XOR / perimeter px", fidelity.get("xor_per_perimeter_px"), metrics.TOLERANCES["xor_per_perimeter_px"]],
            ["Boundary max px", fidelity.get("boundary_max_distance_px"), metrics.TOLERANCES["boundary_max_distance_px"]],
            ["Area error / perimeter px", fidelity.get("area_dev_abs_per_perimeter_px"),
             metrics.TOLERANCES["area_dev_abs_per_perimeter_px"]]])
        for mode in MODES:
            r = rows[(c, mode)]
            lines += [f"**{mode}** — CAD {cad(c, mode)}. Bundle: [{c}/{mode}]({link(c, mode, 'result.json')}).", ""]
            if r.get("argv"):
                lines.append(f"Conversion: `{' '.join(r['argv'])}` (exit {r['exit_code']}).")
            render = r.get("cad", {}).get("render", {})
            if render:
                lines.append(f"OpenSCAD: `{' '.join(render['argv'])}` (exit {render.get('exit_code')}).")
            measured, expected = r.get("cad", {}).get("measured"), r.get("cad", {}).get("expected")
            if measured and expected:
                lines.append(f"Mesh: {measured['solids']} solids (expected {expected['solids']}), genus "
                             f"{measured['genus_sum']} (expected {expected['genus_sum']}), volume "
                             f"{measured['volume_mm3']:.4f} mm³ (expected {expected['volume_mm3']:.4f}), "
                             f"Z planes {measured['z_planes']}.")
            for error in r.get("errors", []) + r.get("cad", {}).get("errors", []):
                lines.append(f"Finding: {error}")
            images = ["out.preview.png", "fidelity.png", "mesh.png"] if mode == "raised" else ["mesh.png"]
            lines += ["", *(f"![{c} {mode} {name}]({link(c, mode, name)})" for name in images), ""]
    lines += ["## Not covered", "",
              "Slicer observations and print review belong to S4.5. Print validation is pending.",
              "The blocking CAD volume uses OpenSCAD effective 2D areas. It checks the 3D assembly, heights and closure.",
              "It does not independently check the 2D clip. Each case JSON keeps the published-area figure beside it.",
              "Baseline test failures `STABL-fjmjolci` and `STABL-bxqlmcmf` are not corpus results.", ""]
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------


def _host(openscad_version: str) -> dict:
    return {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "python": platform.python_version(), "vtracer": package_version("vtracer"),
            "resvg-py": package_version("resvg-py"), "openscad": openscad_version}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", choices=CASE_NAMES, action="append")
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--index-json", type=Path)
    parser.add_argument("--replay", choices=CASE_NAMES, action="append",
                        help="Replay this case from its debug recipe in each run mode.")
    parser.add_argument("--compare-index", type=Path, help="An index JSON from an independent earlier run.")
    parser.add_argument("--bundle-dir", type=Path, help="Write committed bundles here: no STL, no area exports.")
    parser.add_argument("--report", type=Path)
    options = parser.parse_args(argv)
    command = open_scad._openscad()
    if not command:
        print("OpenSCAD is absent. S4.1 corpus acceptance is blocked.", file=sys.stderr)
        return 2
    version_run = subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=30)
    if version_run.returncode:
        print(f"OpenSCAD version command failed: {version_run.stderr}", file=sys.stderr)
        return 2
    openscad_version = (version_run.stdout or version_run.stderr).strip()
    root = options.work_dir.resolve()
    if root.exists() and any(root.iterdir()):
        print(f"Work directory is not empty: {root}", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    cases, modes = options.case or list(CASE_NAMES), options.mode or list(MODES)
    rows = []
    for case in cases:
        for mode in modes:
            try:
                row = run_case(case, mode, root, command)
            except Exception as exc:
                row = {"case": case, "mode": mode, "fidelity": {"failures": ["exception"]}, "cad_passed": False,
                       "evidence_missing": ["exception"], "errors": [f"{type(exc).__name__}: {exc}"]}
            rows.append(row)
            print(f"{case} {mode}: fidelity {row['fidelity']['failures'] or 'PASS'}, "
                  f"CAD {'PASS' if row['cad_passed'] else 'FAIL'}", file=sys.stderr)
    checked, mismatches = 0, []
    for case in cases:
        check = verify_manifest_hashes(root / case)
        checked += check["checked"]
        mismatches += check["mismatches"]
    pinned = {case: names for case, names in pinned_signature().items() if case in cases}
    index = build_index(rows, pinned, {"checked": checked, "mismatches": mismatches})
    index["replay"] = [replay_case(case, mode, root) for case in options.replay or [] for mode in modes]
    if options.compare_index:
        index["regeneration"] = dict(compare_runs(json.loads(options.compare_index.read_text()), index),
                                     against=str(options.compare_index))
    if not all(r["passed"] for r in index["replay"]) or not index.get("regeneration", {"passed": True})["passed"]:
        index["exit_code"] = 1
    index.update(schema_version=1, host=_host(openscad_version), openscad_command=command,
                 invocation=[sys.executable, "-m", "tests.vector_map_s4_1_corpus_probe",
                             *(argv if argv is not None else sys.argv[1:])], work_dir=str(root))
    extra = {options.compare_index.resolve(): "<earlier-index>"} if options.compare_index else None
    index = scrub(index, root, extra)
    try:
        if options.bundle_dir:
            index["bundle_export"] = scrub(export_bundles(root, options.bundle_dir.resolve(), index), root, extra)
        text = json.dumps(index, indent=2, sort_keys=True) + "\n"
        host_path_check(text, root, "index.json")
        for path in (options.index_json, options.bundle_dir and options.bundle_dir / "index.json"):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        if options.report:
            links = Path(os.path.relpath(options.bundle_dir.resolve(), options.report.resolve().parent)) \
                if options.bundle_dir else None
            text = report(index, links)
            host_path_check(text, root, options.report.name)
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(text)
    except ValueError as exc:
        print(f"Evidence not written: {exc}", file=sys.stderr)
        return 1
    print(f"evidence_complete {index['evidence_complete']}, fidelity_matches_pin {index['fidelity_matches_pin']}, "
          f"CAD {index['cad_passed']}, exit {index['exit_code']}", file=sys.stderr)
    return index["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
