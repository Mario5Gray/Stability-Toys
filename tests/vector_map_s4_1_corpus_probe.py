"""Blocking S4.1 corpus acceptance (STABL-etpzsxvn). Run explicitly with a real OpenSCAD executable.

Plan: docs/superpowers/plans/2026-10-15-vector-map-s4-1.md.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
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
    source = CORPUS / case
    return [str(source / "source.png"), str(directory / "out.svg"), "--input-kind", "image",
            "--mask", str(source / "mask.png"), "--width-mm", "100",
            "--layers", "silhouette,structure,detail", "--preset", "relief-0.4",
            *(["--detail-mode", "engraved"] if mode == "engraved" else []),
            "--export-scad", "--debug-bundle", "--preview", "--json"]


def _convert(argv: list[str]) -> tuple[int, str, str, float]:
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
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
    index.update(schema_version=1, host=_host(openscad_version), openscad_command=command,
                 invocation=[sys.executable, "-m", "tests.vector_map_s4_1_corpus_probe",
                             *(argv if argv is not None else sys.argv[1:])], work_dir=str(root))
    if options.index_json:
        options.index_json.parent.mkdir(parents=True, exist_ok=True)
        options.index_json.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"evidence_complete {index['evidence_complete']}, fidelity_matches_pin {index['fidelity_matches_pin']}, "
          f"CAD {index['cad_passed']}, exit {index['exit_code']}", file=sys.stderr)
    return index["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
