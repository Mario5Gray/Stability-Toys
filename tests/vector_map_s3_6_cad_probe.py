"""Blocking S3.6 CAD acceptance. Run explicitly with a real OpenSCAD executable."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import vector_map  # noqa: E402
from tests.fixtures.vector_map.corpus import metrics  # noqa: E402
from tests import test_vector_map_openscad as open_scad  # noqa: E402
from tests.test_vector_map_scad import scene_files, two_island_mask, write_mask  # noqa: E402

CORPUS = ROOT / "tests/fixtures/vector_map/corpus"
ROLES = ("silhouette", "structure", "detail")
SYNTHETIC = (
    "raised_backed", "engraved", "islands_loose", "islands_backed",
    "donut_open", "donut_filled", "preset",
)
CORPUS_NAMES = tuple(sorted(p.name for p in CORPUS.iterdir() if (p / "mask.png").is_file()))
CASES = SYNTHETIC + tuple(f"corpus_{name}" for name in CORPUS_NAMES)
EXPECTED_PLANES = {
    "raised_backed": [0, 1, 3, 3.5, 4],
    "engraved": [0, 1.5, 2, 3],
    "preset": [0, 1.2, 1.4, 1.8],
}


def role_area_mm2(svg: str, canvas: dict) -> float:
    """Convert the published SVG path area from viewBox px² to physical mm²."""
    return metrics.polygon_area(svg) * canvas["mm_per_px"] ** 2


MODULES = {"silhouette": "silhouette_2d", "structure": "structure_2d", "detail": "detail_2d"}


def effective_areas_mm2(directory: Path, command: list[str], roles) -> dict[str, float]:
    """Areas of the 2D regions the SCAD extrudes, in mm².

    The source clips structure and detail to the silhouette and removes structure from detail, so
    the published polygon area overcounts any clipped material. OpenSCAD exports each module as SVG
    with DETAIL_GROW forced to zero, so a growth shows up in the measured mesh, not the expectation.
    """
    text = (directory / "out.scad").read_text()
    head = text[:text.index("module relief")]
    areas = {}
    for role in roles:
        source, svg = directory / f"area-{role}.scad", directory / f"area-{role}.svg"
        source.write_text(f"{head}{MODULES[role]}();\n")
        run = subprocess.run([*command, "-D", "DETAIL_GROW=0", "-o", str(svg), str(source)],
                             capture_output=True, text=True, timeout=300, cwd=directory)
        if run.returncode:
            raise RuntimeError(f"OpenSCAD 2D export of {role} failed: {run.stderr}")
        total = 0.0
        for data in re.findall(r' d="([^"]*)"', svg.read_text()):
            for ring in re.findall(r"M[^M]*", data):
                points = [(float(x), float(y)) for x, y in re.findall(r"(-?[\d.e+-]+),(-?[\d.e+-]+)", ring)]
                total += sum(points[i][0] * points[(i + 1) % len(points)][1]
                             - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))) / 2
        areas[role] = abs(total)
    return areas


def expected_volume(manifest: dict, directory: Path, command: list[str] | None = None) -> tuple[float, dict]:
    """Use the manifest scale, the requested heights, and the regions the SCAD extrudes.

    With a command the areas are effective, after clipping and removal (see effective_areas_mm2).
    Without one they are the published polygon areas, which overcount clipped material.
    """
    canvas = manifest["canvas"]
    relief = manifest["relief"]
    areas = {"canvas": canvas["width_mm"] * canvas["height_mm"]}
    for entry in manifest["layers"]:
        role = entry["id"]
        areas[role] = (role_area_mm2((directory / entry["svg"]).read_text(), canvas)
                       if entry["svg"] else 0.0)
    if command:
        published = [role for role in MODULES if areas.get(role)]
        areas.update(effective_areas_mm2(directory, command, published))
    sign = -1 if relief["detail_mode"] == "engraved" else 1
    volume = relief["backing_mm"] * areas["canvas"]
    volume += (relief["silhouette_thickness_mm"] or 0) * areas.get("silhouette", 0)
    volume += (relief["structure_height_mm"] or 0) * areas.get("structure", 0)
    volume += sign * (relief["detail_height_mm"] or 0) * areas.get("detail", 0)
    return volume, areas


def _mask_expectation(mask_path: Path, canvas: dict, backing: float, speckle: int = 0) -> dict:
    """Expected bounds, solids and genus from the mask.

    The tracer drops components smaller than speckle**2 pixels (vtracer filter_speckle, S1.4). The
    expected solid count skips them. The raw mask count stays in mask_solids.
    """
    material = np.asarray(Image.open(mask_path).convert("L")) >= 128
    components, holes = metrics.topology(material)
    scale = canvas["mm_per_px"]
    if backing:
        bounds = [[0.0, 0.0], [canvas["width_mm"], canvas["height_mm"]]]
        return {"bounds_xy": bounds, "solids": 1, "genus_sum": 0}
    ys, xs = np.nonzero(material)
    if not len(xs):
        raise ValueError(f"Empty silhouette mask: {mask_path}")
    bounds = [[float(xs.min() * scale), float((material.shape[0] - 1 - ys.max()) * scale)],
              [float((xs.max() + 1) * scale), float((material.shape[0] - ys.min()) * scale)]]
    sizes = np.bincount(ndimage.label(material)[0].ravel())[1:]
    traced = int(np.count_nonzero(sizes >= speckle * speckle))
    return {"bounds_xy": bounds, "solids": traced, "mask_solids": int(components),
            "filter_speckle_px": speckle, "genus_sum": int(holes)}


def _save_scene(directory: Path, name: str) -> tuple[dict, list[str]]:
    directory.mkdir(parents=True)
    if name.startswith("corpus_"):
        source_dir = CORPUS / name.removeprefix("corpus_")
        for file in ("source.png", "mask.png"):
            shutil.copyfile(source_dir / file, directory / file)
        return {"source": directory / "source.png", "mask": directory / "mask.png"}, [
            "--width-mm", "100", "--layers", "silhouette,structure,detail",
            "--preset", "relief-0.4",
        ]
    scene = scene_files(directory)
    if name.startswith("islands_"):
        write_mask(scene["mask"], two_island_mask())
    if name.startswith("donut_"):
        ring = np.zeros((24, 32), bool)
        ring[2:22, 2:30] = True
        ring[8:16, 11:21] = False
        write_mask(scene["mask"], ring)
    if name in ("raised_backed", "engraved"):
        extra = ["--layers", "silhouette,structure,detail",
                 "--structure-map", str(scene["structure"]), "--detail-map", str(scene["detail"]),
                 "--silhouette-thickness-mm", "2", "--structure-height-mm", "1",
                 "--detail-height-mm", "0.5"]
        if name == "raised_backed":
            extra += ["--backing-mm", "1"]
        else:
            extra += ["--detail-mode", "engraved"]
    elif name == "preset":
        extra = ["--layers", "silhouette,structure,detail",
                 "--structure-map", str(scene["structure"]), "--detail-map", str(scene["detail"]),
                 "--preset", "relief-0.4"]
    else:
        extra = ["--layers", "silhouette", "--silhouette-thickness-mm", "2"]
        if name in ("islands_backed", "donut_filled"):
            extra += ["--backing-mm", "1"]
    return scene, ["--width-mm", "32", *extra]


def _vertices(stl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    triples = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", stl_path.read_text())
    if not triples:
        raise ValueError(f"OpenSCAD emitted no STL triangles: {stl_path}")
    corners = np.array([[float(value) for value in row] for row in triples])
    vertices, index = np.unique(corners.round(6), axis=0, return_inverse=True)
    return vertices, index.reshape(-1, 3)


def _top_at(vertices: np.ndarray, triangles: np.ndarray, x: float, y: float) -> float | None:
    """Return the highest mesh surface at an XY point away from triangle edges."""
    hits = []
    point = np.array([x, y])
    for triangle in vertices[triangles]:
        a, b, c = triangle
        matrix = np.column_stack(((b - a)[:2], (c - a)[:2]))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-9:
            continue
        u, v = np.linalg.solve(matrix, point - a[:2])
        if u >= -1e-8 and v >= -1e-8 and u + v <= 1 + 1e-8:
            hits.append(float(a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])))
    return max(hits) if hits else None


def _preview(vertices: np.ndarray, triangles: np.ndarray, path: Path) -> None:
    """Rasterize the highest STL surface at each XY pixel as a top-view height map."""
    low, high = vertices[:, :2].min(0), vertices[:, :2].max(0)
    span = np.maximum(high - low, 1e-9)
    scale = min(560 / span[0], 360 / span[1])
    xy = np.column_stack((40 + (vertices[:, 0] - low[0]) * scale,
                          40 + (high[1] - vertices[:, 1]) * scale))
    top = np.full((440, 640), -np.inf)
    for face in triangles:
        p0, p1, p2 = xy[face]
        denominator = ((p1[0] - p0[0]) * (p2[1] - p0[1])
                       - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        if abs(denominator) < 1e-9:  # Vertical faces have no top-view area.
            continue
        x0, x1 = max(0, int(np.floor(xy[face, 0].min()))), min(639, int(np.ceil(xy[face, 0].max())))
        y0, y1 = max(0, int(np.floor(xy[face, 1].min()))), min(439, int(np.ceil(xy[face, 1].max())))
        if x0 > x1 or y0 > y1:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        dx, dy = xx + 0.5 - p0[0], yy + 0.5 - p0[1]
        u = (dx * (p2[1] - p0[1]) - dy * (p2[0] - p0[0])) / denominator
        v = ((p1[0] - p0[0]) * dy - (p1[1] - p0[1]) * dx) / denominator
        inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        z0, z1, z2 = vertices[face, 2]
        z = z0 + u * (z1 - z0) + v * (z2 - z0)
        region = top[y0:y1 + 1, x0:x1 + 1]
        region[inside] = np.maximum(region[inside], z[inside])
    occupied = np.isfinite(top)
    image_pixels = np.full((440, 640, 3), 255, dtype=np.uint8)
    if occupied.any():
        z_min, z_max = float(top[occupied].min()), float(top[occupied].max())
        height_fraction = ((top[occupied] - z_min) / (z_max - z_min)
                           if z_max > z_min else np.ones(occupied.sum()))
        image_pixels[occupied] = np.column_stack((220 - 115 * height_fraction,
                                                   228 - 115 * height_fraction,
                                                   244 - 58 * height_fraction)).astype(np.uint8)
    image = Image.fromarray(image_pixels, "RGB")
    draw = ImageDraw.Draw(image)
    draw.text((40, 8), "Top view - color shows top surface height", fill=(30, 36, 54))
    if occupied.any():
        draw.text((420, 8), f"Z {z_min:g} to {z_max:g} mm", fill=(30, 36, 54))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _hashes(directory: Path) -> dict[str, dict]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(directory))] = {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    return files


def verify_manifest_hashes(root: Path) -> dict:
    checked, mismatches = 0, []
    for manifest in sorted(root.glob("*/out.vector.json")):
        for item in json.loads(manifest.read_text())["artifacts"]:
            path = manifest.parent / item["path"]
            checked += 1
            if (not path.is_file() or path.stat().st_size != item["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]):
                mismatches.append(str(path))
    return {"checked": checked, "mismatches": mismatches}


def _run_case(name: str, root: Path, previews: Path, command: list[str], version: str) -> dict:
    directory = root / name
    scene, extra = _save_scene(directory, name)
    argv = [str(scene["source"]), str(directory / "out.svg"), "--input-kind", "image",
            "--mask", str(scene["mask"]), *extra, "--export-scad", "--debug-bundle", "--json"]
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = vector_map.main(argv)
    conversion_seconds = time.perf_counter() - started
    result = json.loads(stdout.getvalue())
    row = {"case": name, "conversion": {"argv": ["python", "scripts/vector_map.py", *argv],
                                        "exit_code": code, "result": result,
                                        "stderr": stderr.getvalue(), "seconds": conversion_seconds},
           "openscad_version": version, "work_dir": str(directory)}
    if code != 0:
        row["passed"] = False
        row["errors"] = ["Conversion failed."]
        row["artifacts"] = _hashes(directory)
        return row
    manifest = json.loads((directory / "out.vector.json").read_text())
    row["relief"] = manifest["relief"]
    render_argv = [*command, "-o", str(directory / "out.stl"), str(directory / "out.scad")]
    row["render"] = {"argv": render_argv}
    if issue := open_scad.sandbox_error(command, [directory / "out.scad", directory / "out.stl"]):
        row["passed"], row["errors"] = False, [issue]
        return row
    started = time.perf_counter()
    render = subprocess.run(render_argv, capture_output=True, text=True, timeout=300)
    row["render"].update(exit_code=render.returncode, stdout=render.stdout, stderr=render.stderr,
                         seconds=time.perf_counter() - started)
    if render.returncode:
        row["passed"], row["errors"] = False, ["OpenSCAD render failed."]
        row["artifacts"] = _hashes(directory)
        return row
    vertices, triangles = _vertices(directory / "out.stl")
    parts = open_scad.mesh_components(vertices, triangles)
    measured = {"bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                "solids": len(parts), "genus_per_solid": [int(part[2]) for part in parts],
                "genus_sum": int(sum(part[2] for part in parts)),
                "z_planes": sorted({round(float(z), 6) for z in vertices[:, 2]}),
                "volume_mm3": metrics.mesh_volume(vertices[triangles])}
    volume, areas = expected_volume(manifest, directory, command)
    raw_volume, raw_areas = expected_volume(manifest, directory)
    expectation = _mask_expectation(directory / "out.debug/silhouette.png", manifest["canvas"],
                                    manifest["relief"]["backing_mm"],
                                    manifest["vtracer"]["filter_speckle"])
    relief = manifest["relief"]
    ztop = relief["backing_mm"] + (relief["silhouette_thickness_mm"] or 0)
    ztop += max(relief["structure_height_mm"] or 0,
                0 if relief["detail_mode"] == "engraved" else relief["detail_height_mm"] or 0)
    expectation.update(volume_mm3=volume, areas_mm2=areas, z_max=ztop, volume_basis="effective_openscad_2d",
                       published_areas_mm2=raw_areas, published_volume_mm3=raw_volume,
                       published_volume_excess=(raw_volume - volume) / volume if volume else 0.0)
    errors = [line for line in render.stderr.splitlines()
              if line.startswith("ERROR:") or "not be a valid 2-manifold" in line]
    scale = manifest["canvas"]["mm_per_px"]
    xy_tol = scale if name.startswith("corpus_") else 1e-4
    for observed, wanted in ((measured["bounds"][0][:2], expectation["bounds_xy"][0]),
                             (measured["bounds"][1][:2], expectation["bounds_xy"][1])):
        if not np.allclose(observed, wanted, atol=xy_tol, rtol=0):
            errors.append(f"XY bounds differ beyond {xy_tol:g} mm: {observed} versus {wanted}.")
    if abs(measured["bounds"][0][2]) > 1e-6 or abs(measured["bounds"][1][2] - ztop) > 1e-6:
        errors.append(f"Z bounds differ: {measured['bounds']} versus [0, {ztop}].")
    if measured["solids"] != expectation["solids"]:
        errors.append(f"Solid count {measured['solids']} versus {expectation['solids']}.")
    if measured["genus_sum"] != expectation["genus_sum"]:
        errors.append(f"Genus sum {measured['genus_sum']} versus {expectation['genus_sum']}.")
    if not metrics.mesh_volume_ok(measured["volume_mm3"], volume):
        errors.append(f"Volume {measured['volume_mm3']} versus {volume} mm³ exceeds 1e-4 relative "
                      f"(published-area figure {raw_volume}).")
    if name in EXPECTED_PLANES and measured["z_planes"] != EXPECTED_PLANES[name]:
        errors.append(f"Z planes {measured['z_planes']} versus {EXPECTED_PLANES[name]}.")
    if name == "engraved":
        measured["sample_tops"] = {"recess": _top_at(vertices, triangles, 10.5, 8.5),
                                   "ridge": _top_at(vertices, triangles, 10.5, 15.5)}
        if measured["sample_tops"] != {"recess": 1.5, "ridge": 3.0}:
            errors.append(f"Engraved support samples differ: {measured['sample_tops']}.")
    preview = previews / f"{name}.png"
    _preview(vertices, triangles, preview)
    row.update(measured=measured, expected=expectation, artifacts=_hashes(directory),
               preview=str(preview), preview_sha256=hashlib.sha256(preview.read_bytes()).hexdigest(),
               errors=errors, passed=not errors)
    return row


def _report(evidence: dict) -> str:
    lines = ["# S3.6 CAD acceptance", "", f"OpenSCAD: `{evidence['openscad_version']}`.",
             f"Work directory: `{evidence['work_dir']}`.", "",
             "Regenerate from the repository root with an empty work directory: `python -m tests.vector_map_s3_6_cad_probe --work-dir /private/tmp/s36-cad-review --evidence-json /private/tmp/s36-cad-review.json --report /private/tmp/s36-cad-review.md --preview-dir /private/tmp/s36-cad-review-previews`.",
             "The JSON preserves the exact argv used for this run.",
             "Reproduce individual conversions and renders with the exact argv in each case's JSON record.", "",
             f"Result: {sum(row['passed'] for row in evidence['cases'])}/{len(evidence['cases'])} cases passed.",
             f"Manifest artifact hashes: {evidence['manifest_hash_check']['checked']} checked, {len(evidence['manifest_hash_check']['mismatches'])} mismatches.",
             "OpenSCAD may return exit 0 while reporting a non-closed or non-manifold mesh; those diagnostics fail a CAD row.",
             "The blocking volume check uses effective areas: the regions the generated SCAD extrudes, exported by OpenSCAD as 2D with DETAIL_GROW forced to zero. This derives the expectation from the SCAD and OpenSCAD, so it checks the 3D assembly, heights and closure but not the 2D clip itself. Bounds, solid count, genus and Z planes stay independent checks.",
             "Each row also records the published-area figure (the sum of the published SVG areas times the heights) and its excess over the effective figure. The published figure overcounts material that structure or detail loses to clipping, so it is evidence and not a blocking check.", ""]
    lines += ["| Case | CAD result |", "|---|---|"]
    lines += [f"| {row['case']} | {'PASS' if row['passed'] else 'FAIL'} |"
              for row in evidence["cases"]]
    lines.append("")
    for row in evidence["cases"]:
        lines += [f"## {row['case']}", "", f"Result: {'PASS' if row['passed'] else 'FAIL'}.", ""]
        if "conversion" in row:
            lines.append(f"Conversion command: `{' '.join(row['conversion']['argv'])}`.")
            lines.append(f"Conversion exit: {row['conversion']['exit_code']}.")
        if "render" in row:
            lines.append(f"OpenSCAD command: `{' '.join(row['render']['argv'])}`.")
            lines.append(f"OpenSCAD exit: {row['render'].get('exit_code', 'not run')}.")
        if "measured" in row:
            lines += [f"Expected: `{json.dumps(row['expected'], sort_keys=True)}`.",
                      f"Measured: `{json.dumps(row['measured'], sort_keys=True)}`."]
        if row.get("preview"):
            lines.append(f"![{row['case']} mesh](2026-10-14-vector-map-s3-6-cad-acceptance/{row['case']}.png)")
        if row.get("errors"):
            lines.append(f"Findings: {'; '.join(row['errors'])}")
        lines.append("")
    lines += ["## Limits", "", "These renders test CAD geometry on the listed fixtures.",
              "S1.4 tracing fidelity and physical print suitability have separate evidence.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--case", choices=CASES, action="append")
    options = parser.parse_args(argv)
    command = open_scad._openscad()
    if not command:
        print("OpenSCAD is absent. S3.6 CAD acceptance is blocked.", file=sys.stderr)
        return 2
    version_run = subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=30)
    if version_run.returncode:
        print(f"OpenSCAD version command failed: {version_run.stderr}", file=sys.stderr)
        return 2
    version = (version_run.stdout or version_run.stderr).strip()
    root = options.work_dir.resolve()
    if root.exists() and any(root.iterdir()):
        print(f"Work directory is not empty: {root}", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    previews = (options.preview_dir or root / "previews").resolve()
    previews.mkdir(parents=True, exist_ok=True)
    selected = options.case or list(CASES)
    rows = []
    for name in selected:
        try:
            row = _run_case(name, root, previews, command, version)
        except Exception as exc:
            row = {"case": name, "passed": False, "errors": [f"{type(exc).__name__}: {exc}"]}
        rows.append(row)
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'}", file=sys.stderr)
    evidence = {"schema_version": 1, "openscad_version": version, "openscad_command": command,
                "invocation": [sys.executable, "-m", "tests.vector_map_s3_6_cad_probe",
                               *(argv if argv is not None else sys.argv[1:])],
                "work_dir": str(root), "manifest_hash_check": verify_manifest_hashes(root),
                "cases": rows}
    if options.evidence_json:
        options.evidence_json.parent.mkdir(parents=True, exist_ok=True)
        options.evidence_json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(_report(evidence))
    return 0 if all(row["passed"] for row in rows) and not evidence["manifest_hash_check"]["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
