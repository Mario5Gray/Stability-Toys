"""S1.4 phase two: filter_speckle sweep over corpus v1 (STABL-cbwzjyky).

Uses only the locked phase-one rules in tests/fixtures/vector_map/corpus/metrics.py
(commit b95c718). Every conversion goes through the merged adapter in polygon mode.

Run with the dedicated Python 3.12 interpreter (VTracer 0.6.15 crashes on 3.14):
    /Users/darkbit1001/miniforge3/envs/stability-toys/bin/python spikes/vector_map_corpus_sweep.py

Writes tests/fixtures/vector_map/corpus/sweep-results.json.
Add --openscad to check mesh volume and Z bounds at each case's resolved value.

Override rule, declared here before the sweep runs:
A case that fails at the selected default gets a recipe override. Among the grid
values that pass that case, take the one nearest the default by grid index.
On a tie, take the smaller value, because it keeps more detail.
A case that passes at no grid value has no override and is recorded as failing.
"""

import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import vector_map_vtracer as adapter  # noqa: E402
from tests.fixtures.vector_map.corpus import make_corpus, metrics  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "vector_map" / "corpus"
OUT = CORPUS / "sweep-results.json"
PHASE_ONE = "b95c718"
LOCKED_METRICS_SHA256 = "7050d8cc4938a2fc75103c735c64430b275a32d79b74488861e4c8f1ac672898"


def _timed(call):
    """One untimed warm-up, then the locked number of timed calls. Return (last value, median ms)."""
    for _ in range(metrics.TIMING["warmup_calls"]):
        call()
    times, values = [], []
    for _ in range(metrics.TIMING["timed_calls"]):
        start = time.perf_counter()
        values.append(call())
        times.append((time.perf_counter() - start) * 1000)
    return values, statistics.median(times)


def measure(material, speckle):
    height, width = material.shape
    try:
        traces, trace_ms = _timed(lambda: adapter.trace_layer(material, {"filter_speckle": speckle}))
    except Exception as exc:  # recorded per case, never hidden
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}", "failures": ["trace"]}
    svg = traces[-1].svg
    renders, render_ms = _timed(lambda: metrics.render(svg, width, height))
    result = metrics.evaluate(material, renders[-1], metrics.polygon_area(svg))
    result.update(metrics.svg_complexity(svg))
    result.update(
        svg_canvas=list(metrics.svg_canvas(svg)),
        svg_sha256=hashlib.sha256(svg.encode()).hexdigest(),
        trace_repeatable=len({t.svg for t in traces}) == 1,
        render_repeatable=all(np.array_equal(r, renders[0]) for r in renders),
        adapter_median_ms=round(trace_ms, 3),
        resvg_median_ms=round(render_ms, 3),
        upstream_args=traces[-1].upstream_args,
    )
    return result


def override(case, results, default):
    grid = list(metrics.SPECKLE_GRID)
    passing = [value for value in grid if results[value][case]["passed"]]
    if not passing:
        return None
    anchor = grid.index(default)
    return min(passing, key=lambda value: (abs(grid.index(value) - anchor), value))


def _openscad_check(material, speckle, work_dir):
    from tests.test_vector_map_openscad import OPENSCAD, SCAD, physical_svg

    if OPENSCAD is None:
        return {"skipped": "OpenSCAD is absent"}
    height, width = material.shape
    svg = adapter.trace_layer(material, {"filter_speckle": speckle}).svg
    svg_path, stl_path = work_dir / "layer.svg", work_dir / "layer.stl"
    svg_path.write_text(physical_svg(svg, width, height))
    thickness = metrics.EXTRUSION["thickness_mm"]
    command = [*OPENSCAD, "-o", str(stl_path), "-D", f'svg="{svg_path}"', "-D", f"thickness={thickness}", str(SCAD)]
    start = time.perf_counter()
    run = subprocess.run(command, capture_output=True, text=True, timeout=600)
    seconds = time.perf_counter() - start
    if run.returncode:
        return {"error": run.stderr[-2000:], "command": command}
    corners = np.array([[float(v) for v in m] for m in re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", stl_path.read_text())])
    triangles = corners.reshape(-1, 3, 3)
    volume = metrics.mesh_volume(triangles)
    expected = metrics.expected_volume(metrics.polygon_area(svg))
    z_low, z_high = float(corners[:, 2].min()), float(corners[:, 2].max())
    z_ok = abs(z_low) <= metrics.TOLERANCES["mesh_z_abs_mm"] and abs(z_high - thickness) <= metrics.TOLERANCES["mesh_z_abs_mm"]
    return {
        "triangles": int(len(triangles)),
        "volume_mm3": volume,
        "expected_mm3": expected,
        "volume_rel_error": abs(volume - expected) / expected,
        "volume_ok": metrics.mesh_volume_ok(volume, expected),
        "z_bounds_mm": [z_low, z_high],
        "z_ok": z_ok,
        "seconds": round(seconds, 3),
        "stl_sha256": hashlib.sha256(stl_path.read_bytes()).hexdigest(),
        "log_tail": run.stderr.strip().splitlines()[-3:],
    }


def main(argv):
    actual = hashlib.sha256((CORPUS / "metrics.py").read_bytes()).hexdigest()
    if actual != LOCKED_METRICS_SHA256:
        raise SystemExit(f"metrics.py changed since phase one ({actual}). Refuse to measure.")
    masks = {case: np.asarray(Image.open(CORPUS / case / "mask.png")) == 255 for case in sorted(make_corpus.CASES)}

    results = {}
    for speckle in metrics.SPECKLE_GRID:
        results[speckle] = {}
        for case, material in masks.items():
            row = measure(material, speckle)
            results[speckle][case] = row
            print(f"speckle={speckle:<3} {case:<8} {'PASS' if row['passed'] else 'FAIL ' + ','.join(row['failures'])}",
                  flush=True)

    default, reason = metrics.select_default(
        {s: {c: {k: r.get(k, float("inf")) if k != "passed" else r["passed"]
                 for k in ("passed", "xor_per_perimeter_px", "commands")}
             for c, r in cases.items()} for s, cases in results.items()}
    )
    recipes, failing = {}, {}
    if default is not None:
        for case in masks:
            if results[default][case]["passed"]:
                continue
            value = override(case, results, default)
            if value is None:
                failing[case] = {s: results[s][case]["failures"] for s in metrics.SPECKLE_GRID}
            else:
                recipes[case] = {"filter_speckle": value}

    report = {
        "phase_one_commit": PHASE_ONE,
        "metrics_sha256": actual,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "vtracer": version("vtracer"),
            "resvg-py": version("resvg-py"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "pillow": version("pillow"),
        },
        "render": metrics.RENDER,
        "timing": metrics.TIMING,
        "tolerances": metrics.TOLERANCES,
        "grid": list(metrics.SPECKLE_GRID),
        "selection": {"default": default, "reason": reason, "overrides": recipes, "failing_every_value": failing},
        "results": {str(s): cases for s, cases in results.items()},
    }

    if "--openscad" in argv and default is not None:
        import shutil
        import tempfile

        from tests.test_vector_map_openscad import OPENSCAD, choose_work_dir

        base = Path(tempfile.mkdtemp(prefix="s14-openscad-"))
        work, created = choose_work_dir(base, OPENSCAD)
        try:
            report["openscad"] = {}
            for case, material in masks.items():
                value = recipes.get(case, {}).get("filter_speckle")
                if value is None:
                    value = default
                report["openscad"][case] = {"filter_speckle": value, **_openscad_check(material, value, work)}
                print("openscad", case, value, report["openscad"][case].get("volume_ok"), flush=True)
        finally:
            shutil.rmtree(base, ignore_errors=True)
            if created:
                shutil.rmtree(work, ignore_errors=True)

    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("selection:", report["selection"])
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
