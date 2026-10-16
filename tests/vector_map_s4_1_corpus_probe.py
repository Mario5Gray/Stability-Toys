"""Blocking S4.1 corpus acceptance (STABL-etpzsxvn). Run explicitly with a real OpenSCAD executable.

Plan: docs/superpowers/plans/2026-10-15-vector-map-s4-1.md.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import vector_map  # noqa: E402
from tests.fixtures.vector_map.corpus import metrics  # noqa: E402

CORPUS = ROOT / "tests/fixtures/vector_map/corpus"
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
