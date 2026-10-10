"""S3.2 Canny default sweep over the reviewed corpus (STABL-uifsadne).

Plan: docs/superpowers/plans/2026-10-09-vector-map-s3-2.md, "Canny default selection".
Every candidate runs through vector_map_layers.source_rgb and canny_map.canny_edges.
Each reviewed mask is the silhouette. Its full oriented canvas is kept. Gap closing stays disabled.

Run with the dedicated interpreter:
    /Users/darkbit1001/miniforge3/envs/stability-toys/bin/python spikes/vector_map_canny_sweep.py
Add --structure LOW,HIGH,BLUR to record detail counts after removal of that structure candidate.
Add --detail LOW,HIGH,BLUR as well to draw defaults.png: structure and detail after removal per case.

Writes tests/fixtures/vector_map/corpus/canny-sweep.json and labeled contact sheets in
docs/superpowers/reports/2026-10-09-vector-map-s3-2-canny/. It never writes sweep-results.json,
which measures VTracer speckle.
"""

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import canny_map  # noqa: E402
import vector_map_config as config  # noqa: E402
import vector_map_layers as layers  # noqa: E402
import vector_map_preview as preview  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "vector_map" / "corpus"
OUT = CORPUS / "canny-sweep.json"
SHEETS = ROOT / "docs" / "superpowers" / "reports" / "2026-10-09-vector-map-s3-2-canny"
PAIRS = [(50, 100), (75, 150), (100, 200), (125, 250)]
BLURS = [0, 3, 5]
GRID = [layers.CannySpec(low, high, blur) for low, high in PAIRS for blur in BLURS]


def label(spec):
    return f"{spec.low_threshold}/{spec.high_threshold} b{spec.blur}"


def case_settings(name):
    source, mask = CORPUS / name / "source.png", CORPUS / name / "mask.png"
    with Image.open(source) as image:
        width = image.size[0]
    layer = {**dict.fromkeys(config.FIELDS), "mask": mask}
    return config.resolve(config.DEFAULTS, {"input": source, "input_kind": "image", "width_mm": float(width)}, layer)


def prepare(settings, *requests):
    return layers.prepare_layers(settings, list(requests))


def direct(rgb, spec):
    """Direct canny_map call on the exact prepared pixels. Parity reference only."""
    edges = canny_map.canny_edges(Image.fromarray(rgb, "RGB"), low_threshold=spec.low_threshold,
                                  high_threshold=spec.high_threshold, blur=spec.blur, invert=False)
    return np.asarray(edges) >= 128


def components(material):
    return int(cv2.connectedComponents(material.astype(np.uint8), connectivity=8)[0] - 1)


def sheet(name, silhouette, rgb, candidates):
    """Source, silhouette, then the threshold-by-blur grid of clipped candidates."""
    font = preview.label_font()
    height, width = silhouette.shape
    cell_w, cell_h = width + 8, height + 22
    page = Image.new("RGB", (8 + len(BLURS) * cell_w, 8 + (1 + len(PAIRS)) * cell_h), preview.PAGE)
    draw = ImageDraw.Draw(page)
    header = [("source (prepared RGB)", Image.fromarray(rgb, "RGB")),
              ("silhouette (reviewed mask)", Image.fromarray(np.where(silhouette, 0, 255).astype(np.uint8)))]
    for index, (text, image) in enumerate(header):
        left = 8 + index * cell_w
        draw.text((left, 8), text, font=font, fill=preview.TEXT)
        page.paste(image.convert("RGB"), (left, 26))
    for row, (low, high) in enumerate(PAIRS, start=1):
        for column, blur in enumerate(BLURS):
            spec = layers.CannySpec(low, high, blur)
            material = candidates[spec]
            left, top = 8 + column * cell_w, 8 + row * cell_h
            draw.text((left, top), f"{name} {label(spec)} n={int(material.sum())}", font=font, fill=preview.TEXT)
            page.paste(Image.fromarray(np.where(material, 0, 255).astype(np.uint8)).convert("RGB"), (left, top + 18))
    path = SHEETS / f"{name}.png"
    page.save(path, format="PNG", optimize=True)
    return path.relative_to(ROOT).as_posix()


def defaults_sheet(chosen, structure, detail):
    """One row per case: structure default, then detail after structure removal."""
    font = preview.label_font()
    width = max(material.shape[1] for _, material, _ in chosen) + 8
    height = max(material.shape[0] for _, material, _ in chosen) + 22
    page = Image.new("RGB", (8 + 2 * width, 8 + len(chosen) * height), preview.PAGE)
    draw = ImageDraw.Draw(page)
    for row, (name, coarse, fine) in enumerate(chosen):
        top = 8 + row * height
        for column, (text, material) in enumerate(((f"{name} structure {label(structure)}", coarse),
                                                   (f"{name} detail {label(detail)} minus structure", fine))):
            left = 8 + column * width
            draw.text((left, top), f"{text} n={int(material.sum())}", font=font, fill=preview.TEXT)
            page.paste(Image.fromarray(np.where(material, 0, 255).astype(np.uint8)).convert("RGB"), (left, top + 18))
    path = SHEETS / "defaults.png"
    page.save(path, format="PNG", optimize=True)
    return path.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", help="LOW,HIGH,BLUR of the selected structure candidate")
    parser.add_argument("--detail", help="LOW,HIGH,BLUR of the selected detail candidate")
    args = parser.parse_args()
    structure = layers.CannySpec(*map(int, args.structure.split(","))) if args.structure else None
    detail = layers.CannySpec(*map(int, args.detail.split(","))) if args.detail else None
    chosen = []
    SHEETS.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((CORPUS / "corpus.json").read_text())
    cases = {}
    for name in sorted(corpus["cases"]):
        settings = case_settings(name)
        base = prepare(settings)
        silhouette = base.silhouette
        rgb = layers.source_rgb(settings.input, silhouette.oriented_size, silhouette.processed_size)
        area = int(silhouette.material.sum())
        rows, candidates = [], {}
        for spec in GRID:
            prepared = prepare(settings, layers.RoleRequest("structure", canny=spec))
            material = prepared.structure.material
            raw = layers.canny_material(rgb, spec)
            parity = bool(np.array_equal(raw, direct(rgb, spec))
                          and np.array_equal(material, raw & silhouette.material))
            candidates[spec] = material
            rows.append({
                "low_threshold": spec.low_threshold, "high_threshold": spec.high_threshold, "blur": spec.blur,
                "parity": parity,
                "edge_pixels": int(raw.sum()),
                "edge_density": round(float(raw.mean()), 5),
                "clipped_pixels": int(material.sum()),
                "clipped_density": round(float(material.sum()) / area, 5),
                "components": components(material),
            })
        case = {"processing_size": list(silhouette.processed_size), "silhouette_pixels": area,
                "candidates": rows, "contact_sheet": sheet(name, silhouette.material, rgb, candidates)}
        if structure is not None:
            case["structure"] = {"low_threshold": structure.low_threshold,
                                 "high_threshold": structure.high_threshold, "blur": structure.blur}
            case["detail_after_structure"] = []
            for spec in GRID:
                prepared = prepare(settings, layers.RoleRequest("structure", canny=structure),
                                   layers.RoleRequest("detail", canny=spec))
                # Plan rule without width, gaps or constraints: clipped detail minus clipped structure.
                expected = candidates[spec] & ~candidates[structure]
                case["detail_after_structure"].append({
                    "low_threshold": spec.low_threshold, "high_threshold": spec.high_threshold, "blur": spec.blur,
                    "detail_pixels": int(expected.sum()),
                    "structure_pixels": int(candidates[structure].sum()),
                    "detail_parity": bool(np.array_equal(prepared.detail.material, expected)),
                })
        if structure is not None and detail is not None:
            chosen.append((name, candidates[structure], candidates[detail] & ~candidates[structure]))
        cases[name] = case
        print(name, "parity", all(row["parity"] for row in rows))
    result = {
        "version": 1,
        "issue": "STABL-uifsadne",
        "grid": {"pairs": PAIRS, "blurs": BLURS},
        "path": "vector_map_layers.source_rgb -> canny_map.canny_edges, clipped to reviewed mask",
        "gap_closing": "disabled",
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                        "pillow": version("Pillow"), "numpy": np.__version__},
        "cases": cases,
    }
    if chosen:
        result["defaults_sheet"] = defaults_sheet(chosen, structure, detail)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
