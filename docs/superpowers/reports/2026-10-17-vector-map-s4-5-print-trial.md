# S4.5 physical print trial

**FP:** `STABL-wurhddhb` · **Plan:** [2026-10-17-vector-map-s4-5.md](../plans/2026-10-17-vector-map-s4-5.md)
**Date:** 2026-10-17 · **Recorded by:** theta · **Operator:** Mario

**Physical print: PASS for both rows, as reported by the operator.**

This report holds the operator's slicer and print evidence. It is separate from conversion, CAD, and
fidelity evidence. The CLI `physical_print` field stays `pending`. The CLI never reads this report.

## Source rows

Both rows come from the [S4.1 corpus report](2026-10-15-vector-map-s4-1-corpus.md). Both used
`--preset relief-0.4` at 100 mm width.

| Row | OpenSCAD source | S4.1 record | S4.1 fidelity | S4.1 CAD | Solids |
|---|---|---|---|---|---:|
| `badge/raised` | [`out.scad`](2026-10-15-vector-map-s4-1-corpus/badge/raised/out.scad) | [`result.json`](2026-10-15-vector-map-s4-1-corpus/badge/raised/result.json) | PASS | PASS | 6 |
| `brick/engraved` | [`out.scad`](2026-10-15-vector-map-s4-1-corpus/brick/engraved/out.scad) | [`result.json`](2026-10-15-vector-map-s4-1-corpus/brick/engraved/result.json) | FAIL (centroid, topology) | PASS | 1 |

Brick's S1.4 fidelity failure is a known limitation of the raster trace. It is not a print result.

## OpenSCAD STL export

Mario exported the STL for each row with OpenSCAD 2021.01, then sliced it. Theta repeated the export on
2026-10-17 from the committed files, in each row's directory:

```bash
export OPENSCAD=/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD
"$OPENSCAD" --version   # OpenSCAD version 2021.01
"$OPENSCAD" -o <row>.stl out.scad
```

| Row | Imports (all committed) | Exit | OpenSCAD `Volumes` | STL bounds (mm) | Matches S4.1 CAD bounds |
|---|---|---:|---:|---|---|
| `badge/raised` | `out.layers/silhouette.svg`, `out.layers/structure.svg` | 0 | 7 | [3.9062, 3.9062, 0] to [96.0938, 96.0938, 1.8] | Yes |
| `brick/engraved` | `out.layers/detail.svg`, `out.layers/silhouette.svg`, `out.layers/structure.svg` | 0 | 2 | [0, 0, 0] to [100, 100, 1.8] | Yes |

OpenSCAD's `Volumes` count includes the outer volume, so 7 and 2 agree with the 6 and 1 solids in S4.1.
The STL is a temporary slicer input. The committed SCAD and SVG files are the reproducible source. No STL is committed.

## Bambu Studio settings

Bambu Studio default settings. No tuning. The plan copies only these three fields.

| Field | Value | Source |
|---|---|---|
| Application version | Bambu Studio 02.08.04.61 | Saved 3MF metadata |
| Printer preset | Bambu Lab A1 mini 0.4 nozzle | Saved 3MF metadata |
| Filament preset | Generic PLA @BBL A1M | Confirmed by Mario |

The saved 3MF files list two PLA filament presets. Mario confirmed which one was used. He gave one
filament preset for the trial, so this report records the same value for both rows.

## Observations

### `badge/raised`

| Check | Result | Note |
|---|---|---|
| Slicer | PASS | The STL loaded and sliced. Bambu Studio showed no warning. The preview showed every expected piece and the raised structure. No feature was missing or thinned. |
| Print | PASS | The sample was completed. All pieces are intact. The raised relief is visible. Thickness and scale look correct. No defect was reported, so no difference from the slicer preview was reported. |

### `brick/engraved`

| Check | Result | Note |
|---|---|---|
| Slicer | PASS | The STL loaded and sliced. Bambu Studio showed no warning. The preview showed the engraved detail. No feature was missing or thinned. |
| Print | PASS | The sample was completed. The piece is intact. The engraving is intact and visible. Thickness and scale look correct. No defect was reported, so no difference from the slicer preview was reported. |

The brick mesh has Z planes at 1.19 and 1.20 mm. Bambu Studio showed no warning, and Mario reported no
defect. This report makes no claim about how the slicer handled the 0.01 mm interval.

## Evidence limits

- All slicer and print observations are Mario's. This report does not measure the samples independently.
  "Thickness and scale look correct" is a visual check, not a caliper measurement.
- A CAD pass or a slicer preview did not decide either print verdict. Mario's observation of each sample did.
- After the values above were recorded on `STABL-wurhddhb`, the 3MF files were deleted at Mario's request.
  They are not kept as evidence.
- This report changes no converter, preset, corpus bundle, dependency, or CLI evidence value.
