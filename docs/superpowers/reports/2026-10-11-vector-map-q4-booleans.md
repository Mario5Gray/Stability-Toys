# Q4: OpenSCAD 2D Booleans on published S3.3 layers

Issue: `STABL-szyudowd`. Base: merged S3.3 on `main` at `5800f10`. Date: 2026-10-11.
Decision: **accept 2D intersection and difference for S3.4's published-layer assembly**, with the limits below. Do not use raw, pre-subtraction candidates as CAD inputs.

## Reproduce

On the macOS host with OpenSCAD 2021.01 and the `stability-toys` Python environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/darkbit1001/miniforge3/envs/stability-toys/bin/python tests/vector_map_q4_probe.py > /private/tmp/q4_full_result.json
```

The probe creates a temporary directory under `/private/tmp`, invokes the merged S3.3 `vector_map.main` CLI path with `--input-kind image --layers silhouette,structure,detail --structure-map ... --detail-map ...`, then imports the **actual published** `out.layers/{silhouette,structure,detail}.svg` files. All masks pass through S3.3 preparation, separate VTracer traces, and S2.5 normalization. Each SCAD render uses `linear_extrude(height=2)` and invokes `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o <case>.stl <case>.scad`. The script parses STL triangles and reports whole-mesh bounds, connected closed-surface components, and genus per component. An assertion fails if any direct/Boolean metric differs. Native OpenSCAD reported `OpenSCAD version 2021.01`.

The expressions are:

```scad
import("out.layers/structure.svg", center=false);
intersection() {
  import("out.layers/structure.svg", center=false);
  import("out.layers/silhouette.svg", center=false);
}
import("out.layers/detail.svg", center=false);
difference() {
  intersection() {
    import("out.layers/detail.svg", center=false);
    import("out.layers/silhouette.svg", center=false);
  }
  intersection() {
    import("out.layers/structure.svg", center=false);
    import("out.layers/silhouette.svg", center=false);
  }
}
```

Each expression is rendered separately; the direct and Boolean results are compared by role. This is a source-generation decision, not the complete raised/engraved assembly acceptance in S3.6.

## Fixture matrix and measured output

| Fixture | Properties | Silhouette components / genus sum | Structure direct = Boolean | Detail direct = Boolean | Detail components / genus sum |
|---|---|---:|---|---|---:|
| `compound_nested` | Asymmetric, border-touching silhouette; donut hole and nested island; separate structure components; detail donut | 2 / 1 | yes, 2 components | yes | 1 / 1 |
| `crossing` | Structure crosses outer and hole boundaries and nested island; raw detail crosses structure before S3.3 subtraction | 2 / 1 | yes, 3 components | yes | 2 / 0 |
| `border` | Structure touches canvas and outer silhouette borders; detail follows a hole boundary | 1 / 1 | yes, 2 components | yes | 1 / 0 |
| `near` | Independently traced straight shared structure/detail boundary | 1 / 0 | yes, 1 component | yes | 1 / 0 |
| `stair` | Independently traced stepped shared boundary with multiple corners | 1 / 0 | yes, 1 component | yes | 1 / 0 |

All direct and Boolean structure/detail renders also had equal per-component XYZ bounds and genus, including the stepped shared boundary; no extra sliver component appeared. All rendered Z bounds were `[0,2]` mm. Examples of measured whole-mesh XY bounds in mm:

| Fixture | Silhouette XY min → max | Structure XY min → max | Detail XY min → max |
|---|---|---|---|
| `compound_nested` | `(0,0.73999) → (12.95,11.1)` | `(1.10999,3.7) → (11.47,9.98999)` | `(2.95999,1.10999) → (6.65999,3.7)` |
| `crossing` | `(0,0) → (15.17,11.84)` | `(0,5.17999) → (15.17,6.28999)` | `(7.76999,0) → (8.87999,11.84)` |
| `border` | `(0,0.52916) → (9.26041,7.9375)` | `(0,0.52916) → (9.26041,7.40833)` | `(5.82083,1.32291) → (6.87917,6.61458)` |
| `near` | `(0.73999,0.73999) → (15.54,11.1)` | `(1.10999,1.84999) → (7.76999,9.98999)` | `(7.76999,1.84999) → (14.8,9.98999)` |
| `stair` | `(0.73999,0.73999) → (17.02,11.84)` | `(1.48,1.84999) → (7.76999,10.73)` | `(5.92,1.84999) → (15.91,10.73)` |

The probe prints all per-component bounds, genus values, published fixture paths, and exact commands as JSON. Paths are temporary by design; rerunning the command regenerates them.

## Import behavior and limits

The separate open-path control `M1,1 L12,1 L12,12` imported as one component with XY bounds `(0.5,3.5) → (12.5,15.5)`. Closing it with `Z` yielded `(1,4) → (12,15)`. Thus OpenSCAD 2021.01 treats open and closed SVG paths differently. The production S2.5 inspector accepts closed M/L/Z subpaths only; S3.3 publication rejects unsupported path commands, shapes, and styling rather than sending them to OpenSCAD. Its accepted material is black, nonzero fill with limited translate transforms. The proof is confined to that emitted subset; arbitrary authored SVG features are outside Q4.

The five fixtures show that 2D Booleans preserve the measured published-layer topology and bounds on this importer. They do not prove that every independently fitted curve has zero area change below the measured precision, or that a finished relief is one printable solid. S3.6 performs complete raised/engraved mesh acceptance. Retain the CAD Booleans as a final boundary guard, and keep the S3.3 raster clipping and subtraction.
