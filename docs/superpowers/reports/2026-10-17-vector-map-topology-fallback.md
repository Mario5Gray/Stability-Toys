# Vector map topology-preserving fallback — research spike

**FP:** `STABL-wuzmvgxm` (child of `STABL-wzfgcsse`) · **Date:** 2026-10-17 · **Author:** theta
**Script:** `spikes/vector_map_topology_fallback.py` (throwaway research code, imported by nothing in `scripts/`)
**Evidence:** [`results.json`](2026-10-17-vector-map-topology-fallback/results.json), [`openscad.json`](2026-10-17-vector-map-topology-fallback/openscad.json)

```bash
conda activate stability-toys
python spikes/vector_map_topology_fallback.py
```

Environment: Python 3.12.13, macOS 26.0.1 arm64, vtracer 0.6.15, resvg-py 0.5.0, numpy 2.5.1, scipy 1.17.1,
scikit-image 0.26.0, Pillow 12.3.0. OpenSCAD 2021.01 for the extrusion check.

## Question

Can a fallback recover the locked S1.4 fidelity rules on the four hard corpus cases
(bracket, brick, gravel, horse) without a regression on the four passing cases? What does it cost?

## Constraints held

- The eight reviewed S1.4 masks are unchanged. The S1.4 tolerances are unchanged.
- The script refuses to run when `tests/fixtures/vector_map/corpus/metrics.py` differs from the S1.4 hash.
- Every method is scored with `metrics.evaluate`, `metrics.render`, `metrics.polygon_area`, and
  `metrics.svg_complexity`. Timing uses `metrics.TIMING`: one warm-up, median of five calls.
- No production code changed. No dependency changed.

## Calibration

- `vtracer` at speckle 4 reproduces the S1.4 sweep row for horse exactly: `centroid, topology`, 132 commands, 1406 bytes.
- The local-width measure reproduces the production 4x4 diagnostic on badge: 38 material and 5 background pixels.
- Two runs give identical `results.json` except for trace times.

## Methods

| Method | What it does |
|---|---|
| `vtracer` | Current production trace, `filter_speckle` 4. |
| `up{k}_s{s}` | Nearest-neighbour upscale by k, VTracer at speckle s, viewBox maps the vectors back. `s = 4` keeps the speckle in traced pixels. `s = 4k²` keeps it in source pixels. |
| `pixel_edge` | Exact pixel-boundary (crack) tracer in NumPy. Holes wind opposite to their outline. Topology-exact by construction. |
| `router_width` | `vtracer` when material and background minimum widths are both at least 4 px. Otherwise `pixel_edge`. |
| `router_verify` | `vtracer` when it passes S1.4 against its own mask. Otherwise `pixel_edge`. |

## Findings

**1. `pixel_edge` passes all eight cases.** XOR, area error, and boundary error are zero. Topology matches,
including gravel at 225 components and 13 holes. The cost is 3.08x to 8.83x the VTracer command count and
3.1x to 7.1x the bytes. The largest output is gravel: 11,526 commands and 92,815 bytes. That is far below the
per-layer limits of 1,000,000 commands and 20,971,520 bytes.

**2. Upscaling is the pixel staircase, at a higher cost.** Every passing upscale variant has XOR 0 and the same
command count as `pixel_edge`. VTracer smooths in traced pixels, so at ×2 and ×4 it keeps every source step.
It adds trace time (×4 at speckle 4: 22.7–65.4 ms against 2.6–81.2 ms for `pixel_edge`) and coordinate bytes. With speckle scaled to
source pixels it breaks cases: `up2_s16` fails gravel and truchet. `up4_s64` fails bracket, gravel, pcb, and truchet.
Reject: it gives no benefit over `pixel_edge`.

**3. Width detection cannot select the hard cases.** Every case, passing or hard, has pixels below 4 px.
Truchet has 13,197 thin material pixels and passes. Horse has 64 and fails. The skeleton minimum is 2.0 px in
most cases in both sets. `router_width` therefore sends all eight cases to `pixel_edge`. That is the same as
always using `pixel_edge`. Reject as a router.

**4. `router_verify` selects exactly the four hard cases.** The four passing cases keep their VTracer output.
The cost falls only on the hard cases: 3.08x (brick) to 5.92x (horse) commands. The router needs a raster
check at conversion time. The S4.2 contract says normal conversion must not depend on resvg-py
(`docs/superpowers/plans/2026-10-16-vector-map-s4-2.md`, contract item 6). An in-process NumPy or OpenCV
rasterizer could replace resvg-py. This spike did not measure one.

**5. `pixel_edge` output passes the production dialect and a CAD extrusion.** `vector_map_svg.normalize_svg`
accepts all eight: one path, no degenerate subpaths. OpenSCAD 2021.01 extrudes all eight with volume error
at most 2.3e-5 (limit 1e-4) and exact Z bounds. This is the S1.3 single-layer extrusion check. It is not the
S3.6 layered relief gate.

### Minimum widths

Local width is the side of the largest axis-aligned square of the same class that covers a pixel. Off-canvas
pixels count as background. Skeleton width is the minimum inscribed diameter on interior medial-axis pixels.

| Case | Set | Material min | Background min | Material < 4 px | Background < 4 px | Skeleton material | Skeleton background |
|---|---|---:|---:|---:|---:|---:|---:|
| badge | passing | 1 | 1 | 38 | 5 | 2.0 | 7.2 |
| coins | passing | 1 | 3 | 18 | 15 | 18.4 | 2.0 |
| pcb | passing | 4 | 1 | 0 | 128 | 4.0 | 14.0 |
| truchet | passing | 1 | 1 | 13197 | 121 | 2.0 | 2.0 |
| bracket | hard | 2 | 2 | 3367 | 11 | 2.0 | 6.0 |
| brick | hard | 1 | 1 | 10378 | 181 | 2.0 | 2.0 |
| gravel | hard | 1 | 1 | 6310 | 5856 | 2.0 | 2.0 |
| horse | hard | 1 | 1 | 64 | 178 | 2.0 | 2.0 |

### Per-case measurements

| Case | Method | Result | Topology (mask → trace) | XOR / perimeter | Boundary max px | Centroid px | Commands | Bytes | Trace ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| badge | `vtracer` | PASS | (6, 2) → (6, 2) | 0.216 | 0.50 | 0.068 | 156 | 1606 | 3.3 |
| badge | `pixel_edge` | PASS | (6, 2) → (6, 2) | 0.000 | 0.00 | 0.000 | 1378 | 11397 | 4.4 |
| badge | `up2_s4` | PASS | (6, 2) → (6, 2) | 0.000 | 0.00 | 0.000 | 1378 | 11943 | 11.6 |
| badge | `up2_s16` | PASS | (6, 2) → (6, 2) | 0.000 | 0.00 | 0.000 | 1378 | 11943 | 12.0 |
| badge | `up4_s4` | PASS | (6, 2) → (6, 2) | 0.000 | 0.00 | 0.000 | 1378 | 12632 | 37.0 |
| badge | `up4_s64` | PASS | (6, 2) → (6, 2) | 0.000 | 0.00 | 0.000 | 1378 | 12632 | 38.3 |
| coins | `vtracer` | PASS | (24, 0) → (24, 0) | 0.127 | 0.50 | 0.135 | 341 | 3695 | 2.8 |
| coins | `pixel_edge` | PASS | (24, 0) → (24, 0) | 0.000 | 0.00 | 0.000 | 1736 | 14121 | 5.2 |
| coins | `up2_s4` | PASS | (24, 0) → (24, 0) | 0.000 | 0.00 | 0.000 | 1736 | 13780 | 8.6 |
| coins | `up2_s16` | PASS | (24, 0) → (24, 0) | 0.000 | 0.00 | 0.000 | 1736 | 13780 | 9.1 |
| coins | `up4_s4` | PASS | (24, 0) → (24, 0) | 0.000 | 0.00 | 0.000 | 1736 | 14518 | 22.7 |
| coins | `up4_s64` | PASS | (24, 0) → (24, 0) | 0.000 | 0.00 | 0.000 | 1736 | 14518 | 22.7 |
| pcb | `vtracer` | PASS | (30, 64) → (30, 64) | 0.099 | 0.50 | 0.083 | 1082 | 8342 | 4.5 |
| pcb | `pixel_edge` | PASS | (30, 64) → (30, 64) | 0.000 | 0.00 | 0.000 | 3806 | 30935 | 16.8 |
| pcb | `up2_s4` | PASS | (30, 64) → (30, 64) | 0.000 | 0.00 | 0.000 | 3806 | 27614 | 13.9 |
| pcb | `up2_s16` | PASS | (30, 64) → (30, 64) | 0.000 | 0.00 | 0.000 | 3806 | 27614 | 13.6 |
| pcb | `up4_s4` | PASS | (30, 64) → (30, 64) | 0.000 | 0.00 | 0.000 | 3806 | 30032 | 31.8 |
| pcb | `up4_s64` | FAIL boundary, centroid, topology | (30, 64) → (16, 50) | 0.406 | 2.50 | 9.755 | 2966 | 23692 | 32.6 |
| truchet | `vtracer` | PASS | (55, 20) → (55, 20) | 0.256 | 1.50 | 0.074 | 2354 | 19367 | 9.9 |
| truchet | `pixel_edge` | PASS | (55, 20) → (55, 20) | 0.000 | 0.00 | 0.000 | 9245 | 75041 | 41.3 |
| truchet | `up2_s4` | PASS | (55, 20) → (55, 20) | 0.000 | 0.00 | 0.000 | 9245 | 73448 | 30.3 |
| truchet | `up2_s16` | FAIL boundary, topology | (55, 20) → (52, 20) | 0.008 | 1.74 | 0.330 | 9186 | 72935 | 30.4 |
| truchet | `up4_s4` | PASS | (55, 20) → (55, 20) | 0.000 | 0.00 | 0.000 | 9245 | 78220 | 65.4 |
| truchet | `up4_s64` | FAIL boundary, centroid, topology | (55, 20) → (13, 2) | 0.385 | 1.74 | 5.757 | 6897 | 59711 | 53.1 |
| bracket | `vtracer` | FAIL centroid | (6, 3) → (6, 3) | 0.089 | 0.50 | 2.464 | 147 | 1483 | 2.6 |
| bracket | `pixel_edge` | PASS | (6, 3) → (6, 3) | 0.000 | 0.00 | 0.000 | 603 | 5079 | 3.8 |
| bracket | `up2_s4` | PASS | (6, 3) → (6, 3) | 0.000 | 0.00 | 0.000 | 603 | 4947 | 9.1 |
| bracket | `up2_s16` | PASS | (6, 3) → (6, 3) | 0.000 | 0.00 | 0.000 | 603 | 4947 | 9.3 |
| bracket | `up4_s4` | PASS | (6, 3) → (6, 3) | 0.000 | 0.00 | 0.000 | 603 | 5240 | 33.0 |
| bracket | `up4_s64` | FAIL boundary, centroid, topology | (6, 3) → (4, 3) | 0.051 | 1.74 | 0.656 | 573 | 4938 | 31.4 |
| brick | `vtracer` | FAIL centroid, topology | (1, 53) → (5, 53) | 0.283 | 0.91 | 2.262 | 703 | 5518 | 5.2 |
| brick | `pixel_edge` | PASS | (1, 53) → (1, 53) | 0.000 | 0.00 | 0.000 | 2168 | 17231 | 18.9 |
| brick | `up2_s4` | PASS | (1, 53) → (1, 53) | 0.000 | 0.00 | 0.000 | 2168 | 18213 | 15.0 |
| brick | `up2_s16` | PASS | (1, 53) → (1, 53) | 0.000 | 0.00 | 0.000 | 2168 | 18213 | 15.5 |
| brick | `up4_s4` | PASS | (1, 53) → (1, 53) | 0.000 | 0.00 | 0.000 | 2168 | 18826 | 43.5 |
| brick | `up4_s64` | PASS | (1, 53) → (1, 53) | 0.000 | 0.00 | 0.000 | 2168 | 18826 | 44.1 |
| gravel | `vtracer` | FAIL boundary, topology | (225, 13) → (208, 27) | 0.244 | 1.74 | 0.464 | 2808 | 28649 | 10.5 |
| gravel | `pixel_edge` | PASS | (225, 13) → (225, 13) | 0.000 | 0.00 | 0.000 | 11526 | 92815 | 81.2 |
| gravel | `up2_s4` | PASS | (225, 13) → (225, 13) | 0.000 | 0.00 | 0.000 | 11526 | 89183 | 33.4 |
| gravel | `up2_s16` | FAIL boundary, centroid, topology | (225, 13) → (93, 13) | 0.248 | 3.74 | 1.550 | 8748 | 65865 | 26.8 |
| gravel | `up4_s4` | PASS | (225, 13) → (225, 13) | 0.000 | 0.00 | 0.000 | 11526 | 95100 | 57.1 |
| gravel | `up4_s64` | FAIL area, boundary, bounds, centroid, scale, topology, xor | (225, 13) → (21, 13) | 0.825 | 5.50 | 10.896 | 4734 | 38814 | 38.5 |
| horse | `vtracer` | FAIL centroid, topology | (1, 0) → (1, 2) | 0.356 | 0.91 | 0.851 | 132 | 1406 | 2.1 |
| horse | `pixel_edge` | PASS | (1, 0) → (1, 0) | 0.000 | 0.00 | 0.000 | 781 | 6397 | 2.6 |
| horse | `up2_s4` | PASS | (1, 0) → (1, 0) | 0.000 | 0.00 | 0.000 | 781 | 7416 | 7.8 |
| horse | `up2_s16` | PASS | (1, 0) → (1, 0) | 0.000 | 0.00 | 0.000 | 781 | 7416 | 7.9 |
| horse | `up4_s4` | PASS | (1, 0) → (1, 0) | 0.000 | 0.00 | 0.000 | 781 | 7685 | 26.4 |
| horse | `up4_s64` | PASS | (1, 0) → (1, 0) | 0.000 | 0.00 | 0.000 | 781 | 7685 | 25.8 |

### Routers and cost

| Case | Set | `router_width` | `router_verify` | vtracer commands | pixel_edge commands | Ratio | vtracer bytes | pixel_edge bytes |
|---|---|---|---|---:|---:|---:|---:|---:|
| badge | passing | pixel_edge | vtracer | 156 | 1378 | 8.83x | 1606 | 11397 |
| coins | passing | pixel_edge | vtracer | 341 | 1736 | 5.09x | 3695 | 14121 |
| pcb | passing | pixel_edge | vtracer | 1082 | 3806 | 3.52x | 8342 | 30935 |
| truchet | passing | pixel_edge | vtracer | 2354 | 9245 | 3.93x | 19367 | 75041 |
| bracket | hard | pixel_edge | pixel_edge | 147 | 603 | 4.10x | 1483 | 5079 |
| brick | hard | pixel_edge | pixel_edge | 703 | 2168 | 3.08x | 5518 | 17231 |
| gravel | hard | pixel_edge | pixel_edge | 2808 | 11526 | 4.10x | 28649 | 92815 |
| horse | hard | pixel_edge | pixel_edge | 132 | 781 | 5.92x | 1406 | 6397 |

## Not measured

- **Saddle points.** No pixel-edge ring shares a vertex in this corpus. The reviewed masks were selected so
  that 4- and 8-connectivity agree, so they have no diagonal-only pixel contacts. General masks will have them.
  Two rings that touch at one point are the case that CAD pinch repair exists for. Test this before any production use.
- The S3.6 layered relief gate, the S4.1 corpus probe, and the multi-layer path.
- A raster check for `router_verify` without resvg-py.
- Appearance. `pixel_edge` keeps every one-pixel step: 0.390625 mm at the corpus scale.
- Physical print.

## Recommendation

1. Make no production change now. The issue requires a separate decision before any fallback or dependency change.
2. If sub-4-pixel demand appears, the candidate is `router_verify` with the `pixel_edge` fallback, as an explicit opt-in.
3. Before that decision, answer three questions:
   - Which raster check does conversion use: resvg-py, or an in-process rasterizer?
   - Does `pixel_edge` pass the S3.6 relief gate on masks with saddle points?
   - Can a topology-preserving simplification reduce the 3x–9x command cost? This is the next research question.
4. Reject upscaling and width-based routing. The data above shows that neither helps.
