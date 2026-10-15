# S3.6 CAD acceptance

OpenSCAD: `OpenSCAD version 2021.01`.
Work directory: `/private/tmp/s36-cad-final2`.

Regenerate from the repository root with an empty work directory: `python -m tests.vector_map_s3_6_cad_probe --work-dir /private/tmp/s36-cad-review --evidence-json /private/tmp/s36-cad-review.json --report /private/tmp/s36-cad-review.md --preview-dir /private/tmp/s36-cad-review-previews`.
The JSON preserves the exact argv used for this run.
Reproduce individual conversions and renders with the exact argv in each case's JSON record.

Result: 10/15 cases passed.
Manifest artifact hashes: 115 checked, 0 mismatches.
OpenSCAD may return exit 0 while reporting a non-closed or non-manifold mesh; those diagnostics fail a CAD row.
The volume formula uses full published SVG areas. CAD intersections can clip these areas, so a volume mismatch alone does not isolate a defect; inspect the render diagnostic and effective geometry.

| Case | CAD result |
|---|---|
| raised_backed | PASS |
| engraved | PASS |
| islands_loose | PASS |
| islands_backed | PASS |
| donut_open | PASS |
| donut_filled | PASS |
| preset | PASS |
| corpus_badge | PASS |
| corpus_bracket | PASS |
| corpus_brick | FAIL |
| corpus_coins | FAIL |
| corpus_gravel | FAIL |
| corpus_horse | PASS |
| corpus_pcb | FAIL |
| corpus_truchet | FAIL |

## raised_backed

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/raised_backed/source.png /private/tmp/s36-cad-final2/raised_backed/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/raised_backed/mask.png --width-mm 32 --layers silhouette,structure,detail --structure-map /private/tmp/s36-cad-final2/raised_backed/structure.png --detail-map /private/tmp/s36-cad-final2/raised_backed/detail.png --silhouette-thickness-mm 2 --structure-height-mm 1 --detail-height-mm 0.5 --backing-mm 1 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/raised_backed/out.stl /private/tmp/s36-cad-final2/raised_backed/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 768.0, "detail": 60.0, "silhouette": 560.0, "structure": 72.0}, "bounds_xy": [[0.0, 0.0], [32.0, 24.0]], "genus_sum": 0, "solids": 1, "volume_mm3": 1990.0, "z_max": 4.0}`.
Measured: `{"bounds": [[0.0, 0.0, 0.0], [32.0, 24.0, 4.0]], "genus_per_solid": [0], "genus_sum": 0, "solids": 1, "volume_mm3": 1990.0, "z_planes": [0.0, 1.0, 3.0, 3.5, 4.0]}`.
![raised_backed mesh](2026-10-14-vector-map-s3-6-cad-acceptance/raised_backed.png)

## engraved

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/engraved/source.png /private/tmp/s36-cad-final2/engraved/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/engraved/mask.png --width-mm 32 --layers silhouette,structure,detail --structure-map /private/tmp/s36-cad-final2/engraved/structure.png --detail-map /private/tmp/s36-cad-final2/engraved/detail.png --silhouette-thickness-mm 2 --structure-height-mm 1 --detail-height-mm 0.5 --detail-mode engraved --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/engraved/out.stl /private/tmp/s36-cad-final2/engraved/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 768.0, "detail": 60.0, "silhouette": 560.0, "structure": 72.0}, "bounds_xy": [[2.0, 2.0], [30.0, 22.0]], "genus_sum": 0, "solids": 1, "volume_mm3": 1162.0, "z_max": 3.0}`.
Measured: `{"bounds": [[2.0, 2.0, 0.0], [30.0, 22.0, 3.0]], "genus_per_solid": [0], "genus_sum": 0, "sample_tops": {"recess": 1.5, "ridge": 3.0}, "solids": 1, "volume_mm3": 1162.0, "z_planes": [0.0, 1.5, 2.0, 3.0]}`.
![engraved mesh](2026-10-14-vector-map-s3-6-cad-acceptance/engraved.png)

## islands_loose

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/islands_loose/source.png /private/tmp/s36-cad-final2/islands_loose/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/islands_loose/mask.png --width-mm 32 --layers silhouette --silhouette-thickness-mm 2 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/islands_loose/out.stl /private/tmp/s36-cad-final2/islands_loose/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 768.0, "silhouette": 198.0}, "bounds_xy": [[2.0, 2.0], [30.0, 22.0]], "genus_sum": 0, "solids": 2, "volume_mm3": 396.0, "z_max": 2.0}`.
Measured: `{"bounds": [[2.0, 2.0, 0.0], [30.0, 22.0, 2.0]], "genus_per_solid": [0, 0], "genus_sum": 0, "solids": 2, "volume_mm3": 396.0, "z_planes": [0.0, 2.0]}`.
![islands_loose mesh](2026-10-14-vector-map-s3-6-cad-acceptance/islands_loose.png)

## islands_backed

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/islands_backed/source.png /private/tmp/s36-cad-final2/islands_backed/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/islands_backed/mask.png --width-mm 32 --layers silhouette --silhouette-thickness-mm 2 --backing-mm 1 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/islands_backed/out.stl /private/tmp/s36-cad-final2/islands_backed/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 768.0, "silhouette": 198.0}, "bounds_xy": [[0.0, 0.0], [32.0, 24.0]], "genus_sum": 0, "solids": 1, "volume_mm3": 1164.0, "z_max": 3.0}`.
Measured: `{"bounds": [[0.0, 0.0, 0.0], [32.0, 24.0, 3.0]], "genus_per_solid": [0], "genus_sum": 0, "solids": 1, "volume_mm3": 1164.0, "z_planes": [0.0, 1.0, 3.0]}`.
![islands_backed mesh](2026-10-14-vector-map-s3-6-cad-acceptance/islands_backed.png)

## donut_open

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/donut_open/source.png /private/tmp/s36-cad-final2/donut_open/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/donut_open/mask.png --width-mm 32 --layers silhouette --silhouette-thickness-mm 2 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/donut_open/out.stl /private/tmp/s36-cad-final2/donut_open/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 768.0, "silhouette": 480.0}, "bounds_xy": [[2.0, 2.0], [30.0, 22.0]], "genus_sum": 1, "solids": 1, "volume_mm3": 960.0, "z_max": 2.0}`.
Measured: `{"bounds": [[2.0, 2.0, 0.0], [30.0, 22.0, 2.0]], "genus_per_solid": [1], "genus_sum": 1, "solids": 1, "volume_mm3": 960.0, "z_planes": [0.0, 2.0]}`.
![donut_open mesh](2026-10-14-vector-map-s3-6-cad-acceptance/donut_open.png)

## donut_filled

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/donut_filled/source.png /private/tmp/s36-cad-final2/donut_filled/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/donut_filled/mask.png --width-mm 32 --layers silhouette --silhouette-thickness-mm 2 --backing-mm 1 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/donut_filled/out.stl /private/tmp/s36-cad-final2/donut_filled/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 768.0, "silhouette": 480.0}, "bounds_xy": [[0.0, 0.0], [32.0, 24.0]], "genus_sum": 0, "solids": 1, "volume_mm3": 1728.0, "z_max": 3.0}`.
Measured: `{"bounds": [[0.0, 0.0, 0.0], [32.0, 24.0, 3.0]], "genus_per_solid": [0], "genus_sum": 0, "solids": 1, "volume_mm3": 1728.0, "z_planes": [0.0, 1.0, 3.0]}`.
![donut_filled mesh](2026-10-14-vector-map-s3-6-cad-acceptance/donut_filled.png)

## preset

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/preset/source.png /private/tmp/s36-cad-final2/preset/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/preset/mask.png --width-mm 32 --layers silhouette,structure,detail --structure-map /private/tmp/s36-cad-final2/preset/structure.png --detail-map /private/tmp/s36-cad-final2/preset/detail.png --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/preset/out.stl /private/tmp/s36-cad-final2/preset/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 768.0, "detail": 60.0, "silhouette": 560.0, "structure": 72.0}, "bounds_xy": [[2.0, 2.0], [30.0, 22.0]], "genus_sum": 0, "solids": 1, "volume_mm3": 727.2, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[2.0, 2.0, 0.0], [30.0, 22.0, 1.8]], "genus_per_solid": [0], "genus_sum": 0, "solids": 1, "volume_mm3": 727.1999999999999, "z_planes": [0.0, 1.2, 1.4, 1.8]}`.
![preset mesh](2026-10-14-vector-map-s3-6-cad-acceptance/preset.png)

## corpus_badge

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_badge/source.png /private/tmp/s36-cad-final2/corpus_badge/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_badge/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_badge/out.stl /private/tmp/s36-cad-final2/corpus_badge/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 10000.0, "detail": 0.0, "silhouette": 3456.4971923828125, "structure": 531.3873291015625}, "bounds_xy": [[3.90625, 3.90625], [96.09375, 96.09375]], "genus_sum": 2, "solids": 6, "volume_mm3": 4466.6290283203125, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[3.90625, 3.90625, 0.0], [96.0938, 96.0938, 1.8]], "genus_per_solid": [0, 0, 1, 1, 0, 0], "genus_sum": 2, "solids": 6, "volume_mm3": 4466.632948238609, "z_planes": [0.0, 1.2, 1.8]}`.
![corpus_badge mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_badge.png)

## corpus_bracket

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_bracket/source.png /private/tmp/s36-cad-final2/corpus_bracket/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_bracket/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_bracket/out.stl /private/tmp/s36-cad-final2/corpus_bracket/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 10000.0, "detail": 0.0, "silhouette": 549.163818359375, "structure": 543.212890625}, "bounds_xy": [[11.71875, 17.96875], [88.28125, 90.234375]], "genus_sum": 3, "solids": 6, "volume_mm3": 984.92431640625, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[11.7188, 17.9688, 0.0], [88.2812, 90.2344, 1.8]], "genus_per_solid": [0, 1, 1, 0, 1, 0], "genus_sum": 3, "solids": 6, "volume_mm3": 984.8927812349983, "z_planes": [0.0, 1.2, 1.8]}`.
![corpus_bracket mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_bracket.png)

## corpus_brick

Result: FAIL.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_brick/source.png /private/tmp/s36-cad-final2/corpus_brick/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_brick/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_brick/out.stl /private/tmp/s36-cad-final2/corpus_brick/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 10000.0, "detail": 3.814697265625, "silhouette": 1955.718994140625, "structure": 1933.2122802734375}, "bounds_xy": [[0.0, 0.0], [100.0, 100.0]], "genus_sum": 53, "solids": 1, "volume_mm3": 3507.5531005859375, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[0.0, 0.0, 0.0], [100.0, 100.0, 1.8]], "genus_per_solid": [53], "genus_sum": 53, "solids": 1, "volume_mm3": 3500.3688402302364, "z_planes": [0.0, 1.2, 1.4, 1.8]}`.
![corpus_brick mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_brick.png)
Findings: WARNING: Object may not be a valid 2-manifold and may need repair!; EXPORT-WARNING: Exported object may not be a valid 2-manifold and may need repair; Volume 3500.3688402302364 versus 3507.5531005859375 mm³ exceeds 1e-4 relative.

## corpus_coins

Result: FAIL.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_coins/source.png /private/tmp/s36-cad-final2/corpus_coins/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_coins/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_coins/out.stl /private/tmp/s36-cad-final2/corpus_coins/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 7890.625, "detail": 957.6416015625, "silhouette": 2804.4891357421875, "structure": 1364.0594482421875}, "bounds_xy": [[4.6875, 3.515625], [99.609375, 75.0]], "genus_sum": 0, "solids": 24, "volume_mm3": 4375.3509521484375, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[4.6875, 3.51562, 0.0], [99.6094, 75.0, 1.2]], "genus_per_solid": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "genus_sum": 0, "solids": 24, "volume_mm3": 3365.3947863387607, "z_planes": [0.0, 1.2]}`.
![corpus_coins mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_coins.png)
Findings: ERROR: The given mesh is not closed! Unable to convert to CGAL_Nef_Polyhedron.; ERROR: The given mesh is not closed! Unable to convert to CGAL_Nef_Polyhedron.; Z bounds differ: [[4.6875, 3.51562, 0.0], [99.6094, 75.0, 1.2]] versus [0, 1.7999999999999998].; Volume 3365.3947863387607 versus 4375.3509521484375 mm³ exceeds 1e-4 relative.

## corpus_gravel

Result: FAIL.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_gravel/source.png /private/tmp/s36-cad-final2/corpus_gravel/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_gravel/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_gravel/out.stl /private/tmp/s36-cad-final2/corpus_gravel/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 10000.0, "detail": 160.369873046875, "silhouette": 4284.97314453125, "structure": 3169.5556640625}, "bounds_xy": [[0.0, 0.0], [100.0, 100.0]], "genus_sum": 13, "solids": 225, "volume_mm3": 7075.775146484375, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[0.0, 0.0, 0.0], [100.0, 100.0, 1.4]], "genus_per_solid": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "genus_sum": 13, "solids": 209, "volume_mm3": 5170.305583455119, "z_planes": [0.0, 1.19, 1.2, 1.4]}`.
![corpus_gravel mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_gravel.png)
Findings: ERROR: The given mesh is not closed! Unable to convert to CGAL_Nef_Polyhedron.; Z bounds differ: [[0.0, 0.0, 0.0], [100.0, 100.0, 1.4]] versus [0, 1.7999999999999998].; Solid count 209 versus 225.; Volume 5170.305583455119 versus 7075.775146484375 mm³ exceeds 1e-4 relative.

## corpus_horse

Result: PASS.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_horse/source.png /private/tmp/s36-cad-final2/corpus_horse/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_horse/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_horse/out.stl /private/tmp/s36-cad-final2/corpus_horse/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 8203.125, "detail": 0.0, "silhouette": 2803.7261962890625, "structure": 370.78857421875}, "bounds_xy": [[4.296875, 3.90625], [97.265625, 79.6875]], "genus_sum": 0, "solids": 1, "volume_mm3": 3586.944580078125, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[4.29688, 3.90625, 0.0], [97.2656, 79.6875, 1.8]], "genus_per_solid": [0], "genus_sum": 0, "solids": 1, "volume_mm3": 3586.9439566395, "z_planes": [0.0, 1.2, 1.8]}`.
![corpus_horse mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_horse.png)

## corpus_pcb

Result: FAIL.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_pcb/source.png /private/tmp/s36-cad-final2/corpus_pcb/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_pcb/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_pcb/out.stl /private/tmp/s36-cad-final2/corpus_pcb/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 10000.0, "detail": 0.0, "silhouette": 2207.489013671875, "structure": 1584.930419921875}, "bounds_xy": [[5.859375, 5.46875], [94.53125, 94.140625]], "genus_sum": 64, "solids": 30, "volume_mm3": 3599.945068359375, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[5.85938, 5.46875, 0.0], [94.5312, 94.1406, 1.2]], "genus_per_solid": [2, 6, 1, 2, 5, 2, 2, 1, 1, 1, 1, 2, 1, 2, 4, 2, 5, 3, 1, 1, 2, 1, 3, 1, 1, 6, 1, 1, 1, 2], "genus_sum": 64, "solids": 30, "volume_mm3": 2648.988850805997, "z_planes": [0.0, 1.2]}`.
![corpus_pcb mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_pcb.png)
Findings: ERROR: The given mesh is not closed! Unable to convert to CGAL_Nef_Polyhedron.; Z bounds differ: [[5.85938, 5.46875, 0.0], [94.5312, 94.1406, 1.2]] versus [0, 1.7999999999999998].; Volume 2648.988850805997 versus 3599.945068359375 mm³ exceeds 1e-4 relative.

## corpus_truchet

Result: FAIL.

Conversion command: `python scripts/vector_map.py /private/tmp/s36-cad-final2/corpus_truchet/source.png /private/tmp/s36-cad-final2/corpus_truchet/out.svg --input-kind image --mask /private/tmp/s36-cad-final2/corpus_truchet/mask.png --width-mm 100 --layers silhouette,structure,detail --preset relief-0.4 --export-scad --debug-bundle --json`.
Conversion exit: 0.
OpenSCAD command: `/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD -o /private/tmp/s36-cad-final2/corpus_truchet/out.stl /private/tmp/s36-cad-final2/corpus_truchet/out.scad`.
OpenSCAD exit: 0.
Expected: `{"areas_mm2": {"canvas": 10000.0, "detail": 74.6917724609375, "silhouette": 3349.8382568359375, "structure": 3215.789794921875}, "bounds_xy": [[0.0, 0.0], [100.0, 100.0]], "genus_sum": 20, "solids": 55, "volume_mm3": 5964.2181396484375, "z_max": 1.7999999999999998}`.
Measured: `{"bounds": [[0.0, 0.0, 0.0], [100.0, 100.0, 1.8]], "genus_per_solid": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0], "genus_sum": 20, "solids": 55, "volume_mm3": 5962.058863710845, "z_planes": [0.0, 1.2, 1.4, 1.8]}`.
![corpus_truchet mesh](2026-10-14-vector-map-s3-6-cad-acceptance/corpus_truchet.png)
Findings: WARNING: Object may not be a valid 2-manifold and may need repair!; EXPORT-WARNING: Exported object may not be a valid 2-manifold and may need repair; Volume 5962.058863710845 versus 5964.2181396484375 mm³ exceeds 1e-4 relative.

## Limits

These renders test CAD geometry on the listed fixtures.
S1.4 tracing fidelity and physical print suitability have separate evidence.

## Findings and provenance

The five failed CAD rows are tracked in FP as `STABL-ygtexxzr`. Brick and truchet emitted non-manifold export warnings. Coins, gravel, and pcb emitted CGAL non-closed-mesh errors and lost raised material; OpenSCAD still returned exit 0. The gravel mask has 225 raster components while the resulting mesh has 209; this count also includes the separate S1.4 tracing-fidelity effect and is not, by itself, evidence of a CAD assembly defect. No production module was changed in this acceptance branch.

The blocking volume calculation follows the merged plan: `polygon_area(layer_svg) * mm_per_px²` per role, with the canvas area in mm² and relative tolerance `1e-4`. The generated SCAD also intersects roles with the silhouette and subtracts structure from detail. Those operations can make the effective area smaller than a raw published SVG area. A volume mismatch can therefore include both assembly loss and a limitation of the plan's raw-area expectation. The OpenSCAD mesh diagnostics and missing Z levels above establish the defect independently for coins, gravel, and pcb; the non-manifold warnings establish the other two failures.

The [installed-console evidence](2026-10-14-vector-map-s3-6-cad-acceptance/installed_console.json) records a fresh `./scripts[vector]` installation, dependency versions, exact argv, exit codes, and hashes for an explicit-relief corpus run, a `relief-0.4` run, and standalone mask conversion. All three conversions exited 0. The temporary venv was deleted after capture. Mask mode produced conversion artifacts only, as required by the layered-image admission rule.

Host vector-map tests: 1,417 passed and one `$HOME` sandbox failure; that exact OpenSCAD work-directory test passed when rerun with host access. The first broad host run had 2,850 passed, 18 failed, and 9 skipped while CAD rendering and the CPU container ran concurrently. A sequential host rerun of those failures gave 17 passed and one failed: the existing `test_entry_spans.py::test_an_UNHASHABLE_type_does_not_break_the_span` baseline (`STABL-fjmjolci`). The stub cohort passed 213 tests; the live cohort skipped without CUDA.

The CPU container (`docker compose -f docker-compose.test.yml run --rm test-unit`) exited 1. Its real cohort had 2,747 passed, 10 failed, 41 skipped, and 79 deselected; stub had 213 passed and live had one skip. The ten failures are the existing `test_entry_spans` case plus nine observability/logging cases caused by the test image lacking `/app/docs/observability-contract.md` and `/app/env.*` or using an unexpected default log level. The image has no OpenSCAD: the new native CAD tests skipped, while the two new non-render checks passed. The unrelated container packaging gap is `STABL-bxqlmcmf`.
