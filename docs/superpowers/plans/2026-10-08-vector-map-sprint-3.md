# Sprint 3 layered image and relief plan

Parent: `STABL-gqgxdbjl`. Planner: Sigma. Status: proposed for review.
Baseline: `0c2a773` on `main`, checked 2026-10-08.
Authority: planning only. Human assigns each implementation issue.
Contract: [vector-map design](../specs/2026-09-06-vector-map-relief-design.md), sections 4–10 and 11.3.
Sprint 2 evidence: [standalone acceptance](../reports/2026-10-07-vector-map-sprint-2-acceptance.md).

## Goal and boundaries

Image mode produces one selected silhouette and optional structure and detail layers.
Every layer keeps the oriented source canvas, processing size, physical scale, and origin.
The CLI publishes aligned SVGs and can generate an editable OpenSCAD assembly.
Actual CAD renders establish Sprint 3 acceptance.

Keep standalone mask and edge commands compatible.
Keep Torch, server imports, learned segmentation, internal STL generation, and automatic geometry repair outside this sprint.
Keep `STABL-eibrjlsh` separate. It does not block Sprint 3 without a new finding.
The three known full-suite failures remain separately tracked. Record them beside Sprint 3 test results.

## Proposed execution order

| Order | Existing FP issue | Deliverable | Gate |
|---|---|---|---|
| 1 | `STABL-memwrtos` S3.1 | Image silhouette selection | Reviewed mask, alpha, and threshold cases |
| 2 | `STABL-uifsadne` S3.2 | Selected Canny or external feature masks | Raster composition and preview cases |
| 3 | `STABL-qlagdbmh` S3.3 | Combined and separate aligned SVGs | Geometry, manifest, and publication cases |
| Parallel with 1–3 | `STABL-szyudowd` Q4 | OpenSCAD 2D Boolean decision | Real render and mesh evidence |
| 4 | `STABL-gorlibrt` S3.4 | SCAD source generation | S3.3 and accepted Q4 result |
| 5 | `STABL-mknlfcui` S3.5 | Explicit fabrication preset | S3.4 height controls, overrides, provenance |
| Last | `STABL-sxcphzwd` S3.6 | Complete CAD acceptance | S3.4 and S3.5 |

Existing FP dependencies encode every required gate except the proposed S3.4-to-S3.5 gate.
S3.5 currently depends on S3.3. Review should add S3.4 to its dependency list.
Q4 must finish before S3.4 because source generation depends on its Boolean decision.
Q4's description says "before S3.5", but the S3.4 dependency records the effective gate.
No new issue is needed unless review divides an existing issue into smaller independent outputs.

## Proposed interface decisions for review

1. Image mode requires exactly one silhouette method: `--mask`, `--alpha`, or `--threshold`.
   `--threshold` uses oriented source luminance at or above an explicit integer from 0 to 255.
   `--invert` reverses that selection. A missing method exits 2 with an instruction.
   Reject an empty silhouette after constraints with a processing failure.
2. `--layers` selects `silhouette`, `structure`, and `detail` in fixed output order.
   Image mode defaults to `silhouette` only. The silhouette is always selected.
   Optional layers use either a supplied map or an explicit Canny settings object.
   Do not infer structure or detail from image meaning.
3. Add recipe fields for each layer's map, Canny thresholds, blur, nominal band width, and raster constraints.
   Resolve relative paths from the recipe directory. Keep schema version 1.
   CLI values override recipe values. Unsupported fields exit 2.
   Choose Canny default values from the reviewed corpus before implementation.
4. Preserve the standalone JSON fields and exit codes.
   Add relative layer paths and SCAD path under `artifacts` when those files exist.
   Add selected layer identities and resolved heights to the manifest without changing schema version 1.

These choices need review before an implementation task starts.
The threshold name and recipe field shape are proposed policy, not existing behavior.

## File responsibilities

| File | Planned responsibility |
|---|---|
| `scripts/vector_map.py` | Keep CLI thin. Select the standalone or layered route. |
| `scripts/vector_map_config.py` | Validate methods, layer selection, recipe fields, and preset precedence. |
| `scripts/vector_map_raster.py` | Reuse orientation, resizing, constraints, and band expansion. |
| `scripts/vector_map_layers.py` (new) | Produce immutable, aligned binary masks for selected roles. |
| `scripts/vector_map_svg.py` | Assemble named groups from normalized layer SVGs with XML parsing. |
| `scripts/vector_map_artifacts.py` | Extend fixed ownership, input snapshots, manifest, and staged publication. |
| `scripts/vector_map_preview.py` | Show each selected prepared mask and its rendered vector output. |
| `scripts/vector_map_scad.py` (new) | Generate relative imports and editable height parameters. |
| `scripts/pyproject.toml` | Ship new modules in the vector package. |
| `scripts/USAGE.md` | Describe image selection, layers, relief parameters, and CAD render commands. |

Keep VTracer as the only tracer. Preserve raw traced bytes before normalization.
Use `resvg-py==0.5.0` with a pixel-sized SVG root for preview geometry.
Use the existing embedded-font fallback on Pillow 10.0 or without FreeType.
Never parse or assemble SVG with regular expressions.

## S3.1 image silhouette

1. Add RED tests in `tests/test_vector_map_image.py` for all three methods.
   Cover EXIF orientation, matching external-mask dimensions, alpha absence, and explicit threshold polarity.
   Cover missing method, conflicting methods, empty silhouette, and unchanged mask and edge routes.
2. Run `python -m pytest tests/test_vector_map_image.py tests/test_vector_map_raster_cli.py -q`.
   Record tests that fail because image mode remains deferred.
3. Extend configuration and raster preparation with one immutable silhouette result.
   Apply existing include and exclude masks after selection.
   Derive the canvas once from the common processing size.
4. Repeat the focused command to GREEN. Check a real image conversion through VTracer.
5. Check `drift refs` for each edited file. Update bound prose before `drift link`.
6. Commit only S3.1 paths. Report the commit and RED/GREEN evidence in FP.

## S3.2 structure and detail proposals

1. Add RED tests in `tests/test_vector_map_layers.py` for selected roles and external maps.
   Cover mismatched oriented dimensions, empty optional layers, and include/exclude precedence.
2. Compare candidate edges with direct `canny_map.canny_edges` output on the same oriented, resized RGB pixels.
   Test separate coarse and fine settings. Record the exact values in the recipe.
3. Expand feature bands through the existing physical-width function.
   Reapply constraints, intersect each band with the silhouette, then remove structure from detail.
4. Add preview tests for selected intermediate masks.
   S3.3 adds rendered SVG panels after separate layer exports exist.
5. Run `python -m pytest tests/test_vector_map_layers.py tests/test_vector_map_preview.py -q` to RED and GREEN.
6. Commit only S3.2 paths. Report candidate settings, tests, and visual limits in FP.

## S3.3 aligned SVG and bundle

1. Add RED tests in `tests/test_vector_map_layer_export.py` for a common root and `viewBox`.
   Check an asymmetric source, a donut, and a nested island at known physical dimensions.
2. Trace each nonempty selected layer with the pinned VTracer adapter.
   Normalize each layer with the S2.5 inspector. Retain the complete canvas.
3. Build combined SVG named groups with `ElementTree`.
   Treat separate normalized layer SVGs as the authoritative CAD inputs.
   Add preview tests for every selected rendered SVG and its prepared mask.
   Keep pixel-root rendering and overlay alignment at non-binary millimetre scales.
4. Extend `Bundle` to own `<stem>.layers/<role>.svg` only for selected nonempty roles.
   Reject a symlink layer directory. Protect every source and external map from output aliases.
5. Include each published artifact and input hash in the manifest.
   Report omitted optional layers. Do not list stale unrequested files as current outputs.
6. Test collision refusal, `--overwrite`, staged failure, manifest-last publication, and repeated bundle hashes.
   Preserve unrelated files in the layer directory.
7. Run `python -m pytest tests/test_vector_map_layer_export.py tests/test_vector_map_artifacts.py -q` to RED and GREEN.
8. Commit only S3.3 paths. Report geometry and publication evidence in FP.

## Q4 imported-layer Boolean decision

1. Add a focused proof using S2.5-normalized SVGs and real OpenSCAD 2021.01.
   Render imported structure intersected with the silhouette.
   Render detail minus structure. Use compound paths and a nested hole.
2. Include a border-touching region and asymmetric placement.
   Compare STL XY/Z bounds, surface components, and genus with expected geometry.
3. Record OpenSCAD version, exact command, output, and fixture paths in `STABL-szyudowd`.
   Keep Flatpak files under the home directory when that executable is used.
4. Accept 2D booleans only if every required case passes.
   If any case fails, record the failure and review an alternative assembly before S3.4.

## S3.4 SCAD assembly

1. Add RED tests in `tests/test_vector_map_scad.py` for relative imports and `center=false`.
   Require one assembly-level centering transform when requested.
2. Implement Q4's accepted Boolean strategy.
   Constrain structure and detail to the silhouette. Remove structure from detail again.
3. Expose backing `B`, silhouette `S`, structure `H`, and detail `D` as editable millimetre parameters.
   Check the spec's Z intervals for raised and engraved cases.
   Require `0 < D < S` for engraving and positive selected feature heights.
4. Keep engraving out of structure ridges.
   Document small support overlaps without changing the intended top heights.
5. Add an optional full-canvas rectangular backing.
   Warn when a disconnected silhouette without backing may form multiple solids.
6. Add `<stem>.scad` to fixed bundle ownership and manifest hashes.
   Source generation must work without an OpenSCAD executable.
7. Run `python -m pytest tests/test_vector_map_scad.py tests/test_vector_map_artifacts.py -q` to RED and GREEN.
8. Commit only S3.4 paths. Report validated source and remaining CAD evidence in FP.

## S3.5 explicit relief preset

1. Add RED tests in `tests/test_vector_map_preset.py` for no implicit preset.
   Check `--preset relief-0.4` and every CLI or recipe override.
2. Set `S=1.2 mm`, structure width `0.8 mm`, `H=0.6 mm`, detail width `0.5 mm`, and `D=0.2 mm`.
   Keep backing disabled. Do not change S1.4 VTracer defaults.
3. Record all resolved values in the manifest and replay recipe.
   Document that printer suitability requires separate evidence.
4. Run `python -m pytest tests/test_vector_map_preset.py tests/test_vector_map_scad.py -q` to RED and GREEN.
5. Commit only S3.5 paths. Report override and provenance evidence in FP.

## S3.6 CAD acceptance

1. Run complete image and external-mask workflows with the reviewed corpus.
   Save each selected mask, combined SVG, separate SVG, manifest, and SCAD source.
2. Render actual OpenSCAD meshes for raised structure, raised detail, engraved detail, backing on, and backing off.
   Include a preset end-to-end case and a silhouette with holes.
3. Record executable version, exact commands, output, expected and measured XY/Z bounds, components, and genus.
   Treat absent OpenSCAD as blocked acceptance, not a passing skip.
4. Compare each case with its expected geometry. Do not hide a failing case in an aggregate score.
5. Prove installed console behavior in a fresh venv from `./scripts[vector]`.
   The local conda environment contains an older installed script copy.
6. Run `python -m pytest tests/test_vector_map_*.py -q -rs` and `python -m tests.run tests/ -- -q -rs`.
   Run the native CPU test container. Separate baseline full-suite failures from new failures.
7. Run `drift check`, `git diff --check`, and manifest hash verification.
   Record conversion and CAD evidence separately. Leave print validation for Sprint 4.
8. Commit acceptance evidence with issue ID. Report ready for independent review.

## Planning and review gates

Review the four proposed interface decisions before S3.1 starts.
Review Q4's render evidence before S3.4 starts.
Each human-popped implementation task uses inline RED/GREEN work and stops at ready for review.
Do not self-pop the next task or mark an issue done before its review cycle.
No waveplan runtime bundle is configured in this checkout. Do not invent schedule state.
