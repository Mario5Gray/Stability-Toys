# Sprint 3 layered image and relief plan

Parent: `STABL-gqgxdbjl`. Planner: Sigma. Status: S3.1 assigned to Theta. Later gates remain under review.
Baseline: `0c2a773` on `main`, checked 2026-10-08.
Authority: Mario assigned S3.1 implementation to Theta. Sigma retains Sprint 3 planning. Mario assigns later implementation issues.
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

FP now lists S3.3 and S3.4 as dependencies of S3.5.
Q4 must finish before S3.4 because source generation depends on its Boolean decision.
Q4's description now names S3.4 as its gate.
No new issue is needed unless review divides an existing issue into smaller independent outputs.

## Proposed interface decisions for review

1. Image mode requires exactly one silhouette method: `--mask`, `--alpha`, or `--threshold`.
   Recipe fields `mask`, `alpha`, and `threshold` form one method group.
   Each settings source replaces the prior method when it selects a new method.
   Thus CLI `--threshold 200` replaces recipe `alpha: true`.
   Two positive methods from one source exit 2. `--no-alpha` clears inherited alpha selection.
   If alpha was the only method, `--no-alpha` leaves no method and exits 2.
   If the recipe selects a mask or threshold, `--no-alpha` leaves that method selected.
   `--threshold` uses oriented source luminance at or above an integer from 0 to 255.
   Select material before the common binary resize. `--invert` reverses every selected image method.
   Keep existing `--invert` behavior for standalone mask and edge modes.
   Image `--mask` selects luminance at or above 128. Ignore mask-file alpha with the existing `alpha_ignored` warning.
   With image `--mask`, the source supplies the canvas only. Ignore source alpha without an `alpha_ignored` warning.
   Image `--threshold` also reports `alpha_ignored` when the source has alpha.
   Reject `--mask` and `--threshold` in mask and edge modes with exit 2.
   Reject an empty silhouette after constraints with exit 1.
2. `--layers` selects `silhouette`, `structure`, and `detail` in fixed output order.
   Image mode defaults to `silhouette` only. The silhouette is always selected.
   Reject `--layers` outside image mode. Reject a selection without `silhouette` with exit 2.
   Selecting structure or detail chooses Canny unless a map is supplied for that role.
   Recipe `structure` and `detail` objects may select `source: "canny"` or `source: "map"`.
   A map source requires its path. Canny values may be omitted and use reviewed defaults.
   Recipe `layers` is a list. CLI `--layers` is a comma-separated list.
   Normalize either selection to the fixed output order.
   Do not infer structure or detail from image meaning.
3. Add recipe fields for each layer's map, Canny thresholds, blur, nominal band width, and raster constraints.
   Each role object contains `source`, optional `path`, optional `canny`, optional `width_mm`, and optional masks.
   The `canny` object accepts `low_threshold`, `high_threshold`, and `blur`.
   Role masks use `include_mask` and `exclude_mask`.
   `--structure-map` and `--detail-map` select external maps from the CLI.
   `--structure-width-mm` and `--detail-width-mm` override each role's band width.
   Without a role width or preset width, keep the candidate band at its input thickness.
   Keep standalone `line_width_mm` limited to edge mode.
   Validate `canny.blur` as integer 0 or a positive odd integer.
   Resolve relative paths from the recipe directory. Keep schema version 1.
   CLI values override recipe values. Unsupported fields exit 2.
   S3.2 measures corpus candidates and selects separate coarse and fine Canny defaults.
   Record selected defaults and per-image overrides before S3.2 implementation.
   Orient the colour source first. Composite RGBA onto opaque black before Canny.
   Derive processing dimensions with `_processing_size`.
   Resize RGB with Pillow LANCZOS to those exact dimensions.
   Run `canny_map.canny_edges` on those pixels without another resize.
   Prepare silhouette and external masks before scaling. Resize those binary masks with nearest-neighbour.
4. Preserve the standalone JSON fields and exit codes.
   Use `artifacts.layers` as a role-to-path object and `artifacts.scad` as a path when present.
   Use the same path string convention as existing `artifacts.svg`.
   Keep existing `artifacts.svg`, `artifacts.manifest`, `artifacts.preview`, and `artifacts.debug` fields.
   Layered manifest entries contain `id`, `height_mm`, and `mode`.
   Without `--export-scad`, `height_mm` and `mode` are null, even when a preset is selected.
   With `--export-scad`, silhouette mode is `base`, structure mode is `raised`.
   Detail mode is `raised` or `engraved`. Standalone entries retain their existing two-field shape.
   Keep schema version 1. Record backing, centering, and preset in resolved relief settings.
5. Add `--export-scad`, `--preset relief-0.4`, and explicit relief controls.
   Restrict SCAD export and the relief preset to image mode with a selected silhouette.
   Reject either control in standalone mask or edge mode with exit 2.
   CLI and recipe fields are `backing_mm`, `silhouette_thickness_mm`, `structure_height_mm`, and `detail_height_mm`.
   CLI flags use matching dashed names. `backing_mm=0` disables backing.
   CLI `--detail-mode raised|engraved` and recipe `detail_mode` select the detail operation.
   Default detail mode is `raised`. Default centering is false.
   CLI `--center` or `--no-center` and recipe `center` select one assembly transform.
   CLI `--export-scad` or `--no-export-scad` and recipe `export_scad` select source generation.
   Recipe `preset` selects the same preset as CLI `--preset`.
   A recipe preset cannot be cleared from the CLI. No `--no-preset` option exists.
   This limit is accepted while `relief-0.4` is the only preset.
   `--structure-width-mm` and `--detail-width-mm` set physical band widths before tracing.
   `backing_mm` is nonnegative. Zero disables backing.
   Without a preset, require positive heights for selected layers when `--export-scad` is set.
   Require `0 < detail_height_mm < silhouette_thickness_mm` for engraving.
   Resolve built-in defaults, preset, recipe, and CLI in that order.

Review decision 1 before S3.1. Review decisions 2 and 3 before S3.2.
Review decision 4 before S3.3. Review decision 5 before S3.4.
The new flags and recipe fields are proposed policy, not existing behavior.
S3.2 prepares optional masks and candidate preview helpers in-process.
S3.3 exposes `--layers`, role flags, and role recipe fields when layered SVG publication exists.
This order prevents a successful CLI run from silently omitting selected SVG layers.

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

Implementer: Theta. A different agent reviews Theta's implementation.

1. Add RED tests in `tests/test_vector_map_image.py` for all three methods.
   Cover EXIF orientation, matching external-mask dimensions, alpha absence, and explicit threshold polarity.
   Cover recipe alpha replaced by CLI threshold. Cover recipe alpha cleared by `--no-alpha`.
   Cover missing method, same-source conflicts, and empty silhouette with exit 1 before VTracer.
   Reject mask and threshold in mask and edge modes, for CLI flags and recipe fields.
   Assert exit 2 and no output when either unsupported method is selected.
   Test white-mask luminance and ignored mask alpha.
   Test no source-alpha warning with `--mask`, and a source-alpha warning with `--threshold`.
   Keep existing standalone mask and edge routes unchanged.
2. Run `python -m pytest tests/test_vector_map_image.py tests/test_vector_map_raster_cli.py -q`.
   Record tests that fail because image mode remains deferred.
3. Extend configuration and raster preparation with one immutable silhouette result.
   Apply existing include and exclude masks after selection.
   Derive the canvas once from the common processing size.
4. Repeat the focused command to GREEN. Check a real image conversion through VTracer.
5. Check `drift refs` for each edited file. Update bound prose before `drift link`.
6. Commit only S3.1 paths. Report the commit and RED/GREEN evidence in FP.

## S3.2 structure and detail proposals

Use the [S3.2 cold implementation plan](2026-10-09-vector-map-s3-2.md).
Keep the role request internal until S3.3 can publish every selected SVG layer.
Do not add CLI flags or recipe fields that the current CLI would silently ignore.
The S3.2 preview helper shows candidate masks without claiming vector render proof.
S3.3 adds CLI and recipe validation, layered artifacts, and rendered SVG panels.

## S3.3 aligned SVG and bundle

1. Add RED tests in `tests/test_vector_map_layer_export.py` for role CLI flags and recipe fields.
   Test `--layers` order, missing silhouette, duplicate roles, and standalone exit 2.
   Test map source and path rules, recipe-relative paths, CLI overrides, and no silent layer omission.
   Test a common root and `viewBox` for selected layers.
   Check an asymmetric source, a donut, and a nested island at known physical dimensions.
2. Add role CLI flags and recipe decoding with the reviewed decision 2 and 3 rules.
   Resolve requests into the S3.2 internal model. Include external maps and role constraints in input snapshots.
   Reject any selected role that cannot be published by this run.
3. Trace each nonempty selected layer with the pinned VTracer adapter.
   Normalize each layer with the S2.5 inspector. Retain the complete canvas.
4. Build combined SVG named groups with `ElementTree`.
   Treat separate normalized layer SVGs as the authoritative CAD inputs.
   Add preview tests for every selected rendered SVG and its prepared mask.
   Keep pixel-root rendering and overlay alignment at non-binary millimetre scales.
5. Extend `Bundle` to own all three fixed `<stem>.layers/<role>.svg` names in a layered run.
   Reject a symlink layer directory. Protect every source and external map from output aliases.
   Without `--overwrite`, refuse any existing owned role file.
   With `--overwrite`, remove unselected role files after invalidating the old manifest.
   Preserve unrelated files. Do not remove the layer directory.
6. Include each published artifact and input hash in the manifest.
   Report omitted optional layers. List only files from the current run.
   Keep the standalone debug bundle unchanged.
   For a layered debug bundle, save one fixed PNG per selected mask and a replay recipe.
   Own all three role PNG names. Remove unselected role PNGs on overwrite.
   Hash current debug files in the manifest. Preserve unrelated debug files.
7. Test collision refusal, `--overwrite`, staged failure, manifest-last publication, and repeated bundle hashes.
   Test a prior `structure.svg` followed by a run without structure.
   Check removal on overwrite and preservation of unrelated files.
   Check layered debug replay and fixed-name ownership before publication.
   Document that a later standalone run may leave an old `.layers/` directory.
   State that the current manifest identifies the current bundle.
8. Run `python -m pytest tests/test_vector_map_layer_export.py tests/test_vector_map_artifacts.py -q` to RED and GREEN.
9. Commit only S3.3 paths. Report geometry and publication evidence in FP.

## Q4 imported-layer Boolean decision

1. Add a focused proof using S2.5-normalized SVGs and real OpenSCAD 2021.01.
   Render imported structure intersected with the silhouette.
   Render detail minus structure. Use compound paths and a nested hole.
2. Include a border-touching region and asymmetric placement.
   Compare STL XY/Z bounds, surface components, and genus with expected geometry.
3. Add an almost-touching structure/detail boundary after independent VTracer traces.
   Check that intersection and difference leave no extra sliver components.
4. Record OpenSCAD version, exact command, output, and fixture paths in `STABL-szyudowd`.
   Keep Flatpak files under the home directory when that executable is used.
5. Accept 2D booleans only if every required case passes.
   If any case fails, record the failure and review an alternative assembly before S3.4.

## S3.4 SCAD assembly

1. Add RED tests in `tests/test_vector_map_scad.py` for relative imports and `center=false`.
   Require one assembly-level centering transform when requested.
   Cover `--export-scad` through CLI and recipe, including absent OpenSCAD.
2. Implement Q4's accepted Boolean strategy.
   Constrain structure and detail to the silhouette. Remove structure from detail again.
3. Parse and validate the four height controls and `detail_mode` from CLI and recipe.
   Expose backing `B`, silhouette `S`, structure `H`, and detail `D` as editable millimetre parameters.
   Check the spec's Z intervals for raised and engraved cases.
   Require `0 < D < S` for engraving and positive selected feature heights.
4. Keep engraving out of structure ridges.
   Document small support overlaps without changing the intended top heights.
5. Add an optional full-canvas rectangular backing.
   Warn when a disconnected silhouette without backing may form multiple solids.
6. Add `<stem>.scad` to fixed bundle ownership and manifest hashes.
   Add `artifacts.scad`. Record height, mode, backing, and centering in the manifest.
   Source generation must work without an OpenSCAD executable.
7. Run `python -m pytest tests/test_vector_map_scad.py tests/test_vector_map_artifacts.py -q` to RED and GREEN.
8. Commit only S3.4 paths. Report validated source and remaining CAD evidence in FP.

## S3.5 explicit relief preset

1. Add RED tests in `tests/test_vector_map_preset.py` for no implicit preset.
   Check `--preset relief-0.4` through CLI and recipe.
   Check every height, width, backing, detail mode, and centering override.
2. Set `S=1.2 mm`, structure width `0.8 mm`, `H=0.6 mm`, detail width `0.5 mm`, and `D=0.2 mm`.
   Keep backing disabled. Do not change S1.4 VTracer defaults.
3. Record all resolved values in the manifest and replay recipe.
   Document that printer suitability requires separate evidence.
4. Run `python -m pytest tests/test_vector_map_preset.py tests/test_vector_map_scad.py -q` to RED and GREEN.
5. Commit only S3.5 paths. Report override and provenance evidence in FP.

## S3.6 CAD acceptance

1. Run complete image workflows with the reviewed corpus.
   Include image mode with `--mask` as the external-mask case.
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

Decision 1 meets the stated S3.1 review condition. Human assignment still gates S3.1.
Review decisions 2 and 3 before S3.2 starts.
Review decision 4 before S3.3 starts. Review decision 5 before S3.4 starts.
Review Q4's render evidence before S3.4 starts.
Each human-popped implementation task uses inline RED/GREEN work and stops at ready for review.
Do not self-pop the next task or mark an issue done before its review cycle.
No waveplan runtime bundle is configured in this checkout. Do not invent schedule state.
