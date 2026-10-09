# S2.3 raster preparation

Date: 2026-10-06
Owner: Sigma
Issue: STABL-vjpnctjh
Status: Approved after review corrections on 2026-10-06. Corrections incorporated below.

Parent contract: [vector-map design](2026-09-06-vector-map-relief-design.md), sections 5 and 6.1.
Baseline: `9775e02`, merged S2.2.

## Scope and authority

Human selected S2.3 on 2026-10-06. FP issue now in progress.
No matching waveplan plan or SWIM artifact bundle exists in inspected workspace or environment.
No waveplan task claimed. No SWIM row completed.

Replace raster skeleton. Enable mask and edge preparation through existing CLI and recipe.
Retain existing VTracer adapter and SVG inspection boundary.
Manifest publication belongs to S2.6. Preview belongs to S2.7. Image mode belongs to S3.1.

S2.6 supplies optional input byte snapshots to raster preparation and captures recipe bytes during parsing.
Preparation decodes those exact bytes while preserving original paths in diagnostics.
Direct raster callers can continue reading files normally. Raster geometry and recipe precedence remain unchanged.

## Options considered

1. Recommend warning diagnostics from square coverage. Deterministic, conservative, uses existing NumPy and OpenCV dependencies.
2. Reject every flagged input. Prevents conversion of reviewed thin geometry and changes current conversion contract.
3. Measure exact local thickness with medial geometry. Adds substantial geometry machinery beyond wrapper scope.

Option 1 identifies possible small features. It does not certify minimum width or preserve vector topology.
False positives near curved or diagonal boundaries remain possible. Diagnostic text must state this limitation.

## Raster input

Decode PNG and JPEG with Pillow. Reject other decoded formats, regardless of filename suffix.
Accept only pixel modes `1`, `L`, `LA`, `P`, `RGB`, and `RGBA`.
Reject other modes with exit 2 and an explicit instruction to convert to supported 8-bit input.
Never clip 16-bit grayscale into an 8-bit luminance mask.
Retain Pillow decompression-bomb mapping to processing failure.
Apply EXIF orientation independently to source and external masks before dimension checks.
Load pixels before closing image handles.

Default material: luminance >= 128. RGB conversion ignores alpha by default.
Warn when source has alpha but alpha selection is disabled.
Warning: source has alpha, luminance used, pass --alpha to select it.
Explicit `--alpha` selects alpha >= 128, including palette transparency.
Reject `--alpha` when source has no alpha or transparency information.
Apply `--invert` after channel selection and thresholding.
Recommend PNG for masks and edges.

## Common canvas and scale

Record original source dimensions, oriented source dimensions, and processed dimensions.
All external masks must match oriented source dimensions before resize.
Reject mismatches. Never stretch an external mask to repair mismatches.

Threshold before resize. Resize every binary mask with nearest-neighbour sampling onto identical processing dimensions.
`--max-res` limits longest processing dimension. Never upscale.
For downscaling, round shorter dimension to nearest integer, with half values rounded upward. Minimum dimension: one pixel.

Keep one uniform `mm_per_px` scale. Selected physical dimension remains exact.
Derive scale from selected physical dimension divided by selected processed dimension.
Set other physical dimension to its processed pixel count multiplied by `mm_per_px`.
Integer resize rounding can change source aspect ratio slightly. Report original, oriented, and processed dimensions.
Short-side rounding error is at most half one processing pixel, except when minimum dimension clamps to one pixel.
Matching physical and pixel aspect ratios prevent SVG letterboxing and centering.
Derive canvas once. Copy immutable value into downstream stages.
Never crop, center, or recompute canvas from material bounds.

## Inclusion and exclusion

Add `--include-mask` and recipe `include_mask` as inclusion constraint for current layer.
Add `--exclude-mask` and recipe `exclude_mask` for exclusion constraint.
Current rule (S3.1, STABL-memwrtos): image mode accepts `--mask` and recipe `mask`. Mask and edge modes reject them with exit 2.
Superseded S2.3 rule: reserve `mask` for S3.1 and reject it with an S3.1 ownership message.
External constraints use luminance >= 128. Source alpha and inversion options do not change constraint polarity.

Before expansion, compute `(material AND include) AND NOT exclude`.
Missing inclusion permits whole canvas. Missing exclusion removes nothing.
Apply same constraints after expansion. Exclusion always wins.
Internal preparation accepts constraints per layer. S2.3 command still produces one layer.
Protect both external masks from destination aliasing, including symlinks and hard links.

## Edge expansion

Without `--line-width-mm`, preserve prepared edge thickness.
Reject `--line-width-mm` in mask mode.
Require positive finite width in edge mode.

Let `p = requested_width_mm / mm_per_px`.
Choose radius `r = max(0, ceil((p - 1) / 2 - 1e-9))`.
Subtract tolerance in radius units to suppress floating-point noise at exact odd widths.
Widths more than `2e-9` pixels above an odd boundary still expand to next odd width.
Nominal isolated-line width becomes `2*r + 1` pixels.
Rounding selects smallest odd width that reaches requested width within this tolerance. Width below one pixel remains one pixel.

Use centered square dilation kernel with side `2*r + 1`.
Square kernel gives explicit axis-aligned behavior. Diagonal strokes do not have exact constant physical width.
Existing thick bands expand further. Never erode or skeletonize them to reach nominal width.
Clip at common canvas boundary. Reapply inclusion and exclusion after dilation.

Report requested width beside nominal achieved width in millimetres: `(2*r + 1) * mm_per_px`.
Also report radius, kernel side, and expansion per side in millimetres.
Example: requested 3.1 pixels becomes five pixels. Conservative rounding can overshoot requested width substantially.
Report this before tracing on stderr. Expose same metadata through raster result for later manifest work.
Warn when requested width covers fewer than four processing pixels.
Reject nonfinite derived scales or unrepresentable kernel dimensions before allocation.

## Minimum-feature diagnostics

Run diagnostics after resize, expansion, and constraint reapplication.
Check material and background separately.
For each class, identify pixels that no complete axis-aligned 4x4 block of that class covers.
Compute coverage with binary erosion followed by dilation. Specify matching anchors to avoid shifts with even kernel size.
Treat every outside-canvas pixel as background in both checks.
Thin material at canvas edge warns. Thin exterior background margins do not count as gaps.
Wide clipped features remain covered by interior 4x4 blocks. Clipping alone produces no warning.
Thin border-connected channels still warn beyond three pixels of depth.

Emit one warning per affected class with uncovered pixel count and diagnostic method.
Message states possible features below four pixels and possible boundary false positives.
Background check includes enclosed holes and channels connected to border.
Uniform background produces no warning. Uniform material warns only when canvas itself is narrower than four pixels.

Diagnostics never mutate material. Diagnostics never change VTracer controls.
Warn by default. Add no strict rejection option in S2.3.
Include warnings in existing JSON `diagnostics` array and print warnings on stderr.
Keep existing JSON top-level keys and exit codes.
Successful conversion with warnings remains exit 0.
Keep both requested-width warning and material-feature warning when both conditions apply. Pin this behavior in regression test.

## Module contract

`vector_map_raster` owns decoding, orientation, thresholding, common resize, constraints, expansion, and diagnostic results.
Preparation returns binary material, immutable canvas, dimensions, expansion metadata, and warnings.
`vector_map_config` validates supported options and resolves recipe precedence.
`vector_map` coordinates preparation, tracing, output, and diagnostics.
`vector_map_svg` receives existing canvas value. SVG normalization changes remain S2.5 scope.

Invalid settings, missing alpha, dimension mismatches, and unsupported formats return exit 2.
Decode errors, I/O failures, and Pillow pixel-limit refusal return exit 1.
Failure emits one JSON result when requested. Failure publishes no SVG.

## RED/GREEN evidence

Use dedicated `stability-toys` conda environment. Confirm interpreter prefix and Python version before native VTracer calls.

Focused command:

```bash
python -m tests.run tests/test_vector_map_raster.py tests/test_vector_map_cli.py tests/test_vector_map_packaging.py tests/test_vector_map_vtracer.py -- -q
```

Write failing tests before implementation. Record RED failures caused by missing behavior.
After implementation, require focused GREEN and unchanged adapter acceptance.

Required cases:

- All eight EXIF orientations, with asymmetric source and independently oriented external masks.
- PNG, JPEG, unsupported format, malformed bytes, and Pillow pixel-limit refusal.
- Luminance and alpha threshold boundaries, palette transparency, inversion, and missing alpha.
- Dimension mismatch, no upscale, nearest-neighbour binary resize, and awkward aspect-ratio rounding.
- Inclusion/exclusion precedence before and after expansion, canvas margins, and border clipping.
- Isolated and thick edges, fractional widths, odd-width rounding, uniform scale, and requested versus achieved millimetres.
- One-, two-, three-, and four-pixel material bands and background gaps.
- Thin bridges, enclosed holes, border-connected channels, diagonals, and uniform canvases.
- All S1.4 hard cases, with unchanged raster bytes before and after diagnostic evaluation.
- Real CLI edge conversion, JSON warnings, recipe precedence, invalid settings, and all mask alias forms.
- Alpha ignored warning and intentional double warning for requested width below four pixels with remaining thin geometry.

Mutations must prove threshold, EXIF, exclusion order, rounding, background diagnostics, and alias protection assertions.
Run full split suite after focused GREEN. Report unrelated failures separately.
Run `git diff --check` and `drift check` before review handoff.

## Review boundary

FP review approved implementation after three required corrections and three smaller fixes. This version incorporates all six.
After implementation, commit with FP issue and exact next step. Assign commit in FP.
Stop ready for independent review. Do not mark S2.3 done or advance another task.
