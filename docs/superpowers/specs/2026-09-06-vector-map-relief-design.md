# st-vector-map: VTracer integration and layered relief design

Date: 2026-09-06

Status: Revised design draft. Reviewed; FP issues filed 2026-09-26 under
umbrella STABL-umtdoiaf (sprint and task issues plus four open-question
gates, per Section 11.5). The feature has no implementation.

Owner: Sigma

Delivery: Four integration sprints, tracked in FP under the umbrella issue.

Language: This document applies the structural rules in the ASD-STE100 skill.
The rewrite uses plain words and consistent technical terms.
It does not certify compliance with the official ASD dictionary.

## 1. Objective and reuse decision

Deliver `st-vector-map` as a Stability-Toys interface to [VTracer](https://github.com/visioncortex/vtracer).
Add the limited image preparation and OpenSCAD assembly necessary for printable reliefs.
VTracer controls conversion from raster images to vector geometry.
Stability-Toys controls workflow inputs, physical scale, layer selection, reproducible settings, and local artifacts.

The design does not depend on the subject in an image.
The requirements and defaults depend on these properties:

- The input representation.
- The geometry.
- The physical dimensions.
- The output operations.

The same processing contract applies to photographs, illustrations, symbols, technical line art, and abstract patterns.
Do not select algorithms, thresholds, layer meanings, or special behaviour from an object category, example image, or filename.
Keep adjustments for a specific subject in explicit operator recipes.
Do not make those adjustments global defaults.

This revision replaces the custom tracing design and its schedule of six sprints.
Do not implement these alternatives:

- A competing tracer.
- A curve fitter.
- A simplification engine.
- An engine that repairs polygons.
- A general library for vector Boolean operations.

Use VTracer as a dependency.
Do not copy its implementation.
Do not create a fork of its implementation.
Retain mask, Canny, and image inputs.
Retain SVG output.
Retain optional layered OpenSCAD reliefs.

```text
Image + silhouette mask             Existing mask or Canny map
          |                                      |
          +----> normalize orientation/scale <---+
                              |
                 prepare binary layer masks
                 (existing Canny + raster operations)
                              |
                   VTracer: one call per layer
                              |
                  SVG units + common canvas
                              |
                preview + SVG files + manifest
                              |
                 optional OpenSCAD assembly
                 (extrude / intersect / subtract)
                              |
                 render -> slice -> print trial
```

### 1.1 Technical terms

| Term | Meaning in this document |
|---|---|
| Operator | The person who supplies inputs and selects settings. |
| Wrapper | The Stability-Toys code that prepares inputs, calls VTracer, and supplies output artifacts. |
| Upstream | The VTracer project and its supplied software. |
| Adapter | The wrapper code that calls one supported VTracer interface. |
| Raster | An image that represents data as pixels. |
| Binary mask | A raster that identifies foreground and background regions. |
| Polarity | The relation between pixel values and foreground or background. |
| Canvas | The complete image area, including margins, in a common coordinate frame. |
| Topology | The arrangement of connected regions, separate regions, and holes. |
| Fitting | The VTracer operation that represents raster boundaries with vector paths. |
| Recipe | A file that records explicit settings and input paths. |
| Artifact | A file that the workflow produces. |
| Manifest | The JSON artifact that records settings, source information, output information, and hashes. |
| Bundle | The related artifacts from one workflow run. |
| Fixture | A test input with expected results. |
| Acceptance corpus | The varied set of images used to check the workflow. |
| Relief | A solid with raised or engraved features. |
| SCAD assembly | The OpenSCAD source that combines imported layers into a relief. |
| Preset | A named group of initial settings that the operator can override. |
| Extra | An optional dependency group in the Python package. |
| Pin | An exact dependency version selected for use. |
| RED/GREEN TDD | A process with a failing test before code changes and a passing test after those changes. |

## 2. Project fit

[`scripts/pyproject.toml`](../../../scripts/pyproject.toml) registers the existing helpers.
[`scripts/canny_map.py`](../../../scripts/canny_map.py) supplies the Canny operation.
Reuse that operation.
Do not introduce a second Canny implementation.
Follow the [existing helper design](2026-06-30-canny-control-map-script-design.md).
Follow the [CLI-first boundary](../../../project-forward-notes.md#cli-first-always).

Execution remains local and uses the CPU.
This feature requires none of these additions:

- A model download.
- A server endpoint.
- A ControlNet type.
- A GPU worker.
- A frontend.

SVG remains a local artifact.
It is not a new ControlNet attachment type.
Do not interpret existing depth maps as relief heights.

The helper package currently requires Torch globally.
When you add the vector extra, move the Torch requirement into the depth and pose extras.
Preserve `all`.
Preserve the documented installation paths.
Check the imports for each sibling command.
Check the packaging for each sibling command.

This change adjusts dependencies for the helpers.
It does not rewrite the server.

## 3. Upstream capability and version boundary

The inspected upstream README describes these capabilities:

- Conversion from raster images to SVG.
- Binary and colour processing.
- Multiple fitting modes.
- Python, Rust, and CLI interfaces.
- A 1.0 prerelease API.

These capabilities support the reuse decision.
They do not prove compatibility with every published Python wheel.
[Source: upstream README](https://github.com/visioncortex/vtracer#readme).

The inspected PyPI page lists `0.6.15` and the older `convert_image_to_svg_py` interface.
That interface includes binary conversion and fitting options.
Treat `vtracer==0.6.15` as the initial evaluation candidate.
It is not a pin with runtime proof.
Do not copy the repository's 1.0 `Config` examples into an adapter for that candidate.
[Source: Python package](https://pypi.org/project/vtracer/0.6.15/).

Sprint 1 checks these properties:

- The exact API of the installed package.
- Wheel availability for the project Python on macOS ARM64.
- Wheel availability for the project Python on Linux AMD64.
- SVG behaviour.

Record the accepted package version.
Record the results for each platform.
If the candidate fails a required case, record the failure.
Then propose a specific upstream release for evaluation.
Do not install a prerelease without an explicit decision.
Do not select `master` without an explicit decision.

Do not write a replacement tracer.
Repeat the relevant acceptance fixtures after a version change.

Prefer the Python binding because it matches the existing helpers.
Use a CLI adapter only if evaluation identifies a specific Python packaging problem that prevents integration.
Select one production interface.
Do not implement both interfaces without a demonstrated need.
SVG generation must not require the VTracer desktop app or a network service.

## 4. Ownership and limits

| Concern | Responsible component or evidence |
|---|---|
| Raster regions, tracing, fitting, and speckle filtering during tracing | VTracer. |
| Input decoding, orientation, and masks from alpha or thresholds | Pillow and limited wrapper code. |
| Canny candidates, dilation, and raster intersections or exclusions | Existing OpenCV helpers and array operations. |
| Physical canvas, layer names, parameter checks, and artifact metadata | The Stability-Toys wrapper. |
| SVG import as solids, extrusion, and solid intersection or difference | OpenSCAD. |
| Print feasibility | A recorded render, slicer review, and print trial. |

Upstream VTracer can convert a photograph directly into vector artwork.
For reliefs, use separate binary masks.
Colour clusters and SVG paint order do not identify the intended structural regions.
They do not establish Z heights.
These sprints do not add a second Stability-Toys interface for direct colour artwork conversion.

Canny pixels represent edge bands.
They do not represent a recovered silhouette or centrelines.
The wrapper can expand a band into material.
The band does not identify which side the operator intends to retain.
Automatic selection by image meaning remains deferred.

Layer names describe processing roles.
They do not name recognized objects.

| Layer | Role |
|---|---|
| `silhouette` | The support region that the operator defines. |
| `structure` | A selected mask of coarse features. |
| `detail` | A selected mask of fine features. |

A support region may contain multiple components and holes.
It need not describe one isolated object.
Selection of coarse and fine features is optional.
Supplied masks can define the layer roles without inference about image contents.

## 5. Inputs and physical coordinates

Require `--input-kind mask|edges|image` explicitly.

| Kind | Input meaning | Processing route |
|---|---|---|
| `mask` | A foreground region with any intended holes. | Prepare the binary mask. Use VTracer to produce the silhouette SVG. |
| `edges` | An existing Canny map or line map. | Prepare the bands. Widen them if requested. Use VTracer to produce the detail SVG. |
| `image` | An RGB/RGBA source with a selected silhouette method. | Prepare the silhouette and selected detail masks. Call VTracer for each layer. |

Accept PNG and JPEG.
Recommend lossless PNG for masks and edges.
Normalize EXIF orientation first.
In mask mode, the default foreground has luminance >= 128.
Permit explicit inversion.
Permit explicit alpha selection.

Image mode requires one of these silhouette methods:

- `--mask`.
- Explicit alpha selection.
- Explicit thresholding for a simple background.

Reject a fully opaque photograph if the operator supplies no silhouette method.
Give an instruction that explains the required input.
Do not infer learned foreground segmentation.

Require exactly one of `--width-mm` and `--height-mm`.
The selected dimension sizes the complete oriented source canvas, including margins.
Preserve the aspect ratio.
External masks and edge maps must match the oriented source dimensions before the common resize.
Reject dimension mismatches.
Do not stretch mismatched inputs.

`--max-res` limits the processing resolution without increasing the input resolution.
Use nearest-neighbour resampling for binary masks.
Record these dimensions:

- The original dimensions.
- The oriented dimensions.
- The processed dimensions.

All layer SVGs retain the same complete canvas and origin.
Do not crop individual layers.
Do not center individual layers.
Convert processing pixels to physical units once.
Record the scale.
Distances use millimetres unless an option explicitly specifies upstream pixel, angle, or precision units.

## 6. End-to-end processing

### 6.1 Prepare layer masks

Mask mode produces a silhouette.
Edge mode produces detail without a base.
Image mode first establishes a silhouette.
It then reuses Canny with separate coarse and fine settings to propose structure and detail.
These proposals use visual heuristics.
A recipe can instead specify external structure and detail maps.

Support inclusion and exclusion masks for each layer.
Exclusion takes precedence.
Apply these constraints again after band expansion.
Intersect the structure and detail masks with the silhouette.
Remove structure coverage from the fine-detail mask.

Do these operations on the common binary raster before VTracer.
This method avoids a separate implementation of Boolean operations for SVG paths.

Gap closing requires an explicit selection.
The default disables gap closing.
Hole filling and automatic bridges remain deferred.
Expose the VTracer speckle control with its checked upstream meaning.
Do not describe that control as a physical area measurement.

For edge inputs, `--line-width-mm` specifies a nominal width for an isolated input line that is one pixel wide.
Derive raster dilation from the physical scale.
Document the rounding.
Document the effect of input thickness.
This operation does not produce exact constant-width strokes.
Do not silently reduce the thickness of existing wider bands.

Report the effective band expansion.
Warn when a requested width covers fewer than four processing pixels.
Show intermediate masks in a preview for operator review.

### 6.2 Invoke VTracer

Send each final binary layer through the adapter for the selected version.
Use a fixture to check foreground polarity.
The wrapper's white foreground must become foreground material in the selected VTracer binary mode.
Reject output with an unintended background that covers the complete canvas.

VTracer controls contour hierarchy and fitting.
Preserve the generated paths, compound paths, winding, and transforms.
Do not flatten curves into a custom polygon engine.
Start relief evaluation with polygon fitting.
Permit spline fitting only after it passes the same hole, alignment, and OpenSCAD cases.

Expose only controls that pass checks for the pinned API.
The candidate options are:

- `mode`.
- `filter_speckle`.
- `corner_threshold`.
- `length_threshold`.
- `splice_threshold`.
- `path_precision`.

Store these options in a checked `vtracer` object in the recipe.
Record all resolved values.
These options control upstream fitting.
They do not guarantee a maximum physical error at a boundary.

The design no longer promises `--simplify-mm`.
A future upstream tolerance may map to millimetres only after checks establish its meaning and measured error.
The wrapper contains no simplification engine or automatic retry engine.

### 6.3 Normalize and inspect SVG

Set the root dimensions in millimetres.
Preserve the matching pixel viewBox.
Preserve all coordinate transforms for the layers.
Use an existing parser to check these properties:

- The XML structure.
- The required canvas attributes.
- Nonempty foreground paths.
- Closed filled geometry.
- Finite coordinates.
- Supported elements.

Reject unexpected embedded images, external resources, masks, or clipping features.
Do not parse SVG with regular expressions.

Use configurable limits for path counts, file sizes, and path-command counts.
Polygon vertex counts are not a reliable general measure for paths with curves.
If output exceeds a limit, fail with measured counts and suggested upstream settings.
Do not silently discard shapes.
Do not silently change fitting options.

The wrapper does not promise to detect or repair all self-intersections.
Check compatibility with these methods:

- Check fixture topology.
- Compare rendered output.
- Run an actual OpenSCAD render.

Report a failure as incompatible upstream output or an incompatible import.
Do not repair the failure silently.

### 6.4 Export aligned artifacts

Write these artifacts for destination `<stem>.svg`:

| Artifact | Condition and purpose |
|---|---|
| `<stem>.svg` | Write the requested SVG. |
| `<stem>.vector.json` | Write the manifest. |
| `<stem>.layers/silhouette.svg` | Write a selected, nonempty silhouette layer during a layered run. |
| `<stem>.layers/structure.svg` | Write a selected, nonempty structure layer during a layered run. |
| `<stem>.layers/detail.svg` | Write a selected, nonempty detail layer during a layered run. |

A combined SVG contains named groups on the common canvas.
It supports viewing and editing.
The separate layers are the authoritative SCAD inputs.

`--preview` produces `<stem>.preview.png` with an existing SVG renderer.
Select the renderer during Sprint 2 packaging checks.
Pin its version during those checks.
Show prepared masks and rendered SVG together so the operator can see changes from fitting.
Also show an overlay on the source with a legend.
Do not describe a mask-only preview as proof of the final vector output.

An empty silhouette is an error.
Report an empty optional detail layer.
Omit that layer from exports.
Omit that layer from generated imports.

The wrapper can save edge-only output as SVG.
SCAD relief export requires a silhouette from the layered workflow.

### 6.5 Assemble the relief with OpenSCAD

`--export-scad` writes `<stem>.scad` with relative layer imports and editable height parameters.
It generates source only.
Rendering and STL export require explicit subsequent OpenSCAD operations.
SVG and SCAD source generation do not require an installed OpenSCAD executable.

Import every layer with `center=false`.
Apply requested centering once to the complete assembly.
Use OpenSCAD 2D intersection to constrain structure and detail to the imported silhouette.
Use OpenSCAD difference to remove structure from fine detail again.

These operations enforce the final CAD boundaries even if independent fitting changes raster boundaries.
The separate SVG exports retain upstream curves.
Exact clipping after fitting occurs in the SCAD assembly.

The height parameters have these meanings:

| Parameter | Meaning |
|---|---|
| B | The backing thickness, with a default of zero. |
| S | The silhouette thickness. |
| H | The structure height. |
| D | The detail height or engraving depth. |

The layers use these Z intervals:

| Operation | Z interval |
|---|---|
| Silhouette material | [B,B+S] |
| Raised structure | [B+S,B+S+H] |
| Raised detail | [B+S,B+S+D] |
| Engraving removal | [B+S-D,B+S] |

Engraving removes material only from the exposed silhouette.
Require 0 < D < S for engraving.
Detail does not cut structure ridges.
Require positive heights for selected features.

Use small overlaps at support surfaces without changing the intended top heights.
Document those overlaps.
An optional rectangular backing covers the complete canvas and joins separate islands.
Report multiple solids when there is no backing.
Do not add bridges implicitly.

The OpenSCAD SVG importer treats open and closed shapes differently.
It ignores some visual SVG features.
A browser preview alone does not prove compatibility.
[Source: OpenSCAD SVG import](https://files.openscad.org/documentation/manual/SVG_Import.html).

## 7. Presets and CLI

`relief-0.4` proposes these initial settings:

| Setting | Value |
|---|---|
| Silhouette thickness | 1.2 mm |
| Nominal structure width | 0.8 mm |
| Structure height | 0.6 mm |
| Nominal detail width | 0.5 mm |
| Detail height or depth | 0.2 mm |
| Backing | Disabled |

The operator can change these values.
They do not guarantee suitability for every 0.4 mm nozzle, extrusion width, material, or slicer configuration.
Set the VTracer fitting defaults from Sprint 1 evidence.

This preset describes fabrication settings only.
It requires an explicit selection.
Its values must remain overridable.
Choose fitting and Canny defaults from the varied acceptance corpus in Section 10.
Do not choose defaults from one demonstration image.

These proposed commands have no implementation:

```bash
st-canny-map source.png edges.png

st-vector-map edges.png detail.svg \
  --input-kind edges --width-mm 100 --line-width-mm 0.6 --preview

st-vector-map mask.png silhouette.svg \
  --input-kind mask --width-mm 100

st-vector-map source.png result.svg \
  --input-kind image --mask mask.png \
  --layers silhouette,structure,detail --width-mm 100 \
  --preset relief-0.4 --export-scad --preview
```

`--recipe recipe.json` supplies these values for each layer:

- Input paths.
- Masks.
- Raster settings.
- Heights.
- Supported VTracer controls.

Use `schema_version: 1`.
Reject unknown fields.
Resolve recipe paths relative to the recipe directory.
Apply settings in this order, with later values taking precedence:

1. Built-in defaults.
2. The preset.
3. The recipe.
4. Explicit CLI options.

Reject unsupported upstream controls.
Do not ignore them.
Do not pass arbitrary shell commands through the wrapper.

Send progress and diagnostics to stderr.
`--json` emits one final result object with these fields:

- Schema version.
- Status.
- Artifact paths.
- Counts.
- Diagnostics.

Do not change `st gen --json`.
Use these exit codes:

| Code | Meaning |
|---|---|
| 0 | The wrapper completed SVG conversion and published the artifacts. |
| 2 | Arguments or configuration are invalid. |
| 1 | Processing or I/O failed. |

Successful conversion does not prove a successful CAD render or physical print.
Record conversion, CAD render, and physical print evidence separately.

## 8. Provenance and publication

The manifest records these values:

- The VTracer version and wrapper version.
- Input hashes.
- Dimensions and scale.
- Resolved preparation options and upstream options.
- Layer identities and heights.
- Artifact hashes.
- Complexity counts.
- Warnings.

Use a schema for vector artifacts instead of `controlnet_map`.
Exclude timestamps from deterministic artifacts.
Check repeatability with the same inputs, platform, settings, and dependency versions.
Do not promise identical bytes across different native library versions.

Check all output collisions before processing.
Refuse overwrite by default.
`--overwrite` permits replacement of this bundle only.
Preserve unrelated files.

Publish a bundle in this sequence:

1. Stage the files.
2. Invalidate an old manifest before you replace its files.
3. Publish the output files.
4. Publish the manifest last.

A bundle is complete only when every manifest hash matches.
Multiple file renames do not form an atomic transaction.

Make the prepared raster masks and resolved recipe available in a debug bundle to reproduce upstream failures.
Report native errors as concise failures with version and layer information.
Do not publish partial output as a successful conversion.

## 9. Implementation footprint

Use `scripts/vector_map.py` as a thin CLI facade.
Add small modules only where necessary for these responsibilities:

- Configuration.
- Raster preparation.
- The VTracer adapter.
- Artifact handling.
- SCAD template generation.

Retain SVG as the format for vector interchange.
Do not introduce an internal polygon model.
Do not introduce a mandatory Shapely dependency.

The core vector extra needs VTracer and the existing image dependencies.
OpenCV supplies Canny and raster preparation.
The preview renderer may use a separate extra if its native dependencies substantially increase installation size.
Document that choice during Sprint 2.

Core conversion must not load Torch, server modules, or model assets.
Import optional dependencies only when necessary.
Give clear errors for missing dependencies.

## 10. Verification contract

Test the adapter.
Test the workflow.
Run the real upstream converter.
Do not reimplement upstream algorithms in tests.
Do not accept mocks as tracing proof.

Follow RED/GREEN TDD for code tasks.
Use the dedicated `stability-toys` environment.
Run tests with `python -m pytest`.

| Proof | Required cases |
|---|---|
| Upstream compatibility | Check the exact API and version, imports, platform wheels, binary polarity, and supported options. |
| SVG topology | Check a donut, a nested island, separate components, and a region that touches the border. Check actual rendered holes. |
| Canny preparation | Check both polarities, thin and thick edges, dilation rounding, and disabled and enabled gap closing. |
| Coordinates | Check EXIF orientation, mask-size rejection, the common resize, and asymmetric layers. Check a rectangle with known physical dimensions. |
| Composition | Check mask precedence, raster clipping after expansion, and final SCAD intersections. |
| Artifact behaviour | Check empty layers, malformed upstream output, count limits, overwrite refusal, and manifest hashes. |
| Repeatability | Repeat real VTracer conversions with fixed inputs, configuration, and environment. |
| CAD | Run actual OpenSCAD renders. Check XY/Z bounds, holes, raised detail, engraved detail, backing, and connectivity. |
| Packaging | Check conversion without Torch, retained sibling entry points, and diagnostics for preview dependencies. |

Use two acceptance sets.
Synthetic geometric fixtures establish measurable contracts for topology, scale, alignment, and extrusion.
A varied acceptance corpus exercises the same workflow with these image classes:

- Flat graphics and symbols.
- Technical line art.
- Textured photographs.
- Abstract or repeated patterns.

Include these properties across the corpus:

- Sparse and dense detail.
- High and low contrast.
- Holes.
- Disconnected regions.

Supply reviewed masks where necessary.
Acceptance does not require automatic subject recognition or automatic segmentation.

Check each case against its supplied masks, resolved settings, and expected geometry.
Record results for each case so a combined score cannot hide a failure.
Assess defaults across the corpus.
Identify recipes for specific images as overrides.
Acceptance measures geometry and workflow behaviour.
It does not measure resemblance to a preferred subject or artistic composition.

Use an existing renderer to rasterize the SVG at a documented resolution.
Compare its topology and boundary behaviour with the fixture masks.
Record error and complexity together.
This evidence measures acceptance for the tested cases.
It does not guarantee a universal geometric error limit.

Establish acceptable fixture tolerances in Sprint 1 before you choose fitting defaults.
Do not relax those tolerances only to make a candidate pass.

Local OpenSCAD integration checks may skip when the executable is absent.
Relief release acceptance requires an actual render.
Record these values:

- The executable version.
- The command.
- The output.
- The expected mesh bounds and components.

Do not claim success from generated SCAD syntax alone.
Do not claim success from a browser view alone.

Final acceptance uses the reviewed corpus and the resolved recipe for each case.
Keep this evidence for applicable relief cases:

- Previews.
- SVGs.
- Render evidence.
- Counts.
- Timings.
- Slicer observations.

An illustrative example cannot replace final acceptance.
A physical sample is separate evidence.
Mark print validation as pending until the sample exists.
Advanced detection of minimum features and automatic geometry repair remain deferred.
Do not rebuild those functions around VTracer.

## 11. Revised sprint boundaries

These sprints define proposed FP issue boundaries.
They do not claim work or create a waveplan schedule.

| Sprint | Dependency | Result |
|---|---|---|
| 1. VTracer compatibility and relief proof | Design review. | An evaluated upstream integration with a selected version and acceptance evidence. |
| 2. Standalone st-vector-map wrapper | Sprint 1. | The standalone wrapper for masks and edges. |
| 3. Layered image workflow and relief export | Sprint 2. | The layered image workflow and SCAD assembly. |
| 4. Workflow acceptance and release guidance | Sprint 3. | Acceptance evidence and operator documentation. |

### 11.1 Sprint 1: VTracer compatibility and relief proof

Complete these items:

1. Evaluate one upstream integration.
2. Pin the accepted version.
3. Check options, polarity, and topology.
4. Establish the varied acceptance corpus.
5. Select fixture tolerances.
6. Select fitting defaults.

Require this exit evidence:

- Actual conversions.
- Platform and package evidence.
- Corpus results.
- Donut, nested-hole, and asymmetric fixtures.
- An actual basic OpenSCAD import and render.

### 11.2 Sprint 2: Standalone st-vector-map wrapper

Deliver these items:

- The entry point.
- Vector packaging.
- Mask and edge modes.
- Physical scale.
- Raster band preparation.
- Supported VTracer controls in recipes.
- Preview output.
- The manifest.
- Output collision handling.

Require actual VTracer conversion of existing Canny output.
Require passing checks for these properties:

- Dimensions.
- Repeatability.
- CLI failures.
- Artifact hashes.
- Packaging for sibling commands.

### 11.3 Sprint 3: Layered image workflow and relief export

Deliver these items:

- Silhouette methods.
- Reuse of Canny candidates.
- Selection for each layer.
- Aligned SVGs.
- The SCAD assembly.
- The relief preset.

Require conversion through the complete image and mask workflow.
Require actual renders with raised features, engraved features, and backing.
Check XY/Z bounds.
Check connectivity.

### 11.4 Sprint 4: Workflow acceptance and release guidance

Deliver these items:

- Reviewed corpus bundles.
- Diagnostics for complexity and error.
- JSON results and evidence states.
- Regression checks.
- Operator documentation.

Require this exit evidence:

- Evidence for each corpus case.
- An actual render and slicer review.
- Installation checks.
- Drift checks.
- Reproducible recipes.
- A physical print result or an explicit pending status.

### 11.5 Delivery and task authority

Sprint 2 is the first standalone release.
Sprint 3 supplies the complete relief workflow.
Include tests and documentation in each sprint.
These sprints replace the previous custom-tracer schedule.
They do not add work to that schedule.
Estimate sprint duration after you divide the work into atomic tasks.

After document review, create one umbrella FP issue with a link to this document.
Create one child issue for each sprint.
Use the dependencies in the sprint table.

Done 2026-09-26: umbrella STABL-umtdoiaf with one child per sprint, atomic
tasks under each sprint, and four open-question issues (VTracer version and
interface, SVG renderer, torch-extra impact check, OpenSCAD 2D boolean
reliability). The task and question IDs are recorded in a comment on the
umbrella issue.
Include these items in each issue:

- Scope.
- Exclusions.
- Relevant document sections.
- Concrete exit evidence.

Divide sprints into bounded implementation tasks before execution.
Give each task RED/GREEN commands.

A task that the human pops in waveplan remains the authority for implementation.
Keep FP current.
Stop when the task is ready for review.
This document does not claim work or advance waveplan.
It does not authorize implementation before task dependencies permit it.

## 12. Deferred work and review status

Defer these features:

- Learned segmentation.
- Semantic layer recognition.
- Centreline tracing.
- Automatic bridges and hole repair.
- Repair of arbitrary vector geometry.
- Continuous heightfields.
- Automatic mapping from colour to height.
- Internal STL generation.
- Batch processing.
- Server integration.
- Interactive editing.

Evaluate VTracer upgrades as dependency changes.
Require explicit compatibility evidence.

This revision uses evidence from upstream documentation.
It does not include a trial of an installed VTracer package.
Sprint 1 supplies that runtime proof.
This design edit includes no application code, package installation, or FP sprint creation.

## 13. Illustrative application: botanical relief (non-normative)

The original flower workflow is one possible use of the generic pipeline.
An operator could supply these inputs:

- A flower photograph.
- A reviewed support mask.
- Optional masks that select boundaries or texture to raise or engrave.

Those selections are recipe inputs.
The tool does not recognize botanical features.

```bash
st-vector-map flower.png flower.svg \
  --input-kind image --mask flower-mask.png \
  --layers silhouette,structure,detail --width-mm 100 \
  --preset relief-0.4 --export-scad --preview
```

This example explains the original application only.
It defines none of these requirements:

- A required image.
- A tuning target.
- A feature detector.
- A global default.
- An acceptance criterion.

Sections 1–12 govern development and sprint completion.
