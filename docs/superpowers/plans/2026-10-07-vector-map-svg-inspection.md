# S2.5 SVG normalization and inspection plan

Issue: STABL-npoznayt
Planner: Sigma
Executor: Theta
Baseline: `2dc6764` on main, after PR #85.
Status: Review condition addressed. Human verdict: approve after degenerate-subpath fix. Planning only. No implementation started.

Goal: Reject incompatible traced SVG before publication and preserve physical canvas during normalization.
Parent contract: [vector-map design](../specs/2026-09-06-vector-map-relief-design.md), section 6.3.
Process: Theta executes inline with Superpowers executing-plans and TDD. No subagent development.

## Authority and boundaries

Human assigned planning to Sigma and execution to Theta.
FP issue remains todo until Theta claims execution. Human assigned execution to Theta.
Declared dependency STABL-lohtpfiy is complete. S2.2 and S2.3 are also merged.
No matching waveplan plan or SWIM bundle exists in inspected workspace.
Do not invent or advance waveplan state. Record direct issue authority in FP when execution starts.

Keep VTracer 0.6.15, polygon mode, polarity inversion, and all upstream arguments unchanged.
Do not enable splines or introduce a tracer, fitter, polygon repair engine, or vector Boolean engine.
Do not add manifests, previews, layered image mode, MPO support, or raster cleanup.
S2.6 owns bundle publication. S2.7 owns preview output.

## Current seams and evidence

- `scripts/vector_map_svg.py`: `size_svg`, `count_paths`, and six-decimal `_mm` formatting.
- `scripts/vector_map_vtracer.py`: raw tracing, full-canvas guard, `_translate`, and `_subpaths`.
- `scripts/vector_map.py`: independently counts paths and sizes SVG before `_publish`.
- `scripts/vector_map_config.py`: frozen settings, uniform canvas, flat recipe fields, and precedence.
- `tests/test_vector_map_openscad.py`: real render harness and one production sizing test.
- `tests/fixtures/vector_map/corpus/metrics.py`: existing rendering and locked acceptance metrics.

Sigma traced all eight corpus masks through installed adapter during planning.
Observed default output: SVG root, path elements, black fill, uppercase M/L/Z tokens, and path-level translate transforms.

| Case | Raw UTF-8 bytes | Paths | M/L/Z commands |
|---|---:|---:|---:|
| badge | 1,606 | 6 | 156 |
| bracket | 1,483 | 6 | 147 |
| brick | 5,518 | 1 | 703 |
| coins | 3,695 | 24 | 341 |
| gravel | 28,649 | 209 | 2,808 |
| horse | 1,406 | 1 | 132 |
| pcb | 8,342 | 30 | 1,082 |
| truchet | 19,367 | 55 | 2,354 |

These measurements establish corpus compatibility only. They do not establish thresholded-photo compatibility or production performance limits.

Review probe used installed VTracer 0.6.15 and NumPy default_rng(0), with random((64,64)) > 0.5.
At speckle 0, noise produced 258 degenerate subpaths among 388 subpaths.
At default speckle 4, noise produced 56 among 98, including 19 paths with both polygon and degenerate subpaths.
One isolated pixel at speckle 0 produced `M0,0 Z `. At speckle 4, it produced no paths.
Counts identify subpaths with fewer than three distinct vertices. These probes confirm review findings, not reviewer-specific random counts.
Use installed package metadata for provenance. Generator comments can report 0.6.12 while installed package is 0.6.15.

## Proposed design decisions

### Parser choice

Recommend ElementTree plus strict validation of existing adapter polygon dialect.
Extract and tighten `_subpaths` and `_translate` into shared SVG module. Do not create competing path parsers.
This reuses existing XML parser and current adapter grammar without another dependency.

Alternative: add a general path library such as svgelements.
That supports broader geometry but needs dependency qualification and strict rejection tests for permissive parsing.
Current adapter blocks curves already, so broader grammar has no required caller in S2.5.

Use ElementTree for document structure. Never use regex to parse XML, path data, or transforms.
Token checks below validate one explicitly supported upstream dialect. They do not implement general SVG syntax.
Do not copy regex probes from topology or corpus tests into production.

### Supported document contract

Accept exactly one SVG root in namespace `http://www.w3.org/2000/svg`.
Permit SVG `g` and `path` descendants. Paths must be leaves. Reject nested SVG roots. Require whitespace-only `.text` and `.tail` on every element.
Permit XML declaration and comments. Ignore comments during semantic comparison.
Reject DTD declarations and processing instructions through XML parser callbacks, before their contents can affect interpretation.

Use explicit attribute allowlists:

| Element | Accepted attributes |
|---|---|
| svg | version, width, height, viewBox |
| g | id, transform, fill, fill-rule |
| path | id, d, transform, fill, fill-rule |

Namespace declarations are XML syntax, not ordinary attributes.
Version may be absent or `1.1`. Reject other versions as unsupported output.
Reject every unknown attribute, including namespaced resource references and event handlers.
This excludes images, external resources, use/defs, scripts, masks, clipping, filters, styles, opacity, strokes, and hidden shapes.
Do not silently remove unsupported content.

Default fill is black. Permit explicit `black`, `#000`, or `#000000`, case-insensitively.
Permit absent fill-rule or `nonzero`. Reject `none`, other paints, and other fill rules.
Apply inherited group attributes when checking effective path state. Keep original attribute strings during serialization.

### Polygon paths and transforms

Supported path form: whitespace-separated uppercase `M<x>,<y>`, `L<x>,<y>`, and `Z` tokens.
Each subpath starts with M, permits zero or more L commands, and ends with explicit Z.
Reject a new M before prior Z, L before M, unmatched Z, trailing coordinates, or any unconsumed text.
Reject empty path data, implicit closure, relative commands, omitted command letters, and curves as unsupported dialect.
Do not normalize unsupported syntax into accepted syntax silently.

Each coordinate must contain only ASCII decimal-number characters and parse as a finite float.
Reject underscores, NaN, Infinity, overflow, incomplete exponents, missing coordinates, and extra comma fields.
Support signs, decimal fractions, and decimal exponents within this numeric rule.

Accept closed subpaths with one or two distinct vertices, including `M0,0 Z`.
Count these as degenerate_subpaths. Preserve their path data and position without rewriting or dropping them.
Require each path element to contain at least one polygon subpath with three distinct non-collinear vertices.
Reject paths containing only degenerate subpaths, even when another path contains valid polygons.
Reject subpaths with three or more distinct vertices when all vertices are collinear.
On successful CLI normalization, emit one warning per layer with degenerate_subpaths count when nonzero.
Use existing text and JSON diagnostic channels. Two inspections must not duplicate this warning.
Do not suggest filter_speckle as guaranteed repair. Degenerate holes persist at default speckle 4.

Degenerate-only rejection diagnostic (human approval, 2026-10-07):

- State the count of failing paths. Say "VTracer returned paths containing only points or lines."
- Do not claim a cause.
- Offer two options: higher --max-res, or higher filter_speckle.
- Say higher --max-res can help only when processing resolution can increase.
- Say filter_speckle removes small islands on purpose.
- Pin this message with one RED test.
- USAGE.md gives measured examples, not guarantees. On noise probes, speckle 4 cleared these paths. Speckle 2 did not. Speckle 1 matched speckle 0.
Do not reject geometry solely because total signed area is zero: holes and self-intersections can cancel signed area.
This check establishes drawable polygon structure, not positive final Boolean area or absence of self-intersections.

Supported transform: absent, or one `translate(x)` / `translate(x,y)` with finite numeric arguments.
Reject transform lists and other transform types. Pinned VTracer emits only translate.
Preserve every accepted transform string and its element position. Never flatten, recenter, or rewrite path data.
Accumulate group translations only in inspection metadata and full-canvas checks.
Reject nonfinite accumulated translations and translated points.

Count each explicit M, L, and Z as one path command. Count all subpaths.
Do not use vertex count as general SVG complexity. Curve support remains blocked by existing adapter policy.

### Canvas and normalization

Inspector accepts raw VTracer dimensions: positive finite unitless pixel values or equivalent explicit px values.
Both dimensions must equal expected processing dimensions exactly.
Optional existing viewBox must equal `0 0 W H`. Reject displaced origins, wrong extents, and malformed lists.
Do not replace a mismatched viewBox silently.

Normalizer accepts raw traced SVG and existing immutable Canvas value.
Check positive finite dimensions and uniform scale. Do not derive a new canvas from geometry.
Set width and height to supplied physical dimensions in mm.
Set viewBox to `0 0 W H` using processing pixels.
Change no geometry, fill, transform, group order, or path order.
Raw adapter result remains byte-for-byte unchanged. Normalized XML need only preserve semantic structure.

Replace six-decimal rounding with plain decimal serialization of `Decimal(str(value))`.
Keep existing whole-number forms such as `100mm`. Never serialize positive size as zero because of rounding.
Do not use scientific notation in physical length attributes.
Check both physical dimensions against processing pixels multiplied by Canvas.mm_per_px.
Use math.isclose with rel_tol=1e-12 and abs_tol=0.0 after positive finite checks.

Normalization is an upstream-output boundary, not a general external-SVG import command.
Already-normalized mm input is outside this entry point. Do not add idempotent external import scope.

### Limits and error behavior

Proposed defaults are wrapper policy, not measured VTracer or OpenSCAD limits:

| CLI flag / recipe key | Default | Measurement |
|---|---:|---|
| --max-svg-bytes / max_svg_bytes | 20,971,520 | UTF-8 bytes, checked independently for raw and normalized SVG |
| --max-paths / max_paths | 10,000 | Total path elements per layer |
| --max-path-commands / max_path_commands | 1,000,000 | Total supported commands per layer, including M and Z |

Accept positive integers only. Reject zero, negatives, booleans, floats, and null recipe values with exit 2.
No unlimited sentinel. Higher valid limits require explicit operator configuration.
Use existing defaults → preset → recipe → explicit CLI precedence.
Limits belong to wrapper settings. Never pass them as VTracer options.

Check raw byte limit before any XML or polygon parsing in adapter.
Apply path and command limits before storing unbounded derived geometry.
When stopping early, report observed count as a lower bound, not an exact total.
Example: `path count at least 101 exceeds limit 100`.
After serialization, check normalized UTF-8 byte count before publication.
Limits cannot prevent VTracer itself from allocating its output. State this boundary accurately.

Limit failures include measured quantity, configured limit, and actionable suggestions.
Suggest lower --max-res or explicit recipe vtracer.filter_speckle adjustment within existing 0..128 range.
Explain these choices can alter geometry. Never change them automatically or retry with altered settings.

Malformed or incompatible upstream output returns exit 1 and one failed JSON result.
Invalid limit configuration raises ConfigError and returns exit 2 with one invalid JSON result.
Valid configured limits exceeded by SVG raise SvgInspectionError and return exit 1, not exit 2.
Preserve existing top-level JSON keys and layers/paths counts. Put detailed failure counts in diagnostic message.
No failure publishes SVG or replaces existing destination, including with --overwrite.

Adapter still permits zero paths for empty or speckle-filtered material.
CLI normalization rejects zero paths as empty output. Empty path data and paths containing only degenerate subpaths remain invalid.

## Module interfaces and dependency direction

Keep shared SVG types and policy in `scripts/vector_map_svg.py`.
This module must not import configuration or adapter at runtime. TYPE_CHECKING may name Canvas.
Configuration and adapter may import SVG policy, avoiding a circular dependency.

Use these public interfaces:

```text
SvgLimits(max_svg_bytes=20971520, max_paths=10000, max_path_commands=1000000)
SvgMetrics(raw_bytes, normalized_bytes, paths, commands, subpaths, degenerate_subpaths)
SvgInspectionError(ValueError)
inspect_svg(svg, *, width_px, height_px, limits=None, allow_empty=False) -> inspected document
normalize_svg(svg, canvas, *, limits=None) -> normalized SVG plus SvgMetrics
trace_layer(layer, options=None, *, svg_limits=None) -> existing TraceResult
```

Inspected document contains parsed root and validated subpaths with accumulated translations.
Expose inspection counts through its metrics member. Set normalized_bytes to None until normalization completes.
Keep parsed root internal. Preserve immutable limit and metrics values.
Normalizer may reparse raw text after adapter inspection. Avoid adding shared mutable parser state to TraceResult.
Keep `size_svg` as thin compatibility facade over normalizer for existing tests and OpenSCAD callers.
Remove CLI use of `count_paths`. Keep helper only if a remaining caller requires it, with clearly documented scope.

Move full-canvas guard to validated inspection data, preserving existing border-majority semantics.
Do not call old ElementTree or polygon parser before byte-budget and XML policy checks.
Two complete inspections are acceptable here. Correct bounds and unchanged raw bytes matter more than avoiding small repeated parse.

## Execution steps for Theta

Keep all steps within STABL-npoznayt. These are implementation steps, not new FP issues or waveplan tasks.
Update state table and FP comments after each RED/GREEN milestone.

| Step | State | Main files |
|---|---|---|
| 1. Resume and baseline | Pending | FP context, git, dedicated environment |
| 2. Inspector RED/GREEN | Pending | vector_map_svg.py, new test_vector_map_svg.py |
| 3. Adapter boundary RED/GREEN | Pending | vector_map_vtracer.py, test_vector_map_vtracer.py |
| 4. Normalization RED/GREEN | Pending | vector_map_svg.py, test_vector_map_svg.py |
| 5. Limits and CLI RED/GREEN | Pending | vector_map_config.py, vector_map.py, test_vector_map_cli.py |
| 6. Compatibility and handoff | Pending | tests, scripts/USAGE.md, drift, FP |

### Step 1: Resume

Read `fp context STABL-npoznayt`, current AGENTS.md, project-forward-notes.md, and this plan.
Read human conditional approval and Sigma revision note. Degenerate-subpath review fix is incorporated here.
Claim FP issue as Theta. Use `FP_AGENT_NAME='theta'` for Theta's commands.
Create isolated worktree from current main after preserving this draft or its approved commit.
Check current baseline against `2dc6764`. Do not discard later unrelated commits.

```bash
source /Users/darkbit1001/miniforge3/etc/profile.d/conda.sh
conda activate stability-toys
python -m tests.run tests/test_vector_map_cli.py tests/test_vector_map_raster.py tests/test_vector_map_raster_cli.py tests/test_vector_map_packaging.py tests/test_vector_map_vtracer.py tests/test_vector_map_topology.py tests/test_vector_map_corpus.py -- -q --no-cov
```

Reference result at approved S2.3 tip: 410 passed. Rerun to establish Theta's own baseline.
Check `drift refs` before editing each bound module.

### Step 2: Inspector RED/GREEN

Create `tests/test_vector_map_svg.py` with pure XML strings. Do not import VTracer for inspector unit tests.
Use a closed triangle fixture with explicit namespace, dimensions, black fill, and translate.
Keep rejection tests independently parameterized so one defect cannot mask another.

Required RED matrix:

- XML: malformed, wrong namespace/root, nested svg, DTD/entity, processing instruction, non-whitespace `.text` and `.tail`.
- Elements/attributes: image, use, defs, script, mask, clipPath, href/xlink:href, style, opacity, stroke, and unknown attributes.
- Paths: empty, degenerate-only, three-or-more-point collinear, missing Z, unmatched Z, incomplete pair, extra coordinates, relative commands, curves, and trailing junk.
- Mixed subpaths: polygon plus M/Z or two-point subpath passes, preserves data, and increments degenerate_subpaths.
- Per-path rule: valid polygon in another path cannot rescue degenerate-only path.
- Counts: degenerate commands still consume shared command budget.
- Numbers: NaN, Infinity, 1e999, underscores, missing exponent digits, translated overflow.
- Groups: accumulated translate, black fill inheritance, preserved order, unsupported transform rejection.
- Limits: exactly-at-limit passes; one-over fails; Unicode counts UTF-8 bytes; multiple paths share one command budget.
- Empty policy: zero paths accepted only with allow_empty=True. Empty path element always rejected.

Representative assertions:

```python
result = inspect_svg(valid_svg, width_px=20, height_px=10, limits=SvgLimits(max_paths=1))
assert result.metrics.paths == 1
assert result.metrics.commands == 4
assert result.metrics.subpaths == 1
assert result.metrics.degenerate_subpaths == 0
with pytest.raises(SvgInspectionError, match="closed"):
    inspect_svg(open_svg, width_px=20, height_px=10)
```

Run `python -m tests.run tests/test_vector_map_svg.py -- -q --no-cov`.
Record expected failures before implementation. Implement smallest shared XML and polygon validation needed for GREEN.
Use parser callbacks for rejected declarations. Do not scan document text with regex.

### Step 3: Adapter boundary RED/GREEN

Add keyword-only svg_limits without changing positional options or TraceResult raw bytes.
Run inspector before full-canvas analysis. Reuse inspected translated subpaths in existing full-canvas policy.
Retain zero-path success at adapter level and all original polarity tests.

Add tests proving oversized output is rejected before XML parsing.
Add tests proving DTD and forbidden elements cannot reach old full-canvas parser first.
Add malformed-output tests through real adapter boundary with monkeypatched upstream return value.
Keep separate real VTracer tests for conversion evidence.
Pin deterministic noisy-mask fixture containing polygon and degenerate subpaths at default speckle 4.
Pin isolated-pixel output at speckle 0 as degenerate-only rejection, and zero-path acceptance at speckle 4.
Update synthetic adapter fixtures only to satisfy valid document contract. Never weaken original assertions.

Run `python -m tests.run tests/test_vector_map_svg.py tests/test_vector_map_vtracer.py -- -q --no-cov`.

### Step 4: Normalization RED/GREEN

Pin physical dimensions, required viewBox, matching raw dimensions, optional matching viewBox, and mismatched viewBox rejection.
Compare parsed path/group attributes before and after normalization. Do not compare incidental XML whitespace.
Pin nested translation preservation and asymmetric origins.
Pin unchanged d attributes for polygon paths containing degenerate subpaths. Verify degenerate_subpaths survives normalization.
Pin small positive dimensions below 0.000001 mm: output stays positive, not `0mm`.
Pin whole-number formatting, finite canvas checks, wrong raw units, and output-byte limit after serialization.
Pin empty normalized output rejection while adapter empty-output test remains green.

Return measured normalized bytes and command/path counts for later S2.6 use.
Do not add manifest output or new JSON fields.

Run `python -m tests.run tests/test_vector_map_svg.py tests/test_vector_map_cli.py -- -q --no-cov`.

### Step 5: Configuration and CLI RED/GREEN

Add three flat recipe fields and corresponding CLI flags from limits table.
Store resolved limits as frozen value in Settings. Validate after precedence resolution.
Extend recipe integer checks and explicit CLI None handling.
Pass same limits to adapter and normalizer. Use normalizer result for publication and path count.

Add tests for defaults, recipe-relative behavior, explicit override, zero/negative/bool/float/null, and unknown limit names.
Add in-process CLI tests with injected upstream SVG for every processing error category.
Assert return code, one JSON object, measured-limit diagnostic, and unchanged existing destination under --overwrite.
Assert no fallback retry and unchanged VTracer options after every limit failure.
Preserve raster warnings and existing destination alias refusal.
Assert exactly one counted warning per layer containing degenerate subpaths, in text and JSON modes.
Assert no degenerate warning when count is zero. Preserve successful publication for mixed polygon/degenerate paths.

Run inspector, CLI, adapter, and packaging tests together after GREEN.
No package dependency changes expected. Existing vector extra and sibling command isolation must still pass.

### Step 6: Compatibility, mutations, and review handoff

Trace all eight corpus masks and all five topology fixtures through production normalizer.
Render normalized SVG with existing resvg helper at processing dimensions.
Require raw versus normalized rendered-mask equality. Retain existing locked topology and corpus acceptance checks.
Do not change tolerances or fitting options when a fixture fails.

Run all existing OpenSCAD proof cases through production normalizer, not only test-only physical_svg string rewrite.
Preserve raw-header misplacement regression as explicit negative control.
Add OpenSCAD render of normalized path containing polygon plus `M p Z`.
Compare resulting mesh bounds and topology against polygon-only control. Record import success or exact failure.
This render is required acceptance evidence. Do not claim degenerate-subpath import compatibility without it.
Optionally record bounded render timing near one configured limit for S2.6. Record input counts, host, timeout, and elapsed time.
Timing is nonblocking. Do not treat configured limits as OpenSCAD performance guarantees.
Check bounds, components/holes, extrusion Z, and asymmetric marker with existing tolerances.
Use existing Flatpak home-directory handling. No generic /tmp render paths on Flatpak.
OpenSCAD or renderer absence is blocked acceptance, not a passing compatibility result.

Mutations must break tests when these checks are removed:

- Pre-parse byte budget.
- Explicit closure and finite coordinate checks.
- Forbidden resource/attribute check.
- Group translation preservation.
- Global command budget.
- Matching viewBox check.
- Post-normalization byte budget.
- Degenerate preservation, metrics, warning count, and per-path polygon requirement.

Run focused suite including new SVG tests and existing raster, CLI, adapter, packaging, topology, corpus, and OpenSCAD tests.
Then run `python -m tests.run tests/ -- -q --no-cov` once after final code change.
Known baseline: stale `test_entry_spans.py::test_an_UNHASHABLE_type_does_not_break_the_span` assertion, recorded on STABL-gzfzzsdq.
Report new failures separately. Do not repair unrelated tests.

Update scripts/USAGE.md with grammar, limits, counts, suggestions, and incompatibility boundary.
Review bound prose before refreshing drift provenance. Run `drift check` and `git diff --check`.
Commit named files with STABL-npoznayt and exact next review step.
Assign commit in FP. Post STOP/NEXT with RED/GREEN, mutation, renderer, OpenSCAD, full-suite, and drift evidence.
Stop ready for independent review. Do not push, merge, finish issue, or claim S2.6.

## Planning verification and references

Planning inspected live code and generated real VTracer output from eight existing corpus masks.
No implementation code, tests, package configuration, or task state changed during planning.
No renderer or OpenSCAD run performed during this planning session.

ElementTree provides parser callbacks for declarations and structured XML inspection.
Source: [Python ElementTree documentation](https://docs.python.org/3/library/xml.etree.elementtree.html).
SVG path grammar distinguishes explicit closepath from merely returning to start.
Source: [W3C SVG 1.1 paths](https://www.w3.org/TR/SVG11/paths.html#PathData).
Broader parser alternative: [svgelements](https://github.com/meerk40t/svgelements).

Review disposition: recommended degenerate-subpath policy adopted. Width/height keywords, tail validation, and package-metadata provenance clarified.
Optional render timing remains nonblocking. Required OpenSCAD degenerate-subpath render belongs to Theta execution.
