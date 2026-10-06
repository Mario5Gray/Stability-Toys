# S2.3 raster implementation plan

Issue: STABL-vjpnctjh. Execute inline with Superpowers executing-plans and TDD. No subagents.
Goal: Prepare oriented binary masks and physical edge bands before VTracer.
Contract: [approved spec](../specs/2026-10-06-vector-map-raster-design.md), commit `e45e558`.
Stack: Pillow, NumPy, OpenCV, Python 3.12, pytest, existing VTracer adapter.

## Authority and status

Human selected S2.3 and approved corrected spec through FP review.
No S2.3 waveplan bundle exists in inspected workspace. Work remains within selected FP issue.
Implementation stops ready for review. Do not finish issue, push branch, or start another task.

| Step | State | Evidence |
|---|---|---|
| 1. Raster contract RED | Complete | 61 failed, 4 passed. `/tmp/vjpnctjh-raster-red.log` |
| 2. Configuration and raster GREEN | Complete | 65 passed. `/tmp/vjpnctjh-raster-green.log` |
| 3. CLI RED/GREEN | Complete | RED: 14 failed, 3 passed. GREEN: 279 passed. `/tmp/vjpnctjh-focused.log` |
| 4. Mutations and verification | Complete | Current vector suite: 410 passed. Initial mutations and full results below. |
| 5. Commit and review handoff | Ready for review | Commit containing this plan. FP owns independent review state. |

## Files and interfaces

- Modify `scripts/vector_map_config.py`: constraint paths, validation, uniform canvas checks.
- Replace `scripts/vector_map_raster.py`: preparation orchestration and immutable result metadata.
- Create `scripts/vector_map_diagnostics.py`: square-coverage warnings only. Never mutate input.
- Modify `scripts/vector_map.py`: flags, alias checks, diagnostic output, preparation result.
- Modify `scripts/pyproject.toml`: package new diagnostic module.
- Create `tests/test_vector_map_raster.py`: pixel-level preparation and diagnostic assertions.
- Update `tests/test_vector_map_cli.py`: replace deferred S2.3 assertions with working behavior.
- Create `tests/test_vector_map_raster_cli.py`: new CLI cases and input protection.

Preparation interface:

```python
@dataclass(frozen=True)
class PreparedRaster:
    material: np.ndarray
    canvas: Canvas
    original_size: tuple[int, int]
    oriented_size: tuple[int, int]
    processed_size: tuple[int, int]
    expansion: BandExpansion | None
    diagnostics: tuple[dict, ...]

@dataclass(frozen=True)
class BandExpansion:
    requested_width_mm: float
    radius_px: int
    kernel_size_px: int
    achieved_width_mm: float
    expansion_mm: float

def prepare(settings) -> PreparedRaster:
    ...

def feature_diagnostics(material) -> tuple[dict, ...]:
    ...
```

Ellipses above denote interface signatures, not implementation steps.
Retain `prepare_mask(path, *, invert)` for existing regression tests. No production callers remain.
It delegates decoding and returns boolean pixels.
Warnings from compatibility wrapper use Python warnings. CLI uses structured preparation diagnostics.

## 1. Raster RED

Activate dedicated environment for all Python commands:

```bash
source /Users/darkbit1001/miniforge3/etc/profile.d/conda.sh
conda activate stability-toys
```

Write pixel-level tests through `prepare(settings)` and `feature_diagnostics(material)`.
Use module imports inside test bodies for new module, so missing behavior fails tests rather than collection.

Representative contracts:

```python
result = raster.prepare(settings)
assert result.processed_size == (7, 5)
assert result.canvas.mm_per_px == 2
assert result.canvas.width_mm == 14
assert result.canvas.height_mm == 10

result = raster.prepare(edge_settings)
assert result.expansion.kernel_size_px == 5
assert result.expansion.achieved_width_mm == 5
assert result.expansion.requested_width_mm == 3.1
```

Cover all eight EXIF orientations with independently computed expected pixel transforms.
Cover alpha thresholds, palette transparency, luminance warning, inverted alpha, JPEG, bad formats, and bad bytes.
Cover nearest-neighbour resize, no upscale, mismatch rejection, and input immutability.
Cover inclusion and exclusion before and after dilation, source inversion independence, and thick edges.
Cover fractional rounding, overflow, invalid resolutions, and reserved `mask` setting.
Cover material bands and background gaps of widths one through four.
Cover edge strips, exterior margins, channels, narrow canvases, diagonals, and S1.4 hard cases.

Run:

```bash
python -m tests.run tests/test_vector_map_raster.py -- -q
```

Expected RED: missing preparation and diagnostics behavior. Record counts and reasons in FP.

## 2. Raster GREEN

Validate positive finite dimensions, optional edge width, integer positive resolution, and supported constraint paths.
Reserve `mask` for S3.1. Enable edges, alpha, and max resolution.
Reject nonfinite or zero derived scale and physical dimensions.

Decode with Pillow and apply `ImageOps.exif_transpose` before thresholding.
Detect alpha through bands or transparency metadata. Warn when luminance ignores alpha.
Use alpha >= 128 when selected. Use luminance >= 128 otherwise. Invert afterward.
Check independently oriented constraint sizes before nearest-neighbour resize.

Resize longest side to cap. Round shorter side half upward using integer arithmetic.
Derive uniform canvas once with `physical_canvas`.
Apply constraints, expand edges with centered square kernel, then reapply constraints.
Use separable rectangular morphology to avoid quadratic kernel allocation.
Limit effective kernel reach to canvas extent while retaining requested expansion metadata.
Reject integer overflow before constructing OpenCV arrays.

Diagnostics use padded binary rasters. Pad material with zero and background with one.
Padding must include external erosion centers, so exterior margins remain covered during dilation.
Use reflected erosion/dilation anchors for 4x4 coverage. Crop padding after both operations.
Return warning per affected class with count and method. Keep material unchanged.

Rerun raster command. Expected GREEN.
Update plan state and FP evidence after passing run.

## 3. CLI RED/GREEN

Add real subprocess tests for edges, uniform SVG scale, both constraints, recipe-relative paths, and explicit overrides.
Assert one JSON result, unchanged result keys, warnings on stderr, and diagnostics in JSON.
Assert requested width warning and material warning both survive on three-pixel bands.
Assert requested 3.1 mm at 1 mm/px reports achieved 5 mm.
Test source, include-mask, and exclude-mask destination aliases using direct paths, symlinks, and hard links.
Test missing alpha, unsupported format, mismatch, malformed bytes, and no artifact on failure.

Run new CLI tests before CLI changes:

```bash
python -m tests.run tests/test_vector_map_raster_cli.py -- -q
```

Replace deferred S2.3 tests with orientation, alpha warning, and enabled-option checks.
Keep S3.1 image/mask and S2.7 preview rejection tests.
Wire preparation result into tracing. Print expansion metadata before tracing.
Return successful diagnostics in JSON. Preserve warnings when later tracing fails.
Add new constraint paths to alias refusal before any source decoding.
Add diagnostic module to package module list.

Run:

```bash
python -m tests.run tests/test_vector_map_raster.py tests/test_vector_map_raster_cli.py tests/test_vector_map_cli.py tests/test_vector_map_packaging.py tests/test_vector_map_vtracer.py -- -q
```

Expected GREEN. Update plan and FP evidence.

## 4. Verification

Temporarily mutate each behavior. Run targeted tests. Restore source immediately after each mutation.
Mutations: threshold comparison, EXIF transpose, final constraints, odd-width ceiling, exterior background, background diagnostic, and constraint aliases.
Each mutation must cause test failure. Keep mutation logs outside tracked source.

Run focused topology and corpus tests with preparation tests.
Run full split suite:

```bash
python -m tests.run tests/ -- -q --no-cov
git diff --check
drift check
```

Report unrelated failures separately. Do not repair unrelated tests.
Review diff against every approved spec section. Update any stale prose before refreshing drift bindings.

## 5. Handoff

Commit named files only. Include STABL-vjpnctjh and exact review next step.
Assign commit through `fp issue assign STABL-vjpnctjh --rev <sha>`.
Post one final STOP/NEXT comment with focused, full, mutation, and drift evidence.
Keep issue in progress. Report ready for independent review.

## Initial verification record at 32d896e

- Baseline CLI, packaging, adapter: 194 passed.
- Raster RED: 61 failed, four passed. Initial GREEN: 65 passed.
- CLI RED: 14 failed, three passed.
- Constraint-alpha warning clarification: one RED failure, then GREEN.
- Final vector suite, including topology and corpus: 396 passed in 12.97 seconds.
- Full real-library cohort: 1,864 passed, nine skipped, one failure.
- Full stub cohort: 213 passed. Live CUDA acceptance: one skipped.
- Only failure: `test_entry_spans.py::test_an_UNHASHABLE_type_does_not_break_the_span`.
- FP STABL-gzfzzsdq records this unchanged stale assertion. No unrelated repair made.
- Skips: eight SDXL CUDA tests, absent worktree `env.custom`, and live CUDA acceptance.
- Eight mutations caught: threshold, orientation, initial constraints, final constraints, rounding, exterior background, background warnings, and aliases.
- `git diff --check` and `drift check` passed. Coverage and container execution not measured.

Logs: `/tmp/vjpnctjh-vector.log`, `/tmp/vjpnctjh-full.log`, and `/tmp/vjpnctjh-mutation-*.log`.
All eight reviewed corpus masks retain original pixels. Diagnostics never mutate material.
Scale-only CLI fixture explicitly uses `filter_speckle=0`, because default four removes its 15-pixel region.
Production VTracer default remains unchanged.

## Review repair: odd-width rounding

Review reproduced 0.28 mm on 100 mm / 2500 px canvas as nine pixels instead of seven.
Regression RED: one failed, one passed. Exact-arithmetic sweep also failed before fix.
Subtract `1e-9` before radius ceiling. This suppresses floating-point noise at odd boundaries.
Raster GREEN: 80 passed, including 16,800 exact-arithmetic comparisons and genuine above-boundary expansion.
NEXT: reject unsupported sample modes with explicit conversion instruction, then rerun vector suite.

## Review repair: unsupported pixel modes

Reject decoded modes outside `1`, `L`, `LA`, `P`, `RGB`, and `RGBA` before luminance conversion.
Return exit 2 with explicit 8-bit conversion instruction. RGBA PNG instruction preserves optional alpha.
RED: five failures for 16-bit source, both constraints, raster decoding, and CMYK. Six supported-mode controls passed.
GREEN: complete vector suite now passes 410 tests, including topology, corpus, packaging, and CLI failures.
Input bytes remain unchanged. Rejected inputs produce one invalid JSON result and no SVG.
`git diff --check` and `drift check` passed.
Full split suite and containers not rerun for these bounded fixes. Initial full-suite evidence above remains historical.
Logs: `/tmp/vjpnctjh-rounding-red.log`, `/tmp/vjpnctjh-sweep-red.log`, `/tmp/vjpnctjh-modes-red.log`, and `/tmp/vjpnctjh-review-green.log`.
MPO support remains unverified. Compatibility wrapper remains, with test-only ownership clarified.
NEXT: independent re-review of both fix commits. Keep issue in progress and branch local.
