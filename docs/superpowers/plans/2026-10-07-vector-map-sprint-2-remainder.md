# Sprint 2 remaining work plan

Parent: STABL-neizotrw. Planner: Sigma. Status: draft for human review.
Baseline: `d11999f`, PR #86 merge. Remote main checked on 2026-10-07.
Goal: complete standalone mask/edge wrapper through bundle publication, previews, and acceptance.
Authority: planning only. Human assigns each implementation task separately.
Process: inline execution with Superpowers executing-plans and TDD. No subagent development.

Contract: [vector-map design](../specs/2026-09-06-vector-map-relief-design.md), sections 6.4, 8, 9, 10, and 11.2.
This document plans existing FP issues. It does not recreate removed execution ledger.
FP comments and git history retain execution evidence.

## Confirmed starting point

- S2.1 through S2.5: done. Sprint 2: five of eight original tasks done.
- S2.5: `STABL-npoznayt`, closed at merge `d11999f`.
- Documentation fix `877f486`: ancestor of `origin/main`.
- Remote `refs/heads/main`: `d11999f27f4f4ad73eeb2a3a915dfbfca78cf22e` during planning.
- Additional follow-up: `STABL-eibrjlsh`, todo. Keep separate from eight original tasks.
- Follow-up scope: viewBox separators, deep-group diagnostic, optional bounded render timing.
- No cleanup required. No implementation task claimed during planning.

## Order and dependencies

| Order | Existing issue | Result | Required evidence |
|---|---|---|---|
| 1 | S2.6 `STABL-fmjwbrzw` | Deterministic manifest and staged bundle publication | Hash verification, collision refusal, input protection, publication failure tests |
| 2 | S2.7 `STABL-kfrksmnp` | Accurate operator preview within same bundle | Renderer pin, scale regression, overlay alignment, native container renders |
| 3 | S2.8 `STABL-ntgnbxci` | Standalone release acceptance | Repeated real conversions, actual Canny input, packaging and failure evidence |

Recommend serial execution. S2.7 can reuse S2.6 publication rules without an interim publisher.
Preview-first needs temporary publication behavior. Concurrent implementation creates shared CLI and packaging edits.
Neither alternative offers enough benefit for this remaining scope.

Current FP dependencies:

- S2.6 depends on S2.2 and S2.5.
- S2.7 depends on S2.5 and accepted Q2 renderer decision `STABL-dheskftn`.
- S2.8 depends on S2.1 through S2.6.

Proposed additions: S2.7 depends on S2.6. S2.8 depends on S2.7.
Preserve every existing dependency when recording additions after plan review.
Current FP dependency fields remain unchanged during draft planning.

## Existing implementation boundaries

| File | Current role | Planned change |
|---|---|---|
| `scripts/vector_map.py` | CLI orchestration, destination check, direct SVG write, one JSON result | Replace direct publication with bundle orchestration |
| `scripts/vector_map_config.py` | Resolved settings and physical canvas | Add only reviewed debug configuration |
| `scripts/vector_map_raster.py` | Prepared material, dimensions, expansion metrics, diagnostics | Reuse preparation values for provenance and preview alignment |
| `scripts/vector_map_vtracer.py` | Pinned tracing and SVG inspection boundary | Preserve raw trace and upstream options |
| `scripts/vector_map_svg.py` | Normalized SVG and measured metrics | Reuse normalized bytes and metrics |
| `scripts/pyproject.toml` | Vector extra and packaged modules | Register new modules and exact renderer pin |
| `tests/fixtures/vector_map/corpus/metrics.py` | Locked geometry renderer and acceptance metrics | Preserve locked tolerances and rendering policy |

Keep conversion free from Torch, server imports, and model assets.
Keep image mode, multilayer exports, height controls, and SCAD generation within Sprint 3.
Keep existing exit codes: 0 converted, 2 invalid configuration, 1 processing or I/O failure.
Keep stdout JSON as one final object. Preserve existing fields while adding artifact paths.

## S2.6: manifest and artifact bundle

Issue: `STABL-fmjwbrzw`.
Create `scripts/vector_map_artifacts.py` for artifact paths, provenance serialization, collision checks, staging, and publication.
Create `tests/test_vector_map_artifacts.py` for publication and manifest contracts.
Extend CLI and packaging tests. Update `scripts/USAGE.md` with published bundle behavior.

### Proposed contract for review

Required outputs: `<stem>.svg` and `<stem>.vector.json`.
Manifest uses independent vector schema version 1. It does not reuse `controlnet_map`.
Manifest records:

- Installed wrapper and VTracer versions.
- SHA-256 hashes for source, recipe when supplied, and constraint masks when supplied.
- Original, oriented, and processing dimensions, physical canvas, and millimetres per pixel.
- Resolved preparation settings, upstream options, SVG limits, and achieved edge expansion.
- One layer identity for current standalone output. No invented relief heights.
- Artifact relative paths and SHA-256 hashes, excluding manifest self-hash.
- Path, command, byte, and degenerate-subpath counts from production inspector.
- Preparation and SVG warnings.

Use canonical JSON ordering and finite numeric values. Exclude timestamps, staging paths, and elapsed times.
Keep timing measurements in acceptance evidence outside deterministic artifacts.
Hash bytes consumed during processing. Do not hash changed source files after tracing.
Use installed package metadata for versions. Do not trust VTracer generator comments.

Proposed debug interface: `--debug-bundle` writes `<stem>.debug/`.
Contents: final prepared mask PNG and resolved recipe JSON.
Recipe must replay prepared mask without applying original preparation twice.
Record original preparation separately from replay settings.
Preserve debug evidence after tracing failure when preparation succeeded.
Report debug path with failed result. Do not publish successful conversion manifest after failure.
Debug path and failure-retention behavior require review before S2.6 implementation.

### Implementation sequence

1. Add RED tests for manifest fields, measured hashes, deterministic serialization, and absent self-hash.
2. Implement provenance from existing settings, prepared raster, trace, and normalized metrics.
3. Add RED collision tests before any raster processing or tracing.
4. Include SVG, manifest, requested preview, and requested debug paths in collision planning.
5. Reject direct, symlink, and hard-link aliases to every input, including recipe and constraint masks.
6. Implement scoped overwrite checks that preserve unrelated files.
7. Add RED failure injection at staging, old-manifest invalidation, output replacement, and final-manifest publication.
8. Implement staging on destination filesystem.
9. Invalidate old manifest only after all new artifacts finish staging.
10. Publish output files before manifest.
11. Add debug replay tests for mask and edge preparation.
12. Integrate bundle paths into existing JSON result.
13. Register artifact module in packaging.
14. Update operator documentation and review drift bindings.

Before invalidation, failures preserve existing bundle.
After invalidation, failures leave no valid completion manifest.
Several renames do not provide an atomic bundle transaction.
Validate manifest hashes before treating any bundle as complete.
Reject unsafe manifest member paths before using old manifest for ownership decisions.
Never recursively remove unrelated files under destination or debug directory.
Recheck collisions at publication. Do not silently replace files that appeared after initial check.

Focused RED/GREEN command after adding tests:

```bash
python -m pytest tests/test_vector_map_artifacts.py tests/test_vector_map_cli.py tests/test_vector_map_packaging.py -q
```

RED must expose missing contract behavior. GREEN must pass without skips for artifact handling.
Mutation evidence: early manifest publication, omitted input-alias check, and omitted artifact hash must each fail tests.
Stop after independent review request. Human controls merge and next assignment.

## S2.7: preview output

Issue: `STABL-kfrksmnp`.
Create `scripts/vector_map_preview.py` for rendering and preview composition.
Create `tests/test_vector_map_preview.py` for canvas, image content, and renderer failures.
Extend CLI, artifact, packaging, topology, and adapter render tests.

### Accepted decisions and proposed rendering path

Accepted Q2 decision: add `resvg-py==0.5.0` to existing `vector` extra.
No additional extra or system dependency.
Operator previews use default anti-aliasing. Geometry checks use `shape_rendering='crisp_edges'`.
Replace conditional `rsvg-convert` fixture checks with resvg-py checks.
Require those checks in native test container.

S2.5 constraint: locked renderer rejects millimetre roots without explicit DPI.
Other DPI values can alter intrinsic rounding and rendered geometry.
Accepted options: render pixel trace, or render normalized SVG with `dpi = 25.4 / mm_per_px`.
Proposed choice: normalized SVG with exact DPI and processing width/height.
This renders published geometry directly. Compare against raw pixel trace in regression tests.
Do not use fixed 96 DPI. Do not modify locked corpus renderer to conceal scale errors.

Preview contains three labeled panels:

- Prepared material mask.
- Rendered vector output.
- Vector overlay on oriented source, aligned to processing canvas, with legend.

Use source selected for this command. Edge mode source is supplied edge image.
Do not imply access to original photograph used by an earlier Canny command.
Use deterministic layout and bundled or deterministic font selection.
Record renderer version and rendering settings in manifest.

### Implementation sequence

1. Add RED packaging test for exact pin and packaged preview module.
2. Add renderer dependency to vector extra.
3. Add RED canvas tests using asymmetric fixture and several physical scales.
4. Include gravel case that exposed S2.5 intrinsic-size rounding.
5. Implement normalized rendering with exact DPI.
6. Add RED checks that preview contains vector rendering, prepared mask, aligned overlay, and legend.
7. Implement preview composition at documented processing resolution.
8. Add missing-renderer and renderer-failure tests before publication.
9. Integrate preview into S2.6 staging, collisions, manifest hashes, and JSON paths.
10. Replace `rsvg-convert` fixture checks with pinned renderer checks.
11. Verify ordinary conversion does not import renderer when preview is absent.
12. Update usage and native-container dependencies where required.

Focused RED/GREEN command:

```bash
python -m pytest tests/test_vector_map_preview.py tests/test_vector_map_artifacts.py tests/test_vector_map_packaging.py tests/test_vector_map_topology.py tests/test_vector_map_vtracer.py -q
```

Mutation evidence: omitted DPI, mask-only preview, and missing preview hash must each fail tests.
Native container must run fixture renders without missing-renderer skips.
Stop after independent review request. Human controls merge and next assignment.

## S2.8: repeatability and standalone acceptance

Issue: `STABL-ntgnbxci`.
Create `tests/test_vector_map_acceptance.py` for real command and repeated-bundle checks.
Create `docs/superpowers/reports/2026-10-07-vector-map-sprint-2-acceptance.md` during execution.
Record actual execution date within report. Preserve per-case results and artifact locations.

### Acceptance sequence

1. Run each geometric fixture through real VTracer in mask mode.
2. Run committed source images through actual `st-canny-map` command.
3. Feed produced Canny files to edge mode with physical width and band settings.
4. Repeat each selected conversion three times with fixed inputs, settings, platform, and dependency versions.
5. Compare SVG, manifest, requested preview, and debug bytes across runs using identical destination names.
6. Verify every manifest hash independently from publisher implementation.
7. Recheck EXIF orientation, constraint-mask dimensions, common resize, physical rectangle size, and asymmetric alignment.
8. Recheck include/exclude precedence and clipping after edge expansion.
9. Recheck empty material, malformed SVG, exceeded limits, overwrite refusal, and interrupted publication.
10. Record per-case dimensions, counts, warnings, timings, dependency versions, and command exit codes.
11. Verify sibling entry points and Torch-free conversion from installed package.
12. Run focused suite, native container, and full suite with isolated test runner.

Optional empty layers and relief composition remain Sprint 3 acceptance.
Preserve locked corpus failures and tolerances. Do not require every corpus case to pass geometry thresholds unexpectedly.
Report each case separately. Distinguish successful conversion from geometric acceptance.
Do not promise identical bytes across different native dependency versions.
Do not claim CAD or physical-print acceptance from standalone conversion.

Acceptance commands after adding tests:

```bash
python -m pytest tests/test_vector_map_acceptance.py -q
python -m pytest tests/test_vector_map_*.py -q
docker compose -f docker-compose.test.yml build test
docker compose -f docker-compose.test.yml run --rm test
python -m tests.run tests/ -- -q
```

Use dedicated `stability-toys` conda environment for Python commands.
Record each command, exit code, counts, skips, and log path.
Historical full-suite failure `STABL-gzfzzsdq` requires fresh classification if it recurs.
Container or OpenSCAD absence is missing evidence. Do not report required acceptance as passed when unavailable.

Sprint closes only after S2.6, S2.7, and S2.8 pass review and human-directed completion.
`STABL-eibrjlsh` stays separately tracked unless human explicitly adds it to release blockers.

## Review and execution boundary

Review proposed dependency additions, manifest contract, debug interface, and normalized-preview rendering path.
Recommended first assignment: S2.6 `STABL-fmjwbrzw`.
Expand its reviewed contract into small RED/GREEN implementation steps before production edits.
Do not claim S2.7 or S2.8 automatically after S2.6.

Use supplied waveplan task and SWIM bundle when human assigns execution.
No matching runtime bundle was supplied during planning.
Do not recreate removed ledger or invent waveplan transitions.

Planning verification: inspected live FP issues, current source boundaries, remote main, and existing spec.
No conversion, renderer, container, or full-suite execution occurred during this planning session.
