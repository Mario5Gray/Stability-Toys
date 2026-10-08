# S2.6 manifest and bundle implementation steps

Issue: STABL-fmjwbrzw. Implementer: Sigma. Reviewer: Theta.
Baseline: `b64711f`. Approved contract: [Sprint 2 remainder](2026-10-07-vector-map-sprint-2-remainder.md).
Scope: S2.6 only. Human approval covers contract and three re-review implementation notes.
Process: inline RED/GREEN with Superpowers executing-plans. No subagent development.
Stop at ready for independent review. Do not merge, push, finish issue, or claim S2.7.

## Environment and authority

Worktree: `.worktrees/fmjwbrzw-bundle`. Branch: `feat/fmjwbrzw-bundle`.
Python environment: `/Users/darkbit1001/miniforge3/envs/stability-toys`.
Baseline command: `python -m pytest tests/test_vector_map_*.py -q`.
Baseline result: 696 passed. Log: `/tmp/fmjwbrzw-baseline.log`.
SWIM status reports missing schedule, journal, and state.
Human explicitly authorized Sigma to execute directly through FP. Do not recreate removed ledger.

## File responsibilities

| File | Change |
|---|---|
| `scripts/vector_map_artifacts.py` | New bundle paths, input snapshots, manifest, debug replay, staging, and publication |
| `scripts/vector_map.py` | Preflight bundle, supply snapshots, publish results, report retained debug evidence |
| `scripts/vector_map_config.py` | Capture recipe bytes during existing parse without a second read |
| `scripts/vector_map_raster.py` | Decode supplied snapshots while preserving source paths in diagnostics |
| `scripts/pyproject.toml` | Package new artifact module |
| `tests/test_vector_map_artifacts.py` | New ownership, provenance, staging, failure, and debug tests |
| `tests/test_vector_map_cli.py` | Manifest JSON path and failure contract |
| `tests/test_vector_map_packaging.py` | New module ships and standalone conversion remains Torch-free |
| `scripts/USAGE.md` | Bundle, overwrite, hard-link limitation, debug retention, and replay instructions |

Check drift bindings before edits. Update bound prose before refreshing provenance.
Preserve existing raster and adapter call signatures through optional keyword arguments.
Retain source paths for diagnostics. Decode snapshots through `BytesIO`.
Snapshot source and constraint bytes before raster processing. Hash those same bytes.
Capture recipe bytes during parsing. Do not reread recipe for provenance.

## Proposed internal shape

`Bundle` owns destination SVG, sibling manifest, and optional fixed debug paths.
Preflight checks all requested paths before raster processing.
Inputs include source, recipe, include mask, and exclude mask.
Input aliases use `os.path.samefile`, plus resolved-path comparison when an output does not exist.
Reject output directories and symlinked debug directory.
Do not trust old manifest paths for deletion or replacement. Fixed output names establish ownership.
Preserve unrequested artifacts and unrelated files. New manifest lists only this run's artifacts.

Manifest fields: `schema_version`, `versions`, `inputs`, `canvas`, `preparation`, `vtracer`, `svg_limits`, `layers`, `artifacts`, `counts`, `warnings`.
Each input has role, source path, SHA-256, and byte count.
Each artifact has relative path, SHA-256, and byte count.
Layers contain `{"id":"standalone","height_mm":null}`.
Counts come from `NormalizedSvg.metrics`. Canvas and preparation come from existing immutable values.
JSON uses sorted keys, fixed whitespace, UTF-8, and `allow_nan=False`.
Manifest excludes its own hash, clocks, elapsed times, and temporary paths.

Debug outputs: `<stem>.debug/mask.png` and `<stem>.debug/recipe.json`.
Replay recipe selects `mask.png` relative to recipe directory with `input_kind: mask`.
Replay uses final physical width, upstream options, and SVG limits.
Replay disables original inversion, alpha selection, resize, constraints, and edge expansion.
Successful manifest retains original resolved preparation independently of replay settings.
`--debug-bundle` is an output flag. It does not change recipe schema or raster settings.

## Increment 1: manifest provenance and consumed bytes

1. Add CLI test requiring sibling manifest after real donut conversion.
2. Assert nullable height, package versions, physical dimensions, settings, counts, and warnings.
3. Independently hash source bytes and published SVG bytes.
4. Assert manifest lacks self-hash, timestamp, timing, and staging paths.
5. Add recipe and constraint provenance tests.
6. Add test changing source file after snapshot but before decoding.
7. Assert decoding and manifest hash both describe original snapshot.
8. Run focused tests and retain expected RED output.
9. Implement input capture, snapshot decode, and pure manifest serialization.
10. Update success JSON expectation with manifest path.
11. Run focused tests to GREEN.
12. Commit increment with issue ID and next step.

## Increment 2: ownership and collision preflight

1. Add tests for existing SVG, manifest, debug mask, and debug recipe.
2. Prove collisions fail before raster processing or VTracer invocation.
3. Parameterize direct, symlink, and hard-link aliases against each input role.
4. Include manifest and debug outputs in alias checks.
5. Add symlinked debug-directory and output-directory rejection tests.
6. Add overwrite test preserving unrelated sibling and debug files.
7. Run tests to expected RED.
8. Implement fixed-name ownership and preflight checks.
9. Run tests to GREEN.
10. Commit increment with issue ID and next step.

## Increment 3: staged publication and failure order

1. Add failure injection during staging and before old-manifest invalidation.
2. Assert old artifact bytes and manifest remain unchanged.
3. Add failure injection during invalidation, first output, later output, and manifest publication.
4. Assert no successful result follows failure.
5. After invalidation, assert no completion manifest survives failed publication.
6. Observe publication order and independently check hashes when manifest first appears.
7. Create a final file after preflight and before publication.
8. Assert `os.link` fails with EEXIST and preserves competing file bytes.
9. Inject unsupported-hard-link error.
10. Assert diagnostic explains filesystem limitation and explicit `--overwrite` retry through replacement.
11. Assert no automatic retry or rename fallback occurs.
12. Run tests to expected RED.
13. Stage every file on destination filesystem.
14. Invalidate old manifest only after staging succeeds.
15. Publish through `os.link` without overwrite and `os.replace` with overwrite.
16. Publish manifest last.
17. Remove temporary staging files on failure or success.
18. Run tests to GREEN.
19. Commit increment with issue ID and next step.

## Increment 4: debug replay and processing failures

1. Add real mask and edge conversion tests with `--debug-bundle`.
2. Replay saved recipe and compare resulting SVG with original conversion.
3. Independently verify successful manifest hashes for both debug files.
4. Inject VTracer failure after preparation.
5. Assert fixed debug files remain and failure JSON reports their location.
6. Assert diagnostic contains original failure and `--overwrite` retry instruction.
7. Assert retry without overwrite stops before processing.
8. Assert overwrite retry changes only fixed owned files.
9. Start with existing successful bundle, then fail tracing with overwrite and debug retention.
10. Assert old manifest invalidates before referenced debug files change.
11. Inject additional debug publication failure.
12. Assert original processing error remains alongside debug failure.
13. Run tests to expected RED.
14. Implement replay bytes and bounded processing-failure retention.
15. Run tests to GREEN.
16. Commit increment with issue ID and next step.

## Increment 5: packaging, docs, and review evidence

1. Add RED packaging check for artifact module.
2. Register module without adding runtime dependencies.
3. Add three-run same-path repeatability check for deterministic bundle bytes.
4. Save each run's bytes before next overwrite run.
5. Preserve separate broad S2.8 acceptance ownership.
6. Run focused suite and inspect exact counts and exit code.
7. Mutate early manifest publication, omitted alias check, and omitted hash separately.
8. Require relevant tests to fail for each mutation.
9. Restore production code after each mutation.
10. Update operator documentation, including partial output after publication failure.
11. Review bound prose and refresh only affected drift anchors.
12. Run final focused suite, full isolated runner, drift check, and diff check.
13. Record historical full-suite failure separately if it recurs.
14. Commit verified increment and assign revisions in FP.
15. Post STOP/NEXT with logs, counts, limitations, and exact review range.

## Commands

Run each focused command before implementation for RED and after implementation for GREEN.
Add new files to command only after creating their tests.

```bash
source /Users/darkbit1001/miniforge3/etc/profile.d/conda.sh
conda activate stability-toys
python -m pytest tests/test_vector_map_artifacts.py tests/test_vector_map_cli.py tests/test_vector_map_packaging.py -q
python -m pytest tests/test_vector_map_*.py -q
python -m tests.run tests/ -- -q
drift check
git diff --check
```

Required result: all new and focused tests pass. Report full-suite failures and skips without claiming full success.
S2.7 renderer/FreeType provenance and broad S2.8 acceptance stay outside this task.
