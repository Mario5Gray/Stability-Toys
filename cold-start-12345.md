# Cold-start prompt: resume Sigma S2.6

You are Sigma. Resume STABL-fmjwbrzw, S2.6 manifest and artifact bundle.
User paused this session after implementation and mutation checks. Final verification remains unfinished.
Treat this prompt as resume context when user starts next session.

## Authority and stopping boundary

User explicitly said: "You are sigma... execute directly."
Direct FP execution is authorized. Do not request another waveplan exception.
No S2.6 SWIM bundle exists. Do not recreate removed execution ledger or invent waveplan transitions.
Issue remains in-progress. Do not mark done before independent review and human completion.
Implement S2.6 only. Stop ready for Theta review.
Do not push, merge, claim S2.7, or claim S2.8.
Do not use subagent-driven development.

## Workspace

Primary checkout: `/Users/darkbit1001/workspace/Stability-Toys`.
Implementation worktree: `/Users/darkbit1001/workspace/Stability-Toys/.worktrees/fmjwbrzw-bundle`.
Branch: `feat/fmjwbrzw-bundle`.
Run implementation commands in that worktree.
Primary main remains at `b64711f`, two planning commits ahead of origin/main at last check.
Do not change unrelated `.worktrees/vjpnctjh-raster`.

Relevant commits:

- `d11999f`: S2.5 merge. `877f486` belongs to origin/main.
- `cd17650`: remaining Sprint 2 plan.
- `b64711f`: approved plan revision.
- `96c0164`: S2.6 RED/GREEN steps.
- `e40d118`: bundle implementation, provenance, publication, debug replay, tests, docs.
- `bcb9a67`: remove completion manifest when staging cleanup fails after manifest publication.
- This prompt has a later documentation-only checkpoint commit.

All implementation commits remain unpushed.
Check current git status before editing. Do not reset or clean unrelated work.

## First commands

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/.worktrees/fmjwbrzw-bundle
git status --short --branch
git log --oneline -8
FP_AGENT_NAME='sigma' fp context STABL-fmjwbrzw
source /Users/darkbit1001/miniforge3/etc/profile.d/conda.sh
conda activate stability-toys
```

Use `python`, not shared-root Python.
Read repository instructions from primary checkout because ignored instruction files may be absent from worktree.
Use Caveman full and controlled English for new prose.
Use semantic tools for symbol navigation. Use drift before editing bound files.
Skills already used: FP Implement, Executing Plans, TDD, Git Worktrees, Waveplan, Systematic Debugging, Drift.

## Read these documents

- `docs/superpowers/plans/2026-10-07-vector-map-sprint-2-remainder.md`: approved contract.
- `docs/superpowers/plans/2026-10-07-vector-map-bundle.md`: concrete RED/GREEN steps.
- `docs/superpowers/specs/2026-09-06-vector-map-relief-design.md`: sections 6.4, 8, 9, 10, 11.2.
- `scripts/USAGE.md`: current operator contract.
- FP comments on STABL-fmjwbrzw: approval, direct authority, implementation evidence, and pause state.

## Implemented behavior

`scripts/vector_map_artifacts.py` owns:

- Input snapshots and SHA-256 provenance.
- Canonical manifest JSON, including installed versions, actual upstream arguments, canvas, preparation, limits, counts, and warnings.
- Layer record `{"id":"standalone","height_mm":null}`.
- Fixed SVG, manifest, and optional debug paths.
- Collision checks and direct/symlink/hard-link input-alias rejection.
- Staging on destination filesystem.
- `os.link` publication without overwrite. EEXIST preserves competing file.
- Explicit `os.replace` publication with overwrite.
- Old-manifest invalidation before replacements. New manifest publishes last.
- Failure after final link invalidates new completion marker.
- Unsupported-hard-link diagnostic directs explicit `--overwrite` retry. No automatic fallback.

`scripts/vector_map.py` integrates bundle publication and `--debug-bundle`.
Success JSON adds `artifacts.manifest` and optional `artifacts.debug`. Existing top-level fields remain.
Failure JSON reports retained debug directory only after successful debug publication.
Input, recipe, and constraint hashes describe exact consumed bytes.
Recipe parser captures bytes once. Raster decoder accepts optional byte snapshots through `BytesIO`.

Debug ownership is limited to `<stem>.debug/mask.png` and `recipe.json`.
Reject symlinked debug directory. Preserve unrelated files.
Replay recipe traces final prepared mask without applying preparation twice.
Tracing failure can retain debug evidence. Retry requires `--overwrite`.
Before retained debug replaces old referenced files, old completion manifest invalidates.
Debug-publication failure preserves original processing error alongside debug error.

Packaging includes artifact module. No runtime dependency added.
Production VTracer, SVG geometry, renderer, and S2.7 behavior remain unchanged.

## Exact verification state

Fresh pause check on final production code:

```bash
python -m pytest tests/test_vector_map_artifacts.py -q
```

Result: 88 passed, exit 0. Log: `/tmp/fmjwbrzw-pause-check.log`.
Drift check passed all 21 managed documents after final source/prose review.
Staged diff check passed before checkpoint commit.

Earlier focused suite at `e40d118`:

```bash
python -m pytest tests/test_vector_map_*.py -q
```

Result: 782 passed, exit 0. Log: `/tmp/fmjwbrzw-focused.log`.
This result predates two final audit tests and cleanup fix. Do not call it final verification.
Initial baseline at `b64711f`: 696 passed. Log: `/tmp/fmjwbrzw-baseline.log`.

RED evidence:

- `/tmp/fmjwbrzw-red1.log`: 4 failures for absent module and packaging.
- `/tmp/fmjwbrzw-red2.log`: 72 failures for absent publisher, 4 passing provenance tests.
- `/tmp/fmjwbrzw-red3.log`: 10 missing CLI/debug cases failed, 76 passed.
- `/tmp/fmjwbrzw-red4.log`: cleanup failure retained completion marker. 1 failed, 1 passed.

Five mutations killed, each with pytest exit 1:

- Early manifest publication.
- Omitted input-alias check.
- Incorrect artifact hash.
- Replacing rename substituted for no-overwrite hard link.
- Omitted completion invalidation after cleanup failure.

Summary: `/tmp/fmjwbrzw-mutations.json`.
Logs: `/tmp/fmjwbrzw-mutation-*.log`.
Mutation runner restored source in `finally`. Fresh pause check passed afterward.

No final complete focused suite, full isolated suite, container run, or independent review completed yet.
Unsupported-hard-link cases use injected errno values. No actual exFAT or SMB run occurred.
Do not claim crash durability or atomic multi-file transaction.

## Remaining work

1. Inspect implementation and final diff for overlooked failure contracts.
2. Run final complete vector-map suite.
3. Run full isolated suite.
4. Classify any failure against unchanged baseline where necessary.
5. Keep known historical STABL-gzfzzsdq entry-span failure separate if it recurs.
6. Run final drift and diff checks after any fixes.
7. Record exact counts, exit codes, skips, limitations, and logs in FP.
8. Commit any remaining fixes with issue ID and exact next step.
9. Report ready for independent Theta review with baseline-to-head range.

```bash
python -m pytest tests/test_vector_map_*.py -q
python -m tests.run tests/ -- -q
drift check
git diff --check
```

Expected focused collection now includes 88 artifact tests plus original 696 tests.
Verify actual result. Do not substitute expected count for execution evidence.

FP revision trap: `fp issue assign --rev HEAD` resolves primary checkout HEAD, even from worktree.
Always pass explicit implementation SHA.
Accidental `b64711f` association was removed. `96c0164` and `e40d118` were restored explicitly.
Pause checkpoint revisions are assigned explicitly as well.

## Preserve later-task boundaries

S2.7 records Pillow and `PIL.features.version("freetype2")`, plus default-font implementation.
S2.7 uses exact-DPI normalized rendering or pixel trace under approved constraint.
S2.8 saves each run's bytes or hashes before next same-path overwrite run.
These notes already exist on their FP issues. Do not implement them during S2.6.
S2.5 hardening issue STABL-eibrjlsh remains separate and nonblocking.
