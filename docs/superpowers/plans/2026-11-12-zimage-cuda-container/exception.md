# Container suite exception (STABL-tzctkzmz)

This record scopes one exception to the full container suite gate of Task 7 step 5.
The suite is not a pass. It exits 1.

## Scope

The exception covers exactly the 19 test IDs in `baseline-failures.txt`.
Each ID fails on base commit `81cd3dfc4204d1c42e0282d86c2f528e21dd22b1`.
Each ID has the same failure message on the branch.
The list is closed. A failure outside the list fails the gate.
Do not add an ID without a review.

## Evidence

Both runs used `docker compose -f docker-compose.test.yml build test` and `run --rm test`.
Both runs used the same compose file, `env.dev`, mounts, and environment.
`GIT_SHA` and the `TEST_*_HOST_PATH` variables were unset.
The mount directories were empty at the start of each run.

| Item | Baseline | Branch |
|---|---:|---:|
| Commit | 81cd3dfc4204 | 6be2bc64 (Task 7 steps 1-4) |
| Image ID | sha256:9eb7d7c7f90e39d68b549bd9304f2c11c7a01fca2d97cd025259b64870648620 | sha256:1a53cc82a3e327258efa29f00390c7e00c44198ab7e393d5e09886d809abdd40 |
| Exit code | 1 | 1 |
| Passed | 3572 | 3708 |
| Failed | 19 | 19 |
| Skipped | 70 | 70 |

- The failed test IDs and the failure messages are the same in both runs.
- The skipped test IDs are the same in both runs.
- Each test that passes on the baseline also passes on the branch.
- The branch has 136 new passes.
- Totals count the verbose result line of each test. The cohort summary lines are not complete, because some branch output lines are joined.
- The saved logs are not byte-identical. Some traceback sections are interleaved.

## Causes seen in the sample

The test image copies a fixed file list. It has no `README.md`, no `env.*` file, and no `docs/observability-contract.md`.
One governor test reads an environment default of 3600.0 where the test expects 120.0.
The vector-map corpus tests find no report rows.

## Reviewed

Sigma accepted this scoped exception on 2026-11-13 after an independent comparison of the saved logs.
The tag `harbor.lan/stability-toys:test` now names the baseline image.
Use the branch image ID, or rebuild the branch, for later checks.
