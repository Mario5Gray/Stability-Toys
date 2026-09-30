# Test collection isolation implementation plan

Issue: STABL-sgdavnvz. Design approved by human. Waveplan not in use.
Use inline execution. Do not delegate implementation.

Goal: keep stub imports outside real-library and live CUDA test processes.
Architecture: select file cohorts before collection, then launch fresh Python interpreters.
Stack: Python subprocess, pytest, pytest-cov, Make, Docker Compose.

| Step | Files | Evidence | State |
|---|---|---|---|
| 1. Add failing process-boundary tests | tests/test_test_isolation.py | Mixed collection rejected before side effects. Separate processes pass. | Complete |
| 2. Add cohort selector and runner | tests/isolation.py, tests/run.py, tests/conftest.py | Boundary tests pass. Filters and failure codes preserved. | Complete |
| 3. Remove unsafe repair | tests/conftest.py, tests/test_hunyuandit_acceptance.py, tests/test_diffusers_stub_leak.py | Real dependency imports pass without xfail. | Complete |
| 4. Route entry points | Makefile.test, docker-compose.test.yml, Dockerfile.test | Native and CUDA commands use same runner. Watch uses runner. | Complete |
| 5. Check complete cohorts | Tests with hidden stub dependencies | Full inventory executes without new failures or skips. Coverage includes all cohorts. | Complete |
| 6. Verify CUDA and docs | docs/TESTING_IN_DOCKER.md, project-forward-notes.md, docs/notes-ledge.md, drift.lock | Dedicated enigma acceptance and drift check. | Complete |

RED command: `python -m pytest tests/test_test_isolation.py -q -o addopts=''`.
GREEN command: same command after implementation.
Suite command: `python -m tests.run tests -- -q -o log_cli=false`.
CUDA command: `docker compose -f docker-compose.test.yml run --rm test-cuda python -m tests.run tests/test_hunyuandit_acceptance.py -- -q`.

Preserve unrelated worktrees and deployment. Test CUDA changes in isolated container.
Commit verified work with issue ID and next review step. Record evidence in FP.

## Initial verification evidence: c248d26

Initial review approved with small fixes. Follow-up changes need review.

| Check | Result |
|---|---|
| Original worker plus real detector | 26 passed, 1 failed before isolation |
| Boundary RED | 6 failed, 2 passed before runner existed |
| Coverage RED | Separate processes measured 75% each and failed aggregate requirement |
| Full local cohorts | 1,638 passed, 10 skipped, exit 0 |
| Combined application coverage | 78% across backends and server |
| Final focused boundary, entrypoint, and leak checks | 17 passed after stronger failure-order assertions |
| Dedicated enigma CUDA acceptance | 1 passed in 85.66 seconds |
| Drift | All 19 managed documents passed after prose review |
| Compose commands | All six rendered services use runner |
| Local Make command | Selected test path and pytest options preserved |

Full suite ran before one extra exit-first regression and stronger test assertions.
Final focused run covers these test-only changes. Runner implementation stayed unchanged.
Local skips: one absent env.custom, eight SDXL GPU cases, one Hunyuan GPU case.
Absent env.custom skip predates this change. Parent commit already contains this skip in tests/test_env_file_contract.py.
Enigma separately executed Hunyuan acceptance with real model assets.

CUDA check used source mounted into existing harbor.lan/stability-toys:test image.
Image ID: sha256:6b72e5b3e7c12bb2c1138a9685b7e54d75b2463adbef3e726ce860592b320f36.
No image rebuild performed. Deployment stayed running.
Optional Ruff check unavailable in dedicated conda environment. No dependency installed for this check.
Shared Concourse task checks syntax and Ruff only. No shared pipeline pytest change required.

Evidence logs: /tmp/STABL-sgdavnvz-diagnosis/ on implementation host.
Full run: final-suite.log. CUDA run: cuda-acceptance.log.

## Review follow-up

Reported option-parsing failure did not reproduce on c248d26.
`--ignore=x` and `--deselect=...` pass through to pytest. Unknown options receive pytest usage errors.
Partition already splits `::` before checking path existence. No partition change required.

Runner now rejects explicit `--` without preceding paths. No-argument invocation still defaults to tests/.
Regression tests cover these cases and existing option handling.
Concourse note now records verification date and source paths.

RED: one failed, 16 passed. Empty-path test exposed silent default-suite execution.
GREEN: 27 passed, one existing env.custom skip across isolation, Make entrypoint, leak, and environment tests.
Logs: /tmp/STABL-sgdavnvz-diagnosis/review-red.log and review-green.log.
Full suite and CUDA evidence above belong to c248d26. Follow-up changed only empty-path validation, tests, and documentation.
