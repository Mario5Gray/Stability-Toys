# Test interpreter guard implementation plan

Issue: STABL-ylxbzedr. Human approved design. No waveplan. Execute inline.
Goal: reject wrong Python before scientific imports or cohort dispatch.
Architecture: shared standard-library prefix check. Host and container use explicit expected prefixes.
Stack: Python, pytest subprocess tests, Make, Docker Compose.

| Step | Files | Verification | State |
|---|---|---|---|
| 1. Expose missing guard | tests/test_python_env.py | RED: 8 failed, 3 passed | Complete |
| 2. Add shared check | tests/python_env.py, tests/conftest.py, tests/run.py | Wrong prefix exits 4 before sentinels. Valid prefix runs tests | Complete |
| 3. Wire existing launch paths | Makefile.test, Dockerfile.test, docker-compose.test.yml, tests/test_test_isolation.py | Focused checks: 31 passed | Complete |
| 4. Verify and document | docs/TESTING_IN_DOCKER.md, drift.lock | Full suite: 1,655 passed, 10 existing skips. Container: 31 passed. Drift clean | Complete |

Helper compares Path(sys.prefix).resolve() against configured prefix.
Default: Path.home() / 'miniforge3/envs/stability-toys'. Override: ST_TEST_PYTHON_PREFIX.
Error names actual interpreter, expected prefix/bin/python, and conda activation command.
Runner prints error and returns 4 before partition or subprocess dispatch.
Conftest raises pytest.UsageError before NumPy and Pillow imports.
Container image and Compose set ST_TEST_PYTHON_PREFIX=/usr/local.
Make passes EXPECTED_CONDA_PREFIX to shared check and local runner.

RED/GREEN: python -m pytest tests/test_python_env.py -q -o addopts='' -o log_cli=false.
Regression: python -m tests.run tests/test_python_env.py tests/test_test_isolation.py tests/test_makefile_entrypoints.py -- -q -o addopts='' -o log_cli=false.
Full suite: python -m tests.run tests -- -q -o log_cli=false.
Use stability-toys env for local verification. Wrong-interpreter subprocesses are intentional test cases.

## Verification evidence

Implementation ready for review.

| Check | Result |
|---|---|
| Baseline isolation and Make tests | 20 passed |
| New guard RED | 8 failed, 3 passed |
| Final local focused checks | 31 passed |
| Full local cohorts | 1,655 passed, 10 existing skips, exit 0 |
| Combined application coverage | 78% |
| Existing enigma test image with source mount and explicit prefix | 31 passed |
| Rendered Compose config | All six services inherit /usr/local prefix |
| Base Python through runner | Exit 4 with expected interpreter path |
| Base Python through direct pytest, plugin autoload disabled | Exit 4 with expected interpreter path |
| Drift | 19 documents and 110 anchors fresh |

Full suite preceded stronger HOME isolation in Make regression and removal of redundant f-string prefix.
Final focused run covers stronger Make regression. String content stayed identical.
Local skips: absent env.custom, eight SDXL GPU cases, one Hunyuan GPU case.
No GPU inference required for interpreter guard.

Container check used existing harbor.lan/stability-toys:test image.
Image ID: sha256:6b72e5b3e7c12bb2c1138a9685b7e54d75b2463adbef3e726ce860592b320f36.
No image rebuild. Source mounted read-only. Prefix supplied explicitly as Compose now supplies it.
Container warnings: existing event_loop_policy deprecation and cache write denied by read-only source mount.

Direct base pytest with normal plugin autoload fails before conftest in LangSmith imports.
Installed pydantic-core 2.46.3 mismatches Pydantic requirement 2.46.4.
This confirms approved design boundary: conftest cannot intercept earlier plugin loading.
Runner rejects wrong interpreter before pytest starts and avoids that failure.
No host dependencies changed.

Evidence directory: /tmp/STABL-ylxbzedr-verification/.
Logs: red.log, final-focused.log, full-suite.log, container.log, base-pytest.log, base-checked-pytest.log, base-checked-tests.run.log.
Next: review implementation and documented direct-pytest limitation.
