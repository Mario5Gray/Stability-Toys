# STABL-sgdavnvz proposed boundary

Status: human approved design and direct execution. Waveplan not in use.

Measured on main 62edaa4 in dedicated stability-toys environment:

- Hunyuan worker plus leak detector, xfail disabled: 27 passed, 1 failed.
- Hunyuan worker plus model detector: 26 passed, 1 failed in safetensors isinstance.
- Model detector alone: 1 passed.
- Collection excluding eight modules with collection-time sys.modules access: 1424 tests collected without error.
- Collection success does not prove all tests execute successfully in that partition.

Recommendation: separate pytest invocations with file selection before collection.
Use fresh Python interpreters. Do not fork a process that has imported Torch.

Proposed execution contract:

1. Identify explicit stub cohort before importing test modules.
2. Run stub cohort and real-library cohort in separate interpreters.
3. Run live CUDA acceptance in its own clean interpreter.
4. Reject unsafe mixed collection before test imports.
5. Preserve pytest filters, failure exit codes, and combined coverage across cohorts.
6. Wire local Make targets, container Make targets, Compose defaults, and CI entry points.
7. Verify watch and focused-file commands cannot bypass collection boundary.

Initial stub candidates:

- tests/test_cuda_worker_base.py
- tests/test_cuda_worker_capabilities.py
- tests/test_cuda_worker_controlnet.py
- tests/test_hunyuandit_worker.py
- tests/test_model_lifecycle.py
- tests/test_worker_controlnet_metadata.py
- tests/test_worker_handle.py
- tests/test_worker_pool.py

Audit runtime mutations and cross-file imports before fixing final cohort membership.
Tests that depend on another test file installing stubs need explicit fixtures.
Do not hide these failures with skips or additional xfails.

Remove GPU acceptance's unsafe sys.modules repair path.
Remove duplicate bare conftest import.
Never remove real Torch and re-import it in one process.
Make leak detector import real dependencies and pass without xfail or an unimported-module skip.
Use a clean-session guard as equivalent containment. Do not enable destructive autouse restoration.

Verification contract:

- Original mixed-file reproductions demonstrate failures before changes.
- Tests prove excluded files never execute collection-time code in clean process.
- Tests prove unsafe direct mixed collection fails before imports.
- Leak detector passes without xfail.
- Stub and real cohorts pass without new failures or reduced test inventory.
- Local and Compose commands select same cohorts.
- Coverage combines all executed cohorts.
- Dedicated GPU acceptance runs on enigma after CPU checks pass.
- Review affected drift documents and check provenance.

Documentation: Makefile.test, docker-compose.test.yml, Dockerfile.test if runner packaging needs changes, docs/TESTING_IN_DOCKER.md.
Correct historical claim in docs/notes-ledge.md and add current boundary to project-forward-notes.md.

Alternatives:

- Rewrite all collection-time stubs as scoped fixtures. Greater migration scope and test-cache audit.
- pytest-xdist loadfile does not solve collection pollution. Every worker still collects full suite.

Task authority: human approved direct execution under STABL-sgdavnvz. No waveplan task required.

Additional smoke evidence:

Clean cohort, selecting leak detector and model detector: 1 passed, 1 skipped, 1422 deselected.
Leak detector skipped because diffusers had not been imported.
Current detector needs explicit real import to prevent this false assurance.
