# STABL-ylxbzedr: test interpreter guard

Status: human approved. Base: main 8d5f356. Waveplan not in use.

Compare resolved sys.prefix with expected prefix. Do not trust CONDA_PREFIX as interpreter identity.
Default expected prefix: ~/miniforge3/envs/stability-toys.
ST_TEST_PYTHON_PREFIX overrides expected prefix for other hosts and CI.

Use one standard-library helper from tests.run and tests/conftest.py.
Runner checks before launching pytest. Conftest checks before NumPy, Pillow, and test-module imports.
Direct pytest can load plugins before conftest. This guard does not control that earlier pytest phase.
Mismatch returns usage error with actual interpreter, expected interpreter, and conda activation command.

Set ST_TEST_PYTHON_PREFIX=/usr/local in Dockerfile.test and docker-compose.test.yml.
Compose setting supports existing images when changed tests are mounted.
Preserve EXPECTED_CONDA_PREFIX override for local Make targets by forwarding it to shared check.

Alternatives:
- Check CONDA_PREFIX only: stale shell state can identify wrong interpreter as valid.
- ST_ALLOW_ANY_PYTHON=1 in containers: simpler bypass, but disables identity check entirely.
- Explicit expected prefix: selected approach. Keeps one check for host and container paths.

Tests use fresh subprocesses and collection sentinels.
Prove wrong prefix fails before imports and subprocess launch.
Prove valid prefix works without activation variables.
Prove stale CONDA_PREFIX cannot authorize wrong interpreter.
Prove configured container prefix works. Check actual test-container execution where available.
Check direct pytest and tests.run entrypoints. Keep collection-isolation regressions green.
Run dedicated-env focused tests and review affected drift documents.

This checks environment identity. It does not verify installed dependency versions.
