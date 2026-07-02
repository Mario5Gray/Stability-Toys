# Testing in Docker

This repo has two distinct Docker test paths:

- Local/native path: CPU-first, intended for laptops and non-NVIDIA hosts.
- Explicit CUDA path: `linux/amd64` + NVIDIA-only, intended for CI or real GPU builders.

The shared test image name is `harbor.lan/dreamlab-test:latest`.

## Local vs CUDA Test Paths

Use the local/native path by default:

```bash
make test-build
make test
```

Use the explicit CUDA path only on `linux/amd64` hosts with NVIDIA runtime support:

```bash
make test-build-cuda
make test-cuda
```

On Apple Silicon, the local path is the only meaningful default. `test-cuda` forces `linux/amd64` and is meant for an x86_64 CUDA builder, not for day-to-day local validation on a Mac.

That local path verifies container wiring honestly, but it does not prove CPU inference support. Real image generation remains implemented only for `BACKEND=cuda` and `BACKEND=rknn` today.

## Compose Services

`docker-compose.test.yml` defines:

- `test`: local-native CPU service
- `test-unit`: local-native CPU service without websocket-marked tests
- `test-fast`: local-native CPU service with `not slow`
- `test-cuda`: explicit CUDA service for `linux/amd64`

The local services build `Dockerfile.test` with `BACKEND=cpu`. The CUDA service builds the same Dockerfile with `BACKEND=cuda` and loads `env.cuda`.

`BACKEND=cpu` in the local Docker test path is a build and smoke-test scaffold. It is not a supported generation backend for this app yet.

## Runtime Layout

The test container is intentionally close to runtime:

- [`Dockerfile.test`](/Users/darkbit1001/workspace/Stability-Toys/Dockerfile.test) installs the runtime/test dependencies and uses the same CUDA package flow as the main runtime image.
- [`docker-compose.test.yml`](/Users/darkbit1001/workspace/Stability-Toys/docker-compose.test.yml) mounts [`conf/modes-test.yml`](/Users/darkbit1001/workspace/Stability-Toys/conf/modes-test.yml) at `/conf/modes.yml`.
- The local CPU path installs pinned CPU PyTorch wheels before the generic requirements files so Linux arm64 builds do not silently pull a CUDA-heavy wheel set.

## Host Path Overrides

The test compose file uses test-specific host path overrides so it does not inherit the main runtime storage paths:

- `TEST_MODELS_HOST_PATH`
- `TEST_FS_HOST_PATH`
- `TEST_WORKFLOW_HOST_PATH`

If unset, the local test path defaults to repo-local directories:

- `./models`
- `./store`
- `./workflows`

Do not rely on `FS_HOST_PATH` or `WORKFLOW_HOST_PATH` for Docker tests. Those may point at non-portable runtime mounts and can break local Docker Desktop runs.

## Useful Commands

The top-level `Makefile` is the canonical entrypoint. It includes [`Makefile.test`](/Users/darkbit1001/workspace/Stability-Toys/Makefile.test) so you no longer need `make -f Makefile.test ...` for normal Docker test flows.

Run the local CPU image build:

```bash
make test-build
```

Run the local Docker test suite:

```bash
make test
```

Run one local Docker test file:

```bash
make test TEST=tests/test_cuda_worker_controlnet.py PYTEST_ARGS=-q
```

Build the explicit CUDA image:

```bash
make test-build-cuda
```

Run the explicit CUDA Docker test suite:

```bash
make test-cuda
```

Run one explicit CUDA Docker test file:

```bash
make test-cuda TEST=tests/test_cuda_worker_controlnet.py PYTEST_ARGS=-q
```

`TEST` defaults to `tests/`. `PYTEST_ARGS` defaults to the full verbose coverage arguments used by the historical suite targets. Override `PYTEST_ARGS` when you want a narrower invocation such as `-q` or `-x -q`.

Build the local CPU image:

```bash
docker compose -f docker-compose.test.yml build test
```

Run the local test suite:

```bash
docker compose -f docker-compose.test.yml run --rm test
```

Run local non-Docker tests via `Makefile.test` (requires Miniforge base):

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
make -f Makefile.test local-test
```

`local-test` and `local-test-coverage` now enforce `CONDA_PREFIX=$(HOME)/miniforge3` by default (override `EXPECTED_CONDA_PREFIX` if needed) and execute pytest via `python -m pytest` to avoid accidentally using system `python3`.

Run a smoke check that verifies the image starts and reads `/conf/modes.yml`:

```bash
docker compose -f docker-compose.test.yml run --rm test \
  python -c "import torch; from server.mode_config import get_mode_config; cfg = get_mode_config('/conf'); print(f'torch={torch.__version__} cuda={torch.version.cuda} default_mode={cfg.get_default_mode()}')"
```

Build the explicit CUDA image:

```bash
docker compose -f docker-compose.test.yml build test-cuda
```

Run the explicit CUDA suite:

```bash
docker compose -f docker-compose.test.yml run --rm test-cuda
```

## Remote GPU Dev Verification

The bind-mounted dev workflow in [`docker-compose.dev.yml`](../docker-compose.dev.yml) must run from a real repo tree on the Docker host. On a laptop, use the remote helper flow instead of trying to drive the bind mounts directly through Docker context alone.

Prepare or refresh the remote worktree and run the CUDA dev verification:

```bash
scripts/enigma-dev-verify.sh --branch <branch>
```

This wrapper:

- pushes the branch to the selected Git remote
- refreshes a branch worktree on the remote host
- runs `docker compose -f docker-cuda.yml build`
- runs `docker compose -f docker-compose.dev.yml up -d --build`
- waits for `lcm-sd-dev` to report a healthy Docker health status
- prints recent container logs
- prints the remaining manual `conf/modes.yaml` watcher check

Pass `--skip-base-build` after the first successful run if the base CUDA image is already present and you only need to re-run the fast dev-compose check.

The final `modes.yaml` edit is intentionally manual in v1. It keeps the remote config mutation explicit and reversible for the operator.

## Expected Local Warnings

A local smoke run may warn that:

- `/models/loras` does not exist
- the sample test models in `/models/diffusers/...` are not present

Those warnings are expected when the repo-local `./models` directory does not contain test assets. They do not mean the container wiring is broken.

## Troubleshooting

If the local test container fails to start on macOS with a mount error, check whether a test command is still inheriting non-local host paths. The resolved compose config should mount repo-local `models`, `store`, and `workflows` unless you explicitly set `TEST_*` overrides.

If the CUDA path fails on Apple Silicon, that is expected. Use the local CPU path on the laptop and reserve `test-cuda` for an amd64/NVIDIA environment.
