# Testing in Docker

This repo has two distinct Docker test paths:

- Local/native path: CPU-first, intended for laptops and non-NVIDIA hosts.
- Explicit CUDA path: `linux/amd64` + NVIDIA-only, intended for CI or real GPU builders.

The shared test image name is `harbor.lan/stability-toys:test`.

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

The local services build `Dockerfile.test` with `BACKEND=cpu`. The CUDA service
builds the same Dockerfile with `BACKEND=cuda`, loads `env.cuda`, and requests
all NVIDIA GPUs through the Compose device reservation.

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

The shared Miniforge root environment drifts — other projects install into it, and it can end up outside the project pins (transformers 5.x, an older diffusers), which aborts pytest collection. The container is the source of truth; for a matching **host** environment, run [`scripts/local-host.sh`](/Users/darkbit1001/workspace/Stability-Toys/scripts/local-host.sh). It detects OS/architecture and CUDA, asks the operator for anything it cannot infer (no CLI arguments), creates a dedicated env named `stability-toys`, and installs torch plus the requirements in the same order as the image. Point local pytest at that env rather than the shared root.

Run a smoke check that verifies the image starts and reads `/conf/modes.yml`:

```bash
docker compose -f docker-compose.test.yml run --rm test \
  python -c "import torch; from server.mode_config import get_mode_config; cfg = get_mode_config('/conf'); print(f'torch={torch.__version__} cuda={torch.version.cuda} default_mode={cfg.get_default_mode()}')"
```

## Prompt Conditioning Checks

Prompt-conditioning dependency checks are deterministic and split by container
path.

Build the local/native test image and verify the isolated Compel pin imports:

```bash
docker compose -f docker-compose.test.yml build test
docker compose -f docker-compose.test.yml run --rm test \
  python -c "from importlib.metadata import version; import compel; print(version('compel'))"
```

Run the local/native conditioning package slice:

```bash
docker compose -f docker-compose.test.yml run --rm test \
  python -m pytest tests/test_conditioning_contracts.py \
    tests/test_conditioning_registry.py tests/test_conditioning_compel.py \
    tests/test_compel_packaging.py -q
```

Build the explicit CUDA image:

```bash
docker compose -f docker-compose.test.yml build test-cuda
```

Run the explicit CUDA suite:

```bash
docker compose -f docker-compose.test.yml run --rm test-cuda
```

Run the explicit CUDA prompt-conditioning and CUDA-consumer slice:

```bash
docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
  python -m pytest tests/test_conditioning_compel.py \
    tests/test_cuda_worker_capabilities.py \
    tests/test_cuda_worker_controlnet.py -q
```

Production and test image package inspection must show `compel==2.3.1`. The
Compel installation must not introduce Notebook or Jupyter packages; the images
install the dedicated conditioning requirements with `--no-deps`, while Torch,
Diffusers, Transformers, and pyparsing remain under repository requirements.

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
- waits for `stability-toys-dev` to report a healthy Docker health status
- prints recent container logs
- prints the remaining manual `conf/modes.yaml` watcher check

Pass `--skip-base-build` after the first successful run if the base CUDA image is already present and you only need to re-run the fast dev-compose check.

The final `modes.yaml` edit is intentionally manual in v1. It keeps the remote config mutation explicit and reversible for the operator.

Use two terminals for that final watcher check:

```bash
# Terminal A
ssh <host>
cd <remote-worktree>
docker logs -f stability-toys-dev
```

Leave Terminal A running, then make the reversible config edit separately:

```bash
# Terminal B
ssh <host>
cd <remote-worktree>
$EDITOR conf/modes.yaml
```

Confirm Terminal A prints the watcher reload without restarting `stability-toys-dev`.

## Expected Local Warnings

A local smoke run may warn that:

- `/models/loras` does not exist
- the sample test models in `/models/diffusers/...` are not present

Those warnings are expected when the repo-local `./models` directory does not contain test assets. They do not mean the container wiring is broken.

## Troubleshooting

If the local test container fails to start on macOS with a mount error, check whether a test command is still inheriting non-local host paths. The resolved compose config should mount repo-local `models`, `store`, and `workflows` unless you explicitly set `TEST_*` overrides.

If the CUDA path fails on Apple Silicon, that is expected. Use the local CPU path on the laptop and reserve `test-cuda` for an amd64/NVIDIA environment.

### `PermissionError` on `/app/tests` under SELinux

On an SELinux host (`getenforce` returns `Enforcing`), a failure like

```text
PermissionError: [Errno 13] Permission denied: '/app/tests/pytest.toml'
```

is a mount-label problem, not a file-permission problem. Two tells: `pytest.toml` does not exist in this repo — pytest probes that name first when locating config, and probing a missing name in a readable directory returns `ENOENT`, not `EACCES` — and the test image sets no `USER`, so the container is already root.

The bind mounts use lowercase `:z`, the **shared** SELinux label, because every service in `docker-compose.test.yml` extends `test` and therefore mounts the same host paths. Uppercase `:Z` applies a **private** label with a unique MCS category per container; with `compose up` starting six services at once, each relabels the same directory and invalidates the others, so all but one lose access. Keep these mounts on `:z`.

Running a single service with `run --rm test` never exercises this, which is why the failure only appears under `up`.
