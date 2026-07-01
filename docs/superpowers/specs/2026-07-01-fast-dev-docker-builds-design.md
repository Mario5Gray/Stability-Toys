# Fast Dev Docker Builds Design

## Goal

Make Docker builds fast for iterative development while keeping production
builds unchanged. Dev builds should handle both source and dependency changes
without multi-minute rebuilds, and source-only changes should be near-instant.

## Scope

In scope:

- fix `docker-cuda.yml` and `docker-rknn.yml` to pass `BACKEND` as a build arg
- fix `docker/runtime/live-test.Dockerfile` startup command (broken module
  path — see "Fix live-test Dockerfile" below)
- add `docker-compose.dev.yml` for fast CUDA iteration with volume mounts
- use `docker/runtime/live-test.Dockerfile` as the dev build entrypoint (no UI
  build, no deps install — source-only overlay onto a pre-built base image)
- add `make dev` and `make dev-build` Makefile targets
- add contract tests for the compose, Dockerfile, and Makefile changes

Out of scope:

- changes to the production `Dockerfile` layer structure
- changes to `Dockerfile.quick` (it serves the source-only thin-overlay path)
- changes to `docker-compose.test.yml` or `docker-compose.live-test.yml`
- CI pipeline changes
- RKNN dev compose (see "Backend scope" below)

## Existing Context

The repo has a split build architecture:

- **Platform base images** (`docker/platform/python-cuda.Dockerfile`,
  `docker/platform/python-common.Dockerfile`) — install apt, CUDA, pip deps,
  torch+xformers. These are the heavy multi-GB layers that rarely change.
- **Runtime Dockerfiles** (`docker/runtime/app.Dockerfile` for production,
  `docker/runtime/live-test.Dockerfile` for dev) — `FROM ${BASE_IMAGE}`,
  copy source, no deps install. These are thin layers.
- **Root `Dockerfile`** — a compatibility entrypoint that combines everything
  (platform + runtime + UI build) in one monolithic file. The comment on line
  1 says "CI should use docker/runtime/app.Dockerfile."

The root `Dockerfile` always executes the `ui-build` stage (line 11) and
copies from it (line 150), even when the operator only wants to update Python
source. A bind mount can replace the runtime UI files after the container
starts, but it does not avoid paying the UI build cost during
`docker compose build`.

`docker/runtime/live-test.Dockerfile` is the right dev entrypoint: it
`FROM ${BASE_IMAGE}`, copies only Python source (no UI build, no deps
install), and uses `uvicorn --reload` for automatic source-reload. However,
its CMD has a broken module path that must be fixed before it can serve as
the dev entrypoint (see below).

`docker-cuda.yml` and `docker-rknn.yml` reference the root `Dockerfile` but
do not pass `BACKEND` as a build arg — the operator must pass it manually via
`--build-arg BACKEND=cuda`. Without it, the CUDA/torch/xformers layers are
skipped, producing a broken image.

`Dockerfile.quick` (already added) provides a thin source-only overlay onto
an existing image — seconds to build, but cannot handle dependency changes.

`docker-compose.live-test.yml` already demonstrates the volume-mount pattern
for hot-reload: it mounts `server/`, `backends/`, `utils/`, `persistence/`,
`invokers/`, and `tests/` as read-only volumes so source changes take effect
without any rebuild.

## Design

### Backend scope: CUDA only

The dev compose is CUDA-only. It hardcodes `platform: linux/amd64`,
`runtime: nvidia`, `env.cuda`, and NVIDIA device reservations. An RKNN dev
path would need `platform: linux/arm64`, `env.rknn`, `/dev/rknpu` devices,
and no NVIDIA runtime — a fundamentally different compose contract. RKNN dev
is out of scope for this spec; it can be added later as a separate
`docker-compose.dev-rknn.yml` if needed.

### Fix compose build args

`docker-cuda.yml` and `docker-rknn.yml` must pass `BACKEND` as a build arg so
`docker compose build` produces a correct image without manual flags.

`docker-cuda.yml`:
```yaml
build:
  args:
    BACKEND: cuda
    GIT_SHA: ${GIT_SHA:-dev}
```

`docker-rknn.yml`:
```yaml
build:
  args:
    BACKEND: rknn
    GIT_SHA: ${GIT_SHA:-dev}
```

### Fix live-test Dockerfile

The live-test Dockerfile's CMD is broken:

```dockerfile
# Current (broken):
CMD ["uvicorn", "lcm_sr_server:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
```

The module lives at `server/lcm_sr_server.py` and must be imported as
`server.lcm_sr_server` (confirmed: `import lcm_sr_server` fails with
`ModuleNotFoundError`; `import server.lcm_sr_server` succeeds). The
production launcher (`start.sh`) uses `python -m server.run`, which imports
`from server.lcm_sr_server import app`.

Fix: change the bare module name to the qualified package path:

```dockerfile
CMD ["uvicorn", "server.lcm_sr_server:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
```

This is a prerequisite defect — the dev compose depends on this Dockerfile
booting correctly.

### Reload semantics

The live-test Dockerfile uses `uvicorn --reload`, which watches mounted
Python source files and automatically reloads the server on changes. This
means:

- **Python source edits** (`.py` files in mounted `server/`, `backends/`,
  `utils/`, `persistence/`, `invokers/`) — picked up automatically by
  `--reload`. No restart, no rebuild needed.
- **`modes.yaml` edits** under the mounted `./conf` at `/conf` — hot-reloaded
  in-process by the app's own watchdog file watcher
  (`start_config_watcher(MODE_CONFIG_PATH, reload_mode_config)` in
  `server/lcm_sr_server.py`). No restart needed.
- **Other changes** (env vars, base-image swaps, model-root changes that live
  outside the config watcher) — require
  `docker compose -f docker-compose.dev.yml restart`.

The spec's earlier drafts incorrectly described source edits as requiring a
restart. That was wrong — `--reload` handles source edits automatically.

### Dev compose

Create `docker-compose.dev.yml` — a standalone CUDA dev compose file that
uses `docker/runtime/live-test.Dockerfile` as the build entrypoint and
volume-mounts source for `--reload` hot-reload.

Design decisions:

- **Uses `docker/runtime/live-test.Dockerfile`, not the root `Dockerfile`** —
  the root Dockerfile always runs the `ui-build` stage (line 11) and copies
  from it (line 150), even when only Python source changed. The live-test
  Dockerfile skips the UI build entirely — it `FROM ${BASE_IMAGE}`, copies
  only Python source, and is done. This is the critical performance win: dev
  builds never pay the yarn install + Vite build cost.
- **Requires a pre-built base image** — the live-test Dockerfile takes
  `BASE_IMAGE` as a build arg. The dev compose defaults to
  `harbor.lan/lcm-sd-ui:latest` (the production image, which already has all
  deps installed). This means the heavy platform layers (CUDA, torch,
  xformers, pip deps) are baked into the base image and never reinstalled
  during dev builds.
- **Handles dep changes** — when `requirements.txt` changes, the operator
  rebuilds the base image once (`docker compose -f docker-cuda.yml build`),
  then dev builds pick up the new base. The dev compose itself never
  reinstalls pip deps — it only overlays source.
- **Volume-mounts Python source** — `server/`, `backends/`, `utils/`,
  `persistence/`, `invokers/` are mounted read-write so `uvicorn --reload`
  picks up `.py` edits automatically without restart or rebuild.
- **Mounts config at `/conf`, not `/app/conf`** — `env.dev` sets
  `MODE_CONFIG_PATH=/conf` (line 27), and the production compose files mount
  `./conf` at `/conf` (not `/app/conf`). The dev compose follows the same
  convention so config is read from the host-mounted directory. Config
  changes require a container restart (not `--reload`).
- **Mounts pre-built UI dist** — the operator builds it locally
  (`cd lcm-sr-ui && yarn build`) and the compose file mounts
  `lcm-sr-ui/dist/` read-only at `/opt/lcm-sr-server/ui-dist/`. This replaces
  the runtime files after the container starts.
- **Uses `env.dev` + `env.cuda`** — dev env files, not `env.prod`.
- **Image tag** — `lcm-sd-ui:dev` to keep it separate from the production tag.

```yaml
# docker-compose.dev.yml — fast CUDA dev builds with volume-mounted source.
#
# Prerequisite: base image must exist (built once via full compose build):
#   docker compose -f docker-cuda.yml build
#
# Prerequisite: pre-built UI dist (for serving the frontend):
#   cd lcm-sr-ui && yarn build
#
# Usage:
#   make dev          # start dev container (uvicorn --reload on source changes)
#   make dev-build    # rebuild dev image (source-only, seconds)
#   make dev-down     # stop dev container
#
# Reload behavior:
#   - Python source edits (.py) → auto-reloaded by uvicorn --reload
#   - Config/env/model changes → docker compose -f docker-compose.dev.yml restart

services:
  lcm-sd:
    platform: linux/amd64
    container_name: lcm-sd-dev
    runtime: nvidia
    build:
      context: .
      dockerfile: docker/runtime/live-test.Dockerfile
      args:
        BASE_IMAGE: ${BASE_IMAGE:-harbor.lan/lcm-sd-ui:latest}
        GIT_SHA: ${GIT_SHA:-dev}
    image: lcm-sd-ui:dev
    ports:
      - "4200:4200"
    volumes:
      - ${MODELS_HOST_PATH:-./model}:/models:rw,Z
      - ${FS_HOST_PATH:-./store}:/store:rw,Z
      - ${WORKFLOW_HOST_PATH:-./workflows}:/workflows:rw,Z
      - ./conf:/conf:rw,Z
      - ./server:/app/server:rw,Z
      - ./backends:/app/backends:rw,Z
      - ./utils:/app/utils:rw,Z
      - ./persistence:/app/persistence:rw,Z
      - ./invokers:/app/invokers:rw,Z
      - ./lcm-sr-ui/dist:/opt/lcm-sr-server/ui-dist:ro,Z
    env_file:
      - env.dev
      - env.cuda
    restart: unless-stopped
    networks:
      - observ-net
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:4200/docs').read()\""]
      interval: 30s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  observ-net:
    name: observ-net
    external: true
```

### Makefile targets

```makefile
.PHONY: dev
dev: ## Start dev container (uvicorn --reload on Python source changes)
	docker compose -f docker-compose.dev.yml up

.PHONY: dev-build
dev-build: ## Rebuild dev image (source-only overlay, seconds)
	docker compose -f docker-compose.dev.yml build

.PHONY: dev-down
dev-down: ## Stop dev container
	docker compose -f docker-compose.dev.yml down
```

### Build matrix

| Speed | Command | When to use | Handles dep changes? | Build time |
|---|---|---|---|---|
| Auto-reload | `make dev` | Active dev, `.py` edits | No rebuild — `--reload` watches mounted source | Instant (auto-reload) |
| Dev rebuild | `make dev-build && make dev` | Source changed, want clean image | No — source only | Seconds |
| Quick overlay | `make quick-build` | Source changed, want clean image without compose | No — source only | Seconds |
| Full prod build | `docker compose -f docker-cuda.yml build` | Deps changed, production build | Yes — full Dockerfile | Minutes |

When `requirements.txt` changes: rebuild the base image once with
`docker compose -f docker-cuda.yml build`, then resume dev iteration with
`make dev`. The dev compose picks up the new base image automatically.

## Testing

Add contract tests to `tests/test_cuda_packaging_contract.py`:

- `test_docker_cuda_yml_passes_backend_cuda_build_arg` — verifies
  `docker-cuda.yml` contains `BACKEND: cuda` under `build.args`
- `test_docker_rknn_yml_passes_backend_rknn_build_arg` — verifies
  `docker-rknn.yml` contains `BACKEND: rknn` under `build.args`
- `test_live_test_dockerfile_uses_qualified_module_path` — verifies the
  CMD uses `server.lcm_sr_server:app`, not the bare `lcm_sr_server:app`
- `test_dev_compose_uses_live_test_dockerfile` — verifies
  `docker-compose.dev.yml` references `docker/runtime/live-test.Dockerfile`,
  not the root `Dockerfile` (so the UI build stage is skipped)
- `test_dev_compose_mounts_config_at_conf_not_app_conf` — verifies `./conf`
  is mounted at `/conf` (matching `MODE_CONFIG_PATH=/conf` in `env.dev`),
  not at `/app/conf`
- `test_dev_compose_mounts_python_source_volumes` — verifies the dev compose
  mounts `server/`, `backends/`, `utils/`, `persistence/`, `invokers/` as
  volumes
- `test_dev_compose_mounts_prebuilt_ui_dist` — verifies `lcm-sr-ui/dist` is
  mounted read-only (no in-container UI build)
- `test_dev_compose_uses_dev_env_files` — verifies `env.dev` is loaded
- `test_dev_compose_uses_dev_image_tag` — verifies image is tagged `:dev`,
  not the production tag
- `test_dev_compose_takes_base_image_build_arg` — verifies `BASE_IMAGE` is
  a build arg with a default pointing at the production image
- `test_makefile_dev_target_uses_dev_compose` — verifies `make dev` runs
  `docker compose -f docker-compose.dev.yml up`
- `test_makefile_dev_build_target_uses_dev_compose` — verifies `make dev-build`
  runs `docker compose -f docker-compose.dev.yml build`

Tests should verify file content and Makefile dry-run output, not actual
Docker builds.

## Files

Create:

- `docker-compose.dev.yml`

Modify:

- `docker-cuda.yml` — add `BACKEND: cuda` build arg
- `docker-rknn.yml` — add `BACKEND: rknn` build arg
- `docker/runtime/live-test.Dockerfile` — fix CMD module path from
  `lcm_sr_server:app` to `server.lcm_sr_server:app`
- `Makefile` — add `dev`, `dev-build`, `dev-down` targets
- `tests/test_cuda_packaging_contract.py` — add contract tests

## Risks and Constraints

- The dev compose requires a pre-built base image
  (`harbor.lan/lcm-sd-ui:latest`). If it doesn't exist locally, the first
  `make dev-build` will fail. The operator must run a full
  `docker compose -f docker-cuda.yml build` first. This is documented in the
  compose file comments.
- The dev compose requires `lcm-sr-ui/dist/` to exist (pre-built UI). If it
  doesn't, the UI won't be served. The operator must run
  `cd lcm-sr-ui && yarn build` first. This is documented in the compose file
  comments and is the same constraint as `docker-compose.live-test.yml`.
- The dev compose requires the `observ-net` network to exist (same as
  production compose). If it doesn't exist, `docker compose up` will fail.
  This is the same constraint as the existing compose files — not a new risk.
- Volume-mounting source means the container sees the host filesystem state.
  This is intentional for dev but would be wrong for production. The dev
  compose uses a separate `:dev` image tag to prevent accidental production
  use.
- The `BACKEND` build arg fix in `docker-cuda.yml` / `docker-rknn.yml` is a
  behavior change — previously the arg was absent (empty), now it's explicit.
  If any workflow relied on the arg being empty, it would break. However, an
  empty `BACKEND` produced a broken image (no torch), so no correct workflow
  could have relied on it.
- The dev compose is CUDA-only. RKNN dev requires a different platform,
  runtime, and device set. An RKNN dev compose can be added later as a
  separate file.
- The live-test Dockerfile fix (module path) is a behavior change for
  `docker-compose.live-test.yml` as well, since it uses the same Dockerfile.
  The fix is strictly correct — the bare module name never worked — so any
  existing usage of the live-test compose must have been relying on a
  different CMD override or was broken.

## Acceptance

This design is complete when:

- `docker compose -f docker-cuda.yml build` produces a working CUDA image
  without manual `--build-arg BACKEND=cuda`
- `make dev` starts a dev container with volume-mounted source in seconds
- `make dev-build` rebuilds the dev image in seconds (no UI build, no deps
  install — source-only overlay onto the pre-built base image)
- Python source edits (`.py` files) are auto-reloaded by `uvicorn --reload`
  — no restart or rebuild needed
- `modes.yaml` edits under `./conf` are hot-reloaded in-process by the app's
  watchdog file watcher — no restart needed
- Other changes (env vars, base-image swaps, model-root changes outside the
  config watcher) require `docker compose -f docker-compose.dev.yml restart`
- Mode config is read from the host-mounted `./conf` at `/conf`, matching
  `MODE_CONFIG_PATH=/conf`
- The live-test Dockerfile CMD uses `server.lcm_sr_server:app` (qualified
  module path)
- All contract tests pass
