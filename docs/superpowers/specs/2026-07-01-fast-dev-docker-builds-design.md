# Fast Dev Docker Builds Design

## Goal

Make Docker builds fast for iterative development while keeping production
builds unchanged. Dev builds should handle both source and dependency changes
without multi-minute rebuilds, and source-only changes should be near-instant.

## Scope

In scope:

- fix `docker-cuda.yml` and `docker-rknn.yml` to pass `BACKEND` as a build arg
- add `docker-compose.dev.yml` overlay for fast iteration with volume mounts
- add `make dev` and `make dev-build` Makefile targets
- add contract tests for the compose and Makefile changes

Out of scope:

- changes to the production `Dockerfile` layer structure (it already orders
  deps before source; production builds stay as-is)
- changes to `Dockerfile.quick` (it serves the source-only thin-overlay path)
- changes to `docker-compose.test.yml` or `docker-compose.live-test.yml`
- CI pipeline changes (CI uses the split `docker/platform/` + `docker/runtime/`
  architecture, which is separate from the root `Dockerfile` compose flow)

## Existing Context

The root `Dockerfile` is a monolithic build that installs apt packages, CUDA
repos, pip deps, torch+xformers (multi-GB), builds the UI, then copies source.
It uses `ARG BACKEND` to conditionally install backend-specific deps.

`docker-cuda.yml` and `docker-rknn.yml` reference the root `Dockerfile` but do
not pass `BACKEND` as a build arg — the operator must pass it manually via
`--build-arg BACKEND=cuda`. Without it, the CUDA/torch/xformers layers are
skipped, producing a broken image.

`Dockerfile.quick` (already added) provides a thin source-only overlay onto
an existing image — seconds to build, but cannot handle dependency changes.

`docker-compose.live-test.yml` already demonstrates the volume-mount pattern
for hot-reload: it mounts `server/`, `backends/`, `utils/`, `persistence/`,
`invokers/`, and `tests/` as read-only volumes so source changes take effect
on container restart without any rebuild.

`docker-compose.test.yml` also mounts source volumes for the test container.

## Design

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

### Dev compose overlay

Create `docker-compose.dev.yml` — a standalone dev compose file (not an
overlay that requires `-f docker-cuda.yml -f docker-compose.dev.yml`) that
uses the root `Dockerfile` for the base image and volume-mounts source for
hot-reload.

Design decisions:

- **Uses the root `Dockerfile`, not `Dockerfile.quick`** — the root Dockerfile
  handles dependency changes through layer caching. `Dockerfile.quick` is
  source-only and cannot install deps.
- **Volume-mounts Python source** — `server/`, `backends/`, `utils/`,
  `persistence/`, `invokers/`, `conf/` are mounted read-write so changes take
  effect on container restart without any rebuild.
- **Mounts pre-built UI dist** — the UI is not rebuilt inside the container.
  The operator builds it locally (`cd lcm-sr-ui && yarn build`) and the
  compose file mounts `lcm-sr-ui/dist/` read-only. This skips the entire
  `ui-build` stage during dev builds.
- **Passes `BACKEND` as a build arg** — defaults to `cuda` for the CUDA dev
  path. Can be overridden with `BACKEND=rknn` for RKNN dev.
- **Uses `env.dev` + `env.cuda`** — dev env files, not `env.prod`.
- **No `privileged: true`** — dev doesn't need it unless testing GPU access.
  The CUDA compose already has `runtime: nvidia` and device reservations.
- **Image tag** — `lcm-sd-ui:dev` to keep it separate from the production tag.

```yaml
services:
  lcm-sd:
    platform: linux/amd64
    container_name: lcm-sd-dev
    runtime: nvidia
    build:
      context: .
      dockerfile: ./Dockerfile
      args:
        BACKEND: ${BACKEND:-cuda}
        GIT_SHA: ${GIT_SHA:-dev}
    image: lcm-sd-ui:dev
    ports:
      - "4200:4200"
    volumes:
      - ${MODELS_HOST_PATH:-./model}:/models:rw,Z
      - ${FS_HOST_PATH:-./store}:/store:rw,Z
      - ${WORKFLOW_HOST_PATH:-./workflows}:/workflows:rw,Z
      - ./conf:/app/conf:rw,Z
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
dev: ## Start dev container with volume-mounted source (hot-reload on restart)
	docker compose -f docker-compose.dev.yml up

.PHONY: dev-build
dev-build: ## Rebuild dev image (picks up dep changes via layer cache)
	docker compose -f docker-compose.dev.yml build

.PHONY: dev-down
dev-down: ## Stop dev container
	docker compose -f docker-compose.dev.yml down
```

### Three-speed build matrix

| Speed | Command | When to use | Handles dep changes? | Build time |
|---|---|---|---|---|
| Hot-reload | `make dev` | Active dev, source changes | No rebuild needed — volume mounts | Instant (restart only) |
| Dev rebuild | `make dev-build && make dev` | Dep change or clean image | Yes — root Dockerfile layer cache | Seconds (cache hit) to minutes (cache miss) |
| Quick overlay | `make quick-build` | Source changed, want clean image without compose | No — source only | Seconds |
| Full prod build | `docker compose -f docker-cuda.yml build` | Production build | Yes — full Dockerfile | Minutes |

## Testing

Add contract tests to `tests/test_cuda_packaging_contract.py`:

- `test_docker_cuda_yml_passes_backend_cuda_build_arg` — verifies
  `docker-cuda.yml` contains `BACKEND: cuda` under `build.args`
- `test_docker_rknn_yml_passes_backend_rknn_build_arg` — verifies
  `docker-rknn.yml` contains `BACKEND: rknn` under `build.args`
- `test_dev_compose_uses_root_dockerfile` — verifies
  `docker-compose.dev.yml` references `./Dockerfile` (not `Dockerfile.quick`)
- `test_dev_compose_mounts_python_source_volumes` — verifies the dev compose
  mounts `server/`, `backends/`, `utils/`, `persistence/`, `invokers/` as
  volumes
- `test_dev_compose_mounts_prebuilt_ui_dist` — verifies `lcm-sr-ui/dist` is
  mounted read-only (no in-container UI build)
- `test_dev_compose_uses_dev_env_files` — verifies `env.dev` is loaded
- `test_dev_compose_uses_dev_image_tag` — verifies image is tagged `:dev`,
  not the production tag
- `test_makefile_dev_target_uses_dev_compose` — verifies `make dev` runs
  `docker compose -f docker-compose.dev.yml up`
- `test_makefile_dev_build_target_uses_dev_compose` — verifies `make dev-build`
  runs `docker compose -f docker-compose.dev.yml build`

Tests should verify file content and Makefile dry-run output, not actual
Docker builds.

## Files

Create:

- `docker-compose.dev.yml`
- (tests are added to existing file)

Modify:

- `docker-cuda.yml` — add `BACKEND: cuda` build arg
- `docker-rknn.yml` — add `BACKEND: rknn` build arg
- `Makefile` — add `dev`, `dev-build`, `dev-down` targets
- `tests/test_cuda_packaging_contract.py` — add contract tests

## Risks and Constraints

- The dev compose requires the `observ-net` network to exist (same as
  production compose). If it doesn't exist, `docker compose up` will fail.
  This is the same constraint as the existing compose files — not a new risk.
- The dev compose requires `lcm-sr-ui/dist/` to exist (pre-built UI). If it
  doesn't, the UI won't be served. The operator must run
  `cd lcm-sr-ui && yarn build` first. This is documented in the live-test
  compose already and should be noted in the dev compose comments.
- Volume-mounting source means the container sees the host filesystem state.
  This is intentional for dev but would be wrong for production. The dev
  compose uses a separate `:dev` image tag to prevent accidental production
  use.
- The `BACKEND` build arg fix in `docker-cuda.yml` / `docker-rknn.yml` is a
  behavior change — previously the arg was absent (empty), now it's explicit.
  If any workflow relied on the arg being empty, it would break. However, an
  empty `BACKEND` produced a broken image (no torch), so no correct workflow
  could have relied on it.

## Acceptance

This design is complete when:

- `docker compose -f docker-cuda.yml build` produces a working CUDA image
  without manual `--build-arg BACKEND=cuda`
- `make dev` starts a dev container with volume-mounted source in seconds
- `make dev-build` rebuilds the dev image, picking up `requirements.txt`
  changes via layer cache
- Source-only changes require no rebuild — just `docker compose restart`
- All contract tests pass
