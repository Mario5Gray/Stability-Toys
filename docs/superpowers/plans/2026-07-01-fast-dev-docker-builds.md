# Fast Dev Docker Builds Implementation Plan

> **For agentic workers:** REQUIRED SKILL: `superpowers:executing-plans` (inline execution only). **Repo override:** `AGENTS.md` forbids subagent-driven development; do **not** use `superpowers:subagent-driven-development` for this plan. Execute each task inline in the current session with TDD red/green/commit per task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Docker dev builds fast by using the split runtime/platform architecture, fixing broken compose build args and the live-test Dockerfile CMD, and adding a dev compose with volume-mounted source for `uvicorn --reload`.

**Architecture:** The dev compose uses `docker/runtime/live-test.Dockerfile` (source-only overlay onto a pre-built base image — no UI build, no deps install). Python source is volume-mounted so `uvicorn --reload` picks up `.py` edits automatically. The production compose files (`docker-cuda.yml`, `docker-rknn.yml`) are fixed to pass `BACKEND` as a build arg so `docker compose build` works without manual `--build-arg`.

**Tech Stack:** Docker, Docker Compose, Make, pytest.

---

## File Structure

- Modify: `docker-cuda.yml`
  Purpose: add `BACKEND: cuda` build arg so compose build produces a working image without manual flags.
- Modify: `docker-rknn.yml`
  Purpose: add `BACKEND: rknn` build arg for the same reason.
- Modify: `docker/runtime/live-test.Dockerfile`
  Purpose: fix broken CMD module path from `lcm_sr_server:app` to `server.lcm_sr_server:app`.
- Create: `docker-compose.dev.yml`
  Purpose: standalone CUDA dev compose using live-test Dockerfile + volume-mounted source.
- Modify: `Makefile`
  Purpose: add `dev`, `dev-build`, `dev-down` targets.
- Modify: `tests/test_cuda_packaging_contract.py`
  Purpose: contract tests for all compose, Dockerfile, and Makefile changes.

### Task 1: Fix Compose Build Args and Live-Test Dockerfile CMD

**Files:**
- Modify: `docker-cuda.yml`
- Modify: `docker-rknn.yml`
- Modify: `docker/runtime/live-test.Dockerfile`
- Modify: `tests/test_cuda_packaging_contract.py`

- [ ] **Step 1: Write the failing contract tests**

Append to `tests/test_cuda_packaging_contract.py`:

```python
def test_docker_cuda_yml_passes_backend_cuda_build_arg():
    import yaml

    compose = yaml.safe_load(
        (REPO_ROOT / "docker-cuda.yml").read_text(encoding="utf-8")
    )
    svc = compose["services"]["lcm-sd"]
    args = svc["build"]["args"]

    assert args.get("BACKEND") == "cuda"


def test_docker_rknn_yml_passes_backend_rknn_build_arg():
    import yaml

    compose = yaml.safe_load(
        (REPO_ROOT / "docker-rknn.yml").read_text(encoding="utf-8")
    )
    svc = compose["services"]["lcm-sd"]
    args = svc["build"]["args"]

    assert args.get("BACKEND") == "rknn"


def test_live_test_dockerfile_uses_qualified_module_path():
    text = (REPO_ROOT / "docker/runtime/live-test.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "server.lcm_sr_server:app" in text
    # The bare module name must not appear — it never worked
    bare_cmd = '"uvicorn", "lcm_sr_server:app"'
    assert bare_cmd not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py::test_docker_cuda_yml_passes_backend_cuda_build_arg tests/test_cuda_packaging_contract.py::test_docker_rknn_yml_passes_backend_rknn_build_arg tests/test_cuda_packaging_contract.py::test_live_test_dockerfile_uses_qualified_module_path -v --no-cov
```

Expected: FAIL because `docker-cuda.yml` and `docker-rknn.yml` don't have `BACKEND` build args, and the live-test Dockerfile uses the bare `lcm_sr_server:app`.

- [ ] **Step 3: Fix `docker-cuda.yml` — add `BACKEND: cuda` build arg**

In `docker-cuda.yml`, change the `build.args` block from:

```yaml
    build:
      context: .
      dockerfile: ./Dockerfile
      args:
        GIT_SHA: ${GIT_SHA:-dev}
```

to:

```yaml
    build:
      context: .
      dockerfile: ./Dockerfile
      args:
        BACKEND: cuda
        GIT_SHA: ${GIT_SHA:-dev}
```

- [ ] **Step 4: Fix `docker-rknn.yml` — add `BACKEND: rknn` build arg**

In `docker-rknn.yml`, change the `build.args` block from:

```yaml
    build:
      context: .
      dockerfile: ./Dockerfile
      args:
        GIT_SHA: ${GIT_SHA:-dev}
```

to:

```yaml
    build:
      context: .
      dockerfile: ./Dockerfile
      args:
        BACKEND: rknn
        GIT_SHA: ${GIT_SHA:-dev}
```

- [ ] **Step 5: Fix `docker/runtime/live-test.Dockerfile` — fix CMD module path**

In `docker/runtime/live-test.Dockerfile`, change the last line from:

```dockerfile
CMD ["uvicorn", "lcm_sr_server:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
```

to:

```dockerfile
CMD ["uvicorn", "server.lcm_sr_server:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py::test_docker_cuda_yml_passes_backend_cuda_build_arg tests/test_cuda_packaging_contract.py::test_docker_rknn_yml_passes_backend_rknn_build_arg tests/test_cuda_packaging_contract.py::test_live_test_dockerfile_uses_qualified_module_path -v --no-cov
```

Expected: PASS

- [ ] **Step 7: Run the full packaging contract suite to check for regressions**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py -q --no-cov
```

Expected: PASS (all existing + new tests)

- [ ] **Step 8: Commit**

```bash
git add docker-cuda.yml docker-rknn.yml docker/runtime/live-test.Dockerfile tests/test_cuda_packaging_contract.py
git commit -m "fix(docker): pass BACKEND build arg in compose, fix live-test Dockerfile module path"
```

### Task 2: Create the Dev Compose File

**Files:**
- Create: `docker-compose.dev.yml`
- Modify: `tests/test_cuda_packaging_contract.py`

- [ ] **Step 1: Write the failing contract tests**

Append to `tests/test_cuda_packaging_contract.py`:

```python
def test_dev_compose_uses_live_test_dockerfile():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "docker/runtime/live-test.Dockerfile" in text
    # Must NOT use the root Dockerfile (which always runs the UI build stage)
    assert "dockerfile: ./Dockerfile" not in text


def test_dev_compose_mounts_config_at_conf_not_app_conf():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "./conf:/conf" in text
    assert "./conf:/app/conf" not in text


def test_dev_compose_mounts_python_source_volumes():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    for src in ["./server:/app/server", "./backends:/app/backends",
                "./utils:/app/utils", "./persistence:/app/persistence",
                "./invokers:/app/invokers"]:
        assert src in text, f"dev compose must mount {src}"


def test_dev_compose_mounts_prebuilt_ui_dist():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "lcm-sr-ui/dist:/opt/lcm-sr-server/ui-dist" in text
    assert ":ro" in text


def test_dev_compose_uses_dev_env_files():
    import yaml

    compose = yaml.safe_load(
        (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    svc = compose["services"]["lcm-sd"]
    env_files = svc.get("env_file", [])

    assert "env.dev" in env_files
    assert "env.cuda" in env_files


def test_dev_compose_uses_dev_image_tag():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "lcm-sd-ui:dev" in text
    # Must NOT use the production tag
    assert "lcm-sd-ui:latest" not in text.split("image:")[1].split("\n")[0] if "image:" in text else True


def test_dev_compose_takes_base_image_build_arg():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "BASE_IMAGE" in text
    assert "harbor.lan/lcm-sd-ui:latest" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py::test_dev_compose_uses_live_test_dockerfile tests/test_cuda_packaging_contract.py::test_dev_compose_mounts_config_at_conf_not_app_conf tests/test_cuda_packaging_contract.py::test_dev_compose_mounts_python_source_volumes tests/test_cuda_packaging_contract.py::test_dev_compose_mounts_prebuilt_ui_dist tests/test_cuda_packaging_contract.py::test_dev_compose_uses_dev_env_files tests/test_cuda_packaging_contract.py::test_dev_compose_uses_dev_image_tag tests/test_cuda_packaging_contract.py::test_dev_compose_takes_base_image_build_arg -v --no-cov
```

Expected: FAIL because `docker-compose.dev.yml` does not exist yet.

- [ ] **Step 3: Create `docker-compose.dev.yml`**

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
#   - modes.yaml edits under ./conf → hot-reloaded by app watchdog watcher
#   - env/base-image/model-root changes → docker compose -f docker-compose.dev.yml restart

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

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py::test_dev_compose_uses_live_test_dockerfile tests/test_cuda_packaging_contract.py::test_dev_compose_mounts_config_at_conf_not_app_conf tests/test_cuda_packaging_contract.py::test_dev_compose_mounts_python_source_volumes tests/test_cuda_packaging_contract.py::test_dev_compose_mounts_prebuilt_ui_dist tests/test_cuda_packaging_contract.py::test_dev_compose_uses_dev_env_files tests/test_cuda_packaging_contract.py::test_dev_compose_uses_dev_image_tag tests/test_cuda_packaging_contract.py::test_dev_compose_takes_base_image_build_arg -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docker-compose.dev.yml tests/test_cuda_packaging_contract.py
git commit -m "feat(docker): add docker-compose.dev.yml for fast CUDA dev builds"
```

### Task 3: Add Makefile Dev Targets

**Files:**
- Modify: `Makefile`
- Modify: `tests/test_cuda_packaging_contract.py`

- [ ] **Step 1: Write the failing contract tests**

Append to `tests/test_cuda_packaging_contract.py`:

```python
def test_makefile_dev_target_uses_dev_compose():
    result = _make_dry_run("dev")

    assert result.returncode == 0, result.stderr
    assert "docker compose -f docker-compose.dev.yml up" in result.stdout


def test_makefile_dev_build_target_uses_dev_compose():
    result = _make_dry_run("dev-build")

    assert result.returncode == 0, result.stderr
    assert "docker compose -f docker-compose.dev.yml build" in result.stdout


def test_makefile_dev_down_target_uses_dev_compose():
    result = _make_dry_run("dev-down")

    assert result.returncode == 0, result.stderr
    assert "docker compose -f docker-compose.dev.yml down" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py::test_makefile_dev_target_uses_dev_compose tests/test_cuda_packaging_contract.py::test_makefile_dev_build_target_uses_dev_compose tests/test_cuda_packaging_contract.py::test_makefile_dev_down_target_uses_dev_compose -v --no-cov
```

Expected: FAIL because the Makefile doesn't have `dev`, `dev-build`, or `dev-down` targets.

- [ ] **Step 3: Add dev targets to the Makefile**

Append to the end of `Makefile`:

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

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py::test_makefile_dev_target_uses_dev_compose tests/test_cuda_packaging_contract.py::test_makefile_dev_build_target_uses_dev_compose tests/test_cuda_packaging_contract.py::test_makefile_dev_down_target_uses_dev_compose -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Run the full packaging contract suite**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py -q --no-cov
```

Expected: PASS (all existing + new tests)

- [ ] **Step 6: Commit**

```bash
git add Makefile tests/test_cuda_packaging_contract.py
git commit -m "feat(makefile): add dev, dev-build, dev-down targets for fast dev builds"
```

## Self-Review

- Spec coverage:
  - fix `docker-cuda.yml` and `docker-rknn.yml` to pass `BACKEND` as a build arg: Task 1
  - fix `docker/runtime/live-test.Dockerfile` startup command: Task 1
  - add `docker-compose.dev.yml`: Task 2
  - use `docker/runtime/live-test.Dockerfile` as dev entrypoint: Task 2
  - add `make dev` and `make dev-build` Makefile targets: Task 3
  - add contract tests for all changes: Tasks 1, 2, 3
  - reload semantics (Python auto-reload, modes.yaml watcher, restart for env): documented in compose file comments (Task 2 Step 3)
- Placeholder scan:
  - No `TODO`, `TBD`, or "similar to previous task" shortcuts remain.
- Type consistency:
  - `docker-compose.dev.yml`, `docker/runtime/live-test.Dockerfile`, `BACKEND`, `BASE_IMAGE`, `lcm-sd-ui:dev`, `env.dev`, `env.cuda`, `/conf`, `server.lcm_sr_server:app` are used consistently across all tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-fast-dev-docker-builds.md`.

**Execution mode:** Inline only (`superpowers:executing-plans`). Subagent-driven development is forbidden by `AGENTS.md` repo policy. Execute each task in the current session with TDD red/green/commit per task, with review checkpoints between tasks.
