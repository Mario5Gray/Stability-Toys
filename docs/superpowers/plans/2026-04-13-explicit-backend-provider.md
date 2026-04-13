# Explicit Backend Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `cuda`/`rknn` branching with one explicit `BACKEND` provider boundary that owns generation, registry, and super-resolution behavior, while keeping `cuda` and `rknn` working and introducing honest scaffolded `mlx` and `cpu` backends.

**Architecture:** Add a small provider layer selected by explicit `BACKEND`, move generation/runtime selection and model-registry ownership behind that provider, and stop probing hardware at runtime. Keep `WorkerPool` as the CUDA-oriented generation runtime in the first pass, allow RKNN to wrap the existing `PipelineService`, and expose `mlx`/`cpu` through placeholder providers that raise clear not-implemented errors at use time.

**Tech Stack:** Python 3, FastAPI, Pydantic, existing `WorkerPool`, existing `PipelineService`, pytest, drift

## Status

- Completed: Task 1 provider contracts and resolution
- Completed: Task 2 provider-owned generation runtimes and compat routing
- Completed: Task 3 provider-backed registry and status assembly
- Completed: Task 4 explicit SR backend handling and provider-owned SR setup
- Completed: Task 5 honest `cpu` and `mlx` placeholder providers
- In progress: Task 6 docs, drift refresh, and final verification

---

## File Structure

### Backend Provider Layer

- Create: `backends/platforms/base.py`
  Defines `BackendCapabilities`, `BackendProvider`, `GenerationRuntimeProtocol`, and `ModelRegistryProtocol`.
- Create: `backends/platform_registry.py`
  Resolves and caches the explicit `BACKEND` provider; rejects missing and unsupported values.
- Create: `backends/platforms/cuda.py`
  Wraps the current CUDA `WorkerPool`, CUDA worker factory, and CUDA SR service.
- Create: `backends/platforms/rknn.py`
  Wraps the current RKNN `PipelineService` and RKNN SR service without teaching `PipelineService` about `mlx` or `cpu`.
- Create: `backends/platforms/cpu.py`
  Placeholder provider that exposes honest capability flags and not-implemented generation/SR runtimes.
- Create: `backends/platforms/mlx.py`
  Placeholder provider with the same contract as `cpu`, but backend-specific error messaging for Apple Silicon work.

### Runtime And Registry Adaptation

- Modify: `backends/model_registry.py`
  Splits CUDA-specific registry behavior from the backend-neutral registry entrypoint.
- Modify: `backends/worker_pool.py`
  Keeps CUDA runtime behavior, but no longer acts as the only implicit default runtime for the whole app.
- Modify: `backends/worker_factory.py`
  Keeps CUDA worker creation helpers, but stops being the assumed global worker factory for all backends.
- Modify: `server/lcm_sr_server.py`
  Replaces direct backend branching with provider-owned generation runtime and SR runtime setup.
- Modify: `server/model_routes.py`
  Reads backend identity, capabilities, and resource stats from the resolved provider/runtime instead of hardcoded CUDA calls.
- Modify: `server/superres_http.py`
  Stops auto-detecting CUDA, consumes the explicit backend contract.
- Modify: `server/superres_service.py`
  Removes `auto` fallback logic and supports provider-owned backend selection.

### Tests

- Create: `tests/test_platform_registry.py`
  Verifies explicit backend selection and provider resolution.
- Create: `tests/test_backend_runtimes.py`
  Verifies provider-owned generation runtimes, including compat helper routing.
- Modify: `tests/test_model_registry.py`
  Covers CUDA and placeholder registry behavior through the new registry contract.
- Modify: `tests/test_worker_pool.py`
  Verifies CUDA runtime remains functional behind the provider split.
- Modify: `tests/test_model_routes.py`
  Verifies backend identity, capabilities, and registry-backed status serialization.
- Modify: `tests/test_superres_http.py`
  Removes `auto` assumptions and verifies explicit backend runtime settings.
- Modify: `tests/test_superres_service.py`
  Removes `auto` assumptions and verifies provider-facing SR backend resolution.

### Docs

- Modify: `README.md`
  Replaces `BACKEND=auto` language with explicit backend configuration.
- Modify: `docs/WORKER_SELECTION.md`
  Documents the provider boundary and backend truth.
- Modify: `docs/TESTING_IN_DOCKER.md`
  Clarifies that `BACKEND=cpu` is scaffold-only for runtime generation.
- Modify: `docs/CUDA_VERIFICATION.md`
  Notes that CUDA selection is explicit and no longer inferred.

---

### Task 1: Add Explicit Backend Provider Contracts And Resolution

**Files:**
- Create: `backends/platforms/base.py`
- Create: `backends/platform_registry.py`
- Create: `tests/test_platform_registry.py`

- [ ] **Step 1: Write the failing provider-resolution tests**

```python
# tests/test_platform_registry.py
import os
from unittest.mock import patch

import pytest


def test_get_backend_provider_requires_explicit_backend():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="BACKEND must be set explicitly"):
            get_backend_provider()


def test_get_backend_provider_rejects_unsupported_backend():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {"BACKEND": "auto"}, clear=True):
        with pytest.raises(RuntimeError, match="Unsupported BACKEND='auto'"):
            get_backend_provider()


def test_get_backend_provider_resolves_known_backend():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {"BACKEND": "cuda"}, clear=True):
        provider = get_backend_provider()

    assert provider.backend_id == "cuda"
    assert provider.capabilities().supports_generation is True
```

- [ ] **Step 2: Run the provider-resolution tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_platform_registry.py -q
```

Expected:

- import or attribute failures because `backends.platform_registry` does not exist yet

- [ ] **Step 3: Implement the provider contracts and explicit resolver**

```python
# backends/platforms/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    supports_generation: bool
    supports_modes: bool
    supports_superres: bool
    supports_model_registry_stats: bool
    supports_img2img: bool


class ModelRegistryProtocol(Protocol):
    def register_model(self, name: str, model_path: str, vram_bytes: int, worker_id: Optional[int] = None, loras: Optional[list[str]] = None): ...
    def unregister_model(self, name: str): ...
    def get_vram_stats(self) -> dict[str, Any]: ...
    def get_total_vram(self) -> int: ...


class GenerationRuntimeProtocol(Protocol):
    def submit_generate(self, req: Any, *, timeout_s: float = 0.25): ...
    def get_current_mode(self) -> Optional[str]: ...
    def is_model_loaded(self) -> bool: ...
    def get_queue_size(self) -> int: ...
    def shutdown(self) -> None: ...


class BackendProvider(Protocol):
    backend_id: str

    def capabilities(self) -> BackendCapabilities: ...
    def create_model_registry(self) -> ModelRegistryProtocol: ...
    def create_generation_runtime(self, *args, **kwargs) -> GenerationRuntimeProtocol: ...
    def create_superres_runtime(self, *args, **kwargs): ...
```

```python
# backends/platform_registry.py
from __future__ import annotations

import os

from backends.platforms.cpu import CPUProvider
from backends.platforms.cuda import CUDAProvider
from backends.platforms.mlx import MLXProvider
from backends.platforms.rknn import RKNNProvider

_provider = None


def _read_backend() -> str:
    backend = (os.environ.get("BACKEND") or "").strip().lower()
    if not backend:
        raise RuntimeError("BACKEND must be set explicitly to one of: cuda, rknn, mlx, cpu")
    if backend not in {"cuda", "rknn", "mlx", "cpu"}:
        raise RuntimeError(f"Unsupported BACKEND='{backend}'")
    return backend


def get_backend_provider():
    global _provider
    if _provider is None:
        backend = _read_backend()
        mapping = {
            "cuda": CUDAProvider(),
            "rknn": RKNNProvider(),
            "mlx": MLXProvider(),
            "cpu": CPUProvider(),
        }
        try:
            _provider = mapping[backend]
        except KeyError as exc:
            raise RuntimeError(f"Unsupported BACKEND='{backend}'") from exc
    return _provider


def reset_backend_provider():
    global _provider
    _provider = None
```

- [ ] **Step 4: Run the provider-resolution tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_platform_registry.py -q
```

Expected:

- all provider-resolution tests pass

- [ ] **Step 5: Commit**

```bash
git add backends/platforms/base.py backends/platform_registry.py tests/test_platform_registry.py
git commit -m "feat: add explicit backend provider resolution"
```

### Task 2: Add Provider-Owned Generation Runtime Adapters

**Files:**
- Create: `backends/platforms/cuda.py`
- Create: `backends/platforms/rknn.py`
- Create: `tests/test_backend_runtimes.py`
- Modify: `server/lcm_sr_server.py`

- [ ] **Step 1: Write the failing runtime-adapter tests**

```python
# tests/test_backend_runtimes.py
from concurrent.futures import Future
from types import SimpleNamespace


def test_run_generate_from_dict_uses_provider_runtime(monkeypatch):
    from server import lcm_sr_server

    fut = Future()
    fut.set_result((b"png-bytes", 1234))

    runtime = SimpleNamespace(
        submit_generate=lambda req, timeout_s=0.25: fut,
    )
    lcm_sr_server.app.state.generation_runtime = runtime
    lcm_sr_server.app.state.sr_service = None

    out_bytes, seed, headers = lcm_sr_server._run_generate_from_dict({"prompt": "owl"})

    assert out_bytes == b"png-bytes"
    assert seed == 1234
    assert headers["X-Seed"] == "1234"


def test_cuda_provider_creates_runtime_without_server_branching(monkeypatch):
    from backends.platforms.cuda import CUDAProvider

    provider = CUDAProvider()
    runtime = provider.create_generation_runtime(queue_max=4)

    assert runtime.__class__.__name__ == "CudaGenerationRuntime"
```

- [ ] **Step 2: Run the runtime-adapter tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_backend_runtimes.py -q
```

Expected:

- failures because the provider-owned runtime adapter classes do not exist yet
- `_run_generate_from_dict()` still reaches directly into `app.state.service`

- [ ] **Step 3: Implement the CUDA and RKNN runtime adapters and route server generation through them**

```python
# backends/platforms/cuda.py
from __future__ import annotations

from backends.platforms.base import BackendCapabilities


class CudaGenerationRuntime:
    def __init__(self, queue_max: int):
        from backends.worker_pool import get_worker_pool

        self._pool = get_worker_pool()

    def submit_generate(self, req, *, timeout_s: float = 0.25):
        from backends.worker_pool import GenerationJob

        return self._pool.submit_job(GenerationJob(req=req))

    def switch_mode(self, mode_name: str, force: bool = False):
        return self._pool.switch_mode(mode_name, force=force)

    def get_current_mode(self):
        return self._pool.get_current_mode()

    def is_model_loaded(self) -> bool:
        return self._pool.is_model_loaded()

    def get_queue_size(self) -> int:
        return self._pool.get_queue_size()

    def shutdown(self) -> None:
        self._pool.shutdown()


class CUDAProvider:
    backend_id = "cuda"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True, True, True)

    def create_generation_runtime(self, *, queue_max: int, **kwargs):
        return CudaGenerationRuntime(queue_max=queue_max)
```

```python
# backends/platforms/rknn.py
from __future__ import annotations

from backends.platforms.base import BackendCapabilities


class RknnGenerationRuntime:
    def __init__(self, *, paths, num_workers: int, queue_max: int, use_rknn_context_cfgs: bool):
        from server.lcm_sr_server import PipelineService, build_rknn_context_cfgs_for_rk3588

        self._service = PipelineService.get_instance(
            paths=paths,
            num_workers=num_workers,
            queue_max=queue_max,
            rknn_context_cfgs=build_rknn_context_cfgs_for_rk3588(num_workers),
            use_rknn_context_cfgs=use_rknn_context_cfgs,
        )

    def submit_generate(self, req, *, timeout_s: float = 0.25):
        return self._service.submit(req, timeout_s=timeout_s)

    def get_current_mode(self):
        return None

    def is_model_loaded(self) -> bool:
        return True

    def get_queue_size(self) -> int:
        return self._service.q.qsize()

    def shutdown(self) -> None:
        self._service.shutdown()
```

```python
# server/lcm_sr_server.py
provider = get_backend_provider()
app.state.backend_provider = provider
app.state.generation_runtime = provider.create_generation_runtime(
    paths=model_root_path,
    num_workers=NUM_WORKERS,
    queue_max=QUEUE_MAX,
    use_rknn_context_cfgs=USE_RKNN_CONTEXT_CFGS,
)

...

runtime = app.state.generation_runtime
fut = runtime.submit_generate(req, timeout_s=0.25)

...

def _run_generate_from_dict(gen_req: dict):
    req = GenerateRequest(**gen_req)
    runtime = app.state.generation_runtime
    fut = runtime.submit_generate(req, timeout_s=0.25)
```

- [ ] **Step 4: Run the runtime-adapter tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_backend_runtimes.py -q
```

Expected:

- runtime-adapter tests pass
- `_run_generate_from_dict()` no longer depends on `app.state.service`

- [ ] **Step 5: Commit**

```bash
git add backends/platforms/cuda.py backends/platforms/rknn.py server/lcm_sr_server.py tests/test_backend_runtimes.py
git commit -m "feat: add provider-owned generation runtimes"
```

### Task 3: Split The Registry Contract And Remove Backend-Neutral CUDA Assumptions

**Files:**
- Modify: `backends/model_registry.py`
- Modify: `server/model_routes.py`
- Modify: `tests/test_model_registry.py`
- Modify: `tests/test_model_routes.py`

- [ ] **Step 1: Write the failing registry and status tests**

```python
# tests/test_model_registry.py
def test_placeholder_registry_reports_backend_without_vram_fields():
    from backends.model_registry import PlaceholderModelRegistry

    registry = PlaceholderModelRegistry("cpu")
    stats = registry.get_vram_stats()

    assert stats["backend"] == "cpu"
    assert stats["device"] == "CPU placeholder"
    assert stats["models_loaded"] == 0


# tests/test_model_routes.py
async def test_models_status_uses_provider_capabilities_and_registry_stats():
    runtime = Mock()
    runtime.get_current_mode.return_value = None
    runtime.is_model_loaded.return_value = False
    runtime.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {"backend": "cpu", "device": "CPU placeholder", "models_loaded": 0}

    provider = Mock()
    provider.backend_id = "cpu"
    provider.capabilities.return_value = Mock(
        supports_generation=False,
        supports_modes=True,
        supports_superres=False,
        supports_model_registry_stats=False,
        supports_img2img=False,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
         patch("server.model_routes.get_generation_runtime", return_value=runtime), \
         patch("server.model_routes.get_model_registry", return_value=registry):
        data = await model_routes.get_models_status()

    assert data["backend"] == "cpu"
    assert data["capabilities"]["supports_generation"] is False
    assert data["vram"] == {"backend": "cpu", "device": "CPU placeholder", "models_loaded": 0}
```

- [ ] **Step 2: Run the registry and status tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_model_registry.py -k placeholder -q
python3 -m pytest tests/test_model_routes.py -k capabilities -q
```

Expected:

- failures because there is no placeholder registry
- failures because model status still assumes CUDA-oriented registry/state

- [ ] **Step 3: Implement the provider-owned registry entrypoint and provider-aware status assembly**

```python
# backends/model_registry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backends.platform_registry import get_backend_provider


class PlaceholderModelRegistry:
    def __init__(self, backend_id: str):
        self._backend_id = backend_id
        self._loaded = {}

    def register_model(self, name: str, model_path: str, vram_bytes: int, worker_id: Optional[int] = None, loras: Optional[list[str]] = None):
        self._loaded[name] = {"name": name, "model_path": model_path, "loras": loras or []}

    def unregister_model(self, name: str):
        self._loaded.pop(name, None)

    def get_vram_stats(self) -> dict[str, Any]:
        return {
            "backend": self._backend_id,
            "device": f"{self._backend_id.upper()} placeholder",
            "models_loaded": len(self._loaded),
        }

    def get_total_vram(self) -> int:
        return 0


_registry = None


def get_model_registry():
    global _registry
    if _registry is None:
        _registry = get_backend_provider().create_model_registry()
    return _registry
```

```python
# server/model_routes.py
provider = get_backend_provider()
caps = provider.capabilities()
runtime = get_generation_runtime()
registry = get_model_registry()

return {
    "backend": provider.backend_id,
    "backend_version": backend_version,
    "current_mode": runtime.get_current_mode(),
    "is_loaded": runtime.is_model_loaded(),
    "queue_size": runtime.get_queue_size(),
    "capabilities": {
        "supports_generation": caps.supports_generation,
        "supports_modes": caps.supports_modes,
        "supports_superres": caps.supports_superres,
        "supports_model_registry_stats": caps.supports_model_registry_stats,
        "supports_img2img": caps.supports_img2img,
    },
    "vram": registry.get_vram_stats(),
}
```

- [ ] **Step 4: Run the registry and status tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_model_registry.py -k placeholder -q
python3 -m pytest tests/test_model_routes.py -k capabilities -q
```

Expected:

- registry and status tests pass without direct backend-neutral `torch.cuda` assumptions

- [ ] **Step 5: Commit**

```bash
git add backends/model_registry.py server/model_routes.py tests/test_model_registry.py tests/test_model_routes.py
git commit -m "feat: add provider-owned model registry and status"
```

### Task 4: Move Super-Resolution Under The Explicit Backend Contract

**Files:**
- Modify: `server/superres_http.py`
- Modify: `server/superres_service.py`
- Modify: `backends/platforms/cuda.py`
- Modify: `backends/platforms/rknn.py`
- Modify: `tests/test_superres_http.py`
- Modify: `tests/test_superres_service.py`

- [ ] **Step 1: Write the failing SR explicit-backend tests**

```python
# tests/test_superres_http.py
import pytest


def test_load_superres_runtime_settings_requires_explicit_backend():
    from server.superres_http import load_superres_runtime_settings

    with pytest.raises(RuntimeError, match="BACKEND must be set explicitly"):
        load_superres_runtime_settings({}, cuda_available=False)


def test_load_superres_runtime_settings_preserves_explicit_backend():
    from server.superres_http import load_superres_runtime_settings

    settings = load_superres_runtime_settings(
        {"BACKEND": "rknn", "MODEL_ROOT": "/models"},
        cuda_available=False,
    )

    assert settings.backend == "rknn"
    assert settings.use_cuda is False


# tests/test_superres_service.py
import pytest


def test_resolve_superres_backend_rejects_unsupported_backend():
    from server.superres_service import resolve_superres_backend

    with pytest.raises(ValueError, match="Unsupported backend: auto"):
        resolve_superres_backend(backend="auto", use_cuda=False)
```

- [ ] **Step 2: Run the SR tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_superres_http.py -q
python3 -m pytest tests/test_superres_service.py -q
```

Expected:

- failures because SR helpers still accept and derive `auto`

- [ ] **Step 3: Implement explicit SR backend handling and provider-owned SR construction**

```python
# server/superres_http.py
def load_superres_runtime_settings(environ=None, *, cuda_available=None) -> SuperResRuntimeSettings:
    env = environ or os.environ
    backend = (env.get("BACKEND") or "").lower().strip()
    if not backend:
        raise RuntimeError("BACKEND must be set explicitly to one of: cuda, rknn, mlx, cpu")
    if backend not in {"cuda", "rknn", "mlx", "cpu"}:
        raise RuntimeError(f"Unsupported BACKEND='{backend}'")

    use_cuda = backend == "cuda"
    return SuperResRuntimeSettings(
        enabled=(env.get("SR_ENABLED", "1") not in ("0", "false", "False")),
        backend=backend,
        use_cuda=use_cuda,
        sr_model_path=env.get("SR_MODEL_PATH", os.path.join(env.get("MODEL_ROOT", ""), "super-resolution-10.rknn")),
        ...
    )
```

```python
# server/superres_service.py
def resolve_superres_backend(*, backend: str, use_cuda: bool) -> SuperResBackend:
    backend_norm = (backend or "").lower().strip()
    if backend_norm == "cuda":
        return "cuda"
    if backend_norm == "rknn":
        return "rknn"
    raise ValueError(f"Unsupported backend: {backend}")
```

```python
# backends/platforms/cuda.py
def create_superres_runtime(self, *, settings, environ=None, path_exists=os.path.isfile, cuda_factory=None, **kwargs):
    from server.superres_http import initialize_superres_service

    return initialize_superres_service(
        enabled=settings.enabled,
        backend="cuda",
        use_cuda=True,
        sr_model_path=settings.sr_model_path,
        sr_num_workers=settings.sr_num_workers,
        sr_queue_max=settings.sr_queue_max,
        sr_input_size=settings.sr_input_size,
        sr_output_size=settings.sr_output_size,
        sr_max_pixels=settings.sr_max_pixels,
        environ=environ,
        path_exists=path_exists,
        cuda_factory=cuda_factory,
    )
```

- [ ] **Step 4: Run the SR tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_superres_http.py -q
python3 -m pytest tests/test_superres_service.py -q
```

Expected:

- explicit-backend SR tests pass
- no tests assume `auto`

- [ ] **Step 5: Commit**

```bash
git add server/superres_http.py server/superres_service.py backends/platforms/cuda.py backends/platforms/rknn.py tests/test_superres_http.py tests/test_superres_service.py
git commit -m "feat: move super-resolution under explicit backend providers"
```

### Task 5: Add Honest Placeholder CPU And MLX Providers

**Files:**
- Create: `backends/platforms/cpu.py`
- Create: `backends/platforms/mlx.py`
- Modify: `tests/test_platform_registry.py`
- Modify: `tests/test_backend_runtimes.py`

- [ ] **Step 1: Write the failing placeholder-provider tests**

```python
# tests/test_platform_registry.py
def test_cpu_provider_resolves_with_placeholder_capabilities():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {"BACKEND": "cpu"}, clear=True):
        provider = get_backend_provider()

    caps = provider.capabilities()
    assert provider.backend_id == "cpu"
    assert caps.supports_generation is False
    assert caps.supports_superres is False


# tests/test_backend_runtimes.py
import pytest


def test_cpu_generation_runtime_raises_clear_error():
    from backends.platforms.cpu import CPUProvider

    runtime = CPUProvider().create_generation_runtime(queue_max=1)

    with pytest.raises(NotImplementedError, match="BACKEND=cpu generation is not implemented"):
        runtime.submit_generate({"prompt": "owl"})


def test_mlx_generation_runtime_raises_clear_error():
    from backends.platforms.mlx import MLXProvider

    runtime = MLXProvider().create_generation_runtime(queue_max=1)

    with pytest.raises(NotImplementedError, match="BACKEND=mlx generation is not implemented"):
        runtime.submit_generate({"prompt": "owl"})
```

- [ ] **Step 2: Run the placeholder-provider tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_platform_registry.py -k 'cpu_provider' -q
python3 -m pytest tests/test_backend_runtimes.py -k 'not_implemented' -q
```

Expected:

- failures because `cpu` and `mlx` providers do not exist yet

- [ ] **Step 3: Implement the placeholder providers and runtimes**

```python
# backends/platforms/cpu.py
from __future__ import annotations

from backends.model_registry import PlaceholderModelRegistry
from backends.platforms.base import BackendCapabilities


class PlaceholderGenerationRuntime:
    def __init__(self, backend_id: str):
        self._backend_id = backend_id

    def submit_generate(self, req, *, timeout_s: float = 0.25):
        raise NotImplementedError(f"BACKEND={self._backend_id} generation is not implemented")

    def get_current_mode(self):
        return None

    def is_model_loaded(self) -> bool:
        return False

    def get_queue_size(self) -> int:
        return 0

    def shutdown(self) -> None:
        return None


class PlaceholderSuperResRuntime:
    def __init__(self, backend_id: str):
        self._backend_id = backend_id

    def submit(self, *args, **kwargs):
        raise NotImplementedError(f"BACKEND={self._backend_id} super-resolution is not implemented")

    def unload(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class CPUProvider:
    backend_id = "cpu"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(False, True, False, False, False)

    def create_model_registry(self):
        return PlaceholderModelRegistry(self.backend_id)

    def create_generation_runtime(self, *args, **kwargs):
        return PlaceholderGenerationRuntime(self.backend_id)

    def create_superres_runtime(self, *args, **kwargs):
        return PlaceholderSuperResRuntime(self.backend_id)
```

```python
# backends/platforms/mlx.py
from backends.platforms.cpu import PlaceholderGenerationRuntime, PlaceholderSuperResRuntime
from backends.model_registry import PlaceholderModelRegistry
from backends.platforms.base import BackendCapabilities


class MLXProvider:
    backend_id = "mlx"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(False, True, False, False, False)

    def create_model_registry(self):
        return PlaceholderModelRegistry(self.backend_id)

    def create_generation_runtime(self, *args, **kwargs):
        return PlaceholderGenerationRuntime(self.backend_id)

    def create_superres_runtime(self, *args, **kwargs):
        return PlaceholderSuperResRuntime(self.backend_id)
```

- [ ] **Step 4: Run the placeholder-provider tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_platform_registry.py -k 'cpu_provider' -q
python3 -m pytest tests/test_backend_runtimes.py -k 'not_implemented' -q
```

Expected:

- placeholder provider tests pass with clear backend-specific failure messages

- [ ] **Step 5: Commit**

```bash
git add backends/platforms/cpu.py backends/platforms/mlx.py tests/test_platform_registry.py tests/test_backend_runtimes.py
git commit -m "feat: add honest cpu and mlx placeholder providers"
```

### Task 6: Update Docs, Drift Bindings, And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/WORKER_SELECTION.md`
- Modify: `docs/TESTING_IN_DOCKER.md`
- Modify: `docs/CUDA_VERIFICATION.md`
- Modify: `drift.lock`

- [ ] **Step 1: Write the failing documentation checks**

```bash
rg -n 'BACKEND=auto|\"auto\"|auto-detects CUDA|otherwise RKNN' README.md docs/WORKER_SELECTION.md docs/TESTING_IN_DOCKER.md docs/CUDA_VERIFICATION.md
```

Expected:

- existing docs still mention `auto` or imply backend inference

- [ ] **Step 2: Record the verification commands for the final pass**

Run:

```bash
python3 -m pytest tests/test_platform_registry.py tests/test_backend_runtimes.py tests/test_worker_pool.py tests/test_model_registry.py tests/test_model_routes.py tests/test_superres_http.py tests/test_superres_service.py -q
drift check
```

Expected before docs update:

- backend tests should pass after Tasks 1-5
- docs may still be stale or misleading until edited and relinked

- [ ] **Step 3: Update the docs and refresh drift bindings**

```markdown
# README.md
- Replace every `BACKEND=auto` example with an explicit backend value.
- Add a short migration note: `BACKEND` is now required and unsupported values fail normally.

# docs/WORKER_SELECTION.md
- Document that `BACKEND` resolves a provider boundary for generation, registry, and SR.
- Note that `mlx` and `cpu` are scaffold-only in this pass.

# docs/TESTING_IN_DOCKER.md
- Keep `BACKEND=cpu` for the local smoke/build path, but state explicitly that it is not a supported inference backend.

# docs/CUDA_VERIFICATION.md
- Replace any implied runtime autodetection language with explicit `BACKEND=cuda`.
```

Run:

```bash
drift link docs/superpowers/specs/2026-04-13-explicit-backend-provider-design.md
drift link docs/superpowers/plans/2026-04-13-explicit-backend-provider.md README.md
drift link docs/superpowers/plans/2026-04-13-explicit-backend-provider.md docs/WORKER_SELECTION.md
drift link docs/superpowers/plans/2026-04-13-explicit-backend-provider.md docs/TESTING_IN_DOCKER.md
drift link docs/superpowers/plans/2026-04-13-explicit-backend-provider.md docs/CUDA_VERIFICATION.md
```

- [ ] **Step 4: Run the final verification suite**

Run:

```bash
python3 -m pytest tests/test_platform_registry.py tests/test_backend_runtimes.py tests/test_worker_pool.py tests/test_model_registry.py tests/test_model_routes.py tests/test_superres_http.py tests/test_superres_service.py -q
drift check
rg -n 'BACKEND=auto|\"auto\"|auto-detects CUDA|otherwise RKNN' README.md docs/WORKER_SELECTION.md docs/TESTING_IN_DOCKER.md docs/CUDA_VERIFICATION.md
```

Expected:

- targeted backend/provider test suite passes
- `drift check` reports all managed docs `ok`
- final `rg` returns no matches

- [ ] **Step 5: Commit**

```bash
git add README.md docs/WORKER_SELECTION.md docs/TESTING_IN_DOCKER.md docs/CUDA_VERIFICATION.md drift.lock docs/superpowers/plans/2026-04-13-explicit-backend-provider.md
git commit -m "docs: document explicit backend provider contract"
```

---

## Self-Review

### Spec coverage

- explicit `BACKEND` selection and `auto` removal: covered in Tasks 1 and 4
- provider boundary for generation, registry, and SR: covered in Tasks 1 through 4
- CUDA and RKNN stay working: covered in Tasks 2 through 4
- `mlx` and `cpu` scaffold providers: covered in Task 5
- compat helper fix: covered in Task 2
- registry/status cleanup: covered in Task 3
- doc migration and drift refresh: covered in Task 6

### Placeholder scan

- no `TODO`, `TBD`, or “implement later” placeholders remain
- every task includes test commands, expected failures, implementation snippets, and verification commands

### Type consistency

- provider contract names stay consistent: `BackendProvider`, `GenerationRuntimeProtocol`, `ModelRegistryProtocol`, `BackendCapabilities`
- runtime entrypoint stays consistent: `submit_generate`
- placeholder and concrete backends use the same provider surface
