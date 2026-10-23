# Shared Pool Generation Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`. Do not use subagent-driven development.

**Goal:** Extract queue-backed generation behavior into a backend-neutral runtime while preserving CUDA behavior.

**Architecture:** Add `PoolGenerationRuntime` under `backends/platforms/`. Keep `CudaGenerationRuntime` as a compatibility subclass with no behavior.

**Tech Stack:** Python 3.12, pytest, WorkerPool, backend provider protocols.

---

## File Structure

| File | Responsibility |
|---|---|
| `backends/platforms/pool_runtime.py` | Backend-neutral queue runtime behavior. |
| `backends/platforms/cuda.py` | CUDA provider and compatibility runtime name. |
| `tests/test_backend_runtimes.py` | Shared runtime extraction contract. |

### Task 1: Extract The Shared Runtime

**Files:**
- Create: `backends/platforms/pool_runtime.py`
- Modify: `backends/platforms/cuda.py`
- Test: `tests/test_backend_runtimes.py`

- [x] **Step 1: Write the failing extraction test**

Add this test after `test_cuda_provider_creates_runtime_without_server_branching`:

```python
def test_cuda_runtime_uses_shared_pool_runtime():
    from backends.platforms.cuda import CUDAProvider, CudaGenerationRuntime
    from backends.platforms.pool_runtime import PoolGenerationRuntime

    runtime = CUDAProvider().create_generation_runtime(pool=SimpleNamespace())

    assert issubclass(CudaGenerationRuntime, PoolGenerationRuntime)
    assert isinstance(runtime, PoolGenerationRuntime)
```

- [x] **Step 2: Verify the test fails**

Run:

```bash
python -m pytest tests/test_backend_runtimes.py::test_cuda_runtime_uses_shared_pool_runtime -q
```

Expected: FAIL because `backends.platforms.pool_runtime` does not exist.

- [x] **Step 3: Add the shared runtime**

Create `backends/platforms/pool_runtime.py` with the current `CudaGenerationRuntime` behavior.

Use `type(self).__name__` in the missing-snapshot error. This preserves the CUDA error name through the subclass.

```python
from __future__ import annotations

from typing import Any


class PoolGenerationRuntime:
    def __init__(self, *, pool: Any | None = None, **_: Any) -> None:
        if pool is None:
            from backends.worker_pool import get_worker_pool

            pool = get_worker_pool()
        self._pool = pool

    def get_active_model_snapshot(self) -> Any:
        return self._pool.get_active_model_snapshot()

    def wait_for_result(self, fut: Any) -> Any:
        return self._pool.wait_for_result(fut)

    def submit_generate(
        self,
        req: Any,
        *,
        snapshot: Any = None,
        controlnet_bindings: Any = None,
        timeout_s: float | None = None,
    ) -> Any:
        from backends.worker_pool import GenerationJob

        if snapshot is None:
            snapshot = self._pool.get_active_model_snapshot()
        if snapshot is None:
            runtime_name = type(self).__name__
            raise RuntimeError(
                f"{runtime_name}.submit_generate requires an active model "
                "snapshot; no model is loaded"
            )

        if controlnet_bindings is None:
            controlnet_bindings = self._resolve_bindings(req, snapshot)

        job = GenerationJob(
            req=req,
            controlnet_bindings=list(controlnet_bindings),
            resolution_epoch=snapshot.resolution_epoch,
        )
        return self._pool.submit_job(job, timeout_s=timeout_s)

    @staticmethod
    def _resolve_bindings(req: Any, snapshot: Any) -> list:
        if not getattr(req, "controlnets", None):
            return []
        from server.asset_store import get_store
        from server.controlnet_execution import resolve_controlnet_bindings

        return resolve_controlnet_bindings(
            req,
            mode=snapshot.mode,
            store=get_store(),
            active_family=snapshot.resolved.profile.family_id,
        )

    def switch_mode(self, mode_name: str, force: bool = False) -> Any:
        return self._pool.switch_mode(mode_name, force=force)

    def get_current_mode(self) -> str | None:
        return self._pool.get_current_mode()

    def is_model_loaded(self) -> bool:
        return self._pool.is_model_loaded()

    def get_queue_size(self) -> int:
        return self._pool.get_queue_size()

    def shutdown(self) -> None:
        self._pool.shutdown()
```

- [x] **Step 4: Keep the CUDA compatibility class**

Remove the runtime implementation from `backends/platforms/cuda.py`.

Import the shared class and keep this compatibility class:

```python
from backends.platforms.pool_runtime import PoolGenerationRuntime


class CudaGenerationRuntime(PoolGenerationRuntime):
    pass
```

Keep `CUDAProvider.create_generation_runtime()` unchanged.

- [x] **Step 5: Verify the new test passes**

Run:

```bash
python -m pytest tests/test_backend_runtimes.py::test_cuda_runtime_uses_shared_pool_runtime -q
```

Expected: PASS.

- [x] **Step 6: Verify CUDA runtime behavior**

Run:

```bash
python -m tests.run tests/test_backend_runtimes.py tests/test_worker_pool.py -- -q
```

Expected: All selected tests pass.

- [x] **Step 7: Check drift**

Run:

```bash
drift refs backends/platforms/cuda.py
drift refs backends/platforms/pool_runtime.py
drift check
```

Expected: No task-caused stale binding remains.

- [x] **Step 8: Commit the task**

Run:

```bash
git add docs/superpowers/plans/2026-10-21-mps-shared-pool-runtime.md \
  tests/test_backend_runtimes.py \
  backends/platforms/pool_runtime.py \
  backends/platforms/cuda.py
git commit -m "refactor: extract shared pool runtime (STABL-iedbgecb) - next: review runtime extraction"
```
