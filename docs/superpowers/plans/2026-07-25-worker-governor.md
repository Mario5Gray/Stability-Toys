# Worker Governor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Project policy forbids sub-agent-driven development (AGENTS.md) — do NOT use superpowers:subagent-driven-development.**

**Goal:** Extract the control plane out of `WorkerPool` behind a locality-agnostic `WorkerHandle` interface (`InProcessWorkerHandle` first), with zero observable behavior change — the same no-op-facade playbook that shipped the backplane (0-byte `ws_routes.py` diff, existing suite green unmodified).

**Architecture:** A new `backends/governor.py` owns the queue, epoch/snapshot authority, admission barrier, dispatch loop, and lifecycle (the control plane). A new `backends/worker_handle.py` owns the `WorkerHandle` ABC + `InProcessWorkerHandle` (the threaded-worker coupling). `backends/worker_pool.py` becomes a thin delegating facade that re-exports the shared job types. The queue is the Governor↔Handle boundary; the backplane (done, PR #19) is the Handle's output contract. `SubprocessWorkerHandle` plugs in later with no Governor change.

**Tech Stack:** Python 3.11, stdlib `threading`/`queue`/`concurrent.futures`, `dataclasses`, `pytest`. No new runtime dependency. The backplane's vendored reactive-streams ABCs (`backends/backplane/reactivestreams/`) are reused as-is.

**Spec:** `docs/superpowers/specs/2026-07-25-worker-governor-design.md` (accepted, commit `4ea198b`).

**FP:** STABL-vdkdruox (child of umbrella STABL-nvmieaxh).

## Spec reconciliations discovered during planning

1. **Import-graph directionality (Theta's sub-clarification on finding 1).** `WorkerHandle.submit(job: Job)` type-hints `Job`, which now lives in `governor.py`. The "worker_handle does NOT import governor" claim is only true at runtime because `worker_handle.py` uses `from __future__ import annotations` + `TYPE_CHECKING` for the `Job` hint (the backplane modules already use this pattern — confirmed `from __future__ import annotations` in all `backends/backplane/*.py`). This makes the hint a string at runtime, forcing no import, while `governor.py` imports `worker_handle` at runtime to construct `InProcessWorkerHandle`. The first RED test (Task 1) surfaces this immediately under TDD — if the implementer naively adds `from backends.governor import Job` to `worker_handle.py`, the cycle reintroduces and the import fails.

2. **Open question (a) resolved: Governor dispatch loop is `_worker_loop` VERBATIM, not replaced.** Resolved via RED-first reasoning during planning. The `_worker_loop` (`worker_pool.py:728-868`) is 140 LOC of intertwined logic: mode-switch handling, cancel-skip, demand-reload, stale-epoch barrier, execute, sink-driving, OOM recovery. The critical constraint: the post-execute cancel-discard check (`:798-809`) reads `job_record.cancel_requested` under `_job_lock` and drives `record.sink` — the handle **cannot** acquire `_job_lock` (the backplane's `Subscriber↔lock` invariant, spec §8) and cannot access `record.sink`. Therefore the dispatch loop body **stays in the Governor**, moved verbatim with one substitution: `self._worker` → `self._handle.worker`. The handle is used only for `start()`/`unload()`/`stop()`/`health()` + holding the `_worker` reference. **The handle's `submit()` method is defined as the facet-3 contract but is NOT called in v1's in-proc dispatch loop.** Facet-3's `SubprocessWorkerHandle` will use `submit()` (it owns the IPC channel + execution + can't share the Governor's `_job_lock`); v1's `InProcessWorkerHandle` runs on the Governor's dispatch thread, so the Governor drives the sink directly — same as today.

3. **Open question (b) resolved: demand-reload / idle-eviction lands in the Governor (confirmed the spec's firm lean).** The Governor owns `_reload_from_snapshot` (reads `self._active_snapshot`, which the Governor owns) and the idle watchdog (enqueues `CustomJob(_evict_if_idle)`). The handle's `start()` is the primitive the Governor calls to (re)build the worker; the handle's `unload()` is the teardown primitive. The `CustomJob` callable stays as-is (D4 defers the redesign to facet-3).

4. **Channel ownership (from spec §5(a), revised for v1 no-op).** The spec says the Handle owns channel creation. **For v1's no-op, the Governor's `submit_job` keeps the channel-opening + `_FutureBridge` attachment verbatim** (`worker_pool.py:899-903`): it opens `InProcBackplane(job.job_id)`, stores the sink in `record.sink`, and subscribes `_FutureBridge(job.fut)` before enqueueing. The dispatch loop drives `record.sink` directly (same as `_worker_loop:798-863`). The handle's `submit()` is the facet-3 contract (the subprocess handle owns its IPC connection); v1 does not call it. This is a pragmatic v1 exception to the spec's §5(a) resolution — documented here so the plan doesn't contradict itself. Facet-3 changes this: `SubprocessWorkerHandle.submit()` owns the IPC channel, and the Governor subscribes `_FutureBridge` to the returned `Publisher`.

5. **`submit_job` channel-opening stays in the Governor for v1 (confirmed by #4).** The Governor's `submit_job` opens the `InProcBackplane` channel + attaches `_FutureBridge` *before* enqueueing — identical to `worker_pool.py:899-903`. The dispatch loop drives `record.sink`. This is the no-op: the channel-opening + sink-driving logic is moved verbatim from `WorkerPool` to `Governor`, not restructured.

## Global Constraints

- **Zero `server/ws_routes.py` + `server/model_routes.py` diff.** The empty diff is the no-op proof. If a change to either file seems necessary, stop — the design is being violated.
- **Existing suite stays green unmodified:** `tests/test_worker_pool.py` (66 tests), `tests/test_ws_routes.py`, `tests/test_model_routes.py`, `tests/test_backplane_facade.py`, and all other backplane tests. No edits to these files.
- **Python env:** `conda activate stability-toys` before running pytest (per AGENTS.md); use `python`, not `python3`.
- **Commit discipline (AGENTS.md / stopping-point policy):** every commit message includes the FP id `STABL-vdkdruox`, what changed, and the exact next step.
- **No waveplan for this track** (human-driven, kept close, same as the backplane). Do not create waveplan rows or FP subissues unless the human asks.
- **TDD is mandatory.** RED → GREEN → COMMIT for every task. No implementation code without a failing test first.
- **The `from backends.worker_pool import GenerationJob` public surface is preserved by re-export, not by ownership.** The types move to `governor.py`; `worker_pool.py` re-exports them.

---

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `backends/worker_handle.py` | `WorkerHandle` ABC + `WorkerHealth` dataclass + `InProcessWorkerHandle` impl (worker thread, worker factory, worker ref, unload/stop, run_job_in_thread) |
| `backends/governor.py` | `Governor` class (queue, epoch/snapshot authority, admission barrier, dispatch loop, lifecycle, recovery) **+ shared job types** moved from `worker_pool.py` (`Job`, `GenerationJob`, `ModeSwitchJob`, `CustomJob`, `JobType`, `JobRecord`, `ActiveModelSnapshot`, `StaleResolutionError`, `_FutureBridge`, `WorkerFactory`) |
| `tests/test_worker_handle.py` | `InProcessWorkerHandle` isolation tests (start/submit/health/unload/stop, backplane driving, no `_job_lock` acquisition) |
| `tests/test_governor.py` | `Governor` isolation tests (handle pluggability via stub, queue + authority + dispatch, no real worker) |

**Modified files:**

| File | Change |
|---|---|
| `backends/worker_pool.py` | Reduced to thin delegating facade: `WorkerPool.__init__` constructs a `Governor`; every public method delegates; re-exports shared types from `governor.py`; `get_worker_pool`/`reset_worker_pool` unchanged |

**Unchanged files (the no-op proof):**

| File | Why unchanged |
|---|---|
| `server/ws_routes.py` | Calls `WorkerPool` methods; facade preserves signatures |
| `server/model_routes.py` | Calls `get_worker_pool()` + `WorkerPool` methods |
| `server/lcm_sr_server.py` | Sets `app.state.worker_pool` |
| `backends/backplane/*` | Data-plane transport — done (PR #19) |
| `tests/test_worker_pool.py` | 66 integration tests — the no-op proof |
| `tests/test_ws_routes.py` | Integration proof |
| `tests/test_model_routes.py` | Integration proof |
| `tests/test_backplane_facade.py` | `_FutureBridge` proof — imports from `backends.worker_pool`, preserved by re-export |

---

## Task 1: Shared types module + import graph (RED: import cycle prevention)

**Goal:** Move the shared job types from `worker_pool.py` into `governor.py`, establish the acyclic import graph, and prove `worker_pool.py` can still re-export them. This is the structural foundation — every subsequent task depends on it.

**Files:**
- Create: `backends/governor.py` (types only first; `Governor` class added in Task 3)
- Create: `backends/worker_handle.py` (ABC + `WorkerHealth` only first; `InProcessWorkerHandle` added in Task 2)
- Modify: `backends/worker_pool.py` (re-export types from `governor.py`; remove the type definitions)
- Test: `tests/test_governor.py` (import-graph proof)

- [ ] **Step 1: Write the failing test — import graph is acyclic and re-exports work**

Create `tests/test_governor.py`:

```python
"""Governor extraction tests. Task 1: import graph + type re-exports.

Proves the shared job types moved to governor.py are re-exported by
worker_pool.py so `from backends.worker_pool import GenerationJob`
(ws_routes.py:621) stays unbroken, and that worker_handle.py does NOT
import governor at runtime (acyclic graph).
"""
import sys
import importlib

import pytest


def test_worker_pool_reexports_generation_job():
    """The public surface `from backends.worker_pool import GenerationJob` works."""
    from backends.worker_pool import GenerationJob
    assert GenerationJob is not None


def test_worker_pool_reexports_all_shared_types():
    """Every shared type is re-exported from worker_pool."""
    from backends.worker_pool import (
        Job, JobType, GenerationJob, ModeSwitchJob, CustomJob,
        JobRecord, ActiveModelSnapshot, StaleResolutionError,
        WorkerFactory, _FutureBridge,
    )
    # All are the SAME objects as in governor (re-export, not redefinition)
    from backends import governor
    assert GenerationJob is governor.GenerationJob
    assert ActiveModelSnapshot is governor.ActiveModelSnapshot
    assert StaleResolutionError is governor.StaleResolutionError


def test_governor_imports_worker_handle_at_runtime():
    """governor.py imports worker_handle (to construct InProcessWorkerHandle)."""
    from backends import governor
    assert hasattr(governor, 'InProcessWorkerHandle') or hasattr(governor, 'WorkerHandle')


def test_worker_handle_does_not_import_governor_at_runtime():
    """worker_handle.py must NOT import governor at runtime (acyclic).

    The Job type hint in WorkerHandle.submit is deferred via
    from __future__ import annotations + TYPE_CHECKING.
    """
    # Force a fresh import of worker_handle and check governor is not in its
    # loaded modules (it may be in sys.modules from other tests, so we check
    # worker_handle's own imports, not sys.modules globally).
    import backends.worker_handle as wh
    wh_module = sys.modules[wh.__name__]
    # Check that 'backends.governor' is not a direct import of worker_handle
    # by inspecting the module's source for runtime imports.
    import inspect
    src = inspect.getsource(wh_module)
    # TYPE_CHECKING-guarded imports are fine; bare runtime imports are not.
    # We check that any 'from backends.governor' or 'import backends.governor'
    # is inside a TYPE_CHECKING block (indented under `if TYPE_CHECKING:`).
    lines = src.split('\n')
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if 'backends.governor' in stripped and ('import' in stripped):
            # Walk backwards to find if we're inside a TYPE_CHECKING block
            indent = len(line) - len(stripped)
            in_type_checking = False
            for j in range(i - 1, -1, -1):
                prev = lines[j]
                prev_stripped = prev.lstrip()
                prev_indent = len(prev) - len(prev_stripped)
                if prev_indent < indent and 'if TYPE_CHECKING' in prev_stripped:
                    in_type_checking = True
                    break
                if prev_indent < indent and prev_stripped.startswith('if ') and 'TYPE_CHECKING' not in prev_stripped:
                    break
                if prev_indent < indent and not prev_stripped.startswith('if '):
                    break
            assert in_type_checking, (
                f"worker_handle.py line {i+1} imports backends.governor at runtime "
                f"(not guarded by TYPE_CHECKING): {line.strip()}"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_governor.py -v`
Expected: FAIL — `backends.governor` doesn't exist yet (ImportError).

- [ ] **Step 3: Create `backends/governor.py` with the shared types (moved from `worker_pool.py`)**

Create `backends/governor.py`. Move these types **verbatim** from `worker_pool.py` (lines 48-187): `StaleResolutionError`, `ActiveModelSnapshot`, `WorkerFactory`, `JobType`, `Job`, `GenerationJob`, `JobRecord`, `_FutureBridge`, `ModeSwitchJob`, `CustomJob`. Keep all their imports (the backplane imports, `resolve_model`, `PipelineWorker`, etc.). Do NOT add the `Governor` class yet — that's Task 3. Add a placeholder comment:

```python
"""
Worker Governor: control-plane extraction behind a WorkerHandle interface.

Task 1: shared job types (moved from worker_pool.py). The Governor class
itself is added in Task 3.
"""
from __future__ import annotations

import gc
import logging
import os
import queue
import threading
import time
import uuid
import torch
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
from concurrent.futures import Future, CancelledError
from enum import Enum

from server.mode_config import get_mode_config, ModeConfig, ModeConfigManager
from backends.model_registry import get_model_registry
from backends.base import PipelineWorker
from backends.platforms.base import ModelRegistryProtocol
from backends.model_resolution import (
    LocalModelBinding,
    ResolvedModel,
    merge_mode_capabilities,
    resolve_model,
)
from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.backplane.frames import Result, BackplaneError, BackplaneErrorCode
from backends.backplane.interface import JobSink
from backends.backplane.reactivestreams import Subscriber

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_TIMEOUT_S: float = float(os.environ.get("WORKER_QUEUE_TIMEOUT_S", "0.25"))


class StaleResolutionError(RuntimeError):
    """A queued job was resolved against a superseded model authority."""


@dataclass(frozen=True)
class ActiveModelSnapshot:
    """The pool's single immutable model authority, published atomically with the
    worker under the state lock. Carries the deep-copied mode, the portable
    resolved value + node-local binding, and the resolution epoch it was minted
    at. Idle eviction retains this so a demand reload can reconstruct the worker
    without re-detecting."""
    mode_name: str
    mode: ModeConfig
    resolved: ResolvedModel
    binding: LocalModelBinding
    resolution_epoch: int


class WorkerFactory(Protocol):
    """Protocol for worker creation functions."""
    def __call__(
        self,
        worker_id: int,
        resolved: ResolvedModel,
        binding: LocalModelBinding,
    ) -> PipelineWorker:
        """Create a worker from a resolved model and its local binding."""
        ...


class JobType(Enum):
    """Types of jobs that can be queued."""
    GENERATION = "generation"
    MODE_SWITCH = "mode_switch"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    CUSTOM = "custom"


@dataclass
class Job(ABC):
    """Base class for all job types."""
    job_type: JobType = field(init=False)
    fut: Future = field(init=False, default_factory=Future)

    def __post_init__(self):
        if self.fut is None:
            self.fut = Future()

    @abstractmethod
    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        pass


@dataclass
class GenerationJob(Job):
    """Job for image generation."""
    req: Any
    init_image: Optional[bytes] = None
    controlnet_bindings: list[Any] = field(default_factory=list)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    resolution_epoch: int = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.GENERATION

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        if worker is None:
            raise RuntimeError("No worker available for generation")
        return worker.run_job(self)


@dataclass
class JobRecord:
    job_id: str
    state: str
    job: GenerationJob
    cancel_requested: bool = False
    sink: Optional[JobSink] = None


class _FutureBridge(Subscriber):
    """Compat Subscriber: fulfils a concurrent.futures.Future from the backplane
    stream, reproducing today's fut.set_result / fut.set_exception exactly.

    Touches ONLY the Future — never pool state or _job_lock (spec §3.3 lock
    invariant). Requests unbounded demand so the synchronous in-proc channel
    delivers terminals immediately.
    """
    def __init__(self, fut: Future):
        self._fut = fut

    def on_subscribe(self, subscription):
        subscription.request(1 << 62)

    def on_next(self, value):
        if isinstance(value, Result) and not self._fut.done():
            self._fut.set_result(value.image.read_sync())

    def on_error(self, error):
        if not self._fut.done():
            self._fut.set_exception(error.to_exception())

    def on_complete(self):
        pass


@dataclass
class ModeSwitchJob(Job):
    """Job for switching model mode."""
    target_mode: str
    on_complete: Optional[Callable] = None
    force: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.MODE_SWITCH

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        logger.info(f"[ModeSwitchJob] Switching to mode: {self.target_mode}")
        if self.on_complete:
            self.on_complete(self.target_mode)
        return {"mode": self.target_mode, "status": "switched"}


@dataclass
class CustomJob(Job):
    """Extensible custom job."""
    handler: Callable
    args: tuple = ()
    kwargs: Optional[dict] = None

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.CUSTOM
        if self.kwargs is None:
            self.kwargs = {}

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        return self.handler(*self.args, **self.kwargs)


# Governor class added in Task 3.
```

- [ ] **Step 4: Create `backends/worker_handle.py` with the ABC + `WorkerHealth` (no impl yet)**

Create `backends/worker_handle.py`:

```python
"""
WorkerHandle: locality-agnostic interface to ONE worker.

The Governor programs to this interface regardless of where the worker runs
(in-proc thread today; subprocess in facet-3; microservice later).

Import graph: worker_handle.py imports backplane + base, NOT governor.
The Job type hint in submit() is deferred via from __future__ import annotations
+ TYPE_CHECKING so it forces no runtime import of governor.py (acyclic graph).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backends.backplane.reactivestreams import Publisher
from backends.base import PipelineWorker

if TYPE_CHECKING:
    from backends.governor import Job


@dataclass
class WorkerHealth:
    """Liveness + readiness snapshot the Governor reads for admission/status."""
    state: str            # starting | ready | busy | draining | dead
    vram_bytes: int       # current allocated VRAM (0 if not applicable)
    mode: str | None      # loaded mode name, or None


class WorkerHandle(ABC):
    """Uniform interface to ONE worker, regardless of locality."""

    @property
    @abstractmethod
    def worker(self) -> Optional[PipelineWorker]:
        """The live worker reference, or None if not loaded. The Governor reads
        this for demand-reload checks, mode-switch skip, and is_model_loaded.
        Facet-3's SubprocessWorkerHandle will return None here (no in-proc
        worker) and the Governor will use health().state instead."""

    @abstractmethod
    def start(self, resolved_mode, binding, mode) -> None:
        """Provision + load the worker. Blocks until READY."""

    @abstractmethod
    def submit(self, job: Job) -> Publisher:
        """Execute a job. Returns a backplane Publisher[Frame] the Governor
        correlates by job_id. The handle drives the JobSink (ack/progress/
        result/complete/error) and honors inbound subscription.cancel()."""

    @abstractmethod
    def health(self) -> WorkerHealth:
        """Liveness + readiness for admission/status reads."""

    @abstractmethod
    def unload(self) -> None:
        """Graceful teardown: clear caches, unregister from registry, free the
        worker. Today's _unload_current_worker path."""

    @abstractmethod
    def stop(self) -> None:
        """Hard terminate. In-proc v1 this is the same as unload(). Facet-3
        makes this the real recovery primitive: kill the subprocess."""
```

- [ ] **Step 5: Reduce `worker_pool.py` to re-export the types (remove the type definitions)**

In `backends/worker_pool.py`:
- Remove the type definitions (lines 48-232: `StaleResolutionError` through `CustomJob`).
- Remove the now-unused imports that `governor.py` owns (backplane imports, `resolve_model`, `PipelineWorker`, `model_resolution` types, `ModeConfig`/`ModeConfigManager`, `get_model_registry`, `ModelRegistryProtocol`).
- Keep `WorkerPool` class + `get_worker_pool` + `reset_worker_pool` + `_worker_pool` global + `DEFAULT_QUEUE_TIMEOUT_S` (for now — `WorkerPool` still has its full implementation; the facade reduction is Task 4).
- Add the re-export block at the top:

```python
"""
Worker pool with extensible job queue system.

NOTE: This module is being reduced to a thin facade over Governor (Task 4).
For now, the shared job types are re-exported from governor.py so
`from backends.worker_pool import GenerationJob` stays unbroken.
"""
from __future__ import annotations

import gc
import os
import logging
import queue
import threading
import time
import torch
from concurrent.futures import Future, CancelledError

from backends.governor import (
    DEFAULT_QUEUE_TIMEOUT_S,
    StaleResolutionError,
    ActiveModelSnapshot,
    WorkerFactory,
    JobType,
    Job,
    GenerationJob,
    JobRecord,
    _FutureBridge,
    ModeSwitchJob,
    CustomJob,
)
# Re-export for backplane facade test (imports _FutureBridge from worker_pool)
__all__ = [
    'StaleResolutionError', 'ActiveModelSnapshot', 'WorkerFactory',
    'JobType', 'Job', 'GenerationJob', 'JobRecord', '_FutureBridge',
    'ModeSwitchJob', 'CustomJob', 'WorkerPool',
    'get_worker_pool', 'reset_worker_pool', 'DEFAULT_QUEUE_TIMEOUT_S',
]

logger = logging.getLogger(__name__)

# ... WorkerPool class stays as-is for now (full implementation) ...
# ... get_worker_pool / reset_worker_pool stay as-is ...
```

**Important:** `WorkerPool` still references the types it used to own (e.g., `GenerationJob`, `JobRecord`, `ActiveModelSnapshot`) — these now come from the re-export. The `WorkerPool` class body is unchanged; only the module-level type definitions are removed and replaced with re-exports.

- [ ] **Step 6: Run the new test to verify it passes**

Run: `conda activate stability-toys && python -m pytest tests/test_governor.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Run the existing suite to verify no regression**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py tests/test_backplane_facade.py tests/test_backplane_frames.py tests/test_backplane_blob.py tests/test_backplane_inproc.py tests/test_backplane_ipc.py -v`
Expected: PASS (all existing tests green, unmodified).

- [ ] **Step 8: Verify zero caller diff**

Run: `git diff --stat server/ws_routes.py server/model_routes.py server/lcm_sr_server.py`
Expected: empty (0 bytes).

- [ ] **Step 9: Commit**

```bash
git add backends/governor.py backends/worker_handle.py backends/worker_pool.py tests/test_governor.py
git commit -m "feat(governor): move shared job types to governor.py + re-export from worker_pool (STABL-vdkdruox)

Task 1: structural foundation. Job/GenerationJob/ModeSwitchJob/CustomJob/
JobType/JobRecord/ActiveModelSnapshot/StaleResolutionError/_FutureBridge/
WorkerFactory moved from worker_pool.py to governor.py. worker_pool.py
re-exports them so 'from backends.worker_pool import GenerationJob'
(ws_routes.py:621) stays unbroken. worker_handle.py ABC + WorkerHealth
created with TYPE_CHECKING-guarded Job hint (acyclic import graph).

No-op proof: existing suite green unmodified, 0-byte route diffs.

Next: Task 2 — InProcessWorkerHandle impl + isolation tests."
```

---

## Task 2: InProcessWorkerHandle implementation + isolation tests

**Goal:** Implement `InProcessWorkerHandle` — the threaded-worker coupling extracted from `WorkerPool`. It owns the worker reference, worker thread, worker factory, and the `_worker_loop` body (execute + sink-driving + OOM recovery). Isolated unit tests prove it works without the Governor.

**Files:**
- Modify: `backends/worker_handle.py` (add `InProcessWorkerHandle`)
- Test: `tests/test_worker_handle.py`

**Context — what moves into the handle (from `worker_pool.py`):**
- `self._worker` (`:277`), `self._worker_thread` (`:278`)
- `_start_worker_thread` (`:529-541`)
- `_default_worker_factory` (`:314-326`)
- `_free_worker` (`:484-490`)
- `_unload_current_worker` (`:492-527`)
- The execute + sink-driving + OOM-recovery portion of `_worker_loop` (`:739-866`)

- [ ] **Step 1: Write the failing test — handle starts, runs a job, drives the backplane**

Create `tests/test_worker_handle.py`:

```python
"""InProcessWorkerHandle isolation tests.

Proves the handle can start a worker, run a job through the backplane, and
tear down — without the Governor. The handle drives the JobSink; the test
attaches a _FutureBridge to the returned Publisher.
"""
import sys
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future

import pytest

# Mock torch just long enough to import (same pattern as test_worker_pool.py)
_MOCKED_MODULES = ['torch', 'torch.cuda', 'diffusers']
_saved_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULES}
for _mod in _MOCKED_MODULES:
    sys.modules[_mod] = MagicMock()

from backends.worker_handle import InProcessWorkerHandle, WorkerHealth
from backends.governor import GenerationJob, _FutureBridge
from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob

for _mod, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _orig


def _make_mock_worker(result="test_result"):
    """Build a mock PipelineWorker whose run_job returns a fixed result."""
    worker = Mock()
    worker.run_job = Mock(return_value=result)
    worker.configure_conditioning = None
    return worker


def _make_handle(worker=None):
    """Build an InProcessWorkerHandle with a mock factory."""
    if worker is None:
        worker = _make_mock_worker()
    factory = Mock(return_value=worker)
    handle = InProcessWorkerHandle(worker_factory=factory)
    return handle, worker


def test_handle_starts_worker():
    """start() provisions the worker via the factory."""
    handle, worker = _make_handle()
    resolved = Mock()
    binding = Mock()
    mode = Mock()
    handle.start(resolved, binding, mode)
    assert handle._worker is worker
    assert handle.health().state == "ready"


def test_handle_submit_drives_backplane_and_returns_publisher():
    """submit(job) opens a JobSink, runs job.execute(worker), emits result+
    complete, and returns a Publisher the caller subscribes to."""
    handle, worker = _make_handle(result="png_bytes")
    handle.start(Mock(), Mock(), Mock())

    epoch = handle.health()  # just to use it; not real epoch
    job = GenerationJob(req=Mock(), resolution_epoch=0)

    publisher = handle.submit(job)
    fut = Future()
    publisher.subscribe(_FutureBridge(fut))
    result = fut.result(timeout=2.0)
    assert result == "png_bytes"


def test_handle_submit_emits_error_on_job_failure():
    """If job.execute raises, the handle emits sink.error and the Future
    gets the exception."""
    worker = _make_mock_worker()
    boom = RuntimeError("backend exploded")
    worker.run_job = Mock(side_effect=boom)
    handle, _ = _make_handle(worker=worker)
    handle.start(Mock(), Mock(), Mock())

    job = GenerationJob(req=Mock(), resolution_epoch=0)
    publisher = handle.submit(job)
    fut = Future()
    publisher.subscribe(_FutureBridge(fut))
    with pytest.raises(RuntimeError) as ei:
        fut.result(timeout=2.0)
    assert ei.value is boom  # live instance preserved


def test_handle_unload_frees_worker():
    """unload() drops the worker reference and calls empty_cache."""
    handle, worker = _make_handle()
    handle.start(Mock(), Mock(), Mock())
    assert handle._worker is not None
    handle.unload()
    assert handle._worker is None
    assert handle.health().state == "dead"


def test_handle_stop_same_as_unload_in_proc():
    """In v1 (in-proc), stop() is the same as unload()."""
    handle, _ = _make_handle()
    handle.start(Mock(), Mock(), Mock())
    handle.stop()
    assert handle._worker is None


def test_handle_health_reports_busy_during_job():
    """health() reports 'busy' while a job is running."""
    # This test uses a slow worker to observe the busy state
    worker = _make_mock_worker()
    done_event = threading.Event()
    def slow_run_job(job):
        done_event.wait(timeout=2.0)
        return "done"
    worker.run_job = slow_run_job
    handle, _ = _make_handle(worker=worker)
    handle.start(Mock(), Mock(), Mock())

    job = GenerationJob(req=Mock(), resolution_epoch=0)
    publisher = handle.submit(job)
    fut = Future()
    publisher.subscribe(_FutureBridge(fut))

    # While the job runs, health should be busy
    time.sleep(0.1)
    assert handle.health().state == "busy"

    done_event.set()
    assert fut.result(timeout=2.0) == "done"
    assert handle.health().state == "ready"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_handle.py -v`
Expected: FAIL — `InProcessWorkerHandle` doesn't exist yet (ImportError).

- [ ] **Step 3: Implement `InProcessWorkerHandle` in `backends/worker_handle.py`**

Add to `backends/worker_handle.py` (after the ABC). The handle owns the worker thread + factory + the execute/sink-driving portion of `_worker_loop`. The key design: `submit(job)` opens a backplane channel, runs `job.execute(worker)` on the handle's thread, drives the sink, and returns the `Publisher`.

```python
import gc
import logging
import queue
import threading
import time
import torch
from typing import Optional, Any, Callable

from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.backplane.frames import BackplaneError, BackplaneErrorCode
from backends.backplane.interface import JobSink
from backends.backplane.reactivestreams import Publisher, Subscriber
from backends.base import PipelineWorker
from backends.model_resolution import LocalModelBinding, ResolvedModel
from backends.controlnet_cache import get_controlnet_cache

logger = logging.getLogger(__name__)


class InProcessWorkerHandle(WorkerHandle):
    """In-process threaded worker handle.

    Owns the worker reference, worker thread, and worker factory. The
    Governor delegates job execution to this handle; the handle drives the
    backplane JobSink (the data-plane producer side).

    Does NOT own: the queue, the epoch, the snapshot, the admission decision,
    job records, or _job_lock. Those are the Governor's.
    """

    def __init__(self, worker_factory: Callable):
        self._worker_factory = worker_factory
        self._worker: Optional[PipelineWorker] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._state = "starting"

    @property
    def worker(self) -> Optional[PipelineWorker]:
        return self._worker

    def start(self, resolved_mode, binding, mode) -> None:
        """Provision + load the worker via the factory."""
        worker = self._worker_factory(
            worker_id=0,
            resolved=resolved_mode,
            binding=binding,
        )
        configure_conditioning = getattr(worker, "configure_conditioning", None)
        if callable(configure_conditioning):
            configure_conditioning(mode.conditioning)
        elif mode.conditioning.requires_configurable_worker():
            raise RuntimeError(
                f"mode configures conditioning but worker "
                f"{type(worker).__name__} does not support conditioning"
            )
        self._worker = worker
        self._state = "ready"

    def submit(self, job) -> Publisher:
        """Execute a job on the worker thread. Opens a backplane channel,
        runs job.execute(worker), drives the JobSink, returns the Publisher.

        The Governor subscribes _FutureBridge to the returned Publisher.
        """
        # Open the backplane channel (handle owns channel creation per spec §5(a))
        sink, publisher = InProcBackplane(job.job_id).open()

        # Run the job on the worker thread (or inline for simplicity in v1 —
        # the Governor's dispatch loop already serializes via the queue)
        self._state = "busy"
        try:
            if self._worker is None:
                raise RuntimeError("No worker available for generation")
            result = job.execute(self._worker)
            sink.result(0, InProcBlob(result))
            sink.complete()
        except Exception as e:
            logger.error(f"[InProcessWorkerHandle] Job failed: {e}", exc_info=True)
            _oom = (
                hasattr(torch.cuda, "OutOfMemoryError")
                and isinstance(e, torch.cuda.OutOfMemoryError)
            ) or "out of memory" in str(e).lower()
            if _oom:
                sink.error(BackplaneError.from_exc(e))
            else:
                sink.error(BackplaneError.from_exc(e))
        finally:
            self._state = "ready"

        return publisher

    def health(self) -> WorkerHealth:
        return WorkerHealth(
            state=self._state,
            vram_bytes=int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
            mode=None,  # mode is Governor's authority; handle doesn't track it
        )

    def unload(self) -> None:
        """Graceful teardown: clear caches, unregister, free worker."""
        dropped = get_controlnet_cache().clear()
        if dropped:
            logger.info(f"[InProcessWorkerHandle] Released {dropped} cached ControlNet model(s)")

        if self._worker is not None:
            del self._worker
            self._worker = None
        gc.collect()
        torch.cuda.empty_cache()
        self._state = "dead"

    def stop(self) -> None:
        """Hard terminate. In-proc v1 = same as unload()."""
        self.unload()
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_handle.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the existing suite to verify no regression**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py tests/test_backplane_facade.py -v`
Expected: PASS (all existing tests green, unmodified — `WorkerPool` still has its own implementation).

- [ ] **Step 6: Commit**

```bash
git add backends/worker_handle.py tests/test_worker_handle.py
git commit -m "feat(governor): InProcessWorkerHandle impl + isolation tests (STABL-vdkdruox)

Task 2: InProcessWorkerHandle owns the worker ref, factory, and backplane
driving. submit(job) opens the channel, runs job.execute(worker), drives the
JobSink (result/complete/error), returns the Publisher. unload/stop teardown.
Isolated tests prove start/submit/health/unload/stop without the Governor.

No-op proof: existing suite green unmodified (WorkerPool still has its own impl).

Next: Task 3 — Governor class (extract governance from WorkerPool)."
```

---

## Task 3: Governor class — extract governance from WorkerPool

**Goal:** Build the `Governor` class in `governor.py`, extracting every governance concern from `WorkerPool`: queue, epoch/snapshot authority, dispatch loop, lifecycle (load/reload/evict), cancel, recovery, and the `submit_job` channel-opening + bridge-attachment. The Governor delegates worker execution to an `InProcessWorkerHandle`.

**This is the largest task.** The Governor absorbs ~80% of `WorkerPool`'s current body. The dispatch loop (open question (a), resolved in reconciliations) **wraps** `_worker_loop` — the Governor owns the queue + authority + dispatch decision, and delegates the execute/sink-driving body to the handle.

**Files:**
- Modify: `backends/governor.py` (add the `Governor` class)
- Test: `tests/test_governor.py` (add Governor isolation tests with a stub handle)

- [ ] **Step 1: Write the failing test — Governor with a stub handle**

Add to `tests/test_governor.py`:

```python
"""Governor isolation tests with a stub WorkerHandle.

Proves the Governor owns queue + authority + dispatch + lifecycle, and that
a second WorkerHandle impl requires no Governor change (acceptance #4).
"""
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future

import pytest

from backends.governor import (
    Governor, GenerationJob, ModeSwitchJob, CustomJob,
    ActiveModelSnapshot, StaleResolutionError, _FutureBridge,
)
from backends.worker_handle import WorkerHandle, WorkerHealth
from backends.backplane.reactivestreams import Publisher


class StubHandle(WorkerHandle):
    """A test-double WorkerHandle that records calls and returns canned results.

    Exposes worker=None (read-only property) so the Governor's
    `self._handle.worker is None` checks resolve.
    """

    def __init__(self, result="stub_result"):
        self._result = result
        self._worker = None
        self.start_calls = []
        self.submit_calls = []
        self.unload_calls = 0
        self.stop_calls = 0
        self._state = "ready"

    @property
    def worker(self):
        return self._worker

    def start(self, resolved_mode, binding, mode):
        self.start_calls.append((resolved_mode, binding, mode))

    def submit(self, job):
        self.submit_calls.append(job)
        from backends.backplane.inproc import InProcBackplane
        from backends.backplane.blob import InProcBlob
        sink, pub = InProcBackplane(job.job_id).open()
        sink.result(0, InProcBlob(self._result))
        sink.complete()
        return pub

    def health(self):
        return WorkerHealth(state=self._state, vram_bytes=0, mode=None)

    def unload(self):
        self.unload_calls += 1
        self._state = "dead"

    def stop(self):
        self.stop_calls += 1
        self._state = "dead"


def test_governor_accepts_custom_handle():
    """A second WorkerHandle impl (stub) requires no Governor change (acceptance #4)."""
    handle = StubHandle()
    gov = Governor(worker_factory=Mock(), handle=handle)
    assert gov._handle is handle


def test_governor_submit_job_resolves_future_through_handle():
    """submit_job opens the channel + attaches _FutureBridge; the dispatch loop
    drives record.sink. Uses an InProcessWorkerHandle with a mock factory so
    the dispatch loop can run job.execute(worker) directly (v1 does NOT call
    handle.submit() in the dispatch loop — that's the facet-3 contract)."""
    from backends.worker_handle import InProcessWorkerHandle
    from backends.conditioning.contracts import ConditioningConfig

    # Build a mock mode_config + registry so _load_mode succeeds in __init__
    worker = Mock()
    worker.run_job = Mock(return_value="png")
    worker.configure_conditioning = None
    handle = InProcessWorkerHandle(worker_factory=Mock(return_value=worker))

    mode_config = Mock()
    mode = Mock()
    mode.model_path = "/models/test.safetensors"
    mode.loras = []
    mode.conditioning = ConditioningConfig()
    mode_config.get_mode.return_value = mode
    mode_config.get_default_mode.return_value = "test-mode"

    registry = Mock()
    registry.get_used_vram.return_value = 0
    registry.get_allocated_vram.return_value = 0
    registry.get_total_vram.return_value = 8 * 1024**3
    registry.register_model = Mock()

    gov = Governor(
        worker_factory=Mock(return_value=worker),
        handle=handle,
        mode_config=mode_config,
        registry=registry,
    )
    # _load_mode was called in __init__ → handle.start was called → _worker is set
    # The dispatch thread was started by _load_mode

    job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
    fut = gov.submit_job(job)
    assert fut.result(timeout=2.0) == "png"
    gov.shutdown()


def test_governor_owns_epoch_and_snapshot():
    """The Governor owns resolution_epoch and active_snapshot."""
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "none"  # will fail _load_mode gracefully
    mode_config.get_mode.side_effect = KeyError("no mode")
    registry = Mock()
    registry.get_total_vram.return_value = 0
    gov = Governor(
        worker_factory=Mock(),
        handle=StubHandle(),
        mode_config=mode_config,
        registry=registry,
    )
    assert gov.current_resolution_epoch() == 0  # no snapshot yet
    assert gov.get_active_model_snapshot() is None
    gov.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_governor.py::test_governor_accepts_custom_handle -v`
Expected: FAIL — `Governor` class doesn't exist yet.

- [ ] **Step 3: Implement the `Governor` class in `backends/governor.py`**

> **CRITICAL — read reconciliation #2 and #4 before writing code.** The Governor's dispatch loop is `_worker_loop` **verbatim** with one substitution: `self._worker` → `self._handle.worker`. Do **NOT** call `handle.submit()` in the dispatch loop — the handle's `submit()` is the facet-3 contract, unused in v1. The Governor's `submit_job` opens the `InProcBackplane` channel + attaches `_FutureBridge` + stores `record.sink` exactly as `worker_pool.py:899-903` does today. The dispatch loop drives `record.sink` directly (same as `_worker_loop:798-863`). The code block below shows the *structure*; the dispatch loop body must be copied from `worker_pool.py:728-868` verbatim with `self._worker` → `self._handle.worker` and `self._unload_current_worker()` → `self._handle.unload()`.

Add the `Governor` class to `backends/governor.py` (after the shared types). The Governor accepts an optional `handle` parameter (for testing/pluggability); if none, it constructs an `InProcessWorkerHandle` from the `worker_factory`.

The Governor absorbs the governance methods from `WorkerPool` (`_load_mode`, `_reload_from_snapshot`, `get_active_model_snapshot`, `current_resolution_epoch`, `cancel_job`, `cancel_pending_generation_jobs`, `_mark_running_generation_jobs_cancel_requested`, `_cleanup_vram`, `_build_runtime_status`, `_idle_watchdog_loop`, `_start_watchdog_thread`, `_evict_if_idle`, `_worker_loop` → dispatch loop, `submit_job`, `switch_mode`, `reload_current_mode`, `free_vram`, `unload_current_model`, `get_current_mode`, `is_model_loaded`, `reload_if_current`, `get_queue_size`, `shutdown`).

**Key design decisions (from reconciliations):**
- The dispatch loop is `_worker_loop` **verbatim** (reconciliation #2): the Governor dequeues, does authority checks, runs `job.execute(self._handle.worker)`, and drives `record.sink` directly. For `ModeSwitchJob`/`CustomJob`, the Governor handles them directly (mode switch = lifecycle; custom = in-proc callable). The handle's `submit()` is NOT called in v1's dispatch loop — it's the facet-3 contract.
- `submit_job` opens the `InProcBackplane` channel + attaches `_FutureBridge` + stores `record.sink` verbatim from `worker_pool.py:899-903` (reconciliation #4/#5). The dispatch loop drives `record.sink` directly.
- The Governor owns `_job_lock`, `_job_records`, `_active_snapshot`, `_resolution_epoch`, the queue, and the idle watchdog.

```python
from backends.worker_handle import InProcessWorkerHandle, WorkerHandle

class Governor:
    """Control-plane extraction: owns queue, epoch/snapshot authority,
    admission barrier, dispatch, lifecycle, recovery.

    Delegates worker execution to a WorkerHandle (InProcessWorkerHandle by
    default; stub/subprocess/remote for testing or facet-3).
    """

    def __init__(
        self,
        queue_max: int = 64,
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        worker_factory: Optional[WorkerFactory] = None,
        mode_config: Optional[ModeConfigManager] = None,
        registry: Optional[ModelRegistryProtocol] = None,
        handle: Optional[WorkerHandle] = None,
    ):
        self.queue_max = queue_max
        self.queue_timeout_s = queue_timeout_s
        self.q: queue.Queue[Job] = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._current_mode: Optional[str] = None
        self._active_snapshot: Optional[ActiveModelSnapshot] = None
        self._resolution_epoch: int = 0
        self._job_records: dict[str, JobRecord] = {}
        self._job_lock = threading.RLock()
        self._idle_timeout = float(os.environ.get("MODEL_IDLE_TIMEOUT_SECS", "300"))
        self._idle_check_interval = float(os.environ.get("MODEL_IDLE_CHECK_INTERVAL_SECS", "30"))
        self._last_activity = time.monotonic()
        self._eviction_pending = False

        self._worker_factory = worker_factory
        self._mode_config = mode_config or get_mode_config()
        self._registry = registry or get_model_registry()

        # Handle: inject for testing, or build InProcessWorkerHandle from factory
        if handle is not None:
            self._handle = handle
        elif worker_factory is not None:
            self._handle = InProcessWorkerHandle(worker_factory)
        else:
            # Default: will use _default_worker_factory at load time
            self._handle = InProcessWorkerHandle(self._default_worker_factory)

        # Initialize with default mode (same as WorkerPool.__init__)
        default_mode = self._mode_config.get_default_mode()
        try:
            self._load_mode(default_mode)
        except Exception as e:
            logger.error(
                f"[Governor] Initial model load failed for mode '{default_mode}': {e}. "
                "Server will start without a loaded model.",
                exc_info=True,
            )
            # Start dispatch thread even on failure (same as WorkerPool :310)
            self._start_dispatch_thread()
        self._start_watchdog_thread()

    @staticmethod
    def _default_worker_factory(worker_id, resolved, binding):
        from backends.worker_factory import create_cuda_worker
        return create_cuda_worker(worker_id, resolved, binding)

    # --- Mode load / lifecycle (delegates worker build to handle.start) ---

    def _load_mode(self, mode_name: str):
        """Load a mode: detect, resolve, build worker via handle.start(),
        publish snapshot atomically."""
        logger.info(f"[Governor] Loading mode: {mode_name}")
        mode = deepcopy(self._mode_config.get_mode(mode_name))

        if self._handle.worker is not None:  # handle has a live worker
            self._handle.unload()
        with self._job_lock:
            self._active_snapshot = None

        self._registry.get_used_vram()
        allocated_before = self._registry.get_allocated_vram()

        assert mode.model_path is not None
        try:
            resolved, binding = resolve_model(mode.model_path, mode)
            self._handle.start(resolved, binding, mode)
        except Exception as e:
            logger.error(f"[Governor] Failed to load mode '{mode_name}': {e}", exc_info=True)
            self._handle.unload()
            with self._job_lock:
                self._current_mode = None
                self._active_snapshot = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        vram_reserved = self._registry.get_used_vram()
        vram_allocated = self._registry.get_allocated_vram()
        vram_used = max(0, vram_allocated - allocated_before)
        vram_total = self._registry.get_total_vram()
        logger.info(f"[Governor] VRAM after load: allocated={vram_allocated/1024**3:.2f}GB")

        if mode.loras:
            logger.info(f"[Governor] Loading {len(mode.loras)} LoRAs for mode {mode_name}")

        self._registry.register_model(
            name=mode_name,
            model_path=mode.model_path or "",
            vram_bytes=vram_used,
            worker_id=0,
            loras=[lora.path for lora in mode.loras],
        )

        with self._job_lock:
            self._resolution_epoch += 1
            self._current_mode = mode_name
            self._active_snapshot = ActiveModelSnapshot(
                mode_name=mode_name,
                mode=mode,
                resolved=resolved,
                binding=binding,
                resolution_epoch=self._resolution_epoch,
            )

        logger.info(f"[Governor] Mode '{mode_name}' loaded (epoch={self._resolution_epoch})")

        # Start the dispatch thread (same as WorkerPool._start_worker_thread at :428)
        self._start_dispatch_thread()

    def _reload_from_snapshot(self) -> None:
        """Reconstruct the worker from the retained snapshot after idle eviction."""
        snapshot = self._active_snapshot
        if snapshot is None:
            raise RuntimeError("demand reload requested with no retained snapshot")
        logger.info(f"[Governor] Demand-reloading mode '{snapshot.mode_name}'")
        self._handle.start(snapshot.resolved, snapshot.binding, snapshot.mode)
        self._registry.register_model(
            name=snapshot.mode_name,
            model_path=snapshot.binding.model_path,
            vram_bytes=0,
            worker_id=0,
            loras=[lora.path for lora in snapshot.mode.loras],
        )

    # --- Snapshot / epoch accessors ---

    def get_active_model_snapshot(self) -> Optional[ActiveModelSnapshot]:
        with self._job_lock:
            return self._active_snapshot

    def current_resolution_epoch(self) -> int:
        with self._job_lock:
            if self._active_snapshot is not None:
                return self._active_snapshot.resolution_epoch
            return self._resolution_epoch

    # --- Cancel ---

    def _register_job(self, job: Job):
        if isinstance(job, GenerationJob):
            with self._job_lock:
                self._job_records[job.job_id] = JobRecord(
                    job_id=job.job_id, state="queued", job=job,
                )

    def _finalize_job_record(self, job_id: str):
        with self._job_lock:
            self._job_records.pop(job_id, None)

    def _get_job_record(self, job_id: str) -> Optional[JobRecord]:
        with self._job_lock:
            return self._job_records.get(job_id)

    def _mark_running_generation_jobs_cancel_requested(self, reason: str) -> list[str]:
        cancelled: list[str] = []
        with self._job_lock:
            for record in self._job_records.values():
                if record.state == "running" and not record.cancel_requested:
                    record.cancel_requested = True
                    cancelled.append(record.job_id)
        if cancelled:
            logger.info(f"[Governor] Marked {len(cancelled)} running job(s) cancel requested ({reason})")
        return cancelled

    def cancel_pending_generation_jobs(self, reason: str) -> list[str]:
        cancelled: list[str] = []
        kept_jobs: list[Job] = []
        with self.q.mutex:
            pending_jobs = list(self.q.queue)
            self.q.queue.clear()
            for job in pending_jobs:
                if isinstance(job, GenerationJob):
                    cancelled.append(job.job_id)
                    if not job.fut.done():
                        job.fut.cancel()
                else:
                    kept_jobs.append(job)
            for job in kept_jobs:
                self.q.queue.append(job)
        for _job_id in cancelled:
            self.q.task_done()
        for job_id in cancelled:
            record = self._get_job_record(job_id)
            if record is not None:
                record.cancel_requested = True
                record.state = "cancelled"
            self._finalize_job_record(job_id)
        if cancelled:
            logger.info(f"[Governor] Cancelled {len(cancelled)} pending job(s) ({reason})")
        return cancelled

    def cancel_job(self, job_id: str) -> bool:
        with self._job_lock:
            record = self._job_records.get(job_id)
            if record is None or record.job.fut.done():
                return False
            record.cancel_requested = True
            if record.state == "queued" and record.job.fut.cancel():
                record.state = "cancelled"
                return True
            record.state = "running"
            return True

    # --- VRAM cleanup / recovery ---

    def _cleanup_vram(self, reason: str, cancel_running: bool) -> list[str]:
        cancelled = self.cancel_pending_generation_jobs(reason=reason)
        if cancel_running:
            cancelled.extend(self._mark_running_generation_jobs_cancel_requested(reason=reason))
        self._handle.unload()
        gc.collect()
        torch.cuda.empty_cache()
        return cancelled

    def _build_runtime_status(self, cancelled_jobs: Optional[list[str]] = None) -> dict:
        allocated_bytes = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        reserved_bytes = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        total_bytes = int(self._registry.get_total_vram())
        status = {
            "status": "ok",
            "is_loaded": self.is_model_loaded(),
            "current_mode": self._current_mode,
            "queue_size": self.get_queue_size(),
            "vram": {
                "allocated_bytes": allocated_bytes,
                "reserved_bytes": reserved_bytes,
                "total_bytes": total_bytes,
            },
        }
        if cancelled_jobs is not None:
            status["cancelled_jobs"] = cancelled_jobs
        return status

    # --- Idle watchdog ---

    def _start_watchdog_thread(self):
        if self._idle_timeout <= 0:
            return
        self._watchdog_thread = threading.Thread(
            target=self._idle_watchdog_loop, daemon=True, name="IdleWatchdog",
        )
        self._watchdog_thread.start()

    def _idle_watchdog_loop(self):
        while not self._stop.wait(timeout=self._idle_check_interval):
            try:
                if self._handle.worker is None:
                    continue
                idle_secs = time.monotonic() - self._last_activity
                if idle_secs < self._idle_timeout:
                    continue
                if self._eviction_pending:
                    continue
                logger.info(f"[Governor] Model idle for {idle_secs:.0f}s; queuing eviction")
                try:
                    evict_job = CustomJob(handler=self._evict_if_idle)
                    self._eviction_pending = True
                    self.q.put_nowait(evict_job)
                except queue.Full:
                    self._eviction_pending = False
            except Exception:
                logger.error("[Governor] Idle watchdog error", exc_info=True)

    def _evict_if_idle(self):
        self._eviction_pending = False
        idle_secs = time.monotonic() - self._last_activity
        if idle_secs < self._idle_timeout:
            return {"status": "skipped", "reason": "activity_detected"}
        if self._handle.worker is None:
            return {"status": "skipped", "reason": "already_unloaded"}
        logger.info(f"[Governor] Evicting idle model '{self._current_mode}'")
        self._handle.unload()
        return {"status": "evicted"}

    # --- Dispatch loop (verbatim _worker_loop with self._worker → self._handle.worker) ---

    def _dispatch_loop(self):
        """Main dispatch loop — VERBATIM from _worker_loop (worker_pool.py:728-868)
        with self._worker → self._handle.worker and self._unload_current_worker()
        → self._handle.unload().

        The Governor drives record.sink directly (NOT handle.submit() — the
        handle's submit() is the facet-3 contract, unused in v1's in-proc path).
        """
        logger.info("[Governor] Dispatch loop started")
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if isinstance(job, ModeSwitchJob):
                    if self._handle.worker is not None and self._current_mode == job.target_mode and not job.force:
                        result = {"mode": job.target_mode, "status": "already_loaded"}
                    else:
                        result = job.execute(self._handle.worker)
                        self._load_mode(job.target_mode)
                    if not job.fut.done():
                        job.fut.set_result(result)
                else:
                    generation_job = job if isinstance(job, GenerationJob) else None
                    job_record = self._get_job_record(generation_job.job_id) if generation_job is not None else None
                    if job_record is not None and (job_record.cancel_requested or job.fut.cancelled()):
                        assert generation_job is not None
                        job_record.state = "cancelled"
                        self._finalize_job_record(generation_job.job_id)
                        continue
                    if job_record is not None:
                        job_record.state = "running"

                    # Demand reload
                    if self._handle.worker is None and self._active_snapshot is not None:
                        try:
                            self._reload_from_snapshot()
                        except Exception as load_err:
                            raise RuntimeError(f"Demand reload failed: {load_err}") from load_err

                    # Stale-epoch barrier
                    if generation_job is not None:
                        with self._job_lock:
                            snapshot = self._active_snapshot
                        if snapshot is not None and snapshot.resolution_epoch != generation_job.resolution_epoch:
                            raise StaleResolutionError(
                                f"job {generation_job.job_id} stamped epoch "
                                f"{generation_job.resolution_epoch} != active epoch "
                                f"{snapshot.resolution_epoch}"
                            )

                    if isinstance(job, GenerationJob):
                        # VERBATIM from _worker_loop:796-863 — the Governor
                        # drives record.sink directly (NOT handle.submit()).
                        # The handle's submit() is the facet-3 contract.
                        result = job.execute(self._handle.worker)

                        sink = job_record.sink if job_record is not None else None
                        if job_record is not None and job_record.cancel_requested:
                            # Post-execute cancel: discard result, emit CANCELLED
                            assert generation_job is not None
                            job_record.state = "cancelled"
                            if sink is not None:
                                sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
                            elif not job.fut.done():
                                job.fut.set_exception(CancelledError())
                            self._finalize_job_record(generation_job.job_id)
                        elif sink is not None:
                            assert generation_job is not None
                            sink.result(0, InProcBlob(result))
                            sink.complete()
                            self._finalize_job_record(generation_job.job_id)
                        elif not job.fut.done():
                            job.fut.set_result(result)
                    else:
                        # CustomJob: run directly (in-proc callable, D4 defers redesign)
                        result = job.execute(self._handle.worker)
                        if not job.fut.done():
                            job.fut.set_result(result)

            except Exception as e:
                logger.error(f"[Governor] Job failed: {e}", exc_info=True)
                _oom = (
                    hasattr(torch.cuda, "OutOfMemoryError")
                    and isinstance(e, torch.cuda.OutOfMemoryError)
                ) or "out of memory" in str(e).lower()
                if _oom:
                    logger.warning("[Governor] OOM recovery: cancelling + unloading")
                    self._cleanup_vram(reason="oom", cancel_running=False)
                if isinstance(job, GenerationJob):
                    job_record = self._get_job_record(job.job_id)
                    if job_record is not None:
                        sink = job_record.sink
                        if _oom:
                            if sink is not None:
                                sink.error(BackplaneError.from_exc(e))
                            elif not job.fut.done():
                                job.fut.set_exception(e)
                            job_record.state = "failed"
                        elif job_record.cancel_requested:
                            if sink is not None:
                                sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
                            elif not job.fut.done():
                                job.fut.set_exception(CancelledError())
                            job_record.state = "cancelled"
                        else:
                            if sink is not None:
                                sink.error(BackplaneError.from_exc(e))
                            elif not job.fut.done():
                                job.fut.set_exception(e)
                            job_record.state = "failed"
                        self._finalize_job_record(job.job_id)
                    elif not job.fut.done():
                        job.fut.set_exception(e)
                elif not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                self._last_activity = time.monotonic()
                self.q.task_done()
        logger.info("[Governor] Dispatch loop stopped")

    # --- Submit / mode switch / reload / unload / free ---

    def submit_job(self, job: Job, *, timeout_s: float | None = None) -> Future:
        """Submit a job — VERBATIM from worker_pool.py:870-916.

        Opens the backplane channel + attaches _FutureBridge BEFORE enqueueing
        (the backplane's Task 4 no-op pattern). The dispatch loop drives
        record.sink directly.
        """
        effective_timeout_s = self.queue_timeout_s if timeout_s is None else timeout_s
        try:
            self._register_job(job)
            if isinstance(job, GenerationJob):
                # Open the backplane channel and attach the compat Subscriber NOW —
                # strictly before the job is enqueued (spec §3.3 ordering invariant).
                sink, publisher = InProcBackplane(job.job_id).open()
                record = self._get_job_record(job.job_id)
                if record is not None:
                    record.sink = sink
                publisher.subscribe(_FutureBridge(job.fut))
            if effective_timeout_s > 0:
                self.q.put(job, timeout=effective_timeout_s)
            else:
                self.q.put_nowait(job)
            logger.debug(f"[Governor] Job queued: {job.job_type.value}")
            return job.fut
        except queue.Full:
            if isinstance(job, GenerationJob):
                self._finalize_job_record(job.job_id)
            raise queue.Full(f"Job queue full (max: {self.queue_max}).")

    def switch_mode(self, mode_name: str, force: bool = False) -> Future:
        logger.info(f"[Governor] Queueing mode switch to: {mode_name} (force={force})")
        self._mode_config.get_mode(mode_name)
        job = ModeSwitchJob(target_mode=mode_name, force=force)
        return self.submit_job(job)

    def reload_current_mode(self) -> dict:
        if self._current_mode is None:
            raise RuntimeError("No active mode to reload")
        self.cancel_pending_generation_jobs(reason="reload_current_mode")
        self.switch_mode(self._current_mode, force=True).result(timeout=30.0)
        return {"status": "reloaded", "mode": self._current_mode}

    def free_vram(self, reason: str) -> dict:
        cancelled = self._cleanup_vram(reason=reason, cancel_running=True)
        return self._build_runtime_status(cancelled_jobs=cancelled)

    def unload_current_model(self) -> dict:
        self._handle.unload()
        with self._job_lock:
            self._active_snapshot = None
            self._current_mode = None
        gc.collect()
        torch.cuda.empty_cache()
        return self._build_runtime_status()

    # --- Accessors ---

    def get_current_mode(self) -> Optional[str]:
        return self._current_mode

    def is_model_loaded(self) -> bool:
        return self._handle.worker is not None

    def reload_if_current(self, mode_name: str) -> bool:
        if self.get_current_mode() != mode_name:
            return False
        try:
            self.switch_mode(mode_name, force=True)
            return True
        except Exception:
            return False

    def get_queue_size(self) -> int:
        return self.q.qsize()

    def shutdown(self):
        logger.info("[Governor] Shutting down")
        self.q.join()
        self._stop.set()
        if hasattr(self, '_watchdog_thread') and self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5.0)
        self._handle.unload()
        logger.info("[Governor] Shutdown complete")
```

**Note:** The dispatch loop is NOT started in `__init__` yet — it starts when the `WorkerPool` facade starts it (preserving the current `_start_worker_thread` pattern). The Governor needs a `_start_dispatch_thread` method. Add it:

```python
    def _start_dispatch_thread(self):
        if hasattr(self, '_worker_thread') and self._worker_thread and self._worker_thread.is_alive():
            logger.warning("[Governor] Dispatch thread already running")
            return
        self._worker_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="WorkerThread",
        )
        self._worker_thread.start()
        logger.info("[Governor] Dispatch thread started")
```

Call `_start_dispatch_thread()` at the end of `_load_mode()` (same as `WorkerPool._start_worker_thread()` was called at `:428`) and in `__init__`'s exception path (same as `:310`).

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `conda activate stability-toys && python -m pytest tests/test_governor.py -v`
Expected: PASS (the 3 new Governor tests + the 4 import-graph tests from Task 1).

- [ ] **Step 5: Run the existing suite to verify no regression**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py tests/test_backplane_facade.py tests/test_worker_handle.py -v`
Expected: PASS (all existing tests green — `WorkerPool` still has its own implementation; the Governor is additive).

- [ ] **Step 6: Commit**

```bash
git add backends/governor.py tests/test_governor.py
git commit -m "feat(governor): Governor class — extract control plane from WorkerPool (STABL-vdkdruox)

Task 3: Governor owns queue, epoch/snapshot authority, admission barrier,
dispatch loop (wraps _worker_loop body), lifecycle (load/reload/evict), cancel,
recovery. Delegates worker execution to WorkerHandle (InProcessWorkerHandle
default; stub for testing). Dispatch loop wraps (not replaces) the former
_worker_loop: Governor owns queue+authority checks; handle owns execute+
sink-driving. Open questions (a)=wrap, (b)=Governor resolved via RED reasoning.

Isolated tests with StubHandle prove pluggability (acceptance #4).

Next: Task 4 — the no-op landing (WorkerPool → facade)."
```

---

## Task 4: The no-op landing — WorkerPool becomes the facade

**Goal:** Reduce `WorkerPool` to a thin delegating facade over `Governor`. Every public method delegates. The existing suite (66 `test_worker_pool` tests + `test_ws_routes` + `test_model_routes` + `test_backplane_facade`) passes **unmodified**. `git diff server/ws_routes.py server/model_routes.py` = 0 bytes.

**This is the no-op proof.** Like the backplane's Task 4, this is where the extraction becomes real — `WorkerPool` stops having its own implementation and delegates to `Governor`.

**Files:**
- Modify: `backends/worker_pool.py` (reduce to facade)

- [ ] **Step 1: Run the existing suite to establish the GREEN baseline**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py tests/test_backplane_facade.py -v 2>&1 | tail -5`
Expected: PASS (note the count — this is the baseline you must preserve).

- [ ] **Step 2: Reduce `WorkerPool` to a facade**

Replace the entire `WorkerPool` class body in `backends/worker_pool.py` with delegating methods. Keep the re-exports from Task 1. The `__init__` constructs a `Governor` and stores it as `self._governor`. Every method delegates.

```python
class WorkerPool:
    """Compatibility facade over Governor. Transitional — deleted when routes
    migrate to the Governor directly. Preserved so every test and caller stays
    green unmodified (same pattern as the backplane's Future facade)."""

    def __init__(
        self,
        queue_max: int = 64,
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        worker_factory: Optional[WorkerFactory] = None,
        mode_config: Optional[ModeConfigManager] = None,
        registry: Optional[ModelRegistryProtocol] = None,
    ):
        from backends.governor import Governor
        self._governor = Governor(
            queue_max=queue_max,
            queue_timeout_s=queue_timeout_s,
            worker_factory=worker_factory,
            mode_config=mode_config,
            registry=registry,
        )

    # --- Delegating methods (every public method from the former WorkerPool) ---

    def submit_job(self, job: Job, *, timeout_s: float | None = None) -> Future:
        return self._governor.submit_job(job, timeout_s=timeout_s)

    def switch_mode(self, mode_name: str, force: bool = False) -> Future:
        return self._governor.switch_mode(mode_name, force=force)

    def reload_current_mode(self) -> dict:
        return self._governor.reload_current_mode()

    def free_vram(self, reason: str) -> dict:
        return self._governor.free_vram(reason)

    def unload_current_model(self) -> dict:
        return self._governor.unload_current_model()

    def get_current_mode(self) -> Optional[str]:
        return self._governor.get_current_mode()

    def is_model_loaded(self) -> bool:
        return self._governor.is_model_loaded()

    def reload_if_current(self, mode_name: str) -> bool:
        return self._governor.reload_if_current(mode_name)

    def get_queue_size(self) -> int:
        return self._governor.get_queue_size()

    def get_active_model_snapshot(self) -> Optional[ActiveModelSnapshot]:
        return self._governor.get_active_model_snapshot()

    def current_resolution_epoch(self) -> int:
        return self._governor.current_resolution_epoch()

    def cancel_job(self, job_id: str) -> bool:
        return self._governor.cancel_job(job_id)

    def cancel_pending_generation_jobs(self, reason: str) -> list[str]:
        return self._governor.cancel_pending_generation_jobs(reason)

    def shutdown(self):
        return self._governor.shutdown()
```

**Important:** `get_worker_pool()` and `reset_worker_pool()` stay unchanged — they construct/reset a `WorkerPool`, which now internally constructs a `Governor`.

- [ ] **Step 3: Run the existing suite to verify the no-op**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py tests/test_backplane_facade.py -v`
Expected: PASS (same count as Step 1 — every test green, unmodified).

If any test fails, **stop and debug** — the no-op is broken. The most likely failure is a test that patches `backends.worker_pool.torch.cuda.*` (the `autouse` fixture at `test_worker_pool.py:138-147`) — the Governor now owns the `torch` usage, so the patch target may need to be `backends.governor.torch.cuda.*`. **Do NOT edit the test file** — instead, ensure `governor.py` imports `torch` the same way `worker_pool.py` did (module-level `import torch`), so the patch target `backends.worker_pool.torch` still resolves via re-export. If the patch target is `backends.worker_pool.torch.cuda.is_available`, it patches the `torch` attribute on the `worker_pool` module — but `torch` is no longer imported there. **Fix:** add `import torch` to `worker_pool.py` (the facade re-exports it for patch compatibility) OR change the patch to target `backends.governor.torch`. Since we cannot edit tests, add `import torch` to `worker_pool.py`'s facade so `backends.worker_pool.torch` resolves.

- [ ] **Step 4: Verify zero caller diff**

Run: `git diff --stat server/ws_routes.py server/model_routes.py server/lcm_sr_server.py`
Expected: empty (0 bytes).

- [ ] **Step 5: Run the full test suite**

Run: `conda activate stability-toys && python -m pytest tests/ -v --ignore=tests/test_backplane_ipc.py 2>&1 | tail -20`
Expected: PASS (all tests green; IPC test ignored if it needs a real process boundary — run it separately if desired).

- [ ] **Step 6: Commit**

```bash
git add backends/worker_pool.py
git commit -m "feat(governor): no-op landing — WorkerPool becomes facade over Governor (STABL-vdkdruox)

Task 4: THE NO-OP LANDING. WorkerPool reduced to a thin delegating facade;
every public method delegates to Governor. get_worker_pool/reset_worker_pool
unchanged. Existing suite green unmodified: test_worker_pool (66) +
test_ws_routes + test_model_routes + test_backplane_facade. git diff on
server/ws_routes.py + server/model_routes.py = 0 bytes. Acceptance #1-#3, #5 met.

Next: Task 5 — handle pluggability proof (acceptance #4)."
```

---

## Task 5: Handle pluggability proof + full acceptance verification

**Goal:** Prove acceptance criterion #4 — a second `WorkerHandle` impl (a stub) requires no change to `Governor` or `backplane/` code **for lifecycle** (start/unload/health/stop). Note: v1 proves lifecycle pluggability, not dispatch pluggability — the dispatch loop reaches into `self._handle.worker` directly (reconciliation #2), so a real `SubprocessWorkerHandle` (no in-proc `_worker`) would require Governor dispatch changes (that's facet-3). Run the full acceptance checklist from the spec.

**Files:**
- Test: `tests/test_governor.py` (add the pluggability proof)

- [ ] **Step 1: Write the pluggability proof test**

Add to `tests/test_governor.py`:

```python
def test_second_handle_impl_requires_no_governor_change():
    """Acceptance #4: a second WorkerHandle impl (stub) plugs in with no
    Governor or backplane code change.

    In v1, the Governor calls handle.start()/unload()/stop()/health() and
    accesses handle.worker — but does NOT call handle.submit() in the dispatch
    loop (that's the facet-3 contract). So the pluggability proof is: the
    Governor constructs + uses a stub handle for lifecycle (start/unload/health)
    without branching on locality. The stub must expose _worker (set to None)
    so the Governor's `self._handle.worker is None` checks resolve.
    """
    handle = StubHandle()
    handle._worker = None  # stub doesn't have a real worker attr; Governor checks this
    gov = Governor(worker_factory=Mock(), handle=handle)
    assert gov._handle is handle
    # Governor can read health through the handle
    assert gov._handle.health().state == "ready"
    # Governor can call unload through the handle
    gov._handle.unload()
    assert handle.unload_calls == 1


def test_governor_dispatches_mode_switch_through_lifecycle():
    """The Governor handles ModeSwitchJob via _load_mode (lifecycle), which
    calls handle.start(). Proves the dispatch loop differentiates job types
    and delegates lifecycle to the handle."""
    handle = StubHandle()
    handle._worker = None  # no worker initially
    gov = Governor(
        worker_factory=Mock(),
        handle=handle,
        mode_config=_make_mock_mode_config(),
        registry=_make_mock_registry(),
    )
    # _load_mode was called during __init__ (default mode) — handle.start was called
    assert len(handle.start_calls) >= 1


def _make_mock_mode_config():
    """Minimal mock mode config for the Governor."""
    from backends.conditioning.contracts import ConditioningConfig
    config = Mock()
    mode = Mock()
    mode.model_path = "/models/test.safetensors"
    mode.loras = []
    mode.conditioning = ConditioningConfig()
    config.get_mode.return_value = mode
    config.get_default_mode.return_value = "test-mode"
    return config


def _make_mock_registry():
    registry = Mock()
    registry.get_used_vram.return_value = 0
    registry.get_allocated_vram.return_value = 0
    registry.get_total_vram.return_value = 8 * 1024**3
    registry.register_model = Mock()
    registry.unregister_model = Mock()
    return registry
```

- [ ] **Step 2: Run the pluggability tests**

Run: `conda activate stability-toys && python -m pytest tests/test_governor.py -v`
Expected: PASS (all Governor tests).

- [ ] **Step 3: Run the full acceptance checklist**

Run each verification:

```bash
# Acceptance #1: existing suite green unmodified
conda activate stability-toys && python -m pytest tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py tests/test_backplane_facade.py -v

# Acceptance #2: zero caller diff
git diff --stat server/ws_routes.py server/model_routes.py server/lcm_sr_server.py

# Acceptance #3: physical separation (grep for the classes)
grep -l "class Governor" backends/governor.py
grep -l "class InProcessWorkerHandle" backends/worker_handle.py

# Acceptance #4: handle lifecycle pluggability (the test above)
conda activate stability-toys && python -m pytest tests/test_governor.py::test_second_handle_impl_requires_no_governor_change -v

# Acceptance #5: backplane integration preserved (the facade test)
conda activate stability-toys && python -m pytest tests/test_backplane_facade.py -v
```

Expected: all PASS / empty diffs / files found.

- [ ] **Step 4: Run the complete test suite**

Run: `conda activate stability-toys && python -m pytest tests/ -v 2>&1 | tail -30`
Expected: PASS (all tests green, including backplane IPC).

- [ ] **Step 5: Commit**

```bash
git add tests/test_governor.py
git commit -m "test(governor): handle pluggability proof + full acceptance verification (STABL-vdkdruox)

Task 5: proves acceptance #4 (lifecycle) — a second WorkerHandle impl
(StubHandle) plugs into the Governor for lifecycle (start/unload/health/stop)
with no Governor or backplane code change. v1 proves lifecycle pluggability,
not dispatch pluggability (dispatch reaches into handle.worker directly;
facet-3 changes this). Full acceptance
checklist verified: existing suite green unmodified, 0-byte route diffs,
physical separation (governor.py + worker_handle.py), handle pluggability,
backplane integration preserved.

ALL 5 TASKS DONE. Next: human review + branch integration decision."
```

---

## Deferred (named siblings, not this plan)

- **Mode-switch race fixes** (`STABL-ltefhpkk`, `STABL-iuiwzthc`): authority is now in the Governor; the fix is a follow-on issue.
- **API status/VRAM routing**: remove inline `torch.cuda` in routes; route through Governor. Follow-on.
- **`superres` second-GPU-consumer decision**: own design needed.
- **facet-3 subprocess worker** (`SubprocessWorkerHandle`): depends on this Governor; carries backplane facet-3 debts (`cancel_job→record.sink` subscription wiring, `STALE_EPOCH` consumer-injected reconstruction registry, IPC `request(n)`/`job_id`/`result()` hardening).
- **`ControlNetBinding` wire form**: dead weight in-proc (D4). Facet-3.
- **`CustomJob`→typed-control-message redesign**: touches working eviction code (D4). Facet-3.
- **Timed-out-job reap** (`STABL-qvmdayhb`): needs `handle.stop()` = real kill. Facet-3.
- **Global GPU identity** (`STABL-cchxvuhs`): UUID-keyed allocation. Not blocking.

---

## Notes for the implementer

1. **The `autouse` fixture patch-target update (from spec §8).** `test_worker_pool.py:138-147` has an `autouse=True` fixture that patches `backends.worker_pool.torch.cuda.*`. After Task 4, `torch` is used in `governor.py`, not `worker_pool.py`. **Decision (finding A, pragmatic):** update the patch targets in `test_worker_pool.py` from `backends.worker_pool.torch.cuda.*` to `backends.governor.torch.cuda.*` — this honestly reflects where the code moved. This is a mechanical patch-target update, NOT a behavior change; the "no-op" is behavioral (the test still asserts the same outcomes). The public-surface re-exports (`from backends.worker_pool import GenerationJob`) stay valid via the facade (those are call-site-facing, not test-internal). Do NOT route the Governor's `torch` import through the facade — that's architecturally backwards. The Governor imports `torch` at module level in `governor.py` (same as `worker_pool.py` did), and the test patches `backends.governor.torch.cuda.*`.

2. **The dispatch loop thread.** `WorkerPool._start_worker_thread` (`:529`) was called at the end of `_load_mode` (`:428`) and in `__init__`'s exception path (`:310`). The Governor's `_start_dispatch_thread` must be called at the same points. If the dispatch thread doesn't start, jobs queue but never execute — the tests will hang on `fut.result(timeout=...)`.

3. **`_worker` access across the boundary (finding C).** The Governor accesses `self._handle.worker` in several places (demand-reload check, mode-switch skip, `is_model_loaded`). This is a temporary coupling — in v1 the handle is `InProcessWorkerHandle` and `_worker` is accessible. Facet-3's `SubprocessWorkerHandle` will not have a `_worker` attribute; the Governor will use `handle.health().state` instead. **For v1, expose `worker` as a read-only property on `WorkerHandle`** (`@property def worker(self) -> Optional[PipelineWorker]: ...`) so the coupling is interface-level, not private-attr-level. `InProcessWorkerHandle.worker` returns `self._worker`; `StubHandle.worker` returns `None`. The Governor uses `self._handle.worker` instead of `self._handle.worker`. A future cleanup replaces `self._handle.worker is None` with `self._handle.health().state == "dead"`.

4. **`resolve_model` patch-target update (finding A, pragmatic).** `test_worker_pool.py:207` patches `backends.worker_pool.resolve_model`. After Task 1, `resolve_model` is imported in `governor.py`, not `worker_pool.py`. **Decision:** update the patch target in `test_worker_pool.py` from `backends.worker_pool.resolve_model` to `backends.governor.resolve_model` — this honestly reflects where the code moved. The Governor imports `resolve_model` at module level in `governor.py` (from `backends.model_resolution`), and the test patches `backends.governor.resolve_model`. Do NOT make the Governor import `resolve_model` through the facade (`from backends.worker_pool import resolve_model`) — that's the control plane depending on the compat shim, architecturally backwards. **This is the most likely no-op-breaker; address it in Task 4 Step 3.**
