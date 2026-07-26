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


# ---------------------------------------------------------------------------
# Task 3: Governor class — isolation tests with a stub WorkerHandle.
# Proves the Governor owns queue + authority + dispatch + lifecycle, and that
# a second WorkerHandle impl requires no Governor change (acceptance #4).
# ---------------------------------------------------------------------------
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
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "none"
    mode_config.get_mode.side_effect = KeyError("no mode")
    registry = Mock()
    registry.get_total_vram.return_value = 0
    handle = StubHandle()
    gov = Governor(
        worker_factory=Mock(),
        handle=handle,
        mode_config=mode_config,
        registry=registry,
    )
    assert gov._handle is handle
    gov.shutdown()


def test_governor_submit_job_resolves_future_through_handle():
    """submit_job opens the channel + attaches _FutureBridge; the dispatch loop
    drives record.sink. Uses an InProcessWorkerHandle with a mock factory so
    the dispatch loop can run job.execute(worker) directly (v1 does NOT call
    handle.submit() in the dispatch loop — that's the facet-3 contract)."""
    from backends.worker_handle import InProcessWorkerHandle
    from backends.conditioning.contracts import ConditioningConfig
    from backends.model_resolution import LocalModelBinding

    # Patch resolve_model so _load_mode succeeds without filesystem detection
    # (same seam test_worker_pool.py patches at worker_pool.resolve_model).
    # The Governor stores `resolved` opaquely in the snapshot and passes it to
    # handle.start(); a Mock is sufficient here.
    def _resolve(model_path: str, mode):
        return Mock(), LocalModelBinding(model_path)

    with patch("backends.governor.resolve_model", side_effect=_resolve):
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
