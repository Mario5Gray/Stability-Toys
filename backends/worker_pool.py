"""
Worker pool — thin compatibility facade over Governor.

The control plane lives in `backends/governor.py`; this module preserves the
`WorkerPool` surface (and re-exports the shared job types) so every caller and
test stays green unmodified. Transitional — deleted when routes migrate to the
Governor directly. `from backends.worker_pool import GenerationJob`
(ws_routes.py:621) stays unbroken via the re-export below.
"""
from __future__ import annotations

import gc  # kept alongside torch so `backends.worker_pool.gc` resolves for tests
import logging
import os
import torch  # kept so `backends.worker_pool.torch` resolves — tests patch
              # torch.cuda.* / gc.* through this namespace; both are shared
              # module objects so the patch reaches the Governor's code too.
from concurrent.futures import Future
from typing import Optional

from server.mode_config import ModeConfigManager
from backends.platforms.base import ModelRegistryProtocol
from backends.model_resolution import merge_mode_capabilities  # re-exported for callers/tests

# Shared job types now live in governor.py; re-export them so the public surface
# `from backends.worker_pool import GenerationJob` (and the backplane facade test
# importing _FutureBridge) stays unbroken. Re-export, not ownership.
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

__all__ = [
    'StaleResolutionError', 'ActiveModelSnapshot', 'WorkerFactory',
    'JobType', 'Job', 'GenerationJob', 'JobRecord', '_FutureBridge',
    'ModeSwitchJob', 'CustomJob', 'WorkerPool', 'merge_mode_capabilities',
    'get_worker_pool', 'reset_worker_pool', 'DEFAULT_QUEUE_TIMEOUT_S',
]

logger = logging.getLogger(__name__)


class WorkerPool:
    """Compatibility facade over Governor. Transitional — deleted when routes
    migrate to the Governor directly. Preserved so every caller and (white-box)
    test stays green unmodified: public methods delegate to the Governor, and the
    internals the suite reaches into (q, _worker, _current_mode, _last_activity,
    _get_job_record, _load_mode, _evict_if_idle, ...) are exposed as delegating
    properties/methods. Same pattern as the backplane's Future facade."""

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

    # --- Delegating public methods ---

    def submit_job(self, job: Job, *, timeout_s: float | None = None) -> Future:
        return self._governor.submit_job(job, timeout_s=timeout_s)

    def switch_mode(self, mode_name: str, force: bool = False) -> Future:
        return self._governor.switch_mode(mode_name, force=force)

    def reload_current_mode(self) -> dict:
        return self._governor.reload_current_mode()

    def reload_if_current(self, mode_name: str) -> bool:
        return self._governor.reload_if_current(mode_name)

    def free_vram(self, reason: str) -> dict:
        return self._governor.free_vram(reason)

    def unload_current_model(self) -> dict:
        return self._governor.unload_current_model()

    def get_current_mode(self) -> Optional[str]:
        return self._governor.get_current_mode()

    def is_model_loaded(self) -> bool:
        return self._governor.is_model_loaded()

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

    # --- Delegating internals the (white-box) suite reaches into ---

    @staticmethod
    def _default_worker_factory(worker_id, resolved, binding):
        from backends.governor import Governor
        return Governor._default_worker_factory(worker_id, resolved, binding)

    def _load_mode(self, mode_name: str):
        return self._governor._load_mode(mode_name)

    def _start_worker_thread(self):
        return self._governor._start_dispatch_thread()

    def _unload_current_worker(self):
        return self._governor._unload_current_worker()

    def _get_job_record(self, job_id: str):
        return self._governor._get_job_record(job_id)

    def _evict_if_idle(self):
        return self._governor._evict_if_idle()

    @property
    def q(self):
        return self._governor.q

    @property
    def queue_max(self) -> int:
        return self._governor.queue_max

    @property
    def _worker(self):
        return self._governor._handle.worker

    @_worker.setter
    def _worker(self, value):
        # Tests set pool._worker = None to simulate a vanished worker; forward to
        # the in-proc handle's backing ref (the handle owns the worker).
        self._governor._handle._worker = value

    @property
    def _worker_thread(self):
        return getattr(self._governor, "_worker_thread", None)

    @property
    def _current_mode(self) -> Optional[str]:
        return self._governor._current_mode

    @property
    def _active_snapshot(self):
        return self._governor._active_snapshot

    @property
    def _registry(self):
        return self._governor._registry

    @property
    def _mode_config(self):
        return self._governor._mode_config

    @property
    def _resolution_epoch(self) -> int:
        return self._governor._resolution_epoch

    @property
    def _last_activity(self) -> float:
        return self._governor._last_activity

    @_last_activity.setter
    def _last_activity(self, value: float) -> None:
        self._governor._last_activity = value


# Global worker pool instance
_worker_pool: Optional[WorkerPool] = None


def get_worker_pool(
    worker_factory: Optional[WorkerFactory] = None,
    mode_config: Optional[ModeConfigManager] = None,
    registry: Optional[ModelRegistryProtocol] = None,
) -> WorkerPool:
    """
    Get global worker pool instance.

    Singleton accessor with optional dependency injection support.
    If called multiple times with different dependencies, the first
    call wins (singleton is not recreated).

    Args:
        worker_factory: Optional factory for creating workers (for testing)
        mode_config: Optional mode configuration manager (for testing)
        registry: Optional model registry (for testing)

    Returns:
        Global WorkerPool instance

    Note:
        For production use, call without arguments to use defaults.
        For testing, pass mocked dependencies on first call.

    Example:
        # Production (uses defaults)
        pool = get_worker_pool()

        # Testing (inject mocks)
        pool = get_worker_pool(
            worker_factory=mock_factory,
            mode_config=mock_config,
            registry=mock_registry,
        )
    """
    global _worker_pool
    if _worker_pool is None:
        queue_max = int(os.environ.get("QUEUE_MAX", "64"))
        queue_timeout_s = DEFAULT_QUEUE_TIMEOUT_S
        _worker_pool = WorkerPool(
            queue_max=queue_max,
            queue_timeout_s=queue_timeout_s,
            worker_factory=worker_factory,
            mode_config=mode_config,
            registry=registry,
        )
    return _worker_pool


def reset_worker_pool():
    """
    Reset global worker pool instance.

    Useful for testing to ensure clean state between tests.
    Should NOT be used in production code.
    """
    global _worker_pool
    if _worker_pool is not None:
        try:
            _worker_pool.shutdown()
        except Exception:
            pass  # Ignore shutdown errors during reset
    _worker_pool = None
