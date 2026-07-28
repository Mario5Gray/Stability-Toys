"""
WorkerHandle: locality-agnostic interface to ONE worker.

The Governor programs to this interface regardless of where the worker runs
(in-proc thread today; subprocess in facet-3; microservice later).

Import graph: worker_handle.py imports backplane + base, NOT governor.
The Job type hint in submit() is deferred via from __future__ import annotations
+ TYPE_CHECKING so it forces no runtime import of governor.py (acyclic graph).
"""
from __future__ import annotations

import gc
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.backplane.frames import BackplaneError
from backends.backplane.reactivestreams import Publisher
from backends.base import PipelineWorker

if TYPE_CHECKING:
    from backends.governor import Job

logger = logging.getLogger(__name__)


@dataclass
class WorkerHealth:
    """Liveness + readiness snapshot the Governor reads for admission/status.
    VRAM is DRIVER TRUTH (mem_get_info), aligning with STABL-sqqlkmdl — not the
    torch allocator."""
    state: str                 # starting | ready | busy | draining | dead
    vram_free_bytes: int       # driver-truth free (what admission needs); 0 if N/A
    vram_total_bytes: int      # driver-truth total; 0 if N/A
    mode: str | None           # loaded mode name, or None


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


class InProcessWorkerHandle(WorkerHandle):
    """In-process threaded worker handle.

    Owns the worker reference, worker thread, and worker factory. The handle
    drives the backplane JobSink (the data-plane producer side) — it is the
    threaded-worker coupling extracted from WorkerPool.

    Does NOT own: the queue, the epoch, the snapshot, the admission decision,
    job records, the registry, or _job_lock. Those are the Governor's authority.

    NOTE on v1 dispatch: the Governor's in-proc dispatch loop runs job.execute
    on its OWN dispatch thread and drives record.sink directly (reconciliation
    #2/#4) — it does NOT call submit() in v1. submit() is the facet-3 contract
    (a SubprocessWorkerHandle owns its own execution + IPC channel), exercised
    here in isolation. It runs the job on a background thread so it returns a
    Publisher immediately (async submit), matching the out-of-process shape.
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
        """Provision + load the worker via the factory (the worker-creation +
        conditioning-configuration portion of WorkerPool._load_mode)."""
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

    def submit(self, job: Job) -> Publisher:
        """Execute a job. Opens a backplane channel, runs job.execute(worker) on
        a background thread, drives the JobSink, and returns the Publisher
        immediately. The Governor subscribes _FutureBridge to it.

        The 'ready' transition is published BEFORE the result frame so an
        observer that unblocks on the result never races a still-'busy' state.
        """
        sink, publisher = InProcBackplane(job.job_id).open()

        def _run() -> None:
            self._state = "busy"
            try:
                if self._worker is None:
                    raise RuntimeError("No worker available for generation")
                result = job.execute(self._worker)
            except Exception as e:  # noqa: BLE001 — error rides the sink terminal
                logger.error(
                    f"[InProcessWorkerHandle] Job {job.job_id} failed: {e}",
                    exc_info=True,
                )
                # OOM gets the same error frame in v1; facet-3's
                # SubprocessWorkerHandle turns OOM into kill+respawn — the durable
                # recovery empty_cache() cannot do on a poisoned context.
                self._state = "ready"
                sink.error(BackplaneError.from_exc(e))
                return
            self._state = "ready"
            sink.result(0, InProcBlob(result))
            sink.complete()

        thread = threading.Thread(
            target=_run, name=f"handle-job-{job.job_id}", daemon=True
        )
        self._worker_thread = thread
        thread.start()
        return publisher

    def health(self) -> WorkerHealth:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
        else:
            free_b, total_b = 0, 0
        return WorkerHealth(
            state=self._state,
            vram_free_bytes=int(free_b),
            vram_total_bytes=int(total_b),
            mode=None,  # mode is the Governor's authority; the handle doesn't track it
        )

    def unload(self) -> None:
        """Graceful teardown: clear the ControlNet cache, drop the worker
        reference, and flush the GPU allocator (WorkerPool._unload_current_worker
        + _free_worker, minus the registry-unregister which is Governor authority)."""
        from backends.controlnet_cache import get_controlnet_cache

        dropped = get_controlnet_cache().clear()
        if dropped:
            logger.info(
                f"[InProcessWorkerHandle] Released {dropped} cached ControlNet model(s)"
            )

        if self._worker is not None:
            del self._worker
            self._worker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._state = "dead"

    def stop(self) -> None:
        """Hard terminate. In-proc v1 = same as unload(). Facet-3 makes this the
        real recovery primitive: kill the subprocess."""
        self.unload()
