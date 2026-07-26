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
from typing import TYPE_CHECKING, Optional

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
