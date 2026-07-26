"""
Worker Governor: control-plane extraction behind a WorkerHandle interface.

Task 1: shared job types (moved verbatim from worker_pool.py). The Governor
class itself is added in Task 3. Imports here are forward-looking — they cover
the types moved in Task 1 and the Governor class that lands in Task 3.
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


# Type hints for dependency injection
class WorkerFactory(Protocol):
    """Protocol for worker creation functions.

    The final contract takes a portable ``ResolvedModel`` plus a node-local
    ``LocalModelBinding``. Threading these through ``_load_mode`` (via
    ``resolve_model``) and the default factory lands with the active snapshot in
    Task 5; this task establishes the protocol shape only.
    """

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
    """
    Base class for all job types.

    Extensible job system - subclass this to create new job types.
    """
    job_type: JobType = field(init=False)
    fut: Future = field(init=False, default_factory=Future)  # Result future

    def __post_init__(self):
        if self.fut is None:
            self.fut = Future()

    @abstractmethod
    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        """
        Execute the job.

        Args:
            worker: Current worker (may be None for non-generation jobs)

        Returns:
            Job result
        """
        pass


@dataclass
class GenerationJob(Job):
    """Job for image generation."""
    req: Any  # GenerateRequest
    init_image: Optional[bytes] = None  # Optional init image bytes for img2img
    controlnet_bindings: list[Any] = field(default_factory=list)  # Resolved ControlNetBinding list (T3 — populated by CudaGenerationRuntime.submit_generate)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Required, keyword-only: the resolution epoch this job was admitted against.
    # Stamped from the active snapshot at submission; enforced at the last safe
    # boundary before run_job. No implicit default — unstamped jobs never queue.
    resolution_epoch: int = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.GENERATION

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        """Execute generation job."""
        if worker is None:
            raise RuntimeError("No worker available for generation")
        return worker.run_job(self)  # type: ignore[arg-type]


@dataclass
class JobRecord:
    job_id: str
    state: str
    job: GenerationJob
    cancel_requested: bool = False
    sink: Optional[JobSink] = None  # backplane producer handle (attached in submit_job)


class _FutureBridge(Subscriber):
    """Compat Subscriber: fulfils a concurrent.futures.Future from the backplane
    stream, reproducing today's fut.set_result / fut.set_exception exactly.

    Touches ONLY the Future — never pool state or _job_lock (spec §3.3 lock
    invariant). Requests unbounded demand so the synchronous in-proc channel
    delivers terminals immediately (spec §3.3 must-deliver-on-return).

    Result is carried OPAQUELY: the worker's run_job return value rides the frame's
    image (InProcBlob) untouched, so `fut.set_result(<opaque>)` matches today whether
    run_job returns a (png, seed) tuple (production) or "test_result" (tests). This
    is the reconciliation of spec §5, which assumed the Subscriber decomposes into
    (png_bytes, seed) — decomposition is incompatible with the no-op and is deferred
    to the streaming/IPC consumers that actually need seed + PNG separated.
    """

    def __init__(self, fut: Future):
        self._fut = fut

    def on_subscribe(self, subscription):
        subscription.request(1 << 62)  # unbounded — the Future is single-valued

    def on_next(self, value):
        if isinstance(value, Result) and not self._fut.done():
            self._fut.set_result(value.image.read_sync())  # opaque passthrough

    def on_error(self, error):
        if not self._fut.done():
            self._fut.set_exception(error.to_exception())  # live instance in-proc

    def on_complete(self):
        pass


@dataclass
class ModeSwitchJob(Job):
    """Job for switching model mode."""
    target_mode: str
    on_complete: Optional[Callable] = None
    force: bool = False  # Reload even if target_mode == current_mode

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.MODE_SWITCH

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        """
        Execute mode switch.

        This doesn't use the worker directly - it triggers worker recreation.
        """
        logger.info(f"[ModeSwitchJob] Switching to mode: {self.target_mode}")
        if self.on_complete:
            self.on_complete(self.target_mode)
        return {"mode": self.target_mode, "status": "switched"}


@dataclass
class CustomJob(Job):
    """
    Extensible custom job.

    Allows other parts of the app to queue arbitrary work.
    """
    handler: Callable
    args: tuple = ()
    kwargs: Optional[dict] = None

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.CUSTOM
        if self.kwargs is None:
            self.kwargs = {}

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        """Execute custom handler."""
        return self.handler(*self.args, **self.kwargs)  # type: ignore[arg-type]


# Import the WorkerHandle interface so `governor.WorkerHandle` /
# `governor.InProcessWorkerHandle` resolve (acyclic: worker_handle does NOT
# import governor at runtime). InProcessWorkerHandle lands in Task 2; the ABC
# is present from Task 1.
from backends.worker_handle import WorkerHandle, WorkerHealth  # noqa: E402


# Governor class added in Task 3.
