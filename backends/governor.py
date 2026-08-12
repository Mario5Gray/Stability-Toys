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
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
from concurrent.futures import Future, CancelledError
from enum import Enum

from server.mode_config import get_mode_config, ModeConfig, ModeConfigManager
from server.metrics import get_metrics
from server import log_context, tracing
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
from backends.backplane.frames import Progress, Result, BackplaneError, BackplaneErrorCode
from backends.backplane.interface import JobSink
from backends.backplane.reactivestreams import Subscriber

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_TIMEOUT_S: float = float(os.environ.get("WORKER_QUEUE_TIMEOUT_S", "0.25"))

# STABL-atzqpcte: one clock became two. DEFAULT_TIMEOUT keeps its name and its 120s
# default and now bounds only EXECUTION — the meaning its name always claimed. Queue
# wait (which includes a mode switch's model load, minutes for a cold checkpoint) is
# bounded separately and generously. Bounded rather than unbounded so a job wedged
# behind a hung ModeSwitchJob still fails instead of pinning a connection forever.
DEFAULT_EXECUTION_TIMEOUT_S: float = float(os.environ.get("DEFAULT_TIMEOUT", "120"))
DEFAULT_ADMISSION_TIMEOUT_S: float = float(os.environ.get("ADMISSION_TIMEOUT_S", "900"))


class StaleResolutionError(RuntimeError):
    """A queued job was resolved against a superseded model authority."""


class ModeLoadFailedError(RuntimeError):
    """A queued job was admitted against an authority whose load never completed."""


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
    def execute(self, worker: Optional[PipelineWorker], progress=None) -> Any:
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

    def execute(self, worker: Optional[PipelineWorker], progress=None,
                should_cancel=None) -> Any:
        """Execute generation job. `progress(step, total, stage)` is the sink's
        Progress emitter, threaded into run_job so the diffusion step callback can
        report (STABL-zueslhah); None when no consumer is attached.

        `should_cancel()` is the reap predicate (STABL-jredufxb), consulted at the
        same step boundary; None when the job is not cancellable."""
        if worker is None:
            raise RuntimeError("No worker available for generation")
        return worker.run_job(self, progress=progress, should_cancel=should_cancel)  # type: ignore[arg-type]


@dataclass
class JobRecord:
    job_id: str
    state: str
    job: GenerationJob
    cancel_requested: bool = False
    sink: Optional[JobSink] = None  # backplane producer handle (attached in submit_job)
    # STABL-zueslhah: the WS progress consumer, stored at submit so the SUBPROCESS
    # dispatch path (which builds the bridge later, in _dispatch_loop) can attach it
    # to _SubprocessFutureBridge. The in-proc path passes it to _FutureBridge directly.
    on_progress: Optional[Callable] = None
    # STABL-atzqpcte: monotonic timestamp of TRUE execution start, stamped after the
    # demand reload and after the stale-epoch barrier so neither is charged to the
    # execution budget. Deliberately NOT `state == "running"`, which is set earlier
    # (before both) and whose transition cancel_job depends on for queued-job cancel.
    executing_since: Optional[float] = None
    # STABL-asawxgvp: monotonic enqueue time, stamped in _register_job. Paired with
    # executing_since this splits the two budgets into two observable durations —
    # queue wait and execution — which nothing could derive before.
    enqueued_at: Optional[float] = None


def _is_oom(exc: BaseException) -> bool:
    """OOM classification that CANNOT raise (STABL-hdzggeir).

    `torch.cuda.OutOfMemoryError` is not guaranteed to be a type. In a shared pytest
    session this module's `torch` can be a MagicMock — every backend module binds
    whatever `sys.modules['torch']` held when IT was first imported — and
    `isinstance(exc, <MagicMock>)` raises `TypeError: isinstance() arg 2 must be a
    type`. Raising here is far worse than misclassifying: this runs inside the
    dispatch loop's error handler, so the throw escaped the loop and killed the
    dispatch thread.

    The `isinstance(oom_cls, type)` guard is the fix; the message-substring check is
    the pre-existing fallback and still covers OOMs raised as plain RuntimeError.
    """
    oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_cls, type) and isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower()


def _terminal_outcome(fut) -> str:
    """Terminal outcome from a RESOLVED future (STABL-asawxgvp).

    `fut.cancelled()` is checked FIRST because `fut.exception()` raises
    CancelledError on a cancelled future — the obvious ordering is a bug.

    `timeout` is deliberately NOT an outcome here. A timeout is a waiter-side
    budget breach raised by `_expire`, counted by `st_governor_wait_expired_total`;
    what subsequently reaches the dispatch loop for that job is a CANCEL. Counting
    both would double-count the same job.
    """
    if fut.cancelled():
        return "cancelled"
    exc = fut.exception()
    if exc is None:
        return "ok"
    if isinstance(exc, CancelledError):
        return "cancelled"
    if _is_oom(exc):
        return "oom"
    return "error"


def _cancel_predicate(record: JobRecord) -> Callable[[], bool]:
    """The reap predicate handed to the worker (STABL-jredufxb).

    Read LOCK-FREE by design. `cancel_job` writes `cancel_requested` under
    `_job_lock`, but the worker must never acquire that lock — the backplane
    Subscriber<->lock invariant. A bool read is atomic under the GIL, and a
    one-step-late read is harmless: the next denoise step catches it.

    A named factory rather than an inline lambda so the record is captured per
    call, not per dispatch-loop iteration.
    """
    return lambda: record.cancel_requested


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

    def __init__(self, fut: Future, on_progress=None):
        self._fut = fut
        # STABL-zueslhah: forward non-terminal Progress frames to a consumer
        # (the WS streamer in Task 4) instead of dropping them. None preserves
        # today's Future-only behaviour exactly.
        self._on_progress = on_progress

    def on_subscribe(self, subscription):
        subscription.request(1 << 62)  # unbounded — the Future is single-valued

    def on_next(self, value):
        if isinstance(value, Progress):
            if self._on_progress is not None:
                self._on_progress(value.step, value.total, value.stage)
            return
        if isinstance(value, Result) and not self._fut.done():
            self._fut.set_result(value.image.read_sync())  # opaque passthrough

    def on_error(self, error):
        if not self._fut.done():
            self._fut.set_exception(error.to_exception())  # live instance in-proc

    def on_complete(self):
        pass


def _worker_allocated(snapshot) -> int:
    """Worker consumer's live allocations from a DeviceMemory snapshot.
    Load-time measurement reads a FRESH snapshot() (the one fan-out exception);
    /status-shaped readers use cached_snapshot() instead."""
    return next((c.allocated_bytes for c in snapshot.consumers if c.label == "worker"), 0)


@dataclass
class ModeSwitchJob(Job):
    """Job for switching model mode."""
    target_mode: str
    on_complete: Optional[Callable] = None
    force: bool = False  # Reload even if target_mode == current_mode
    # The authority this switch WILL publish, reserved atomically with its enqueue.
    # None for switches built outside the Governor (tests, legacy callers): _load_mode
    # then reserves inline.
    reservation: Optional["ActiveModelSnapshot"] = None

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.MODE_SWITCH

    def execute(self, worker: Optional[PipelineWorker], progress=None) -> Any:
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

    def execute(self, worker: Optional[PipelineWorker], progress=None) -> Any:
        """Execute custom handler."""
        return self.handler(*self.args, **self.kwargs)  # type: ignore[arg-type]


# Import the WorkerHandle interface so `governor.WorkerHandle` /
# `governor.InProcessWorkerHandle` resolve (acyclic: worker_handle does NOT
# import governor at runtime — the Job hint is TYPE_CHECKING-guarded).
# InProcessWorkerHandle is imported at module top (not lazily) now that Task 2
# has landed; the Governor constructs it by default when no handle is injected.
from backends.worker_handle import (  # noqa: E402
    InProcessWorkerHandle,
    WorkerHandle,
    WorkerHealth,
)


class Governor:
    """Control-plane extraction: owns queue, epoch/snapshot authority,
    admission barrier, dispatch, lifecycle, recovery.

    Delegates worker execution to a WorkerHandle (InProcessWorkerHandle by
    default; stub/subprocess/remote for testing or facet-3).

    Dispatch loop is ``_worker_loop`` VERBATIM (reconciliation #2) with one
    substitution: ``self._worker`` -> ``self._handle.worker`` and
    ``self._unload_current_worker()`` -> ``self._handle.unload()``. The handle's
    ``submit()`` is the facet-3 contract and is NOT called in v1's in-proc
    dispatch loop — the Governor drives ``record.sink`` directly (reconciliation
    #4/#5). ``submit_job`` opens the ``InProcBackplane`` channel + attaches
    ``_FutureBridge`` BEFORE enqueueing, verbatim from ``worker_pool.py``.
    """

    def __init__(
        self,
        queue_max: int = 64,
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        worker_factory: Optional[WorkerFactory] = None,
        mode_config: Optional[ModeConfigManager] = None,
        registry: Optional[ModelRegistryProtocol] = None,
        handle: Optional[WorkerHandle] = None,
        device_memory=None,
    ):
        self.queue_max = queue_max
        self.queue_timeout_s = queue_timeout_s
        self.q: queue.Queue[Job] = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._current_mode: Optional[str] = None
        self._active_snapshot: Optional[ActiveModelSnapshot] = None
        self._resolution_epoch: int = 0
        # Reservations: authority that a queued mode switch WILL publish. The
        # terminal reservation is what a job admitted NOW executes against.
        self._pending_authorities: list[ActiveModelSnapshot] = []
        self._dead_epochs: set[int] = set()
        self._job_records: dict[str, JobRecord] = {}
        self._job_lock = threading.RLock()
        self._idle_timeout = float(os.environ.get("MODEL_IDLE_TIMEOUT_SECS", "300"))
        self._idle_check_interval = float(os.environ.get("MODEL_IDLE_CHECK_INTERVAL_SECS", "30"))
        self._last_activity = time.monotonic()
        self._eviction_pending = False

        self._worker_factory = worker_factory
        self._mode_config = mode_config or get_mode_config()
        self._registry = registry or get_model_registry()
        if device_memory is None:
            from backends.device_memory import get_device_memory
            device_memory = get_device_memory()
        self._dm = device_memory
        # One publication point, because provider selection is a singleton and the
        # uuid never changes for the life of the process (STABL-bpsfmoke). getattr,
        # not attribute access: a provider without the attribute must degrade to an
        # ABSENT field, not raise.
        self._log_field("device_uuid", getattr(self._dm, "device_uuid", None))

        # Handle: inject for testing/pluggability, or build InProcessWorkerHandle
        # from the factory. Injected handles (stub/subprocess) never touch the
        # default path. (Acceptance #4: a second WorkerHandle impl plugs in with
        # no Governor change — proven by test_second_handle_impl_requires_no_governor_change.)
        if handle is not None:
            self._handle = handle
        elif worker_factory is not None:
            self._handle = InProcessWorkerHandle(worker_factory, device_memory=self._dm)
        else:
            self._handle = InProcessWorkerHandle(self._default_worker_factory, device_memory=self._dm)

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

    # --- Authority reservation (spec §3.1-§3.3) ---

    def _resolve_target(self, mode_name: str):
        """Detect + resolve a target mode. Performs disk I/O (detect_model) and
        therefore MUST NOT be called while holding _job_lock."""
        mode = deepcopy(self._mode_config.get_mode(mode_name))
        assert mode.model_path is not None
        resolved, binding = resolve_model(mode.model_path, mode)
        return mode, resolved, binding

    def _terminal_authority(self) -> Optional[ActiveModelSnapshot]:
        """The authority a job admitted NOW will execute against: the last queued
        reservation, else the published snapshot. Caller MUST hold _job_lock."""
        if self._pending_authorities:
            return self._pending_authorities[-1]
        return self._active_snapshot

    def _reserve_authority(self, mode_name: str) -> ActiveModelSnapshot:
        """Reserve an epoch + resolved model.

        Used by callers that pair the reservation with their own enqueue, and by the
        reservation-less paths (_load_mode's __init__ / force-reload callers) where
        there is no enqueue to pair with at all.
        """
        mode, resolved, binding = self._resolve_target(mode_name)
        with self._job_lock:
            self._resolution_epoch += 1
            reservation = ActiveModelSnapshot(
                mode_name=mode_name,
                mode=mode,
                resolved=resolved,
                binding=binding,
                resolution_epoch=self._resolution_epoch,
            )
            self._pending_authorities.append(reservation)
            self._last_activity = time.monotonic()
            return reservation

    def _drop_reservation(self, reservation: ActiveModelSnapshot, *, dead: bool) -> None:
        """Remove a reservation by IDENTITY. The epoch is never rolled back — epochs
        are monotone and never reused, including across failures."""
        with self._job_lock:
            self._pending_authorities = [
                r for r in self._pending_authorities if r is not reservation
            ]
            if dead:
                self._dead_epochs.add(reservation.resolution_epoch)

    def get_pending_mode(self) -> Optional[str]:
        """The mode a queued switch is heading to, or None. Distinct from
        get_current_mode(), which stays 'the actually-loaded mode'."""
        with self._job_lock:
            if self._pending_authorities:
                return self._pending_authorities[-1].mode_name
            return None

    def _switch_shortcircuit(self, mode_name: str, worker_ok: bool) -> Optional[dict]:
        """The result to return INSTEAD of reserving, or None to proceed with a
        reservation (spec §3.3). Caller MUST hold _job_lock.

        `worker_ok` is passed in rather than read here: _worker_available() calls into
        the handle, and the handle must never be invoked while _job_lock is held
        (backplane Subscriber<->lock invariant). The dispatch fast-path at :606 also
        reads it unlocked, so the same slight staleness already governs this decision.
        """
        terminal = self._terminal_authority()
        if terminal is None or terminal.mode_name != mode_name:
            return None
        if self._pending_authorities:
            # A switch to this mode is already queued — bind to it, reserve nothing.
            return {"mode": mode_name, "status": "already_queued"}
        if worker_ok:
            # Already active with a live worker: the dispatch fast-path would return
            # already_loaded WITHOUT calling _load_mode, so a reservation made here
            # could never be published.
            return {"mode": mode_name, "status": "already_loaded"}
        # Active mode, but the worker was idle-evicted: fall through and reload.
        return None

    # --- Mode load / lifecycle (delegates worker build to handle.start) ---

    def _load_mode(self, mode_name: str, reservation: Optional[ActiveModelSnapshot] = None):
        """Load a mode: build the worker via handle.start(), then publish the
        reservation as the active snapshot.

        The epoch is reserved by the caller (or inline here) BEFORE the load, so a
        generate admitted against the reservation carries the epoch this publishes.
        """
        logger.info(f"[Governor] Loading mode: {mode_name}")
        _load_started = time.monotonic()
        if reservation is None:
            reservation = self._reserve_authority(mode_name)
        mode = reservation.mode
        resolved, binding = reservation.resolved, reservation.binding

        self._unload_current_worker(reason="switch")  # unregister old mode + tear down worker
        with self._job_lock:
            self._active_snapshot = None

        # Load-time measurement reads a FRESH snapshot() — the one sanctioned
        # fan-out exception (spec §4.1 / MUST-FIX-2). This is NOT the admission
        # path (a load already blocks on model I/O), so fan-out is permitted.
        allocated_before = _worker_allocated(self._dm.snapshot())

        try:
            self._handle.start(resolved, binding, mode)
        except Exception as e:
            logger.error(f"[Governor] Failed to load mode '{mode_name}': {e}", exc_info=True)
            self._handle.unload()
            with self._job_lock:
                self._current_mode = None
                self._active_snapshot = None
            self._drop_reservation(reservation, dead=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        vram_allocated = _worker_allocated(self._dm.snapshot())
        vram_used = max(0, vram_allocated - allocated_before)
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

        # Process-global, not per-job: threads with no job of their own (sampler,
        # watchdog, uvicorn) still belong to the resident mode (STABL-bpsfmoke).
        self._log_field("mode", mode_name)

        with self._job_lock:
            self._current_mode = mode_name
            self._active_snapshot = reservation
            self._pending_authorities = [
                r for r in self._pending_authorities if r is not reservation
            ]
            # Prune dead epochs below the published one: monotone epochs plus
            # terminal-only admission mean no NEW job can carry them (spec §3.5).
            self._dead_epochs = {
                e for e in self._dead_epochs if e >= reservation.resolution_epoch
            }

        logger.info(f"[Governor] Mode '{mode_name}' loaded (epoch={reservation.resolution_epoch})")

        # STABL-asawxgvp. Reached only on a SUCCESSFUL load — a load that raises has
        # no duration to report, and its failure is visible as mode_active staying 0.
        _epoch = reservation.resolution_epoch
        self._metric(lambda met: (
            met.mode_load_seconds.labels(mode=mode_name).observe(
                time.monotonic() - _load_started),
            met.resolution_epoch.set(_epoch),
            self._publish_mode_active(met, mode_name),
        ))

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
        _mode_name = snapshot.mode_name
        # Republished here for the same reason _publish_mode_active is: this path
        # brings the worker back WITHOUT going through _load_mode, and the eviction
        # that preceded it cleared the field. Omit this and every line after an
        # evict/reload cycle claims no mode is resident while one is (STABL-bpsfmoke).
        self._log_field("mode", _mode_name)
        self._metric(lambda met: (
            met.demand_reload_total.labels(mode=_mode_name).inc(),
            self._publish_mode_active(met, _mode_name),
        ))

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
                    enqueued_at=time.monotonic(),
                )

    def _finalize_job_record(self, job_id: str):
        with self._job_lock:
            record = self._job_records.pop(job_id, None)
        # Observed OUTSIDE _job_lock: the backplane's Subscriber<->lock invariant,
        # and there is no reason to hold the lock across a metrics write.
        if record is not None:
            self._observe_job_terminal(record)

    def _observe_job_terminal(self, record) -> None:
        """Emit terminal metrics for a finished job (STABL-asawxgvp).

        This is the SINGLE instrumentation point for job terminals: every branch
        of the dispatch loop — early cancel, in-proc success, post-execute cancel,
        subprocess terminal, and _deliver_job_failure — funnels through
        _finalize_job_record, and by then the future is resolved on all of them.

        MUST NOT raise (STABL-hdzggeir): a throw here reaches the dispatch loop's
        try, kills the thread, and permanently deadens the queue. The whole body
        is guarded, not just the metric calls — a malformed record must not
        escape either.
        """
        try:
            fut = record.job.fut
            if record.enqueued_at is None or not fut.done():
                return          # e.g. the submit_job queue.Full rollback
            mode = self._current_mode or "unknown"
            met = get_metrics()
            if record.executing_since is not None:
                met.job_queue_wait_seconds.labels(mode=mode).observe(
                    max(0.0, record.executing_since - record.enqueued_at))
                met.job_execution_seconds.labels(mode=mode).observe(
                    max(0.0, time.monotonic() - record.executing_since))
            met.job_terminal_total.labels(
                mode=mode, outcome=_terminal_outcome(fut)).inc()
        except Exception:
            logger.debug("[Governor] terminal metrics failed", exc_info=True)

    def _metric(self, fn) -> None:
        """Run a metrics side effect that must never reach the dispatch loop.

        Every lifecycle counter goes through here rather than growing its own
        try/except at the call site — one place to be certain about, and the call
        sites stay one line (STABL-hdzggeir).
        """
        try:
            fn(get_metrics())
        except Exception:
            logger.debug("[Governor] metrics side effect failed", exc_info=True)

    @staticmethod
    @contextmanager
    def _span(name: str):
        """Open a Governor span without letting tracing failures reach the loop.

        Mirrors _metric and _log_field: observability code in lifecycle paths must
        never be allowed to deaden the dispatch thread.
        """
        try:
            with tracing.get_tracer(__name__).start_as_current_span(name) as span:
                yield span
        except Exception:
            logger.debug("[Governor] span %s failed", name, exc_info=True)
            yield tracing._NoopSpan()

    @staticmethod
    def _log_field(name: str, value) -> None:
        """Publish a process-wide structured-log field (STABL-bpsfmoke).

        Guarded exactly like _metric, for the same reason: this runs on the
        dispatch thread and inside lifecycle paths, and STABL-hdzggeir says nothing
        added there may be allowed to raise.
        """
        try:
            log_context.set_static_field(name, value)
        except Exception:
            logger.debug("[Governor] log field %s failed", name, exc_info=True)

    def _count_worker_recovery(self, *, oom: bool) -> None:
        self._metric(lambda met: met.worker_recovery_total.labels(
            reason="oom" if oom else "dead").inc())

    def _publish_mode_active(self, met, active: Optional[str]) -> None:
        """0/1 for EVERY configured mode.

        A single gauge labelled with the current mode leaves a stale 1 on the
        previous mode's series forever. conf/modes.yml holds 4 modes, so
        reporting all of them is cheap (spec resolved question 3).
        """
        for name in self._mode_config.list_modes():
            met.mode_active.labels(mode=name).set(1 if name == active else 0)

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

    # --- Result waiting: two budgets, split at true execution start ---

    def _record_for_future(self, fut) -> Optional[JobRecord]:
        """Find a job's record by FUTURE IDENTITY.

        Deliberately not by job id: the HTTP path's `runtime.submit_generate()`
        returns only a future and never exposes an id, so an id-keyed lookup would
        fix the WebSocket transport and silently leave HTTP on the old semantics.
        """
        with self._job_lock:
            for record in self._job_records.values():
                if record.job.fut is fut:
                    return record
        return None

    def wait_for_result(
        self,
        fut,
        *,
        admission_timeout_s: Optional[float] = None,
        execution_timeout_s: Optional[float] = None,
        poll_interval_s: float = 0.25,
    ):
        """Wait for a job's result under TWO budgets.

        A generate's future does not resolve until everything ahead of it in the
        queue has run, so a single `fut.result(timeout=...)` charges queue wait and
        model load to a budget meant for generation — the STABL-atzqpcte defect,
        observed in the field as a WebSocket timeout during a HunyuanDiT load.

        While the job has not begun executing it is judged against the ADMISSION
        budget; once `JobRecord.executing_since` is stamped (after the demand reload
        and the stale-epoch barrier) it is judged against the EXECUTION budget,
        measured from when execution actually began rather than from when this
        waiter noticed.

        On expiry the job is asked to cancel and `TimeoutError` is raised — the type
        callers already handle. NOTE that cancelling does NOT stop a generation that
        is already running: `cancel_requested` is read only at job boundaries and
        `run_job` never checks it, so the worker runs to completion holding VRAM.
        Reaping it for real is STABL-jredufxb.
        """
        admission = (
            DEFAULT_ADMISSION_TIMEOUT_S if admission_timeout_s is None else admission_timeout_s
        )
        execution = (
            DEFAULT_EXECUTION_TIMEOUT_S if execution_timeout_s is None else execution_timeout_s
        )
        waiting_since = time.monotonic()
        vanished_since: Optional[float] = None

        while True:
            try:
                return fut.result(timeout=poll_interval_s)
            except TimeoutError:
                if fut.done():
                    # The JOB raised TimeoutError; that is its result, not our budget.
                    return fut.result()

            record = self._record_for_future(fut)
            now = time.monotonic()

            if record is not None:
                started = record.executing_since
            else:
                # The record is finalized on completion, so it can vanish while the
                # future is briefly unresolved. Treat that as executing — never as
                # still-queued, which would hand the generous admission budget to a
                # job that has already run.
                vanished_since = now if vanished_since is None else vanished_since
                started = vanished_since

            if started is None:
                if now - waiting_since >= admission:
                    self._expire(record, "admission", admission, now - waiting_since)
            elif now - started >= execution:
                self._expire(record, "execution", execution, now - started)

    def _expire(self, record: Optional[JobRecord], budget: str, limit_s: float, waited_s: float):
        """Ask the job to stop, then raise. Cancelling a still-QUEUED job takes it off
        the queue entirely, so the work is never done at all."""
        knob = "ADMISSION_TIMEOUT_S" if budget == "admission" else "DEFAULT_TIMEOUT"
        who = f"job {record.job_id}" if record is not None else "job"
        # STABL-asawxgvp: a timeout is a WAITER-side budget breach, counted here and
        # deliberately not as a job_terminal_total outcome — what reaches the
        # dispatch loop for this job afterwards is a cancel.
        self._metric(lambda met: met.wait_expired_total.labels(budget=budget).inc())
        if record is not None:
            self.cancel_job(record.job_id)
        raise TimeoutError(
            f"{who} exceeded its {budget} budget of {limit_s:g}s after {waited_s:.1f}s "
            f"(raise {knob} if this is legitimate)"
        )

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation.

        A QUEUED job is taken off the queue outright, so the work is never done.
        A RUNNING job has the flag flipped that the in-proc reap predicate reads,
        and — for a locality that cannot see that flag — the handle is signalled
        by job id (STABL-jredufxb).
        """
        signal_handle = False
        with self._job_lock:
            record = self._job_records.get(job_id)
            if record is None or record.job.fut.done():
                return False
            record.cancel_requested = True
            if record.state == "queued" and record.job.fut.cancel():
                record.state = "cancelled"
                return True
            record.state = "running"
            signal_handle = True

        # OUTSIDE _job_lock, deliberately. The subprocess handle's cancel takes
        # _control_lock, which an in-flight stats reply can hold for
        # _STATS_REPLY_TIMEOUT_S — signalling under _job_lock would let a
        # /api/models/status fan-out stall the dispatch loop.
        if signal_handle:
            handle_cancel = getattr(self._handle, "cancel_job", None)
            if callable(handle_cancel):
                try:
                    handle_cancel(job_id)
                except Exception:  # noqa: BLE001 — a failed signal must not fail the cancel
                    logger.warning(
                        "[Governor] handle.cancel_job(%s) failed", job_id, exc_info=True
                    )
        return True

    # --- VRAM cleanup / recovery ---

    def _worker_available(self) -> bool:
        """Locality-agnostic 'is a live, loaded worker present?'. For InProcess
        this equals worker-is-not-None (start->ready, unload->dead); for Subprocess
        it reads true liveness (its .worker is always None). Preserves the spec
        §9.2 semantic split at each call site.

        Defensive fallback: if health() raises (e.g. CPU-only host where
        torch.cuda.is_available() is True but mem_get_info() returns empty),
        preserve the in-proc equivalence by checking handle.worker directly.
        """
        try:
            return self._handle.health().state in ("ready", "busy")
        except Exception:
            return self._handle.worker is not None

    def _unload_current_worker(self, reason: str = "explicit") -> None:
        """Unload the current worker. Registry-unregister is Governor authority
        (mirrors WorkerPool._unload_current_worker:322); worker + ControlNet-cache
        teardown is delegated to the handle. Guarded on worker-presence like the
        original, so a no-worker pool still clears the cache without unregistering.

        `reason` labels st_governor_unload_total (STABL-asawxgvp). It is a parameter
        rather than a constant because the five callers are operationally distinct —
        a mode switch, an idle eviction and an OOM cleanup are not the same event,
        and a hardcoded label would make the dimension dead weight.
        """
        mode = self._current_mode
        if self._worker_available() and mode:
            self._registry.unregister_model(mode)
        self._handle.unload()
        # Nothing is resident now. REMOVES the field rather than blanking it, and
        # mirrors _publish_mode_active(met, None) below: _load_mode calls this
        # before every load and republishes on success, so a failed load correctly
        # leaves the field absent (STABL-bpsfmoke).
        self._log_field("mode", None)
        # Only when something was actually loaded: _load_mode unloads before every
        # load, and the first-ever load has no outgoing mode to report churn for.
        if mode:
            self._metric(lambda met: (
                met.unload_total.labels(mode=mode, reason=reason).inc(),
                # Nothing is loaded now. _load_mode calls this before every load and
                # republishes the new mode on success, so a switch reads 0 -> 1 and a
                # failed load correctly leaves every mode at 0.
                self._publish_mode_active(met, None),
            ))

    def _cleanup_vram(self, reason: str, cancel_running: bool) -> list[str]:
        cancelled = self.cancel_pending_generation_jobs(reason=reason)
        if cancel_running:
            cancelled.extend(self._mark_running_generation_jobs_cancel_requested(reason=reason))
        self._unload_current_worker(reason=reason)
        gc.collect()
        self._dm.reclaim()  # soft pool-trim of LIVE consumers; teardown already flushed inline
        return cancelled

    def _build_runtime_status(
        self, cancelled_jobs: Optional[list[str]] = None, *, status: str = "ok"
    ) -> dict:
        snap = self._dm.snapshot()  # refreshes the cache; /status is the refresh point
        worker = next((c for c in snap.consumers if c.label == "worker"), None)
        payload = {
            "status": status,
            "is_loaded": self.is_model_loaded(),
            "current_mode": self._current_mode,
            # The mode a queued switch is heading to. Distinguishes "nothing loaded"
            # from "loading X" during the window _load_mode leaves between
            # unregistering the outgoing mode and registering the new one — which was
            # silent, and read as a spontaneous unload.
            "pending_mode": self.get_pending_mode(),
            "queue_size": self.get_queue_size(),
            "vram": {
                "allocated_bytes": worker.allocated_bytes if worker else 0,
                "reserved_bytes": worker.reserved_bytes if worker else 0,
                "total_bytes": int(self._registry.get_total_vram()),
                "stale": worker.stale if worker else False,
            },
        }
        if cancelled_jobs is not None:
            payload["cancelled_jobs"] = cancelled_jobs
        return payload

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
                if not self._worker_available():
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
        if not self._worker_available():
            return {"status": "skipped", "reason": "already_unloaded"}
        logger.info(f"[Governor] Evicting idle model '{self._current_mode}'")
        self._unload_current_worker(reason="idle_evict")
        return {"status": "evicted"}

    # --- Dispatch loop (verbatim _worker_loop with self._worker -> self._handle.worker) ---

    def _dispatch_loop(self):
        """Main dispatch loop — VERBATIM from _worker_loop (worker_pool.py:728-868)
        with self._worker -> self._handle.worker and self._unload_current_worker()
        -> self._handle.unload().

        The Governor drives record.sink directly (NOT handle.submit() — the
        handle's submit() is the facet-3 contract, unused in v1's in-proc path).
        """
        logger.info("[Governor] Dispatch loop started")
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            # STABL-bpsfmoke: bind the correlation id for THIS iteration. The token
            # is set here rather than inside a `with` so the ~170-line body keeps
            # its indentation and the diff stays reviewable; the reset lands in the
            # existing finally, next to task_done(), which is the only place that
            # runs on every exit path (including the `continue` at the cancel
            # check). Without the reset, job N's id appears on job N+1's lines and
            # on the loop's own idle lines — plausible, and wrong. Neither
            # getattr(..., None) nor ContextVar.set can raise, so this needs no
            # wrapper to satisfy STABL-hdzggeir.
            _log_token = log_context.job_id_var.set(getattr(job, "job_id", None))
            try:
                if isinstance(job, ModeSwitchJob):
                    # Mode-switch fast-path: a live worker already holds the target mode.
                    # _worker_available() is required for subprocess handles, whose
                    # .worker is always None even when a live process is loaded.
                    if self._worker_available() and self._current_mode == job.target_mode and not job.force:
                        result = {"mode": job.target_mode, "status": "already_loaded"}
                    else:
                        result = job.execute(self._handle.worker)
                        self._load_mode(job.target_mode, reservation=job.reservation)
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
                    if not self._worker_available() and self._active_snapshot is not None:
                        # STABL-zueslhah: model-load progress. This generate triggers
                        # its own (re)load, so surface a 'load' stage over on_progress
                        # (parent-side, mode-agnostic) — the client sees loading, not
                        # silence, before denoise steps stream. total=-1 => indeterminate.
                        _op = job_record.on_progress if job_record is not None else None
                        if _op is not None:
                            try:
                                _op(0, -1, "load")
                            except Exception:
                                pass
                        try:
                            self._reload_from_snapshot()
                        except Exception as load_err:
                            raise RuntimeError(f"Demand reload failed: {load_err}") from load_err
                        if _op is not None:
                            try:
                                _op(1, 1, "load")
                            except Exception:
                                pass

                    # Stale-epoch barrier. The dead-epoch / no-authority guard runs
                    # FIRST: a job whose target mode failed to load has no authority to
                    # run against, and the old `snapshot is not None` conjunct let it
                    # fall through the barrier entirely — reaching the paths below with
                    # no epoch check at all.
                    if generation_job is not None:
                        with self._job_lock:
                            snapshot = self._active_snapshot
                            dead = generation_job.resolution_epoch in self._dead_epochs
                        if dead:
                            raise ModeLoadFailedError(
                                f"job {generation_job.job_id} was admitted against epoch "
                                f"{generation_job.resolution_epoch}, whose mode load did "
                                f"not complete"
                            )
                        if snapshot is None:
                            # No authority at all: explicit unload, or nothing ever
                            # loaded. Keeps the established operator-facing wording —
                            # the condition is unchanged, only the point of rejection
                            # moved here from the handle, and every real path that
                            # clears the snapshot also drops the worker.
                            raise ModeLoadFailedError(
                                f"No worker available for generation: job "
                                f"{generation_job.job_id} has no active model authority "
                                f"(admitted against epoch {generation_job.resolution_epoch})"
                            )
                        if snapshot.resolution_epoch != generation_job.resolution_epoch:
                            raise StaleResolutionError(
                                f"job {generation_job.job_id} stamped epoch "
                                f"{generation_job.resolution_epoch} != active epoch "
                                f"{snapshot.resolution_epoch}"
                            )

                    if isinstance(job, GenerationJob):
                        # STABL-atzqpcte: the execution clock starts HERE — after the
                        # demand reload above and after the stale-epoch barrier, so
                        # neither is charged to the execution budget. A waiter reads
                        # this to tell "still queued" from "actually running".
                        if job_record is not None:
                            job_record.executing_since = time.monotonic()
                        if self._handle.worker is not None:
                            # --- IN-PROC PATH (v1, unchanged) ---
                            # STABL-zueslhah: pass the sink's Progress emitter so
                            # the worker's step callback streams through the same
                            # channel _FutureBridge already drives. None-safe.
                            _sink = job_record.sink if job_record is not None else None
                            result = job.execute(
                                self._handle.worker,
                                progress=(_sink.progress if _sink is not None else None),
                                should_cancel=(
                                    _cancel_predicate(job_record)
                                    if job_record is not None else None
                                ),
                            )

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
                            # --- SUBPROCESS PATH (facet-3): handle owns the IPC channel ---
                            from backends.worker_handle_subprocess import _SubprocessFutureBridge
                            _op = job_record.on_progress if job_record is not None else None
                            bridge = _SubprocessFutureBridge(job.fut, on_progress=_op)
                            self._handle.submit(job).subscribe(bridge)
                            # Serialize (recon #5E): the single subprocess runs one job at
                            # a time — block on THIS job's terminal before dequeuing the next.
                            try:
                                job.fut.result()          # bridge fulfils it; wait for the terminal
                            except Exception:
                                pass                       # already delivered to the caller via the bridge
                            if job_record is not None:
                                self._finalize_job_record(job.job_id)
                            # Task 7: durable OOM / frameless-death recovery.
                            # In-band OOM leaves the child alive but poisoned; frameless
                            # death leaves it dead. Both require explicit unregister +
                            # kill + demand-reload so the next job runs on a fresh process.
                            oom = bridge.terminal_error_code == BackplaneErrorCode.OOM
                            if oom or not self._worker_available():
                                logger.warning(
                                    "[Governor] Subprocess needs recovery "
                                    f"(oom={oom}, alive={self._worker_available()}); kill+respawn"
                                )
                                self._count_worker_recovery(oom=oom)
                                if self._current_mode:
                                    self._registry.unregister_model(self._current_mode)  # idempotent; recon #4 dirty-death complement
                                self._handle.stop()                     # kills the poisoned-but-alive OR already-dead process
                                if self._active_snapshot is not None:
                                    self._reload_from_snapshot()        # -> handle.start() respawns + re-registers
                    else:
                        # CustomJob: run directly (in-proc callable, D4 defers redesign)
                        result = job.execute(self._handle.worker)
                        if not job.fut.done():
                            job.fut.set_result(result)

            except Exception as e:
                logger.error(f"[Governor] Job failed: {e}", exc_info=True)
                # STABL-hdzggeir: the handler is wrapped because a raise INSIDE it
                # escapes this try, escapes the while loop, and kills the dispatch
                # thread — after which the queue is permanently dead, every later job
                # hangs until its own timeout with no error surfaced, and shutdown()
                # blocks forever on q.join(). Whatever fails here, the caller's future
                # must still be resolved and the loop must still be running.
                try:
                    self._deliver_job_failure(job, e)
                except Exception:
                    logger.exception(
                        "[Governor] job error handling itself failed; delivering the "
                        "original error so the caller is not left waiting"
                    )
                    if not job.fut.done():
                        job.fut.set_exception(e)
            finally:
                log_context.job_id_var.reset(_log_token)
                self._last_activity = time.monotonic()
                self.q.task_done()
        logger.info("[Governor] Dispatch loop stopped")

    def _deliver_job_failure(self, job, e: Exception) -> None:
        """Classify a failed job, drive its terminal, and finalize its record.

        Extracted from the dispatch loop's except handler (STABL-hdzggeir) so the
        caller can wrap it: nothing in here may be allowed to kill the loop.
        """
        oom = _is_oom(e)
        if oom:
            logger.warning("[Governor] OOM recovery: cancelling + unloading")
            self._cleanup_vram(reason="oom", cancel_running=False)
        if isinstance(job, GenerationJob):
            job_record = self._get_job_record(job.job_id)
            if job_record is not None:
                sink = job_record.sink
                if oom:
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
                    # STABL-jredufxb: unwinding returned the intermediates to
                    # torch's caching allocator, not to the driver. Trim so the
                    # reaped bytes show up as free VRAM again. No-op for a
                    # subprocess consumer by design — see the spec.
                    self._dm.reclaim()
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

    def _start_dispatch_thread(self):
        if hasattr(self, '_worker_thread') and self._worker_thread and self._worker_thread.is_alive():
            logger.debug("[Governor] Dispatch thread already running")
            return
        self._worker_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="WorkerThread",
        )
        self._worker_thread.start()
        logger.info("[Governor] Dispatch thread started")

    # --- Submit / mode switch / reload / unload / free ---

    def submit_job(self, job: Job, *, timeout_s: float | None = None, on_progress=None) -> Future:
        """Submit a job — VERBATIM from worker_pool.py:870-916.

        Opens the backplane channel + attaches _FutureBridge BEFORE enqueueing
        (the backplane's Task 4 no-op pattern). The dispatch loop drives
        record.sink directly. `on_progress(step, total, stage)` (STABL-zueslhah),
        when given, receives the streamed Progress via the bridge; None preserves
        today's Future-only behaviour.
        """
        effective_timeout_s = self.queue_timeout_s if timeout_s is None else timeout_s
        try:
            self._register_job(job)
            if isinstance(job, GenerationJob):
                # Store on_progress on the record so BOTH paths reach it: the in-proc
                # branch below passes it to _FutureBridge; the subprocess dispatch
                # (which builds _SubprocessFutureBridge later) reads record.on_progress.
                record = self._get_job_record(job.job_id)
                if record is not None:
                    record.on_progress = on_progress
                if self._handle.worker is not None:
                    # Open the backplane channel and attach the compat Subscriber NOW —
                    # strictly before the job is enqueued (spec §3.3 ordering invariant).
                    # Subprocess path: handle.submit() owns the IPC channel; do NOT open
                    # an in-proc channel here (recon #2).
                    sink, publisher = InProcBackplane(job.job_id).open()
                    if record is not None:
                        record.sink = sink
                    publisher.subscribe(_FutureBridge(job.fut, on_progress=on_progress))
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
        self._mode_config.get_mode(mode_name)  # KeyError for unknown mode (unchanged)

        if not force:
            worker_ok = self._worker_available()  # handle call — NOT under _job_lock
            with self._job_lock:
                shortcircuit = self._switch_shortcircuit(mode_name, worker_ok)
            if shortcircuit is not None:
                fut: Future = Future()
                fut.set_result(shortcircuit)
                return fut

        return self._reserve_and_enqueue_switch(mode_name, force=force)

    def _reserve_and_enqueue_switch(self, mode_name: str, *, force: bool = False) -> Future:
        """Reserve the target's authority and enqueue its switch as ONE critical
        section (spec §3.4).

        Holding _job_lock across the bounded q.put cannot deadlock: the dispatch loop
        never holds _job_lock while blocked on q.get. Splitting the two would let a
        concurrent admitter interleave, inverting queue order against
        _pending_authorities — the queue would then load a different mode last than
        terminal authority claims.

        Bypasses submit_job deliberately — submit_job cannot hold the lock across its
        put, and a ModeSwitchJob needs no backplane channel (that is GenerationJob-only).
        """
        mode, resolved, binding = self._resolve_target(mode_name)  # disk I/O, no lock
        worker_ok = self._worker_available()  # handle call — NOT under _job_lock
        with self._job_lock:
            if not force:
                # Re-check under the lock: another thread may have reserved or
                # published this target while we were resolving. Without it, a
                # reservation could be minted for a mode the dispatch fast-path will
                # short-circuit — never published, and doom for anything bound to it.
                shortcircuit = self._switch_shortcircuit(mode_name, worker_ok)
                if shortcircuit is not None:
                    fut: Future = Future()
                    fut.set_result(shortcircuit)
                    return fut

            self._resolution_epoch += 1
            reservation = ActiveModelSnapshot(
                mode_name=mode_name,
                mode=mode,
                resolved=resolved,
                binding=binding,
                resolution_epoch=self._resolution_epoch,
            )
            self._pending_authorities.append(reservation)
            self._last_activity = time.monotonic()

            job = ModeSwitchJob(target_mode=mode_name, force=force, reservation=reservation)
            try:
                if self.queue_timeout_s > 0:
                    self.q.put(job, timeout=self.queue_timeout_s)
                else:
                    self.q.put_nowait(job)
            except queue.Full:
                self._pending_authorities = [
                    r for r in self._pending_authorities if r is not reservation
                ]
                self._dead_epochs.add(reservation.resolution_epoch)
                raise
            logger.debug(
                f"[Governor] Mode switch queued: {mode_name} "
                f"(epoch={reservation.resolution_epoch})"
            )
        # Counted OUTSIDE the critical section (STABL-asawxgvp). Moving the return
        # past the `with` is behaviour-preserving — nothing else sits between — and
        # keeps a metrics write out of the one lock the dispatch loop contends on.
        # Labelled by TARGET only: a {from,to} pair squares the cardinality to buy a
        # transition matrix nobody asked for.
        self._metric(lambda met: met.mode_switch_total.labels(mode=mode_name).inc())
        return job.fut

    def admit_generation(self, target_mode: Optional[str]) -> Optional[ActiveModelSnapshot]:
        """The authority a generate admitted NOW will execute against (spec §3.4).

        target_mode is None            -> the active snapshot (today's behavior). A
                                          generate naming no mode means 'the current
                                          mode'; binding it to a pending switch would
                                          silently run it on the wrong model.
        target_mode == terminal's mode -> the terminal authority (active OR a switch
                                          to it already queued).
        anything else                  -> reserve + enqueue the switch; return the
                                          reservation.

        Side-effecting by design: reserve-and-enqueue must be atomic, and a split
        accessor-plus-switch API reintroduces the interleave window this closes.
        """
        if target_mode is None:
            return self.get_active_model_snapshot()
        with self._job_lock:
            terminal = self._terminal_authority()
            if terminal is not None and terminal.mode_name == target_mode:
                return terminal
        self._reserve_and_enqueue_switch(target_mode)
        with self._job_lock:
            return self._terminal_authority()

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
        """Fully unload the model and drop the authority (mirrors
        WorkerPool.unload_current_model:783 — status 'unloaded', snapshot cleared)."""
        self._unload_current_worker()
        with self._job_lock:
            self._active_snapshot = None
            self._current_mode = None
        gc.collect()
        self._dm.reclaim()  # soft trim; the worker's own pool was flushed inline at teardown
        return self._build_runtime_status(status="unloaded")

    # --- Accessors ---

    def get_current_mode(self) -> Optional[str]:
        return self._current_mode

    def is_model_loaded(self) -> bool:
        return self._worker_available()

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
        if hasattr(self, '_worker_thread') and self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        if hasattr(self, '_watchdog_thread') and self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5.0)
        self._unload_current_worker(reason="shutdown")
        logger.info("[Governor] Shutdown complete")
