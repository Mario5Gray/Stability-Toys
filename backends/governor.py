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
        if reservation is None:
            reservation = self._reserve_authority(mode_name)
        mode = reservation.mode
        resolved, binding = reservation.resolved, reservation.binding

        self._unload_current_worker()  # unregister old mode + tear down worker
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

    def _unload_current_worker(self) -> None:
        """Unload the current worker. Registry-unregister is Governor authority
        (mirrors WorkerPool._unload_current_worker:322); worker + ControlNet-cache
        teardown is delegated to the handle. Guarded on worker-presence like the
        original, so a no-worker pool still clears the cache without unregistering."""
        if self._worker_available() and self._current_mode:
            self._registry.unregister_model(self._current_mode)
        self._handle.unload()

    def _cleanup_vram(self, reason: str, cancel_running: bool) -> list[str]:
        cancelled = self.cancel_pending_generation_jobs(reason=reason)
        if cancel_running:
            cancelled.extend(self._mark_running_generation_jobs_cancel_requested(reason=reason))
        self._unload_current_worker()
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
        self._unload_current_worker()
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
                        try:
                            self._reload_from_snapshot()
                        except Exception as load_err:
                            raise RuntimeError(f"Demand reload failed: {load_err}") from load_err

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
                        if self._handle.worker is not None:
                            # --- IN-PROC PATH (v1, unchanged) ---
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
                            # --- SUBPROCESS PATH (facet-3): handle owns the IPC channel ---
                            from backends.worker_handle_subprocess import _SubprocessFutureBridge
                            bridge = _SubprocessFutureBridge(job.fut)
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

    def submit_job(self, job: Job, *, timeout_s: float | None = None) -> Future:
        """Submit a job — VERBATIM from worker_pool.py:870-916.

        Opens the backplane channel + attaches _FutureBridge BEFORE enqueueing
        (the backplane's Task 4 no-op pattern). The dispatch loop drives
        record.sink directly.
        """
        effective_timeout_s = self.queue_timeout_s if timeout_s is None else timeout_s
        try:
            self._register_job(job)
            if isinstance(job, GenerationJob) and self._handle.worker is not None:
                # Open the backplane channel and attach the compat Subscriber NOW —
                # strictly before the job is enqueued (spec §3.3 ordering invariant).
                # Subprocess path: handle.submit() owns the IPC channel; do NOT open
                # an in-proc channel here (recon #2).
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
        """TEMPORARY (Task 2) — Task 4 makes reserve+enqueue one critical section."""
        reservation = self._reserve_authority(mode_name)
        job = ModeSwitchJob(target_mode=mode_name, force=force, reservation=reservation)
        try:
            return self.submit_job(job)
        except Exception:
            self._drop_reservation(reservation, dead=True)
            raise

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
        self._unload_current_worker()
        logger.info("[Governor] Shutdown complete")
