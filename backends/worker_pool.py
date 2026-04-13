"""
Worker pool with extensible job queue system.

Manages worker lifecycle and job execution with support for:
- Generation jobs
- Mode switch jobs
- Custom job types

The queue is extensible - other parts of the app can submit jobs.
"""

import gc
import os
import logging
import queue
import threading
import time
import uuid
from copy import deepcopy
import torch
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
from concurrent.futures import Future, CancelledError
from enum import Enum

from server.mode_config import get_mode_config, ModeConfig, ModeConfigManager
from backends.model_registry import get_model_registry, ModelRegistry
from backends.base import PipelineWorker
from utils.model_detector import ModelInfo, detect_model

logger = logging.getLogger(__name__)


# Type hints for dependency injection
class WorkerFactory(Protocol):
    """Protocol for worker creation functions."""
    def __call__(
        self,
        worker_id: int,
        model_path: str,
        model_info: Optional[ModelInfo] = None,
    ) -> PipelineWorker:
        """Create a worker with the given ID and resolved model path."""
        ...


def merge_mode_capabilities(model_info: ModelInfo, mode: ModeConfig) -> ModelInfo:
    """Overlay authoritative mode-level capability overrides onto detected model info."""
    resolved = deepcopy(model_info)
    for field in (
        "loader_format",
        "checkpoint_precision",
        "checkpoint_variant",
        "scheduler_profile",
        "recommended_size",
        "runtime_quantize",
        "runtime_offload",
        "runtime_attention_slicing",
        "runtime_enable_xformers",
        "negative_prompt_templates",
        "default_negative_prompt_template",
        "allow_custom_negative_prompt",
        "allowed_scheduler_ids",
        "default_scheduler_id",
    ):
        value = getattr(mode, field, None)
        if value is not None:
            setattr(resolved, field, value)
    return resolved


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
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

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


class WorkerPool:
    """
    Manages worker lifecycle and extensible job queue.

    Features:
    - Single worker mode (recreate on mode switch)
    - Extensible job queue (generation, mode switch, custom)
    - Mode switching with automatic worker recreation
    - VRAM tracking via ModelRegistry
    - Dependency injection support for testing
    """

    def __init__(
        self,
        queue_max: int = 64,
        worker_factory: Optional[WorkerFactory] = None,
        mode_config: Optional[ModeConfigManager] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize worker pool.

        Args:
            queue_max: Maximum queue size
            worker_factory: Optional factory function for creating workers.
                           Defaults to create_cuda_worker from worker_factory module.
            mode_config: Optional mode configuration manager.
                        Defaults to global singleton from get_mode_config().
            registry: Optional model registry for VRAM tracking.
                     Defaults to global singleton from get_model_registry().

        Note:
            When all optional parameters are None (default), uses global singletons
            for backward compatibility. For testing, inject mocked dependencies.
        """
        self.queue_max = queue_max
        self.q: queue.Queue[Job] = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._worker: Optional[PipelineWorker] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._current_mode: Optional[str] = None
        self._job_records: dict[str, JobRecord] = {}
        self._job_lock = threading.RLock()
        # Idle eviction config — 0 disables eviction
        self._idle_timeout = float(os.environ.get("MODEL_IDLE_TIMEOUT_SECS", "300"))
        self._idle_check_interval = float(os.environ.get("MODEL_IDLE_CHECK_INTERVAL_SECS", "30"))
        self._last_activity = time.monotonic()
        self._eviction_pending = False

        # Dependency injection with defaults to singletons
        self._worker_factory = worker_factory or self._default_worker_factory
        self._mode_config = mode_config or get_mode_config()
        self._registry = registry or get_model_registry()

        # Initialize with default mode — failure is non-fatal so the server
        # can still start and accept mode-switch or load requests via API.
        default_mode = self._mode_config.get_default_mode()
        try:
            self._load_mode(default_mode)
        except Exception as e:
            logger.error(
                f"[WorkerPool] Initial model load failed for mode '{default_mode}': {e}. "
                "Server will start without a loaded model. "
                "Use the /api/modes/switch endpoint to load a model.",
                exc_info=True,
            )
            # _load_mode already cleaned up; pool starts in no-model state
            self._start_worker_thread()

        self._start_watchdog_thread()

    @staticmethod
    def _default_worker_factory(
        worker_id: int,
        model_path: str,
        model_info: Optional[ModelInfo] = None,
    ) -> PipelineWorker:
        """
        Default worker factory.

        Imports and calls create_cuda_worker from worker_factory module.
        This is the default behavior when no factory is injected.

        Args:
            worker_id: Worker ID to assign
            model_path: Resolved absolute path to the model
            model_info: Optional pre-resolved model capabilities

        Returns:
            Created PipelineWorker instance
        """
        from backends.worker_factory import create_cuda_worker
        return create_cuda_worker(worker_id, model_path, model_info=model_info)

    def _load_mode(self, mode_name: str):
        """
        Load a mode by creating appropriate worker.

        Args:
            mode_name: Name of mode to load

        Raises:
            Exception: Re-raises any load failure after cleaning up partial state.
                       On failure, worker is None and current_mode is None.
        """
        logger.info(f"[WorkerPool] Loading mode: {mode_name}")

        # Get mode configuration
        mode = self._mode_config.get_mode(mode_name)

        # Unload current worker if exists
        if self._worker is not None:
            self._unload_current_worker()

        # Track VRAM before worker creation
        self._registry.get_used_vram()
        allocated_before = self._registry.get_allocated_vram()

        assert mode.model_path is not None, f"model_path not resolved for mode '{mode_name}'"
        try:
            model_info = merge_mode_capabilities(detect_model(mode.model_path), mode)
            # Create worker using injected factory, passing fully-resolved model path
            self._worker = self._worker_factory(
                worker_id=0,
                model_path=mode.model_path,
                model_info=model_info,
            )
        except Exception as e:
            logger.error(
                f"[WorkerPool] Failed to load mode '{mode_name}': {e}",
                exc_info=True,
            )
            # Clean up any partially allocated GPU memory
            self._free_worker()
            self._current_mode = None
            raise

        vram_reserved = self._registry.get_used_vram()
        vram_allocated = self._registry.get_allocated_vram()
        vram_used = max(0, vram_allocated - allocated_before)
        vram_total = self._registry.get_total_vram()
        logger.info(
            f"[WorkerPool] VRAM after load: "
            f"allocated={vram_allocated/1024**3:.2f}GB "
            f"reserved={vram_reserved/1024**3:.2f}GB "
            f"total={vram_total/1024**3:.2f}GB "
            f"model_delta={vram_used/1024**3:.2f}GB"
        )

        # Load LoRAs if specified in mode
        if mode.loras:
            logger.info(f"[WorkerPool] Loading {len(mode.loras)} LoRAs for mode {mode_name}")
            # LoRAs are loaded by worker during initialization from STYLE_REGISTRY
            # TODO: Support dynamic LoRA loading from mode config

        # Register model in registry
        self._registry.register_model(
            name=mode_name,
            model_path=mode.model_path or "",
            vram_bytes=vram_used,
            worker_id=0,
            loras=[lora.path for lora in mode.loras],
        )

        self._current_mode = mode_name

        # Start worker thread
        self._start_worker_thread()

        logger.info(
            f"[WorkerPool] Mode '{mode_name}' loaded successfully "
            f"(VRAM: {vram_used / 1024**3:.2f} GB)"
        )

    def _free_worker(self):
        """Drop the worker reference and flush the GPU allocator cache."""
        if self._worker is not None:
            del self._worker
            self._worker = None
        gc.collect()
        torch.cuda.empty_cache()

    def _unload_current_worker(self):
        """Unload current worker and free VRAM."""
        if self._worker is None:
            return

        logger.info(f"[WorkerPool] Unloading current worker (mode: {self._current_mode})")

        # Unregister from registry
        if self._current_mode:
            self._registry.unregister_model(self._current_mode)

        self._free_worker()

        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"[WorkerPool] Worker unloaded — "
                f"allocated={vram_allocated:.2f}GB reserved={vram_reserved:.2f}GB "
                f"(reserved>allocated means PyTorch cache; call empty_cache to release)"
            )
        else:
            logger.info("[WorkerPool] Worker unloaded, VRAM freed")

    def _start_worker_thread(self):
        """Start worker thread for processing jobs."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("[WorkerPool] Worker thread already running")
            return

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="WorkerThread",
        )
        self._worker_thread.start()
        logger.info("[WorkerPool] Worker thread started")

    def _start_watchdog_thread(self):
        """Start idle eviction watchdog thread."""
        if self._idle_timeout <= 0:
            logger.info("[WorkerPool] Idle eviction disabled (MODEL_IDLE_TIMEOUT_SECS=0)")
            return

        self._watchdog_thread = threading.Thread(
            target=self._idle_watchdog_loop,
            daemon=True,
            name="IdleWatchdog",
        )
        self._watchdog_thread.start()
        logger.info(
            f"[WorkerPool] Idle watchdog started "
            f"(timeout={self._idle_timeout:.0f}s, interval={self._idle_check_interval:.0f}s)"
        )

    def _idle_watchdog_loop(self):
        """Background thread: evicts model after idle timeout."""
        logger.debug("[WorkerPool] Idle watchdog loop running")

        while not self._stop.wait(timeout=self._idle_check_interval):
            try:
                if self._worker is None:
                    continue

                idle_secs = time.monotonic() - self._last_activity
                if idle_secs < self._idle_timeout:
                    continue

                if self._eviction_pending:
                    continue

                logger.info(
                    f"[WorkerPool] Model idle for {idle_secs:.0f}s "
                    f"(timeout={self._idle_timeout:.0f}s); queuing eviction"
                )
                try:
                    evict_job = CustomJob(handler=self._evict_if_idle)
                    self._eviction_pending = True
                    self.q.put_nowait(evict_job)
                except queue.Full:
                    self._eviction_pending = False
                    logger.warning("[WorkerPool] Queue full; skipping idle eviction this cycle")
            except Exception:
                logger.error("[WorkerPool] Idle watchdog error", exc_info=True)

        logger.debug("[WorkerPool] Idle watchdog loop stopped")

    def _register_job(self, job: Job):
        if isinstance(job, GenerationJob):
            with self._job_lock:
                self._job_records[job.job_id] = JobRecord(
                    job_id=job.job_id,
                    state="queued",
                    job=job,
                )

    def _finalize_job_record(self, job_id: str):
        with self._job_lock:
            self._job_records.pop(job_id, None)

    def _get_job_record(self, job_id: str) -> Optional[JobRecord]:
        with self._job_lock:
            return self._job_records.get(job_id)

    def _mark_running_generation_jobs_cancel_requested(self, reason: str) -> list[str]:
        """Mark running generation jobs so their futures resolve as cancelled."""
        cancelled: list[str] = []
        with self._job_lock:
            for record in self._job_records.values():
                if record.state == "running" and not record.cancel_requested:
                    record.cancel_requested = True
                    cancelled.append(record.job_id)

        if cancelled:
            logger.info(
                f"[WorkerPool] Marked {len(cancelled)} running generation job(s) cancel requested "
                f"({reason})"
            )

        return cancelled

    def cancel_pending_generation_jobs(self, reason: str) -> list[str]:
        """Cancel queued generation jobs that have not started yet."""
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
            logger.info(
                f"[WorkerPool] Cancelled {len(cancelled)} pending generation job(s) ({reason})"
            )

        return cancelled

    def _cleanup_vram(self, reason: str, cancel_running: bool) -> list[str]:
        """Shared cleanup path for explicit free-VRAM and OOM recovery."""
        cancelled = self.cancel_pending_generation_jobs(reason=reason)
        if cancel_running:
            cancelled.extend(self._mark_running_generation_jobs_cancel_requested(reason=reason))
        self._unload_current_worker()
        gc.collect()
        torch.cuda.empty_cache()
        return cancelled

    def _build_runtime_status(self, cancelled_jobs: Optional[list[str]] = None) -> dict:
        """Return a stable runtime snapshot used by recovery endpoints."""
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

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running generation job by backend job ID."""
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

    def _evict_if_idle(self):
        """
        Evict the loaded model if the pool is still idle.

        Runs on the worker thread (via CustomJob) to serialise with generation.
        Re-checks the idle condition in case a job arrived after the watchdog
        enqueued this eviction.
        """
        self._eviction_pending = False
        idle_secs = time.monotonic() - self._last_activity
        if idle_secs < self._idle_timeout:
            logger.debug("[WorkerPool] Eviction skipped: activity detected since enqueue")
            return {"status": "skipped", "reason": "activity_detected"}
        if self._worker is None:
            return {"status": "skipped", "reason": "already_unloaded"}

        logger.info(f"[WorkerPool] Evicting idle model '{self._current_mode}'")
        self._unload_current_worker()
        return {"status": "evicted"}

    def _worker_loop(self):
        """Main worker loop - processes jobs from queue."""
        logger.info("[WorkerPool] Worker loop started")

        while not self._stop.is_set():
            try:
                # Get job with timeout to allow checking stop flag
                job = self.q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if isinstance(job, ModeSwitchJob):
                    # Skip only if worker is live and already on the right mode
                    if self._worker is not None and self._current_mode == job.target_mode and not job.force:
                        logger.info(
                            f"[WorkerPool] Already in mode '{job.target_mode}', "
                            "skipping mode switch"
                        )
                        result = {"mode": job.target_mode, "status": "already_loaded"}
                    else:
                        result = job.execute(self._worker)
                        self._load_mode(job.target_mode)

                    if not job.fut.done():
                        job.fut.set_result(result)

                else:
                    job_record = self._get_job_record(job.job_id) if isinstance(job, GenerationJob) else None
                    if job_record is not None and (job_record.cancel_requested or job.fut.cancelled()):
                        logger.info(f"[WorkerPool] Skipping cancelled generation job: {job.job_id}")
                        job_record.state = "cancelled"
                        self._finalize_job_record(job.job_id)
                        continue

                    if job_record is not None:
                        job_record.state = "running"

                    # Demand reload: worker may have been evicted since last job
                    if self._worker is None and self._current_mode is not None:
                        logger.info(
                            f"[WorkerPool] Worker was evicted; "
                            f"demand-reloading mode '{self._current_mode}'"
                        )
                        try:
                            self._load_mode(self._current_mode)
                        except Exception as load_err:
                            raise RuntimeError(
                                f"Demand reload of '{self._current_mode}' failed: {load_err}"
                            ) from load_err

                    result = job.execute(self._worker)

                    if job_record is not None and job_record.cancel_requested:
                        job_record.state = "cancelled"
                        if not job.fut.done():
                            job.fut.set_exception(CancelledError())
                        self._finalize_job_record(job.job_id)
                    elif not job.fut.done():
                        job.fut.set_result(result)
                        if job_record is not None:
                            self._finalize_job_record(job.job_id)

            except Exception as e:
                logger.error(f"[WorkerPool] Job failed: {e}", exc_info=True)
                _oom = (
                    hasattr(torch.cuda, "OutOfMemoryError")
                    and isinstance(e, torch.cuda.OutOfMemoryError)
                ) or "out of memory" in str(e).lower()
                if _oom:
                    logger.warning(
                        "[WorkerPool] OOM recovery: cancelling queued jobs and unloading worker — "
                        f"allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB "
                        f"reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB"
                    )
                    # OOM can leave the pipeline allocator state partially poisoned.
                    # Use the same cleanup path as explicit free-VRAM, but keep the
                    # failing job's original exception so callers see the OOM.
                    self._cleanup_vram(reason="oom", cancel_running=False)
                if isinstance(job, GenerationJob):
                    job_record = self._get_job_record(job.job_id)
                    if job_record is not None:
                        if _oom:
                            if not job.fut.done():
                                job.fut.set_exception(e)
                            job_record.state = "failed"
                        elif job_record.cancel_requested:
                            if not job.fut.done():
                                job.fut.set_exception(CancelledError())
                            job_record.state = "cancelled"
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

        logger.info("[WorkerPool] Worker loop stopped")

    def submit_job(self, job: Job) -> Future:
        """
        Submit a job to the queue.

        Extensible - accepts any Job subclass.

        Args:
            job: Job to execute

        Returns:
            Future for job result

        Raises:
            queue.Full if queue is full
        """
        try:
            self._register_job(job)
            self.q.put_nowait(job)
            logger.debug(f"[WorkerPool] Job queued: {job.job_type.value}")
            return job.fut
        except queue.Full:
            if isinstance(job, GenerationJob):
                self._finalize_job_record(job.job_id)
            raise queue.Full(
                f"Job queue full (max: {self.queue_max}). "
                "Try again later or increase QUEUE_MAX."
            )

    def switch_mode(self, mode_name: str, force: bool = False) -> Future:
        """
        Queue a mode switch.

        Args:
            mode_name: Target mode name
            force: Reload the worker even if mode_name is already current.
                   Use this when the mode's config has changed on disk.

        Returns:
            Future that completes when mode switch is done
        """
        logger.info(f"[WorkerPool] Queueing mode switch to: {mode_name} (force={force})")

        # Validate mode exists
        self._mode_config.get_mode(mode_name)  # Raises if not found

        job = ModeSwitchJob(target_mode=mode_name, force=force)

        return self.submit_job(job)

    def reload_current_mode(self) -> dict:
        """Reload the currently loaded mode in place."""
        if self._current_mode is None:
            raise RuntimeError("No active mode to reload")

        self.cancel_pending_generation_jobs(reason="reload_current_mode")
        self.switch_mode(self._current_mode, force=True).result(timeout=30.0)
        return {"status": "reloaded", "mode": self._current_mode}

    def free_vram(self, reason: str) -> dict:
        """Cancel queued work, unload the worker, and return a runtime snapshot."""
        cancelled = self._cleanup_vram(reason=reason, cancel_running=True)
        return self._build_runtime_status(cancelled_jobs=cancelled)

    def unload_current_model(self) -> dict:
        """Unload the live worker without canceling queued or running jobs."""
        self._unload_current_worker()
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "unloaded",
            "is_loaded": self.is_model_loaded(),
            "current_mode": self._current_mode,
            "queue_size": self.get_queue_size(),
            "vram": {
                "allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
                "reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
                "total_bytes": int(self._registry.get_total_vram()),
            },
        }

    def get_current_mode(self) -> Optional[str]:
        """Get currently loaded mode name.

        Note: returns the mode name even after idle eviction, so the pool can
        demand-reload the same mode on the next request. Use is_model_loaded()
        to distinguish "in VRAM" from "evicted but name retained".
        """
        return self._current_mode

    def is_model_loaded(self) -> bool:
        """True if a worker is currently live in GPU memory."""
        return self._worker is not None

    def reload_if_current(self, mode_name: str) -> bool:
        """Queue a force-reload if mode_name is the currently loaded mode.

        Returns True if a reload was queued, False otherwise.
        Intended for route handlers that need to hot-reload after a config change.
        """
        if self.get_current_mode() != mode_name:
            return False
        logger.info(f"[WorkerPool] Config changed for loaded mode '{mode_name}'; queuing reload")
        try:
            self.switch_mode(mode_name, force=True)
            return True
        except Exception as e:
            logger.warning(f"[WorkerPool] Could not queue reload for mode '{mode_name}': {e}")
            return False

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.q.qsize()

    def shutdown(self):
        """
        Shutdown worker pool.

        Waits for pending jobs to complete before shutting down.
        """
        logger.info("[WorkerPool] Shutting down")

        # Wait for queue to drain (pending jobs complete)
        logger.debug(f"[WorkerPool] Waiting for {self.q.qsize()} jobs to complete")
        self.q.join()

        # Signal worker thread to stop
        self._stop.set()

        # Wait for worker and watchdog threads to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5.0)

        # Unload worker
        self._unload_current_worker()

        logger.info("[WorkerPool] Shutdown complete")


# Global worker pool instance
_worker_pool: Optional[WorkerPool] = None


def get_worker_pool(
    worker_factory: Optional[WorkerFactory] = None,
    mode_config: Optional[ModeConfigManager] = None,
    registry: Optional[ModelRegistry] = None,
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
        _worker_pool = WorkerPool(
            queue_max=queue_max,
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
