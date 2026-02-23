"""
Worker pool with extensible job queue system.

Manages worker lifecycle and job execution with support for:
- Generation jobs
- Mode switch jobs
- Custom job types

The queue is extensible - other parts of the app can submit jobs.
"""

import os
import logging
import queue
import threading
import torch
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
from concurrent.futures import Future
from enum import Enum

from server.mode_config import get_mode_config, ModeConfig, ModeConfigManager
from backends.model_registry import get_model_registry, ModelRegistry
from backends.base import PipelineWorker

logger = logging.getLogger(__name__)


# Type hints for dependency injection
class WorkerFactory(Protocol):
    """Protocol for worker creation functions."""
    def __call__(self, worker_id: int) -> PipelineWorker:
        """Create a worker with the given ID."""
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
    fut: Future = field(init=False, default=None)  # Result future

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

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.GENERATION

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        """Execute generation job."""
        if worker is None:
            raise RuntimeError("No worker available for generation")
        return worker.run_job(self)


@dataclass
class ModeSwitchJob(Job):
    """Job for switching model mode."""
    target_mode: str
    on_complete: Optional[Callable] = None

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
    kwargs: dict = None

    def __post_init__(self):
        super().__post_init__()
        self.job_type = JobType.CUSTOM
        if self.kwargs is None:
            self.kwargs = {}

    def execute(self, worker: Optional[PipelineWorker]) -> Any:
        """Execute custom handler."""
        return self.handler(*self.args, **self.kwargs)


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
        self._current_mode: Optional[str] = None
        self._lock = threading.Lock()

        # Dependency injection with defaults to singletons
        self._worker_factory = worker_factory or self._default_worker_factory
        self._mode_config = mode_config or get_mode_config()
        self._registry = registry or get_model_registry()

        # Initialize with default mode
        default_mode = self._mode_config.get_default_mode()
        self._load_mode(default_mode)

    @staticmethod
    def _default_worker_factory(worker_id: int) -> PipelineWorker:
        """
        Default worker factory.

        Imports and calls create_cuda_worker from worker_factory module.
        This is the default behavior when no factory is injected.

        Args:
            worker_id: Worker ID to assign

        Returns:
            Created PipelineWorker instance
        """
        from backends.worker_factory import create_cuda_worker
        return create_cuda_worker(worker_id)

    def _load_mode(self, mode_name: str):
        """
        Load a mode by creating appropriate worker.

        Args:
            mode_name: Name of mode to load
        """
        logger.info(f"[WorkerPool] Loading mode: {mode_name}")

        # Get mode configuration
        mode = self._mode_config.get_mode(mode_name)

        # Unload current worker if exists
        if self._worker is not None:
            self._unload_current_worker()

        # Set environment variables from mode config
        os.environ["MODEL_ROOT"] = self._mode_config.config.model_root
        os.environ["MODEL"] = mode.model

        # Track VRAM before worker creation
        vram_before = self._registry.get_used_vram()

        # Create worker using injected factory
        self._worker = self._worker_factory(worker_id=0)

        vram_after = self._registry.get_used_vram()
        vram_used = vram_after - vram_before

        # Load LoRAs if specified in mode
        if mode.loras:
            logger.info(f"[WorkerPool] Loading {len(mode.loras)} LoRAs for mode {mode_name}")
            # LoRAs are loaded by worker during initialization from STYLE_REGISTRY
            # TODO: Support dynamic LoRA loading from mode config

        # Register model in registry
        self._registry.register_model(
            name=mode_name,
            model_path=mode.model_path,
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

    def _unload_current_worker(self):
        """Unload current worker and free VRAM."""
        if self._worker is None:
            return

        logger.info(f"[WorkerPool] Unloading current worker (mode: {self._current_mode})")

        # Unregister from registry
        if self._current_mode:
            self._registry.unregister_model(self._current_mode)

        # Delete worker and free VRAM
        del self._worker
        self._worker = None

        # Force garbage collection and VRAM release
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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
                # Check if this is a mode switch job
                if isinstance(job, ModeSwitchJob):
                    # Check if already in target mode (skip if so)
                    if self._current_mode == job.target_mode:
                        logger.info(
                            f"[WorkerPool] Already in mode '{job.target_mode}', "
                            "skipping mode switch"
                        )
                        result = {"mode": job.target_mode, "status": "already_loaded"}
                    else:
                        # Execute mode switch
                        result = job.execute(self._worker)

                        # Reload worker with new mode
                        self._load_mode(job.target_mode)

                    # Set result
                    if not job.fut.done():
                        job.fut.set_result(result)

                else:
                    # Regular job - execute with current worker
                    result = job.execute(self._worker)

                    if not job.fut.done():
                        job.fut.set_result(result)

            except Exception as e:
                logger.error(f"[WorkerPool] Job failed: {e}", exc_info=True)
                if not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                # Mark task as done for queue.join() to work
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
            self.q.put_nowait(job)
            logger.debug(f"[WorkerPool] Job queued: {job.job_type.value}")
            return job.fut
        except queue.Full:
            raise queue.Full(
                f"Job queue full (max: {self.queue_max}). "
                "Try again later or increase QUEUE_MAX."
            )

    def switch_mode(self, mode_name: str) -> Future:
        """
        Queue a mode switch.

        Args:
            mode_name: Target mode name

        Returns:
            Future that completes when mode switch is done
        """
        logger.info(f"[WorkerPool] Queueing mode switch to: {mode_name}")

        # Validate mode exists
        self._mode_config.get_mode(mode_name)  # Raises if not found

        # Create mode switch job
        job = ModeSwitchJob(target_mode=mode_name)

        return self.submit_job(job)

    def get_current_mode(self) -> Optional[str]:
        """Get currently loaded mode name."""
        return self._current_mode

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

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

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
