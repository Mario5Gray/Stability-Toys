from __future__ import annotations

import queue
from typing import Any

from backends.platforms.base import BackendCapabilities


class CudaGenerationRuntime:
    def __init__(self, *, pool: Any | None = None, **_: Any) -> None:
        if pool is None:
            from backends.worker_pool import get_worker_pool

            pool = get_worker_pool()
        self._pool = pool

    def submit_generate(self, req: Any, *, timeout_s: float = 0.25):
        from backends.worker_pool import GenerationJob

        job = GenerationJob(req=req)
        if timeout_s is None or timeout_s <= 0:
            return self._pool.submit_job(job)

        try:
            self._pool._register_job(job)  # type: ignore[attr-defined]
            self._pool.q.put(job, timeout=timeout_s)  # type: ignore[attr-defined]
            return job.fut
        except queue.Full:
            self._pool._finalize_job_record(job.job_id)  # type: ignore[attr-defined]
            max_size = getattr(self._pool, "queue_max", "unknown")
            raise queue.Full(
                f"Job queue full (max: {max_size}). "
                "Try again later or increase QUEUE_MAX."
            )

    def switch_mode(self, mode_name: str, force: bool = False):
        return self._pool.switch_mode(mode_name, force=force)

    def get_current_mode(self):
        return self._pool.get_current_mode()

    def is_model_loaded(self) -> bool:
        return self._pool.is_model_loaded()

    def get_queue_size(self) -> int:
        return self._pool.get_queue_size()

    def shutdown(self) -> None:
        self._pool.shutdown()


class CUDAProvider:
    backend_id = "cuda"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True, True, True)

    def create_worker_factory(self, *args: Any, **kwargs: Any):
        from backends.worker_factory import create_cuda_worker

        return create_cuda_worker

    def create_model_registry(self):
        from backends.model_registry import ModelRegistry

        return ModelRegistry()

    def create_generation_runtime(self, **kwargs: Any):
        return CudaGenerationRuntime(**kwargs)

    def create_superres_runtime(self, *, settings: Any, **kwargs: Any):
        from server.superres_http import initialize_superres_service

        return initialize_superres_service(
            enabled=settings.enabled,
            backend="cuda",
            use_cuda=True,
            sr_model_path=settings.sr_model_path,
            sr_num_workers=settings.sr_num_workers,
            sr_queue_max=settings.sr_queue_max,
            sr_input_size=settings.sr_input_size,
            sr_output_size=settings.sr_output_size,
            sr_max_pixels=settings.sr_max_pixels,
            **kwargs,
        )
