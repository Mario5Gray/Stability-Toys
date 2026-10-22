from __future__ import annotations

from typing import Any

from backends.platforms.base import (
    BackendCapabilities,
    FamilyPlatformBinding,
    GenerationRuntimeProtocol,
    ModelRegistryProtocol,
)
from backends.platforms.pool_runtime import PoolGenerationRuntime


class CudaGenerationRuntime(PoolGenerationRuntime):
    pass


class CUDAProvider:
    backend_id: str = "cuda"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True, True)

    def family_binding(self, family_id: str) -> FamilyPlatformBinding | None:
        from backends.platforms.cuda_bindings import CUDA_FAMILY_BINDINGS

        return CUDA_FAMILY_BINDINGS.get(family_id)

    def create_worker_factory(self, *args: Any, **kwargs: Any) -> Any:
        from backends.worker_factory import create_cuda_worker

        return create_cuda_worker

    def create_model_registry(self) -> ModelRegistryProtocol:
        from backends.model_registry import ModelRegistry

        return ModelRegistry()

    def create_generation_runtime(self, **kwargs: Any) -> GenerationRuntimeProtocol:
        return CudaGenerationRuntime(**kwargs)

    def create_superres_runtime(self, *, settings: Any, **kwargs: Any) -> Any:
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
