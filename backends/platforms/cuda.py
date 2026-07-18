from __future__ import annotations

from typing import Any

from backends.platforms.base import (
    BackendCapabilities,
    FamilyPlatformBinding,
    GenerationRuntimeProtocol,
    ModelRegistryProtocol,
)


class CudaGenerationRuntime:
    def __init__(self, *, pool: Any | None = None, **_: Any) -> None:
        if pool is None:
            from backends.worker_pool import get_worker_pool

            pool = get_worker_pool()
        self._pool = pool

    def submit_generate(self, req: Any, *, timeout_s: float | None = None) -> Any:
        from server.asset_store import get_store
        from server.controlnet_execution import (
            active_model_family_from_variant,
            resolve_controlnet_bindings,
        )
        from server.mode_config import get_mode_config
        from utils.model_detector import detect_model
        from backends.worker_pool import GenerationJob

        bindings = []
        if getattr(req, "controlnets", None):
            mode_name = self._pool.get_current_mode()
            if mode_name is None:
                raise RuntimeError(
                    "CudaGenerationRuntime received a ControlNet request before any mode was loaded"
                )
            mode = get_mode_config().get_mode(mode_name)
            if mode.model_path is None:
                raise RuntimeError(f"Mode '{mode_name}' does not have a resolved model_path")
            family = active_model_family_from_variant(detect_model(mode.model_path).variant.value)
            bindings = resolve_controlnet_bindings(
                req,
                mode=mode,
                store=get_store(),
                active_family=family,
            )

        job = GenerationJob(
            req=req,
            controlnet_bindings=bindings,
            resolution_epoch=self._pool.current_resolution_epoch(),
        )
        return self._pool.submit_job(job, timeout_s=timeout_s)

    def switch_mode(self, mode_name: str, force: bool = False) -> Any:
        return self._pool.switch_mode(mode_name, force=force)

    def get_current_mode(self) -> str | None:
        return self._pool.get_current_mode()

    def is_model_loaded(self) -> bool:
        return self._pool.is_model_loaded()

    def get_queue_size(self) -> int:
        return self._pool.get_queue_size()

    def shutdown(self) -> None:
        self._pool.shutdown()


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
