from __future__ import annotations

from typing import Any

from backends.platforms.base import BackendCapabilities


def _bool_arg(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in ("0", "false", "no", "off")
    return bool(value)


class RknnGenerationRuntime:
    def __init__(
        self,
        *,
        paths,
        num_workers: int,
        queue_max: int,
        use_rknn_context_cfgs: bool,
    ) -> None:
        from backends.rknn_runtime import PipelineService, build_rknn_context_cfgs_for_rk3588

        self._service = PipelineService.get_instance(
            paths=paths,
            num_workers=num_workers,
            queue_max=queue_max,
            rknn_context_cfgs=build_rknn_context_cfgs_for_rk3588(num_workers),
            use_rknn_context_cfgs=use_rknn_context_cfgs,
        )

    def submit_generate(self, req: Any, *, timeout_s: float = 0.25):
        return self._service.submit(req, timeout_s=timeout_s)

    def get_current_mode(self):
        return None

    def is_model_loaded(self) -> bool:
        return True

    def get_queue_size(self) -> int:
        return self._service.q.qsize()

    def shutdown(self) -> None:
        self._service.shutdown()


class RKNNProvider:
    backend_id = "rknn"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, False, True, False, False)

    def create_worker_factory(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("RKNN worker-factory wiring lands in a later task")

    def create_model_registry(self):
        from backends.model_registry import PlaceholderModelRegistry

        return PlaceholderModelRegistry(self.backend_id)

    def create_generation_runtime(self, *, paths, num_workers: int, queue_max: int, **kwargs: Any):
        return RknnGenerationRuntime(
            paths=paths,
            num_workers=num_workers,
            queue_max=queue_max,
            use_rknn_context_cfgs=_bool_arg(kwargs.get("use_rknn_context_cfgs"), True),
        )

    def create_superres_runtime(self, *, settings: Any, **kwargs: Any):
        from server.superres_http import initialize_superres_service

        return initialize_superres_service(
            enabled=settings.enabled,
            backend="rknn",
            use_cuda=False,
            sr_model_path=settings.sr_model_path,
            sr_num_workers=settings.sr_num_workers,
            sr_queue_max=settings.sr_queue_max,
            sr_input_size=settings.sr_input_size,
            sr_output_size=settings.sr_output_size,
            sr_max_pixels=settings.sr_max_pixels,
            **kwargs,
        )
