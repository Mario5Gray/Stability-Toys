import os
from typing import Callable, Mapping, Optional

from server.superres_service import (
    SuperResServiceProtocol,
    create_superres_service,
    load_cuda_superres_config,
    resolve_superres_backend,
)


def initialize_superres_service(
    *,
    enabled: bool,
    backend: str,
    use_cuda: bool,
    sr_model_path: str,
    sr_num_workers: int,
    sr_queue_max: int,
    sr_input_size: int,
    sr_output_size: int,
    sr_max_pixels: int,
    environ: Optional[Mapping[str, str]] = None,
    path_exists: Callable[[str], bool] = os.path.isfile,
    rknn_factory=None,
    cuda_factory=None,
) -> Optional[SuperResServiceProtocol]:
    if not enabled:
        return None

    backend_kind = resolve_superres_backend(backend=backend, use_cuda=use_cuda)
    if backend_kind == "rknn":
        if not path_exists(sr_model_path):
            raise RuntimeError(f"SR model not found at SR_MODEL_PATH={sr_model_path}")

        return create_superres_service(
            backend_kind="rknn",
            model_path=sr_model_path,
            num_workers=sr_num_workers,
            queue_max=sr_queue_max,
            input_size=sr_input_size,
            output_size=sr_output_size,
            max_pixels=sr_max_pixels,
            rknn_factory=rknn_factory,
        )

    cuda_config = load_cuda_superres_config(environ)
    if not cuda_config.model_path:
        raise RuntimeError("CUDA_SR_MODEL must be set for CUDA super-resolution")
    if not path_exists(cuda_config.model_path):
        raise RuntimeError(f"CUDA SR model not found at CUDA_SR_MODEL={cuda_config.model_path}")

    return create_superres_service(
        backend_kind="cuda",
        model_path=cuda_config.model_path,
        num_workers=1,
        queue_max=sr_queue_max,
        input_size=sr_input_size,
        output_size=sr_output_size,
        cuda_factory=cuda_factory,
        cuda_config=cuda_config,
    )


def submit_superres(
    *,
    sr_service: SuperResServiceProtocol,
    image_bytes: bytes,
    out_format: str,
    quality: int,
    magnitude: int,
    queue_timeout_s: float,
    request_timeout_s: float,
) -> bytes:
    fut = sr_service.submit(
        image_bytes=image_bytes,
        out_format=out_format,
        quality=quality,
        magnitude=magnitude,
        timeout_s=queue_timeout_s,
    )
    return fut.result(timeout=request_timeout_s)


def build_superres_headers(sr_service: object, *, magnitude: int, out_format: str) -> dict[str, str]:
    model_path = getattr(sr_service, "model_path", "")
    scale_per_pass = getattr(sr_service, "scale_per_pass", 1)
    return {
        "X-SR-Model": os.path.basename(model_path),
        "X-SR-Passes": str(int(magnitude)),
        "X-SR-Scale-Per-Pass": str(scale_per_pass),
        "X-SR-Format": out_format,
    }
