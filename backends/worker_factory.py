"""
Worker factory with automatic model type detection.

Handles automatic selection of SD1.5 vs SDXL workers based on model inspection.
"""

import os
import logging
from typing import TYPE_CHECKING, Optional

from utils.model_detector import ModelInfo, detect_model

if TYPE_CHECKING:
    from backends.base import PipelineWorker

logger = logging.getLogger(__name__)


def inspect_model(model_path: str) -> ModelInfo:
    """Inspect the model once and return its resolved capability metadata."""
    return detect_model(model_path)


def _worker_type_from_info(info: ModelInfo) -> str:
    """Map detected model architecture to the appropriate worker family."""
    logger.info(f"[ModelDetection] Detected variant: {info.variant.value}")
    logger.info(f"[ModelDetection] Cross-attention dim: {info.cross_attention_dim}")
    logger.info(f"[ModelDetection] Confidence: {info.confidence:.2f}")

    if info.cross_attention_dim in (2048, 1280):
        logger.info("[ModelDetection] Using SDXL worker")
        return "sdxl"
    if info.cross_attention_dim in (768, 1024):
        logger.info("[ModelDetection] Using SD1.5 worker")
        return "sd15"
    raise RuntimeError(
        f"Unsupported cross_attention_dim: {info.cross_attention_dim}. "
        f"Expected 768 (SD1.5), 1024 (SD2.x), 1280 (SDXL Refiner), or 2048 (SDXL Base)"
    )


def detect_worker_type(model_path: str) -> str:
    """
    Detect which worker to use based on the model file.

    Inspects the model at the given path to determine if it's SD1.5 or SDXL.

    Args:
        model_path: Resolved absolute path to the model file or directory

    Returns:
        "sdxl" if SDXL model (cross_attention_dim=2048)
        "sd15" if SD1.5/2.x model (cross_attention_dim=768/1024)

    Raises:
        RuntimeError if model not found or detection fails
    """
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at: {model_path}")

    logger.info(f"[ModelDetection] Detecting model type for: {model_path}")

    try:
        return _worker_type_from_info(inspect_model(model_path))
    except Exception as e:
        logger.error(f"[ModelDetection] Failed to detect model: {e}")
        raise RuntimeError(f"Model detection failed: {e}")


def create_cuda_worker(
    worker_id: int,
    model_path: str,
    model_info: Optional[ModelInfo] = None,
) -> "PipelineWorker":
    """
    Create a CUDA worker with automatic SD1.5/SDXL detection.

    Inspects the model at the given path, then creates the appropriate worker class.

    Args:
        worker_id: Worker ID to assign
        model_path: Resolved absolute path to the model

    Returns:
        DiffusersCudaWorker (SD1.5) or DiffusersSDXLCudaWorker (SDXL)

    Raises:
        RuntimeError if detection fails
    """
    if model_info is None:
        logger.info(f"[ModelDetection] Detecting model type for: {model_path}")
        try:
            model_info = inspect_model(model_path)
        except Exception as e:
            logger.error(f"[ModelDetection] Failed to detect model: {e}")
            raise RuntimeError(f"Model detection failed: {e}")

    worker_type = _worker_type_from_info(model_info)

    if worker_type == "sdxl":
        from backends.cuda_worker import DiffusersSDXLCudaWorker
        worker = DiffusersSDXLCudaWorker(
            worker_id=worker_id,
            model_path=model_path,
            model_info=model_info,
        )
        logger.info(f"[WorkerFactory] Created DiffusersSDXLCudaWorker (worker {worker_id})")
    else:  # sd15
        from backends.cuda_worker import DiffusersCudaWorker
        worker = DiffusersCudaWorker(
            worker_id=worker_id,
            model_path=model_path,
            model_info=model_info,
        )
        logger.info(f"[WorkerFactory] Created DiffusersCudaWorker (worker {worker_id})")

    return worker
