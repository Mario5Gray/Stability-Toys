"""
Worker factory with automatic model type detection.

Handles automatic selection of SD1.5 vs SDXL workers based on model inspection.
"""

import os
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backends.base import PipelineWorker

logger = logging.getLogger(__name__)


def detect_worker_type() -> str:
    """
    Detect which worker to use based on the model file.

    Uses MODEL_ROOT and MODEL environment variables to locate the model,
    then inspects it to determine if it's SD1.5 or SDXL.

    Returns:
        "sdxl" if SDXL model (cross_attention_dim=2048)
        "sd15" if SD1.5/2.x model (cross_attention_dim=768/1024)

    Raises:
        RuntimeError if MODEL_ROOT or MODEL not set, or detection fails
    """
    from utils.model_detector import detect_model

    model_root = os.environ.get("MODEL_ROOT", "").strip()
    model_name = os.environ.get("MODEL", "").strip()

    if not model_root:
        raise RuntimeError("MODEL_ROOT environment variable is required")
    if not model_name:
        raise RuntimeError("MODEL environment variable is required")

    model_path = os.path.join(model_root, model_name)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at: {model_path}")

    logger.info(f"[ModelDetection] Detecting model type for: {model_path}")

    try:
        info = detect_model(model_path)

        logger.info(f"[ModelDetection] Detected variant: {info.variant.value}")
        logger.info(f"[ModelDetection] Cross-attention dim: {info.cross_attention_dim}")
        logger.info(f"[ModelDetection] Confidence: {info.confidence:.2f}")

        if info.cross_attention_dim == 2048:
            logger.info(f"[ModelDetection] Using SDXL worker")
            return "sdxl"
        elif info.cross_attention_dim in (768, 1024):
            logger.info(f"[ModelDetection] Using SD1.5 worker")
            return "sd15"
        else:
            raise RuntimeError(
                f"Unsupported cross_attention_dim: {info.cross_attention_dim}. "
                f"Expected 768 (SD1.5), 1024 (SD2.x), or 2048 (SDXL)"
            )
    except Exception as e:
        logger.error(f"[ModelDetection] Failed to detect model: {e}")
        raise RuntimeError(f"Model detection failed: {e}")


def create_cuda_worker(worker_id: int) -> "PipelineWorker":
    """
    Create a CUDA worker with automatic SD1.5/SDXL detection.

    Inspects the model specified by MODEL_ROOT and MODEL environment
    variables, then creates the appropriate worker class.

    Args:
        worker_id: Worker ID to assign

    Returns:
        DiffusersCudaWorker (SD1.5) or DiffusersSDXLCudaWorker (SDXL)

    Raises:
        RuntimeError if detection fails
    """
    worker_type = detect_worker_type()

    if worker_type == "sdxl":
        from backends.cuda_worker import DiffusersSDXLCudaWorker
        worker = DiffusersSDXLCudaWorker(worker_id=worker_id)
        logger.info(f"[WorkerFactory] Created DiffusersSDXLCudaWorker (worker {worker_id})")
    else:  # sd15
        from backends.cuda_worker import DiffusersCudaWorker
        worker = DiffusersCudaWorker(worker_id=worker_id)
        logger.info(f"[WorkerFactory] Created DiffusersCudaWorker (worker {worker_id})")

    return worker
