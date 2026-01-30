"""
Model management API endpoints.

Provides REST API for managing models, modes, and VRAM.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.mode_config import get_mode_config, reload_mode_config
from backends.model_registry import get_model_registry
from backends.worker_pool import get_worker_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ModeSwitchRequest(BaseModel):
    """Request to switch to a different mode."""
    mode: str


class ModelLoadRequest(BaseModel):
    """Request to load a specific model."""
    model_path: str
    mode_name: Optional[str] = None  # Optional mode name for registration


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/models/status")
async def get_models_status():
    """
    Get current model status and VRAM statistics.

    Returns:
        Current mode, loaded models, VRAM usage
    """
    pool = get_worker_pool()
    registry = get_model_registry()

    current_mode = pool.get_current_mode()
    vram_stats = registry.get_vram_stats()
    queue_size = pool.get_queue_size()

    return {
        "current_mode": current_mode,
        "queue_size": queue_size,
        "vram": vram_stats,
    }


@router.get("/modes")
async def list_modes():
    """
    List all available modes from configuration.

    Returns:
        List of mode names and their configurations
    """
    config = get_mode_config()

    modes_dict = config.to_dict()

    return {
        "default_mode": modes_dict["default_mode"],
        "modes": {
            name: {
                "model": mode_data["model"],
                "loras": mode_data["loras"],
                "default_size": mode_data["default_size"],
                "default_steps": mode_data["default_steps"],
                "default_guidance": mode_data["default_guidance"],
            }
            for name, mode_data in modes_dict["modes"].items()
        },
    }


@router.post("/modes/switch")
async def switch_mode(request: ModeSwitchRequest):
    """
    Switch to a different mode.

    Queues the mode switch - will execute after current jobs complete.

    Args:
        request: Mode switch request with target mode name

    Returns:
        Status message
    """
    pool = get_worker_pool()
    config = get_mode_config()

    # Validate mode exists
    try:
        config.get_mode(request.mode)
    except KeyError:
        available = config.list_modes()
        raise HTTPException(
            status_code=404,
            detail=f"Mode '{request.mode}' not found. Available modes: {available}",
        )

    # Check if already in this mode
    current = pool.get_current_mode()
    if current == request.mode:
        return {
            "status": "already_loaded",
            "mode": request.mode,
            "message": f"Already in mode '{request.mode}'",
        }

    # Queue mode switch
    try:
        future = pool.switch_mode(request.mode)
        logger.info(f"[API] Mode switch queued: {current} -> {request.mode}")

        return {
            "status": "queued",
            "from_mode": current,
            "to_mode": request.mode,
            "message": f"Mode switch queued. Will switch after {pool.get_queue_size()} pending jobs.",
        }
    except Exception as e:
        logger.error(f"[API] Mode switch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/modes/reload")
async def reload_modes_config():
    """
    Reload modes.yaml configuration from disk.

    Useful after editing the configuration file.

    Returns:
        Status message with loaded modes
    """
    try:
        reload_mode_config()
        config = get_mode_config()
        modes = config.list_modes()

        logger.info(f"[API] Configuration reloaded: {len(modes)} modes")

        return {
            "status": "reloaded",
            "modes_count": len(modes),
            "modes": modes,
            "default_mode": config.get_default_mode(),
        }
    except Exception as e:
        logger.error(f"[API] Config reload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload configuration: {e}",
        )


@router.get("/vram")
async def get_vram_stats():
    """
    Get detailed VRAM statistics.

    Returns:
        VRAM usage, available space, loaded models breakdown
    """
    registry = get_model_registry()
    return registry.get_vram_stats()


@router.post("/models/unload")
async def unload_current_model():
    """
    Unload the currently loaded model.

    WARNING: This will cause generation requests to fail until a new mode is loaded.

    Returns:
        Status message
    """
    pool = get_worker_pool()
    current_mode = pool.get_current_mode()

    if current_mode is None:
        raise HTTPException(status_code=400, detail="No model currently loaded")

    # TODO: Implement explicit unload in worker pool
    # For now, switching to a lightweight mode is recommended instead

    return {
        "status": "not_implemented",
        "message": "Model unload not yet implemented. Use mode switching instead.",
        "current_mode": current_mode,
    }


@router.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """
    Load a specific model.

    This is a low-level API - prefer using mode switching instead.

    Args:
        request: Model load request

    Returns:
        Status message
    """
    # TODO: Implement direct model loading without mode
    # For now, use modes.yaml and mode switching

    raise HTTPException(
        status_code=501,
        detail="Direct model loading not implemented. Use mode switching via /api/modes/switch",
    )
