"""
Model management API endpoints.

Provides REST API for managing models, modes, and VRAM.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

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
    model_config = ConfigDict(protected_namespaces=())
    model_path: str
    mode_name: Optional[str] = None  # Optional mode name for registration

#
#
#
def scan_models(models_root: Path) -> Dict[str, List[Path]]:
    """
    Scan a models directory for available models.

    Expected structure (ComfyUI-compatible):
      models/
        checkpoints/       - single-file .safetensors / .ckpt
        diffusers/         - pipeline roots (dirs with model_index.json)
        diffusion_models/  - UNet-format models (FLUX, etc.)
        loras/             - LoRA .safetensors files

    Returns:
      {
        "checkpoints": [Path, ...],
        "diffusers": [Path, ...],          # pipeline root dirs
        "diffusion_models": [Path, ...],
        "loras": [Path, ...],
      }
    """
    models_root = Path(models_root)
    results: Dict[str, List[Path]] = {
        "checkpoints": [],
        "diffusers": [],
        "diffusion_models": [],
        "loras": [],
    }

    # ---- loras (.safetensors) ---
    loras_dir = models_root / "loras"
    if loras_dir.exists():
        results["loras"] = sorted(
            p for p in loras_dir.rglob("*.safetensors") if p.is_file()
        )

    # ---- Checkpoints (.safetensors) ----
    checkpoints_dir = models_root / "checkpoints"
    if checkpoints_dir.exists():
        results["checkpoints"] = sorted(
            p for p in checkpoints_dir.rglob("*.safetensors") if p.is_file()
        )

    # ---- Diffusers (pipeline root = has model_index.json) ----
    diffusers_dir = models_root / "diffusers"
    if diffusers_dir.exists():
        roots = set()
        for mi in diffusers_dir.rglob("model_index.json"):
            if mi.is_file():
                roots.add(mi.parent)
        results["diffusers"] = sorted(roots)

    # ---- Diffusion models / UNet-format (.safetensors) ----
    diffusion_models_dir = models_root / "diffusion_models"
    if diffusion_models_dir.exists():
        results["diffusion_models"] = sorted(
            p for p in diffusion_models_dir.rglob("*.safetensors") if p.is_file()
        )

    return results
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
        "is_loaded": pool.is_model_loaded(),
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
                "loader_format": mode_data.get("loader_format"),
                "checkpoint_precision": mode_data.get("checkpoint_precision"),
                "checkpoint_variant": mode_data.get("checkpoint_variant"),
                "scheduler_profile": mode_data.get("scheduler_profile"),
                "recommended_size": mode_data.get("recommended_size"),
                "negative_prompt_templates": mode_data.get("negative_prompt_templates", {}),
                "default_negative_prompt_template": mode_data.get("default_negative_prompt_template"),
                "allow_custom_negative_prompt": mode_data.get("allow_custom_negative_prompt", False),
                "allowed_scheduler_ids": mode_data.get("allowed_scheduler_ids"),
                "default_scheduler_id": mode_data.get("default_scheduler_id"),
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
        pool.switch_mode(request.mode)
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


# ============================================================================
# Inventory Endpoints
# ============================================================================

@router.get("/inventory/models")
async def get_inventory_models():
    """Scan MODEL_ROOT for available model directories."""
    config = get_mode_config()
    model_root = Path(config.config.model_root)

    models = scan_models(model_root)
    all_models = []
    for category in ("checkpoints", "diffusers", "diffusion_models"):
        for p in models[category]:
            all_models.append(str(p.relative_to(model_root)))

    return {"models": all_models, "model_root": str(model_root)}


@router.get("/inventory/loras")
async def get_inventory_loras():
    """Scan LORAS_ROOT for available LoRA files."""
    config = get_mode_config()
    lora_root = Path(config.config.lora_root)

    loras = scan_models(lora_root.parent)["loras"]
    lora_strings = [str(p.relative_to(lora_root)) for p in loras]

    return {"loras": lora_strings, "lora_root": str(lora_root)}


# ============================================================================
# Mode CRUD Endpoints
# ============================================================================

class ModeCreateRequest(BaseModel):
    model: str
    loras: List[Dict[str, Any]] = []
    default_size: str = "512x512"
    default_steps: int = 4
    default_guidance: float = 1.0


class ModesBulkSaveRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_root: str
    lora_root: str
    default_mode: str
    modes: Dict[str, Any]


@router.put("/modes")
async def save_all_modes(request: ModesBulkSaveRequest):
    """Save full modes config, write to disk and reload."""
    config = get_mode_config()
    pool = get_worker_pool()
    data = request.model_dump()

    if not data.get("modes"):
        raise HTTPException(status_code=400, detail="At least one mode must exist")

    if data["default_mode"] not in data["modes"]:
        raise HTTPException(
            status_code=400,
            detail=f"default_mode '{data['default_mode']}' not found in modes",
        )

    # Snapshot current mode config before saving so we can detect changes
    current_mode = pool.get_current_mode()
    old_model = None
    old_loras = None
    if current_mode:
        try:
            old_cfg = config.get_mode(current_mode)
            old_model = old_cfg.model
            old_loras = [lora.path for lora in old_cfg.loras]
        except Exception:
            pass

    try:
        config.save_config(data)
    except Exception as e:
        logger.error(f"[API] Save modes failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # If the currently loaded mode was edited, reload the worker
    reload_queued = False
    if current_mode and old_model is not None:
        new_mode_data = data["modes"].get(current_mode)
        if new_mode_data is None:
            # Current mode was deleted in this bulk save — switch to default
            default = data["default_mode"]
            logger.info(f"[API] Current mode '{current_mode}' removed; switching to default '{default}'")
            try:
                pool.switch_mode(default)
                reload_queued = True
            except Exception as e:
                logger.warning(f"[API] Could not queue switch to default after mode removal: {e}")
        else:
            new_model = new_mode_data.get("model", "")
            new_loras = [
                (lora["path"] if isinstance(lora, dict) else lora)
                for lora in new_mode_data.get("loras", [])
            ]
            if new_model != old_model or new_loras != old_loras:
                reload_queued = pool.reload_if_current(current_mode)

    return {
        "status": "saved",
        "modes": list(data["modes"].keys()),
        "reload_queued": reload_queued,
    }


@router.post("/modes/{name}")
async def create_or_update_mode(name: str, request: ModeCreateRequest):
    """Create or update a single mode."""
    config = get_mode_config()
    pool = get_worker_pool()
    data = config.to_dict()
    data["modes"][name] = request.model_dump()

    try:
        config.save_config(data)
    except Exception as e:
        logger.error(f"[API] Save mode '{name}' failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # If this mode is currently loaded, reload the worker with the new config
    reload_queued = pool.reload_if_current(name)

    return {"status": "saved", "mode": name, "reload_queued": reload_queued}


@router.delete("/modes/{name}")
async def delete_mode(name: str):
    """Delete a mode. Cannot delete the default mode."""
    config = get_mode_config()
    pool = get_worker_pool()
    data = config.to_dict()

    if name not in data["modes"]:
        raise HTTPException(status_code=404, detail=f"Mode '{name}' not found")

    if name == data["default_mode"]:
        raise HTTPException(status_code=400, detail="Cannot delete the default mode")

    was_loaded = name == pool.get_current_mode()
    del data["modes"][name]

    try:
        config.save_config(data)
    except Exception as e:
        logger.error(f"[API] Delete mode '{name}' failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # If the deleted mode was running, switch to the default
    switched_to = None
    if was_loaded:
        default = data["default_mode"]
        logger.info(f"[API] Deleted loaded mode '{name}'; switching to default '{default}'")
        try:
            pool.switch_mode(default)
            switched_to = default
        except Exception as e:
            logger.warning(f"[API] Could not queue switch to default after delete: {e}")

    return {"status": "deleted", "mode": name, "switched_to": switched_to}
