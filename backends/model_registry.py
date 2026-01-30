"""
Model registry with VRAM tracking.

Tracks loaded models and their actual VRAM usage.
No artificial limits - uses all available VRAM.
"""

import logging
import torch
from typing import Dict, Optional, List
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Information about a loaded model."""
    name: str  # Mode name or model identifier
    model_path: str
    vram_bytes: int
    worker_id: Optional[int] = None
    loras: List[str] = None  # List of loaded LoRA paths

    def __post_init__(self):
        if self.loras is None:
            self.loras = []


class ModelRegistry:
    """
    Tracks loaded models and VRAM usage.

    Thread-safe registry for managing model lifecycle and VRAM accounting.
    Uses actual torch.cuda measurements - no artificial limits.
    """

    def __init__(self):
        """Initialize model registry."""
        self._loaded: Dict[str, LoadedModel] = {}
        self._lock = Lock()
        self._device_index = 0  # Default CUDA device

        # Detect GPU
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(self._device_index)
            self._total_vram = device_props.total_memory
            self._device_name = device_props.name
            logger.info(
                f"[ModelRegistry] GPU detected: {self._device_name} "
                f"({self._total_vram / 1024**3:.2f} GB VRAM)"
            )
        else:
            self._total_vram = 0
            self._device_name = "No GPU"
            logger.warning("[ModelRegistry] No CUDA GPU detected")

    def register_model(
        self,
        name: str,
        model_path: str,
        vram_bytes: int,
        worker_id: Optional[int] = None,
        loras: Optional[List[str]] = None,
    ):
        """
        Register a loaded model.

        Args:
            name: Model identifier (typically mode name)
            model_path: Path to model file
            vram_bytes: Actual VRAM used by model
            worker_id: Worker ID if applicable
            loras: List of loaded LoRA paths
        """
        with self._lock:
            model = LoadedModel(
                name=name,
                model_path=model_path,
                vram_bytes=vram_bytes,
                worker_id=worker_id,
                loras=loras or [],
            )
            self._loaded[name] = model
            logger.info(
                f"[ModelRegistry] Registered model '{name}': "
                f"{vram_bytes / 1024**3:.2f} GB VRAM"
            )

    def unregister_model(self, name: str):
        """
        Unregister a model.

        Args:
            name: Model identifier
        """
        with self._lock:
            if name in self._loaded:
                model = self._loaded.pop(name)
                logger.info(
                    f"[ModelRegistry] Unregistered model '{name}': "
                    f"freed {model.vram_bytes / 1024**3:.2f} GB VRAM"
                )
            else:
                logger.warning(f"[ModelRegistry] Model '{name}' not registered")

    def get_loaded_models(self) -> Dict[str, LoadedModel]:
        """Get all loaded models."""
        with self._lock:
            return dict(self._loaded)

    def get_model(self, name: str) -> Optional[LoadedModel]:
        """Get specific loaded model."""
        with self._lock:
            return self._loaded.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if model is loaded."""
        with self._lock:
            return name in self._loaded

    def get_used_vram(self) -> int:
        """
        Get currently used VRAM in bytes.

        Uses actual torch.cuda.memory_allocated() for accurate measurement.
        """
        if not torch.cuda.is_available():
            return 0

        # Get actual allocated memory
        allocated = torch.cuda.memory_allocated(self._device_index)
        return allocated

    def get_total_vram(self) -> int:
        """Get total GPU VRAM in bytes."""
        return self._total_vram

    def get_available_vram(self) -> int:
        """Get available VRAM in bytes."""
        if not torch.cuda.is_available():
            return 0

        used = self.get_used_vram()
        return self._total_vram - used

    def can_fit(self, estimated_bytes: int) -> bool:
        """
        Check if estimated model size can fit in available VRAM.

        Args:
            estimated_bytes: Estimated model size in bytes

        Returns:
            True if model can fit
        """
        if not torch.cuda.is_available():
            return False

        available = self.get_available_vram()
        can_load = estimated_bytes <= available

        logger.debug(
            f"[ModelRegistry] VRAM check: need {estimated_bytes / 1024**3:.2f} GB, "
            f"available {available / 1024**3:.2f} GB, "
            f"fits: {can_load}"
        )

        return can_load

    def estimate_model_vram(self, model_path: str) -> int:
        """
        Estimate VRAM requirement for a model.

        Uses file size as rough estimate. Actual usage will be measured
        after loading via torch.cuda.memory_allocated().

        Args:
            model_path: Path to model file

        Returns:
            Estimated VRAM in bytes
        """
        import os

        if not os.path.exists(model_path):
            logger.warning(f"[ModelRegistry] Model not found: {model_path}")
            return 0

        file_size = os.path.getsize(model_path)

        # Rough estimate: model file size + 20% overhead for inference
        estimated = int(file_size * 1.2)

        logger.debug(
            f"[ModelRegistry] Estimated VRAM for {model_path}: "
            f"{estimated / 1024**3:.2f} GB"
        )

        return estimated

    def get_vram_stats(self) -> Dict[str, any]:
        """
        Get comprehensive VRAM statistics.

        Returns:
            Dictionary with VRAM stats
        """
        used = self.get_used_vram()
        total = self.get_total_vram()
        available = self.get_available_vram()

        # Get breakdown by loaded models
        models_breakdown = []
        with self._lock:
            for name, model in self._loaded.items():
                models_breakdown.append({
                    "name": name,
                    "model_path": model.model_path,
                    "vram_gb": round(model.vram_bytes / 1024**3, 2),
                    "loras": model.loras,
                })

        return {
            "device": self._device_name,
            "total_gb": round(total / 1024**3, 2),
            "used_gb": round(used / 1024**3, 2),
            "available_gb": round(available / 1024**3, 2),
            "usage_percent": round((used / total * 100) if total > 0 else 0, 1),
            "models_loaded": len(self._loaded),
            "models": models_breakdown,
        }

    def clear(self):
        """Clear all registered models (does not unload, just clears registry)."""
        with self._lock:
            self._loaded.clear()
            logger.info("[ModelRegistry] Cleared all registrations")


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
