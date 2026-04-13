"""
Model registry implementations.

CUDA uses a VRAM-aware registry. Other backends can expose a lightweight
placeholder registry that reports backend identity without pretending to have
CUDA allocator metrics.
"""

import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from threading import Lock

try:
    import torch
except ImportError:  # pragma: no cover - exercised on non-torch environments
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Information about a loaded model."""
    name: str  # Mode name or model identifier
    model_path: str
    vram_bytes: int
    worker_id: Optional[int] = None
    loras: List[str] = field(default_factory=list)  # List of loaded LoRA paths


class PlaceholderModelRegistry:
    """Backend-neutral registry for runtimes without CUDA VRAM accounting."""

    def __init__(self, backend_id: str):
        self._backend_id = backend_id
        self._loaded: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def register_model(
        self,
        name: str,
        model_path: str,
        vram_bytes: int = 0,
        worker_id: Optional[int] = None,
        loras: Optional[List[str]] = None,
    ):
        with self._lock:
            self._loaded[name] = {
                "name": name,
                "model_path": model_path,
                "vram_bytes": int(vram_bytes),
                "worker_id": worker_id,
                "loras": list(loras or []),
            }

    def unregister_model(self, name: str):
        with self._lock:
            self._loaded.pop(name, None)

    def list_models(self) -> List[str]:
        with self._lock:
            return sorted(self._loaded.keys())

    def get_total_vram(self) -> int:
        return 0

    def get_vram_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": self._backend_id,
                "device": f"{self._backend_id.upper()} placeholder",
                "models_loaded": len(self._loaded),
                "models": list(self._loaded.values()),
            }

    def clear(self):
        with self._lock:
            self._loaded.clear()


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

        if torch is None:
            self._total_vram = 0
            self._device_name = "CUDA unavailable"
            logger.warning("[ModelRegistry] torch is not installed")
            return

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

    def get_reserved_vram(self) -> int:
        """
        Get allocator-reserved VRAM in bytes for this process.

        This includes cached blocks held by the PyTorch allocator, not just
        live tensors.
        """
        if torch is None or not torch.cuda.is_available():
            return 0

        return torch.cuda.memory_reserved(self._device_index)

    def get_used_vram(self) -> int:
        """
        Backward-compatible alias for allocator-reserved VRAM.

        Process-level reporting now uses allocator metrics only:
        - reserved: allocator-held memory
        - allocated: live tensor allocations
        """
        return self.get_reserved_vram()
    
    def get_allocated_vram(self) -> int:
        '''
        Get currently allocated VRAM by this app

        uses torch.cuda.memory_allocated                
        '''
        if torch is None or not torch.cuda.is_available():
            return 0

        # Get actual allocated memory
        used_vram = torch.cuda.memory_allocated(self._device_index)
        return used_vram
        
    def get_total_vram(self) -> int:
        """Get total GPU VRAM in bytes."""
        return self._total_vram

    def get_available_vram(self) -> int:
        """Get available VRAM in bytes."""
        if torch is None or not torch.cuda.is_available():
            return 0

        reserved = self.get_reserved_vram()
        return self._total_vram - reserved

    def can_fit(self, estimated_bytes: int) -> bool:
        """
        Check if estimated model size can fit in available VRAM.

        Args:
            estimated_bytes: Estimated model size in bytes

        Returns:
            True if model can fit
        """
        if torch is None or not torch.cuda.is_available():
            return False

        available = self.get_available_vram()
        can_load = estimated_bytes < available + (available*.05)
        
        logger.info(
            "[ModelRegistry] VRAM check: need %.2f GB, available %.2f GB, fits: %s",
            estimated_bytes / 1024**3,
            available / 1024**3,
            can_load,
        )
        
        return can_load

    def estimate_model_vram(self, model_path: str) -> int:
        """
        Estimate VRAM requirement for a model.

        Uses file size with a dtype-aware multiplier:
          fp8  (float8_e4m3fn / float8_e5m2) → 1.1  (file IS the compressed size)
          fp16 / bf16                         → 1.2  (standard overhead)
          fp32                                → 0.6  (loaded as fp16, halved)
          fallback / non-safetensors          → 1.2

        Actual usage is measured after loading via torch.cuda.memory_allocated().

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
        multiplier = self._safetensors_vram_multiplier(model_path)
        estimated = int(file_size * multiplier)

        logger.debug(
            f"[ModelRegistry] Estimated VRAM for {model_path}: "
            f"{estimated / 1024**3:.2f} GB (multiplier={multiplier})"
        )

        return estimated

    def _safetensors_vram_multiplier(self, model_path: str) -> float:
        """Return a dtype-aware file-size multiplier for safetensors files.

        Reads only the safetensors header (no tensor data loaded).
        Falls back to 1.2 on any error or for non-safetensors files.
        """
        if not model_path.endswith(".safetensors"):
            return 1.2

        try:
            import json
            import struct

            with open(model_path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))

            sample_dtypes = {
                v["dtype"].upper()
                for k, v in header.items()
                if k != "__metadata__" and isinstance(v, dict) and "dtype" in v
            }

            if sample_dtypes & {"F8_E4M3", "F8_E5M2"}:
                return 1.1
            if sample_dtypes & {"F32"}:
                return 0.6
            return 1.2
        except Exception:
            return 1.2

    def get_vram_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive VRAM statistics.

        Returns:
            Dictionary with VRAM stats
        """
        reserved = self.get_reserved_vram()
        total = self.get_total_vram()
        available = self.get_available_vram()
        allocated = self.get_allocated_vram()

        # Get breakdown by loaded models
        models_breakdown = []

        to_gb = lambda x: x / (1024**3)

        with self._lock:
            for name, model in self._loaded.items():
                models_breakdown.append({
                    "name": name,
                    "model_path": model.model_path,
                    "vram_gb": to_gb(model.vram_bytes),
                    "loras": model.loras,
                })

        return {
            "device": self._device_name,
            "total_gb": to_gb(total),
            "allocated_gb": to_gb(allocated),
            "reserved_gb": to_gb(reserved),
            "used_gb": to_gb(reserved),
            "available_gb": to_gb(available),
            "usage_percent": round((reserved / total * 100) if total > 0 else 0, 1),
            "models_loaded": len(self._loaded),
            "models": models_breakdown,
        }

    def clear(self):
        """Clear all registered models (does not unload, just clears registry)."""
        with self._lock:
            self._loaded.clear()
            logger.info("[ModelRegistry] Cleared all registrations")

    def list_models(self) -> List[str]:
        with self._lock:
            return sorted(self._loaded.keys())


# Global registry instance
_registry: Optional[object] = None


def get_model_registry():
    """Get the singleton registry for the active backend."""
    global _registry
    if _registry is None:
        from backends.platform_registry import get_backend_provider

        _registry = get_backend_provider().create_model_registry()
    return _registry


def reset_model_registry() -> None:
    global _registry
    _registry = None
