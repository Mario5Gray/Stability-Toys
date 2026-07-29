"""
Model registry implementations.

CUDA uses a VRAM-aware registry. Other backends can expose a lightweight
placeholder registry that reports backend identity without pretending to have
CUDA allocator metrics.

ModelRegistry is a pure VIEW over DeviceMemory (STABL-hjldxurg): it holds no
torch and no pynvml. Driver truth comes from cached_snapshot()/
available_for_load(); per-consumer pool stats come from the worker's entry in
cached_snapshot().consumers[]. The view NEVER calls fresh snapshot() — no
fan-out from the registry (spec §4.1).
"""

import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from threading import Lock

from backends.platforms.base import ModelRegistryProtocol

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
    ) -> None:
        with self._lock:
            self._loaded[name] = {
                "name": name,
                "model_path": model_path,
                "vram_bytes": int(vram_bytes),
                "worker_id": worker_id,
                "loras": list(loras or []),
            }

    def unregister_model(self, name: str) -> None:
        with self._lock:
            self._loaded.pop(name, None)

    def list_models(self) -> List[str]:
        with self._lock:
            return sorted(self._loaded.keys())

    def get_total_vram(self) -> int:
        return 0

    def get_used_vram(self) -> int:
        return 0

    def get_allocated_vram(self) -> int:
        return 0

    def get_vram_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": self._backend_id,
                "device": f"{self._backend_id.upper()} placeholder",
                "models_loaded": len(self._loaded),
                "models": list(self._loaded.values()),
            }

    def clear(self) -> None:
        with self._lock:
            self._loaded.clear()


class ModelRegistry:
    """
    Tracks loaded models and VRAM usage.

    Thread-safe registry for managing model lifecycle. VRAM accounting is a
    pure view over DeviceMemory — no torch, no direct driver reads.
    """

    def __init__(self, device_memory=None) -> None:
        """Initialize model registry as a pure view over DeviceMemory."""
        self._loaded: Dict[str, LoadedModel] = {}
        self._lock = Lock()
        if device_memory is None:
            from backends.device_memory import get_device_memory
            device_memory = get_device_memory()
        self._dm = device_memory
        logger.info(
            "[ModelRegistry] Device: %s (%.2f GB)",
            self._dm.device_name,
            self._dm.cached_snapshot().total_bytes / 1024**3,
        )

    def register_model(
        self,
        name: str,
        model_path: str,
        vram_bytes: int = 0,
        worker_id: Optional[int] = None,
        loras: Optional[List[str]] = None,
    ) -> None:
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

    def unregister_model(self, name: str) -> None:
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

    def _worker_entry(self):
        """The worker consumer's entry in the last computed snapshot (cached —
        NO fan-out from this view; /status and diagnostics refresh the cache)."""
        return next(
            (c for c in self._dm.cached_snapshot().consumers if c.label == "worker"),
            None,
        )

    def get_reserved_vram(self) -> int:
        """Allocator-reserved VRAM of the worker consumer (last snapshot).

        Includes cached blocks held by the PyTorch allocator, not just live
        tensors.
        """
        entry = self._worker_entry()
        return entry.reserved_bytes if entry else 0

    def get_used_vram(self) -> int:
        """Backward-compatible alias for allocator-reserved VRAM."""
        return self.get_reserved_vram()

    def get_allocated_vram(self) -> int:
        """Live torch allocations of the worker consumer (last snapshot)."""
        entry = self._worker_entry()
        return entry.allocated_bytes if entry else 0

    def get_total_vram(self) -> int:
        """Total device VRAM in bytes (last snapshot; correct from startup
        because the provider pre-seeds its cache)."""
        return self._dm.cached_snapshot().total_bytes

    def get_available_vram(self) -> int:
        """Driver-truth free VRAM in bytes — NVML free via DeviceMemory.

        Reports the driver's free memory, accounting for the CUDA context,
        cuDNN/cuBLAS/xformers workspaces, and every other process on the GPU.
        Never fans out to consumers (spec §2.2 invariant #1).
        """
        return self._dm.available_for_load()

    def can_fit(self, estimated_bytes: int) -> bool:
        """
        Check if estimated model size can fit in available VRAM.

        Args:
            estimated_bytes: Estimated model size in bytes

        Returns:
            True if model can fit
        """
        # No CUDA guard: the Null provider reports 0 free, which rejects the
        # same way the old torch.cuda.is_available() guard did.
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
        # Device-used is the driver truth (total - driver_free), which includes the
        # CUDA context, library workspaces, and other processes — not just this
        # process's torch reserved pool. reserved_gb/allocated_gb below stay as the
        # torch-specific numbers.
        used = max(0, total - available)

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
            "device": self._dm.device_name,
            "total_gb": to_gb(total),
            "allocated_gb": to_gb(allocated),
            "reserved_gb": to_gb(reserved),
            "used_gb": to_gb(used),
            "available_gb": to_gb(available),
            "usage_percent": round((used / total * 100) if total > 0 else 0, 1),
            "models_loaded": len(self._loaded),
            "models": models_breakdown,
        }

    def clear(self) -> None:
        """Clear all registered models (does not unload, just clears registry)."""
        with self._lock:
            self._loaded.clear()
            logger.info("[ModelRegistry] Cleared all registrations")

    def list_models(self) -> List[str]:
        with self._lock:
            return sorted(self._loaded.keys())


# Global registry instance
_registry: Optional[ModelRegistryProtocol] = None


def get_model_registry() -> ModelRegistryProtocol:
    """Get the singleton registry for the active backend."""
    global _registry
    if _registry is None:
        from backends.platform_registry import get_backend_provider

        _registry = get_backend_provider().create_model_registry()
    assert _registry is not None
    return _registry


def reset_model_registry() -> None:
    global _registry
    _registry = None
