from __future__ import annotations

import os

from backends.platforms.base import (
    BackendCapabilities,
    BackendProvider,
)

_VALID_BACKENDS = ("cuda", "rknn", "mlx", "cpu")

_provider: BackendProvider | None = None


def _read_backend() -> str:
    backend = (os.environ.get("BACKEND") or "").strip().lower()
    allowed = ", ".join(_VALID_BACKENDS)
    if not backend:
        raise RuntimeError(f"BACKEND must be set explicitly to one of: {allowed}")
    if backend not in _VALID_BACKENDS:
        raise RuntimeError(f"Unsupported BACKEND='{backend}'; supported values: {allowed}")
    return backend


def get_backend_provider() -> BackendProvider:
    global _provider
    if _provider is None:
        backend = _read_backend()
        if backend == "cuda":
            from backends.platforms.cuda import CUDAProvider

            _provider = CUDAProvider()
        elif backend == "rknn":
            from backends.platforms.rknn import RKNNProvider

            _provider = RKNNProvider()
        elif backend == "cpu":
            from backends.platforms.cpu import CPUProvider

            _provider = CPUProvider()
        elif backend == "mlx":
            from backends.platforms.mlx import MLXProvider

            _provider = MLXProvider()
        else:
            raise RuntimeError(f"Unsupported BACKEND='{backend}'")
    return _provider


def reset_backend_provider() -> None:
    global _provider
    _provider = None
