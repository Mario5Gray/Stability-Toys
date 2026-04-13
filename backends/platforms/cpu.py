from __future__ import annotations

from typing import Any

from backends.model_registry import PlaceholderModelRegistry
from backends.platforms.base import BackendCapabilities


class PlaceholderGenerationRuntime:
    def __init__(self, backend_id: str):
        self._backend_id = backend_id

    def submit_generate(self, req: Any, *, timeout_s: float = 0.25):
        raise NotImplementedError(f"BACKEND={self._backend_id} generation is not implemented")

    def get_current_mode(self):
        return None

    def is_model_loaded(self) -> bool:
        return False

    def get_queue_size(self) -> int:
        return 0

    def shutdown(self) -> None:
        return None


class PlaceholderSuperResRuntime:
    def __init__(self, backend_id: str):
        self._backend_id = backend_id

    def submit(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(f"BACKEND={self._backend_id} super-resolution is not implemented")

    def unload(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class CPUProvider:
    backend_id = "cpu"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(False, True, False, False, False)

    def create_worker_factory(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("BACKEND=cpu worker factory is not implemented")

    def create_model_registry(self):
        return PlaceholderModelRegistry(self.backend_id)

    def create_generation_runtime(self, *args: Any, **kwargs: Any):
        return PlaceholderGenerationRuntime(self.backend_id)

    def create_superres_runtime(self, *args: Any, **kwargs: Any):
        return PlaceholderSuperResRuntime(self.backend_id)
