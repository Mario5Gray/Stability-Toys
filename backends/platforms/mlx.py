from __future__ import annotations

from backends.model_registry import PlaceholderModelRegistry
from backends.platforms.base import BackendCapabilities
from backends.platforms.cpu import PlaceholderGenerationRuntime, PlaceholderSuperResRuntime


class MLXProvider:
    backend_id = "mlx"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(False, True, False, False, False)

    def create_worker_factory(self, *args, **kwargs):
        raise NotImplementedError("BACKEND=mlx worker factory is not implemented")

    def create_model_registry(self):
        return PlaceholderModelRegistry(self.backend_id)

    def create_generation_runtime(self, *args, **kwargs):
        return PlaceholderGenerationRuntime(self.backend_id)

    def create_superres_runtime(self, *args, **kwargs):
        return PlaceholderSuperResRuntime(self.backend_id)
