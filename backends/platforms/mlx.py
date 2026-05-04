from __future__ import annotations

from typing import Any

from backends.model_registry import PlaceholderModelRegistry
from backends.platforms.base import BackendCapabilities, GenerationRuntimeProtocol, ModelRegistryProtocol
from backends.platforms.cpu import PlaceholderGenerationRuntime, PlaceholderSuperResRuntime


class MLXProvider:
    backend_id: str = "mlx"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(False, True, False, False, False)

    def create_worker_factory(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("BACKEND=mlx worker factory is not implemented")

    def create_model_registry(self) -> ModelRegistryProtocol:
        return PlaceholderModelRegistry(self.backend_id)

    def create_generation_runtime(self, *args: Any, **kwargs: Any) -> GenerationRuntimeProtocol:
        return PlaceholderGenerationRuntime(self.backend_id)

    def create_superres_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return PlaceholderSuperResRuntime(self.backend_id)
