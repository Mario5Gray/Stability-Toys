from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    supports_generation: bool
    supports_modes: bool
    supports_superres: bool
    supports_model_registry_stats: bool
    supports_img2img: bool


class ModelRegistryProtocol(Protocol):
    def register_model(
        self,
        name: str,
        model_path: str,
        vram_bytes: int = 0,
        worker_id: Optional[int] = None,
        loras: Optional[list[str]] = None,
    ) -> None:
        ...

    def unregister_model(self, name: str) -> None:
        ...

    def list_models(self) -> list[str]:
        ...

    def get_vram_stats(self) -> dict[str, Any]:
        ...

    def get_total_vram(self) -> int:
        ...


class GenerationRuntimeProtocol(Protocol):
    def submit_generate(self, req: Any, *, timeout_s: float = 0.25) -> Any:
        ...

    def get_current_mode(self) -> Optional[str]:
        ...

    def is_model_loaded(self) -> bool:
        ...

    def get_queue_size(self) -> int:
        ...

    def shutdown(self) -> None:
        ...


class BackendProvider(Protocol):
    backend_id: str

    def capabilities(self) -> BackendCapabilities:
        ...

    def create_worker_factory(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def create_model_registry(self) -> ModelRegistryProtocol:
        ...

    def create_generation_runtime(self, *args: Any, **kwargs: Any) -> GenerationRuntimeProtocol:
        ...

    def create_superres_runtime(self, *args: Any, **kwargs: Any) -> Any:
        ...
