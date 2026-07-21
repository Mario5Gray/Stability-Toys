from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


class UnsupportedFamilyError(Exception):
    """A known neutral family has no binding on the selected platform."""


@dataclass(frozen=True)
class BackendCapabilities:
    """Platform-wide services only. Per-family execution claims (img2img,
    ControlNet, combined) live on FamilyPlatformBinding, not here."""

    supports_generation: bool
    supports_modes: bool
    supports_superres: bool
    supports_model_registry_stats: bool


@dataclass(frozen=True)
class ExecutionCapabilities:
    supports_img2img: bool
    supports_controlnet: bool
    supports_img2img_and_controlnet: bool


@dataclass(frozen=True)
class FamilyPlatformBinding:
    worker_ref: str
    execution_capabilities: ExecutionCapabilities


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

    def get_used_vram(self) -> int:
        ...

    def get_allocated_vram(self) -> int:
        ...


class GenerationRuntimeProtocol(Protocol):
    def submit_generate(self, req: Any, *, timeout_s: float | None = None) -> Any:
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

    def family_binding(self, family_id: str) -> Optional[FamilyPlatformBinding]:
        ...

    def create_worker_factory(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def create_model_registry(self) -> ModelRegistryProtocol:
        ...

    def create_generation_runtime(self, *args: Any, **kwargs: Any) -> GenerationRuntimeProtocol:
        ...

    def create_superres_runtime(self, *args: Any, **kwargs: Any) -> Any:
        ...
