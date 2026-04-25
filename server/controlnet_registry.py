from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml


@dataclass(frozen=True)
class ControlNetModelSpec:
    model_id: str
    path: str
    control_types: tuple[str, ...]
    compatible_with: tuple[str, ...]


class ControlNetRegistry:
    def __init__(self, specs: Dict[str, ControlNetModelSpec], validation_mode: str) -> None:
        self._specs = specs
        self.validation_mode = validation_mode

    def get(self, model_id: str) -> Optional[ControlNetModelSpec]:
        return self._specs.get(model_id)

    def get_required(self, model_id: str) -> ControlNetModelSpec:
        spec = self.get(model_id)
        if spec is None:
            raise ValueError(f"unknown ControlNet model_id '{model_id}'")
        if self.validation_mode == "lazy":
            _validate_local_path(spec)
        return spec


def _validate_local_path(spec: ControlNetModelSpec) -> None:
    if not Path(spec.path).exists():
        raise ValueError(f"ControlNet model path does not exist: {spec.path}")


def load_controlnet_registry(*, config_path: str = "conf/controlnets.yaml", validation_mode: str = "strict") -> ControlNetRegistry:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    models = raw.get("models") or {}
    specs: Dict[str, ControlNetModelSpec] = {}
    for model_id, data in models.items():
        spec = ControlNetModelSpec(
            model_id=model_id,
            path=str(data["path"]),
            control_types=tuple(data["control_types"]),
            compatible_with=tuple(data["compatible_with"]),
        )
        if validation_mode == "strict":
            _validate_local_path(spec)
        specs[model_id] = spec
    return ControlNetRegistry(specs=specs, validation_mode=validation_mode)


_registry_singleton: Optional[ControlNetRegistry] = None


def get_controlnet_registry() -> ControlNetRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        import os

        _registry_singleton = load_controlnet_registry(
            config_path=os.environ.get("CONTROLNET_REGISTRY_PATH", "conf/controlnets.yaml"),
            validation_mode=os.environ.get("CONTROLNET_REGISTRY_VALIDATION", "strict").strip().lower(),
        )
    return _registry_singleton


def reset_controlnet_registry() -> None:
    global _registry_singleton
    _registry_singleton = None