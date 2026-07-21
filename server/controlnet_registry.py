from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import os

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


def default_controlnet_registry_path() -> str:
    config_root = os.environ.get("MODE_CONFIG_PATH", "conf")
    return str(Path(config_root) / "controlnets.yaml")


def load_controlnet_registry(*, config_path: Optional[str] = None, validation_mode: str = "strict") -> ControlNetRegistry:
    if config_path is None:
        config_path = default_controlnet_registry_path()
    from backends.family_profiles import validate_family_id

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
        # Every compatible_with entry must name a known neutral family, so a typo
        # fails at load rather than silently never matching at admission.
        for family_id in spec.compatible_with:
            validate_family_id(family_id)
        if validation_mode == "strict":
            _validate_local_path(spec)
        specs[model_id] = spec
    return ControlNetRegistry(specs=specs, validation_mode=validation_mode)


_registry_singleton: Optional[ControlNetRegistry] = None


def get_controlnet_registry() -> ControlNetRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = load_controlnet_registry(
            config_path=os.environ.get("CONTROLNET_REGISTRY_PATH") or default_controlnet_registry_path(),
            validation_mode=os.environ.get("CONTROLNET_REGISTRY_VALIDATION", "strict").strip().lower(),
        )
    return _registry_singleton


def validate_controlnet_mode_references(*, mode_config=None, registry: Optional[ControlNetRegistry] = None) -> None:
    from backends.family_profiles import FamilyResolutionError, resolve_family
    from server.mode_config import get_mode_config
    from utils.model_detector import detect_model

    if registry is None:
        registry = get_controlnet_registry()
    if mode_config is None:
        mode_config = get_mode_config()

    for mode_name in mode_config.list_modes():
        mode = mode_config.get_mode(mode_name)
        policy = getattr(mode, "controlnet_policy", None)
        if policy is None or not policy.enabled:
            continue

        if mode.model_path is None:
            raise ValueError(f"Mode '{mode.name}' does not have a resolved model_path")

        # Family is resolved from detector facts, never from checkpoint_variant
        # (mode policy can overlay checkpoint_variant, so it is not authoritative).
        try:
            active_family = resolve_family(detect_model(mode.model_path)).family_id
        except FamilyResolutionError as e:
            raise ValueError(
                f"Mode '{mode.name}' model does not resolve to a supported ControlNet family"
            ) from e

        for control_type, type_policy in policy.allowed_control_types.items():
            if (
                type_policy.default_model_id is not None
                and type_policy.allowed_model_ids
                and type_policy.default_model_id not in type_policy.allowed_model_ids
            ):
                raise ValueError(
                    f"Mode '{mode.name}' control_type '{control_type}' default_model_id "
                    f"'{type_policy.default_model_id}' is not present in allowed_model_ids"
                )

            referenced_model_ids = list(type_policy.allowed_model_ids)
            if (
                type_policy.default_model_id is not None
                and type_policy.default_model_id not in referenced_model_ids
            ):
                referenced_model_ids.append(type_policy.default_model_id)

            for model_id in referenced_model_ids:
                spec = registry.get_required(model_id)
                if control_type not in spec.control_types:
                    raise ValueError(
                        f"Mode '{mode.name}' control_type '{control_type}' references model_id "
                        f"'{model_id}' which does not support that control_type"
                    )
                if active_family not in spec.compatible_with:
                    raise ValueError(
                        f"Mode '{mode.name}' control_type '{control_type}' references model_id "
                        f"'{model_id}' which is incompatible with mode family '{active_family}'"
                    )


def reset_controlnet_registry() -> None:
    global _registry_singleton
    _registry_singleton = None
