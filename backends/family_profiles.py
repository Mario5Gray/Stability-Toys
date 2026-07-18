"""Neutral family registry, profiles, and the exact-one resolver.

Import-clean by contract: this module must not import Torch, Diffusers, CUDA
workers, or server state. It carries only pure, comparable, wire-safe family
data plus registry-local selection predicates. ``FamilyProfile`` values may be
embedded verbatim in request traces and consumed by a remote processor without
re-detection, so no callable or host-local authority is ever stored on them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

from utils.model_detector import ModelInfo


@dataclass(frozen=True)
class FamilyProfile:
    """Pure, comparable, wire-safe family data. No callables, ever."""

    family_id: str
    encoder_roles: tuple[str, ...]
    pooled_required: bool
    pooled_projection_role: str | None
    control_image_kwarg: str


@dataclass(frozen=True)
class FamilyRegistration:
    """A profile plus its registry-local selection predicate.

    ``detect`` is never serialized; selection behavior stays out of the wire
    form by living here rather than on :class:`FamilyProfile`.
    """

    profile: FamilyProfile
    detect: Callable[[ModelInfo], bool]


class FamilyResolutionError(Exception):
    """Raised when raw detector facts match zero or multiple neutral families."""

    def __init__(self, path: str, matches: tuple[FamilyRegistration, ...]) -> None:
        self.path = path
        self.matches = matches
        matched_ids = ", ".join(m.profile.family_id for m in matches)
        super().__init__(
            f"expected exactly one neutral family for {path!r}; "
            f"matched {len(matches)}: [{matched_ids}]"
        )


class UnknownFamilyError(Exception):
    """Raised when a family id is absent from the registry."""

    def __init__(self, family_id: str) -> None:
        self.family_id = family_id
        super().__init__(f"unknown neutral family id: {family_id!r}")


# --- Canonical profiles ------------------------------------------------------
# `sd15` is an execution family fronting SD1.5, SD2.0, and SD2.1. `sdxl` fronts
# SDXL Base and Refiner. Lineage remains in ModelInfo.variant, never here.
# Task 9 adds HUNYUANDIT_PROFILE after the Phase 2 gate.

SD15_PROFILE = FamilyProfile(
    family_id="sd15",
    encoder_roles=("text_encoder",),
    pooled_required=False,
    pooled_projection_role=None,
    control_image_kwarg="image",
)

SDXL_PROFILE = FamilyProfile(
    family_id="sdxl",
    encoder_roles=("text_encoder", "text_encoder_2"),
    pooled_required=True,
    pooled_projection_role="text_encoder_2",
    control_image_kwarg="image",
)


# --- Predicates --------------------------------------------------------------
# Predicates read only detector-owned architecture facts. `checkpoint_variant`
# is forbidden here because mode policy can overlay it.


def _is_sd15(info: ModelInfo) -> bool:
    return info.base_arch == "unet" and info.cross_attention_dim in (768, 1024)


def _is_sdxl(info: ModelInfo) -> bool:
    return info.base_arch == "unet" and info.cross_attention_dim in (1280, 2048)


FAMILY_REGISTRY: tuple[FamilyRegistration, ...] = (
    FamilyRegistration(SD15_PROFILE, _is_sd15),
    FamilyRegistration(SDXL_PROFILE, _is_sdxl),
)


# --- Resolution and validation ----------------------------------------------

_JSON_SAFE = (str, int, float, bool, type(None))


def _assert_profile_is_pure(profile: FamilyProfile) -> None:
    for spec in fields(profile):
        value = getattr(profile, spec.name)
        items = value if isinstance(value, tuple) else (value,)
        for item in items:
            if not isinstance(item, _JSON_SAFE):
                raise TypeError(
                    f"family profile {profile.family_id!r} field {spec.name!r} "
                    f"holds non-wire-safe value {item!r}"
                )


def _validate_registry(registry: tuple[FamilyRegistration, ...]) -> None:
    seen: set[str] = set()
    for registration in registry:
        profile = registration.profile
        _assert_profile_is_pure(profile)
        if profile.family_id in seen:
            raise ValueError(f"duplicate family id: {profile.family_id!r}")
        seen.add(profile.family_id)
        if not profile.encoder_roles:
            raise ValueError(f"family {profile.family_id!r} has empty encoder roles")
        role = profile.pooled_projection_role
        if role is not None:
            if not profile.pooled_required:
                raise ValueError(
                    f"family {profile.family_id!r} sets a projection role "
                    "without requiring a pooled embedding"
                )
            if role not in profile.encoder_roles:
                raise ValueError(
                    f"family {profile.family_id!r} projection role {role!r} "
                    "is absent from its encoder roles"
                )


_validate_registry(FAMILY_REGISTRY)


def family_ids() -> tuple[str, ...]:
    return tuple(registration.profile.family_id for registration in FAMILY_REGISTRY)


def resolve_family(
    model_info: ModelInfo,
    registry: tuple[FamilyRegistration, ...] = FAMILY_REGISTRY,
) -> FamilyProfile:
    """Return the single family whose predicate matches ``model_info``.

    Requires exactly one match. Registry order only stabilizes error text.
    """

    matches = tuple(r for r in registry if r.detect(model_info))
    if len(matches) != 1:
        raise FamilyResolutionError(model_info.path, matches)
    return matches[0].profile


def validate_family_id(family_id: str) -> str:
    if family_id not in family_ids():
        raise UnknownFamilyError(family_id)
    return family_id
