"""Portable model-resolution values and the single pre-overlay resolver.

This module emits a wire-safe ``ResolvedModel`` that may appear verbatim in a
request trace and be consumed by a future remote processor without re-detecting
or re-resolving family. It therefore carries no callables, no live ``ModelInfo``,
and no host-local authority: host paths live only in ``LocalModelBinding``,
which never serializes.

Codec discipline (per design §3):
- Named ``to_json_dict`` / ``from_json_dict`` per wire value. We never
  recursively dump ``__dict__``, call ``asdict()`` as an implicit contract, or
  reuse the lossy ``ModelInfo.to_dict()``.
- ``ModelInfo.path`` is stripped at freeze time; a host path would make the
  identical resolution hash differently on different nodes.
- ``resolution_id = sha256(JCS(payload))`` (RFC 8785) over schema, model_ref,
  raw_info, the *complete* embedded profile, and enriched info.
- Pickle is forbidden for this contract.
"""

from __future__ import annotations

import hashlib
import math
import os
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, fields
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import rfc8785

from server.mode_config import ModeConfig
from utils.model_detector import ModelInfo, ModelVariant, detect_model

from .family_profiles import FamilyProfile, resolve_family, validate_family_id

RESOLVED_MODEL_SCHEMA_VERSION = 1


class ResolutionCompatibilityError(Exception):
    """Raised when a traced ResolvedModel cannot be safely consumed/executed."""


# ---------------------------------------------------------------------------
# JSON-safety
# ---------------------------------------------------------------------------

def _assert_json_safe(value: Any, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return
    if isinstance(value, float):
        # RFC 8785 (and JSON) cannot represent NaN/Inf; reject at freeze time so
        # invalid detector data fails now, not later when canonical bytes build.
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite float {value!r} is not JSON-safe")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: non-string mapping key {key!r}")
            _assert_json_safe(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_safe(item, f"{where}[{index}]")
        return
    raise TypeError(f"{where}: value {value!r} of type {type(value).__name__} is not JSON-safe")


def _freeze_mapping(value: Mapping[str, Any], where: str) -> MappingProxyType:
    _assert_json_safe(value, where)
    return MappingProxyType({str(k): v for k, v in value.items()})


# ---------------------------------------------------------------------------
# ModelInfoSnapshot: frozen, JSON-safe capture of detector output (no path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfoSnapshot:
    variant: str
    cross_attention_dim: int | None
    text_encoder_hidden_size: int | None
    text_encoder_2_hidden_size: int | None
    unet_in_channels: int | None
    unet_out_channels: int | None
    vae_latent_channels: int | None
    format: str
    is_lora: bool
    confidence: float
    detected_by: tuple[str, ...]
    metadata: MappingProxyType
    loader_format: str
    checkpoint_precision: str
    checkpoint_variant: str
    scheduler_profile: str
    recommended_size: str | None
    runtime_quantize: str | None
    runtime_offload: str | None
    runtime_attention_slicing: bool | None
    runtime_enable_xformers: bool | None
    negative_prompt_templates: MappingProxyType
    default_negative_prompt_template: str | None
    allow_custom_negative_prompt: bool
    allowed_scheduler_ids: tuple[str, ...] | None
    default_scheduler_id: str | None
    compatible_worker: str | None
    required_cross_attention_dim: int | None
    base_arch: str
    transformer_kind: str | None


# Fields carried verbatim (scalar) between ModelInfo and the snapshot.
_SCALAR_SNAPSHOT_FIELDS = (
    "cross_attention_dim",
    "text_encoder_hidden_size",
    "text_encoder_2_hidden_size",
    "unet_in_channels",
    "unet_out_channels",
    "vae_latent_channels",
    "format",
    "is_lora",
    "confidence",
    "loader_format",
    "checkpoint_precision",
    "checkpoint_variant",
    "scheduler_profile",
    "recommended_size",
    "runtime_quantize",
    "runtime_offload",
    "runtime_attention_slicing",
    "runtime_enable_xformers",
    "default_negative_prompt_template",
    "allow_custom_negative_prompt",
    "default_scheduler_id",
    "compatible_worker",
    "required_cross_attention_dim",
    "base_arch",
    "transformer_kind",
)


def freeze_model_info(info: ModelInfo) -> ModelInfoSnapshot:
    """Capture detector output as immutable JSON-safe data, stripping ``path``.

    Non-JSON-safe metadata fails here — at resolution, not later at trace time.
    """

    scalars = {name: getattr(info, name) for name in _SCALAR_SNAPSHOT_FIELDS}
    allowed = info.allowed_scheduler_ids
    return ModelInfoSnapshot(
        variant=info.variant.value,
        detected_by=tuple(info.detected_by),
        metadata=_freeze_mapping(info.metadata, "metadata"),
        negative_prompt_templates=_freeze_mapping(
            info.negative_prompt_templates, "negative_prompt_templates"
        ),
        allowed_scheduler_ids=tuple(allowed) if allowed is not None else None,
        **scalars,
    )


def thaw_model_info(snapshot: ModelInfoSnapshot, binding: "LocalModelBinding") -> ModelInfo:
    """Rehydrate a live ModelInfo, restoring ``path`` from node-local authority.

    Deliberately partial: there is no ``snapshot -> ModelInfo`` inverse without a
    binding, because the snapshot carries no path by design.
    """

    return ModelInfo(
        path=binding.model_path,
        variant=ModelVariant(snapshot.variant),
        detected_by=list(snapshot.detected_by),
        metadata=dict(snapshot.metadata),
        negative_prompt_templates=dict(snapshot.negative_prompt_templates),
        allowed_scheduler_ids=(
            list(snapshot.allowed_scheduler_ids)
            if snapshot.allowed_scheduler_ids is not None
            else None
        ),
        **{name: getattr(snapshot, name) for name in _SCALAR_SNAPSHOT_FIELDS},
    )


def snapshot_to_json_dict(snapshot: ModelInfoSnapshot) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    for spec in fields(snapshot):
        value = getattr(snapshot, spec.name)
        if isinstance(value, MappingProxyType):
            wire[spec.name] = dict(value)
        elif isinstance(value, tuple):
            wire[spec.name] = list(value)
        else:
            wire[spec.name] = value
    return wire


def snapshot_from_json_dict(wire: Mapping[str, Any]) -> ModelInfoSnapshot:
    expected = {spec.name for spec in fields(ModelInfoSnapshot)}
    missing = expected - set(wire)
    if missing:
        raise ValueError(f"snapshot wire form missing keys: {sorted(missing)}")
    allowed = wire["allowed_scheduler_ids"]
    return ModelInfoSnapshot(
        variant=wire["variant"],
        detected_by=tuple(wire["detected_by"]),
        metadata=_freeze_mapping(wire["metadata"], "metadata"),
        negative_prompt_templates=_freeze_mapping(
            wire["negative_prompt_templates"], "negative_prompt_templates"
        ),
        allowed_scheduler_ids=tuple(allowed) if allowed is not None else None,
        **{name: wire[name] for name in _SCALAR_SNAPSHOT_FIELDS},
    )


# ---------------------------------------------------------------------------
# ModelArtifactRef: location-neutral identity + weak/strong tiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArtifactRef:
    kind: str  # "hub" | "local-file" | "local-dir"
    name: str
    revision: str | None
    fingerprint: str
    digest: str | None


@dataclass(frozen=True)
class LocalModelBinding:
    """Node-local load authority. NEVER serialized into traces."""

    model_path: str


def _weak_hash(payload: Any) -> str:
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def _local_dir_manifest(root: str) -> list[list[Any]]:
    manifest: list[list[Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                raise ValueError(
                    f"symlink in model directory has no portable identity: {full}"
                )
        for name in filenames:
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, root)
            rel_posix = unicodedata.normalize("NFC", rel.replace(os.sep, "/"))
            manifest.append([rel_posix, os.path.getsize(full)])
    manifest.sort(key=lambda entry: entry[0].encode("utf-8"))
    return manifest


def local_artifact_ref(model_path: str) -> ModelArtifactRef:
    """Build a location-neutral ref with a weak structural fingerprint."""

    if os.path.islink(model_path):
        raise ValueError(f"symlinked model path has no portable identity: {model_path}")
    name = os.path.basename(os.path.normpath(model_path))
    if os.path.isdir(model_path):
        fingerprint = _weak_hash(_local_dir_manifest(model_path))
        return ModelArtifactRef("local-dir", name, None, fingerprint, None)
    fingerprint = _weak_hash([name, os.path.getsize(model_path)])
    return ModelArtifactRef("local-file", name, None, fingerprint, None)


def hub_ref(repo_id: str, revision: str | None, digest: str | None = None) -> ModelArtifactRef:
    fingerprint = f"{repo_id}@{revision}"
    return ModelArtifactRef("hub", repo_id, revision, fingerprint, digest)


def model_ref_to_json_dict(ref: ModelArtifactRef) -> dict[str, Any]:
    return {
        "kind": ref.kind,
        "name": ref.name,
        "revision": ref.revision,
        "fingerprint": ref.fingerprint,
        "digest": ref.digest,
    }


def model_ref_from_json_dict(wire: Mapping[str, Any]) -> ModelArtifactRef:
    return ModelArtifactRef(
        kind=wire["kind"],
        name=wire["name"],
        revision=wire["revision"],
        fingerprint=wire["fingerprint"],
        digest=wire["digest"],
    )


_FULL_COMMIT_HASH_LEN = 40


def _is_full_commit_hash(revision: str | None) -> bool:
    return (
        isinstance(revision, str)
        and len(revision) == _FULL_COMMIT_HASH_LEN
        and all(c in "0123456789abcdef" for c in revision.lower())
    )


def has_strong_identity(ref: ModelArtifactRef) -> bool:
    if ref.digest is not None:
        return True
    return ref.kind == "hub" and _is_full_commit_hash(ref.revision)


# ---------------------------------------------------------------------------
# Profile wire form
# ---------------------------------------------------------------------------


def profile_to_json_dict(profile: FamilyProfile) -> dict[str, Any]:
    return {
        "family_id": profile.family_id,
        "encoder_roles": list(profile.encoder_roles),
        "pooled_required": profile.pooled_required,
        "pooled_projection_role": profile.pooled_projection_role,
        "control_image_kwarg": profile.control_image_kwarg,
    }


def profile_from_json_dict(wire: Mapping[str, Any]) -> FamilyProfile:
    validate_family_id(wire["family_id"])
    return FamilyProfile(
        family_id=wire["family_id"],
        encoder_roles=tuple(wire["encoder_roles"]),
        pooled_required=wire["pooled_required"],
        pooled_projection_role=wire["pooled_projection_role"],
        control_image_kwarg=wire["control_image_kwarg"],
    )


# ---------------------------------------------------------------------------
# ResolvedModel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedModel:
    schema_version: int
    resolution_id: str
    model_ref: ModelArtifactRef
    raw_info: ModelInfoSnapshot
    profile: FamilyProfile
    info: ModelInfoSnapshot


def _resolution_payload(
    *,
    model_ref: ModelArtifactRef,
    raw_info: ModelInfoSnapshot,
    profile: FamilyProfile,
    info: ModelInfoSnapshot,
) -> dict[str, Any]:
    return {
        "schema_version": RESOLVED_MODEL_SCHEMA_VERSION,
        "model_ref": model_ref_to_json_dict(model_ref),
        "raw_info": snapshot_to_json_dict(raw_info),
        "profile": profile_to_json_dict(profile),
        "info": snapshot_to_json_dict(info),
    }


def canonical_resolution_bytes(resolved: ResolvedModel) -> bytes:
    return rfc8785.dumps(
        _resolution_payload(
            model_ref=resolved.model_ref,
            raw_info=resolved.raw_info,
            profile=resolved.profile,
            info=resolved.info,
        )
    )


def build_resolved(
    *,
    model_ref: ModelArtifactRef,
    raw_info: ModelInfo,
    profile: FamilyProfile,
    info: ModelInfo,
) -> ResolvedModel:
    frozen_raw = freeze_model_info(raw_info)
    frozen_info = freeze_model_info(info)
    payload = _resolution_payload(
        model_ref=model_ref, raw_info=frozen_raw, profile=profile, info=frozen_info
    )
    resolution_id = hashlib.sha256(rfc8785.dumps(payload)).hexdigest()
    return ResolvedModel(
        schema_version=RESOLVED_MODEL_SCHEMA_VERSION,
        resolution_id=resolution_id,
        model_ref=model_ref,
        raw_info=frozen_raw,
        profile=profile,
        info=frozen_info,
    )


def resolved_model_to_json_dict(resolved: ResolvedModel) -> dict[str, Any]:
    return {
        "schema_version": resolved.schema_version,
        "resolution_id": resolved.resolution_id,
        "model_ref": model_ref_to_json_dict(resolved.model_ref),
        "raw_info": snapshot_to_json_dict(resolved.raw_info),
        "profile": profile_to_json_dict(resolved.profile),
        "info": snapshot_to_json_dict(resolved.info),
    }


def expected_resolution_id(resolved: ResolvedModel) -> str:
    return hashlib.sha256(canonical_resolution_bytes(resolved)).hexdigest()


def verify_resolution_id(resolved: ResolvedModel) -> None:
    """Recompute the content hash and reject a trace whose id has drifted.

    ``resolution_id`` is integrity-bearing portable identity, so a consumer must
    never trust the incoming value — a tampered or stale id is a compatibility
    failure, not silently canonical.
    """

    expected = expected_resolution_id(resolved)
    if resolved.resolution_id != expected:
        raise ResolutionCompatibilityError(
            "resolution_id integrity check failed: "
            f"traced {resolved.resolution_id!r} != recomputed {expected!r}"
        )


def resolved_model_from_json_dict(wire: Mapping[str, Any]) -> ResolvedModel:
    resolved = ResolvedModel(
        schema_version=wire["schema_version"],
        resolution_id=wire["resolution_id"],
        model_ref=model_ref_from_json_dict(wire["model_ref"]),
        raw_info=snapshot_from_json_dict(wire["raw_info"]),
        profile=profile_from_json_dict(wire["profile"]),
        info=snapshot_from_json_dict(wire["info"]),
    )
    verify_resolution_id(resolved)
    return resolved


# ---------------------------------------------------------------------------
# Trace validation + consumption
# ---------------------------------------------------------------------------


def validate_resolved_model_trace(
    resolved: ResolvedModel, *, for_execution: bool = False
) -> None:
    """Validate a traced ResolvedModel. Diagnostics are always allowed.

    ``for_execution=True`` requires strong identity (digest or immutable hub
    commit) and raises before any local binding/load attempt.
    """

    if resolved.schema_version != RESOLVED_MODEL_SCHEMA_VERSION:
        raise ResolutionCompatibilityError(
            f"unsupported resolved-model schema version {resolved.schema_version}"
        )
    verify_resolution_id(resolved)
    if for_execution and not has_strong_identity(resolved.model_ref):
        raise ResolutionCompatibilityError(
            "cross-node execution requires strong identity (digest or immutable "
            f"hub commit); model_ref {resolved.model_ref.fingerprint!r} is weak"
        )


def consume_resolved_model(
    resolved: ResolvedModel, registry_profiles: Mapping[str, FamilyProfile]
) -> FamilyProfile:
    """Validate a trace against a consumer's own registry; never re-detects.

    Verifies schema, that the family is known, and that the traced profile equals
    the consumer's canonical row field-for-field. Returns the consumer row.
    """

    validate_resolved_model_trace(resolved, for_execution=False)
    family_id = resolved.profile.family_id
    canonical = registry_profiles.get(family_id)
    if canonical is None:
        raise ResolutionCompatibilityError(f"unknown neutral family: {family_id!r}")
    if resolved.profile != canonical:
        raise ResolutionCompatibilityError(
            f"traced profile for {family_id!r} differs from the consumer's "
            "canonical registry row"
        )
    return canonical


# ---------------------------------------------------------------------------
# Overlay ownership (moved from worker_pool) + single resolver entrypoint
# ---------------------------------------------------------------------------


def merge_mode_capabilities(model_info: ModelInfo, mode: ModeConfig) -> ModelInfo:
    """Overlay authoritative mode-level capability overrides onto detected model info."""
    resolved = deepcopy(model_info)
    for field in (
        "loader_format",
        "checkpoint_precision",
        "checkpoint_variant",
        "scheduler_profile",
        "recommended_size",
        "runtime_quantize",
        "runtime_offload",
        "runtime_attention_slicing",
        "runtime_enable_xformers",
        "negative_prompt_templates",
        "default_negative_prompt_template",
        "allow_custom_negative_prompt",
        "allowed_scheduler_ids",
        "default_scheduler_id",
    ):
        value = getattr(mode, field, None)
        if value is not None:
            setattr(resolved, field, value)
    existing_metadata = getattr(resolved, "metadata", None)
    if not isinstance(existing_metadata, Mapping):
        existing_metadata = {}
    mode_metadata = getattr(mode, "metadata", None)
    if isinstance(mode_metadata, Mapping) and mode_metadata:
        resolved.metadata = {
            **dict(existing_metadata),
            **dict(mode_metadata),
        }
    elif existing_metadata:
        resolved.metadata = dict(existing_metadata)
    return resolved


def resolve_model(
    model_path: str, mode: ModeConfig
) -> tuple[ResolvedModel, LocalModelBinding]:
    """Detect, resolve family (pre-overlay), then overlay mode capabilities.

    The ordering is structural: family resolution consumes detector output before
    mode overlays. ``raw_info`` and ``info`` are distinct frozen captures.
    """

    raw = detect_model(model_path)
    profile = resolve_family(raw)
    enriched = merge_mode_capabilities(raw, mode)
    resolved = build_resolved(
        model_ref=local_artifact_ref(model_path),
        raw_info=raw,
        profile=profile,
        info=enriched,
    )
    return resolved, LocalModelBinding(model_path=model_path)
