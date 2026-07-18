"""Tests for the portable ResolvedModel codec, artifact identity, and resolver.

Covers Task 3: path-free wire data, deterministic descriptor identity, explicit
weak/strong artifact identity, and the single pre-overlay resolver entrypoint.
"""

import hashlib
import os
from types import SimpleNamespace
from typing import Any

import pytest
import rfc8785

from backends.family_profiles import SD15_PROFILE, SDXL_PROFILE
from backends.model_resolution import (
    RESOLVED_MODEL_SCHEMA_VERSION,
    LocalModelBinding,
    ModelArtifactRef,
    ModelInfoSnapshot,
    ResolutionCompatibilityError,
    ResolvedModel,
    canonical_resolution_bytes,
    consume_resolved_model,
    freeze_model_info,
    hub_ref,
    local_artifact_ref,
    profile_to_json_dict,
    resolve_model,
    resolved_model_from_json_dict,
    resolved_model_to_json_dict,
    snapshot_to_json_dict,
    thaw_model_info,
    validate_resolved_model_trace,
)
from utils.model_detector import ModelInfo, ModelVariant


def _info(path="/host/models/secret-model", **overrides) -> ModelInfo:
    base: dict[str, Any] = dict(
        variant=ModelVariant.SD15,
        cross_attention_dim=768,
        text_encoder_hidden_size=768,
        base_arch="unet",
        format="diffusers",
        confidence=0.9,
        detected_by=["DiffusersDetector"],
        metadata={"source": "unit-test", "rank": 3},
    )
    base.update(overrides)
    return ModelInfo(path=path, **base)


# --- Snapshot codec ----------------------------------------------------------


def test_every_model_info_field_except_path_freezes_json_safe():
    snapshot = freeze_model_info(_info())
    assert isinstance(snapshot, ModelInfoSnapshot)
    assert not hasattr(snapshot, "path")
    wire = snapshot_to_json_dict(snapshot)
    # rfc8785 accepts the wire dict -> proves it is JSON-safe end to end.
    rfc8785.dumps(wire)
    assert "path" not in wire
    assert wire["base_arch"] == "unet"
    assert wire["transformer_kind"] is None


def test_snapshot_is_frozen():
    snapshot = freeze_model_info(_info())
    with pytest.raises(Exception):
        snapshot.base_arch = "transformer"  # type: ignore[misc]


def test_model_variant_round_trips_by_value():
    snapshot = freeze_model_info(_info(variant=ModelVariant.SDXL_REFINER))
    assert snapshot_to_json_dict(snapshot)["variant"] == "sdxl-refiner"
    restored = thaw_model_info(snapshot, LocalModelBinding("/node/local/path"))
    assert restored.variant is ModelVariant.SDXL_REFINER


def test_non_json_metadata_fails_during_freeze_not_later():
    with pytest.raises((TypeError, ValueError)):
        freeze_model_info(_info(metadata={"bad": object()}))


def test_thaw_restores_only_local_path_authority():
    snapshot = freeze_model_info(_info(path="/host/A/model"))
    binding = LocalModelBinding("/node/B/model")
    restored = thaw_model_info(snapshot, binding)
    assert restored.path == "/node/B/model"
    assert restored.base_arch == "unet"
    assert restored.cross_attention_dim == 768


# --- Resolved wire form + identity ------------------------------------------


def _resolved(tmp_path, profile=SD15_PROFILE) -> ResolvedModel:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_bytes(b"{}")
    raw = _info(path=str(model_dir))
    info = _info(path=str(model_dir), checkpoint_variant="fp16")
    ref = local_artifact_ref(str(model_dir))
    from backends.model_resolution import build_resolved

    return build_resolved(model_ref=ref, raw_info=raw, profile=profile, info=info)


def test_serialized_resolved_model_hides_path_and_binding(tmp_path):
    resolved = _resolved(tmp_path)
    wire = resolved_model_to_json_dict(resolved)
    raw_bytes = rfc8785.dumps(wire)
    assert str(tmp_path).encode() not in raw_bytes
    assert b"model_path" not in raw_bytes
    assert b"LocalModelBinding" not in raw_bytes


def test_resolved_model_wire_round_trips(tmp_path):
    resolved = _resolved(tmp_path)
    restored = resolved_model_from_json_dict(resolved_model_to_json_dict(resolved))
    assert restored == resolved


def test_golden_canonical_bytes_and_hash_pin_the_encoding(tmp_path):
    resolved = _resolved(tmp_path)
    expected_payload = {
        "schema_version": RESOLVED_MODEL_SCHEMA_VERSION,
        "model_ref": {
            "kind": "local-dir",
            "name": "model",
            "revision": None,
            "fingerprint": resolved.model_ref.fingerprint,
            "digest": None,
        },
        "raw_info": snapshot_to_json_dict(resolved.raw_info),
        "profile": profile_to_json_dict(SD15_PROFILE),
        "info": snapshot_to_json_dict(resolved.info),
    }
    expected_bytes = rfc8785.dumps(expected_payload)
    assert canonical_resolution_bytes(resolved) == expected_bytes
    assert resolved.resolution_id == hashlib.sha256(expected_bytes).hexdigest()


def test_every_profile_field_contributes_to_resolution_id(tmp_path):
    baseline = _resolved(tmp_path, profile=SD15_PROFILE)
    from dataclasses import replace

    for field_name in (
        "family_id",
        "encoder_roles",
        "pooled_required",
        "pooled_projection_role",
        "control_image_kwarg",
    ):
        mutated_profile = {
            "family_id": replace(SD15_PROFILE, family_id="sdxl"),
            "encoder_roles": replace(SD15_PROFILE, encoder_roles=("text_encoder", "x")),
            "pooled_required": replace(SD15_PROFILE, pooled_required=True),
            "pooled_projection_role": replace(
                SD15_PROFILE, pooled_projection_role="text_encoder"
            ),
            "control_image_kwarg": replace(SD15_PROFILE, control_image_kwarg="control_image"),
        }[field_name]
        other = _resolved(tmp_path, profile=mutated_profile)
        assert other.resolution_id != baseline.resolution_id, field_name


# --- Weak/strong artifact identity ------------------------------------------


def test_local_file_fingerprint_stable_across_path_and_mtime(tmp_path):
    # Same basename + size in different parent dirs, different mtime -> identical
    # weak fingerprint (the ref fingerprints [name, size], not the host path/mtime).
    a = tmp_path / "nodeA" / "weights.safetensors"
    b = tmp_path / "nodeB" / "weights.safetensors"
    a.parent.mkdir()
    b.parent.mkdir()
    payload = b"x" * 4096
    a.write_bytes(payload)
    b.write_bytes(payload)
    os.utime(b, (1, 1))
    assert local_artifact_ref(str(a)).fingerprint == local_artifact_ref(str(b)).fingerprint

    # A different basename is part of identity and must change the fingerprint.
    c = tmp_path / "nodeA" / "renamed.safetensors"
    c.write_bytes(payload)
    assert local_artifact_ref(str(c)).fingerprint != local_artifact_ref(str(a)).fingerprint


def test_local_dir_manifest_sorts_and_rejects_symlink(tmp_path):
    root = tmp_path / "model"
    (root / "sub").mkdir(parents=True)
    (root / "b.json").write_bytes(b"bb")
    (root / "sub" / "a.bin").write_bytes(b"aaaa")
    ref = local_artifact_ref(str(root))
    assert ref.kind == "local-dir"
    # Fingerprint recomputes deterministically.
    assert local_artifact_ref(str(root)).fingerprint == ref.fingerprint

    link_root = tmp_path / "linked"
    link_root.mkdir()
    (link_root / "real.bin").write_bytes(b"z")
    (link_root / "link.bin").symlink_to(link_root / "real.bin")
    with pytest.raises((ValueError, OSError)):
        local_artifact_ref(str(link_root))


def test_weak_fingerprint_traces_but_fails_execution_without_strong_identity(tmp_path):
    resolved = _resolved(tmp_path)  # local-dir, digest=None
    # Diagnostics view is always allowed.
    validate_resolved_model_trace(resolved, for_execution=False)
    with pytest.raises(ResolutionCompatibilityError):
        validate_resolved_model_trace(resolved, for_execution=True)


def test_hub_immutable_commit_is_strong_but_tag_is_weak(tmp_path):
    commit = "a" * 40
    strong = ResolvedModel(
        schema_version=RESOLVED_MODEL_SCHEMA_VERSION,
        resolution_id="deadbeef",
        model_ref=hub_ref("org/repo", commit),
        raw_info=freeze_model_info(_info()),
        profile=SD15_PROFILE,
        info=freeze_model_info(_info()),
    )
    validate_resolved_model_trace(strong, for_execution=True)

    weak = ResolvedModel(
        schema_version=RESOLVED_MODEL_SCHEMA_VERSION,
        resolution_id="deadbeef",
        model_ref=hub_ref("org/repo", "v1.0"),  # mutable tag
        raw_info=freeze_model_info(_info()),
        profile=SD15_PROFILE,
        info=freeze_model_info(_info()),
    )
    with pytest.raises(ResolutionCompatibilityError):
        validate_resolved_model_trace(weak, for_execution=True)


# --- Consumption never re-detects -------------------------------------------


def test_consumption_mismatches_fail_without_calling_detection(tmp_path, monkeypatch):
    import utils.model_detector as detector

    def _boom(*a, **k):  # detection must never run during consumption
        raise AssertionError("consumption re-detected the model")

    monkeypatch.setattr(detector, "detect_model", _boom)

    resolved = _resolved(tmp_path, profile=SD15_PROFILE)

    # Happy path: traced profile equals the consumer's canonical row.
    assert consume_resolved_model(resolved, {"sd15": SD15_PROFILE}) is SD15_PROFILE

    # Unknown family in the consumer registry.
    with pytest.raises(ResolutionCompatibilityError):
        consume_resolved_model(resolved, {"sdxl": SDXL_PROFILE})

    # Schema mismatch.
    from dataclasses import replace

    bumped = replace(resolved, schema_version=RESOLVED_MODEL_SCHEMA_VERSION + 1)
    with pytest.raises(ResolutionCompatibilityError):
        consume_resolved_model(bumped, {"sd15": SD15_PROFILE})

    # Field-for-field canonical profile mismatch (consumer row differs).
    drifted = replace(SD15_PROFILE, control_image_kwarg="control_image")
    with pytest.raises(ResolutionCompatibilityError):
        consume_resolved_model(resolved, {"sd15": drifted})


# --- resolve_model ordering + moved overlay ---------------------------------


def test_resolve_model_orders_detect_family_overlay(tmp_path, monkeypatch):
    model_dir = tmp_path / "sd15-model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_bytes(b"{}")

    import backends.model_resolution as mr

    detected = _info(path=str(model_dir), base_arch="unet", cross_attention_dim=768)
    monkeypatch.setattr(mr, "detect_model", lambda p: detected)

    mode = SimpleNamespace(scheduler_profile="karras", metadata={"mode": "portrait"})
    resolved, binding = resolve_model(str(model_dir), mode)

    assert isinstance(binding, LocalModelBinding)
    assert binding.model_path == str(model_dir)
    assert resolved.profile is SD15_PROFILE
    # raw_info is pre-overlay, info is post-overlay (mode wins).
    assert resolved.raw_info.scheduler_profile == "unknown"
    assert resolved.info.scheduler_profile == "karras"


def test_merge_mode_capabilities_is_importable_from_resolution_and_pool():
    from backends.model_resolution import merge_mode_capabilities as mr_merge
    from backends.worker_pool import merge_mode_capabilities as pool_merge

    assert mr_merge is pool_merge
