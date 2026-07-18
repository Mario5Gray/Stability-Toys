"""Tests for the neutral family registry and exact-one resolver.

Phase 2 gate: only SD families are registered here. The Task 1 HunyuanDiT
transformer fixture stays a zero-match unsupported case until Task 9 adds the
Hunyuan data row.
"""

import subprocess
import sys

import pytest

from utils.model_detector import ModelInfo


def _info(**overrides) -> ModelInfo:
    return ModelInfo(path="/models/test", **overrides)


def test_family_profiles_module_is_import_clean():
    # Verified in a fresh interpreter: an in-session assertion is unreliable
    # because sibling tests (compel) pull torch into this process first.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import backends.family_profiles; "
            "assert 'torch' not in sys.modules and 'diffusers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registered_profiles_are_pure_data():
    from backends.family_profiles import FAMILY_REGISTRY

    json_safe = (str, int, float, bool, type(None))
    for registration in FAMILY_REGISTRY:
        profile = registration.profile
        for value in vars(profile).values():
            if isinstance(value, tuple):
                assert all(isinstance(item, json_safe) for item in value)
            else:
                assert isinstance(value, json_safe)


def test_sd15_and_sdxl_profile_shapes_match_the_design():
    from backends.family_profiles import SD15_PROFILE, SDXL_PROFILE

    assert SD15_PROFILE.family_id == "sd15"
    assert SD15_PROFILE.encoder_roles == ("text_encoder",)
    assert SD15_PROFILE.pooled_required is False
    assert SD15_PROFILE.pooled_projection_role is None
    assert SD15_PROFILE.control_image_kwarg == "image"

    assert SDXL_PROFILE.family_id == "sdxl"
    assert SDXL_PROFILE.encoder_roles == ("text_encoder", "text_encoder_2")
    assert SDXL_PROFILE.pooled_required is True
    assert SDXL_PROFILE.pooled_projection_role == "text_encoder_2"
    assert SDXL_PROFILE.control_image_kwarg == "image"


@pytest.mark.parametrize("cad", [768, 1024])
def test_unet_low_cad_resolves_to_sd15(cad):
    from backends.family_profiles import SD15_PROFILE, resolve_family

    resolved = resolve_family(_info(base_arch="unet", cross_attention_dim=cad))
    assert resolved is SD15_PROFILE


@pytest.mark.parametrize("cad", [1280, 2048])
def test_unet_high_cad_resolves_to_sdxl(cad):
    from backends.family_profiles import SDXL_PROFILE, resolve_family

    resolved = resolve_family(_info(base_arch="unet", cross_attention_dim=cad))
    assert resolved is SDXL_PROFILE


def test_zero_matches_raise_family_resolution_error():
    from backends.family_profiles import FamilyResolutionError, resolve_family

    hunyuan = _info(
        base_arch="transformer",
        transformer_kind="hunyuandit",
        cross_attention_dim=1024,
    )
    with pytest.raises(FamilyResolutionError):
        resolve_family(hunyuan)


def test_multiple_matches_raise_family_resolution_error():
    from backends.family_profiles import (
        SD15_PROFILE,
        SDXL_PROFILE,
        FamilyRegistration,
        FamilyResolutionError,
        resolve_family,
    )

    overlapping = (
        FamilyRegistration(SD15_PROFILE, lambda info: True),
        FamilyRegistration(SDXL_PROFILE, lambda info: True),
    )
    with pytest.raises(FamilyResolutionError):
        resolve_family(_info(base_arch="unet", cross_attention_dim=768),
                       registry=overlapping)


def test_checkpoint_variant_is_not_read_by_predicates():
    from backends.family_profiles import SD15_PROFILE, resolve_family

    # An overlaid checkpoint_variant must not steer neutral resolution.
    resolved = resolve_family(
        _info(base_arch="unet", cross_attention_dim=768, checkpoint_variant="sdxl")
    )
    assert resolved is SD15_PROFILE


def test_validate_family_id_accepts_known_and_rejects_unknown():
    from backends.family_profiles import UnknownFamilyError, validate_family_id

    assert validate_family_id("sd15") == "sd15"
    assert validate_family_id("sdxl") == "sdxl"
    with pytest.raises(UnknownFamilyError):
        validate_family_id("hunyuandit")
