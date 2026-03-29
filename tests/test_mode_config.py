"""
Tests for ModeConfigManager capability field parsing and serialization.
"""
import pytest


def test_mode_config_parses_loader_capability_overrides(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
modes:
  sdxl-fp8:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    default_steps: 30
    default_guidance: 7.5
    loader_format: single_file
    checkpoint_precision: fp8
    checkpoint_variant: sdxl-base
    scheduler_profile: native
    recommended_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl-fp8")

    assert mode.loader_format == "single_file"
    assert mode.checkpoint_precision == "fp8"
    assert mode.checkpoint_variant == "sdxl-base"
    assert mode.scheduler_profile == "native"
    assert mode.recommended_size == "512x512"


def test_mode_config_capability_fields_default_to_none(tmp_path):
    """Modes without capability overrides should have None for all new fields."""
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
modes:
  sd15:
    model: checkpoints/sd15/model.safetensors
    default_size: 512x512
    default_steps: 20
    default_guidance: 7.5
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sd15")

    assert mode.loader_format is None
    assert mode.checkpoint_precision is None
    assert mode.checkpoint_variant is None
    assert mode.scheduler_profile is None
    assert mode.recommended_size is None


def test_mode_config_to_dict_includes_capability_fields(tmp_path):
    """to_dict() must round-trip all capability fields."""
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
modes:
  sdxl-fp8:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    default_steps: 30
    default_guidance: 7.5
    loader_format: single_file
    checkpoint_precision: fp8
    checkpoint_variant: sdxl-base
    scheduler_profile: native
    recommended_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    d = manager.to_dict()
    mode_d = d["modes"]["sdxl-fp8"]

    assert mode_d["loader_format"] == "single_file"
    assert mode_d["checkpoint_precision"] == "fp8"
    assert mode_d["checkpoint_variant"] == "sdxl-base"
    assert mode_d["scheduler_profile"] == "native"
    assert mode_d["recommended_size"] == "512x512"
