"""
Tests for ModeConfigManager capability field parsing and serialization.
"""
import pytest
import yaml


def test_mode_config_parses_loader_capability_overrides(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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


def test_mode_config_parses_runtime_policy_overrides(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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
    runtime_quantize: none
    runtime_offload: model
    runtime_attention_slicing: true
    runtime_enable_xformers: true
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl-fp8")

    assert mode.runtime_quantize == "none"
    assert mode.runtime_offload == "model"
    assert mode.runtime_attention_slicing is True
    assert mode.runtime_enable_xformers is True


def test_mode_config_parses_chat_block(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-chat
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl-chat:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    chat:
      endpoint: http://localhost:11434/v1
      model: llama3.2
      api_key_env: OPENAI_API_KEY
      max_tokens: 768
      temperature: 0.4
      system_prompt: You are concise.
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl-chat")

    assert mode.chat is not None
    assert mode.chat.endpoint == "http://localhost:11434/v1"
    assert mode.chat.model == "llama3.2"
    assert mode.chat.api_key_env == "OPENAI_API_KEY"
    assert mode.chat.max_tokens == 768
    assert mode.chat.temperature == 0.4
    assert mode.chat.system_prompt == "You are concise."


def test_mode_config_chat_numeric_field_errors_include_mode_name(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-chat
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl-chat:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    chat:
      endpoint: http://localhost:11434/v1
      model: llama3.2
      max_tokens: not-a-number
""".strip()
    )

    from server.mode_config import ModeConfigManager

    with pytest.raises(ValueError) as exc:
        ModeConfigManager(str(tmp_path))
    assert "Mode 'sdxl-chat'" in str(exc.value)
    assert "max_tokens" in str(exc.value)


def test_mode_config_parses_maximum_len(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    default_size: 512x512
    default_steps: 30
    default_guidance: 7.5
    maximum_len: 240
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    assert manager.get_mode("sdxl").maximum_len == 240


def test_mode_config_maximum_len_defaults_to_none(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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
    assert manager.get_mode("sd15").maximum_len is None


def test_mode_config_parses_maximum_len(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    default_size: 512x512
    default_steps: 30
    default_guidance: 7.5
    maximum_len: 240
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    assert manager.get_mode("sdxl").maximum_len == 240


def test_mode_config_maximum_len_defaults_to_none(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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
    assert manager.get_mode("sd15").maximum_len is None


def test_mode_config_parses_maximum_len(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    default_size: 512x512
    default_steps: 30
    default_guidance: 7.5
    maximum_len: 240
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    assert manager.get_mode("sdxl").maximum_len == 240


def test_mode_config_maximum_len_defaults_to_none(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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
    assert manager.get_mode("sd15").maximum_len is None


def test_mode_config_capability_fields_default_to_none(tmp_path):
    """Modes without capability overrides should have None for all new fields."""
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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


def test_mode_config_runtime_policy_fields_default_to_none(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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

    assert mode.runtime_quantize is None
    assert mode.runtime_offload is None
    assert mode.runtime_attention_slicing is None
    assert mode.runtime_enable_xformers is None


def test_mode_config_to_dict_includes_capability_fields(tmp_path):
    """to_dict() must round-trip all capability fields."""
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
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


def test_mode_config_parses_resolution_sets_and_round_trips_them(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
  sdxl:
    - size: 1024x1024
      aspect_ratio: "1:1"
    - size: 896x1152
      aspect_ratio: "7:9"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    resolution_set: sdxl
    default_size: 1024x1024
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl")
    mode_d = manager.to_dict()["modes"]["sdxl"]

    assert mode.resolution_set == "sdxl"
    assert mode.resolution_options == [
        {"size": "1024x1024", "aspect_ratio": "1:1"},
        {"size": "896x1152", "aspect_ratio": "7:9"},
    ]
    assert mode_d["resolution_set"] == "sdxl"
    assert mode_d["resolution_options"] == [
        {"size": "1024x1024", "aspect_ratio": "1:1"},
        {"size": "896x1152", "aspect_ratio": "7:9"},
    ]


def test_mode_config_to_dict_includes_top_level_resolution_sets(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
  sdxl:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    resolution_set: sdxl
    default_size: 1024x1024
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    d = manager.to_dict()

    assert d["resolution_sets"] == {
        "default": [{"size": "512x512", "aspect_ratio": "1:1"}],
        "sdxl": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
    }


def test_mode_config_save_config_round_trips_to_dict_output(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
  sdxl:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    resolution_set: sdxl
    default_size: 1024x1024
    negative_prompt_templates:
      safe_photo: "blurry, distorted, low quality"
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    manager.save_config(manager.to_dict())

    saved = yaml.safe_load(cfg.read_text())
    reloaded = ModeConfigManager(str(tmp_path)).get_mode("sdxl")

    assert saved["resolution_sets"] == {
        "default": [{"size": "512x512", "aspect_ratio": "1:1"}],
        "sdxl": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
    }
    assert saved["modes"]["sdxl"]["resolution_set"] == "sdxl"
    assert "resolution_options" not in saved["modes"]["sdxl"]
    assert reloaded.resolution_set == "sdxl"
    assert reloaded.resolution_options == [{"size": "1024x1024", "aspect_ratio": "1:1"}]
    assert reloaded.negative_prompt_templates == {
        "safe_photo": "blurry, distorted, low quality"
    }


def test_mode_config_requires_default_resolution_set(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: base
resolution_sets:
  sdxl:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  base:
    model: checkpoints/base/model.safetensors
    default_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    with pytest.raises(ValueError, match=r"resolution_sets\.default"):
        ModeConfigManager(str(tmp_path))


def test_mode_config_rejects_unknown_resolution_set(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: base
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  base:
    model: checkpoints/base/model.safetensors
    resolution_set: missing
    default_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    with pytest.raises(ValueError, match=r"unknown resolution_set 'missing'"):
        ModeConfigManager(str(tmp_path))


def test_mode_config_rejects_default_size_outside_resolution_set(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: base
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
  narrow:
    - size: 896x1152
      aspect_ratio: "7:9"
modes:
  base:
    model: checkpoints/base/model.safetensors
    resolution_set: narrow
    default_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    with pytest.raises(ValueError, match=r"default_size '512x512'"):
        ModeConfigManager(str(tmp_path))


def test_mode_config_to_dict_includes_runtime_policy_fields(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl-fp8:
    model: checkpoints/sdxl/sdxl-base.safetensors
    runtime_quantize: none
    runtime_offload: model
    runtime_attention_slicing: true
    runtime_enable_xformers: true
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode_d = manager.to_dict()["modes"]["sdxl-fp8"]

    assert mode_d["runtime_quantize"] == "none"
    assert mode_d["runtime_offload"] == "model"
    assert mode_d["runtime_attention_slicing"] is True
    assert mode_d["runtime_enable_xformers"] is True


def test_mode_config_parses_generation_control_policy_fields(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    negative_prompt_templates:
      safe_photo: "blurry, distorted, low quality"
      illustration_clean: "text, watermark, extra fingers"
    default_negative_prompt_template: safe_photo
    allow_custom_negative_prompt: true
    allowed_scheduler_ids:
      - euler
      - dpmpp_2m
    default_scheduler_id: euler
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl")

    assert mode.negative_prompt_templates == {
        "safe_photo": "blurry, distorted, low quality",
        "illustration_clean": "text, watermark, extra fingers",
    }
    assert mode.default_negative_prompt_template == "safe_photo"
    assert mode.allow_custom_negative_prompt is True
    assert mode.allowed_scheduler_ids == ["euler", "dpmpp_2m"]
    assert mode.default_scheduler_id == "euler"


def test_mode_config_generation_control_fields_default_safely(tmp_path):
    """Absent policy fields should preserve open/unspecified backend semantics."""
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sd15:
    model: checkpoints/sd15/model.safetensors
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sd15")

    assert mode.negative_prompt_templates == {}
    assert mode.default_negative_prompt_template is None
    assert mode.allow_custom_negative_prompt is False
    assert mode.allowed_scheduler_ids is None
    assert mode.default_scheduler_id is None


def test_mode_config_to_dict_preserves_empty_scheduler_allowlist(tmp_path):
    """Explicit empty allowlists must round-trip distinctly from absent/null."""
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    negative_prompt_templates:
      base: "bad anatomy"
    allow_custom_negative_prompt: false
    allowed_scheduler_ids: []
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode_d = manager.to_dict()["modes"]["sdxl"]

    assert mode_d["negative_prompt_templates"] == {"base": "bad anatomy"}
    assert mode_d["default_negative_prompt_template"] is None
    assert mode_d["allow_custom_negative_prompt"] is False
    assert mode_d["allowed_scheduler_ids"] == []
    assert mode_d["default_scheduler_id"] is None


def test_mode_config_save_config_round_trips_generation_control_fields(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: base
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  base:
    model: checkpoints/base.safetensors
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    manager.save_config(
        {
            "model_root": "/models",
            "lora_root": "/models/loras",
            "default_mode": "base",
            "resolution_sets": {
                "default": [
                    {"size": "512x512", "aspect_ratio": "1:1"},
                ],
                "base": [
                    {"size": "512x512", "aspect_ratio": "1:1"},
                ],
            },
            "modes": {
                "base": {
                    "model": "checkpoints/base.safetensors",
                    "resolution_set": "base",
                    "default_size": "512x512",
                    "default_steps": 20,
                    "default_guidance": 7.0,
                    "negative_prompt_templates": {
                        "safe_photo": "blurry, watermark",
                    },
                    "default_negative_prompt_template": "safe_photo",
                    "allow_custom_negative_prompt": True,
                    "allowed_scheduler_ids": ["euler"],
                    "default_scheduler_id": "euler",
                }
            },
        }
    )

    saved = yaml.safe_load(cfg.read_text())
    reloaded = ModeConfigManager(str(tmp_path)).get_mode("base")

    assert saved["resolution_sets"] == {
        "default": [{"size": "512x512", "aspect_ratio": "1:1"}],
        "base": [{"size": "512x512", "aspect_ratio": "1:1"}],
    }
    assert saved["modes"]["base"]["resolution_set"] == "base"
    assert "resolution_options" not in saved["modes"]["base"]
    assert reloaded.negative_prompt_templates == {"safe_photo": "blurry, watermark"}
    assert reloaded.default_negative_prompt_template == "safe_photo"
    assert reloaded.allow_custom_negative_prompt is True
    assert reloaded.allowed_scheduler_ids == ["euler"]
    assert reloaded.default_scheduler_id == "euler"
    assert reloaded.resolution_set == "base"
    assert reloaded.resolution_options == [{"size": "512x512", "aspect_ratio": "1:1"}]
