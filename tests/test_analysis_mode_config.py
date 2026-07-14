"""Tests for analysis_connections / analysis_delegates / analysis_profiles parsing."""
import textwrap

import pytest

from server.mode_config import ModeConfigManager

BASE_YAML = textwrap.dedent("""\
    model_root: /tmp/models
    lora_root: /tmp/loras
    default_mode: SDXL
    resolution_sets:
      default:
        - size: 512x512
          aspect_ratio: "1:1"
    analysis_connections:
      local_vlm:
        endpoint: "http://node2.lan:8080/v1"
        api_key_env: "OPENAI_API_KEY"
      local_detector:
        endpoint: "http://node2.lan:8090"
    analysis_delegates:
      vlm_caption:
        connection: local_vlm
        kind: caption
        model: qwen2.5-vl
      yolo_detect:
        connection: local_detector
        kind: detect
        model: yolo11x
    analysis_profiles:
      default:
        task_routes:
          caption: vlm_caption
          detect: yolo_detect
    modes:
      SDXL:
        model: sdxl/model.safetensors
        analysis_profile: default
""")


def load(tmp_path, yaml_text):
    # ModeConfigManager takes the config *directory* and appends modes.yml
    # itself (server/mode_config.py:164) — pass tmp_path, not the file.
    (tmp_path / "modes.yml").write_text(yaml_text)
    return ModeConfigManager(str(tmp_path))


def test_parses_analysis_sections(tmp_path):
    mgr = load(tmp_path, BASE_YAML)
    cfg = mgr.config
    assert cfg.analysis_connections["local_vlm"].endpoint == "http://node2.lan:8080/v1"
    assert cfg.analysis_delegates["vlm_caption"].kind == "caption"
    assert cfg.analysis_delegates["vlm_caption"].connection == "local_vlm"
    assert cfg.analysis_profiles["default"].task_routes == {
        "caption": "vlm_caption", "detect": "yolo_detect",
    }
    assert cfg.modes["SDXL"].analysis_profile == "default"


def test_sections_default_empty(tmp_path):
    # Drop every analysis_* section and the mode's analysis_profile line.
    yaml_text = BASE_YAML[: BASE_YAML.index("analysis_connections:")] + BASE_YAML[BASE_YAML.index("modes:"):]
    yaml_text = yaml_text.replace("    analysis_profile: default\n", "")
    cfg = load(tmp_path, yaml_text).config
    assert cfg.analysis_connections == {}
    assert cfg.analysis_profiles == {}
    assert cfg.modes["SDXL"].analysis_profile is None


def test_to_dict_includes_analysis_sections(tmp_path):
    d = load(tmp_path, BASE_YAML).to_dict()
    assert d["analysis_connections"]["local_vlm"] == {
        "endpoint": "http://node2.lan:8080/v1", "api_key_env": "OPENAI_API_KEY",
    }
    assert d["analysis_delegates"]["yolo_detect"] == {
        "connection": "local_detector", "kind": "detect", "model": "yolo11x",
    }
    assert d["analysis_profiles"]["default"] == {
        "task_routes": {"caption": "vlm_caption", "detect": "yolo_detect"},
    }
    assert d["modes"]["SDXL"]["analysis_profile"] == "default"


def test_save_config_round_trip_preserves_analysis(tmp_path):
    mgr = load(tmp_path, BASE_YAML)
    # The ordinary save path: export, save, reload.
    mgr.save_config(mgr.to_dict())
    cfg = mgr.config
    assert set(cfg.analysis_connections) == {"local_vlm", "local_detector"}
    assert cfg.analysis_delegates["vlm_caption"].kind == "caption"
    assert cfg.analysis_profiles["default"].task_routes["detect"] == "yolo_detect"
    assert cfg.modes["SDXL"].analysis_profile == "default"


def test_save_config_rejects_unknown_mode_analysis_profile(tmp_path):
    mgr = load(tmp_path, BASE_YAML)
    data = mgr.to_dict()
    data["modes"]["SDXL"]["analysis_profile"] = "nope"
    with pytest.raises(ValueError, match="unknown analysis_profile"):
        mgr.save_config(data)


@pytest.mark.parametrize(
    "needle,replacement,err_fragment",
    [
        # delegate references unknown connection
        ("connection: local_vlm", "connection: nope", "unknown connection"),
        # delegate kind outside closed enum
        ("kind: caption", "kind: segment", "kind"),
        # profile routes to unknown delegate
        ("caption: vlm_caption", "caption: nope", "unknown delegate"),
        # route key != delegate kind -> analysis_delegate_kind_mismatch
        # (mutate the *detect* route so YAML last-key-wins can't mask it)
        ("detect: yolo_detect", "detect: vlm_caption", "analysis_delegate_kind_mismatch"),
        # mode references unknown profile
        ("analysis_profile: default", "analysis_profile: nope", "unknown analysis_profile"),
    ],
)
def test_fail_fast_validation(tmp_path, needle, replacement, err_fragment):
    bad = BASE_YAML.replace(needle, replacement, 1)
    with pytest.raises(ValueError, match=err_fragment):
        load(tmp_path, bad)


PROVIDER_YAML = BASE_YAML.replace(
    "    model: qwen2.5-vl\n",
    "    model: qwen2.5-vl\n"
    "    provider: openai_vlm\n"
    "    options:\n"
    "      max_tokens: 256\n"
    "      temperature: 0.0\n"
    "      timeout_s: 90\n"
    "      system_prompt: \"Describe for a catalog.\"\n",
)


def test_delegate_provider_defaults_to_stub(tmp_path):
    cfg = load(tmp_path, BASE_YAML).config
    assert cfg.analysis_delegates["vlm_caption"].provider == "stub"
    assert cfg.analysis_delegates["vlm_caption"].options == {}


def test_delegate_provider_and_options_parse(tmp_path):
    cfg = load(tmp_path, PROVIDER_YAML).config
    d = cfg.analysis_delegates["vlm_caption"]
    assert d.provider == "openai_vlm"
    assert d.options == {
        "max_tokens": 256,
        "temperature": 0.0,
        "timeout_s": 90,
        "system_prompt": "Describe for a catalog.",
    }


def test_unknown_provider_fails_load(tmp_path):
    bad = BASE_YAML.replace(
        "    model: qwen2.5-vl\n",
        "    model: qwen2.5-vl\n    provider: nonsense\n",
    )
    with pytest.raises(ValueError, match="provider"):
        load(tmp_path, bad)


def test_openai_vlm_on_non_caption_kind_fails_load(tmp_path):
    bad = BASE_YAML.replace(
        "    model: yolo11x\n",
        "    model: yolo11x\n    provider: openai_vlm\n",
    )
    with pytest.raises(ValueError, match="openai_vlm"):
        load(tmp_path, bad)


@pytest.mark.parametrize("options_yaml, match", [
    ("      bogus_key: 1\n", "bogus_key"),
    ("      max_tokens: 0\n", "max_tokens"),
    ("      max_tokens: not-a-number\n", "max_tokens"),
    ("      temperature: -1\n", "temperature"),
    ("      timeout_s: 0\n", "timeout_s"),
    ("      system_prompt: \"\"\n", "system_prompt"),
])
def test_bad_options_fail_load(tmp_path, options_yaml, match):
    bad = BASE_YAML.replace(
        "    model: qwen2.5-vl\n",
        "    model: qwen2.5-vl\n    options:\n" + options_yaml,
    )
    with pytest.raises(ValueError, match=match):
        load(tmp_path, bad)


def test_options_accepted_without_provider_field(tmp_path):
    # options is a delegate-tuning surface, not an openai_vlm exclusive.
    ok = BASE_YAML.replace(
        "    model: qwen2.5-vl\n",
        "    model: qwen2.5-vl\n    options:\n      max_tokens: 128\n",
    )
    cfg = load(tmp_path, ok).config
    assert cfg.analysis_delegates["vlm_caption"].options == {"max_tokens": 128}
    assert cfg.analysis_delegates["vlm_caption"].provider == "stub"


def test_provider_and_options_survive_export_save_reload(tmp_path):
    # Spec: round-trip persistence is definition-of-done.
    mgr = load(tmp_path, PROVIDER_YAML)
    exported = mgr.to_dict()
    d = exported["analysis_delegates"]["vlm_caption"]
    assert d["provider"] == "openai_vlm"
    assert d["options"]["max_tokens"] == 256
    # Default-valued delegates omit the fields (clean exports).
    assert "provider" not in exported["analysis_delegates"]["yolo_detect"]
    assert "options" not in exported["analysis_delegates"]["yolo_detect"]

    mgr.save_config(exported)
    reloaded = mgr.config.analysis_delegates["vlm_caption"]
    assert reloaded.provider == "openai_vlm"
    assert reloaded.options["system_prompt"] == "Describe for a catalog."
