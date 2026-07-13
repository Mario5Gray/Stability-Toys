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
