import pytest


def test_registry_loads_local_controlnet_specs(tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    model_dir = tmp_path / "models" / "sdxl-canny"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  sdxl-canny:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )

    from server.controlnet_registry import load_controlnet_registry

    registry = load_controlnet_registry(config_path=str(config_path), validation_mode="strict")
    spec = registry.get_required("sdxl-canny")
    assert spec.model_id == "sdxl-canny"
    assert spec.control_types == ("canny",)
    assert spec.compatible_with == ("sdxl",)


def test_registry_rejects_missing_local_path_in_strict_mode(tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    config_path.write_text(
        "models:\n"
        "  sdxl-depth:\n"
        "    path: /does/not/exist\n"
        "    control_types: [depth]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )

    from server.controlnet_registry import load_controlnet_registry

    with pytest.raises(ValueError, match="does not exist"):
        load_controlnet_registry(config_path=str(config_path), validation_mode="strict")
