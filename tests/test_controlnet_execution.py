import pytest


@pytest.fixture
def configured_controlnet_registry(monkeypatch, tmp_path):
    from server.controlnet_registry import reset_controlnet_registry

    models_dir = tmp_path / "models"
    registry_paths = {
        "sdxl-canny": models_dir / "sdxl-canny",
        "sdxl-depth": models_dir / "sdxl-depth",
        "sd15-canny": models_dir / "sd15-canny",
        "sd15-depth": models_dir / "sd15-depth",
    }
    for path in registry_paths.values():
        path.mkdir(parents=True)

    config_path = tmp_path / "controlnets.yaml"
    config_path.write_text(
        "models:\n"
        f"  sdxl-canny:\n"
        f"    path: {registry_paths['sdxl-canny']}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n"
        f"  sdxl-depth:\n"
        f"    path: {registry_paths['sdxl-depth']}\n"
        "    control_types: [depth]\n"
        "    compatible_with: [sdxl]\n"
        f"  sd15-canny:\n"
        f"    path: {registry_paths['sd15-canny']}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sd15]\n"
        f"  sd15-depth:\n"
        f"    path: {registry_paths['sd15-depth']}\n"
        "    control_types: [depth]\n"
        "    compatible_with: [sd15]\n",
        encoding="utf-8",
    )

    reset_controlnet_registry()
    monkeypatch.setenv("CONTROLNET_REGISTRY_PATH", str(config_path))
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "strict")
    yield
    reset_controlnet_registry()


def test_resolve_controlnet_bindings_rejects_wrong_family(configured_controlnet_registry):
    from server.asset_store import AssetStore
    from server.controlnet_models import ControlNetAttachment
    from server.controlnet_execution import resolve_controlnet_bindings

    store = AssetStore()
    ref = store.insert("control_map", b"png-bytes")
    req = type(
        "Req",
        (),
        {
            "controlnets": [
                ControlNetAttachment(
                    attachment_id="cn_1",
                    control_type="canny",
                    model_id="sd15-canny",
                    map_asset_ref=ref,
                )
            ]
        },
    )()
    mode = type("Mode", (), {"name": "sdxl-mode", "model_path": "/tmp/sdxl.safetensors"})()

    with pytest.raises(ValueError, match="incompatible with active mode family"):
        resolve_controlnet_bindings(req, mode=mode, store=store, active_family="sdxl")


def test_resolve_controlnet_bindings_preserves_request_order(configured_controlnet_registry):
    from server.asset_store import AssetStore
    from server.controlnet_models import ControlNetAttachment
    from server.controlnet_execution import resolve_controlnet_bindings

    store = AssetStore()
    ref1 = store.insert("control_map", b"first-map")
    ref2 = store.insert("control_map", b"second-map")
    req = type(
        "Req",
        (),
        {
            "controlnets": [
                ControlNetAttachment(
                    attachment_id="cn_1",
                    control_type="canny",
                    model_id="sdxl-canny",
                    map_asset_ref=ref1,
                ),
                ControlNetAttachment(
                    attachment_id="cn_2",
                    control_type="depth",
                    model_id="sdxl-depth",
                    map_asset_ref=ref2,
                ),
            ]
        },
    )()
    mode = type("Mode", (), {"name": "sdxl-mode", "model_path": "/tmp/sdxl.safetensors"})()

    bindings = resolve_controlnet_bindings(req, mode=mode, store=store, active_family="sdxl")
    assert [binding.attachment_id for binding in bindings] == ["cn_1", "cn_2"]
    assert bindings[0].control_image_bytes == b"first-map"
    assert bindings[1].control_image_bytes == b"second-map"


def test_active_model_family_from_variant_maps_supported_prefixes():
    from server.controlnet_execution import active_model_family_from_variant

    assert active_model_family_from_variant("sdxl-turbo") == "sdxl"
    assert active_model_family_from_variant("sd15-base") == "sd15"
    assert active_model_family_from_variant("sd2-inpaint") == "sd15"


def test_active_model_family_from_variant_rejects_unknown_prefix():
    from server.controlnet_execution import active_model_family_from_variant

    with pytest.raises(ValueError, match="unsupported active model family for ControlNet"):
        active_model_family_from_variant("flux-dev")
