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
    from server.asset_store import InMemoryAssetStore
    from server.controlnet_models import ControlNetAttachment
    from server.controlnet_execution import resolve_controlnet_bindings

    store = InMemoryAssetStore()
    ref = store.write("control_map", b"png-bytes")
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
    from server.asset_store import InMemoryAssetStore
    from server.controlnet_models import ControlNetAttachment
    from server.controlnet_execution import resolve_controlnet_bindings

    store = InMemoryAssetStore()
    ref1 = store.write("control_map", b"first-map")
    ref2 = store.write("control_map", b"second-map")
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


def _hunyuan_combined_admission(attachment):
    from server.controlnet_execution import admit_generation_operation
    from tests.snapshot_test_helpers import make_active_snapshot, make_family_provider

    mode = type("Mode", (), {"name": "hunyuandit-mode"})()
    snapshot = make_active_snapshot(mode, family_id="hunyuandit")
    provider = make_family_provider(family_id="hunyuandit", supports_combined=True)
    req = type("Req", (), {"controlnets": [attachment]})()
    return admit_generation_operation(req, snapshot=snapshot, provider=provider, has_init_image=True)


def test_admit_generation_operation_hunyuan_combined_canny_default_window_admits():
    from server.controlnet_models import ControlNetAttachment

    attachment = ControlNetAttachment(attachment_id="cn", control_type="canny", map_asset_ref="map")
    assert _hunyuan_combined_admission(attachment) == "img2img+controlnet"


def test_admit_generation_operation_hunyuan_combined_rejects_non_canny():
    from server.controlnet_models import ControlNetAttachment

    attachment = ControlNetAttachment(attachment_id="cn", control_type="depth", map_asset_ref="map")
    with pytest.raises(ValueError, match="Canny"):
        _hunyuan_combined_admission(attachment)


def test_admit_generation_operation_hunyuan_combined_rejects_nondefault_start_percent():
    from server.controlnet_models import ControlNetAttachment

    attachment = ControlNetAttachment(
        attachment_id="cn", control_type="canny", map_asset_ref="map", start_percent=0.1
    )
    with pytest.raises(ValueError, match="control timing window"):
        _hunyuan_combined_admission(attachment)


def test_admit_generation_operation_hunyuan_combined_rejects_nondefault_end_percent():
    from server.controlnet_models import ControlNetAttachment

    attachment = ControlNetAttachment(
        attachment_id="cn", control_type="canny", map_asset_ref="map", end_percent=0.9
    )
    with pytest.raises(ValueError, match="control timing window"):
        _hunyuan_combined_admission(attachment)


# --- MPS sd15 cell: text-to-image and img2img (STABL-tukrgrjb) --------------
# These use the real MPSProvider, so the live MPS_FAMILY_BINDINGS table is the
# gate under test, not a hand-built capability triple.


def _mps_admission(*, has_init_image, controlnets=()):
    from backends.platforms.mps import MPSProvider
    from server.controlnet_execution import admit_generation_operation
    from tests.snapshot_test_helpers import make_active_snapshot

    mode = type("Mode", (), {"name": "lcm-general"})()
    snapshot = make_active_snapshot(mode, family_id="sd15")
    req = type("Req", (), {"controlnets": list(controlnets)})()
    return admit_generation_operation(
        req,
        snapshot=snapshot,
        provider=MPSProvider(),
        has_init_image=has_init_image,
    )


def test_admit_generation_operation_mps_sd15_admits_text_to_image():
    assert _mps_admission(has_init_image=False) == "txt2img"


def test_admit_generation_operation_mps_sd15_admits_img2img():
    assert _mps_admission(has_init_image=True) == "img2img"


def test_admit_generation_operation_mps_sd15_admits_one_canny_controlnet():
    from server.controlnet_models import ControlNetAttachment

    attachment = ControlNetAttachment(
        attachment_id="cn_1", control_type="canny", map_asset_ref="map"
    )
    admitted = _mps_admission(has_init_image=False, controlnets=(attachment,))
    assert admitted == "controlnet"


def test_admit_generation_operation_mps_sd15_rejects_a_depth_controlnet():
    """Through the real MPSProvider, so the platform scope is the thing tested."""
    from server.controlnet_models import ControlNetAttachment

    attachment = ControlNetAttachment(
        attachment_id="cn_1", control_type="depth", map_asset_ref="map"
    )
    with pytest.raises(ValueError, match="control_type 'depth'"):
        _mps_admission(has_init_image=False, controlnets=(attachment,))


def test_admit_generation_operation_mps_sd15_rejects_two_controlnet_attachments():
    """conf/modes.yml permits two attachments. The MPS scope permits one."""
    from server.controlnet_models import ControlNetAttachment

    attachments = tuple(
        ControlNetAttachment(
            attachment_id=f"cn_{index}", control_type="canny", map_asset_ref="map"
        )
        for index in (1, 2)
    )
    with pytest.raises(ValueError, match="at most 1 controlnet"):
        _mps_admission(has_init_image=False, controlnets=attachments)


def test_admit_generation_operation_mps_sd15_rejects_the_combined_operation():
    from server.controlnet_models import ControlNetAttachment

    attachment = ControlNetAttachment(
        attachment_id="cn_1", control_type="canny", map_asset_ref="map"
    )
    with pytest.raises(
        ValueError, match="family 'sd15' does not support the img2img\\+controlnet operation"
    ):
        _mps_admission(has_init_image=True, controlnets=(attachment,))


def _scoped_admission(*, controlnets, scope=None, backend_id="mps", family_id="sd15"):
    """Admission through a fake provider, so the scope is reachable before the
    MPS cell flips in Task 4."""
    from server.controlnet_execution import admit_generation_operation
    from tests.snapshot_test_helpers import make_active_snapshot, make_family_provider

    mode = type("Mode", (), {"name": "lcm-general"})()
    snapshot = make_active_snapshot(mode, family_id=family_id)
    provider = make_family_provider(
        family_id=family_id,
        supports_controlnet=True,
        backend_id=backend_id,
        controlnet_scope=scope,
    )
    req = type("Req", (), {"controlnets": list(controlnets)})()
    return admit_generation_operation(
        req, snapshot=snapshot, provider=provider, has_init_image=False
    )


def _canny_only_scope():
    from backends.platforms.base import ControlNetScope

    return ControlNetScope(allowed_control_types=("canny",), max_attachments=1)


def _attachment(control_type, attachment_id="cn_1"):
    from server.controlnet_models import ControlNetAttachment

    return ControlNetAttachment(
        attachment_id=attachment_id, control_type=control_type, map_asset_ref="map"
    )


def test_admit_generation_operation_rejects_a_control_type_outside_the_platform_scope():
    with pytest.raises(ValueError, match="depth"):
        _scoped_admission(controlnets=(_attachment("depth"),), scope=_canny_only_scope())


def test_admit_generation_operation_rejects_more_attachments_than_the_platform_scope_allows():
    """conf/modes.yml permits two attachments. The platform cap is one."""
    attachments = (_attachment("canny", "cn_1"), _attachment("canny", "cn_2"))
    with pytest.raises(ValueError, match="1"):
        _scoped_admission(controlnets=attachments, scope=_canny_only_scope())


def test_admit_generation_operation_admits_one_canny_attachment_within_the_platform_scope():
    admitted = _scoped_admission(
        controlnets=(_attachment("canny"),), scope=_canny_only_scope()
    )
    assert admitted == "controlnet"


def test_admit_generation_operation_leaves_cuda_depth_admitted():
    """The scope is an MPS limit. An unnarrowed CUDA binding admits depth."""
    admitted = _scoped_admission(
        controlnets=(_attachment("depth"),), scope=None, backend_id="cuda"
    )
    assert admitted == "controlnet"
