import pytest

from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest
from server.controlnet_constraints import enforce_controlnet_policy
from server.mode_config import (
    ControlNetControlTypePolicy,
    ControlNetPolicy,
    ModeConfig,
)


def _make_mode(policy: ControlNetPolicy) -> ModeConfig:
    return ModeConfig(
        name="m",
        model="model.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=policy,
    )


def _req(controlnets):
    class R:
        pass
    r = R()
    r.controlnets = controlnets
    return r


def _canny_policy() -> ControlNetPolicy:
    return ControlNetPolicy(
        enabled=True,
        max_attachments=2,
        allow_reuse_emitted_maps=True,
        allowed_control_types={
            "canny": ControlNetControlTypePolicy(
                default_model_id="sdxl-canny",
                allowed_model_ids=["sdxl-canny"],
                allow_preprocess=True,
                default_strength=0.8,
                min_strength=0.0,
                max_strength=1.5,
            )
        },
    )


def test_none_controlnets_is_noop():
    enforce_controlnet_policy(_req(None), _make_mode(ControlNetPolicy()))


def test_empty_list_is_noop():
    enforce_controlnet_policy(_req([]), _make_mode(ControlNetPolicy()))


def test_rejects_when_policy_disabled():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
    )
    with pytest.raises(ValueError, match="does not enable ControlNet"):
        enforce_controlnet_policy(_req([att]), _make_mode(ControlNetPolicy()))


def test_rejects_unknown_control_type():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="pose",
        map_asset_ref="asset_a",
    )
    with pytest.raises(ValueError, match="control_type 'pose'"):
        enforce_controlnet_policy(_req([att]), _make_mode(_canny_policy()))


def test_rejects_model_id_not_in_allowed_list():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
        model_id="rogue-canny",
    )
    with pytest.raises(ValueError, match="model_id 'rogue-canny'"):
        enforce_controlnet_policy(_req([att]), _make_mode(_canny_policy()))


def test_applies_default_model_id_when_omitted():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
    )
    req = _req([att])
    enforce_controlnet_policy(req, _make_mode(_canny_policy()))
    assert req.controlnets[0].model_id == "sdxl-canny"


def test_rejects_strength_outside_policy_bounds():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
        strength=1.8,
    )
    with pytest.raises(ValueError, match="strength"):
        enforce_controlnet_policy(_req([att]), _make_mode(_canny_policy()))


def test_rejects_exceeding_max_attachments():
    atts = [
        ControlNetAttachment(
            attachment_id=f"cn_{i}",
            control_type="canny",
            map_asset_ref=f"asset_{i}",
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="max_attachments"):
        enforce_controlnet_policy(_req(atts), _make_mode(_canny_policy()))


def test_rejects_duplicate_attachment_id():
    atts = [
        ControlNetAttachment(
            attachment_id="cn_dup",
            control_type="canny",
            map_asset_ref="asset_a",
        ),
        ControlNetAttachment(
            attachment_id="cn_dup",
            control_type="canny",
            map_asset_ref="asset_b",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate attachment_id"):
        enforce_controlnet_policy(_req(atts), _make_mode(_canny_policy()))


def test_rejects_preprocess_when_type_policy_forbids():
    policy = _canny_policy()
    policy.allowed_control_types["canny"].allow_preprocess = False
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref="asset_src",
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    with pytest.raises(ValueError, match="preprocessing not allowed"):
        enforce_controlnet_policy(_req([att]), _make_mode(policy))


def test_valid_attachment_passes_through_unchanged():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
        model_id="sdxl-canny",
        strength=1.0,
        start_percent=0.0,
        end_percent=0.75,
    )
    req = _req([att])
    enforce_controlnet_policy(req, _make_mode(_canny_policy()))
    assert req.controlnets[0].model_id == "sdxl-canny"
    assert req.controlnets[0].end_percent == 0.75
