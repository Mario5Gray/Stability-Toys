import pytest


def test_ws_build_rejects_disabled_mode_controlnet():
    from server.controlnet_constraints import enforce_controlnet_policy
    from server.controlnet_models import ControlNetAttachment
    from server.mode_config import ControlNetPolicy, ModeConfig

    class R:
        pass
    req = R()
    req.controlnets = [
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
        )
    ]
    mode = ModeConfig(
        name="m",
        model="x.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=ControlNetPolicy(enabled=False),
    )
    with pytest.raises(ValueError, match="does not enable ControlNet"):
        enforce_controlnet_policy(req, mode)
