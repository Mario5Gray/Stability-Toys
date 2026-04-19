import pytest
from unittest.mock import patch, MagicMock


def test_http_generate_rejects_controlnets_when_no_current_mode():
    """Dispatch stub fires even when current_mode is None (e.g. RKNN backend)."""
    from server.controlnet_models import ControlNetAttachment
    import server.lcm_sr_server as srv
    from fastapi import HTTPException

    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
    )

    mock_runtime = MagicMock()
    mock_runtime.get_current_mode.return_value = None  # RKNN returns None
    mock_app = MagicMock()
    mock_app.state.generation_runtime = mock_runtime

    req = srv.GenerateRequest(prompt="a cat", controlnets=[att])

    with patch.object(srv, "app", mock_app):
        with pytest.raises(HTTPException) as exc_info:
            srv.generate(req)
    assert exc_info.value.status_code == 501
    assert "ControlNet provider not yet implemented" in exc_info.value.detail


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


async def test_ws_run_generate_rejects_controlnets_on_non_mode_system():
    """_run_generate (non-mode-system/RKNN WS path) must 501-stub controlnets."""
    from server.controlnet_models import ControlNetAttachment
    import server.ws_routes as ws

    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
    )

    mock_ws = MagicMock()
    mock_state = MagicMock()
    mock_state.use_mode_system = False
    mock_ws.app.state = mock_state

    sent_messages = []

    async def fake_send(client_id, msg):
        sent_messages.append(msg)

    with patch.object(ws.hub, "send", side_effect=fake_send):
        with patch("server.ws_routes._get_app_state", return_value=mock_state):
            await ws._run_generate(mock_ws, "client1", "job1", {
                "prompt": "a cat",
                "controlnets": [att.model_dump()],
            })

    errors = [m for m in sent_messages if m.get("type") == "job:error"]
    assert errors, "expected a job:error message"
    assert "ControlNet provider not yet implemented" in errors[0]["error"]
