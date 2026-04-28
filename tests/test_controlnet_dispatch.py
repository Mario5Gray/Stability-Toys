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


async def test_ws_handle_job_submit_rejects_controlnets_when_mode_system_has_no_current_mode():
    """Mode-system WS pre-submit path must still 501-stub when no model is loaded."""
    import server.ws_routes as ws

    mock_ws = MagicMock()
    mock_state = MagicMock()
    mock_state.use_mode_system = True
    mock_state.worker_pool.get_current_mode.return_value = None
    mock_ws.app.state = mock_state

    sent_messages = []

    async def fake_send(client_id, msg):
        sent_messages.append(msg)

    with patch.object(ws.hub, "send", side_effect=fake_send):
        with patch("server.ws_routes._get_app_state", return_value=mock_state):
            await ws.handle_job_submit(
                mock_ws,
                {
                    "id": "corr1",
                    "jobType": "generate",
                    "params": {
                        "prompt": "a cat",
                        "controlnets": [
                            {
                                "attachment_id": "cn_1",
                                "control_type": "canny",
                                "map_asset_ref": "asset_a",
                            }
                        ],
                    },
                },
                "client1",
            )

    errors = [m for m in sent_messages if m.get("type") == "job:error"]
    assert errors, "expected a job:error message"
    assert "ControlNet provider not yet implemented" in errors[0]["error"]


async def test_ws_handle_job_submit_rejects_controlnets_on_mode_system_with_current_mode():
    """Mode-system WS pre-submit path should return the stub error, not TypeError."""
    import server.ws_routes as ws
    from server.mode_config import ControlNetControlTypePolicy, ControlNetPolicy

    mock_ws = MagicMock()
    mock_state = MagicMock()
    mock_state.use_mode_system = True
    mock_state.worker_pool.get_current_mode.return_value = "sdxl-general"
    mock_ws.app.state = mock_state

    sent_messages = []

    async def fake_send(client_id, msg):
        sent_messages.append(msg)

    policy = ControlNetPolicy(
        enabled=True,
        max_attachments=1,
        allow_reuse_emitted_maps=True,
        allowed_control_types={
            "canny": ControlNetControlTypePolicy(
                default_model_id="sdxl-canny",
                allowed_model_ids=["sdxl-canny"],
                allow_preprocess=True,
                default_strength=0.8,
                min_strength=0.0,
                max_strength=2.0,
            )
        },
    )

    with patch.object(ws.hub, "send", side_effect=fake_send):
        with patch("server.ws_routes._get_app_state", return_value=mock_state):
            with patch("server.ws_routes.get_mode_config") as get_mode_config:
                get_mode_config.return_value = MagicMock(
                    get_mode=MagicMock(
                        return_value=MagicMock(
                            name="sdxl-general",
                            default_size="1024x1024",
                            default_steps=30,
                            default_guidance=7.0,
                            resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
                            controlnet_policy=policy,
                        )
                    )
                )
                await ws.handle_job_submit(
                    mock_ws,
                    {
                        "id": "corr2",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "controlnets": [
                                {
                                    "attachment_id": "cn_1",
                                    "control_type": "canny",
                                    "map_asset_ref": "asset_a",
                                }
                            ],
                        },
                    },
                    "client1",
                )

    errors = [m for m in sent_messages if m.get("type") == "job:error"]
    assert errors, "expected a job:error message"
    assert "ControlNet provider not yet implemented" in errors[0]["error"]
    assert "missing 1 required keyword-only argument" not in errors[0]["error"]
