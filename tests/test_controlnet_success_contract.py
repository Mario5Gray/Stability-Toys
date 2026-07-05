import concurrent.futures
import asyncio
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image as PILImage

from server.controlnet_models import ControlNetArtifactRef
from server.controlnet_preprocessors import ControlMapResult
from server.mode_config import (
    ControlNetControlTypePolicy,
    ControlNetPolicy,
    ModeConfig,
)


def _make_png(w: int = 8, h: int = 8) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _valid_controlnet_request(source_ref: str) -> dict:
    return {
        "prompt": "a cat",
        "controlnets": [
            {
                "attachment_id": "cn_1",
                "control_type": "canny",
                "source_asset_ref": source_ref,
                "preprocess": {"id": "canny", "options": {}},
            }
        ],
    }


def _make_mode_with_canny() -> ModeConfig:
    return ModeConfig(
        name="sdxl-cn-test",
        model="checkpoints/sdxl.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=ControlNetPolicy(
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
        ),
    )


def _make_registry(preprocessor_id: str = "canny") -> MagicMock:
    fake_result = ControlMapResult(
        preprocessor_id=preprocessor_id,
        control_type=preprocessor_id,
        image_bytes=_make_png(),
        width=8,
        height=8,
    )
    mock_preprocessor = MagicMock()
    mock_preprocessor.run.return_value = fake_result

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_preprocessor
    return mock_registry


def _controlnet_provider() -> SimpleNamespace:
    return SimpleNamespace(capabilities=lambda: SimpleNamespace(supports_controlnet=True))


def test_http_generate_success_exposes_controlnet_artifacts_header():
    from server.asset_store import InMemoryAssetStore
    from server import lcm_sr_server

    store = InMemoryAssetStore()
    source_ref = store.write("upload", _make_png())

    fut = concurrent.futures.Future()
    fut.set_result((b"png-bytes", 1234))

    mock_runtime = MagicMock(spec=["switch_mode", "get_current_mode", "submit_generate"])
    mock_runtime.get_current_mode.return_value = "sdxl-cn-test"
    mock_runtime.submit_generate.return_value = fut

    mock_mode_config = MagicMock()
    mock_mode_config.get_mode.return_value = _make_mode_with_canny()

    original_runtime = getattr(lcm_sr_server.app.state, "generation_runtime", None)
    original_provider = getattr(lcm_sr_server.app.state, "backend_provider", None)
    original_storage = getattr(lcm_sr_server.app.state, "storage", None)
    original_sr = getattr(lcm_sr_server.app.state, "sr_service", None)
    lcm_sr_server.app.state.generation_runtime = mock_runtime
    lcm_sr_server.app.state.backend_provider = _controlnet_provider()
    lcm_sr_server.app.state.storage = None
    lcm_sr_server.app.state.sr_service = None
    try:
        with (
            patch("server.lcm_sr_server.get_mode_config", return_value=mock_mode_config),
            patch("server.asset_store.get_store", return_value=store),
            patch("server.controlnet_preprocessing.DEFAULT_REGISTRY", _make_registry()),
        ):
            req = lcm_sr_server.GenerateRequest(**_valid_controlnet_request(source_ref))
            resp = lcm_sr_server.generate(req)
    finally:
        if original_runtime is None:
            del lcm_sr_server.app.state.generation_runtime
        else:
            lcm_sr_server.app.state.generation_runtime = original_runtime
        lcm_sr_server.app.state.backend_provider = original_provider
        lcm_sr_server.app.state.storage = original_storage
        lcm_sr_server.app.state.sr_service = original_sr

    assert resp.headers["X-Seed"] == "1234"
    assert resp.headers["X-ControlNet-Artifacts"]


def test_ws_job_complete_includes_controlnet_artifacts():
    import server.ws_routes as ws_routes
    from server import lcm_sr_server

    fut = concurrent.futures.Future()
    fut.set_result((b"png-bytes", 1234))

    mock_ws = MagicMock()
    mock_ws.app.state = SimpleNamespace(
        storage=None,
        sr_service=None,
    )
    sent: list[dict] = []

    async def _fake_send(client_id: str, msg: dict) -> None:
        sent.append(msg)

    async def _exercise() -> None:
        req = lcm_sr_server.GenerateRequest(prompt="a cat")
        req.controlnets = [
            lcm_sr_server.ControlNetAttachment(
                attachment_id="cn_1",
                control_type="canny",
                map_asset_ref="artifact-ref-1",
            )
        ]
        req._controlnet_artifacts = [
            ControlNetArtifactRef(
                attachment_id="cn_1",
                asset_ref="artifact-ref-1",
                control_type="canny",
                preprocessor_id="canny",
                source_asset_ref="source-ref-1",
            )
        ]
        fake_loop = SimpleNamespace(
            run_in_executor=MagicMock(return_value=asyncio.sleep(0, result=(b"png-bytes", 1234)))
        )
        with (
            patch.object(ws_routes.hub, "send", side_effect=_fake_send),
            patch("server.ws_routes.asyncio.get_running_loop", return_value=fake_loop),
        ):
            await ws_routes._finish_generate(mock_ws, "client-1", "job-1", req, fut)

    asyncio.run(_exercise())

    complete = sent[0]
    assert complete["type"] == "job:complete"
    assert complete["jobId"] == "job-1"
    assert complete["controlnet_artifacts"][0]["attachment_id"] == "cn_1"
