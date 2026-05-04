# tests/test_controlnet_http_contract.py
"""
Route contract tests for ControlNet wiring in lcm_sr_server.py.

All external dependencies (mode config, asset store, preprocessor registry,
worker pool) are mocked explicitly. Tests call the handler directly so they
stay focused on the HTTP status/detail contract instead of full app startup.

Run with: pytest tests/test_controlnet_http_contract.py -v
Skip in unit-only CI: pytest -m "not integration"
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image as PILImage

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


@pytest.mark.integration
def test_http_generate_501_includes_controlnet_artifacts():
    """
    /generate with a valid ControlNet attachment:
    - preprocessing runs and emits an artifact
    - the raised HTTPException includes controlnet_artifacts with correct fields
    """
    from server.asset_store import AssetStore
    from server import lcm_sr_server

    store = AssetStore(byte_budget=64 * 1024 * 1024)
    source_ref = store.insert("upload", _make_png())

    fake_result = ControlMapResult(
        preprocessor_id="canny",
        control_type="canny",
        image_bytes=_make_png(),
        width=8,
        height=8,
    )
    mock_preprocessor = MagicMock()
    mock_preprocessor.run.return_value = fake_result

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_preprocessor

    mock_mode_config = MagicMock()
    mock_mode_config.get_mode.return_value = _make_mode_with_canny()

    mock_runtime = MagicMock(spec=["switch_mode", "get_current_mode", "submit_generate"])
    mock_runtime.get_current_mode.return_value = "sdxl-cn-test"

    original_runtime = getattr(lcm_sr_server.app.state, "generation_runtime", None)
    lcm_sr_server.app.state.generation_runtime = mock_runtime
    try:
        with (
            patch("server.lcm_sr_server.get_mode_config", return_value=mock_mode_config),
            patch("server.asset_store.get_store", return_value=store),
            patch("server.controlnet_preprocessing.DEFAULT_REGISTRY", mock_registry),
        ):
            req = lcm_sr_server.GenerateRequest(
                prompt="a cat",
                controlnets=[{
                    "attachment_id": "cn_1",
                    "control_type": "canny",
                    "source_asset_ref": source_ref,
                    "preprocess": {"id": "canny", "options": {}},
                }],
            )
            with pytest.raises(HTTPException) as excinfo:
                lcm_sr_server.generate(req)
    finally:
        if original_runtime is None:
            del lcm_sr_server.app.state.generation_runtime
        else:
            lcm_sr_server.app.state.generation_runtime = original_runtime

    exc = excinfo.value
    assert exc.status_code == 501
    detail = exc.detail
    assert isinstance(detail, dict), f"expected dict detail, got: {detail!r}"
    assert "controlnet_artifacts" in detail
    arts = detail["controlnet_artifacts"]
    assert len(arts) == 1
    assert arts[0]["attachment_id"] == "cn_1"
    assert arts[0]["control_type"] == "canny"
    assert arts[0]["preprocessor_id"] == "canny"
    assert arts[0]["asset_ref"]


@pytest.mark.integration
def test_http_generate_400_when_controlnet_policy_disabled():
    """
    /generate with controlnets on a mode that has controlnet_policy.enabled=False
    raises 400, not 501, before preprocessing runs.
    """
    from server.asset_store import AssetStore
    from server import lcm_sr_server

    store = AssetStore(byte_budget=64 * 1024 * 1024)
    source_ref = store.insert("upload", _make_png())

    disabled_mode = ModeConfig(
        name="sdxl-plain",
        model="checkpoints/sdxl.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=ControlNetPolicy(enabled=False),
    )

    mock_mode_config = MagicMock()
    mock_mode_config.get_mode.return_value = disabled_mode

    mock_runtime = MagicMock(spec=["switch_mode", "get_current_mode", "submit_generate"])
    mock_runtime.get_current_mode.return_value = "sdxl-plain"

    mock_registry = MagicMock()

    original_runtime = getattr(lcm_sr_server.app.state, "generation_runtime", None)
    lcm_sr_server.app.state.generation_runtime = mock_runtime
    try:
        with (
            patch("server.lcm_sr_server.get_mode_config", return_value=mock_mode_config),
            patch("server.asset_store.get_store", return_value=store),
            patch("server.controlnet_preprocessing.DEFAULT_REGISTRY", mock_registry),
        ):
            req = lcm_sr_server.GenerateRequest(
                prompt="a cat",
                controlnets=[{
                    "attachment_id": "cn_1",
                    "control_type": "canny",
                    "source_asset_ref": source_ref,
                    "preprocess": {"id": "canny", "options": {}},
                    "model_id": "sdxl-canny",
                }],
            )
            with pytest.raises(HTTPException) as excinfo:
                lcm_sr_server.generate(req)
    finally:
        if original_runtime is None:
            del lcm_sr_server.app.state.generation_runtime
        else:
            lcm_sr_server.app.state.generation_runtime = original_runtime

    exc = excinfo.value
    assert exc.status_code == 400
    assert "does not enable ControlNet" in exc.detail
    mock_registry.get.assert_not_called()
