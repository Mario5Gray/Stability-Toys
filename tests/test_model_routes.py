"""
Unit tests for model route serialization.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class _FakeAPIRouter:
    def __init__(self, *args, **kwargs):
        pass

    def _decorator(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    get = _decorator
    post = _decorator
    put = _decorator
    delete = _decorator
    patch = _decorator


class _FakeHTTPException(Exception):
    def __init__(self, status_code=None, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


sys.modules.setdefault(
    "fastapi",
    SimpleNamespace(APIRouter=_FakeAPIRouter, HTTPException=_FakeHTTPException),
)
sys.modules.setdefault(
    "backends.model_registry",
    SimpleNamespace(get_model_registry=lambda: None),
)
sys.modules.setdefault(
    "backends.worker_pool",
    SimpleNamespace(get_worker_pool=lambda: None),
)

from server.model_routes import router
from server import model_routes


async def test_models_status_includes_backend_version():
    pool = Mock()
    pool.get_current_mode.return_value = "SDXL"
    pool.is_model_loaded.return_value = True
    pool.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {"allocated_gb": 1.5, "reserved_gb": 2.0}

    with patch("server.model_routes.get_worker_pool", return_value=pool), \
            patch("server.model_routes.get_model_registry", return_value=registry), \
            patch.dict(os.environ, {"BACKEND_VERSION": "abc1234"}, clear=False):
        data = await model_routes.get_models_status()

    assert data["backend_version"] == "abc1234"


async def test_models_status_defaults_backend_version_to_dev_when_unset():
    pool = Mock()
    pool.get_current_mode.return_value = None
    pool.is_model_loaded.return_value = False
    pool.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {}

    with patch("server.model_routes.get_worker_pool", return_value=pool), \
            patch("server.model_routes.get_model_registry", return_value=registry), \
            patch.dict(os.environ, {}, clear=True):
        data = await model_routes.get_models_status()

    assert data["backend_version"] == "dev"


async def test_list_modes_includes_generation_control_policy_fields():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sdxl",
        "modes": {
            "sdxl": {
                "model": "checkpoints/sdxl/model.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 20,
                "default_guidance": 7.0,
                "loader_format": "single_file",
                "checkpoint_precision": "fp8",
                "checkpoint_variant": "sdxl-base",
                "scheduler_profile": "native",
                "recommended_size": "512x512",
                "runtime_quantize": "none",
                "runtime_offload": "model",
                "runtime_attention_slicing": True,
                "runtime_enable_xformers": True,
                "negative_prompt_templates": {"safe_photo": "blurry, watermark"},
                "default_negative_prompt_template": "safe_photo",
                "allow_custom_negative_prompt": True,
                "allowed_scheduler_ids": ["euler", "dpmpp_2m"],
                "default_scheduler_id": "euler",
            },
            "sd15": {
                "model": "checkpoints/sd15/model.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 20,
                "default_guidance": 7.0,
                "loader_format": None,
                "checkpoint_precision": None,
                "checkpoint_variant": None,
                "scheduler_profile": None,
                "recommended_size": None,
                "runtime_quantize": None,
                "runtime_offload": None,
                "runtime_attention_slicing": None,
                "runtime_enable_xformers": None,
                "negative_prompt_templates": {},
                "default_negative_prompt_template": None,
                "allow_custom_negative_prompt": False,
                "allowed_scheduler_ids": None,
                "default_scheduler_id": None,
            },
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    sdxl = data["modes"]["sdxl"]
    sd15 = data["modes"]["sd15"]

    assert sdxl["negative_prompt_templates"] == {"safe_photo": "blurry, watermark"}
    assert sdxl["default_negative_prompt_template"] == "safe_photo"
    assert sdxl["allow_custom_negative_prompt"] is True
    assert sdxl["allowed_scheduler_ids"] == ["euler", "dpmpp_2m"]
    assert sdxl["default_scheduler_id"] == "euler"
    assert sdxl["runtime_quantize"] == "none"
    assert sdxl["runtime_offload"] == "model"
    assert sdxl["runtime_attention_slicing"] is True
    assert sdxl["runtime_enable_xformers"] is True

    assert sd15["negative_prompt_templates"] == {}
    assert sd15["default_negative_prompt_template"] is None
    assert sd15["allow_custom_negative_prompt"] is False
    assert sd15["allowed_scheduler_ids"] is None
    assert sd15["default_scheduler_id"] is None
    assert sd15["runtime_quantize"] is None
    assert sd15["runtime_offload"] is None
    assert sd15["runtime_attention_slicing"] is None
    assert sd15["runtime_enable_xformers"] is None


async def test_reload_and_free_vram_routes_call_pool_methods():
    pool = Mock()
    pool.reload_current_mode.return_value = {
        "status": "reloaded",
        "mode": "sdxl-general",
    }
    pool.unload_current_model.return_value = {
        "status": "unloaded",
        "is_loaded": False,
        "current_mode": "sdxl-general",
        "queue_size": 0,
        "vram": {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "total_bytes": 8 * 1024**3,
        },
    }
    pool.free_vram.return_value = {
        "status": "ok",
        "is_loaded": False,
        "current_mode": "sdxl-general",
        "vram": {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "total_bytes": 8 * 1024**3,
        },
    }

    with patch("server.model_routes.get_worker_pool", return_value=pool):
        assert (await model_routes.reload_current_model())["status"] == "reloaded"
        assert (await model_routes.unload_current_model())["status"] == "unloaded"
        assert (await model_routes.free_vram())["status"] == "ok"

    pool.unload_current_model.assert_called_once()
    pool.free_vram.assert_called_once_with(reason="manual_free_vram")


async def test_cancel_job_route_calls_worker_pool():
    pool = Mock()
    pool.cancel_job.return_value = True

    with patch("server.model_routes.get_worker_pool", return_value=pool):
        result = await model_routes.cancel_job("abc123")

    assert result == {"job_id": "abc123", "status": "canceled"}
    pool.cancel_job.assert_called_once_with("abc123")


async def test_cancel_job_route_reports_not_found_when_pool_cannot_cancel():
    pool = Mock()
    pool.cancel_job.return_value = False

    with patch("server.model_routes.get_worker_pool", return_value=pool):
        result = await model_routes.cancel_job("missing")

    assert result == {"job_id": "missing", "status": "not_found"}
    pool.cancel_job.assert_called_once_with("missing")
