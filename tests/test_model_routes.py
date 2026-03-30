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

    assert sd15["negative_prompt_templates"] == {}
    assert sd15["default_negative_prompt_template"] is None
    assert sd15["allow_custom_negative_prompt"] is False
    assert sd15["allowed_scheduler_ids"] is None
    assert sd15["default_scheduler_id"] is None
