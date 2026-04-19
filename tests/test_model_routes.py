"""
Unit tests for model route serialization.
"""

import os
import sys
from unittest.mock import Mock, patch
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import model_routes


def _status_request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(generation_runtime=None)))


async def test_models_status_includes_backend_version():
    runtime = Mock()
    runtime.get_current_mode.return_value = "SDXL"
    runtime.is_model_loaded.return_value = True
    runtime.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {"allocated_gb": 1.5, "reserved_gb": 2.0}

    provider = Mock()
    provider.backend_id = "cuda"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=True,
        supports_modes=True,
        supports_superres=True,
        supports_model_registry_stats=True,
        supports_img2img=True,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry), \
            patch.dict(os.environ, {"BACKEND_VERSION": "abc1234"}, clear=False):
        data = await model_routes.get_models_status(_status_request())

    assert data["backend_version"] == "abc1234"


async def test_models_status_defaults_backend_version_to_dev_when_empty():
    runtime = Mock()
    runtime.get_current_mode.return_value = None
    runtime.is_model_loaded.return_value = False
    runtime.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {}

    provider = Mock()
    provider.backend_id = "cuda"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=True,
        supports_modes=True,
        supports_superres=True,
        supports_model_registry_stats=True,
        supports_img2img=True,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry), \
            patch.dict(os.environ, {"BACKEND_VERSION": ""}, clear=False):
        data = await model_routes.get_models_status(_status_request())

    assert data["backend_version"] == "dev"


async def test_models_status_defaults_backend_version_to_dev_when_unset():
    runtime = Mock()
    runtime.get_current_mode.return_value = None
    runtime.is_model_loaded.return_value = False
    runtime.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {}

    provider = Mock()
    provider.backend_id = "cuda"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=True,
        supports_modes=True,
        supports_superres=True,
        supports_model_registry_stats=True,
        supports_img2img=True,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry), \
            patch.dict(os.environ, {}, clear=True):
        data = await model_routes.get_models_status(_status_request())

    assert data["backend_version"] == "dev"


async def test_models_status_uses_provider_capabilities_and_registry_stats():
    runtime = Mock()
    runtime.get_current_mode.return_value = None
    runtime.is_model_loaded.return_value = False
    runtime.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {
        "backend": "cpu",
        "device": "CPU placeholder",
        "models_loaded": 0,
    }

    provider = Mock()
    provider.backend_id = "cpu"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=False,
        supports_modes=True,
        supports_superres=False,
        supports_model_registry_stats=False,
        supports_img2img=False,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry):
        data = await model_routes.get_models_status(_status_request())

    assert data["backend"] == "cpu"
    assert data["capabilities"]["supports_generation"] is False
    assert data["vram"] == {
        "backend": "cpu",
        "device": "CPU placeholder",
        "models_loaded": 0,
    }


async def test_list_modes_includes_generation_control_policy_fields_and_resolution_sets():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sdxl",
        "resolution_sets": {
            "default": [
                {"size": "512x512", "aspect_ratio": "1:1"},
                {"size": "512x768", "aspect_ratio": "2:3"},
            ],
            "sdxl": [
                {"size": "1024x1024", "aspect_ratio": "1:1"},
                {"size": "896x1152", "aspect_ratio": "7:9"},
            ],
        },
        "modes": {
            "sdxl": {
                "model": "checkpoints/sdxl/model.safetensors",
                "loras": [],
                "default_size": "1024x1024",
                "default_steps": 20,
                "default_guidance": 7.0,
                "loader_format": "single_file",
                "checkpoint_precision": "fp8",
                "checkpoint_variant": "sdxl-base",
                "scheduler_profile": "native",
                "recommended_size": "896x1152",
                "runtime_quantize": "none",
                "runtime_offload": "model",
                "runtime_attention_slicing": True,
                "runtime_enable_xformers": True,
                "resolution_set": "sdxl",
                "resolution_options": [
                    {"size": "1024x1024", "aspect_ratio": "1:1"},
                    {"size": "896x1152", "aspect_ratio": "7:9"},
                ],
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
                "resolution_set": "default",
                "resolution_options": [
                    {"size": "512x512", "aspect_ratio": "1:1"},
                    {"size": "512x768", "aspect_ratio": "2:3"},
                ],
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

    assert data["resolution_sets"] == {
        "default": [
            {"size": "512x512", "aspect_ratio": "1:1"},
            {"size": "512x768", "aspect_ratio": "2:3"},
        ],
        "sdxl": [
            {"size": "1024x1024", "aspect_ratio": "1:1"},
            {"size": "896x1152", "aspect_ratio": "7:9"},
        ],
    }
    assert sdxl["negative_prompt_templates"] == {"safe_photo": "blurry, watermark"}
    assert sdxl["default_negative_prompt_template"] == "safe_photo"
    assert sdxl["allow_custom_negative_prompt"] is True
    assert sdxl["allowed_scheduler_ids"] == ["euler", "dpmpp_2m"]
    assert sdxl["default_scheduler_id"] == "euler"
    assert sdxl["runtime_quantize"] == "none"
    assert sdxl["runtime_offload"] == "model"
    assert sdxl["runtime_attention_slicing"] is True
    assert sdxl["runtime_enable_xformers"] is True
    assert sdxl["resolution_set"] == "sdxl"
    assert sdxl["resolution_options"] == [
        {"size": "1024x1024", "aspect_ratio": "1:1"},
        {"size": "896x1152", "aspect_ratio": "7:9"},
    ]
    assert sdxl["default_size"] in {option["size"] for option in sdxl["resolution_options"]}
    assert sdxl["recommended_size"] in {option["size"] for option in sdxl["resolution_options"]}

    assert sd15["negative_prompt_templates"] == {}
    assert sd15["default_negative_prompt_template"] is None
    assert sd15["allow_custom_negative_prompt"] is False
    assert sd15["allowed_scheduler_ids"] is None
    assert sd15["default_scheduler_id"] is None
    assert sd15["runtime_quantize"] is None
    assert sd15["runtime_offload"] is None
    assert sd15["runtime_attention_slicing"] is None
    assert sd15["runtime_enable_xformers"] is None
    assert sd15["resolution_set"] == "default"
    assert sd15["resolution_options"] == [
        {"size": "512x512", "aspect_ratio": "1:1"},
        {"size": "512x768", "aspect_ratio": "2:3"},
    ]


async def test_list_modes_includes_maximum_len():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sdxl",
        "resolution_sets": {
            "default": [
                {"size": "512x512", "aspect_ratio": "1:1"},
            ],
        },
        "modes": {
            "sdxl": {
                "model": "checkpoints/sdxl/model.safetensors",
                "loras": [],
                "default_size": "1024x1024",
                "default_steps": 20,
                "default_guidance": 7.0,
                "loader_format": "single_file",
                "checkpoint_precision": "fp8",
                "checkpoint_variant": "sdxl-base",
                "scheduler_profile": "native",
                "recommended_size": "896x1152",
                "runtime_quantize": "none",
                "runtime_offload": "model",
                "runtime_attention_slicing": True,
                "runtime_enable_xformers": True,
                "resolution_set": "default",
                "resolution_options": [
                    {"size": "1024x1024", "aspect_ratio": "1:1"},
                ],
                "negative_prompt_templates": {"safe_photo": "blurry, watermark"},
                "default_negative_prompt_template": "safe_photo",
                "allow_custom_negative_prompt": True,
                "allowed_scheduler_ids": ["euler", "dpmpp_2m"],
                "default_scheduler_id": "euler",
                "maximum_len": 240,
            },
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    assert data["modes"]["sdxl"]["maximum_len"] == 240


async def test_list_modes_includes_chat_enabled_flag():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sdxl-chat",
        "chat": {
            "sdxl-chat": {
                "endpoint": "http://localhost:11434/v1",
                "model": "llama3.2",
            },
        },
        "resolution_sets": {
            "default": [{"size": "512x512", "aspect_ratio": "1:1"}],
        },
        "modes": {
            "sdxl-chat": {
                "model": "checkpoints/sdxl/model.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 20,
                "default_guidance": 7.0,
                "resolution_set": "default",
                "resolution_options": [{"size": "512x512", "aspect_ratio": "1:1"}],
                "negative_prompt_templates": {},
                "default_negative_prompt_template": None,
                "allow_custom_negative_prompt": False,
                "allowed_scheduler_ids": None,
                "default_scheduler_id": None,
            },
            "sd15": {
                "model": "checkpoints/sd15/model.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 20,
                "default_guidance": 7.0,
                "resolution_set": "default",
                "resolution_options": [{"size": "512x512", "aspect_ratio": "1:1"}],
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

    assert data["modes"]["sdxl-chat"]["chat_enabled"] is True
    assert data["modes"]["sd15"]["chat_enabled"] is False


async def test_save_all_modes_passes_resolution_sets_to_save_config():
    config = Mock()
    config.to_dict.return_value = {"chat": {}}
    pool = Mock()
    pool.get_current_mode.return_value = None
    request = model_routes.ModesBulkSaveRequest.model_validate({
        "model_root": "/models",
        "lora_root": "/loras",
        "default_mode": "sdxl",
        "resolution_sets": {
            "default": [{"size": "512x512", "aspect_ratio": "1:1"}],
            "sdxl": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
        },
        "modes": {
            "sdxl": {
                "model": "checkpoints/sdxl/model.safetensors",
                "loras": [],
                "default_size": "1024x1024",
                "default_steps": 24,
                "default_guidance": 6.5,
                "resolution_set": "sdxl",
            },
        },
    })

    with patch("server.model_routes.get_mode_config", return_value=config), \
            patch("server.model_routes.get_worker_pool", return_value=pool):
        await model_routes.save_all_modes(request)

    saved_payload = config.save_config.call_args.args[0]
    assert saved_payload["resolution_sets"] == {
        "default": [{"size": "512x512", "aspect_ratio": "1:1"}],
        "sdxl": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
    }


async def test_save_all_modes_prunes_stale_chat_bindings_when_chat_omitted():
    config = Mock()
    config.to_dict.return_value = {
        "chat": {
            "sdxl": {"endpoint": "http://localhost:11434/v1", "model": "llama3.2"},
            "sd15": {"endpoint": "http://localhost:11434/v1", "model": "llama3.2"},
        }
    }
    pool = Mock()
    pool.get_current_mode.return_value = None
    request = model_routes.ModesBulkSaveRequest.model_validate({
        "model_root": "/models",
        "lora_root": "/loras",
        "default_mode": "sdxl",
        "resolution_sets": {
            "default": [{"size": "512x512", "aspect_ratio": "1:1"}],
        },
        "modes": {
            "sdxl": {
                "model": "checkpoints/sdxl/model.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 24,
                "default_guidance": 6.5,
            },
        },
    })

    with patch("server.model_routes.get_mode_config", return_value=config), \
            patch("server.model_routes.get_worker_pool", return_value=pool):
        await model_routes.save_all_modes(request)

    saved_payload = config.save_config.call_args.args[0]
    assert saved_payload["chat"] == {
        "sdxl": {"endpoint": "http://localhost:11434/v1", "model": "llama3.2"}
    }


async def test_create_or_update_mode_preserves_existing_resolution_and_policy_fields():
    config = Mock()
    config.to_dict.return_value = {
        "model_root": "/models",
        "lora_root": "/loras",
        "default_mode": "sdxl",
        "resolution_sets": {
            "sdxl": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
        },
        "modes": {
            "sdxl": {
                "model": "checkpoints/old/model.safetensors",
                "loras": [{"path": "old/style.safetensors", "strength": 1.0}],
                "default_size": "1024x1024",
                "default_steps": 20,
                "default_guidance": 7.0,
                "resolution_set": "sdxl",
                "resolution_options": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
                "loader_format": "single_file",
                "checkpoint_precision": "fp16",
                "scheduler_profile": "native",
                "negative_prompt_templates": {"safe_photo": "blurry, watermark"},
                "default_negative_prompt_template": "safe_photo",
                "allow_custom_negative_prompt": True,
                "allowed_scheduler_ids": ["euler"],
                "default_scheduler_id": "euler",
            }
        },
    }
    pool = Mock()
    pool.reload_if_current.return_value = False
    request = model_routes.ModeCreateRequest(
        model="checkpoints/new/model.safetensors",
        loras=[{"path": "new/style.safetensors", "strength": 0.8}],
        default_size="1024x1024",
        default_steps=28,
        default_guidance=5.5,
    )

    with patch("server.model_routes.get_mode_config", return_value=config), \
            patch("server.model_routes.get_worker_pool", return_value=pool):
        await model_routes.create_or_update_mode("sdxl", request)

    saved_mode = config.save_config.call_args.args[0]["modes"]["sdxl"]
    assert saved_mode["model"] == "checkpoints/new/model.safetensors"
    assert saved_mode["loras"] == [{"path": "new/style.safetensors", "strength": 0.8}]
    assert saved_mode["default_steps"] == 28
    assert saved_mode["default_guidance"] == 5.5
    assert saved_mode["resolution_set"] == "sdxl"
    assert saved_mode["loader_format"] == "single_file"
    assert saved_mode["checkpoint_precision"] == "fp16"
    assert saved_mode["scheduler_profile"] == "native"
    assert saved_mode["negative_prompt_templates"] == {"safe_photo": "blurry, watermark"}
    assert saved_mode["default_negative_prompt_template"] == "safe_photo"
    assert saved_mode["allow_custom_negative_prompt"] is True
    assert saved_mode["allowed_scheduler_ids"] == ["euler"]
    assert saved_mode["default_scheduler_id"] == "euler"


async def test_delete_mode_prunes_chat_config_for_deleted_mode():
    config = Mock()
    config.to_dict.return_value = {
        "model_root": "/models",
        "lora_root": "/loras",
        "default_mode": "sdxl",
        "resolution_sets": {"default": [{"size": "512x512", "aspect_ratio": "1:1"}]},
        "chat": {
            "sdxl": {"endpoint": "http://localhost:11434/v1", "model": "llama3.2"},
            "sd15": {"endpoint": "http://localhost:11434/v1", "model": "llama3.2"},
        },
        "modes": {
            "sdxl": {"model": "checkpoints/sdxl/model.safetensors"},
            "sd15": {"model": "checkpoints/sd15/model.safetensors"},
        },
    }
    pool = Mock()
    pool.get_current_mode.return_value = "sdxl"

    with patch("server.model_routes.get_mode_config", return_value=config), \
            patch("server.model_routes.get_worker_pool", return_value=pool):
        result = await model_routes.delete_mode("sd15")

    saved_payload = config.save_config.call_args.args[0]
    assert "sd15" not in saved_payload["modes"]
    assert saved_payload["chat"] == {
        "sdxl": {"endpoint": "http://localhost:11434/v1", "model": "llama3.2"}
    }
    assert result["status"] == "deleted"
    assert result["switched_to"] is None


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


async def test_list_modes_includes_controlnet_policy():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sdxl-cn",
        "resolution_sets": {"default": [{"size": "1024x1024", "aspect_ratio": "1:1"}]},
        "chat_connections": {},
        "chat_delegates": {},
        "modes": {
            "sdxl-cn": {
                "model": "checkpoints/sdxl.safetensors",
                "model_path": "/models/checkpoints/sdxl.safetensors",
                "loras": [],
                "default_size": "1024x1024",
                "default_steps": 20,
                "default_guidance": 7.0,
                "maximum_len": None,
                "loader_format": None,
                "checkpoint_precision": None,
                "checkpoint_variant": None,
                "scheduler_profile": None,
                "recommended_size": None,
                "runtime_quantize": None,
                "runtime_offload": None,
                "runtime_attention_slicing": None,
                "runtime_enable_xformers": None,
                "resolution_set": "default",
                "resolution_options": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
                "negative_prompt_templates": {},
                "default_negative_prompt_template": None,
                "allow_custom_negative_prompt": False,
                "allowed_scheduler_ids": None,
                "default_scheduler_id": None,
                "chat_delegate": None,
                "metadata": {},
                "controlnet_policy": {
                    "enabled": True,
                    "max_attachments": 2,
                    "allow_reuse_emitted_maps": True,
                    "allowed_control_types": {
                        "canny": {
                            "default_model_id": "sdxl-canny",
                            "allowed_model_ids": ["sdxl-canny"],
                            "allow_preprocess": True,
                            "default_strength": 0.8,
                            "min_strength": 0.0,
                            "max_strength": 2.0,
                        }
                    },
                },
            },
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    policy = data["modes"]["sdxl-cn"]["controlnet_policy"]
    assert policy["enabled"] is True
    assert policy["allowed_control_types"]["canny"]["default_model_id"] == "sdxl-canny"


async def test_list_modes_controlnet_policy_defaults_when_absent():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sd15",
        "resolution_sets": {"default": [{"size": "512x512", "aspect_ratio": "1:1"}]},
        "chat_connections": {},
        "chat_delegates": {},
        "modes": {
            "sd15": {
                "model": "checkpoints/sd15.safetensors",
                "model_path": "/models/checkpoints/sd15.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 20,
                "default_guidance": 7.0,
                "maximum_len": None,
                "loader_format": None,
                "checkpoint_precision": None,
                "checkpoint_variant": None,
                "scheduler_profile": None,
                "recommended_size": None,
                "runtime_quantize": None,
                "runtime_offload": None,
                "runtime_attention_slicing": None,
                "runtime_enable_xformers": None,
                "resolution_set": "default",
                "resolution_options": [{"size": "512x512", "aspect_ratio": "1:1"}],
                "negative_prompt_templates": {},
                "default_negative_prompt_template": None,
                "allow_custom_negative_prompt": False,
                "allowed_scheduler_ids": None,
                "default_scheduler_id": None,
                "chat_delegate": None,
                "metadata": {},
                # no controlnet_policy key
            },
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    policy = data["modes"]["sd15"]["controlnet_policy"]
    assert policy["enabled"] is False
    assert policy["allowed_control_types"] == {}
