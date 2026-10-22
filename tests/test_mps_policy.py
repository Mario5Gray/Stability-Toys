"""Strict MPS execution policy tests."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from backends.platforms.mps_policy import MPS_TORCH_POLICY
from backends.torch_policy import TorchExecutionPolicy


@pytest.fixture
def mps_ready():
    clean_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("CUDA_")
    }
    with (
        patch.dict(os.environ, clean_env, clear=True),
        patch("torch.backends.mps.is_built", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        yield


def test_mps_policy_satisfies_contract_and_uses_float16(mps_ready):
    assert isinstance(MPS_TORCH_POLICY, TorchExecutionPolicy)
    assert MPS_TORCH_POLICY.backend_id == "mps"

    settings = MPS_TORCH_POLICY.resolve_settings(None)

    assert settings.device == "mps"
    assert settings.dtype is torch.float16
    assert settings.dtype_str == "fp16"
    assert settings.enable_xformers is False
    assert settings.quantize == "none"
    assert settings.offload == "none"


@pytest.mark.parametrize("missing", ["is_built", "is_available"])
def test_mps_policy_requires_runtime(mps_ready, missing):
    with patch(f"torch.backends.mps.{missing}", return_value=False):
        with pytest.raises(RuntimeError, match="MPS"):
            MPS_TORCH_POLICY.resolve_settings(None)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CUDA_OFFLOAD", "sequential", "offload"),
        ("CUDA_OFFLOAD", "model", "offload"),
        ("CUDA_QUANTIZE", "fp8", "quantize"),
        ("CUDA_ENABLE_XFORMERS", "1", "xformers"),
    ],
)
def test_mps_policy_rejects_cuda_env_option(mps_ready, name, value, message):
    with patch.dict(os.environ, {name: value}):
        with pytest.raises(ValueError, match=message):
            MPS_TORCH_POLICY.resolve_settings(None)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("runtime_offload", "model", "offload"),
        ("runtime_offload", "sequential", "offload"),
        ("runtime_quantize", "fp8", "quantize"),
        ("runtime_enable_xformers", True, "xformers"),
    ],
)
def test_mps_policy_rejects_model_override(mps_ready, name, value, message):
    with pytest.raises(ValueError, match=message):
        MPS_TORCH_POLICY.resolve_settings(SimpleNamespace(**{name: value}))


@pytest.mark.parametrize(
    ("env_name", "env_value", "override_name", "override_value", "message"),
    [
        ("CUDA_OFFLOAD", "model", "runtime_offload", "none", "offload"),
        ("CUDA_QUANTIZE", "fp8", "runtime_quantize", "none", "quantize"),
        ("CUDA_ENABLE_XFORMERS", "1", "runtime_enable_xformers", False, "xformers"),
    ],
)
def test_mps_policy_rejects_invalid_env_even_with_safe_override(
    mps_ready, env_name, env_value, override_name, override_value, message
):
    with patch.dict(os.environ, {env_name: env_value}):
        with pytest.raises(ValueError, match=message):
            MPS_TORCH_POLICY.resolve_settings(
                SimpleNamespace(**{override_name: override_value})
            )


def test_mps_policy_accepts_attention_slicing(mps_ready):
    settings = MPS_TORCH_POLICY.resolve_settings(
        SimpleNamespace(runtime_attention_slicing=True)
    )
    assert settings.attention_slicing is True


def test_mps_policy_places_pipeline_only_on_mps():
    pipe = MagicMock()
    placed = pipe.to.return_value

    assert MPS_TORCH_POLICY.place_pipeline(pipe, device="mps", offload="none") is placed
    pipe.to.assert_called_once_with("mps")
    pipe.enable_model_cpu_offload.assert_not_called()
    pipe.enable_sequential_cpu_offload.assert_not_called()


@pytest.mark.parametrize(("device", "offload"), [("cpu", "none"), ("mps", "model")])
def test_mps_policy_rejects_invalid_placement(device, offload):
    pipe = MagicMock()
    with pytest.raises(ValueError, match="MPS"):
        MPS_TORCH_POLICY.place_pipeline(pipe, device=device, offload=offload)
    pipe.to.assert_not_called()


def test_mps_policy_releases_mps_cache():
    with patch("torch.mps.empty_cache") as empty_cache:
        MPS_TORCH_POLICY.release_cache()
    empty_cache.assert_called_once_with()


def test_mps_policy_rejects_quantization_and_xformers():
    with pytest.raises(RuntimeError, match="MPS"):
        MPS_TORCH_POLICY.quantize_modules((MagicMock(),))
    with pytest.raises(RuntimeError, match="MPS"):
        MPS_TORCH_POLICY.enable_memory_efficient_attention(MagicMock())
