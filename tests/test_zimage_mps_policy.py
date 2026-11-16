"""Z-Image bf16 MPS policy tests (STABL-lcfvhjog).

Real Torch. The device probe and the bf16 operation check are mocked, so no
MPS device is needed. Inherited validation runs in test_mps_policy.py.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from backends.platforms.mps_policy import MpsTorchPolicy
from backends.platforms.zimage_mps_policy import ZIMAGE_MPS_TORCH_POLICY

_SETTING_ENV = (
    "CUDA_DTYPE",
    "CUDA_OFFLOAD",
    "CUDA_QUANTIZE",
    "CUDA_ENABLE_XFORMERS",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


@pytest.fixture
def mps_ready(monkeypatch):
    for name in _SETTING_ENV:
        monkeypatch.delenv(name, raising=False)
    with (
        patch("torch.backends.mps.is_built", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("backends.platforms.zimage_mps_policy._check_bf16_available") as check,
    ):
        yield check


def test_zimage_policy_uses_bf16(mps_ready):
    settings = ZIMAGE_MPS_TORCH_POLICY.resolve_settings(SimpleNamespace())
    mps_ready.assert_called_once_with()
    assert settings.device == "mps"
    assert settings.dtype is torch.bfloat16
    assert settings.dtype_str == "bf16"
    assert settings.offload == settings.quantize == "none"
    assert settings.enable_xformers is False


def test_zimage_policy_is_an_mps_policy():
    assert isinstance(ZIMAGE_MPS_TORCH_POLICY, MpsTorchPolicy)
    assert ZIMAGE_MPS_TORCH_POLICY.backend_id == "mps"


@pytest.mark.parametrize("dtype", [None, "fp16", "bf16", "fp32"])
def test_cuda_dtype_does_not_select_zimage_precision(mps_ready, monkeypatch, dtype):
    if dtype is not None:
        monkeypatch.setenv("CUDA_DTYPE", dtype)
    settings = ZIMAGE_MPS_TORCH_POLICY.resolve_settings(SimpleNamespace())
    assert settings.dtype is torch.bfloat16
    assert settings.dtype_str == "bf16"


def test_cpu_fallback_is_refused(mps_ready, monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="CPU fallback"):
        ZIMAGE_MPS_TORCH_POLICY.resolve_settings(SimpleNamespace())
    mps_ready.assert_not_called()


def test_bf16_probe_failure_is_refused(mps_ready):
    mps_ready.side_effect = RuntimeError("bf16 probe failed")
    with pytest.raises(RuntimeError, match="bf16 probe failed"):
        ZIMAGE_MPS_TORCH_POLICY.resolve_settings(SimpleNamespace())


@pytest.mark.parametrize("missing", ["is_built", "is_available"])
def test_mps_runtime_is_required_before_the_probe(mps_ready, missing):
    with patch(f"torch.backends.mps.{missing}", return_value=False):
        with pytest.raises(RuntimeError, match="MPS"):
            ZIMAGE_MPS_TORCH_POLICY.resolve_settings(SimpleNamespace())
    mps_ready.assert_not_called()
