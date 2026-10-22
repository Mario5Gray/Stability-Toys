"""Torch execution policy seam (STABL-xkiwjvrm).

Shared Diffusers worker code owns family behavior. The policy owns device
actions: settings, quantization, attention backend, placement, cache release.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]

_CLEAN_CUDA_ENV = {
    name: value
    for name, value in os.environ.items()
    if not name.startswith("CUDA_")
}


def _settings(**overrides):
    from backends.torch_policy import TorchExecutionSettings

    values = dict(
        device="cuda:0",
        dtype="fp16_sentinel",
        dtype_str="fp16",
        enable_xformers=False,
        attention_slicing=False,
        quantize="none",
        offload="none",
    )
    values.update(overrides)
    return TorchExecutionSettings(**values)


def test_policy_modules_import_without_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import backends.torch_policy; "
            "import backends.platforms.cuda_policy; "
            "import backends.platforms.mps_policy; "
            "assert 'torch' not in sys.modules, 'torch imported eagerly'",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cuda_policy_satisfies_contract():
    from backends.platforms.cuda_policy import CUDA_TORCH_POLICY
    from backends.torch_policy import TorchExecutionPolicy

    assert isinstance(CUDA_TORCH_POLICY, TorchExecutionPolicy)
    assert CUDA_TORCH_POLICY.backend_id == "cuda"


def test_cuda_policy_resolves_env_defaults():
    import torch

    from backends.platforms.cuda_policy import CUDA_TORCH_POLICY

    with patch.dict(os.environ, _CLEAN_CUDA_ENV, clear=True):
        settings = CUDA_TORCH_POLICY.resolve_settings(None)

    assert settings.device == "cuda:0"
    assert settings.dtype is torch.float16
    assert settings.dtype_str == "fp16"
    assert settings.enable_xformers is False
    assert settings.attention_slicing is False
    assert settings.quantize == "none"
    assert settings.offload == "none"


def test_cuda_policy_runtime_overrides_beat_env():
    from backends.platforms.cuda_policy import CUDA_TORCH_POLICY

    env = {**_CLEAN_CUDA_ENV, "CUDA_OFFLOAD": "none", "CUDA_QUANTIZE": "none"}
    model_info = SimpleNamespace(
        runtime_offload="model",
        runtime_quantize="FP8",
        runtime_enable_xformers=True,
        runtime_attention_slicing=True,
    )
    with patch.dict(os.environ, env, clear=True):
        settings = CUDA_TORCH_POLICY.resolve_settings(model_info)

    assert settings.offload == "model"
    assert settings.quantize == "fp8"
    assert settings.enable_xformers is True
    assert settings.attention_slicing is True


def test_cuda_policy_places_with_device_index():
    from backends.platforms.cuda_policy import CUDA_TORCH_POLICY

    pipe = MagicMock()
    placed = CUDA_TORCH_POLICY.place_pipeline(pipe, device="cuda:2", offload="model")

    pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=2)
    pipe.to.assert_not_called()
    assert placed is pipe


def test_cuda_policy_release_cache_calls_cuda_empty_cache():
    from backends.platforms.cuda_policy import CUDA_TORCH_POLICY

    with patch("torch.cuda.empty_cache") as empty_cache:
        CUDA_TORCH_POLICY.release_cache()

    empty_cache.assert_called_once_with()


class _RecordingPolicy:
    backend_id = "recording"

    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    def resolve_settings(self, model_info):
        self.calls.append(("resolve_settings", model_info))
        return self.settings

    def quantize_modules(self, modules):
        self.calls.append(("quantize_modules", modules))

    def enable_memory_efficient_attention(self, pipe):
        self.calls.append(("enable_memory_efficient_attention", pipe))

    def place_pipeline(self, pipe, *, device, offload):
        self.calls.append(("place_pipeline", (device, offload)))
        return pipe

    def release_cache(self):
        self.calls.append(("release_cache",))


def test_worker_delegates_device_actions_to_policy():
    from backends.cuda_worker import CudaWorkerBase

    policy = _RecordingPolicy(
        _settings(device="rec:0", quantize="fp8", offload="model", enable_xformers=True)
    )

    class _PolicyWorker(CudaWorkerBase):
        execution_policy = policy

    model_info = SimpleNamespace(checkpoint_precision="fp16")
    worker = _PolicyWorker(worker_id=0, model_info=model_info)

    assert worker.device == "rec:0"
    assert worker.dtype == "fp16_sentinel"
    assert worker._offload == "model"
    assert worker._quantize == "fp8"

    pipe = MagicMock()
    placed = worker._setup_pipe_memory_opts(pipe)

    assert placed is pipe
    assert [call[0] for call in policy.calls] == [
        "resolve_settings",
        "quantize_modules",
        "enable_memory_efficient_attention",
        "place_pipeline",
    ]
    assert policy.calls[1][1] == (pipe.unet,)
    assert policy.calls[3][1] == ("rec:0", "model")
    pipe.to.assert_not_called()


def test_cuda_worker_module_makes_no_direct_cuda_calls():
    import backends.cuda_worker as cuda_worker

    assert "torch.cuda." not in inspect.getsource(cuda_worker)
