"""
Unit tests for CudaWorkerBase._setup_pipe_memory_opts and _device_index.

These tests mock the pipeline object so no GPU, diffusers install, or model
file is needed.  They guard against:
  - API typos on pipe.vae (enable_vae_slicing vs enable_slicing)
  - Offload mode routing (none / model / sequential)
  - CUDA_DEVICE index not being forwarded to offload calls
  - Attention-slicing flag propagation
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out diffusers and project deps before importing cuda_worker.
# sys.modules.setdefault means real installs (if present) are not disturbed.
# ---------------------------------------------------------------------------
_STUBS = [
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.PngImagePlugin",
    "diffusers",
    "diffusers.schedulers",
    "diffusers.schedulers.scheduling_lcm",
    "diffusers.pipelines",
    "diffusers.pipelines.stable_diffusion",
    "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
    "diffusers.pipelines.stable_diffusion_xl",
    "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl",
    "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img",
    "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img",
    "backends.styles",
]
for _mod in _STUBS:
    sys.modules.setdefault(_mod, MagicMock())

# torch: stub only if not already installed; expose dtype sentinels _parse_env assigns.
if "torch" not in sys.modules:
    _torch_stub = MagicMock()
    _torch_stub.float16 = "fp16_sentinel"
    _torch_stub.bfloat16 = "bf16_sentinel"
    _torch_stub.float32 = "fp32_sentinel"
    sys.modules["torch"] = _torch_stub

# Wire the specific names the module imports from each stub.
sys.modules["diffusers.schedulers.scheduling_lcm"].LCMScheduler = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion"].StableDiffusionPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl"].StableDiffusionXLPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img"].StableDiffusionImg2ImgPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img"].StableDiffusionXLImg2ImgPipeline = MagicMock
sys.modules["backends.styles"].STYLE_REGISTRY = {}

import pytest
from backends.cuda_worker import CudaWorkerBase  # noqa: E402 — stubs must be set first

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ENV = {
    "CUDA_DEVICE": "cuda:0",
    "CUDA_DTYPE": "fp16",
    "CUDA_ENABLE_XFORMERS": "0",
    "CUDA_ATTENTION_SLICING": "0",
    "CUDA_QUANTIZE": "none",
    "CUDA_OFFLOAD": "none",
}


def _make_base(extra_env=None, model_info=None):
    """Instantiate CudaWorkerBase with a clean env overlay."""
    env = {**_BASE_ENV, **(extra_env or {})}
    with patch.dict(os.environ, env, clear=False):
        return CudaWorkerBase(worker_id=0, model_info=model_info)


def _make_pipe():
    """Mock pipeline with the attributes _setup_pipe_memory_opts touches."""
    pipe = MagicMock()
    pipe.vae = MagicMock()
    pipe.unet = MagicMock()
    del pipe.text_encoder_2   # absent = SD1.5 path; MagicMock del makes hasattr False
    pipe.to.return_value = pipe
    return pipe


# ---------------------------------------------------------------------------
# VAE memory opts — regression guard for enable_vae_slicing vs enable_slicing
# ---------------------------------------------------------------------------

class TestVaeMemoryOpts:
    def test_enable_tiling_called(self):
        pipe = _make_pipe()
        _make_base()._setup_pipe_memory_opts(pipe)
        pipe.vae.enable_tiling.assert_called_once()

    def test_enable_slicing_called(self):
        pipe = _make_pipe()
        _make_base()._setup_pipe_memory_opts(pipe)
        pipe.vae.enable_slicing.assert_called_once()

    def test_enable_vae_slicing_NOT_called(self):
        """enable_vae_slicing does not exist on AutoencoderKL; calling it crashes startup."""
        pipe = _make_pipe()
        _make_base()._setup_pipe_memory_opts(pipe)
        pipe.vae.enable_vae_slicing.assert_not_called()


# ---------------------------------------------------------------------------
# Offload routing
# ---------------------------------------------------------------------------

class TestOffloadRouting:
    def test_no_offload_places_pipe_on_device(self):
        pipe = _make_pipe()
        _make_base({"CUDA_OFFLOAD": "none", "CUDA_DEVICE": "cuda:0"})._setup_pipe_memory_opts(pipe)
        pipe.to.assert_called_once_with("cuda:0")
        pipe.enable_model_cpu_offload.assert_not_called()
        pipe.enable_sequential_cpu_offload.assert_not_called()

    def test_model_offload(self):
        pipe = _make_pipe()
        _make_base({"CUDA_OFFLOAD": "model"})._setup_pipe_memory_opts(pipe)
        pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=0)
        pipe.to.assert_not_called()

    def test_sequential_offload(self):
        pipe = _make_pipe()
        _make_base({"CUDA_OFFLOAD": "sequential"})._setup_pipe_memory_opts(pipe)
        pipe.enable_sequential_cpu_offload.assert_called_once_with(gpu_id=0)
        pipe.to.assert_not_called()

    def test_model_offload_respects_device_index(self):
        pipe = _make_pipe()
        _make_base({"CUDA_OFFLOAD": "model", "CUDA_DEVICE": "cuda:1"})._setup_pipe_memory_opts(pipe)
        pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=1)

    def test_sequential_offload_respects_device_index(self):
        pipe = _make_pipe()
        _make_base({"CUDA_OFFLOAD": "sequential", "CUDA_DEVICE": "cuda:2"})._setup_pipe_memory_opts(pipe)
        pipe.enable_sequential_cpu_offload.assert_called_once_with(gpu_id=2)


# ---------------------------------------------------------------------------
# Device index parsing
# ---------------------------------------------------------------------------

class TestDeviceIndex:
    def test_cuda_colon_0(self):
        assert _make_base({"CUDA_DEVICE": "cuda:0"})._device_index() == 0

    def test_cuda_colon_1(self):
        assert _make_base({"CUDA_DEVICE": "cuda:1"})._device_index() == 1

    def test_bare_cuda_defaults_to_0(self):
        assert _make_base({"CUDA_DEVICE": "cuda"})._device_index() == 0

    def test_non_numeric_suffix_defaults_to_0(self):
        assert _make_base({"CUDA_DEVICE": "cuda:bad"})._device_index() == 0


# ---------------------------------------------------------------------------
# Attention slicing
# ---------------------------------------------------------------------------

class TestAttentionSlicing:
    def test_disabled_by_default(self):
        pipe = _make_pipe()
        _make_base()._setup_pipe_memory_opts(pipe)
        pipe.enable_attention_slicing.assert_not_called()

    def test_enabled_when_set(self):
        pipe = _make_pipe()
        _make_base({"CUDA_ATTENTION_SLICING": "1"})._setup_pipe_memory_opts(pipe)
        pipe.enable_attention_slicing.assert_called_once()


class TestRuntimePolicyOverrides:
    def test_model_info_runtime_policy_overrides_env_defaults(self):
        pipe = _make_pipe()
        model_info = SimpleNamespace(
            runtime_quantize="none",
            runtime_offload="model",
            runtime_attention_slicing=True,
            runtime_enable_xformers=True,
            checkpoint_precision="fp8",
        )

        _make_base(
            {
                "CUDA_QUANTIZE": "fp8",
                "CUDA_OFFLOAD": "none",
                "CUDA_ATTENTION_SLICING": "0",
                "CUDA_ENABLE_XFORMERS": "0",
            },
            model_info=model_info,
        )._setup_pipe_memory_opts(pipe)

        pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=0)
        pipe.enable_attention_slicing.assert_called_once_with(1)
        pipe.enable_xformers_memory_efficient_attention.assert_called_once()
        pipe.to.assert_not_called()
