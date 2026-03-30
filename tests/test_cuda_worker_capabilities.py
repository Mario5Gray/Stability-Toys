"""
Unit tests for capability-driven CUDA worker behavior.

These tests keep diffusers/torch fully stubbed so they can run without GPU
dependencies while still exercising the loader and scheduler branches added for
capability-aware SDXL checkpoints.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

# Stub heavy dependencies before importing cuda_worker.
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

if "torch" not in sys.modules:
    _torch_stub = MagicMock()
    _torch_stub.float16 = "fp16_sentinel"
    _torch_stub.bfloat16 = "bf16_sentinel"
    _torch_stub.float32 = "fp32_sentinel"
    sys.modules["torch"] = _torch_stub

sys.modules["diffusers.schedulers.scheduling_lcm"].LCMScheduler = MagicMock()
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion"].StableDiffusionPipeline = MagicMock()
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl"].StableDiffusionXLPipeline = MagicMock()
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img"].StableDiffusionImg2ImgPipeline = MagicMock()
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img"].StableDiffusionXLImg2ImgPipeline = MagicMock()
sys.modules["backends.styles"].STYLE_REGISTRY = {}

from backends.cuda_worker import CudaWorkerBase, DiffusersSDXLCudaWorker  # noqa: E402

_BASE_ENV = {
    "CUDA_DEVICE": "cuda:0",
    "CUDA_DTYPE": "fp16",
    "CUDA_ENABLE_XFORMERS": "0",
    "CUDA_ATTENTION_SLICING": "0",
    "CUDA_QUANTIZE": "none",
    "CUDA_OFFLOAD": "none",
}


def _make_base(extra_env=None):
    env = {**_BASE_ENV, **(extra_env or {})}
    with patch.dict(os.environ, env, clear=False):
        return CudaWorkerBase(worker_id=0)


def _make_pipe():
    pipe = MagicMock()
    pipe.scheduler = MagicMock()
    pipe.scheduler.config = {"name": "base"}
    pipe.vae = MagicMock()
    pipe.unet = MagicMock()
    pipe.unet.config = SimpleNamespace(cross_attention_dim=2048)
    pipe.text_encoder = MagicMock()
    pipe.text_encoder.config = SimpleNamespace(hidden_size=768)
    pipe.text_encoder_2 = MagicMock()
    pipe.text_encoder_2.config = SimpleNamespace(hidden_size=1280)
    pipe.to.return_value = pipe
    return pipe


class TestCapabilityAwareMemoryOpts:
    def test_fp8_single_file_checkpoint_still_applies_runtime_quanto_when_requested(self):
        pipe = _make_pipe()
        base = _make_base({"CUDA_QUANTIZE": "fp8"})
        base.model_info = SimpleNamespace(checkpoint_precision="fp8", loader_format="single_file")
        fake_qfloat8 = object()
        fake_quanto = SimpleNamespace(
            quantize=Mock(),
            freeze=Mock(),
            qfloat8=fake_qfloat8,
        )

        with patch.dict(sys.modules, {"optimum.quanto": fake_quanto}):
            base._setup_pipe_memory_opts(pipe)

        fake_quanto.quantize.assert_any_call(pipe.unet, weights=fake_qfloat8)
        fake_quanto.quantize.assert_any_call(pipe.text_encoder_2, weights=fake_qfloat8)
        assert fake_quanto.quantize.call_count == 2
        assert fake_quanto.freeze.call_count == 2
        pipe.to.assert_called_once_with("cuda:0")


class TestSdxlCapabilityLoader:
    def test_sdxl_single_file_native_scheduler_keeps_pipeline_scheduler(self):
        pipe = _make_pipe()
        model_info = SimpleNamespace(loader_format="single_file", scheduler_profile="native")

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_single_file", return_value=pipe) as mock_single, \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_pretrained", side_effect=AssertionError("from_pretrained should not be called")), \
             patch("backends.cuda_worker.LCMScheduler.from_config") as mock_lcm, \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", return_value=pipe):
            worker = DiffusersSDXLCudaWorker(
                worker_id=0,
                model_path="/models/checkpoints/sdxl-base.safetensors",
                model_info=model_info,
            )

        mock_single.assert_called_once_with(
            "/models/checkpoints/sdxl-base.safetensors",
            torch_dtype="fp16_sentinel",
        )
        mock_lcm.assert_not_called()
        assert worker.pipe is pipe

    def test_sdxl_diffusers_dir_lcm_scheduler_switches_scheduler(self):
        pipe = _make_pipe()
        original_config = pipe.scheduler.config
        lcm_scheduler = object()
        model_info = SimpleNamespace(loader_format="diffusers_dir", scheduler_profile="lcm")

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_pretrained", return_value=pipe) as mock_pretrained, \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_single_file", side_effect=AssertionError("from_single_file should not be called")), \
             patch("backends.cuda_worker.LCMScheduler.from_config", return_value=lcm_scheduler) as mock_lcm, \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", return_value=pipe):
            worker = DiffusersSDXLCudaWorker(
                worker_id=0,
                model_path="/models/diffusers/sdxl-base",
                model_info=model_info,
            )

        mock_pretrained.assert_called_once_with(
            "/models/diffusers/sdxl-base",
            torch_dtype="fp16_sentinel",
            use_safetensors=True,
            variant="fp16",
        )
        mock_lcm.assert_called_once_with(original_config)
        assert worker.pipe.scheduler is lcm_scheduler
