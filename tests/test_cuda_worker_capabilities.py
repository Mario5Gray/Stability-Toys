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

import pytest

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


def _make_base(extra_env=None, model_info=None):
    env = {**_BASE_ENV, **(extra_env or {})}
    with patch.dict(os.environ, env, clear=False):
        return CudaWorkerBase(worker_id=0, model_info=model_info)


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
    def test_fp8_single_file_checkpoint_skips_runtime_quanto_when_requested(self):
        pipe = _make_pipe()
        model_info = SimpleNamespace(
            checkpoint_precision="fp8",
            loader_format="single_file",
            runtime_quantize="fp8",
        )
        base = _make_base({"CUDA_QUANTIZE": "fp8"}, model_info=model_info)
        fake_qfloat8 = object()
        fake_quanto = SimpleNamespace(
            quantize=Mock(),
            freeze=Mock(),
            qfloat8=fake_qfloat8,
        )

        with patch.dict(sys.modules, {"optimum.quanto": fake_quanto}):
            base._setup_pipe_memory_opts(pipe)

        fake_quanto.quantize.assert_not_called()
        fake_quanto.freeze.assert_not_called()
        pipe.to.assert_called_once_with("cuda:0")


class TestSchedulerSelection:
    def test_explicit_scheduler_id_is_applied_under_open_policy(self):
        pipe = _make_pipe()
        base = _make_base()
        base.pipe = pipe
        base.model_info = SimpleNamespace(
            default_scheduler_id=None,
            allowed_scheduler_ids=None,
        )
        base._baseline_scheduler_class = MagicMock()
        base._baseline_scheduler_config = {"name": "base"}

        req = SimpleNamespace(scheduler_id="euler")
        built_scheduler = object()

        with patch("backends.cuda_worker.build_scheduler", return_value=built_scheduler) as mock_build:
            selected = base._apply_request_scheduler(req)

        assert selected == "euler"
        mock_build.assert_called_once_with("euler", {"name": "base"})
        assert pipe.scheduler is built_scheduler

    def test_disallowed_scheduler_id_is_rejected(self):
        pipe = _make_pipe()
        base = _make_base()
        base.pipe = pipe
        base.model_info = SimpleNamespace(
            default_scheduler_id=None,
            allowed_scheduler_ids=["lcm"],
        )
        base._baseline_scheduler_class = MagicMock()
        base._baseline_scheduler_config = {"name": "base"}

        with pytest.raises(RuntimeError, match="not allowed"):
            base._apply_request_scheduler(SimpleNamespace(scheduler_id="euler"))

    def test_missing_scheduler_selection_restores_baseline(self):
        pipe = _make_pipe()
        base = _make_base()
        base.pipe = pipe
        restored_scheduler = object()
        baseline_cls = MagicMock()
        baseline_cls.from_config.return_value = restored_scheduler
        base._baseline_scheduler_class = baseline_cls
        base._baseline_scheduler_config = {"name": "base"}
        base.model_info = SimpleNamespace(
            default_scheduler_id=None,
            allowed_scheduler_ids=[],
        )

        selected = base._apply_request_scheduler(SimpleNamespace(scheduler_id=None))

        assert selected is None
        baseline_cls.from_config.assert_called_once_with({"name": "base"})
        assert pipe.scheduler is restored_scheduler


class TestSdxlCapabilityLoader:
    def test_sdxl_single_file_native_scheduler_keeps_pipeline_scheduler(self):
        pipe = _make_pipe()
        model_info = SimpleNamespace(loader_format="single_file", scheduler_profile="native")

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_single_file", return_value=pipe, create=True) as mock_single, \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_pretrained", side_effect=AssertionError("from_pretrained should not be called"), create=True), \
             patch("backends.cuda_worker.LCMScheduler.from_config", create=True) as mock_lcm, \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", return_value=pipe):
            worker = DiffusersSDXLCudaWorker(
                worker_id=0,
                model_path="/models/checkpoints/sdxl-base.safetensors",
                model_info=model_info,
            )

        mock_single.assert_called_once_with(
            "/models/checkpoints/sdxl-base.safetensors",
            torch_dtype="fp16_sentinel",
            local_files_only=True,
        )
        mock_lcm.assert_not_called()
        assert worker.pipe is pipe

    def test_sdxl_single_file_missing_local_assets_raises_clear_error(self):
        model_info = SimpleNamespace(loader_format="single_file", scheduler_profile="native")

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch(
                 "backends.cuda_worker.StableDiffusionXLPipeline.from_single_file",
                 side_effect=OSError("missing local config"),
                 create=True,
             ), \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", side_effect=AssertionError("_setup_pipe_memory_opts should not be reached")):
            with pytest.raises(RuntimeError, match="local-only SDXL single-file load failed"):
                DiffusersSDXLCudaWorker(
                    worker_id=0,
                    model_path="/models/checkpoints/sdxl-base.safetensors",
                    model_info=model_info,
                )

    def test_sdxl_diffusers_dir_lcm_scheduler_switches_scheduler(self):
        pipe = _make_pipe()
        original_config = pipe.scheduler.config
        lcm_scheduler = object()
        model_info = SimpleNamespace(loader_format="diffusers_dir", scheduler_profile="lcm")

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_pretrained", return_value=pipe, create=True) as mock_pretrained, \
             patch("backends.cuda_worker.StableDiffusionXLPipeline.from_single_file", side_effect=AssertionError("from_single_file should not be called"), create=True), \
             patch("backends.cuda_worker.LCMScheduler.from_config", return_value=lcm_scheduler, create=True) as mock_lcm, \
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


class TestNegativePromptForwarding:
    def test_sdxl_run_job_forwards_negative_prompt(self):
        worker = DiffusersSDXLCudaWorker.__new__(DiffusersSDXLCudaWorker)
        worker.device = "cuda:0"
        worker.worker_id = 0
        worker.pipe = MagicMock()
        worker.pipe.return_value = SimpleNamespace(images=[MagicMock()])
        worker._img2img_pipe = None
        worker._apply_style = Mock()
        worker._apply_request_scheduler = Mock(return_value="euler")

        req = SimpleNamespace(
            prompt="a castle",
            negative_prompt="blurry, watermark",
            size="512x512",
            num_inference_steps=8,
            guidance_scale=3.0,
            seed=123,
            style_lora=None,
        )
        job = SimpleNamespace(req=req, init_image=None)

        fake_generator = MagicMock()
        fake_generator.manual_seed.return_value = fake_generator

        with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
             patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
             patch("backends.cuda_worker.torch.cuda.empty_cache"), \
             patch("backends.cuda_worker.PngImagePlugin.PngInfo") as mock_pnginfo:
            mock_inference.return_value.__enter__.return_value = None
            mock_inference.return_value.__exit__.return_value = None
            pnginfo = mock_pnginfo.return_value

            worker.run_job(job)

        worker._apply_request_scheduler.assert_called_once_with(req)
        assert worker.pipe.call_args.kwargs["negative_prompt"] == "blurry, watermark"
        pnginfo.add_text.assert_called_once()

    def test_sdxl_img2img_normalizes_vae_dtype_before_execution(self):
        worker = DiffusersSDXLCudaWorker.__new__(DiffusersSDXLCudaWorker)
        worker.device = "cuda:0"
        worker.dtype = "fp16_sentinel"
        worker.worker_id = 0
        worker.pipe = _make_pipe()
        worker._img2img_pipe = MagicMock()
        worker._img2img_pipe.return_value = SimpleNamespace(images=[MagicMock()])
        worker._apply_style = Mock()
        worker._apply_request_scheduler = Mock(return_value="ddim")

        req = SimpleNamespace(
            prompt="a castle",
            negative_prompt="blurry, watermark",
            size="512x512",
            num_inference_steps=8,
            guidance_scale=3.0,
            seed=123,
            style_lora=None,
            denoise_strength=0.75,
        )
        job = SimpleNamespace(req=req, init_image=b"fake-image")

        fake_generator = MagicMock()
        fake_generator.manual_seed.return_value = fake_generator
        fake_init = MagicMock()
        fake_init.convert.return_value.resize.return_value = fake_init

        with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
             patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
             patch("backends.cuda_worker.torch.cuda.empty_cache"), \
             patch("backends.cuda_worker.Image.open", return_value=fake_init), \
             patch("backends.cuda_worker.PngImagePlugin.PngInfo"):
            mock_inference.return_value.__enter__.return_value = None
            mock_inference.return_value.__exit__.return_value = None

            worker.run_job(job)

        worker.pipe.vae.to.assert_called_once_with("cuda:0", dtype="fp16_sentinel")
        assert worker._img2img_pipe.vae is worker.pipe.vae
        assert worker._img2img_pipe.call_args.kwargs["negative_prompt"] == "blurry, watermark"
