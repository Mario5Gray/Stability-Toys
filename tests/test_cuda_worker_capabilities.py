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
import torch

sys.modules.setdefault("torch", torch)

# Stub heavy dependencies before importing cuda_worker.
_STUBS = [
    "numpy",
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

import backends.cuda_worker as cuda_worker_module  # noqa: E402
from backends.conditioning.artifacts import (  # noqa: E402
    ConditioningCompatibility,
    MaterializedConditioning,
)
from backends.cuda_worker import CudaWorkerBase, DiffusersSDXLCudaWorker  # noqa: E402

_BASE_ENV = {
    "CUDA_DEVICE": "cuda:0",
    "CUDA_DTYPE": "fp16",
    "CUDA_ENABLE_XFORMERS": "0",
    "CUDA_ATTENTION_SLICING": "0",
    "CUDA_QUANTIZE": "none",
    "CUDA_OFFLOAD": "none",
}

_EXPECTED_FP16 = cuda_worker_module.torch.float16


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
    pipe.tokenizer = object()
    pipe.text_encoder_2 = MagicMock()
    pipe.text_encoder_2.config = SimpleNamespace(hidden_size=1280)
    pipe.tokenizer_2 = object()
    pipe.to.return_value = pipe
    return pipe


def _fake_config(*, name, hidden_size, projection_dim=None):
    config = SimpleNamespace(_name_or_path=name, hidden_size=hidden_size)
    if projection_dim is not None:
        config.projection_dim = projection_dim
    return config


def _fake_module(*, name, hidden_size, dtype, projection_dim=None):
    return SimpleNamespace(
        dtype=dtype,
        config=_fake_config(
            name=name,
            hidden_size=hidden_size,
            projection_dim=projection_dim,
        ),
    )


def _make_sdxl_worker_with_fake_pipe(dtype=None, worker_cls=DiffusersSDXLCudaWorker):
    dtype = dtype or cuda_worker_module.torch.float16
    worker = worker_cls.__new__(worker_cls)
    worker.device = "cuda:0"
    worker.dtype = dtype
    worker._img2img_pipe = None
    worker.pipe = SimpleNamespace(
        tokenizer=SimpleNamespace(model_max_length=77),
        tokenizer_2=SimpleNamespace(model_max_length=77),
        text_encoder=_fake_module(
            name="local/sdxl-text-encoder",
            hidden_size=768,
            dtype=dtype,
        ),
        text_encoder_2=_fake_module(
            name="local/sdxl-text-encoder-2",
            hidden_size=1280,
            projection_dim=1280,
            dtype=dtype,
        ),
        unet=SimpleNamespace(dtype=dtype),
        vae=SimpleNamespace(dtype=dtype),
    )
    return worker


def _materialized_sdxl(worker, *, dtype=None, slots=None, compatibility=None):
    torch_mod = cuda_worker_module.torch
    dtype = dtype or torch_mod.float16
    slots = slots or {
        "prompt_embeds": torch_mod.zeros((1, 77, 2048), dtype=dtype),
        "negative_prompt_embeds": torch_mod.zeros((1, 77, 2048), dtype=dtype),
        "pooled_prompt_embeds": torch_mod.zeros((1, 1280), dtype=dtype),
        "negative_pooled_prompt_embeds": torch_mod.zeros((1, 1280), dtype=dtype),
    }
    compatibility = compatibility or worker._describe_conditioning_consumer(
        worker.pipe
    ).compatibility
    return MaterializedConditioning(slots=slots, compatibility=compatibility)


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
    def test_sdxl_single_file_uses_local_companion_config_when_provided(self):
        pipe = _make_pipe()
        model_info = SimpleNamespace(
            loader_format="single_file",
            scheduler_profile="native",
            metadata={"single_file_config": "configs/sdxl-base"},
        )
        fake_pipeline_cls = MagicMock()
        fake_pipeline_cls.from_single_file.return_value = pipe
        fake_lcm_cls = MagicMock()

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker.os.path.isdir", side_effect=lambda path: path == "/models/checkpoints/configs/sdxl-base"), \
             patch("backends.cuda_worker._sdxl_pipeline_cls", return_value=fake_pipeline_cls) as mock_pipe_cls, \
             patch("backends.cuda_worker._lcm_scheduler_cls", return_value=fake_lcm_cls) as mock_lcm_cls, \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", return_value=pipe):
            worker = DiffusersSDXLCudaWorker(
                worker_id=0,
                model_path="/models/checkpoints/sdxl-base.safetensors",
                model_info=model_info,
            )

        mock_pipe_cls.assert_called_once_with()
        fake_pipeline_cls.from_single_file.assert_called_once_with(
            "/models/checkpoints/sdxl-base.safetensors",
            torch_dtype=_EXPECTED_FP16,
            local_files_only=True,
            config="/models/checkpoints/configs/sdxl-base",
        )
        mock_lcm_cls.assert_not_called()
        assert worker.pipe is pipe

    def test_sdxl_single_file_native_scheduler_keeps_pipeline_scheduler(self):
        pipe = _make_pipe()
        model_info = SimpleNamespace(loader_format="single_file", scheduler_profile="native")
        fake_pipeline_cls = MagicMock()
        fake_pipeline_cls.from_single_file.return_value = pipe
        fake_lcm_cls = MagicMock()

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker._sdxl_pipeline_cls", return_value=fake_pipeline_cls) as mock_pipe_cls, \
             patch("backends.cuda_worker._lcm_scheduler_cls", return_value=fake_lcm_cls) as mock_lcm_cls, \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", return_value=pipe):
            worker = DiffusersSDXLCudaWorker(
                worker_id=0,
                model_path="/models/checkpoints/sdxl-base.safetensors",
                model_info=model_info,
            )

        mock_pipe_cls.assert_called_once_with()
        fake_pipeline_cls.from_single_file.assert_called_once_with(
            "/models/checkpoints/sdxl-base.safetensors",
            torch_dtype=_EXPECTED_FP16,
            local_files_only=True,
        )
        mock_lcm_cls.assert_not_called()
        assert worker.pipe is pipe

    def test_sdxl_single_file_missing_required_components_raises_clear_error(self):
        pipe = _make_pipe()
        pipe.tokenizer_2 = None
        model_info = SimpleNamespace(loader_format="single_file", scheduler_profile="native", metadata={})
        fake_pipeline_cls = MagicMock()
        fake_pipeline_cls.from_single_file.return_value = pipe

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker._sdxl_pipeline_cls", return_value=fake_pipeline_cls), \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", return_value=pipe):
            with pytest.raises(RuntimeError, match="missing required SDXL components: tokenizer_2"):
                DiffusersSDXLCudaWorker(
                    worker_id=0,
                    model_path="/models/checkpoints/sdxl-base.safetensors",
                    model_info=model_info,
                )

    def test_sdxl_single_file_missing_local_assets_raises_clear_error(self):
        model_info = SimpleNamespace(loader_format="single_file", scheduler_profile="native")
        fake_pipeline_cls = MagicMock()
        fake_pipeline_cls.from_single_file.side_effect = OSError("missing local config")

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker._sdxl_pipeline_cls", return_value=fake_pipeline_cls), \
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
        fake_pipeline_cls = MagicMock()
        fake_pipeline_cls.from_pretrained.return_value = pipe
        fake_lcm_cls = MagicMock()
        fake_lcm_cls.from_config.return_value = lcm_scheduler

        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("backends.cuda_worker._sdxl_pipeline_cls", return_value=fake_pipeline_cls) as mock_pipe_cls, \
             patch("backends.cuda_worker._lcm_scheduler_cls", return_value=fake_lcm_cls) as mock_lcm_cls, \
             patch.object(DiffusersSDXLCudaWorker, "_setup_pipe_memory_opts", return_value=pipe):
            worker = DiffusersSDXLCudaWorker(
                worker_id=0,
                model_path="/models/diffusers/sdxl-base",
                model_info=model_info,
            )

        mock_pipe_cls.assert_called_once_with()
        fake_pipeline_cls.from_pretrained.assert_called_once_with(
            "/models/diffusers/sdxl-base",
            torch_dtype=_EXPECTED_FP16,
            use_safetensors=True,
            variant="fp16",
        )
        mock_lcm_cls.assert_called_once_with()
        fake_lcm_cls.from_config.assert_called_once_with(original_config)
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
        worker.dtype = _EXPECTED_FP16
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

        worker.pipe.vae.to.assert_called_once_with("cuda:0", dtype=_EXPECTED_FP16)
        assert worker._img2img_pipe.vae is worker.pipe.vae
        assert worker._img2img_pipe.call_args.kwargs["negative_prompt"] == "blurry, watermark"


class TestSdxlConditioningContextAndAcceptance:
    def test_sdxl_conditioning_model_context_survives_worker_subclassing(self):
        class CustomSdxlWorker(DiffusersSDXLCudaWorker):
            pass

        worker = _make_sdxl_worker_with_fake_pipe(worker_cls=CustomSdxlWorker)

        context = worker._build_conditioning_context()

        assert context.descriptor.model_family == "sdxl"
        assert context.descriptor.hidden_dimensions == (768, 1280)
        assert context.descriptor.pooled_required is True
        assert context.local_encoder_bundle is not None
        assert context.local_encoder_bundle.text_encoders() == (
            worker.pipe.text_encoder,
            worker.pipe.text_encoder_2,
        )

    def test_sdxl_conditioning_model_context_describes_both_encoders_and_pooled_output(self):
        worker = _make_sdxl_worker_with_fake_pipe(dtype=cuda_worker_module.torch.float16)

        context = worker._build_conditioning_context()

        assert context.descriptor.model_family == "sdxl"
        assert context.descriptor.tokenizer_max_length == 77
        assert context.descriptor.hidden_dimensions == (768, 1280)
        assert context.descriptor.pooled_required is True
        assert context.descriptor.encode_dtype_name == "float16"
        assert context.descriptor.encoder_identities == (
            "local/sdxl-text-encoder",
            "local/sdxl-text-encoder-2",
        )
        assert context.local_encoder_bundle is not None
        assert context.local_encoder_bundle.tokenizers() == (
            worker.pipe.tokenizer,
            worker.pipe.tokenizer_2,
        )
        assert context.local_encoder_bundle.text_encoders() == (
            worker.pipe.text_encoder,
            worker.pipe.text_encoder_2,
        )
        assert worker.pipe not in context.__dict__.values()

    def test_accept_conditioning_materialized_sdxl_returns_exact_pipeline_kwargs(self):
        worker = _make_sdxl_worker_with_fake_pipe()
        artifact = _materialized_sdxl(worker)

        kwargs = worker._accept_conditioning_artifact(worker.pipe, artifact)

        assert kwargs == artifact.slots

    @pytest.mark.parametrize(
        ("artifact_factory", "match"),
        [
            (
                lambda worker: _materialized_sdxl(
                    worker,
                    slots={
                        "prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 77, 2048), dtype=cuda_worker_module.torch.float16
                        ),
                        "negative_prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 77, 2048), dtype=cuda_worker_module.torch.float16
                        ),
                    },
                ),
                "slots",
            ),
            (
                lambda worker: _materialized_sdxl(
                    worker,
                    compatibility=ConditioningCompatibility(
                        model_family="sd15",
                        encoder_identities=(
                            "local/sdxl-text-encoder",
                            "local/sdxl-text-encoder-2",
                        ),
                        hidden_dimensions=(768, 1280),
                        pooled_required=True,
                        dtype_name="float16",
                    ),
                ),
                "compatibility",
            ),
            (
                lambda worker: _materialized_sdxl(
                    worker,
                    slots={
                        "prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 77, 2048), dtype=cuda_worker_module.torch.float16
                        ),
                        "negative_prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 77, 2048), dtype=cuda_worker_module.torch.float16
                        ),
                        "pooled_prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 1024), dtype=cuda_worker_module.torch.float16
                        ),
                        "negative_pooled_prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 1024), dtype=cuda_worker_module.torch.float16
                        ),
                    },
                ),
                "pooled",
            ),
            (
                lambda worker: _materialized_sdxl(
                    worker,
                    slots={
                        "prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 77, 2048), dtype=cuda_worker_module.torch.float16
                        ),
                        "negative_prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 76, 2048), dtype=cuda_worker_module.torch.float16
                        ),
                        "pooled_prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 1280), dtype=cuda_worker_module.torch.float16
                        ),
                        "negative_pooled_prompt_embeds": cuda_worker_module.torch.zeros(
                            (1, 1280), dtype=cuda_worker_module.torch.float16
                        ),
                    },
                ),
                "sequence",
            ),
        ],
    )
    def test_accept_conditioning_materialized_sdxl_rejects_invalid_artifacts_fail_closed(
        self,
        artifact_factory,
        match,
    ):
        worker = _make_sdxl_worker_with_fake_pipe()
        target_pipe = Mock()
        target_pipe.tokenizer = worker.pipe.tokenizer
        target_pipe.tokenizer_2 = worker.pipe.tokenizer_2
        target_pipe.text_encoder = worker.pipe.text_encoder
        target_pipe.text_encoder_2 = worker.pipe.text_encoder_2
        target_pipe.unet = worker.pipe.unet
        target_pipe.vae = worker.pipe.vae

        with pytest.raises(ValueError, match=match):
            worker._accept_conditioning_artifact(target_pipe, artifact_factory(worker))

        target_pipe.assert_not_called()
