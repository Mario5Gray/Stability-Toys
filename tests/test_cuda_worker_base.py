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
from unittest.mock import MagicMock, Mock, patch

import torch

sys.modules.setdefault("torch", torch)

# ---------------------------------------------------------------------------
# Stub out diffusers and project deps before importing cuda_worker.
# sys.modules.setdefault means real installs (if present) are not disturbed.
# ---------------------------------------------------------------------------
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

import pytest  # noqa: E402
from backends.conditioning.artifacts import (  # noqa: E402 — stubs must be set first
    ConditioningCompatibility,
    DelegatedConditioning,
    MaterializedConditioning,
)
from backends.cuda_worker import (  # noqa: E402 — stubs must be set first
    CudaWorkerBase,
    DiffusersCudaWorker,
)

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


def _make_sd15_worker_with_fake_pipe(dtype=None):
    dtype = dtype or cuda_worker_torch().float16
    worker = DiffusersCudaWorker.__new__(DiffusersCudaWorker)
    worker.device = "cuda:0"
    worker.dtype = dtype
    worker._img2img_pipe = None
    worker.pipe = SimpleNamespace(
        tokenizer=SimpleNamespace(model_max_length=77),
        text_encoder=_fake_module(
            name="local/sd15-text-encoder",
            hidden_size=768,
            dtype=dtype,
        ),
        unet=SimpleNamespace(dtype=dtype),
        vae=SimpleNamespace(dtype=dtype),
    )
    return worker


def cuda_worker_torch():
    from backends import cuda_worker as cuda_worker_module

    return cuda_worker_module.torch


def _materialized_sd15(worker, *, dtype=None, slots=None, compatibility=None):
    torch_mod = cuda_worker_torch()
    dtype = dtype or torch_mod.float16
    slots = slots or {
        "prompt_embeds": torch_mod.zeros((1, 77, 768), dtype=dtype),
        "negative_prompt_embeds": torch_mod.zeros((1, 77, 768), dtype=dtype),
    }
    compatibility = compatibility or worker._describe_conditioning_consumer(
        worker.pipe
    ).compatibility
    return MaterializedConditioning(slots=slots, compatibility=compatibility)


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


class TestConditioningContextAndAcceptance:
    def test_sd15_conditioning_model_context_exposes_plain_descriptor_and_local_bundle(self):
        torch_mod = cuda_worker_torch()
        worker = _make_sd15_worker_with_fake_pipe(dtype=torch_mod.float16)

        context = worker._build_conditioning_context()

        assert context.descriptor.model_family == "sd15"
        assert context.descriptor.tokenizer_max_length == 77
        assert context.descriptor.hidden_dimensions == (768,)
        assert context.descriptor.pooled_required is False
        assert context.descriptor.encode_dtype_name == "float16"
        assert context.descriptor.device == "cuda:0"
        assert context.descriptor.encoder_identities == ("local/sd15-text-encoder",)
        assert context.local_encoder_bundle is not None
        assert context.local_encoder_bundle.tokenizers() == (worker.pipe.tokenizer,)
        assert context.local_encoder_bundle.text_encoders() == (worker.pipe.text_encoder,)
        assert context.local_encoder_bundle.live_dtype() == torch_mod.float16
        assert worker.pipe not in context.__dict__.values()

    def test_accept_conditioning_delegated_returns_only_prompt_kwargs(self):
        worker = _make_sd15_worker_with_fake_pipe()

        kwargs = worker._accept_conditioning_artifact(
            worker.pipe,
            DelegatedConditioning("cat", None),
        )

        assert kwargs == {"prompt": "cat", "negative_prompt": None}

    def test_accept_conditioning_materialized_sd15_returns_exact_pipeline_kwargs(self):
        worker = _make_sd15_worker_with_fake_pipe()
        artifact = _materialized_sd15(worker)

        kwargs = worker._accept_conditioning_artifact(worker.pipe, artifact)

        assert kwargs == artifact.slots

    def test_accept_conditioning_materialized_rechecks_live_dtype_after_artifact_creation(self):
        torch_mod = cuda_worker_torch()
        worker = _make_sd15_worker_with_fake_pipe(dtype=torch_mod.float16)
        artifact = _materialized_sd15(worker, dtype=torch_mod.float16)
        worker.pipe.text_encoder.dtype = torch_mod.float32

        with pytest.raises(ValueError, match="dtype"):
            worker._accept_conditioning_artifact(worker.pipe, artifact)

    @pytest.mark.parametrize(
        ("artifact_factory", "match"),
        [
            (lambda worker: object(), "unknown conditioning artifact"),
            (
                lambda worker: _materialized_sd15(
                    worker,
                    slots={
                        "prompt_embeds": cuda_worker_torch().zeros(
                            (1, 77, 768), dtype=cuda_worker_torch().float16
                        )
                    },
                ),
                "slots",
            ),
            (
                lambda worker: _materialized_sd15(
                    worker,
                    compatibility=ConditioningCompatibility(
                        model_family="sdxl",
                        encoder_identities=("local/sd15-text-encoder",),
                        hidden_dimensions=(768,),
                        pooled_required=False,
                        dtype_name="float16",
                    ),
                ),
                "compatibility",
            ),
            (
                lambda worker: _materialized_sd15(
                    worker,
                    compatibility=ConditioningCompatibility(
                        model_family="sd15",
                        encoder_identities=("other-encoder",),
                        hidden_dimensions=(768,),
                        pooled_required=False,
                        dtype_name="float16",
                    ),
                ),
                "compatibility",
            ),
            (
                lambda worker: _materialized_sd15(
                    worker,
                    slots={
                        "prompt_embeds": cuda_worker_torch().zeros(
                            (1, 77, 1024), dtype=cuda_worker_torch().float16
                        ),
                        "negative_prompt_embeds": cuda_worker_torch().zeros(
                            (1, 77, 1024), dtype=cuda_worker_torch().float16
                        ),
                    },
                ),
                "hidden",
            ),
            (
                lambda worker: _materialized_sd15(
                    worker,
                    slots={
                        "prompt_embeds": object(),
                        "negative_prompt_embeds": cuda_worker_torch().zeros(
                            (1, 77, 768), dtype=cuda_worker_torch().float16
                        ),
                    },
                ),
                "tensor",
            ),
            (
                lambda worker: _materialized_sd15(
                    worker,
                    slots={
                        "prompt_embeds": cuda_worker_torch().zeros(
                            (1, 77, 768), dtype=cuda_worker_torch().float32
                        ),
                        "negative_prompt_embeds": cuda_worker_torch().zeros(
                            (1, 77, 768), dtype=cuda_worker_torch().float32
                        ),
                    },
                ),
                "dtype",
            ),
        ],
    )
    def test_accept_conditioning_materialized_sd15_rejects_invalid_artifacts_fail_closed(
        self,
        artifact_factory,
        match,
    ):
        worker = _make_sd15_worker_with_fake_pipe()
        target_pipe = Mock()
        target_pipe.tokenizer = worker.pipe.tokenizer
        target_pipe.text_encoder = worker.pipe.text_encoder
        target_pipe.unet = worker.pipe.unet
        target_pipe.vae = worker.pipe.vae

        with pytest.raises(ValueError, match=match):
            worker._accept_conditioning_artifact(target_pipe, artifact_factory(worker))

        target_pipe.assert_not_called()
