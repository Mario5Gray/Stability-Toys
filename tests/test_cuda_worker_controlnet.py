"""
Unit tests for ordered ControlNet execution in CUDA workers.

These tests stub heavy diffusers/torch dependencies and assert only the
request-shaping contract for ControlNet bindings. Task T5.1 expects these to
fail until the worker threads binding data into pipeline kwargs.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest


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


class _FakePipelineBase:
    def __init__(self):
        self.components = {}
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[MagicMock()])


class _FakeStableDiffusionPipeline(_FakePipelineBase):
    pass


class _FakeStableDiffusionXLPipeline(_FakePipelineBase):
    pass


class _FakeStableDiffusionImg2ImgPipeline(_FakePipelineBase):
    pass


class _FakeStableDiffusionXLImg2ImgPipeline(_FakePipelineBase):
    pass


class _FakeStableDiffusionControlNetPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        return cls()


class _FakeStableDiffusionXLControlNetPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        return cls()


sys.modules["diffusers.schedulers.scheduling_lcm"].LCMScheduler = MagicMock()
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion"].StableDiffusionPipeline = _FakeStableDiffusionPipeline
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl"].StableDiffusionXLPipeline = _FakeStableDiffusionXLPipeline
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img"].StableDiffusionImg2ImgPipeline = _FakeStableDiffusionImg2ImgPipeline
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img"].StableDiffusionXLImg2ImgPipeline = _FakeStableDiffusionXLImg2ImgPipeline
sys.modules["diffusers"].StableDiffusionControlNetPipeline = _FakeStableDiffusionControlNetPipeline
sys.modules["diffusers"].StableDiffusionXLControlNetPipeline = _FakeStableDiffusionXLControlNetPipeline
sys.modules["diffusers"].ControlNetModel = MagicMock()
sys.modules["backends.styles"].STYLE_REGISTRY = {}

from backends.cuda_worker import (  # noqa: E402
    DiffusersCudaWorker,
    DiffusersSDXLCudaWorker,
    _decode_control_image,
)


def _make_req():
    return SimpleNamespace(
        prompt="an owl",
        negative_prompt="blurry",
        size="512x512",
        num_inference_steps=8,
        guidance_scale=3.0,
        seed=123,
        style_lora=None,
    )


def _make_worker(worker_cls):
    worker = worker_cls.__new__(worker_cls)
    worker.device = "cuda:0"
    worker.dtype = "fp16_sentinel"
    worker.worker_id = 0
    if worker_cls is DiffusersSDXLCudaWorker:
        worker.pipe = _FakeStableDiffusionXLPipeline()
    else:
        worker.pipe = _FakeStableDiffusionPipeline()
    worker._img2img_pipe = None
    worker._apply_style = Mock()
    worker._apply_request_scheduler = Mock(return_value="euler")
    return worker


def _make_binding(prefix, strength, start, end):
    return SimpleNamespace(
        attachment_id=f"{prefix}-attachment",
        model_id=f"{prefix}-model",
        model_path=f"/models/{prefix}",
        control_image_bytes=f"{prefix}-bytes".encode(),
        strength=strength,
        start_percent=start,
        end_percent=end,
    )


def _fake_control_image(label):
    opened = MagicMock(name=f"{label}_opened")
    converted = MagicMock(name=f"{label}_converted")
    resized = MagicMock(name=f"{label}_resized")
    opened.convert.return_value = converted
    converted.resize.return_value = resized
    return opened, resized


def _fake_cache():
    cache = MagicMock()
    cache.acquire.side_effect = lambda model_id, model_path, loader: f"loaded:{model_id}"
    return cache


def test_decode_control_image_converts_rgb_and_resizes():
    opened, resized = _fake_control_image("decode")

    with patch("backends.cuda_worker.Image.open", return_value=opened):
        result = _decode_control_image(b"control-map", (512, 512))

    opened.convert.assert_called_once_with("RGB")
    opened.convert.return_value.resize.assert_called_once_with((512, 512))
    assert result is resized


def test_load_controlnet_model_uses_process_cache():
    worker = _make_worker(DiffusersCudaWorker)
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    cache = _fake_cache()

    with patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache):
        loaded = worker._load_controlnet_model(binding)

    assert loaded == "loaded:canny-model"
    cache.acquire.assert_called_once()
    args, kwargs = cache.acquire.call_args
    assert args[:2] == ("canny-model", "/models/canny")
    assert callable(kwargs["loader"])


def test_sd15_worker_passes_single_controlnet_kwargs():
    worker = _make_worker(DiffusersCudaWorker)
    req = _make_req()
    job = SimpleNamespace(
        req=req,
        init_image=None,
        controlnet_bindings=[_make_binding("canny", 0.4, 0.0, 0.8)],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    opened, resized = _fake_control_image("single")
    cn_pipe = MagicMock()
    cn_pipe.return_value = SimpleNamespace(images=[MagicMock()])

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open", return_value=opened), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch.object(_FakeStableDiffusionControlNetPipeline, "from_pipe", return_value=cn_pipe) as from_pipe:
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    from_pipe.assert_called_once_with(worker.pipe, controlnet="loaded:canny-model")
    assert worker.pipe.calls == []
    kwargs = cn_pipe.call_args.kwargs
    assert "controlnet" not in kwargs
    assert kwargs["control_image"] is resized
    assert kwargs["controlnet_conditioning_scale"] == 0.4
    assert kwargs["control_guidance_start"] == 0.0
    assert kwargs["control_guidance_end"] == 0.8


def test_sdxl_worker_passes_controlnet_lists_in_request_order():
    worker = _make_worker(DiffusersSDXLCudaWorker)
    req = _make_req()
    job = SimpleNamespace(
        req=req,
        init_image=None,
        controlnet_bindings=[
            _make_binding("canny", 0.4, 0.0, 0.8),
            _make_binding("depth", 0.9, 0.1, 1.0),
        ],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    opened_a, resized_a = _fake_control_image("first")
    opened_b, resized_b = _fake_control_image("second")
    cn_pipe = MagicMock()
    cn_pipe.return_value = SimpleNamespace(images=[MagicMock()])

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open", side_effect=[opened_a, opened_b]), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch.object(_FakeStableDiffusionXLControlNetPipeline, "from_pipe", return_value=cn_pipe) as from_pipe:
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    from_pipe.assert_called_once_with(
        worker.pipe,
        controlnet=["loaded:canny-model", "loaded:depth-model"],
    )
    assert worker.pipe.calls == []
    kwargs = cn_pipe.call_args.kwargs
    assert "controlnet" not in kwargs
    assert kwargs["control_image"] == [resized_a, resized_b]
    assert kwargs["controlnet_conditioning_scale"] == [0.4, 0.9]
    assert kwargs["control_guidance_start"] == [0.0, 0.1]
    assert kwargs["control_guidance_end"] == [0.8, 1.0]


def test_sd15_worker_without_bindings_calls_base_pipeline_directly():
    worker = _make_worker(DiffusersCudaWorker)
    req = _make_req()
    job = SimpleNamespace(req=req, init_image=None, controlnet_bindings=[])
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch.object(_FakeStableDiffusionControlNetPipeline, "from_pipe") as from_pipe:
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    from_pipe.assert_not_called()
    assert len(worker.pipe.calls) == 1
    kwargs = worker.pipe.calls[0]
    assert "controlnet" not in kwargs
    assert "control_image" not in kwargs


def test_sd15_worker_rejects_controlnet_on_img2img_path():
    worker = _make_worker(DiffusersCudaWorker)
    req = _make_req()
    job = SimpleNamespace(
        req=req,
        init_image=b"init-bytes",
        controlnet_bindings=[_make_binding("canny", 0.4, 0.0, 0.8)],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open"):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        with pytest.raises(NotImplementedError):
            worker.run_job(job)
