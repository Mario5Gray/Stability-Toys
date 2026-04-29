"""
Unit tests for ordered ControlNet execution in CUDA workers.

These tests stub heavy diffusers/torch dependencies and assert only the
request-shaping contract for ControlNet bindings. Task T5.1 expects these to
fail until the worker threads binding data into pipeline kwargs.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


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

from backends.cuda_worker import DiffusersCudaWorker, DiffusersSDXLCudaWorker  # noqa: E402


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
    worker.pipe = MagicMock()
    worker.pipe.return_value = SimpleNamespace(images=[MagicMock()])
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

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open", return_value=opened), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    kwargs = worker.pipe.call_args.kwargs
    assert kwargs["controlnet"] == "loaded:canny-model"
    assert kwargs["image"] is resized
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

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open", side_effect=[opened_a, opened_b]), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    kwargs = worker.pipe.call_args.kwargs
    assert kwargs["controlnet"] == ["loaded:canny-model", "loaded:depth-model"]
    assert kwargs["image"] == [resized_a, resized_b]
    assert kwargs["controlnet_conditioning_scale"] == [0.4, 0.9]
    assert kwargs["control_guidance_start"] == [0.0, 0.1]
    assert kwargs["control_guidance_end"] == [0.8, 1.0]
