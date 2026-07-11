"""
Unit tests for ordered ControlNet execution in CUDA workers.

These tests stub heavy diffusers/torch dependencies and assert only the
request-shaping contract for ControlNet bindings. Task T5.1 expects these to
fail until the worker threads binding data into pipeline kwargs.
"""

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image as PILImage

from backends.conditioning.invocation import CompletedInvocation


_MISSING = object()


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
    def from_pipe(cls, pipe, controlnet, *, torch_dtype=_MISSING):
        assert torch_dtype is None
        return cls()


class _FakeStableDiffusionXLControlNetPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet, *, torch_dtype=_MISSING):
        assert torch_dtype is None
        return cls()


class _FakeStableDiffusionControlNetImg2ImgPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet, *, torch_dtype=_MISSING):
        assert torch_dtype is None
        return cls()


class _FakeStableDiffusionXLControlNetImg2ImgPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet, *, torch_dtype=_MISSING):
        assert torch_dtype is None
        return cls()


sys.modules["diffusers.schedulers.scheduling_lcm"].LCMScheduler = MagicMock()
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion"].StableDiffusionPipeline = _FakeStableDiffusionPipeline
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl"].StableDiffusionXLPipeline = _FakeStableDiffusionXLPipeline
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img"].StableDiffusionImg2ImgPipeline = _FakeStableDiffusionImg2ImgPipeline
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img"].StableDiffusionXLImg2ImgPipeline = _FakeStableDiffusionXLImg2ImgPipeline
sys.modules["diffusers"].StableDiffusionControlNetPipeline = _FakeStableDiffusionControlNetPipeline
sys.modules["diffusers"].StableDiffusionXLControlNetPipeline = _FakeStableDiffusionXLControlNetPipeline
sys.modules["diffusers"].StableDiffusionControlNetImg2ImgPipeline = _FakeStableDiffusionControlNetImg2ImgPipeline
sys.modules["diffusers"].StableDiffusionXLControlNetImg2ImgPipeline = _FakeStableDiffusionXLControlNetImg2ImgPipeline
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


class _ReturningConditioningChain:
    def __init__(self, artifact):
        self.artifact = artifact
        self.requests = []

    def invoke(self, request, context):
        self.requests.append((request, context))
        return CompletedInvocation.success(self.artifact)


def _materialized_artifact(family):
    slots = {
        "prompt_embeds": object(),
        "negative_prompt_embeds": object(),
    }
    if family == "sdxl":
        slots.update(
            {
                "pooled_prompt_embeds": object(),
                "negative_pooled_prompt_embeds": object(),
            }
        )
    return SimpleNamespace(slots=slots)


def _install_materialized_conditioning(worker, family):
    artifact = _materialized_artifact(family)
    chain = _ReturningConditioningChain(artifact)
    context = object()
    worker._conditioning_chain = chain
    worker._conditioning_context = context
    worker._accept_conditioning_artifact = Mock(return_value=artifact.slots)
    return artifact, chain, context


def _make_binding(prefix, strength, start, end):
    return SimpleNamespace(
        attachment_id=f"{prefix}-attachment",
        control_type=prefix,
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


def _make_png_bytes(width, height):
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


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


def test_load_controlnet_model_loader_moves_to_worker_device():
    """The cache loader must place the ControlNet on the worker's CUDA device,
    otherwise from_pipe leaves it on CPU and _execution_device resolves to cpu
    while the rest of the pipeline is on cuda:0 (device-mismatch at encode_prompt)."""
    worker = _make_worker(DiffusersCudaWorker)
    binding = _make_binding("canny", 0.4, 0.0, 0.8)

    captured = {}
    cache = MagicMock()
    cache.acquire.side_effect = lambda model_id, model_path, loader: captured.setdefault("loader", loader)

    fake_model = MagicMock(name="controlnet_model")
    fake_model.to.return_value = fake_model
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_model

    with patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch("backends.cuda_worker._import_attr", return_value=fake_cls):
        worker._load_controlnet_model(binding)

    result = captured["loader"]("/models/canny")
    fake_model.to.assert_called_once_with("cuda:0")
    assert result is fake_model


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
    fake_pipe_cls = MagicMock()
    fake_pipe_cls.from_pipe.return_value = cn_pipe

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open", return_value=opened), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch("backends.cuda_worker._import_attr", return_value=fake_pipe_cls):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    fake_pipe_cls.from_pipe.assert_called_once_with(
        worker.pipe, controlnet="loaded:canny-model", torch_dtype=None
    )
    assert worker.pipe.calls == []
    kwargs = cn_pipe.call_args.kwargs
    assert "controlnet" not in kwargs
    assert kwargs["image"] is resized
    assert kwargs["controlnet_conditioning_scale"] == 0.4
    assert kwargs["control_guidance_start"] == 0.0
    assert kwargs["control_guidance_end"] == 0.8


# A future family (HunyuanDiT) declares its control-map kwarg by overriding the
# class constant, not the instance — model the real override surface here.
class _ControlImageKwargWorker(DiffusersCudaWorker):
    _CONTROL_IMAGE_KWARG = "control_image"


def test_control_image_kwarg_follows_worker_constant():
    """The control map must be routed under the kwarg named by the worker's
    _CONTROL_IMAGE_KWARG, so a future family (HunyuanDiT) can declare
    'control_image' via a class-level override without re-hardcoding the assembly."""
    worker = _make_worker(_ControlImageKwargWorker)
    assert type(worker)._CONTROL_IMAGE_KWARG == "control_image"  # class-level, not instance
    req = _make_req()
    job = SimpleNamespace(
        req=req,
        init_image=None,
        controlnet_bindings=[_make_binding("canny", 0.4, 0.0, 0.8)],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    opened, resized = _fake_control_image("kwarg")
    cn_pipe = MagicMock()
    cn_pipe.return_value = SimpleNamespace(images=[MagicMock()])
    fake_pipe_cls = MagicMock()
    fake_pipe_cls.from_pipe.return_value = cn_pipe

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open", return_value=opened), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch("backends.cuda_worker._import_attr", return_value=fake_pipe_cls):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    fake_pipe_cls.from_pipe.assert_called_once_with(
        worker.pipe, controlnet="loaded:canny-model", torch_dtype=None
    )
    kwargs = cn_pipe.call_args.kwargs
    assert kwargs["control_image"] is resized
    assert "image" not in kwargs


def test_build_controlnet_kwargs_image_kwarg_override_routes_to_control_image():
    """The combined img2img+ControlNet path needs the control map under
    control_image (image= is the init image there), not under
    _CONTROL_IMAGE_KWARG unchanged — the image_kwarg override is the seam."""
    worker = _make_worker(DiffusersCudaWorker)
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    cache = _fake_cache()
    opened, resized = _fake_control_image("override")

    with patch("backends.cuda_worker.Image.open", return_value=opened), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache):
        kwargs = worker._build_controlnet_kwargs(
            [binding], (512, 512), [], image_kwarg="control_image"
        )

    assert kwargs["control_image"] is resized
    assert "image" not in kwargs


def test_build_controlnet_kwargs_without_override_uses_class_constant():
    worker = _make_worker(DiffusersCudaWorker)
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    cache = _fake_cache()
    opened, resized = _fake_control_image("default")

    with patch("backends.cuda_worker.Image.open", return_value=opened), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache):
        kwargs = worker._build_controlnet_kwargs([binding], (512, 512), [])

    assert kwargs["image"] is resized
    assert "control_image" not in kwargs


def test_validate_control_image_aspect_ratio_passes_within_tolerance():
    from backends.cuda_worker import _validate_control_image_aspect_ratio

    init_bytes = _make_png_bytes(1024, 768)  # ratio 1.333
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(1000, 750)  # ratio 1.333

    _validate_control_image_aspect_ratio(init_bytes, [binding])  # must not raise


def test_validate_control_image_aspect_ratio_rejects_beyond_tolerance():
    from backends.cuda_worker import _validate_control_image_aspect_ratio

    init_bytes = _make_png_bytes(1024, 768)  # ratio 1.333
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(512, 512)  # ratio 1.0

    with pytest.raises(ValueError, match="canny-attachment"):
        _validate_control_image_aspect_ratio(init_bytes, [binding])


def test_partial_controlnet_load_failure_releases_already_pinned_models():
    """If the Nth ControlNet fails to load, the N-1 already pinned into the cache
    must still be released by the finally block — otherwise they leak VRAM."""
    worker = _make_worker(DiffusersCudaWorker)
    req = _make_req()
    job = SimpleNamespace(
        req=req,
        init_image=None,
        controlnet_bindings=[
            _make_binding("canny", 0.4, 0.0, 0.8),   # loads OK
            _make_binding("depth", 0.9, 0.1, 1.0),   # raises during load
        ],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    release_cache = MagicMock()

    def fake_load(binding):
        if binding.model_id == "depth-model":
            raise RuntimeError("boom loading second controlnet")
        return f"loaded:{binding.model_id}"

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker._decode_control_image", return_value=MagicMock()), \
         patch.object(worker, "_load_controlnet_model", side_effect=fake_load), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=release_cache):
        with pytest.raises(RuntimeError, match="boom"):
            worker.run_job(job)

    # The first ControlNet was pinned before the second failed — it must be released.
    release_cache.release.assert_any_call("canny-model")


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
    fake_pipe_cls = MagicMock()
    fake_pipe_cls.from_pipe.return_value = cn_pipe

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.cuda_worker.Image.open", side_effect=[opened_a, opened_b]), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch("backends.cuda_worker._import_attr", return_value=fake_pipe_cls):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    fake_pipe_cls.from_pipe.assert_called_once_with(
        worker.pipe,
        controlnet=["loaded:canny-model", "loaded:depth-model"],
        torch_dtype=None,
    )
    assert worker.pipe.calls == []
    kwargs = cn_pipe.call_args.kwargs
    assert "controlnet" not in kwargs
    assert kwargs["image"] == [resized_a, resized_b]
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
    assert "image" not in kwargs


@pytest.mark.parametrize(
    ("worker_cls", "family", "branch"),
    [
        (DiffusersCudaWorker, "sd15", "txt2img"),
        (DiffusersCudaWorker, "sd15", "txt2img_controlnet"),
        (DiffusersCudaWorker, "sd15", "img2img"),
        (DiffusersCudaWorker, "sd15", "img2img_controlnet"),
        (DiffusersSDXLCudaWorker, "sdxl", "txt2img"),
        (DiffusersSDXLCudaWorker, "sdxl", "txt2img_controlnet"),
        (DiffusersSDXLCudaWorker, "sdxl", "img2img"),
        (DiffusersSDXLCudaWorker, "sdxl", "img2img_controlnet"),
    ],
)
def test_materialized_conditioning_reaches_every_cuda_run_job_target(
    worker_cls,
    family,
    branch,
):
    worker = _make_worker(worker_cls)
    artifact, chain, context = _install_materialized_conditioning(worker, family)
    req = _make_req()
    if family == "sdxl":
        req.size = "1024x1024"
    req.denoise_strength = 0.6
    bindings = []
    init_image = None
    target_pipe = worker.pipe
    controlnet_pipe = MagicMock()
    controlnet_pipe.return_value = SimpleNamespace(images=[MagicMock()])
    img2img_pipe = MagicMock()
    img2img_pipe.return_value = SimpleNamespace(images=[MagicMock()])

    if "controlnet" in branch:
        binding = _make_binding("canny", 0.4, 0.0, 0.8)
        binding.control_image_bytes = _make_png_bytes(1024, 1024)
        bindings = [binding]
    if "img2img" in branch:
        init_image = _make_png_bytes(1024, 1024)
        target_pipe = img2img_pipe
        worker._img2img_pipe = img2img_pipe
    if branch == "txt2img_controlnet":
        target_pipe = controlnet_pipe
        worker._build_controlnet_pipe = Mock(return_value=controlnet_pipe)
    if branch == "img2img_controlnet":
        target_pipe = controlnet_pipe

    job = SimpleNamespace(req=req, init_image=init_image, controlnet_bindings=bindings)
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    fake_pipe_cls = MagicMock()
    fake_pipe_cls.from_pipe.return_value = controlnet_pipe

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch("backends.cuda_worker._import_attr", return_value=fake_pipe_cls):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    assert chain.requests[0][0].prompt == "an owl"
    assert chain.requests[0][0].negative_prompt == "blurry"
    assert chain.requests[0][1] is context
    worker._accept_conditioning_artifact.assert_called_once_with(target_pipe, artifact)
    kwargs = target_pipe.call_args.kwargs if hasattr(target_pipe, "call_args") else target_pipe.calls[0]
    assert "prompt" not in kwargs
    assert "negative_prompt" not in kwargs
    assert kwargs["prompt_embeds"] is artifact.slots["prompt_embeds"]
    assert kwargs["negative_prompt_embeds"] is artifact.slots["negative_prompt_embeds"]
    if family == "sdxl":
        assert kwargs["pooled_prompt_embeds"] is artifact.slots["pooled_prompt_embeds"]
        assert (
            kwargs["negative_pooled_prompt_embeds"]
            is artifact.slots["negative_pooled_prompt_embeds"]
        )


def test_sd15_combined_img2img_controlnet_keeps_init_image_and_control_map_distinct():
    """Regression guard for the image/control_image kwarg collision: the combined
    pipeline's image= is the init image and control_image= is the control map —
    a naive **controlnet_kwargs merge would silently let one overwrite the other."""
    worker = _make_worker(DiffusersCudaWorker)
    req = _make_req()
    req.denoise_strength = 0.6
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(512, 512)
    job = SimpleNamespace(
        req=req,
        init_image=_make_png_bytes(512, 512),
        controlnet_bindings=[binding],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    combined_pipe = MagicMock()
    combined_pipe.return_value = SimpleNamespace(images=[MagicMock()])
    fake_pipe_cls = MagicMock()
    fake_pipe_cls.from_pipe.return_value = combined_pipe

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch("backends.cuda_worker._import_attr", return_value=fake_pipe_cls):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        _, seed = worker.run_job(job)

    assert seed == 123
    fake_pipe_cls.from_pipe.assert_called_once_with(
        worker.pipe, controlnet="loaded:canny-model", torch_dtype=None
    )
    assert worker.pipe.calls == []  # base txt2img pipe must not be invoked
    kwargs = combined_pipe.call_args.kwargs
    assert "controlnet" not in kwargs
    assert kwargs["image"] is not kwargs["control_image"]
    assert kwargs["strength"] == 0.6
    assert kwargs["controlnet_conditioning_scale"] == 0.4
    assert kwargs["control_guidance_start"] == 0.0
    assert kwargs["control_guidance_end"] == 0.8


def test_sd15_combined_path_normalizes_vae_dtype_before_execution():
    """The combined SD1.5 pipeline encodes the init image at prompt_embeds.dtype,
    which diffusers derives from text_encoder.dtype when present. The worker must
    align the shared VAE to that encode dtype, not blindly to self.dtype, or the
    combined path hits the float-vs-half crash seen in ./error."""
    worker = _make_worker(DiffusersCudaWorker)
    worker.pipe.vae = MagicMock()
    worker.pipe.text_encoder = SimpleNamespace(dtype="fp32_sentinel")
    req = _make_req()
    req.denoise_strength = 0.6
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(512, 512)
    job = SimpleNamespace(
        req=req,
        init_image=_make_png_bytes(512, 512),
        controlnet_bindings=[binding],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    combined_pipe = MagicMock()
    combined_pipe.return_value = SimpleNamespace(images=[MagicMock()])

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch.object(
             _FakeStableDiffusionControlNetImg2ImgPipeline, "from_pipe", return_value=combined_pipe
         ):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    worker.pipe.vae.to.assert_called_once_with(worker.device, dtype="fp32_sentinel")


def test_sd15_combined_path_preserves_shared_module_dtypes_across_from_pipe():
    """from_pipe shares module objects with self.pipe. The worker must pass an
    explicit torch_dtype that avoids the diffusers float32 default, or the
    combined path permanently upcasts the shared text encoder / UNet / VAE."""
    worker = _make_worker(DiffusersCudaWorker)
    worker.pipe.text_encoder = SimpleNamespace(dtype="fp16_sentinel")
    worker.pipe.unet = SimpleNamespace(dtype="fp16_sentinel")
    worker.pipe.vae = MagicMock(dtype="fp16_sentinel")
    req = _make_req()
    req.denoise_strength = 0.6
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(512, 512)
    job = SimpleNamespace(
        req=req,
        init_image=_make_png_bytes(512, 512),
        controlnet_bindings=[binding],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()

    class _MutatingCombinedPipe(_FakePipelineBase):
        @classmethod
        def from_pipe(cls, pipe, controlnet, *, torch_dtype=_MISSING):
            if torch_dtype is _MISSING:
                pipe.text_encoder.dtype = "fp32_sentinel"
                pipe.unet.dtype = "fp32_sentinel"
                pipe.vae.dtype = "fp32_sentinel"
            elif torch_dtype is not None:
                pipe.text_encoder.dtype = torch_dtype
                pipe.unet.dtype = torch_dtype
                pipe.vae.dtype = torch_dtype
            return cls()

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch.object(
             _FakeStableDiffusionControlNetImg2ImgPipeline,
             "from_pipe",
             side_effect=_MutatingCombinedPipe.from_pipe,
         ):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    assert worker.pipe.text_encoder.dtype == "fp16_sentinel"
    assert worker.pipe.unet.dtype == "fp16_sentinel"
    assert worker.pipe.vae.dtype == "fp16_sentinel"


def test_sd15_combined_path_rejects_mismatched_aspect_ratio_before_dispatch():
    worker = _make_worker(DiffusersCudaWorker)
    req = _make_req()
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(512, 512)  # ratio 1.0
    job = SimpleNamespace(
        req=req,
        init_image=_make_png_bytes(1024, 768),  # ratio 1.333, diverges > 2%
        controlnet_bindings=[binding],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch.object(
             _FakeStableDiffusionControlNetImg2ImgPipeline, "from_pipe"
         ) as from_pipe:
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        with pytest.raises(ValueError, match="canny-attachment"):
            worker.run_job(job)

    from_pipe.assert_not_called()


def test_sdxl_combined_img2img_controlnet_keeps_init_image_and_control_map_distinct():
    """SDXL mirror of the SD1.5 kwarg-collision regression guard: image= is the
    init image, control_image= is the control map."""
    worker = _make_worker(DiffusersSDXLCudaWorker)
    req = _make_req()
    req.size = "1024x1024"
    req.denoise_strength = 0.6
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(1024, 1024)
    job = SimpleNamespace(
        req=req,
        init_image=_make_png_bytes(1024, 1024),
        controlnet_bindings=[binding],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    combined_pipe = MagicMock()
    combined_pipe.return_value = SimpleNamespace(images=[MagicMock()])

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch.object(
             _FakeStableDiffusionXLControlNetImg2ImgPipeline, "from_pipe", return_value=combined_pipe
         ) as from_pipe:
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        _, seed = worker.run_job(job)

    assert seed == 123
    from_pipe.assert_called_once_with(worker.pipe, controlnet="loaded:canny-model", torch_dtype=None)
    assert worker.pipe.calls == []  # base txt2img pipe must not be invoked
    kwargs = combined_pipe.call_args.kwargs
    assert "controlnet" not in kwargs
    assert kwargs["image"] is not kwargs["control_image"]
    assert kwargs["strength"] == 0.6
    assert kwargs["controlnet_conditioning_scale"] == 0.4
    assert kwargs["control_guidance_start"] == 0.0
    assert kwargs["control_guidance_end"] == 0.8


def test_sdxl_combined_path_normalizes_vae_dtype_before_execution():
    """SDXL mirror of the SD1.5 shared-VAE dtype fix-up: from_pipe shares
    self.pipe's components, so the combined path needs _normalize_img2img_modules()
    just like the plain img2img path (review blocker on STABL-vgbxamoz)."""
    worker = _make_worker(DiffusersSDXLCudaWorker)
    worker.pipe.vae = MagicMock()
    req = _make_req()
    req.size = "1024x1024"
    req.denoise_strength = 0.6
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(1024, 1024)
    job = SimpleNamespace(
        req=req,
        init_image=_make_png_bytes(1024, 1024),
        controlnet_bindings=[binding],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    combined_pipe = MagicMock()
    combined_pipe.return_value = SimpleNamespace(images=[MagicMock()])

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch.object(
             _FakeStableDiffusionXLControlNetImg2ImgPipeline, "from_pipe", return_value=combined_pipe
         ):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    worker.pipe.vae.to.assert_called_once_with(worker.device, dtype=worker.dtype)


def test_sdxl_combined_path_rejects_mismatched_aspect_ratio_before_dispatch():
    worker = _make_worker(DiffusersSDXLCudaWorker)
    req = _make_req()
    req.size = "1024x1024"
    binding = _make_binding("canny", 0.4, 0.0, 0.8)
    binding.control_image_bytes = _make_png_bytes(512, 512)  # ratio 1.0
    job = SimpleNamespace(
        req=req,
        init_image=_make_png_bytes(1024, 768),  # ratio 1.333, diverges > 2%
        controlnet_bindings=[binding],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo"), \
         patch.object(
             _FakeStableDiffusionXLControlNetImg2ImgPipeline, "from_pipe"
         ) as from_pipe:
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        with pytest.raises(ValueError, match="canny-attachment"):
            worker.run_job(job)

    from_pipe.assert_not_called()


def test_normalize_img2img_modules_skips_noop_cast_when_already_aligned():
    """Steady-state guard: when the shared VAE already matches the target
    device/dtype, normalize must not call vae.to() at all — diffusers logs a
    scary 'Casting directly with to()' warning on every .to(), and keeping
    steady-state logs quiet preserves that warning's diagnostic value for
    real casts (the STABL-crdsypux poisoning signature)."""
    worker = _make_worker(DiffusersCudaWorker)
    worker.pipe.text_encoder = SimpleNamespace(dtype="fp16_sentinel")
    vae = MagicMock()
    vae.dtype = "fp16_sentinel"
    vae.device = "cuda:0"
    worker.pipe.vae = vae

    worker._normalize_img2img_modules()

    vae.to.assert_not_called()


def test_normalize_img2img_modules_still_casts_on_dtype_drift():
    worker = _make_worker(DiffusersCudaWorker)
    worker.pipe.text_encoder = SimpleNamespace(dtype="fp16_sentinel")
    vae = MagicMock()
    vae.dtype = "fp32_sentinel"  # drifted
    vae.device = "cuda:0"
    worker.pipe.vae = vae

    worker._normalize_img2img_modules()

    vae.to.assert_called_once_with("cuda:0", dtype="fp16_sentinel")
