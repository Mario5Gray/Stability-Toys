"""Contract tests for the MPS HunyuanDiT text-to-image worker (STABL-appbjuws).

Torch is real, but MPS availability and the Diffusers pipeline class are
patched, so the file runs on any host. The pipeline-class helper is patched on
``backends.cuda_worker`` because the inherited ``__init__`` calls it there.
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

MODEL_PATH = "/node/models/diffusers/HunyuanDiT-v1.1-Diffusers"


class _FakeVae:
    def __init__(self):
        self.tiling = False
        self.slicing = False

    def enable_tiling(self):
        self.tiling = True

    def enable_slicing(self):
        self.slicing = True


class _FakeScheduler:
    def __init__(self, config=None):
        self.config = config if config is not None else {"scheduler": "ddpm"}

    @classmethod
    def from_config(cls, config):
        return cls(config)


class _FakeHunyuanPipe:
    """Minimal stand-in for HunyuanDiTPipeline."""

    def __init__(self, image=None):
        self.vae = _FakeVae()
        self.transformer = SimpleNamespace()
        self.scheduler = _FakeScheduler()
        self.to = Mock(side_effect=lambda device: self)
        self.calls = []
        self.inference_mode_flags = []
        self._image = image

    def __call__(self, *, callback_on_step_end=None, **kwargs):
        self.calls.append({"callback_on_step_end": callback_on_step_end, **kwargs})
        self.inference_mode_flags.append(torch.is_inference_mode_enabled())
        if callback_on_step_end is not None:
            callback_on_step_end(self, 0, 0, {})
        return SimpleNamespace(images=[self._image])


class _RecordingGenerator:
    """Stand-in for torch.Generator, so the test needs no live MPS device."""

    instances = []

    def __init__(self, device=None):
        self.device = device
        self.seed = None
        _RecordingGenerator.instances.append(self)

    def manual_seed(self, seed):
        self.seed = seed
        return self


def _model_info():
    return SimpleNamespace(
        loader_format="diffusers_dir",
        checkpoint_precision="fp16",
        checkpoint_variant="unknown",
        scheduler_profile="native",
        default_scheduler_id=None,
        allowed_scheduler_ids=None,
        runtime_offload=None,
        runtime_quantize=None,
        runtime_enable_xformers=None,
        runtime_attention_slicing=None,
        metadata={},
    )


def _request(**overrides):
    base = dict(
        prompt="a red apple on a wooden table",
        negative_prompt=None,
        scheduler_id=None,
        size="1024x1024",
        num_inference_steps=4,
        guidance_scale=6.0,
        seed=11,
        style_lora=None,
        denoise_strength=0.75,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _job(req=None, init_image=None, controlnet_bindings=None):
    return SimpleNamespace(
        req=req if req is not None else _request(),
        init_image=init_image,
        controlnet_bindings=controlnet_bindings if controlnet_bindings is not None else [],
    )


@pytest.fixture
def mps_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.fixture
def fake_pipeline(monkeypatch, mps_available):
    import backends.cuda_worker as cuda_worker

    pipe = _FakeHunyuanPipe(image=Image.new("RGB", (1024, 1024), "red"))
    pipeline_class = SimpleNamespace(from_pretrained=Mock(return_value=pipe))
    monkeypatch.setattr(cuda_worker, "_hunyuandit_pipeline_cls", lambda: pipeline_class)
    return SimpleNamespace(pipe=pipe, pipeline_class=pipeline_class)


@pytest.fixture
def recording_generator(monkeypatch):
    _RecordingGenerator.instances = []
    monkeypatch.setattr(torch, "Generator", _RecordingGenerator)
    return _RecordingGenerator


@pytest.fixture
def released_cache(monkeypatch):
    from backends.platforms.mps_policy import MPS_TORCH_POLICY

    release = Mock()
    monkeypatch.setattr(MPS_TORCH_POLICY, "release_cache", release)
    return release


def _build_worker():
    from backends.family_profiles import HUNYUANDIT_PROFILE
    from backends.hunyuan_mps_worker import DiffusersHunyuanDiTMpsWorker

    return DiffusersHunyuanDiTMpsWorker(
        worker_id=0,
        model_path=MODEL_PATH,
        model_info=_model_info(),
        family_profile=HUNYUANDIT_PROFILE,
    )


def test_hunyuan_mps_worker_loads_fp16_on_mps(fake_pipeline):
    from backends.platforms.mps_policy import MPS_TORCH_POLICY

    worker = _build_worker()

    assert worker.device == "mps"
    assert worker.dtype is torch.float16
    assert worker.execution_policy is MPS_TORCH_POLICY
    fake_pipeline.pipeline_class.from_pretrained.assert_called_once_with(
        MODEL_PATH, torch_dtype=torch.float16
    )
    fake_pipeline.pipe.to.assert_called_once_with("mps")
    assert fake_pipeline.pipe.vae.tiling is True
    assert fake_pipeline.pipe.vae.slicing is True


def test_hunyuan_mps_worker_renders_text_to_image_png(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker()

    png_bytes, seed = worker.run_job(_job())

    assert seed == 11
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.format == "PNG"
        metadata = json.loads(image.info["lcm"])
    assert metadata == {
        "prompt": "a red apple on a wooden table",
        "seed": 11,
        "size": "1024x1024",
        "steps": 4,
        "cfg": 6.0,
        "negative_prompt": None,
        "scheduler_id": None,
    }
    call = fake_pipeline.pipe.calls[0]
    assert call["width"] == 1024
    assert call["height"] == 1024
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 6.0
    assert call["use_resolution_binning"] is True
    assert "control_image" not in call
    assert fake_pipeline.pipe.inference_mode_flags == [True]
    released_cache.assert_called_once_with()


def test_hunyuan_mps_worker_streams_step_progress(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker()
    seen = []

    worker.run_job(_job(), progress=lambda step, total, stage: seen.append((step, total, stage)))

    call = fake_pipeline.pipe.calls[0]
    assert call["callback_on_step_end"] is not None
    assert seen == [(1, 4, "denoise")]


def test_hunyuan_mps_worker_seeds_an_mps_generator(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker()

    worker.run_job(_job())

    generator = recording_generator.instances[-1]
    assert generator.device == "mps"
    assert generator.seed == 11


def _init_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1024, 1024), "green").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_img2img(monkeypatch):
    """Stand-in for the repo-owned Hunyuan img2img loop."""
    import backends.hunyuandit_img2img as img2img

    calls = []

    def _run(pipe, image, conditioning, **kwargs):
        calls.append({"pipe": pipe, "image": image, "conditioning": conditioning, **kwargs})
        return [Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")]

    monkeypatch.setattr(img2img, "run_hunyuan_img2img", _run)
    return SimpleNamespace(calls=calls)


def test_hunyuan_mps_worker_runs_img2img_through_the_shared_loop(
    fake_pipeline, fake_img2img, recording_generator, released_cache, mps_provider
):
    worker = _build_worker()
    job = _job(init_image=_init_png(), req=_request(denoise_strength=0.6))

    png_bytes, seed = worker.run_job(job)

    assert seed == 11
    call = fake_img2img.calls[0]
    assert call["strength"] == 0.6
    assert call["steps"] == 4
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["image"].size == (1024, 1024)
    with Image.open(io.BytesIO(png_bytes)) as image:
        metadata = json.loads(image.info["lcm"])
    assert metadata["denoise_strength"] == 0.6
    assert metadata["seed"] == 11


class _FakeHunyuanControlNetPipe:
    """Stand-in for HunyuanDiTControlNetPipeline.from_pipe()."""

    def __init__(self, image=None, *, base_pipe=None, controlnet=None, torch_dtype=None):
        self.base_pipe = base_pipe
        self.controlnet = controlnet
        self.torch_dtype = torch_dtype
        self.calls = []
        self._image = image

    def __call__(self, *, callback_on_step_end=None, **kwargs):
        self.calls.append({"callback_on_step_end": callback_on_step_end, **kwargs})
        return SimpleNamespace(images=[self._image])


class _FakeControlNetCache:
    """Records pins so a test can prove the release happens."""

    def __init__(self):
        self.acquired = []
        self.released = []

    def acquire(self, model_id, model_path, *, loader):
        self.acquired.append(model_id)
        return SimpleNamespace(model_id=model_id, model_path=model_path)

    def release(self, model_id):
        self.released.append(model_id)


def _control_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1024, 1024), "white").save(buf, format="PNG")
    return buf.getvalue()


def _canny_binding(**overrides):
    fields = dict(
        attachment_id="cn_1",
        control_type="canny",
        model_id="hunyuandit-canny",
        model_path="/node/models/controlnets/HunyuanDiT-v1.1-ControlNet-Canny",
        control_image_bytes=_control_png(),
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def mps_provider(monkeypatch):
    """Resolve the MPS binding table without setting BACKEND.

    The worker reads its cell through `get_backend_provider()`, and this file
    sets no BACKEND, so the registry module is patched to the MPS provider.
    """
    import backends.platform_registry as platform_registry
    from backends.platforms.mps import MPSProvider

    monkeypatch.setattr(platform_registry, "get_backend_provider", MPSProvider)


@pytest.fixture
def canny_controlnet(monkeypatch, fake_pipeline, mps_provider):
    """Patch the ControlNet pipeline class and its cache."""
    import backends.cuda_worker as cuda_worker
    import backends.controlnet_cache as controlnet_cache

    built_pipes = []

    class _ControlNetPipeClass:
        @staticmethod
        def from_pipe(base_pipe, *, controlnet, torch_dtype):
            built = _FakeHunyuanControlNetPipe(
                image=Image.new("RGB", (1024, 1024), "green"),
                base_pipe=base_pipe,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
            )
            built_pipes.append(built)
            return built

    cache = _FakeControlNetCache()
    monkeypatch.setattr(controlnet_cache, "get_controlnet_cache", lambda: cache)
    monkeypatch.setattr(
        cuda_worker, "_hunyuandit_controlnet_pipeline_cls", lambda: _ControlNetPipeClass
    )
    return SimpleNamespace(pipes=built_pipes, cache=cache)


def test_hunyuan_mps_worker_renders_text_to_image_with_one_canny_binding(
    fake_pipeline, canny_controlnet, recording_generator, released_cache
):
    worker = _build_worker()

    png_bytes, seed = worker.run_job(_job(controlnet_bindings=[_canny_binding()]))

    assert seed == 11
    # The control-map path composes a ControlNet pipeline; it does not call the
    # base pipe directly and it does not enter the repo-owned img2img loop.
    assert fake_pipeline.pipe.calls == []
    assert len(canny_controlnet.pipes) == 1
    pipe = canny_controlnet.pipes[0]
    assert pipe.base_pipe is fake_pipeline.pipe
    assert pipe.controlnet.model_id == "hunyuandit-canny"
    # torch_dtype=None skips the fp32 recast that would OOM (STABL-kfekehhc).
    assert pipe.torch_dtype is None
    call = pipe.calls[0]
    assert isinstance(call["control_image"], Image.Image)
    assert call["control_image"].size == (1024, 1024)
    assert call["controlnet_conditioning_scale"] == 1.0
    # HunyuanDiTControlNetPipeline has no per-step guidance window.
    assert "control_guidance_start" not in call
    assert "control_guidance_end" not in call
    assert call["num_inference_steps"] == 4
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.format == "PNG"
        controlnet_chunk = json.loads(image.info["controlnet"])
    assert controlnet_chunk[0]["control_type"] == "canny"
    assert controlnet_chunk[0]["generation"]["model_id"] == "hunyuandit-canny"
    assert canny_controlnet.cache.acquired == ["hunyuandit-canny"]
    assert canny_controlnet.cache.released == ["hunyuandit-canny"]


def test_hunyuan_mps_worker_rejects_combined_img2img_and_canny_before_the_pipe_call(
    fake_pipeline, canny_controlnet, recording_generator, released_cache
):
    worker = _build_worker()

    with pytest.raises(
        ValueError, match="does not support the img2img\\+controlnet operation"
    ):
        worker.run_job(
            _job(init_image=_init_png(), controlnet_bindings=[_canny_binding()])
        )

    assert canny_controlnet.cache.acquired == []
    assert canny_controlnet.pipes == []
    assert fake_pipeline.pipe.calls == []


def test_hunyuan_mps_worker_rejects_a_second_canny_attachment(
    fake_pipeline, canny_controlnet, recording_generator, released_cache
):
    worker = _build_worker()
    bindings = [_canny_binding(), _canny_binding(attachment_id="cn_2")]

    with pytest.raises(ValueError, match="admits at most 1 controlnet attachment"):
        worker.run_job(_job(controlnet_bindings=bindings))

    assert canny_controlnet.cache.acquired == []


def test_hunyuan_mps_worker_rejects_a_depth_control_type(
    fake_pipeline, canny_controlnet, recording_generator, released_cache
):
    worker = _build_worker()
    binding = _canny_binding(control_type="depth", model_id="hunyuandit-depth")

    with pytest.raises(
        ValueError, match="does not support control_type 'depth'; allowed: canny"
    ):
        worker.run_job(_job(controlnet_bindings=[binding]))

    assert canny_controlnet.cache.acquired == []


def test_hunyuan_mps_worker_rejects_control_timing_windows(
    fake_pipeline, canny_controlnet, recording_generator, released_cache
):
    worker = _build_worker()
    binding = _canny_binding(start_percent=0.2, end_percent=0.8)

    with pytest.raises(ValueError, match="does not support control timing windows"):
        worker.run_job(_job(controlnet_bindings=[binding]))

    assert canny_controlnet.cache.acquired == []
    assert canny_controlnet.pipes == []
