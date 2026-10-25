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


def test_hunyuan_mps_worker_rejects_img2img_before_the_pipe_call(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker()

    with pytest.raises(ValueError, match="does not support the img2img operation"):
        worker.run_job(_job(init_image=b"not-an-image"))

    assert fake_pipeline.pipe.calls == []


def test_hunyuan_mps_worker_rejects_controlnet_before_the_pipe_call(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker()
    binding = SimpleNamespace(
        attachment_id="cn_1",
        control_type="canny",
        model_id="hunyuandit-canny",
        model_path="/node/models/controlnets/HunyuanDiT-v1.1-ControlNet-Canny",
        control_image_bytes=b"not-an-image",
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
    )

    with pytest.raises(NotImplementedError, match="does not support the controlnet operation"):
        worker.run_job(_job(controlnet_bindings=[binding]))

    assert fake_pipeline.pipe.calls == []
