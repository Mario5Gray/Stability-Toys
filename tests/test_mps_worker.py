"""Contract tests for the MPS SD1.5 LCM worker (STABL-hvvogszy).

Torch is real, but MPS availability and the Diffusers pipeline classes are
patched, so the file runs on any host. The pipeline-class helpers are patched on
``backends.mps_worker`` because the worker calls them unqualified there. Patching
``backends.cuda_worker`` instead would miss, and the test would load real weights.
"""

import io
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

MODEL_PATH = "/node/models/diffusers/lcm-dreamshaper-v7"


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
        self.config = config if config is not None else {"scheduler": "base"}


class _FakePipe:
    """Minimal stand-in for a StableDiffusionPipeline."""

    def __init__(self, image=None):
        self.vae = _FakeVae()
        self.unet = SimpleNamespace(config=SimpleNamespace(cross_attention_dim=768))
        self.text_encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=768))
        self.scheduler = _FakeScheduler()
        self.to = Mock(side_effect=lambda device: self)
        self.calls = []
        self.inference_mode_flags = []
        self.error = None
        self._image = image
        self.disable_lora = Mock()

    # Python looks up __call__ on the type, so a per-test override must go
    # through attributes set here, not through assignment to pipe.__call__.
    def __call__(self, *, callback_on_step_end=None, **kwargs):
        self.calls.append({"callback_on_step_end": callback_on_step_end, **kwargs})
        self.inference_mode_flags.append(torch.is_inference_mode_enabled())
        if self.error is not None:
            raise self.error
        if callback_on_step_end is not None:
            callback_on_step_end(self, 0, 0, {})
        return SimpleNamespace(images=[self._image])


def _model_info(loader_format="diffusers_dir"):
    return SimpleNamespace(
        loader_format=loader_format,
        checkpoint_precision="fp16",
        checkpoint_variant="unknown",
        scheduler_profile="lcm",
        default_scheduler_id="lcm",
        allowed_scheduler_ids=["lcm", "euler"],
        runtime_offload=None,
        runtime_quantize=None,
        runtime_enable_xformers=None,
        runtime_attention_slicing=None,
        metadata={},
    )


@pytest.fixture
def mps_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.fixture
def fake_pipeline(monkeypatch, mps_available):
    """Patch both pipeline-class helpers and hand back the recording doubles."""
    import backends.mps_worker as mps_worker

    pipe = _FakePipe(image=Image.new("RGB", (768, 768), "red"))
    pipeline_class = SimpleNamespace(
        from_pretrained=Mock(return_value=pipe),
        from_single_file=Mock(return_value=pipe),
    )
    lcm_scheduler = _FakeScheduler({"scheduler": "lcm"})
    scheduler_class = SimpleNamespace(from_config=Mock(return_value=lcm_scheduler))
    monkeypatch.setattr(mps_worker, "_sd_pipeline_cls", lambda: pipeline_class)
    monkeypatch.setattr(mps_worker, "_lcm_scheduler_cls", lambda: scheduler_class)
    return SimpleNamespace(
        pipe=pipe,
        pipeline_class=pipeline_class,
        scheduler_class=scheduler_class,
        lcm_scheduler=lcm_scheduler,
    )


def _build_worker(doubles, loader_format="diffusers_dir"):
    from backends.mps_worker import DiffusersMpsWorker

    return DiffusersMpsWorker(
        worker_id=0,
        model_path=MODEL_PATH,
        model_info=_model_info(loader_format),
        family_profile=None,
    )


def test_mps_worker_loads_lcm_on_mps(fake_pipeline):
    from backends.platforms.mps_policy import MPS_TORCH_POLICY

    doubles = fake_pipeline
    worker = _build_worker(doubles)
    pipe = doubles.pipe

    assert worker.device == "mps"
    assert worker.dtype is torch.float16
    assert worker.execution_policy is MPS_TORCH_POLICY
    doubles.pipeline_class.from_pretrained.assert_called_once_with(
        "/node/models/diffusers/lcm-dreamshaper-v7",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to.assert_called_once_with("mps")
    assert worker.pipe is pipe
    doubles.scheduler_class.from_config.assert_called_once_with({"scheduler": "base"})
    assert pipe.scheduler is doubles.lcm_scheduler
    assert pipe.vae.tiling is True
    assert pipe.vae.slicing is True
    doubles.pipeline_class.from_single_file.assert_not_called()


def test_mps_worker_loads_single_file_checkpoint(fake_pipeline):
    doubles = fake_pipeline
    worker = _build_worker(doubles, loader_format="single_file")

    doubles.pipeline_class.from_single_file.assert_called_once_with(
        "/node/models/diffusers/lcm-dreamshaper-v7",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    doubles.pipeline_class.from_pretrained.assert_not_called()
    assert worker.pipe is doubles.pipe


def _request(**overrides):
    base = dict(
        prompt="a bright red sailboat",
        negative_prompt=None,
        scheduler_id=None,
        size="768x768",
        num_inference_steps=8,
        guidance_scale=3.0,
        seed=7,
        style_lora=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _job(req=None, init_image=None, controlnet_bindings=None):
    return SimpleNamespace(
        req=req if req is not None else _request(),
        init_image=init_image,
        controlnet_bindings=controlnet_bindings if controlnet_bindings is not None else [],
    )


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


def test_mps_worker_renders_png_with_lcm_metadata(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker(fake_pipeline)
    png_bytes, seed = worker.run_job(_job())

    assert seed == 7
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.format == "PNG"
        assert image.size == (768, 768)
        metadata = json.loads(image.info["lcm"])
    assert metadata == {
        "prompt": "a bright red sailboat",
        "seed": 7,
        "size": "768x768",
        "steps": 8,
        "cfg": 3.0,
        "negative_prompt": None,
        "scheduler_id": "lcm",
    }
    call = fake_pipeline.pipe.calls[0]
    assert (call["width"], call["height"]) == (768, 768)
    assert call["num_inference_steps"] == 8
    assert call["guidance_scale"] == 3.0
    assert call["prompt"] == "a bright red sailboat"
    released_cache.assert_called_once_with()


def test_mps_worker_seeds_an_mps_generator(fake_pipeline, recording_generator, released_cache):
    worker = _build_worker(fake_pipeline)
    worker.run_job(_job())

    generator = recording_generator.instances[-1]
    assert generator.device == "mps"
    assert generator.seed == 7
    assert fake_pipeline.pipe.calls[0]["generator"] is generator


def test_mps_worker_runs_inference_under_inference_mode(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker(fake_pipeline)
    worker.run_job(_job())

    assert fake_pipeline.pipe.inference_mode_flags == [True]


def test_mps_worker_rejects_non_lcm_scheduler(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker(fake_pipeline)
    with pytest.raises(ValueError, match="MPS SD1.5 supports LCM scheduler only"):
        worker.run_job(_job(req=_request(scheduler_id="euler")))
    assert fake_pipeline.pipe.calls == []


def test_mps_worker_rejects_init_image(fake_pipeline, recording_generator, released_cache):
    worker = _build_worker(fake_pipeline)
    with pytest.raises(ValueError, match="MPS SD1.5 does not support img2img"):
        worker.run_job(_job(init_image=b"not-an-image"))
    assert fake_pipeline.pipe.calls == []
    released_cache.assert_not_called()


def test_mps_worker_rejects_controlnet_bindings(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker(fake_pipeline)
    with pytest.raises(NotImplementedError, match="MPS SD1.5 does not support ControlNet"):
        worker.run_job(_job(controlnet_bindings=[SimpleNamespace(model_id="canny")]))
    assert fake_pipeline.pipe.calls == []
    released_cache.assert_not_called()


def test_mps_worker_cancels_at_step_boundary(
    fake_pipeline, recording_generator, released_cache
):
    from concurrent.futures import CancelledError

    worker = _build_worker(fake_pipeline)
    with pytest.raises(CancelledError, match="job cancelled at step boundary"):
        worker.run_job(_job(), should_cancel=lambda: True)
    released_cache.assert_called_once_with()


def test_mps_worker_releases_cache_after_pipeline_failure(
    fake_pipeline, recording_generator, released_cache
):
    fake_pipeline.pipe.error = RuntimeError("pipeline exploded")
    worker = _build_worker(fake_pipeline)
    with pytest.raises(RuntimeError, match="pipeline exploded"):
        worker.run_job(_job())
    released_cache.assert_called_once_with()
