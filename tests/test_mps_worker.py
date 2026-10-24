"""Contract tests for the MPS SD1.5 LCM worker (STABL-hvvogszy, STABL-tukrgrjb).

Torch is real, but MPS availability and the Diffusers pipeline classes are
patched, so the file runs on any host. The pipeline-class helpers are patched on
``backends.mps_worker`` because the worker calls them unqualified there. Patching
``backends.cuda_worker`` instead would miss, and the test would load real weights.
"""

import io
import json
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
        self.to = Mock()

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

    @property
    def components(self):
        """Live, like the diffusers property. The worker swaps the scheduler
        after load, so a dict frozen at construction would hand the img2img
        pipeline the pre-LCM scheduler."""
        return {
            "vae": self.vae,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "scheduler": self.scheduler,
        }

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


class _FakeImg2ImgPipe:
    """Stand-in for StableDiffusionImg2ImgPipeline built from shared components."""

    def __init__(self, image=None, **components):
        self.built_from = components
        self.vae = components.get("vae")
        self.scheduler = components.get("scheduler")
        self.calls = []
        self.inference_mode_flags = []
        self.error = None
        self._image = image

    def __call__(self, *, callback_on_step_end=None, **kwargs):
        self.calls.append({"callback_on_step_end": callback_on_step_end, **kwargs})
        self.inference_mode_flags.append(torch.is_inference_mode_enabled())
        if self.error is not None:
            raise self.error
        if callback_on_step_end is not None:
            callback_on_step_end(self, 0, 0, {})
        return SimpleNamespace(images=[self._image])


class _FakeControlNetPipe:
    """Stand-in for StableDiffusionControlNetPipeline.from_pipe()."""

    def __init__(self, image=None, *, base_pipe=None, controlnet=None, torch_dtype=None):
        self.base_pipe = base_pipe
        self.controlnet = controlnet
        self.torch_dtype = torch_dtype
        self.calls = []
        self.error: BaseException | None = None
        self._image = image

    def __call__(self, *, callback_on_step_end=None, **kwargs):
        self.calls.append({"callback_on_step_end": callback_on_step_end, **kwargs})
        if self.error is not None:
            raise self.error
        if callback_on_step_end is not None:
            callback_on_step_end(self, 0, 0, {})
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
    Image.new("RGB", (768, 768), "white").save(buf, format="PNG")
    return buf.getvalue()


def _binding(control_type="canny", attachment_id="cn_1", model_id="sd15-canny"):
    return SimpleNamespace(
        attachment_id=attachment_id,
        control_type=control_type,
        model_id=model_id,
        model_path=f"/models/controlnets/{model_id}",
        control_image_bytes=_control_png(),
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
    )


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

    img2img_pipes = []

    def _build_img2img(**components):
        built = _FakeImg2ImgPipe(image=Image.new("RGB", (768, 768), "blue"), **components)
        img2img_pipes.append(built)
        return built

    img2img_class = Mock(side_effect=_build_img2img)

    controlnet_pipes = []

    class _ControlNetPipeClass:
        """Recording stand-in for StableDiffusionControlNetPipeline."""

        @staticmethod
        def from_pipe(base_pipe, *, controlnet, torch_dtype):
            built = _FakeControlNetPipe(
                image=Image.new("RGB", (768, 768), "green"),
                base_pipe=base_pipe,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
            )
            controlnet_pipes.append(built)
            return built

    # The worker reads its platform scope through the registry, and this file
    # sets no BACKEND, so get_backend_provider() would raise RuntimeError. Patch
    # the registry module: the worker imports the function lazily inside
    # run_job, so the lookup happens there, not on backends.mps_worker.
    import backends.platform_registry as platform_registry
    from backends.platforms.mps import MPSProvider

    monkeypatch.setattr(platform_registry, "get_backend_provider", MPSProvider)

    # The inherited _load_controlnet_model goes through the ControlNet cache.
    # Patch the cache itself, so no loader runs and no real weights load.
    import backends.controlnet_cache as controlnet_cache

    cache = _FakeControlNetCache()
    monkeypatch.setattr(controlnet_cache, "get_controlnet_cache", lambda: cache)

    monkeypatch.setattr(mps_worker, "_sd_pipeline_cls", lambda: pipeline_class)
    monkeypatch.setattr(mps_worker, "_lcm_scheduler_cls", lambda: scheduler_class)
    monkeypatch.setattr(mps_worker, "_sd_img2img_pipeline_cls", lambda: img2img_class)
    monkeypatch.setattr(
        mps_worker, "_sd_controlnet_pipeline_cls", lambda: _ControlNetPipeClass
    )
    return SimpleNamespace(
        pipe=pipe,
        pipeline_class=pipeline_class,
        scheduler_class=scheduler_class,
        lcm_scheduler=lcm_scheduler,
        img2img_class=img2img_class,
        img2img_pipes=img2img_pipes,
        controlnet_pipes=controlnet_pipes,
        controlnet_cache=cache,
    )


def _build_worker(doubles, loader_format="diffusers_dir", family_profile=None):
    """`family_profile` stays None for the text-to-image and img2img tests, which
    never read it. The ControlNet path does read it, through the inherited
    `_build_controlnet_kwargs` and the scope lookup, so those tests pass
    SD15_PROFILE."""
    from backends.mps_worker import DiffusersMpsWorker

    return DiffusersMpsWorker(
        worker_id=0,
        model_path=MODEL_PATH,
        model_info=_model_info(loader_format),
        family_profile=family_profile,
    )


def _build_controlnet_worker(doubles):
    from backends.family_profiles import SD15_PROFILE

    return _build_worker(doubles, family_profile=SD15_PROFILE)


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


# --- img2img (STABL-tukrgrjb) ----------------------------------------------


def _init_png(size=(512, 512), color="green"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_mps_worker_renders_img2img_from_init_image(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker(fake_pipeline)
    png_bytes, seed = worker.run_job(_job(init_image=_init_png()))

    assert seed == 7
    # The text-to-image pipeline stays untouched. Only the img2img pipeline ran.
    assert fake_pipeline.pipe.calls == []
    assert len(fake_pipeline.img2img_pipes) == 1
    img2img = fake_pipeline.img2img_pipes[0]
    assert img2img.built_from == fake_pipeline.pipe.components
    assert worker._img2img_pipe is img2img

    call = img2img.calls[0]
    assert call["image"].size == (768, 768)
    assert call["strength"] == 0.75
    assert call["num_inference_steps"] == 8
    assert call["guidance_scale"] == 3.0
    assert call["prompt"] == "a bright red sailboat"
    assert call["generator"] is recording_generator.instances[-1]
    # The img2img pipeline reads the size from the init image, not from kwargs.
    assert "width" not in call
    assert "height" not in call
    assert img2img.inference_mode_flags == [True]

    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.format == "PNG"
        assert image.size == (768, 768)
        metadata = json.loads(image.info["lcm"])
    assert metadata["seed"] == 7
    assert metadata["scheduler_id"] == "lcm"
    released_cache.assert_called_once_with()


def test_mps_worker_reads_the_request_denoise_strength(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_worker(fake_pipeline)
    worker.run_job(_job(req=_request(denoise_strength=0.4), init_image=_init_png()))

    assert fake_pipeline.img2img_pipes[0].calls[0]["strength"] == 0.4


def test_mps_worker_builds_one_img2img_pipe_for_two_jobs(
    fake_pipeline, recording_generator, released_cache
):
    """The img2img pipeline shares the loaded weights, so it is built once."""
    worker = _build_worker(fake_pipeline)
    worker.run_job(_job(init_image=_init_png()))
    worker.run_job(_job(init_image=_init_png()))

    assert fake_pipeline.img2img_class.call_count == 1
    assert len(fake_pipeline.img2img_pipes[0].calls) == 2


def test_mps_worker_aligns_the_vae_before_the_img2img_encode(
    fake_pipeline, recording_generator, released_cache
):
    """A prior run can upcast the shared VAE. The encode fails if the init-image
    tensor dtype no longer matches the VAE bias dtype."""
    worker = _build_worker(fake_pipeline)
    worker.run_job(_job(init_image=_init_png()))

    fake_pipeline.pipe.vae.to.assert_called_with("mps", dtype=torch.float16)
    assert fake_pipeline.img2img_pipes[0].vae is fake_pipeline.pipe.vae


def test_mps_worker_cancels_img2img_at_step_boundary(
    fake_pipeline, recording_generator, released_cache
):
    from concurrent.futures import CancelledError

    worker = _build_worker(fake_pipeline)
    with pytest.raises(CancelledError, match="job cancelled at step boundary"):
        worker.run_job(_job(init_image=_init_png()), should_cancel=lambda: True)
    released_cache.assert_called_once_with()


def test_mps_worker_rejects_an_init_image_with_controlnet_bindings(
    fake_pipeline, recording_generator, released_cache
):
    """The combined operation stays rejected. The cell reports it as false."""
    worker = _build_worker(fake_pipeline)
    with pytest.raises(
        NotImplementedError,
        match="MPS SD1.5 does not support the img2img\\+ControlNet operation",
    ):
        worker.run_job(
            _job(
                init_image=_init_png(),
                controlnet_bindings=[SimpleNamespace(model_id="canny")],
            )
        )
    assert fake_pipeline.img2img_pipes == []
    released_cache.assert_not_called()


def test_mps_worker_renders_one_canny_controlnet(
    fake_pipeline, recording_generator, released_cache
):
    """The ControlNet pipe comes from self.pipe through from_pipe, so the render
    costs no second copy of the weights."""
    worker = _build_controlnet_worker(fake_pipeline)
    png, seed = worker.run_job(_job(controlnet_bindings=[_binding()]))

    assert png.startswith(b"\x89PNG")
    assert seed == 7
    assert len(fake_pipeline.controlnet_pipes) == 1
    built = fake_pipeline.controlnet_pipes[0]
    assert built.base_pipe is fake_pipeline.pipe
    assert built.torch_dtype is None
    # The base text-to-image pipe must not run for a ControlNet job.
    assert fake_pipeline.pipe.calls == []
    call = built.calls[0]
    assert call["image"].size == (768, 768)
    assert call["controlnet_conditioning_scale"] == 1.0
    assert call["control_guidance_start"] == 0.0
    assert call["control_guidance_end"] == 1.0
    assert fake_pipeline.controlnet_cache.acquired == ["sd15-canny"]
    assert fake_pipeline.controlnet_cache.released == ["sd15-canny"]
    released_cache.assert_called_once_with()


def test_mps_worker_rejects_a_depth_controlnet_binding(
    fake_pipeline, recording_generator, released_cache
):
    """Defense in depth. Admission rejects it first, from the same scope."""
    worker = _build_controlnet_worker(fake_pipeline)
    with pytest.raises(ValueError, match="depth"):
        worker.run_job(_job(controlnet_bindings=[_binding(control_type="depth")]))
    assert fake_pipeline.controlnet_pipes == []
    assert fake_pipeline.controlnet_cache.acquired == []
    released_cache.assert_not_called()


def test_mps_worker_rejects_two_controlnet_bindings(
    fake_pipeline, recording_generator, released_cache
):
    """The mode permits two attachments. The platform scope permits one."""
    worker = _build_controlnet_worker(fake_pipeline)
    bindings = [_binding(attachment_id="cn_1"), _binding(attachment_id="cn_2")]
    with pytest.raises(ValueError, match="1"):
        worker.run_job(_job(controlnet_bindings=bindings))
    assert fake_pipeline.controlnet_pipes == []
    assert fake_pipeline.controlnet_cache.acquired == []
    released_cache.assert_not_called()


def test_mps_worker_releases_the_controlnet_pin_after_a_failed_render(
    fake_pipeline, recording_generator, released_cache, monkeypatch
):
    """A mid-pipeline raise must not leak the ControlNet weights."""
    import backends.mps_worker as mps_worker

    worker = _build_controlnet_worker(fake_pipeline)

    def _exploding_from_pipe(base_pipe, *, controlnet, torch_dtype):
        built = _FakeControlNetPipe(base_pipe=base_pipe, controlnet=controlnet,
                                   torch_dtype=torch_dtype)
        built.error = RuntimeError("controlnet exploded")
        fake_pipeline.controlnet_pipes.append(built)
        return built

    # monkeypatch, not assignment: a leaked module attribute would reach every
    # later test in the session.
    monkeypatch.setattr(
        mps_worker,
        "_sd_controlnet_pipeline_cls",
        lambda: SimpleNamespace(from_pipe=_exploding_from_pipe),
    )
    with pytest.raises(RuntimeError, match="controlnet exploded"):
        worker.run_job(_job(controlnet_bindings=[_binding()]))

    assert fake_pipeline.controlnet_cache.acquired == ["sd15-canny"]
    assert fake_pipeline.controlnet_cache.released == ["sd15-canny"]
    released_cache.assert_called_once_with()


def test_mps_worker_cancels_a_controlnet_render_at_a_step_boundary(
    fake_pipeline, recording_generator, released_cache
):
    from concurrent.futures import CancelledError

    worker = _build_controlnet_worker(fake_pipeline)
    with pytest.raises(CancelledError, match="job cancelled at step boundary"):
        worker.run_job(
            _job(controlnet_bindings=[_binding()]), should_cancel=lambda: True
        )
    assert fake_pipeline.controlnet_cache.released == ["sd15-canny"]
    released_cache.assert_called_once_with()


def test_mps_worker_writes_the_controlnet_png_chunk(
    fake_pipeline, recording_generator, released_cache
):
    worker = _build_controlnet_worker(fake_pipeline)
    png, _ = worker.run_job(_job(controlnet_bindings=[_binding()]))

    entries = json.loads(Image.open(io.BytesIO(png)).text["controlnet"])
    assert [entry["attachment_id"] for entry in entries] == ["cn_1"]
    assert entries[0]["control_type"] == "canny"
    assert entries[0]["generation"]["model_id"] == "sd15-canny"


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


def test_the_load_log_reports_no_style_api(fake_pipeline, caplog):
    """The MPS worker runs no style-LoRA load loop, so it never reaches the
    CUDA style-API detection and the field kept its `unknown` default. The
    HunyuanDiT worker has the same shape and states `none` for the same reason.
    `unknown` reads as a failed probe in a log line that is otherwise fact."""
    doubles = fake_pipeline

    with caplog.at_level("INFO", logger="backends.mps_worker"):
        worker = _build_worker(doubles)

    assert worker._style_api == "none"
    assert "style_api=none" in caplog.text
    assert "style_api=unknown" not in caplog.text
