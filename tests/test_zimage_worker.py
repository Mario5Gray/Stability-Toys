"""Unit tests for the Z-Image CUDA worker.

These stub the heavy diffusers and torch stack. No test loads real weights.
Each test patches the lazy resolvers and ``load_int8_text_encoder`` in
``backends.zimage_cuda_worker``, where the call happens.
"""

import concurrent.futures
import importlib
import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image as PILImage

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
    "transformers",
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

sys.modules["diffusers.schedulers.scheduling_lcm"].LCMScheduler = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion"].StableDiffusionPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl"].StableDiffusionXLPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img"].StableDiffusionImg2ImgPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img"].StableDiffusionXLImg2ImgPipeline = MagicMock
sys.modules["backends.styles"].STYLE_REGISTRY = {}

_ENV_KEYS = (
    "CUDA_DTYPE",
    "CUDA_OFFLOAD",
    "CUDA_QUANTIZE",
    "CUDA_ATTENTION_SLICING",
    "CUDA_ENABLE_XFORMERS",
)


class _FakePipelineBase:
    def __init__(self):
        self.scheduler = SimpleNamespace(name="baseline")
        self.vae = MagicMock()
        self.transformer = MagicMock()
        self.text_encoder = MagicMock()
        self.unet = MagicMock()
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    def enable_attention_slicing(self, *args, **kwargs):
        self.calls.append(("enable_attention_slicing", args))

    def enable_xformers_memory_efficient_attention(self, *args, **kwargs):
        self.calls.append(("enable_xformers_memory_efficient_attention", args))

    def __call__(self, callback_on_step_end=None, **kwargs):
        # The signature lets inject_step_progress add a step callback.
        self.calls.append(kwargs)
        if callback_on_step_end is not None:
            callback_on_step_end(self, 0, 0, {})
        return SimpleNamespace(images=[PILImage.new("RGB", (8, 8))])


def _fake_pipeline_cls():
    class _FakeZImage(_FakePipelineBase):
        load_calls: list = []

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.load_calls.append((path, kwargs))
            return cls()

    _FakeZImage.load_calls = []
    return _FakeZImage


def _model_info(**overrides):
    base = dict(
        loader_format="diffusers_dir",
        scheduler_profile="native",
        runtime_quantize=None,
        runtime_offload=None,
        runtime_attention_slicing=None,
        runtime_enable_xformers=None,
        text_encoder_quantize=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def zw():
    return importlib.import_module("backends.zimage_cuda_worker")


@pytest.fixture
def model_dir(tmp_path):
    """A minimal valid Diffusers folder: a directory with model_index.json."""
    root = tmp_path / "repos" / "Z-Image-Turbo-BF16"
    root.mkdir(parents=True)
    (root / "model_index.json").write_text('{"_class_name": "ZImagePipeline"}')
    return str(root)


@pytest.fixture
def fake_load(zw, model_dir):
    """Patch the pipeline resolvers and the encoder store. Yield the handles."""
    pipe_cls = _fake_pipeline_cls()
    encoder = object()
    with patch.object(zw, "_zimage_pipeline_cls", return_value=pipe_cls), \
         patch.object(zw, "_zimage_img2img_pipeline_cls", return_value=MagicMock()), \
         patch.object(zw, "load_int8_text_encoder", return_value=encoder) as store:
        yield SimpleNamespace(pipe_cls=pipe_cls, encoder=encoder, store=store, model_dir=model_dir)


def _build(zw, model_path, **info):
    return zw.DiffusersZImageCudaWorker(
        worker_id=0, model_path=model_path, model_info=_model_info(**info)
    )


# --- Lazy import -----------------------------------------------------------


def test_selecting_sd_family_does_not_import_zimage_worker(monkeypatch):
    import backends.cuda_worker as cw
    from backends.family_profiles import SD15_PROFILE
    from backends.model_resolution import LocalModelBinding, build_resolved, hub_ref
    from backends.worker_factory import create_cuda_worker
    from utils.model_detector import ModelInfo, ModelVariant

    monkeypatch.delitem(sys.modules, "backends.zimage_cuda_worker", raising=False)

    class _FakeSD(_FakePipelineBase):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    info = ModelInfo(
        path="/host/only",
        variant=ModelVariant.SD15,
        cross_attention_dim=768,
        base_arch="unet",
        loader_format="diffusers_dir",
        scheduler_profile="native",
    )
    resolved = build_resolved(
        model_ref=hub_ref("org/repo", None), raw_info=info, profile=SD15_PROFILE, info=info
    )
    with patch.object(cw, "_sd_pipeline_cls", return_value=_FakeSD), \
         patch.object(cw, "STYLE_REGISTRY", {}):
        create_cuda_worker(0, resolved, LocalModelBinding("/models/sd15"))

    assert "backends.zimage_cuda_worker" not in sys.modules


# --- Dependency preflight --------------------------------------------------


def test_preflight_failure_reports_versions(zw, model_dir):
    with patch.object(zw, "_zimage_pipeline_cls", side_effect=ImportError("no ZImagePipeline")), \
         patch.object(zw, "load_int8_text_encoder") as store:
        with pytest.raises(zw.ZImageDependencyError) as info:
            _build(zw, model_dir)
    message = str(info.value)
    assert "diffusers" in message
    assert "transformers" in message
    assert "no ZImagePipeline" in message
    store.assert_not_called()


def test_preflight_checks_img2img_class(zw, model_dir):
    with patch.object(zw, "_zimage_pipeline_cls", return_value=_fake_pipeline_cls()), \
         patch.object(zw, "_zimage_img2img_pipeline_cls", side_effect=ImportError("no img2img")), \
         patch.object(zw, "load_int8_text_encoder"):
        with pytest.raises(zw.ZImageDependencyError, match="no img2img"):
            _build(zw, model_dir)


# --- Admission -------------------------------------------------------------


def _assert_no_load(fake_load):
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


@pytest.mark.parametrize("hint", ["diffusers_dir", "unknown"])
def test_missing_path_is_refused_whatever_the_hint(zw, fake_load, tmp_path, hint):
    missing = tmp_path / "missing" / "model.safetensors"
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="Diffusers folder"):
        _build(zw, str(missing), loader_format=hint)
    _assert_no_load(fake_load)


@pytest.mark.parametrize("hint", ["diffusers_dir", "unknown"])
def test_regular_file_path_is_refused_whatever_the_hint(zw, fake_load, tmp_path, hint):
    single = tmp_path / "model.safetensors"
    single.write_bytes(b"weights")
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="Diffusers folder"):
        _build(zw, str(single), loader_format=hint)
    _assert_no_load(fake_load)


@pytest.mark.parametrize("hint", ["diffusers_dir", "unknown"])
def test_folder_without_model_index_is_refused_whatever_the_hint(zw, fake_load, tmp_path, hint):
    folder = tmp_path / "Z-Image-Turbo-BF16"
    (folder / "transformer").mkdir(parents=True)
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="model_index.json"):
        _build(zw, str(folder), loader_format=hint)
    _assert_no_load(fake_load)


def test_model_index_that_is_a_folder_is_refused(zw, fake_load, tmp_path):
    folder = tmp_path / "Z-Image-Turbo-BF16"
    (folder / "model_index.json").mkdir(parents=True)
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="model_index.json"):
        _build(zw, str(folder))
    _assert_no_load(fake_load)


@pytest.mark.parametrize("hint", ["single_file", "gguf"])
def test_valid_folder_with_other_format_hint_is_refused(zw, fake_load, hint):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match=f"loader_format={hint}"):
        _build(zw, fake_load.model_dir, loader_format=hint)
    _assert_no_load(fake_load)


@pytest.mark.parametrize("hint", ["unknown", None])
def test_valid_folder_with_absent_hint_is_admitted(zw, fake_load, hint):
    _build(zw, fake_load.model_dir, loader_format=hint)
    assert len(fake_load.pipe_cls.load_calls) == 1


@pytest.mark.parametrize("offload", ["model", "sequential"])
def test_offload_is_refused_before_load(zw, fake_load, offload):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError) as info:
        _build(zw, fake_load.model_dir, runtime_offload=offload)
    assert str(info.value) == f"Z-Image does not support offload={offload}. Admitted: none."
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_env_offload_is_refused_before_load(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_OFFLOAD", "model")
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="offload=model"):
        _build(zw, fake_load.model_dir)
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_runtime_quantize_is_refused_before_load(zw, fake_load):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError) as info:
        _build(zw, fake_load.model_dir, runtime_quantize="fp8")
    assert str(info.value) == "Z-Image does not support quantize=fp8. Admitted: none."
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_env_quantize_is_refused_before_load(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_QUANTIZE", "fp8")
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="quantize=fp8"):
        _build(zw, fake_load.model_dir)
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_unknown_text_encoder_quantize_is_refused_before_load(zw, fake_load):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError) as info:
        _build(zw, fake_load.model_dir, text_encoder_quantize="fp8")
    assert str(info.value) == (
        "Z-Image does not support text_encoder_quantize=fp8. Admitted: int8, none."
    )
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


# --- Load ------------------------------------------------------------------


def test_dtype_is_bf16_even_when_env_asks_fp16(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_DTYPE", "fp16")
    worker = _build(zw, fake_load.model_dir)
    (_, kwargs), = fake_load.pipe_cls.load_calls
    assert kwargs["torch_dtype"] is zw.torch.bfloat16
    assert worker.dtype is zw.torch.bfloat16
    assert worker.dtype_str == "bf16"
    assert fake_load.store.call_args.kwargs["dtype"] is zw.torch.bfloat16


def test_dtype_override_logs_a_warning(zw, fake_load, monkeypatch, caplog):
    monkeypatch.setenv("CUDA_DTYPE", "fp16")
    with caplog.at_level("WARNING", logger=zw.logger.name):
        _build(zw, fake_load.model_dir)
    assert any("CUDA_DTYPE" in r.getMessage() for r in caplog.records)


def test_default_uses_int8_store(zw, fake_load):
    worker = _build(zw, fake_load.model_dir)
    fake_load.store.assert_called_once()
    args, kwargs = fake_load.store.call_args
    assert args == (fake_load.model_dir,)
    assert kwargs["device"] == worker.device
    (_, load_kwargs), = fake_load.pipe_cls.load_calls
    assert load_kwargs["text_encoder"] is fake_load.encoder


def test_explicit_int8_uses_store(zw, fake_load):
    _build(zw, fake_load.model_dir, text_encoder_quantize="int8")
    fake_load.store.assert_called_once()


def test_none_loads_native_encoder(zw, fake_load):
    _build(zw, fake_load.model_dir, text_encoder_quantize="none")
    fake_load.store.assert_not_called()
    (_, kwargs), = fake_load.pipe_cls.load_calls
    assert "text_encoder" not in kwargs


def test_from_pretrained_is_offline(zw, fake_load):
    _build(zw, fake_load.model_dir)
    (path, kwargs), = fake_load.pipe_cls.load_calls
    assert path == fake_load.model_dir
    assert kwargs["local_files_only"] is True


def test_worker_state_after_load(zw, fake_load):
    worker = _build(zw, fake_load.model_dir)
    assert isinstance(worker.pipe, fake_load.pipe_cls)
    assert worker._style_api == "none"
    assert worker.family_profile.family_id == "zimage"
    assert worker.supports_attention_processor_swap is False


def test_attention_swaps_are_skipped(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_ATTENTION_SLICING", "1")
    monkeypatch.setenv("CUDA_ENABLE_XFORMERS", "1")
    worker = _build(zw, fake_load.model_dir)
    swap_calls = [c for c in worker.pipe.calls if isinstance(c, tuple)]
    assert swap_calls == []


def test_quantization_targets_raises(zw, fake_load):
    worker = _build(zw, fake_load.model_dir)
    with pytest.raises(zw.ZImageUnsupportedRuntimeError):
        worker._quantization_targets(worker.pipe)


# --- Execution (Task 6) ----------------------------------------------------

LOG_TAG_PREFIX = "[zimage-cuda] job="


def _png_bytes(size=(16, 16)):
    buf = io.BytesIO()
    PILImage.new("RGB", size, (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_img2img_cls():
    class _FakeImg2Img(_FakePipelineBase):
        from_pipe_calls: list = []

        @classmethod
        def from_pipe(cls, base, **kwargs):
            cls.from_pipe_calls.append((base, kwargs))
            inst = cls()
            inst.scheduler = base.scheduler  # diffusers shares components
            return inst

    _FakeImg2Img.from_pipe_calls = []
    return _FakeImg2Img


def _fake_torch():
    fake = MagicMock()
    fake.inference_mode = MagicMock
    fake.cuda.max_memory_allocated.return_value = 2**30
    fake.cuda.max_memory_reserved.return_value = 2 * 2**30
    return fake


def _req(**overrides):
    base = dict(
        prompt="a red panda",
        negative_prompt=None,
        size="1024x768",
        seed=42,
        num_inference_steps=9,
        guidance_scale=0.0,
        scheduler_id=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _job(req=None, init_image=None, bindings=None):
    return SimpleNamespace(
        job_id="job-1",
        req=req or _req(),
        init_image=init_image,
        controlnet_bindings=bindings if bindings is not None else [],
    )


def _img_job(**req_overrides):
    return _job(_req(denoise_strength=0.6, **req_overrides), init_image=_png_bytes())


@pytest.fixture
def env(zw, fake_load):
    from backends.conditioning.artifacts import DelegatedConditioning

    i2i = _fake_img2img_cls()
    with patch.object(zw, "_zimage_img2img_pipeline_cls", return_value=i2i):
        worker = _build(zw, fake_load.model_dir)
        worker.execution_policy = MagicMock()
        worker._conditioning_artifact_for_request = Mock(
            side_effect=lambda req: DelegatedConditioning(
                prompt=req.prompt, negative_prompt=req.negative_prompt
            )
        )
        yield SimpleNamespace(zw=zw, worker=worker, i2i=i2i, torch=_fake_torch())


def _run(env, job, **kwargs):
    with patch.object(env.zw, "torch", env.torch):
        return env.worker.run_job(job, **kwargs)


def _job_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(LOG_TAG_PREFIX)]


def test_txt2img_kwargs(env):
    png, seed = _run(env, _job())
    (call,) = env.worker.pipe.calls
    assert seed == 42
    assert png.startswith(b"\x89PNG")
    assert call == {
        "prompt": "a red panda",
        "width": 1024,
        "height": 768,
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "generator": env.torch.Generator.return_value,
    }


def test_turbo_mode_drops_negative_prompt(env):
    # Review Focus 4: an old client sends a negative prompt to a Turbo mode.
    env.worker.model_info.allow_custom_negative_prompt = False
    env.worker.model_info.negative_prompt_templates = {}
    _run(env, _job(_req(negative_prompt="blurry")))
    assert "negative_prompt" not in env.worker.pipe.calls[0]


def test_base_mode_sends_negative_prompt(env):
    env.worker.model_info.negative_prompt_templates = {"default": "x"}
    _run(env, _job(_req(negative_prompt="blurry, low quality, watermark")))
    assert env.worker.pipe.calls[0]["negative_prompt"] == "blurry, low quality, watermark"


def test_custom_negative_flag_sends_negative_prompt(env):
    env.worker.model_info.allow_custom_negative_prompt = True
    _run(env, _job(_req(negative_prompt="blurry")))
    assert env.worker.pipe.calls[0]["negative_prompt"] == "blurry"


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_negative_prompt_is_not_sent(env, empty):
    env.worker.model_info.allow_custom_negative_prompt = True
    _run(env, _job(_req(negative_prompt=empty)))
    assert "negative_prompt" not in env.worker.pipe.calls[0]


def test_img2img_uses_img2img_pipeline_without_recast(env):
    _run(env, _img_job())
    ((base, kwargs),) = env.i2i.from_pipe_calls
    assert base is env.worker.pipe
    assert kwargs == {"torch_dtype": None}
    (call,) = env.worker._img2img_pipe.calls
    assert "image" in call
    assert call["strength"] == 0.6
    assert env.worker.pipe.calls == []


def test_img2img_resizes_init_image_to_request_size(env):
    _run(env, _img_job(size="64x32"))
    assert env.worker._img2img_pipe.calls[0]["image"].size == (64, 32)
    assert env.worker._img2img_pipe.calls[0]["image"].mode == "RGB"


def test_img2img_pipeline_is_built_once(env):
    _run(env, _img_job())
    _run(env, _img_job())
    assert len(env.i2i.from_pipe_calls) == 1


def test_img2img_pipeline_is_kept_in_base_attribute(env):
    assert env.worker._img2img_pipe is None
    _run(env, _img_job())
    assert isinstance(env.worker._img2img_pipe, env.i2i)


def test_img2img_default_strength_is_075_when_absent(env):
    _run(env, _job(_req(), init_image=_png_bytes()))
    _run(env, _job(_req(denoise_strength=None), init_image=_png_bytes()))
    calls = env.worker._img2img_pipe.calls
    assert [c["strength"] for c in calls] == [0.75, 0.75]


def test_img2img_zero_strength_is_refused_not_defaulted(env):
    with pytest.raises(ValueError, match="0 < denoise_strength"):
        _run(env, _job(_req(denoise_strength=0.0), init_image=_png_bytes()))
    assert env.i2i.from_pipe_calls == []


@pytest.mark.parametrize("strength", [-0.1, 1.5])
def test_img2img_out_of_range_strength_is_refused(env, strength):
    with pytest.raises(ValueError, match="0 < denoise_strength <= 1"):
        _run(env, _job(_req(denoise_strength=strength), init_image=_png_bytes()))
    assert env.i2i.from_pipe_calls == []


def test_img2img_strength_one_is_admitted(env):
    _run(env, _job(_req(denoise_strength=1.0), init_image=_png_bytes()))
    assert env.worker._img2img_pipe.calls[0]["strength"] == 1.0


def test_img2img_zero_effective_steps_is_refused(env):
    with pytest.raises(ValueError) as info:
        _run(env, _job(_req(denoise_strength=0.1), init_image=_png_bytes()))
    message = str(info.value)
    assert "9" in message and "0.1" in message and "0" in message
    assert env.i2i.from_pipe_calls == []


def test_img2img_smallest_valid_strength_runs(env):
    _run(env, _job(_req(denoise_strength=0.12), init_image=_png_bytes()))
    assert env.worker._img2img_pipe.calls[0]["strength"] == 0.12


def test_img2img_logs_effective_step_count(env, caplog):
    with caplog.at_level("INFO", logger=env.zw.logger.name):
        _run(env, _img_job())
    # int(9 * 0.6) == 5
    assert any("effective_steps=5" in r.getMessage() for r in caplog.records)


def test_img2img_progress_total_is_effective_steps(env):
    progress = Mock()
    _run(env, _img_job(), progress=progress)
    progress.assert_called_once_with(1, 5, "denoise")


def test_txt2img_progress_total_is_requested_steps(env):
    progress = Mock()
    _run(env, _job(), progress=progress)
    progress.assert_called_once_with(1, 9, "denoise")


def test_scheduler_stays_in_step_over_mixed_jobs(env):
    worker = env.worker
    with patch(
        "backends.cuda_worker.build_scheduler",
        side_effect=lambda sid, cfg: SimpleNamespace(id=sid, token=object()),
    ):
        for init in (None, _png_bytes(), None, _png_bytes()):
            req = _req(scheduler_id="flow_match_euler", denoise_strength=0.6)
            _run(env, _job(req, init_image=init))
            assert worker.pipe.scheduler.id == "flow_match_euler"
            if worker._img2img_pipe is not None:
                assert worker._img2img_pipe.scheduler is worker.pipe.scheduler


def test_cancel_then_next_job_runs(env):
    with pytest.raises(concurrent.futures.CancelledError):
        _run(env, _job(), should_cancel=lambda: True)
    env.worker.execution_policy.release_cache.assert_called_once()
    png, _ = _run(env, _job())
    assert png.startswith(b"\x89PNG")


def test_failed_img2img_then_txt2img_runs(env):
    _run(env, _img_job())
    i2i_pipe = env.worker._img2img_pipe
    i2i_pipe.__class__.__call__ = Mock(side_effect=RuntimeError("img2img broke"))
    with pytest.raises(RuntimeError, match="img2img broke"):
        _run(env, _img_job())
    png, _ = _run(env, _job())
    assert png.startswith(b"\x89PNG")
    assert env.worker._img2img_pipe is i2i_pipe


def test_job_logs_peak_allocated(env, caplog):
    env.worker._read_peak_memory = Mock(return_value=(19 * 2**30, 20 * 2**30))
    with caplog.at_level("INFO", logger=env.zw.logger.name):
        _run(env, _job())
    (line,) = _job_lines(caplog)
    assert "job=job-1" in line
    assert "op=txt2img" in line
    assert "size=1024x768" in line
    assert "steps=9" in line
    assert "peak_allocated_gib=19.00" in line
    assert "peak_reserved_gib=20.00" in line


def test_img2img_job_line_names_op_and_effective_steps(env, caplog):
    with caplog.at_level("INFO", logger=env.zw.logger.name):
        _run(env, _img_job())
    (line,) = _job_lines(caplog)
    assert "op=img2img" in line
    assert "steps=5" in line


def test_peak_is_reset_before_and_read_after_the_pipe_call(env):
    order = []
    env.worker._reset_peak_memory = Mock(side_effect=lambda: order.append("reset"))
    env.worker._read_peak_memory = Mock(side_effect=lambda: order.append("read") or (1, 2))
    env.worker.execution_policy.release_cache.side_effect = lambda: order.append("release")
    pipe_cls = type(env.worker.pipe)
    original = pipe_cls.__call__

    def _recording_call(self, **kwargs):
        order.append("pipe")
        return original(self, **kwargs)

    with patch.object(pipe_cls, "__call__", _recording_call):
        _run(env, _job())
    assert order == ["reset", "pipe", "read", "release"]


def test_peak_is_logged_when_the_pipe_raises(env, caplog):
    env.worker._read_peak_memory = Mock(return_value=(3 * 2**30, 4 * 2**30))
    with patch.object(type(env.worker.pipe), "__call__", side_effect=ValueError("boom")):
        with caplog.at_level("INFO", logger=env.zw.logger.name):
            with pytest.raises(ValueError, match="boom"):
                _run(env, _job())
    (line,) = _job_lines(caplog)
    assert "peak_allocated_gib=3.00" in line


def test_job_log_has_exact_bytes(env, caplog):
    env.worker._read_peak_memory = Mock(return_value=(19 * 2**30 + 1, 20 * 2**30))
    with caplog.at_level("INFO", logger=env.zw.logger.name):
        _run(env, _job())
    (line,) = _job_lines(caplog)
    assert "peak_allocated_bytes=20401094657" in line
    assert "peak_allocated_gib=19.00" in line


def test_memory_read_failure_does_not_fail_the_job(env, caplog):
    env.worker._read_peak_memory = Mock(side_effect=RuntimeError("no cuda"))
    with caplog.at_level("INFO", logger=env.zw.logger.name):
        png, seed = _run(env, _job())
    assert png.startswith(b"\x89PNG") and seed == 42
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    (line,) = _job_lines(caplog)
    assert "peak_allocated_bytes=unavailable" in line
    env.worker.execution_policy.release_cache.assert_called_once()


def test_memory_reset_failure_does_not_fail_the_job(env):
    env.worker._reset_peak_memory = Mock(side_effect=RuntimeError("no cuda"))
    png, _ = _run(env, _job())
    assert png.startswith(b"\x89PNG")
    assert len(env.worker.pipe.calls) == 1


def test_memory_read_failure_does_not_mask_a_generation_error(env):
    env.worker._read_peak_memory = Mock(side_effect=RuntimeError("no cuda"))
    with patch.object(type(env.worker.pipe), "__call__", side_effect=ValueError("boom")):
        with pytest.raises(ValueError, match="boom"):
            _run(env, _job())
    env.worker.execution_policy.release_cache.assert_called_once()


def test_peak_helpers_use_cuda_counters_for_worker_device(env):
    worker = env.worker
    with patch.object(env.zw, "torch", env.torch):
        worker._reset_peak_memory()
        assert worker._read_peak_memory() == (2**30, 2 * 2**30)
    cuda = env.torch.cuda
    cuda.reset_peak_memory_stats.assert_called_once_with(worker.device)
    cuda.max_memory_allocated.assert_called_once_with(worker.device)
    cuda.max_memory_reserved.assert_called_once_with(worker.device)
    assert cuda.synchronize.call_count == 2


def test_invalid_init_image_bytes_raise(env):
    # Review Focus 3
    job = _job(_req(denoise_strength=0.6), init_image=b"not an image")
    with pytest.raises(RuntimeError, match="invalid init image bytes"):
        _run(env, job)
    assert env.i2i.from_pipe_calls == []


def test_controlnet_binding_is_refused(env):
    # Review Focus 5
    with pytest.raises(RuntimeError, match="does not support ControlNet"):
        _run(env, _job(bindings=[object()]))
    assert env.worker.pipe.calls == []


def test_controlnet_request_field_is_refused(env):
    with pytest.raises(RuntimeError, match="does not support ControlNet"):
        _run(env, _job(_req(controlnets=[SimpleNamespace(attachment_id="cn_1")])))
    assert env.worker.pipe.calls == []


def test_invalid_size_raises(env):
    with pytest.raises(RuntimeError, match="Invalid size"):
        _run(env, _job(_req(size="big")))


def test_materialized_conditioning_is_refused(env):
    from backends.conditioning.artifacts import MaterializedConditioning

    env.worker._conditioning_artifact_for_request = Mock(
        return_value=MagicMock(spec=MaterializedConditioning)
    )
    with pytest.raises(ValueError, match="native conditioning only"):
        _run(env, _job())
    assert env.worker.pipe.calls == []


def test_png_metadata_records_request(env):
    import json

    png, _ = _run(env, _job(_req(scheduler_id=None)))
    meta = json.loads(PILImage.open(io.BytesIO(png)).text["lcm"])
    assert meta["prompt"] == "a red panda"
    assert meta["seed"] == 42
    assert meta["size"] == "1024x768"
    assert meta["steps"] == 9
    assert meta["cfg"] == 0.0
    assert "scheduler_id" in meta
    assert "denoise_strength" not in meta


def test_img2img_png_metadata_records_strength(env):
    import json

    png, _ = _run(env, _img_job())
    meta = json.loads(PILImage.open(io.BytesIO(png)).text["lcm"])
    assert meta["denoise_strength"] == 0.6


def test_release_cache_runs_after_failure(env):
    with patch.object(type(env.worker.pipe), "__call__", side_effect=ValueError("boom")):
        with pytest.raises(ValueError):
            _run(env, _job())
    env.worker.execution_policy.release_cache.assert_called_once()
