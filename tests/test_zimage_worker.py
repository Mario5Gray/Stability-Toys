"""Unit tests for the Z-Image CUDA worker.

These stub the heavy diffusers and torch stack. No test loads real weights.
Each test patches the lazy resolvers and ``load_int8_text_encoder`` in
``backends.zimage_cuda_worker``, where the call happens.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[MagicMock()])


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
def fake_load(zw):
    """Patch the pipeline resolvers and the encoder store. Yield the handles."""
    pipe_cls = _fake_pipeline_cls()
    encoder = object()
    with patch.object(zw, "_zimage_pipeline_cls", return_value=pipe_cls), \
         patch.object(zw, "_zimage_img2img_pipeline_cls", return_value=MagicMock()), \
         patch.object(zw, "load_int8_text_encoder", return_value=encoder) as store:
        yield SimpleNamespace(pipe_cls=pipe_cls, encoder=encoder, store=store)


def _build(zw, model_path="/models/repos/Z-Image-Turbo-BF16", **info):
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


def test_preflight_failure_reports_versions(zw):
    with patch.object(zw, "_zimage_pipeline_cls", side_effect=ImportError("no ZImagePipeline")), \
         patch.object(zw, "load_int8_text_encoder") as store:
        with pytest.raises(zw.ZImageDependencyError) as info:
            _build(zw)
    message = str(info.value)
    assert "diffusers" in message
    assert "transformers" in message
    assert "no ZImagePipeline" in message
    store.assert_not_called()


def test_preflight_checks_img2img_class(zw):
    with patch.object(zw, "_zimage_pipeline_cls", return_value=_fake_pipeline_cls()), \
         patch.object(zw, "_zimage_img2img_pipeline_cls", side_effect=ImportError("no img2img")), \
         patch.object(zw, "load_int8_text_encoder"):
        with pytest.raises(zw.ZImageDependencyError, match="no img2img"):
            _build(zw)


# --- Admission -------------------------------------------------------------


def test_non_folder_path_is_refused(zw, fake_load):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="Diffusers"):
        _build(zw, model_path="/x/model.safetensors", loader_format="single_file")
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


@pytest.mark.parametrize("offload", ["model", "sequential"])
def test_offload_is_refused_before_load(zw, fake_load, offload):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError) as info:
        _build(zw, runtime_offload=offload)
    assert str(info.value) == f"Z-Image does not support offload={offload}. Admitted: none."
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_env_offload_is_refused_before_load(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_OFFLOAD", "model")
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="offload=model"):
        _build(zw)
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_runtime_quantize_is_refused_before_load(zw, fake_load):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError) as info:
        _build(zw, runtime_quantize="fp8")
    assert str(info.value) == "Z-Image does not support quantize=fp8. Admitted: none."
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_env_quantize_is_refused_before_load(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_QUANTIZE", "fp8")
    with pytest.raises(zw.ZImageUnsupportedRuntimeError, match="quantize=fp8"):
        _build(zw)
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


def test_unknown_text_encoder_quantize_is_refused_before_load(zw, fake_load):
    with pytest.raises(zw.ZImageUnsupportedRuntimeError) as info:
        _build(zw, text_encoder_quantize="fp8")
    assert str(info.value) == (
        "Z-Image does not support text_encoder_quantize=fp8. Admitted: int8, none."
    )
    assert fake_load.pipe_cls.load_calls == []
    fake_load.store.assert_not_called()


# --- Load ------------------------------------------------------------------


def test_dtype_is_bf16_even_when_env_asks_fp16(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_DTYPE", "fp16")
    worker = _build(zw)
    (_, kwargs), = fake_load.pipe_cls.load_calls
    assert kwargs["torch_dtype"] is zw.torch.bfloat16
    assert worker.dtype is zw.torch.bfloat16
    assert worker.dtype_str == "bf16"
    assert fake_load.store.call_args.kwargs["dtype"] is zw.torch.bfloat16


def test_dtype_override_logs_a_warning(zw, fake_load, monkeypatch, caplog):
    monkeypatch.setenv("CUDA_DTYPE", "fp16")
    with caplog.at_level("WARNING", logger=zw.logger.name):
        _build(zw)
    assert any("CUDA_DTYPE" in r.getMessage() for r in caplog.records)


def test_default_uses_int8_store(zw, fake_load):
    worker = _build(zw)
    fake_load.store.assert_called_once()
    args, kwargs = fake_load.store.call_args
    assert args == ("/models/repos/Z-Image-Turbo-BF16",)
    assert kwargs["device"] == worker.device
    (_, load_kwargs), = fake_load.pipe_cls.load_calls
    assert load_kwargs["text_encoder"] is fake_load.encoder


def test_explicit_int8_uses_store(zw, fake_load):
    _build(zw, text_encoder_quantize="int8")
    fake_load.store.assert_called_once()


def test_none_loads_native_encoder(zw, fake_load):
    _build(zw, text_encoder_quantize="none")
    fake_load.store.assert_not_called()
    (_, kwargs), = fake_load.pipe_cls.load_calls
    assert "text_encoder" not in kwargs


def test_from_pretrained_is_offline(zw, fake_load):
    _build(zw)
    (path, kwargs), = fake_load.pipe_cls.load_calls
    assert path == "/models/repos/Z-Image-Turbo-BF16"
    assert kwargs["local_files_only"] is True


def test_worker_state_after_load(zw, fake_load):
    worker = _build(zw)
    assert isinstance(worker.pipe, fake_load.pipe_cls)
    assert worker._style_api == "none"
    assert worker.family_profile.family_id == "zimage"
    assert worker.supports_attention_processor_swap is False


def test_attention_swaps_are_skipped(zw, fake_load, monkeypatch):
    monkeypatch.setenv("CUDA_ATTENTION_SLICING", "1")
    monkeypatch.setenv("CUDA_ENABLE_XFORMERS", "1")
    worker = _build(zw)
    swap_calls = [c for c in worker.pipe.calls if isinstance(c, tuple)]
    assert swap_calls == []


def test_quantization_targets_raises(zw, fake_load):
    worker = _build(zw)
    with pytest.raises(zw.ZImageUnsupportedRuntimeError):
        worker._quantization_targets(worker.pipe)
