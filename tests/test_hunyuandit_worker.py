"""Unit tests for the lazy native-conditioning HunyuanDiT CUDA worker.

These stub the heavy diffusers/torch stack and assert only the family-owned
construction and request-shaping contract:

  - selecting an SD family never imports the Hunyuan pipeline/tokenizer stack
  - Hunyuan construction runs the dependency preflight before any download
  - the preflight validates all three Diffusers classes and a callable
    ``T5Tokenizer.from_pretrained``, and reports installed versions on failure
  - the worker defaults to the canonical Hunyuan profile before conditioning setup
  - quantization targets ``pipe.transformer`` only (mT5 excluded)
  - the ControlNet class is ``HunyuanDiT2DControlNetModel``
  - a Diffusers directory is required; the worker does not inherit SDXL
  - base + ControlNet load fp16, composition uses
    ``HunyuanDiTControlNetPipeline.from_pipe`` with no post-composition recast
  - zero/one Canny ControlNet txt2img uses ``control_image`` +
    ``use_resolution_binning=True``
  - every init image fails explicitly in the worker
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

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


class _FakePipelineBase:
    def __init__(self):
        self.vae = MagicMock()
        self.transformer = MagicMock()
        self.unet = MagicMock()
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    # Memory-optimization hooks _setup_pipe_memory_opts may call. Whether it
    # does depends on CUDA_ATTENTION_SLICING / CUDA_ENABLE_XFORMERS, which are
    # unset locally but both 1 in env.cuda — so a fake lacking these passes on a
    # laptop and fails in the CUDA container. Record calls instead of omitting
    # the methods; the SD/Hunyuan swap assertions patch over these anyway.
    def enable_attention_slicing(self, *args, **kwargs):
        self.calls.append(("enable_attention_slicing", args))

    def enable_xformers_memory_efficient_attention(self, *args, **kwargs):
        self.calls.append(("enable_xformers_memory_efficient_attention", args))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[MagicMock()])


# Wire the SD names cuda_worker imports at module load (needed to import it).
sys.modules["diffusers.schedulers.scheduling_lcm"].LCMScheduler = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion"].StableDiffusionPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl"].StableDiffusionXLPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img"].StableDiffusionImg2ImgPipeline = MagicMock
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img"].StableDiffusionXLImg2ImgPipeline = MagicMock
sys.modules["backends.styles"].STYLE_REGISTRY = {}


def _diffusers_dir_model_info():
    return SimpleNamespace(loader_format="diffusers_dir", scheduler_profile="native")


def _make_req(size="1024x1024"):
    return SimpleNamespace(
        prompt="a red panda",
        negative_prompt="blurry",
        size=size,
        num_inference_steps=25,
        guidance_scale=5.0,
        seed=7,
        style_lora=None,
    )


# --------------------------------------------------------------------------
# Lazy import: SD selection must not pull in the Hunyuan stack
# --------------------------------------------------------------------------


def test_sd_worker_construction_does_not_run_hunyuan_preflight():
    import backends.cuda_worker as cw

    class _FakeSD(_FakePipelineBase):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    with patch.object(cw, "_sd_pipeline_cls", return_value=_FakeSD), \
         patch.object(cw, "STYLE_REGISTRY", {}), \
         patch.object(cw, "_hunyuandit_dependency_preflight") as preflight, \
         patch.object(cw, "_hunyuandit_pipeline_cls") as h_pipe, \
         patch.object(cw, "_hunyuandit_controlnet_pipeline_cls") as h_cn_pipe, \
         patch.object(cw, "_hunyuandit_controlnet_model_cls") as h_cn_model, \
         patch.object(cw, "_t5_tokenizer_cls") as h_tok:
        cw.DiffusersCudaWorker(
            worker_id=0,
            model_path="/models/sd15",
            model_info=_diffusers_dir_model_info(),
        )

    preflight.assert_not_called()
    for getter in (h_pipe, h_cn_pipe, h_cn_model, h_tok):
        getter.assert_not_called()


# --------------------------------------------------------------------------
# Dependency preflight
# --------------------------------------------------------------------------


def test_preflight_passes_when_all_classes_and_tokenizer_are_present():
    import backends.cuda_worker as cw

    tok = SimpleNamespace(from_pretrained=lambda *a, **k: object())
    with patch.object(cw, "_hunyuandit_pipeline_cls", return_value=object()), \
         patch.object(cw, "_hunyuandit_controlnet_pipeline_cls", return_value=object()), \
         patch.object(cw, "_hunyuandit_controlnet_model_cls", return_value=object()), \
         patch.object(cw, "_t5_tokenizer_cls", return_value=tok):
        cw._hunyuandit_dependency_preflight()  # must not raise


def test_preflight_raises_dependency_error_when_a_pipeline_class_is_missing():
    import backends.cuda_worker as cw

    def _boom():
        raise ImportError("cannot import name 'HunyuanDiT2DControlNetModel'")

    with patch.object(cw, "_hunyuandit_pipeline_cls", return_value=object()), \
         patch.object(cw, "_hunyuandit_controlnet_pipeline_cls", return_value=object()), \
         patch.object(cw, "_hunyuandit_controlnet_model_cls", side_effect=_boom), \
         patch.object(cw, "_t5_tokenizer_cls", return_value=SimpleNamespace(from_pretrained=lambda *a: None)):
        with pytest.raises(cw.HunyuanDiTDependencyError) as exc:
            cw._hunyuandit_dependency_preflight()

    # Reports installed versions of the whole family stack.
    msg = str(exc.value)
    assert "diffusers=" in msg
    assert "transformers=" in msg
    assert "sentencepiece=" in msg


def test_preflight_rejects_placeholder_tokenizer_loader():
    import backends.cuda_worker as cw

    # SentencePiece-less transformers installs a placeholder whose loader is not
    # callable; the preflight must reject it before any model download.
    placeholder = SimpleNamespace(from_pretrained=None)
    with patch.object(cw, "_hunyuandit_pipeline_cls", return_value=object()), \
         patch.object(cw, "_hunyuandit_controlnet_pipeline_cls", return_value=object()), \
         patch.object(cw, "_hunyuandit_controlnet_model_cls", return_value=object()), \
         patch.object(cw, "_t5_tokenizer_cls", return_value=placeholder):
        with pytest.raises(cw.HunyuanDiTDependencyError) as exc:
            cw._hunyuandit_dependency_preflight()

    assert "sentencepiece" in str(exc.value).lower()


def test_construction_runs_preflight_before_base_model_download():
    import backends.cuda_worker as cw

    order: list[str] = []

    class _FakeHunyuanPipeline(_FakePipelineBase):
        @classmethod
        def from_pretrained(cls, path, *, torch_dtype=_MISSING):
            order.append(f"base_load:{torch_dtype}")
            return cls()

    def _record_preflight():
        order.append("preflight")

    with patch.object(cw, "_hunyuandit_dependency_preflight", side_effect=_record_preflight), \
         patch.object(cw, "_hunyuandit_pipeline_cls", return_value=_FakeHunyuanPipeline):
        cw.DiffusersHunyuanDiTCudaWorker(
            worker_id=0,
            model_path="/models/hunyuan",
            model_info=_diffusers_dir_model_info(),
        )

    assert order[0] == "preflight"
    assert any(step.startswith("base_load") for step in order)
    assert order.index("preflight") < next(
        i for i, step in enumerate(order) if step.startswith("base_load")
    )


# --------------------------------------------------------------------------
# Construction: profile, dtype, loader-format, scheduler
# --------------------------------------------------------------------------


def _build_worker(order=None, model_info=None):
    """Construct a Hunyuan worker with a fake base pipeline (fp16 recorded)."""
    import backends.cuda_worker as cw

    captured = {}

    class _FakeHunyuanPipeline(_FakePipelineBase):
        @classmethod
        def from_pretrained(cls, path, *, torch_dtype=_MISSING):
            captured["base_path"] = path
            captured["base_dtype"] = torch_dtype
            inst = cls()
            captured["pipe"] = inst
            return inst

    with patch.object(cw, "_hunyuandit_dependency_preflight"), \
         patch.object(cw, "_hunyuandit_pipeline_cls", return_value=_FakeHunyuanPipeline):
        worker = cw.DiffusersHunyuanDiTCudaWorker(
            worker_id=0,
            model_path="/models/hunyuan",
            model_info=model_info or _diffusers_dir_model_info(),
        )
    return worker, captured


def test_worker_defaults_to_canonical_hunyuandit_profile():
    from backends.family_profiles import HUNYUANDIT_PROFILE

    worker, _ = _build_worker()
    assert worker.family_profile is HUNYUANDIT_PROFILE
    # Native conditioning was set up from that profile (family + no pooled).
    descriptor = worker._conditioning_context.descriptor
    assert descriptor.model_family == "hunyuandit"
    assert descriptor.pooled_required is False


def test_base_pipeline_loads_in_worker_dtype():
    import backends.cuda_worker as cw

    worker, captured = _build_worker()
    assert captured["base_dtype"] == worker.dtype
    assert captured["base_path"] == "/models/hunyuan"


def test_single_file_checkpoint_is_rejected_diffusers_directory_only():
    import backends.cuda_worker as cw

    with patch.object(cw, "_hunyuandit_dependency_preflight"), \
         patch.object(cw, "_hunyuandit_pipeline_cls", return_value=MagicMock()):
        with pytest.raises(RuntimeError, match="[Dd]iffusers direct"):
            cw.DiffusersHunyuanDiTCudaWorker(
                worker_id=0,
                model_path="/models/hunyuan.safetensors",
                model_info=SimpleNamespace(loader_format="single_file"),
            )


def test_worker_does_not_inherit_the_sdxl_worker():
    import backends.cuda_worker as cw

    assert not issubclass(cw.DiffusersHunyuanDiTCudaWorker, cw.DiffusersSDXLCudaWorker)
    assert issubclass(cw.DiffusersHunyuanDiTCudaWorker, cw.CudaWorkerBase)


# --------------------------------------------------------------------------
# Family hooks: quantization targets and ControlNet class
# --------------------------------------------------------------------------


def test_quantization_targets_are_transformer_only_mt5_excluded():
    import backends.cuda_worker as cw

    worker = cw.DiffusersHunyuanDiTCudaWorker.__new__(cw.DiffusersHunyuanDiTCudaWorker)
    transformer = object()
    text_encoder_2 = object()
    pipe = SimpleNamespace(transformer=transformer, text_encoder_2=text_encoder_2, unet=object())

    targets = worker._quantization_targets(pipe)
    assert targets == (transformer,)
    assert text_encoder_2 not in targets


def test_controlnet_model_class_is_hunyuandit_2d_controlnet_model():
    import backends.cuda_worker as cw

    sentinel = object()
    sys.modules["diffusers"].HunyuanDiT2DControlNetModel = sentinel
    worker = cw.DiffusersHunyuanDiTCudaWorker.__new__(cw.DiffusersHunyuanDiTCudaWorker)
    assert worker._controlnet_model_cls() is sentinel


# --------------------------------------------------------------------------
# Composition: from_pipe with no post-composition dtype recast
# --------------------------------------------------------------------------


def test_controlnet_composition_uses_from_pipe_without_dtype_recast():
    import backends.cuda_worker as cw

    recorded = {}

    class _FakeHunyuanControlNetPipeline(_FakePipelineBase):
        @classmethod
        def from_pipe(cls, pipe, controlnet, *, torch_dtype=_MISSING):
            recorded["base"] = pipe
            recorded["controlnet"] = controlnet
            recorded["torch_dtype"] = torch_dtype
            return cls()

    sys.modules["diffusers"].HunyuanDiTControlNetPipeline = _FakeHunyuanControlNetPipeline
    worker = cw.DiffusersHunyuanDiTCudaWorker.__new__(cw.DiffusersHunyuanDiTCudaWorker)
    base = _FakePipelineBase()
    worker.pipe = base
    cn_obj = object()

    composed = worker._build_controlnet_pipe(cn_obj)

    assert isinstance(composed, _FakeHunyuanControlNetPipeline)
    assert recorded["base"] is base
    assert recorded["controlnet"] is cn_obj
    # No recast: a dtype must never be forced onto the composed pipeline.
    assert recorded["torch_dtype"] in (_MISSING, None)


# --------------------------------------------------------------------------
# ControlNet kwargs: only what HunyuanDiTControlNetPipeline.__call__ accepts
# --------------------------------------------------------------------------


def test_controlnet_kwargs_omit_unsupported_guidance_window():
    """HunyuanDiTControlNetPipeline.__call__ accepts control_image and
    controlnet_conditioning_scale but has NO control_guidance_start/end (no
    per-step guidance window). The worker's control kwargs must not emit those
    keys or the live pipe(**kwargs) raises TypeError. Regression: the T10 CUDA
    acceptance failed with 'unexpected keyword argument control_guidance_start'.

    Exercises the REAL _build_controlnet_kwargs (earlier run_job tests mocked it,
    which is exactly what hid this incompatibility)."""
    import backends.cuda_worker as cw

    worker = cw.DiffusersHunyuanDiTCudaWorker.__new__(cw.DiffusersHunyuanDiTCudaWorker)
    worker.family_profile = __import__(
        "backends.family_profiles", fromlist=["HUNYUANDIT_PROFILE"]
    ).HUNYUANDIT_PROFILE
    worker._load_controlnet_model = Mock(return_value=object())

    binding = SimpleNamespace(
        model_id="hunyuandit-canny",
        control_image_bytes=b"\x89PNG-canny",
        strength=0.8,
        start_percent=0.0,
        end_percent=1.0,
    )
    with patch("backends.cuda_worker._decode_control_image", return_value=MagicMock()):
        kwargs = worker._build_controlnet_kwargs([binding], (1024, 1024), [])

    # Accepted by the HunyuanDiTControlNetPipeline signature.
    assert "controlnet" in kwargs  # popped in run_job before the call
    assert "control_image" in kwargs
    assert "controlnet_conditioning_scale" in kwargs
    # Not accepted — presence would TypeError at pipe(**kwargs).
    assert "control_guidance_start" not in kwargs
    assert "control_guidance_end" not in kwargs


# --------------------------------------------------------------------------
# run_job: control_image kwarg, resolution binning, init-image rejection
# --------------------------------------------------------------------------


def _run_new_worker():
    import backends.cuda_worker as cw

    worker = cw.DiffusersHunyuanDiTCudaWorker.__new__(cw.DiffusersHunyuanDiTCudaWorker)
    worker.device = "cuda:0"
    worker.dtype = "fp16_sentinel"
    worker.worker_id = 0
    worker.family_profile = __import__(
        "backends.family_profiles", fromlist=["HUNYUANDIT_PROFILE"]
    ).HUNYUANDIT_PROFILE
    worker._apply_request_scheduler = Mock(return_value=None)
    worker._conditioning_artifact_for_request = Mock(return_value=object())
    worker._accept_conditioning_artifact = Mock(return_value={"prompt": "a red panda"})
    return worker


def test_txt2img_without_controlnet_passes_resolution_binning():
    import torch  # stub

    worker = _run_new_worker()
    pipe = _FakePipelineBase()
    worker.pipe = pipe
    job = SimpleNamespace(req=_make_req(), init_image=None, controlnet_bindings=[])

    with patch("backends.cuda_worker.torch", MagicMock(inference_mode=MagicMock)):
        worker.run_job(job)

    assert pipe.calls, "expected the base pipeline to be invoked"
    call = pipe.calls[0]
    assert call["use_resolution_binning"] is True
    assert call["width"] == 1024 and call["height"] == 1024
    assert "control_image" not in call


def test_single_controlnet_txt2img_uses_control_image_kwarg():
    import backends.cuda_worker as cw

    worker = _run_new_worker()
    base = _FakePipelineBase()
    worker.pipe = base

    cn_obj = object()
    control_map = MagicMock(name="control_map")
    # Family-driven control-map kwarg is "control_image" for HunyuanDiT.
    worker._build_controlnet_kwargs = Mock(return_value={
        "controlnet": cn_obj,
        "control_image": control_map,
        "controlnet_conditioning_scale": 0.8,
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
    })
    cn_pipe = _FakePipelineBase()
    worker._build_controlnet_pipe = Mock(return_value=cn_pipe)
    # Provenance stamping is exercised elsewhere; keep it out of this kwarg test.
    worker._controlnet_metadata = Mock(return_value=[])

    binding = SimpleNamespace(model_id="hunyuan-canny", attachment_id="cn_1",
                              control_type="canny")
    job = SimpleNamespace(req=_make_req(), init_image=None,
                          controlnet_bindings=[binding])

    with patch("backends.cuda_worker.torch", MagicMock(inference_mode=MagicMock)):
        worker.run_job(job)

    worker._build_controlnet_pipe.assert_called_once_with(cn_obj)
    assert cn_pipe.calls, "expected the composed ControlNet pipeline to run"
    call = cn_pipe.calls[0]
    assert call["control_image"] is control_map
    assert call["use_resolution_binning"] is True
    assert "controlnet" not in call  # popped before the pipeline call


def test_worker_rejects_init_image_explicitly():
    worker = _run_new_worker()
    worker.pipe = _FakePipelineBase()
    job = SimpleNamespace(req=_make_req(), init_image=b"\x89PNG init",
                          controlnet_bindings=[])

    with pytest.raises(RuntimeError, match="init image|img2img"):
        worker.run_job(job)


def _memory_opt_probe_pipe():
    """Pipe stub recording which attention-processor swaps were requested."""
    pipe = MagicMock()
    pipe.to.return_value = pipe
    return pipe


def _worker_for_memory_opts(cls, *, enable_xformers=True, attention_slicing=True):
    worker = cls.__new__(cls)
    worker.worker_id = 0
    worker._enable_xformers = enable_xformers
    worker._attention_slicing = attention_slicing
    worker._quantize = "none"
    worker._checkpoint_precision = "fp16"
    worker._offload = "none"
    worker.device = "cuda:0"
    return worker


def test_hunyuandit_worker_declares_attention_processor_swap_unsafe():
    import backends.cuda_worker as cw

    # HunyuanDiT2DModel routes rotary positional embeddings through
    # cross_attention_kwargs["image_rotary_emb"]. Swapped-in processors
    # (XFormersAttnProcessor / SlicedAttnProcessor) silently drop that kwarg.
    assert cw.CudaWorkerBase.supports_attention_processor_swap is True
    assert cw.DiffusersHunyuanDiTCudaWorker.supports_attention_processor_swap is False


def test_hunyuandit_memory_opts_skip_processor_swaps():
    import backends.cuda_worker as cw

    worker = _worker_for_memory_opts(cw.DiffusersHunyuanDiTCudaWorker)
    pipe = _memory_opt_probe_pipe()

    worker._setup_pipe_memory_opts(pipe)

    pipe.enable_xformers_memory_efficient_attention.assert_not_called()
    pipe.enable_attention_slicing.assert_not_called()
    # VAE-level memory opts do not touch attention processors and stay on.
    pipe.vae.enable_tiling.assert_called_once()
    pipe.vae.enable_slicing.assert_called_once()


def test_sd_memory_opts_still_apply_processor_swaps():
    import backends.cuda_worker as cw

    worker = _worker_for_memory_opts(cw.DiffusersCudaWorker)
    pipe = _memory_opt_probe_pipe()

    worker._setup_pipe_memory_opts(pipe)

    pipe.enable_xformers_memory_efficient_attention.assert_called_once()
    pipe.enable_attention_slicing.assert_called_once_with(1)


def _fake_pipe_for_state():
    """Pipe stand-in exposing the attributes the state snapshot reads."""
    pipe = MagicMock()
    class DDPMScheduler:
        config = {"beta_schedule": "scaled_linear", "num_train_timesteps": 1000}

    pipe.scheduler = DDPMScheduler()
    pipe.vae.use_tiling = True
    pipe.vae.use_slicing = True
    param = SimpleNamespace(dtype="torch.float16", device="cuda:0")
    # Fresh iterator per call: each captured field calls parameters() again.
    pipe.transformer.parameters.side_effect = lambda: iter([param])
    pipe.vae.parameters.side_effect = lambda: iter([param])
    pipe.transformer.attn_processors = {"blk.0": object()}
    return pipe


def test_debug_dump_disabled_by_default(monkeypatch, tmp_path):
    import backends.cuda_worker as cw

    monkeypatch.delenv("HUNYUAN_DEBUG_DUMP", raising=False)
    assert cw._hunyuan_debug_enabled() is False
    # Disabled must not create the directory at all.
    monkeypatch.setenv("HUNYUAN_DEBUG_ROOT", str(tmp_path / "dumps"))
    assert cw._hunyuan_debug_dir("job123") is None
    assert not (tmp_path / "dumps").exists()


def test_debug_dump_enabled_creates_job_directory(monkeypatch, tmp_path):
    import backends.cuda_worker as cw

    monkeypatch.setenv("HUNYUAN_DEBUG_DUMP", "1")
    monkeypatch.setenv("HUNYUAN_DEBUG_ROOT", str(tmp_path / "dumps"))
    target = cw._hunyuan_debug_dir("job123")
    assert target is not None
    assert target.is_dir()
    assert target.name == "job123"


def test_pipe_state_snapshot_reports_required_fields():
    import backends.cuda_worker as cw

    state = cw._hunyuan_pipe_state(_fake_pipe_for_state(), controlnet=None)

    assert state["scheduler_class"] == "DDPMScheduler"
    assert state["scheduler_config"]["beta_schedule"] == "scaled_linear"
    assert state["vae_tiling"] is True
    assert state["vae_slicing"] is True
    assert state["transformer_dtype"] == "torch.float16"
    assert state["transformer_device"] == "cuda:0"
    assert state["attention_processor_classes"] == ["object"]


def test_pipe_state_snapshot_never_raises_on_broken_pipe():
    import backends.cuda_worker as cw

    # Diagnostics must never be able to fail a job.
    broken = SimpleNamespace()
    state = cw._hunyuan_pipe_state(broken, controlnet=None)
    assert isinstance(state, dict)


def test_debug_dump_json_survives_unserializable_values(monkeypatch, tmp_path):
    import backends.cuda_worker as cw

    monkeypatch.setenv("HUNYUAN_DEBUG_DUMP", "1")
    monkeypatch.setenv("HUNYUAN_DEBUG_ROOT", str(tmp_path / "dumps"))
    target = cw._hunyuan_debug_dir("job456")
    cw._hunyuan_debug_write_json(target, "call_kwargs.json", {"generator": object(), "width": 1024})

    written = json.loads((target / "call_kwargs.json").read_text())
    assert written["width"] == 1024
    assert isinstance(written["generator"], str)


def test_run_job_writes_no_diagnostics_when_flag_unset(monkeypatch, tmp_path):
    """Disabled diagnostics must be fully inert on the production path.

    Not just "writes no files" — _hunyuan_pipe_state must never be invoked,
    since arguments evaluate eagerly and it walks the pipeline on every call.
    """
    import backends.cuda_worker as cw

    monkeypatch.delenv("HUNYUAN_DEBUG_DUMP", raising=False)
    root = tmp_path / "dumps"
    monkeypatch.setenv("HUNYUAN_DEBUG_ROOT", str(root))

    state_calls: list[int] = []
    monkeypatch.setattr(
        cw, "_hunyuan_pipe_state", lambda *a, **k: (state_calls.append(1), {})[1]
    )

    worker = _run_new_worker()
    pipe = _FakePipelineBase()
    worker.pipe = pipe
    job = SimpleNamespace(
        req=_make_req(), init_image=None, controlnet_bindings=[], job_id="jobX"
    )

    with patch("backends.cuda_worker.torch", MagicMock(inference_mode=MagicMock)):
        worker.run_job(job)

    assert pipe.calls, "the job must still run normally"
    assert state_calls == [], "pipe state must not be read when diagnostics are off"
    assert not root.exists(), "no dump directory may be created"


def test_run_job_dumps_control_image_and_state_when_enabled(monkeypatch, tmp_path):
    import backends.cuda_worker as cw

    monkeypatch.setenv("HUNYUAN_DEBUG_DUMP", "1")
    monkeypatch.setenv("HUNYUAN_DEBUG_ROOT", str(tmp_path / "dumps"))

    worker = _run_new_worker()
    pipe = _FakePipelineBase()
    worker.pipe = pipe
    job = SimpleNamespace(
        req=_make_req(), init_image=None, controlnet_bindings=[], job_id="jobY"
    )

    with patch("backends.cuda_worker.torch", MagicMock(inference_mode=MagicMock)):
        worker.run_job(job)

    target = tmp_path / "dumps" / "jobY"
    assert (target / "pipe_state_before_scheduler.json").exists()
    assert (target / "pipe_state_after_scheduler.json").exists()
    assert (target / "call_kwargs.json").exists()
    assert (target / "conditioning_keys.json").exists()
    assert (target / "pipe_state_at_call.json").exists()

    kwargs = json.loads((target / "call_kwargs.json").read_text())
    assert kwargs["width"] == 1024
    assert kwargs["use_resolution_binning"] is True

    cond = json.loads((target / "conditioning_keys.json").read_text())
    assert cond["conditioning_keys"] == ["prompt"]
    assert cond["seed"] == 7
