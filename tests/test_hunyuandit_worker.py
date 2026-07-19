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
