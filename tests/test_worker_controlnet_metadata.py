import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

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
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])


class _FakeStableDiffusionPipeline(_FakePipelineBase):
    pass


class _FakeStableDiffusionXLPipeline(_FakePipelineBase):
    pass


class _FakeStableDiffusionControlNetPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        return cls()


class _FakeStableDiffusionXLControlNetPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        return cls()


sys.modules["diffusers.schedulers.scheduling_lcm"].LCMScheduler = MagicMock()
sys.modules["diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion"].StableDiffusionPipeline = _FakeStableDiffusionPipeline
sys.modules["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl"].StableDiffusionXLPipeline = _FakeStableDiffusionXLPipeline
sys.modules["diffusers"].StableDiffusionControlNetPipeline = _FakeStableDiffusionControlNetPipeline
sys.modules["diffusers"].StableDiffusionXLControlNetPipeline = _FakeStableDiffusionXLControlNetPipeline
sys.modules["diffusers"].ControlNetModel = MagicMock()
sys.modules["backends.styles"].STYLE_REGISTRY = {}

from cn_metadata import build_map_metadata, save_with_metadata  # noqa: E402
from backends.cuda_worker import CudaWorkerBase, DiffusersCudaWorker  # noqa: E402
from server.controlnet_execution import ControlNetBinding  # noqa: E402


def _stamped_png(tmp_path: Path) -> bytes:
    payload = build_map_metadata(
        tool="canny_map",
        control_type="canny",
        source_size=(8, 8),
        params={"low_threshold": 100, "high_threshold": 200},
    )
    dest = tmp_path / "map.png"
    save_with_metadata(Image.new("RGB", (8, 8), "white"), dest, payload)
    return dest.read_bytes()


def _bare_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


def _binding(
    attachment_id: str,
    control_type: str,
    model_id: str,
    image_bytes: bytes,
    strength: float,
) -> ControlNetBinding:
    return ControlNetBinding(
        attachment_id=attachment_id,
        control_type=control_type,
        model_id=model_id,
        model_path="/x",
        control_image_bytes=image_bytes,
        strength=strength,
        start_percent=0.0,
        end_percent=0.7,
    )


def _make_req():
    return SimpleNamespace(
        prompt="an owl",
        negative_prompt="blurry",
        size="8x8",
        num_inference_steps=8,
        guidance_scale=3.0,
        seed=123,
        style_lora=None,
    )


def _make_worker():
    worker = DiffusersCudaWorker.__new__(DiffusersCudaWorker)
    worker.device = "cuda:0"
    worker.dtype = "fp16_sentinel"
    worker.worker_id = 0
    worker.pipe = _FakeStableDiffusionPipeline()
    worker._img2img_pipe = None
    worker._apply_style = Mock()
    worker._apply_request_scheduler = Mock(return_value="euler")
    return worker


def _fake_cache():
    cache = MagicMock()
    cache.acquire.side_effect = lambda model_id, model_path, loader: f"loaded:{model_id}"
    return cache


def _fake_control_image(label: str):
    opened = MagicMock(name=f"{label}_opened")
    converted = MagicMock(name=f"{label}_converted")
    resized = MagicMock(name=f"{label}_resized")
    opened.convert.return_value = converted
    converted.resize.return_value = resized
    return opened, resized


def test_controlnet_metadata_source_populated_and_null(tmp_path: Path):
    worker = CudaWorkerBase.__new__(CudaWorkerBase)
    bindings = [
        _binding("cn-1", "canny", "sdxl-canny", _stamped_png(tmp_path), 0.8),
        _binding("cn-2", "depth", "sdxl-depth", _bare_png(), 1.0),
    ]

    out = worker._controlnet_metadata(bindings)

    assert len(out) == 2
    assert out[0]["attachment_id"] == "cn-1"
    assert out[0]["control_type"] == "canny"
    assert out[0]["generation"] == {
        "model_id": "sdxl-canny",
        "strength": 0.8,
        "start_percent": 0.0,
        "end_percent": 0.7,
    }
    assert out[0]["source"]["tool"] == "canny_map"
    assert out[1]["source"] is None


def test_controlnet_metadata_empty_bindings():
    worker = CudaWorkerBase.__new__(CudaWorkerBase)
    assert worker._controlnet_metadata([]) == []


def test_run_job_writes_controlnet_chunk_when_bindings_present(tmp_path: Path):
    worker = _make_worker()
    req = _make_req()
    job = SimpleNamespace(
        req=req,
        init_image=None,
        controlnet_bindings=[_binding("cn-1", "canny", "sdxl-canny", _stamped_png(tmp_path), 0.8)],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()
    cn_pipe = MagicMock()
    cn_pipe.return_value = SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])
    resized = MagicMock(name="decoded_resized")

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo") as mock_pnginfo, \
         patch("backends.cuda_worker._decode_control_image", return_value=resized), \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache), \
         patch("backends.cuda_worker.DiffusersCudaWorker._build_controlnet_pipe", return_value=cn_pipe):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    add_text_calls = mock_pnginfo.return_value.add_text.call_args_list
    assert add_text_calls[0].args[0] == "lcm"
    assert add_text_calls[1].args[0] == "controlnet"
    payload = json.loads(add_text_calls[1].args[1])
    assert payload[0]["attachment_id"] == "cn-1"
    assert payload[0]["source"]["tool"] == "canny_map"


def test_run_job_skips_controlnet_chunk_without_bindings():
    worker = _make_worker()
    req = _make_req()
    job = SimpleNamespace(req=req, init_image=None, controlnet_bindings=[])
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo") as mock_pnginfo:
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    add_text_calls = mock_pnginfo.return_value.add_text.call_args_list
    assert len(add_text_calls) == 1
    assert add_text_calls[0].args[0] == "lcm"
