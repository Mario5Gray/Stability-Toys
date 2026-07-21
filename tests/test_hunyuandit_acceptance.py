"""Live HunyuanDiT production-path acceptance.

Runs only on a real CUDA host with the production model directories mounted
under /models. This exercises the production WorkerPool path against the
Task-10 Hunyuan mode defined in /app/conf, not the generic test config mounted
at /conf/modes.yml.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import time
from pathlib import Path

import pytest
import torch
from hunyuan_control_map import control_map_png
from hunyuan_warning_policy import blocking_ignored_warnings
from PIL import Image, PngImagePlugin

from backends.model_registry import ModelRegistry
from backends.platforms.cuda import CUDAProvider
from backends.worker_pool import (
    GenerationJob,
    StaleResolutionError,
    WorkerPool,
    reset_worker_pool,
)
from server.asset_store import InMemoryAssetStore
from server.controlnet_execution import (
    admit_generation_operation,
    resolve_controlnet_bindings,
)
from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest
from server.controlnet_registry import reset_controlnet_registry
from server.lcm_sr_server import GenerateRequest
from server.mode_config import ModeConfigManager


def _cuda_functional() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.tensor([0.0], device="cuda")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.requires_gpu,
    pytest.mark.skipif(not _cuda_functional(), reason="CUDA not functional"),
]


def _production_conf_dir() -> Path:
    for candidate in (Path("/app/conf"), Path(__file__).resolve().parents[1] / "conf"):
        if (candidate / "modes.yml").exists() and (candidate / "controlnets.yaml").exists():
            return candidate
    raise AssertionError("production conf/ not available")


def _require_runtime_path(path: Path, label: str) -> None:
    if not path.exists():
        pytest.skip(f"{label} not mounted at {path}")


def _write_png(path: Path, png_bytes: bytes) -> None:
    img = Image.open(io.BytesIO(png_bytes))
    img.load()
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in img.text.items():
        pnginfo.add_text(key, value)
    img.save(path, format="PNG", pnginfo=pnginfo)


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _captured_diffusers_warnings():
    """Collect diffusers warnings raised anywhere, including worker threads.

    The diffusers library root logger sets propagate=False, so caplog does not
    see these records; attach directly. Spec section 8 requires acceptance to
    fail on new config incompatibilities, and the ones that matter most here
    are warn-and-continue, not raise.
    """
    handler = _RecordingHandler()
    logger = logging.getLogger("diffusers")
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _pick_switch_target(mode_config: ModeConfigManager) -> str:
    for mode_name in mode_config.list_modes():
        if mode_name == "HunyuanDiT":
            continue
        mode = mode_config.get_mode(mode_name)
        if mode.model_path and Path(mode.model_path).exists():
            return mode_name
    pytest.skip("no secondary production mode mounted for stale-epoch switch proof")


def _job(req: GenerateRequest, *, bindings, epoch: int) -> GenerationJob:
    return GenerationJob(
        req=req,
        controlnet_bindings=list(bindings),
        resolution_epoch=epoch,
    )


def test_hunyuandit_workerpool_acceptance(monkeypatch, tmp_path):
    conf_dir = _production_conf_dir()
    monkeypatch.setenv("CONTROLNET_REGISTRY_PATH", str(conf_dir / "controlnets.yaml"))
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "strict")
    monkeypatch.setenv("MODE_CONFIG_PATH", str(conf_dir))
    reset_controlnet_registry()
    reset_worker_pool()

    mode_config = ModeConfigManager(str(conf_dir))
    hunyuan_mode = mode_config.get_mode("HunyuanDiT")
    _require_runtime_path(Path(hunyuan_mode.model_path), "HunyuanDiT base model")
    _require_runtime_path(
        Path("/models/controlnets/HunyuanDiT-v1.1-ControlNet-Canny"),
        "HunyuanDiT Canny ControlNet",
    )

    provider = CUDAProvider()
    pool = WorkerPool(queue_max=2, mode_config=mode_config, registry=ModelRegistry())

    try:
        pool.switch_mode("HunyuanDiT", force=True).result(timeout=600.0)
        snapshot = pool.get_active_model_snapshot()
        assert snapshot is not None
        assert snapshot.mode_name == "HunyuanDiT"
        assert snapshot.resolved.profile.family_id == "hunyuandit"
        caps = provider.family_binding("hunyuandit").execution_capabilities
        assert caps.supports_img2img is False
        assert caps.supports_controlnet is True
        assert caps.supports_img2img_and_controlnet is False

        # Admission rejects the unsupported Hunyuan img2img paths before any
        # preprocessing/store activity is needed.
        source_control = ControlNetAttachment(
            attachment_id="cn-src",
            control_type="canny",
            source_asset_ref="upload-ref",
            preprocess=ControlNetPreprocessRequest(id="canny"),
        )
        with pytest.raises(ValueError, match="img2img operation"):
            admit_generation_operation(
                GenerateRequest(prompt="Profile of a smiling man wearing a taupe color cap."),
                snapshot=snapshot,
                provider=provider,
                has_init_image=True,
            )
        with pytest.raises(ValueError, match="img2img\\+controlnet operation"):
            admit_generation_operation(
                GenerateRequest(prompt="Profile of a smiling man wearing a taupe color cap.", controlnets=[source_control]),
                snapshot=snapshot,
                provider=provider,
                has_init_image=True,
            )

        store = InMemoryAssetStore()
        map_ref = store.write("control_map", control_map_png())
        attachment = ControlNetAttachment(
            attachment_id="cn-map",
            control_type="canny",
            model_id="hunyuandit-canny",
            map_asset_ref=map_ref,
            strength=1.0,
        )
        req = GenerateRequest(
            prompt="architectural study, crisp edges, daylight",
            negative_prompt="blurry, noisy, low quality",
            size="1024x1024",
            num_inference_steps=30,
            guidance_scale=5.0,
            seed=1337,
            controlnets=[attachment],
        )

        admitted = admit_generation_operation(
            req,
            snapshot=snapshot,
            provider=provider,
            has_init_image=False,
        )
        assert admitted == "controlnet"
        bindings = resolve_controlnet_bindings(
            req,
            mode=snapshot.mode,
            store=store,
            active_family=snapshot.resolved.profile.family_id,
        )
        assert len(bindings) == 1
        assert bindings[0].model_id == "hunyuandit-canny"

        epoch = snapshot.resolution_epoch
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with _captured_diffusers_warnings() as diffusers_warnings:
            png_bytes, seed = pool.submit_job(
                _job(req, bindings=bindings, epoch=epoch)
            ).result(timeout=900.0)
        elapsed_s = time.monotonic() - started
        peak_allocated = int(torch.cuda.max_memory_allocated())

        out_dir = Path(os.environ.get("ACCEPTANCE_OUT_DIR", tmp_path))
        output_path = out_dir / "hunyuandit-canny-1024-acceptance.png"
        _write_png(output_path, png_bytes)

        # Printed before any assertion: a failing run must still surrender its
        # measurement, since the re-baselined VRAM figure is itself a Task 10
        # deliverable.
        print(
            "[acceptance] "
            f"artifact={output_path} elapsed_s={elapsed_s:.2f} "
            f"peak_allocated_bytes={peak_allocated} "
            f"torch={torch.__version__} cuda={torch.version.cuda}"
        )

        img = Image.open(io.BytesIO(png_bytes))
        img.load()
        assert img.size == (1024, 1024)
        assert seed == 1337
        assert "lcm" in img.text
        assert "controlnet" in img.text

        # Size, seed, and PNG text chunks all hold for pure noise, so the
        # warning stream is the only in-test signal for a silently dropped
        # denoiser input. Policy (spec section 8) lives in hunyuan_warning_policy
        # so it stays unit-testable off-GPU: the known learn_sigma/norm_type
        # config notice is allowed, a dropped cross_attention_kwarg never is.
        blocking = blocking_ignored_warnings(diffusers_warnings.messages)
        assert not blocking, (
            f"diffusers reported {len(blocking)} blocking ignored-input "
            f"warning(s); first: {blocking[0]}"
        )

        status = pool.free_vram("hunyuandit-acceptance")
        assert status["is_loaded"] is False
        assert status["vram"]["allocated_bytes"] <= peak_allocated

        old_epoch = pool.current_resolution_epoch()
        switch_target = _pick_switch_target(mode_config)
        pool.switch_mode(switch_target, force=True).result(timeout=600.0)
        assert pool.current_resolution_epoch() > old_epoch

        stale_future = pool.submit_job(_job(req, bindings=bindings, epoch=old_epoch))
        with pytest.raises(StaleResolutionError):
            stale_future.result(timeout=60.0)
    finally:
        pool.shutdown()
        reset_worker_pool()
        reset_controlnet_registry()
