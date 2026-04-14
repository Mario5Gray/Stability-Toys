"""
lcm_sr_server.py — RKNN LCM Stable Diffusion FastAPI server (queued, multi-worker safe)

Key goals:
- One pipeline per worker thread (no shared RKNN objects across threads)
- Determin guarantee: per-request seed -> np.RandomState
- Deterministic input ordering handled in RKNN2Model (recommended)
- Explicit data_format per model (UNet + VAE commonly NHWC on RKNN)
- Queue backpressure (429 on overflow)
- Clean startup/shutdown (FastAPI lifespan)
- Returns image bytes + X-Seed header
- Super-resolution:
  - As postprocess on /generate (req.superres=true)
  - As standalone ingest endpoint /superres (multipart upload)
  - Magnitude (1/2/3) controls number of SR passes; defaults to 2

Env:
  MODEL_ROOT=/models/lcm_rknn
  PORT=4200
  NUM_WORKERS=1..3
  QUEUE_MAX=64
  DEFAULT_SIZE=512x512
  DEFAULT_STEPS=4
  DEFAULT_GUIDANCE=1.0
  DEFAULT_TIMEOUT=120

  # CUDA Backend (auto-detects SD1.5 vs SDXL):
  MODEL_ROOT=/path/to/models
  MODEL=model.safetensors           (SD1.5 or SDXL - automatically detected)
  CUDA_DEVICE=cuda:0
  CUDA_DTYPE=fp16|fp32|bf16

  SR_ENABLED=true|false
  SR_MODEL_PATH=/models/lcm_rknn/super-resolution-10.rknn
  SR_INPUT_SIZE=224
  SR_OUTPUT_SIZE=672
  SR_NUM_WORKERS=1..N
  SR_QUEUE_MAX=32
  SR_REQUEST_TIMEOUT=120
  SR_MAX_PIXELS=24000000
"""

import asyncio
import os
import json
import time
import queue
import threading
from concurrent.futures import Future
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from transformers import CLIPTokenizer
from server.comfy_routes import router as comfy_router

from server.ws_routes import ws_router, register_job_hook, _build_status
from server.ws_hub import hub
from server.upload_routes import upload_router, cleanup_uploads_loop

# lcm_sr_server.py (add near imports)
from server.compat_endpoints import CompatEndpoints
from utils.request_logger import RequestLogger  # and optionally RequestLoggerConfig

from persistence.storage_provider import StorageProvider

from backends.base import ModelPaths, Job


from backends.base import PipelineWorker

import logging
import signal

# Mode system imports
from server.mode_config import get_mode_config, reload_mode_config
from server.model_routes import router as model_router
from server.advisor_routes import router as advisor_router
from server.telemetry_routes import router as telemetry_router
from server.workflow_routes import router as workflow_router
from server.file_watcher import start_config_watcher, stop_config_watcher
from server.generation_constraints import finalize_mode_generate_request
from backends.platform_registry import get_backend_provider
from server.superres_http import (
    build_superres_headers,
    load_superres_runtime_settings,
    submit_superres,
)

from .mode_config import MODE_CONFIG_PATH

BACKEND = (os.environ.get("BACKEND") or "").lower().strip()
COMFYUI_ENABLED = os.environ.get("COMFYUI_ENABLED", "false").lower().strip()

logger = logging.getLogger(__name__)
#logging.basicConfig(filename='myapp.log', level=logging.INFO)

# Backend Worker Wrapper


class StyleLoraRequest(BaseModel):
    """
    Exclusive style LoRA selector.
    level: 0 => off, 1..N => preset strength index
    """
    style: Optional[str] = Field(default=None, description="Style id, e.g. 'papercut'. Null/None disables.")
    level: int = Field(default=0, ge=0, le=8, description="0=off, 1..N=style strength preset index")

# -----------------------------
# Request schema (HTTP)
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(default=None, description="Optional negative prompt passed through to the diffusion pipeline.")
    mode: Optional[str] = Field(default=None, description="Optional mode to switch to before generation (e.g. 'sdxl-portrait')")
    scheduler_id: Optional[str] = Field(default=None, description="Optional canonical scheduler override for the active mode.")
    size: str = Field(default=os.environ.get("DEFAULT_SIZE", "512x512"), pattern=r"^\d+x\d+$")
    num_inference_steps: int = Field(default=int(os.environ.get("DEFAULT_STEPS", "4")), ge=1, le=50)
    guidance_scale: float = Field(default=float(os.environ.get("DEFAULT_GUIDANCE", "1.0")), ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    # ---- lora ----
    style_lora: StyleLoraRequest = Field(default_factory=StyleLoraRequest)
    # ---- postprocess ----
    superres: bool = Field(default=False, description="If true, run super-resolution as a postprocess step.")
    superres_format: str = Field(default="png", pattern=r"^(png|jpeg)$")
    superres_quality: int = Field(default=92, ge=1, le=100, description="JPEG quality if superres_format=jpeg.")
    superres_magnitude: int = Field(
        default=2,
        ge=1,
        le=3,
        description="SR magnitude (1..3). Interpreted as number of SR passes. Default=2.",
    )
    denoise_strength: float = Field(
        default=0.75,
        ge=0.01,
        le=1.0,
        description="Denoise strength for img2img (0.01=keep image, 1.0=fully regenerate).",
    )

# -----------------------------
# RKNN multi-context configuration
# -----------------------------
def build_rknn_context_cfgs_for_rk3588(num_workers: int) -> List[dict]:
    core_masks = ["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"]
    cfgs = []
    for i in range(num_workers):
        cfgs.append(
            {
                "multi_context": True,
                "core_mask": core_masks[i % len(core_masks)],
                "context_name": f"w{i}",
                "worker_id": i,
            }
        )
    return cfgs

# -----------------------------
# Singleton Service (LCM generation)
# -----------------------------
class PipelineService:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        paths: ModelPaths,
        num_workers: int,
        queue_max: int,
        rknn_context_cfgs: Optional[List[dict]] = None,
        use_rknn_context_cfgs: bool = True,
    ):
        self.workers: List[PipelineWorker] = []
        self.threads: List[threading.Thread] = []
        self._stop = threading.Event()

        self.paths = paths
        self.num_workers = max(1, int(num_workers))
        self.q: "queue.Queue[Job]" = queue.Queue(maxsize=int(queue_max))

        # decide backend
        use_cuda = False
        if BACKEND == "cuda":
            use_cuda = True
        elif BACKEND == "rknn":
            use_cuda = False
        else:
            try:
                import torch
                use_cuda = torch.cuda.is_available()
            except Exception:
                use_cuda = False

        # 1) enforce sweet-spot policy
        if use_cuda and self.num_workers != 1:
            print(f"[PipelineService] CUDA detected; forcing NUM_WORKERS {self.num_workers} -> 1")
            self.num_workers = 1

        print(f"[PipelineService] BACKEND={BACKEND} use_cuda={use_cuda} workers={self.num_workers} queue_max={queue_max}")

        # 2) only init RKNN assets if needed
        if not use_cuda:
            with open(self.paths.scheduler_config, "r") as f:
                self.scheduler_config = json.load(f)
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

            if rknn_context_cfgs is None:
                rknn_context_cfgs = build_rknn_context_cfgs_for_rk3588(self.num_workers)
            if len(rknn_context_cfgs) != self.num_workers:
                raise ValueError("rknn_context_cfgs must match num_workers length")
        else:
            self.scheduler_config = None
            self.tokenizer = None
            rknn_context_cfgs = None

        # 3) create exactly one worker for cuda, N for rknn
        for i in range(self.num_workers):
            if use_cuda:
                from backends.worker_factory import create_cuda_worker
                w = create_cuda_worker(worker_id=i)  # type: ignore[call-arg]
            else:
                from backends.rknn_worker import RKNNPipelineWorker
                w = RKNNPipelineWorker(
                    worker_id=i,
                    paths=self.paths,
                    scheduler_config=self.scheduler_config,  # type: ignore[arg-type]
                    tokenizer=self.tokenizer,  # type: ignore[arg-type]
                    rknn_context_cfg=rknn_context_cfgs[i],  # type: ignore[index]
                    use_rknn_context_cfgs=use_rknn_context_cfgs,
                )
            self.workers.append(w)

        # 4) thread loop stays the same
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

    @classmethod
    def get_instance(
        cls,
        paths: ModelPaths,
        num_workers: int,
        queue_max: int,
        rknn_context_cfgs: Optional[List[dict]] = None,
        use_rknn_context_cfgs: bool = True,
    ) -> "PipelineService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    paths=paths,
                    num_workers=num_workers,
                    queue_max=queue_max,
                    rknn_context_cfgs=rknn_context_cfgs,
                    use_rknn_context_cfgs=use_rknn_context_cfgs,
                )
            return cls._instance

    def shutdown(self):
        self._stop.set()
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if not job.fut.done():
                job.fut.set_exception(RuntimeError("Service shutting down"))
            self.q.task_done()

    def submit(self, req: GenerateRequest, timeout_s: float = 0.25) -> Future:
        fut: Future = Future()
        job = Job(req=req, fut=fut, submitted_at=time.time())
        try:
            self.q.put(job, timeout=timeout_s)
        except queue.Full:
            fut.set_exception(RuntimeError("Queue full"))
        return fut

    def _worker_loop(self, worker_idx: int):
        worker = self.workers[worker_idx]
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if job.fut.cancelled():
                self.q.task_done()
                continue

            try:
                png, seed = worker.run_job(job)  # type: ignore[arg-type]
                if not job.fut.done():
                    job.fut.set_result((png, seed))
            except Exception as e:
                logger.error(f"Worker {worker_idx} job failed: {e}", exc_info=True)
                if not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                self.q.task_done()


# -----------------------------
# FastAPI server config
# -----------------------------
MODEL_ROOT = os.path.join(os.environ.get('MODEL_ROOT', ''), os.environ.get('MODEL', ''))

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "1"))
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "64"))
PORT = int(os.environ.get("PORT", "4200"))
REQUEST_TIMEOUT = float(os.environ.get("DEFAULT_TIMEOUT", "120"))

USE_RKNN_CONTEXT_CFGS = os.environ.get("USE_RKNN_CONTEXT_CFGS", "1") not in ("0", "false", "False")
model_root_path = ModelPaths(root=MODEL_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI server lifespan...")
    logger.info(f"BACKEND={BACKEND}, NUM_WORKERS={NUM_WORKERS}, LOG_LEVEL={os.getenv('LOG_LEVEL', 'INFO')}")

    try:
        provider = get_backend_provider()
        runtime = provider.create_generation_runtime(
            paths=model_root_path,
            num_workers=NUM_WORKERS,
            queue_max=QUEUE_MAX,
            use_rknn_context_cfgs=USE_RKNN_CONTEXT_CFGS,
        )
        app.state.backend_provider = provider
        app.state.generation_runtime = runtime
        app.state.worker_pool = getattr(runtime, "_pool", None)
        app.state.service = getattr(runtime, "_service", None)
        app.state.use_mode_system = provider.backend_id == "cuda" and app.state.worker_pool is not None

        if app.state.use_mode_system:
            try:
                mode_config = get_mode_config()
                logger.info(f"Mode system initialized: {len(mode_config.list_modes())} modes available")
                logger.info(f"Default mode: {mode_config.get_default_mode()}")
            except FileNotFoundError:
                logger.warning("modes.yaml not found - mode system disabled")
                app.state.use_mode_system = False
            else:
                def sighup_handler(signum, frame):
                    logger.info("Received SIGHUP - reloading modes.yaml")
                    try:
                        reload_mode_config()
                        logger.info("Configuration reloaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to reload configuration: {e}", exc_info=True)

                signal.signal(signal.SIGHUP, sighup_handler)
                logger.info("SIGHUP handler registered for config reload")

                try:
                    start_config_watcher(MODE_CONFIG_PATH, reload_mode_config)
                    logger.info("File watcher started for modes.yaml")
                except Exception as e:
                    logger.warning(f"Failed to start file watcher: {e}")

        logger.info("Generation runtime initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize generation runtime: {e}", exc_info=True)
        raise

    app.state.sr_service = None
    app.state.sr_settings = None
    try:
        sr_settings = load_superres_runtime_settings(os.environ)
        app.state.sr_settings = sr_settings
        app.state.sr_service = provider.create_superres_runtime(settings=sr_settings)
        if app.state.sr_service is not None:
            logger.info("Super-resolution service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SR service: {e}", exc_info=True)
        raise

    try:
        app.state.storage = StorageProvider.make_storage_provider_from_env()
        logger.info("Storage provider initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize storage provider: {e}", exc_info=True)
        raise

    # Wire up WS job update hook and start background tasks
    register_job_hook()
    upload_cleanup_task = asyncio.create_task(cleanup_uploads_loop())
    status_broadcast_task = asyncio.create_task(_status_broadcaster(app))
    logger.info("WebSocket hub, upload cleanup, and status broadcaster started")

    logger.info("Server startup complete")
    yield

    # Cancel background tasks
    upload_cleanup_task.cancel()
    status_broadcast_task.cancel()

    # shutdown
    logger.info("Starting server shutdown...")
    try:
        app.state.storage.close()  # type: ignore[union-attr]
    except Exception as e:
        logger.error(f"Error closing storage: {e}", exc_info=True)

    if getattr(app.state, "use_mode_system", False):
        try:
            stop_config_watcher()
            logger.info("File watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}", exc_info=True)

    if getattr(app.state, "generation_runtime", None) is not None:
        try:
            app.state.generation_runtime.shutdown()
            logger.info("Generation runtime shut down")
        except Exception as e:
            logger.error(f"Error shutting down generation runtime: {e}", exc_info=True)

    if app.state.sr_service is not None:
        try:
            app.state.sr_service.shutdown()
            logger.info("SR service shut down")
        except Exception as e:
            logger.error(f"Error shutting down SR service: {e}", exc_info=True)

async def _status_broadcaster(app: FastAPI):
    """Broadcast system:status to all WS clients every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        try:
            if hub.client_count > 0:
                msg = _build_status(app.state)
                await hub.broadcast(msg)
        except Exception:
            pass


app = FastAPI(lifespan=lifespan, title="LCM_Stable_Diffusion and Super_Resolution Service")

# Global exception handler to ensure all errors are logged
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

RequestLogger.install(app)

def _store_image_blob(
    storage: Optional[StorageProvider],
    *,
    out_bytes: bytes,
    media_type: str,
    req: GenerateRequest,
    seed: int,
    did_superres: bool,
    sr_mag: int,
) -> Optional[str]:
    if storage is None:
        return None

    image_key = StorageProvider._new_key("lcm_image")
    storage.put(
        image_key,
        out_bytes,
        content_type=media_type,
        meta={
            "prompt": req.prompt,
            "seed": seed,
            "size": req.size,
            "steps": req.num_inference_steps,
            "cfg": req.guidance_scale,
            "superres": bool(did_superres),
            "sr_magnitude": int(sr_mag) if did_superres else 0,
        },
    )
    return image_key


@app.post("/generate", responses={200: {"content": {"image/png": {}, "image/jpeg": {}}}})
def generate(req: GenerateRequest):    
    runtime = app.state.generation_runtime
    supports_modes = hasattr(runtime, "switch_mode")

    if supports_modes and req.mode is not None:
        current_mode = runtime.get_current_mode()
        if current_mode != req.mode:
            try:
                switch_fut = runtime.switch_mode(req.mode)
                switch_fut.result(timeout=30.0)
                logger.info(f"[/generate] Switched to mode: {req.mode}")
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Mode '{req.mode}' not found in modes.yaml"
                )
            except Exception as e:
                logger.error(f"[/generate] Mode switch failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Mode switch failed: {e}"
                )

    current_mode = runtime.get_current_mode() if supports_modes else None
    if current_mode:
        mode_config = get_mode_config()
        mode = mode_config.get_mode(current_mode)

        try:
            finalize_mode_generate_request(
                req,
                mode,
                env_default_size=os.environ.get("DEFAULT_SIZE", "512x512"),
                env_default_steps=int(os.environ.get("DEFAULT_STEPS", "4")),
                env_default_guidance=float(os.environ.get("DEFAULT_GUIDANCE", "1.0")),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        fut = runtime.submit_generate(req, timeout_s=0.25)
    except queue.Full:
        raise HTTPException(status_code=429, detail="Too many requests (queue full). Try again.")

    mode_used = current_mode

    # Wait for generation result
    try:
        png_bytes, seed = fut.result(timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logger.error(f"Generate endpoint failed: {e}", exc_info=True)
        msg = str(e)
        if "Queue full" in msg:
            raise HTTPException(status_code=429, detail="Too many requests (queue full). Try again.")
        raise HTTPException(status_code=500, detail=f"Generation failed: {msg}")

    out_bytes = png_bytes
    media_type = "image/png"

    storage = getattr(app.state, "storage", None)

    sr_mag = int(req.superres_magnitude or 2)
    if sr_mag < 1 or sr_mag > 3:
        raise HTTPException(status_code=400, detail="superres_magnitude must be 1..3")
    
    did_superres = False

    if req.superres:
        sr = getattr(app.state, "sr_service", None)
        if sr is None:
            raise HTTPException(status_code=503, detail="Super-resolution requested but SR service is disabled")

        try:
            out_bytes = submit_superres(
                sr_service=sr,
                image_bytes=png_bytes,
                out_format=req.superres_format,
                quality=req.superres_quality,
                magnitude=sr_mag,
                queue_timeout_s=0.25,
                request_timeout_s=app.state.sr_settings.sr_request_timeout,
            )
            did_superres = True
            media_type = "image/jpeg" if req.superres_format == "jpeg" else "image/png"

        except Exception as e:
            logger.error(f"Super-resolution failed in /generate: {e}", exc_info=True)
            msg = str(e)
            if "Queue full" in msg:
                raise HTTPException(status_code=429, detail="Too many requests (SR queue full). Try again.")
            raise HTTPException(status_code=500, detail=f"Super-resolution failed: {msg}")

    image_key = _store_image_blob(
        storage,
        out_bytes=out_bytes,
        media_type=media_type,
        req=req,
        seed=int(seed),
        did_superres=did_superres,
        sr_mag=sr_mag
    )

    headers = {
        "Cache-Control": "no-store",
        "X-Seed": str(seed),
        "X-SuperRes": "1" if did_superres else "0",
    }

    # Add mode header if using mode system
    if mode_used:
        headers["X-Mode"] = mode_used

    if image_key:
        headers["X-LCM-Image-Key"] = image_key

    if did_superres:
        headers.update(
            build_superres_headers(sr, magnitude=sr_mag, out_format=req.superres_format)
        )

    return Response(content=out_bytes, media_type=media_type, headers=headers)


@app.post("/superres", responses={200: {"content": {"image/png": {}, "image/jpeg": {}}}})
async def superres(
    file: UploadFile = File(...),
    magnitude: int = Form(2),  # default=2
    out_format: str = Form("png"),
    quality: int = Form(92),
):
    sr = getattr(app.state, "sr_service", None)
    if sr is None:
        raise HTTPException(status_code=503, detail="Super-resolution disabled")

    # Manual validation (FastAPI Form() doesn't enforce ge/le)
    try:
        magnitude = int(magnitude)
    except Exception:
        raise HTTPException(status_code=400, detail="magnitude must be an integer 1..3")
    if magnitude < 1 or magnitude > 3:
        raise HTTPException(status_code=400, detail="magnitude must be 1..3")

    out_format = (out_format or "png").lower().strip()
    if out_format not in ("png", "jpeg"):
        raise HTTPException(status_code=400, detail="out_format must be png or jpeg")

    try:
        quality = int(quality)
    except Exception:
        raise HTTPException(status_code=400, detail="quality must be an integer 1..100")
    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="quality must be 1..100")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        out_bytes = submit_superres(
            sr_service=sr,
            image_bytes=data,
            out_format=out_format,
            quality=quality,
            magnitude=magnitude,
            queue_timeout_s=0.25,
            request_timeout_s=app.state.sr_settings.sr_request_timeout,
        )
    except Exception as e:
        logger.error(f"Super-resolution failed in /superres: {e}", exc_info=True)
        msg = str(e)
        if "Queue full" in msg:
            raise HTTPException(status_code=429, detail="Too many requests (SR queue full). Try again.")
        raise HTTPException(status_code=500, detail=f"Super-resolution failed: {msg}")

    media_type = "image/jpeg" if out_format == "jpeg" else "image/png"

    storage = getattr(app.state, "storage", None)
    image_key = None

    if storage is not None:
        image_key = StorageProvider._new_key("lcm_image")
        storage.put(
            image_key,
            out_bytes,
            content_type=media_type,
            meta={
                "source_upload": file.filename,
                "sr_only": True,
                "sr_magnitude": magnitude,
                "out_format": out_format,
                "quality": quality,
            }
        )

    headers = {
        "Cache-Control": "no-store",
        "X-SR-Magnitude": str(magnitude),
    }
    headers.update(build_superres_headers(sr, magnitude=magnitude, out_format=out_format))

    if image_key:
        headers["X-LCM-Image-Key"] = image_key

    return Response(
        content=out_bytes,
        media_type=media_type,
        headers=headers,
    )


@app.post("/v1/superres", responses={200: {"content": {"image/png": {}, "image/jpeg": {}}}})
async def superres_v1(
    file: UploadFile = File(...),
    magnitude: int = Form(2),
    out_format: str = Form("png"),
    quality: int = Form(92),
):
    return await superres(file=file, magnitude=magnitude, out_format=out_format, quality=quality)

def _run_generate_from_dict(gen_req: dict):
    """
    Shared internal runner used by external compat endpoints.
    Returns: (bytes, seed_used, meta_headers)
    """
    # Build internal request
    req = GenerateRequest(**gen_req)

    runtime = app.state.generation_runtime

    # ---- base SD generation ----
    fut = runtime.submit_generate(req, timeout_s=0.25)
    png_bytes, seed = fut.result(timeout=REQUEST_TIMEOUT)

    # ---- optional SR postprocess ----
    out_bytes = png_bytes

    meta_headers = {
        "X-Seed": str(seed),
        "X-SuperRes": "0",
    }

    if req.superres:
        sr = getattr(app.state, "sr_service", None)
        if sr is None:
            # For compat callers, raise a normal exception (will become 500)
            raise RuntimeError("Super-resolution requested but SR service is disabled")

        sr_mag = int(req.superres_magnitude or 2)

        out_bytes = submit_superres(
            sr_service=sr,
            image_bytes=png_bytes,
            out_format=req.superres_format,
            quality=req.superres_quality,
            magnitude=sr_mag,
            queue_timeout_s=0.25,
            request_timeout_s=app.state.sr_settings.sr_request_timeout,
        )

        meta_headers.update(
            {"X-SuperRes": "1", **build_superres_headers(sr, magnitude=sr_mag, out_format=req.superres_format)}
        )

    return out_bytes, seed, meta_headers

@app.get("/storage/health")
def storage_health():
    st: StorageProvider = app.state.storage
    return st.health()  # type: ignore[attr-defined]

@app.put("/storage/{key}")
def storage_put(key: str, payload: str = Body(..., embed=True)):
    st: StorageProvider = app.state.storage
    st.put(key, payload.encode("utf-8"), content_type="text/plain", ttl_s=300)
    return {"ok": True, "key": key}

@app.get("/storage/{key}")
def storage_get(key: str):
    if not os.environ.get("STORAGE_ENABLE_HTTP", 1):
        raise HTTPException(404, detail="storage http disabled")

    st = getattr(app.state, "storage", None)
    if st is None:
        raise HTTPException(503, detail="storage unavailable")

    item = st.get(key)
    if not item:
        raise HTTPException(404, detail="not found")

    return Response(
        content=item.value,
        media_type=item.content_type,
        headers={
            "Cache-Control": "no-store",
            "X-Storage-Key": item.key,
            "X-Storage-Created-At": str(item.created_at),
        },
    )

@app.get("/health")
def health():
    return {"status": "ok"}

# OpenAI compatible endpoint
CompatEndpoints(app=app, run_generate=_run_generate_from_dict).mount()

# Model management API
app.include_router(model_router)
app.include_router(advisor_router)
app.include_router(telemetry_router)
app.include_router(workflow_router)
logger.info("Model management API mounted at /api")

# Comfyui invoker
if COMFYUI_ENABLED:
    app.include_router(comfy_router)

# WebSocket + upload routes
app.include_router(ws_router)
app.include_router(upload_router)
logger.info("WebSocket endpoint mounted at /v1/ws, upload at /v1/upload")

# UI static mount (serves Vite dist)
_ui_dist = "/opt/lcm-sr-server/ui-dist"
if os.path.isdir(_ui_dist):
    app.mount(
        "/",
        StaticFiles(directory=_ui_dist, html=True),
        name="ui",
    )
else:
    logger.warning(f"UI dist not found at {_ui_dist}; skipping static mount")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # CHANGE ME!!!
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    from logging_config import LOGGING_CONFIG

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=LOGGING_CONFIG,
    )
