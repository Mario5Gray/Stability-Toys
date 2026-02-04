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
import io
import os
import json
import time
import queue
import threading
from dataclasses import dataclass
from concurrent.futures import Future
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from transformers import CLIPTokenizer
from PIL import Image

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
from backends.worker_pool import get_worker_pool, GenerationJob
from server.model_routes import router as model_router
from server.workflow_routes import router as workflow_router
from server.file_watcher import start_config_watcher, stop_config_watcher

# Try to import RKNNLite for super-resolution
try:
    from rknnlite.api import RKNNLite
    RKNNLITE_AVAILABLE = True
except ImportError:
    RKNNLITE_AVAILABLE = False
    RKNNLite = None

from .mode_config import MODE_CONFIG_PATH

BACKEND = os.environ.get("BACKEND", "auto").lower().strip()  # auto|rknn|cuda
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
    mode: Optional[str] = Field(default=None, description="Optional mode to switch to before generation (e.g. 'sdxl-portrait')")
    size: str = Field(default=os.environ.get("DEFAULT_SIZE", "512x512"), pattern=r"^\d+x\d+$")
    num_inference_steps: int = Field(default=int(os.environ.get("DEFAULT_STEPS", "4")), ge=1, le=50)
    guidance_scale: float = Field(default=float(os.environ.get("DEFAULT_GUIDANCE", "1.0")), ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    # ---- lora ----
    style_lora: StyleLoraRequest = Field(default_factory=StyleLoraRequest)
    # ---- postprocess ----
    superres: bool = Field(default=False, description="If true, run RKNN super-resolution as a postprocess step.")
    superres_format: str = Field(default="png", pattern=r"^(png|jpeg)$")
    superres_quality: int = Field(default=92, ge=1, le=100, description="JPEG quality if superres_format=jpeg.")
    superres_magnitude: int = Field(
        default=2,
        ge=1,
        le=3,
        description="SR magnitude (1..3). Interpreted as number of SR passes. Default=2.",
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
                w = create_cuda_worker(worker_id=i)
            else:
                from backends.rknn_worker import RKNNPipelineWorker
                w = RKNNPipelineWorker(
                    worker_id=i,
                    paths=self.paths,
                    scheduler_config=self.scheduler_config,
                    tokenizer=self.tokenizer,
                    rknn_context_cfg=rknn_context_cfgs[i],
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
                png, seed = worker.run_job(job)
                if not job.fut.done():
                    job.fut.set_result((png, seed))
            except Exception as e:
                logger.error(f"Worker {worker_idx} job failed: {e}", exc_info=True)
                if not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                self.q.task_done()


# -----------------------------
# Super-resolution worker/service
# -----------------------------
@dataclass
class SRJob:
    image_bytes: bytes
    out_format: str
    quality: int
    magnitude: int  # 1..3
    fut: Future
    submitted_at: float


class SuperResWorker:
    def __init__(self, worker_id: int, model_path: str, input_size: int, output_size: int):
        if not RKNNLITE_AVAILABLE:
            raise RuntimeError("RKNNLite not available for SR - install rknnlite package")

        self.worker_id = worker_id
        self.model_path = model_path
        self.input_size = int(input_size)
        self.output_size = int(output_size)

        self.rknn = RKNNLite()
        self._init_runtime()

    def _init_runtime(self):
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"SR load_rknn failed: {ret}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"SR init_runtime failed: {ret}")
        print(f"[SR] worker {self.worker_id} loaded {self.model_path}")

    def close(self):
        rel = getattr(self.rknn, "release", None)
        if callable(rel):
            rel()

    def _plan_tiles(self, w: int, h: int):
        tile = self.input_size
        step = tile
        xs = list(range(0, max(1, w - tile + 1), step))
        ys = list(range(0, max(1, h - tile + 1), step))
        if not xs or xs[-1] != w - tile:
            xs.append(max(0, w - tile))
        if not ys or ys[-1] != h - tile:
            ys.append(max(0, h - tile))
        return [(x, y) for y in ys for x in xs]

    def upscale_once(self, image_bytes: bytes, out_format: str = "png", quality: int = 92) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if img.width * img.height > SR_MAX_PIXELS:
            raise RuntimeError(
                f"Image too large: {img.width}x{img.height} exceeds SR_MAX_PIXELS={SR_MAX_PIXELS}"
            )

        img_ycc = img.convert("YCbCr")
        img_y, img_cb, img_cr = img_ycc.split()
        img_y_np = (np.array(img_y, dtype=np.float32) / 255.0)

        in_w, in_h = img.width, img.height
        scale = self.output_size / self.input_size
        out_w = int(round(in_w * scale))
        out_h = int(round(in_h * scale))

        out_y = np.zeros((out_h, out_w), dtype=np.float32)

        tiles = self._plan_tiles(in_w, in_h)
        for (x0, y0) in tiles:
            crop = img_y_np[y0 : y0 + self.input_size, x0 : x0 + self.input_size]
            inp = crop[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,H,W)

            pred = self.rknn.inference(inputs=[inp])[0]
            tile_out = pred[0, 0] if pred.ndim == 4 else pred[0][0]

            ox0 = int(round(x0 * scale))
            oy0 = int(round(y0 * scale))
            out_y[oy0 : oy0 + self.output_size, ox0 : ox0 + self.output_size] = tile_out

        out_y_u8 = np.uint8(np.clip(out_y * 255.0, 0, 255.0))
        out_img = Image.merge(
            "YCbCr",
            [
                Image.fromarray(out_y_u8, mode="L"),
                img_cb.resize((out_w, out_h), Image.BICUBIC),
                img_cr.resize((out_w, out_h), Image.BICUBIC),
            ],
        ).convert("RGB")

        buf = io.BytesIO()
        if out_format == "jpeg":
            out_img.save(buf, format="JPEG", quality=int(quality))
        else:
            out_img.save(buf, format="PNG")
        return buf.getvalue()

    def upscale_bytes(self, image_bytes: bytes, *, magnitude: int, out_format: str, quality: int) -> bytes:
        mag = int(magnitude)
        if mag < 1 or mag > 3:
            raise RuntimeError("magnitude must be 1..3")
        out = image_bytes
        for _ in range(mag):
            out = self.upscale_once(out, out_format=out_format, quality=quality)
        return out


class SuperResService:
    def __init__(self, model_path: str, num_workers: int, queue_max: int, input_size: int, output_size: int):
        self.model_path = model_path
        self.num_workers = max(1, int(num_workers))
        self.q: "queue.Queue[SRJob]" = queue.Queue(maxsize=int(queue_max))

        self.workers: List[SuperResWorker] = []
        self.threads: List[threading.Thread] = []
        self._stop = threading.Event()

        for i in range(self.num_workers):
            w = SuperResWorker(
                worker_id=i,
                model_path=self.model_path,
                input_size=input_size,
                output_size=output_size,
            )
            self.workers.append(w)

        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

    def shutdown(self):
        self._stop.set()
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if not job.fut.done():
                job.fut.set_exception(RuntimeError("SuperResService shutting down"))
            self.q.task_done()

        for w in self.workers:
            try:
                w.close()
            except Exception:
                pass

    def submit(
        self,
        image_bytes: bytes,
        *,
        out_format: str,
        quality: int,
        magnitude: int,
        timeout_s: float = 0.25,
    ) -> Future:
        fut: Future = Future()
        job = SRJob(
            image_bytes=image_bytes,
            out_format=out_format,
            quality=int(quality),
            magnitude=int(magnitude),
            fut=fut,
            submitted_at=time.time(),
        )
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
                out_bytes = worker.upscale_bytes(
                    job.image_bytes,
                    magnitude=job.magnitude,
                    out_format=job.out_format,
                    quality=job.quality,
                )
                if not job.fut.done():
                    job.fut.set_result(out_bytes)
            except Exception as e:
                logger.error(f"SR Worker {worker_idx} job failed: {e}", exc_info=True)
                if not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                self.q.task_done()


# -----------------------------
# FastAPI server config
# -----------------------------
MODEL_ROOT = os.path.join(os.environ.get('MODEL_ROOT'), os.environ.get('MODEL'))

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "1"))
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "64"))
PORT = int(os.environ.get("PORT", "4200"))
REQUEST_TIMEOUT = float(os.environ.get("DEFAULT_TIMEOUT", "120"))

SR_ENABLED = os.environ.get("SR_ENABLED", "1") not in ("0", "false", "False")
SR_MODEL_PATH = os.environ.get("SR_MODEL_PATH", os.path.join(os.environ.get('MODEL_ROOT'), "super-resolution-10.rknn"))
SR_INPUT_SIZE = int(os.environ.get("SR_INPUT_SIZE", "224"))
SR_OUTPUT_SIZE = int(os.environ.get("SR_OUTPUT_SIZE", "672"))
SR_NUM_WORKERS = int(os.environ.get("SR_NUM_WORKERS", "1"))
SR_QUEUE_MAX = int(os.environ.get("SR_QUEUE_MAX", "32"))
SR_REQUEST_TIMEOUT = float(os.environ.get("SR_REQUEST_TIMEOUT", "120"))
SR_MAX_PIXELS = int(os.environ.get("SR_MAX_PIXELS", "24000000"))
USE_RKNN_CONTEXT_CFGS = os.environ.get("USE_RKNN_CONTEXT_CFGS", "1") not in ("0", "false", "False")
model_root_path = ModelPaths(root=MODEL_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI server lifespan...")
    logger.info(f"BACKEND={BACKEND}, NUM_WORKERS={NUM_WORKERS}, LOG_LEVEL={os.getenv('LOG_LEVEL', 'INFO')}")

    # Determine if using CUDA backend
    use_cuda = False
    if BACKEND == "cuda":
        use_cuda = True
    elif BACKEND == "auto":
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

    # Initialize pipeline service
    try:
        if use_cuda:
            # CUDA backend: Use new mode system with WorkerPool
            logger.info("Using CUDA backend with mode system")

            # Initialize mode configuration
            try:
                mode_config = get_mode_config()
                logger.info(f"Mode system initialized: {len(mode_config.list_modes())} modes available")
                logger.info(f"Default mode: {mode_config.get_default_mode()}")
            except FileNotFoundError:
                logger.warning("modes.yaml not found - mode system disabled, falling back to legacy behavior")
                # Fallback to direct PipelineService
                app.state.service = PipelineService.get_instance(
                    paths=model_root_path,
                    num_workers=NUM_WORKERS,
                    queue_max=QUEUE_MAX,
                    rknn_context_cfgs=build_rknn_context_cfgs_for_rk3588(NUM_WORKERS),
                    use_rknn_context_cfgs=USE_RKNN_CONTEXT_CFGS,
                )
                use_cuda = False  # Disable mode system features
            else:
                # Initialize worker pool with mode system
                app.state.worker_pool = get_worker_pool()
                app.state.service = None  # Not using PipelineService

                # Setup SIGHUP handler for config reload
                def sighup_handler(signum, frame):
                    logger.info("Received SIGHUP - reloading modes.yaml")
                    try:
                        reload_mode_config()
                        logger.info("Configuration reloaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to reload configuration: {e}", exc_info=True)

                signal.signal(signal.SIGHUP, sighup_handler)
                logger.info("SIGHUP handler registered for config reload")

                # Start file watcher for hot-reload
                try:
                    start_config_watcher(MODE_CONFIG_PATH, reload_mode_config)
                    logger.info("File watcher started for modes.yaml")
                except Exception as e:
                    logger.warning(f"Failed to start file watcher: {e}")
        else:
            # RKNN backend: Use legacy PipelineService
            logger.info("Using RKNN backend with legacy PipelineService")
            app.state.service = PipelineService.get_instance(
                paths=model_root_path,
                num_workers=NUM_WORKERS,
                queue_max=QUEUE_MAX,
                rknn_context_cfgs=build_rknn_context_cfgs_for_rk3588(NUM_WORKERS),
                use_rknn_context_cfgs=USE_RKNN_CONTEXT_CFGS,
            )

        # Store backend type for later use
        app.state.use_mode_system = use_cuda and hasattr(app.state, 'worker_pool')
        logger.info("Pipeline service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline service: {e}", exc_info=True)
        raise

    app.state.sr_service = None
    if SR_ENABLED:
        if not os.path.isfile(SR_MODEL_PATH):
            logger.error(f"SR model not found at SR_MODEL_PATH={SR_MODEL_PATH}")
            raise RuntimeError(f"SR model not found at SR_MODEL_PATH={SR_MODEL_PATH}")

        try:
            app.state.sr_service = SuperResService(
                model_path=SR_MODEL_PATH,
                num_workers=SR_NUM_WORKERS,
                queue_max=SR_QUEUE_MAX,
                input_size=SR_INPUT_SIZE,
                output_size=SR_OUTPUT_SIZE,
            )
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
        app.state.storage.close()
    except Exception as e:
        logger.error(f"Error closing storage: {e}", exc_info=True)

    # Shutdown pipeline service or worker pool
    if app.state.use_mode_system and hasattr(app.state, 'worker_pool'):
        try:
            # Stop file watcher
            stop_config_watcher()
            logger.info("File watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}", exc_info=True)

        try:
            app.state.worker_pool.shutdown()
            logger.info("Worker pool shut down")
        except Exception as e:
            logger.error(f"Error shutting down worker pool: {e}", exc_info=True)
    elif app.state.service is not None:
        try:
            app.state.service.shutdown()
            logger.info("Pipeline service shut down")
        except Exception as e:
            logger.error(f"Error shutting down pipeline service: {e}", exc_info=True)

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
    # Check if using mode system (CUDA with modes.yaml)
    if getattr(app.state, 'use_mode_system', False):
        pool = app.state.worker_pool
        # Handle mode switching if requested
        if req.mode is not None:
            current_mode = pool.get_current_mode()
            if current_mode != req.mode:
                # Queue mode switch
                try:
                    switch_fut = pool.switch_mode(req.mode)
                    # Wait for mode switch to complete
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
        # Apply mode defaults if not specified in request
        current_mode = pool.get_current_mode()
        if current_mode:
            mode_config = get_mode_config()
            mode = mode_config.get_mode(current_mode)

            # Apply defaults from mode if not provided
            if req.size == os.environ.get("DEFAULT_SIZE", "512x512"):
                req.size = mode.default_size
            if req.num_inference_steps == int(os.environ.get("DEFAULT_STEPS", "4")):
                req.num_inference_steps = mode.default_steps
            if req.guidance_scale == float(os.environ.get("DEFAULT_GUIDANCE", "1.0")):
                req.guidance_scale = mode.default_guidance

        # Submit generation job
        job = GenerationJob(req=req)
        try:
            fut = pool.submit_job(job)
        except queue.Full:
            raise HTTPException(status_code=429, detail="Too many requests (queue full). Try again.")

        mode_used = current_mode
    else:
        # Legacy PipelineService
        print("legacy pipeline service.")
        logger.info("Legacy Pipeline Service")
        service: PipelineService = app.state.service
        fut = service.submit(req, timeout_s=0.25)
        mode_used = None

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
        sr: Optional[SuperResService] = getattr(app.state, "sr_service", None)
        if sr is None:
            raise HTTPException(status_code=503, detail="Super-resolution requested but SR service is disabled")

        sr_fut = sr.submit(
            image_bytes=png_bytes,
            out_format=req.superres_format,
            quality=req.superres_quality,
            magnitude=sr_mag,
            timeout_s=0.25,
        )
        try:
            out_bytes = sr_fut.result(timeout=SR_REQUEST_TIMEOUT)
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
            {
                "X-SR-Model": os.path.basename(SR_MODEL_PATH),
                "X-SR-Passes": str(sr_mag),
                "X-SR-Scale-Per-Pass": (
                    str(SR_OUTPUT_SIZE // SR_INPUT_SIZE)
                    if SR_OUTPUT_SIZE % SR_INPUT_SIZE == 0
                    else str(SR_OUTPUT_SIZE / SR_INPUT_SIZE)
                ),
                "X-SR-Format": req.superres_format,
            }
        )

    return Response(content=out_bytes, media_type=media_type, headers=headers)


@app.post("/superres", responses={200: {"content": {"image/png": {}, "image/jpeg": {}}}})
async def superres(
    file: UploadFile = File(...),
    magnitude: int = Form(2),  # default=2
    out_format: str = Form("png"),
    quality: int = Form(92),
):
    sr: Optional[SuperResService] = getattr(app.state, "sr_service", None)
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

    fut = sr.submit(
        data,
        out_format=out_format,
        quality=quality,
        magnitude=magnitude,
        timeout_s=0.25,
    )
    try:
        out_bytes = fut.result(timeout=SR_REQUEST_TIMEOUT)
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
        "X-SR-Model": os.path.basename(SR_MODEL_PATH),
        "X-SR-Magnitude": str(magnitude),
        "X-SR-Passes": str(magnitude),
        "X-SR-Scale-Per-Pass": (
            str(SR_OUTPUT_SIZE // SR_INPUT_SIZE)
            if SR_OUTPUT_SIZE % SR_INPUT_SIZE == 0
            else str(SR_OUTPUT_SIZE / SR_INPUT_SIZE)
        ),
    }

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

    service: PipelineService = app.state.service

    # ---- base SD generation ----
    fut = service.submit(req, timeout_s=0.25)
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

        sr_fut = sr.submit(
            image_bytes=png_bytes,
            out_format=req.superres_format,
            quality=req.superres_quality,
            magnitude=sr_mag,
            timeout_s=0.25,
        )
        out_bytes = sr_fut.result(timeout=SR_REQUEST_TIMEOUT)

        meta_headers.update(
            {
                "X-SuperRes": "1",
                "X-SR-Passes": str(sr_mag),
                "X-SR-Model": os.path.basename(SR_MODEL_PATH),
                "X-SR-Scale-Per-Pass": (
                    str(SR_OUTPUT_SIZE // SR_INPUT_SIZE)
                    if SR_OUTPUT_SIZE % SR_INPUT_SIZE == 0
                    else str(SR_OUTPUT_SIZE / SR_INPUT_SIZE)
                ),
                "X-SR-Format": req.superres_format,
            }
        )

    return out_bytes, seed, meta_headers

@app.get("/storage/health")
def storage_health():
    st: StorageProvider = app.state.storage
    return st.health()

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
app.mount(
    "/",
    StaticFiles(directory="/opt/lcm-sr-server/ui-dist", html=True),
    name="ui",
)

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
