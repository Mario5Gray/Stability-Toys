"""
lcn_server.py — RKNN LCM Stable Diffusion FastAPI server (queued, multi-worker safe)

Key goals:
- One pipeline per worker thread (no shared RKNN objects across threads)
- Determin guarantee: per-request seed -> np.RandomState
- Deterministic input ordering handled in RKNN2Model (recommended)
- Explicit data_format per model (UNet + VAE commonly NHWC on RKNN)
- Queue backpressure (429 on overflow)
- Clean startup/shutdown (FastAPI lifespan)
- Returns PNG bytes + X-Seed header

Env:
  MODEL_ROOT=/models/lcm_rknn
  PORT=4200
  NUM_WORKERS=1..3
  QUEUE_MAX=64
  DEFAULT_SIZE=512x512
  DEFAULT_STEPS=4
  DEFAULT_GUIDANCE=1.0
  DEFAULT_TIMEOUT=120
"""

import io
import os
import json
import time
import queue
import threading
from dataclasses import dataclass
from concurrent.futures import Future
from typing import Optional, List, Dict, Tuple
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel, Field

from diffusers import LCMScheduler
from transformers import CLIPTokenizer

from rknnlcm import RKNN2Model, RKNN2LatentConsistencyPipeline


# -----------------------------
# Request schema (HTTP)
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    size: str = Field(default=os.environ.get("DEFAULT_SIZE", "512x512"), pattern=r"^\d+x\d+$")
    num_inference_steps: int = Field(default=int(os.environ.get("DEFAULT_STEPS", "4")), ge=1, le=50)
    guidance_scale: float = Field(default=float(os.environ.get("DEFAULT_GUIDANCE", "1.0")), ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


@dataclass(frozen=True)
class ModelPaths:
    root: str

    @property
    def scheduler_config(self) -> str:
        return os.path.join(self.root, "scheduler", "scheduler_config.json")

    @property
    def text_encoder(self) -> str:
        return os.path.join(self.root, "text_encoder")

    @property
    def unet(self) -> str:
        return os.path.join(self.root, "unet")

    @property
    def vae_decoder(self) -> str:
        return os.path.join(self.root, "vae_decoder")


@dataclass
class Job:
    req: GenerateRequest
    fut: Future
    submitted_at: float


# -----------------------------
# RKNN multi-context configuration
# -----------------------------
def build_rknn_context_cfgs_for_rk3588(num_workers: int) -> List[dict]:
    """
    You must map these fields inside RKNN2Model if you actually support them.
    If your RKNN2Model does NOT accept these kwargs, set USE_RKNN_CONTEXT_CFGS=0.
    """
    core_masks = ["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"]
    cfgs = []
    for i in range(num_workers):
        cfgs.append(
            {
                "multi_context": True,
                # binding per-core is optional; if unstable, keep AUTO
                "core_mask": core_masks[i % len(core_masks)],
                # "core_mask": "NPU_CORE_AUTO",
                "context_name": f"w{i}",
                "worker_id": i,
            }
        )
    return cfgs


def parse_size(size_str: str) -> Tuple[int, int]:
    """
    Parse 'WIDTHxHEIGHT' -> (width, height)
    """
    w_str, h_str = size_str.lower().split("x")
    w, h = int(w_str), int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")
    return w, h


def gen_seed_8_digits() -> int:
    # 0..99,999,999 inclusive
    return int(np.random.randint(0, 100_000_000))


# -----------------------------
# Pipeline Worker
# -----------------------------
class PipelineWorker:
    """
    Owns ONE pipeline instance. Execute jobs sequentially on this worker.
    """

    def __init__(
        self,
        worker_id: int,
        paths: ModelPaths,
        scheduler_config: Dict,
        tokenizer: CLIPTokenizer,
        rknn_context_cfg: Optional[dict] = None,
        use_rknn_context_cfgs: bool = True,
    ):
        self.worker_id = worker_id
        self.paths = paths
        self.scheduler_config = scheduler_config
        self.tokenizer = tokenizer
        self.rknn_context_cfg = rknn_context_cfg or {}
        self.use_rknn_context_cfgs = use_rknn_context_cfgs

        self.pipe = None
        self._init_pipeline()

    def _mk_model(self, model_path: str, *, data_format: str) -> RKNN2Model:
        """
        Create one RKNN2Model with explicit data_format.
        If your RKNN2Model supports multi_context/core_mask/etc, it will receive them.
        """
        if self.use_rknn_context_cfgs:
            return RKNN2Model(model_path, data_format=data_format, **self.rknn_context_cfg)
        return RKNN2Model(model_path, data_format=data_format)

    def _init_pipeline(self):
        # IMPORTANT: per-worker scheduler instance (avoid shared mutable state)
        scheduler = LCMScheduler.from_config(self.scheduler_config)

        # Per-model explicit formats:
        # - text encoder is token/embedding, format mostly irrelevant; keep nchw
        # - unet + vae_decoder commonly require nhwc on RKNN
        self.pipe = RKNN2LatentConsistencyPipeline(
            text_encoder=self._mk_model(self.paths.text_encoder, data_format="nchw"),
            unet=self._mk_model(self.paths.unet, data_format="nhwc"),
            vae_decoder=self._mk_model(self.paths.vae_decoder, data_format="nhwc"),
            scheduler=scheduler,
            tokenizer=self.tokenizer,
        )

    def run_job(self, job: Job) -> Tuple[bytes, int]:
        # Parse WIDTHxHEIGHT
        width, height = parse_size(job.req.size)

        # Deterministic per-request RNG
        seed = job.req.seed if job.req.seed is not None else gen_seed_8_digits()
        rng = np.random.RandomState(seed)

        result = self.pipe(
            prompt=job.req.prompt,
            height=height,
            width=width,
            num_inference_steps=job.req.num_inference_steps,
            guidance_scale=job.req.guidance_scale,
            generator=rng,
        )

        pil_image = result["images"][0]
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue(), seed


# -----------------------------
# Singleton Service
# -----------------------------
class PipelineService:
    """
    Singleton-ish service that:
      - loads scheduler_config + tokenizer once
      - starts N worker threads
      - queues requests and runs them on worker-owned pipelines
    """

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
        self.paths = paths
        self.num_workers = max(1, int(num_workers))
        self.q: "queue.Queue[Job]" = queue.Queue(maxsize=int(queue_max))

        # Load scheduler config once (immutable dict)
        with open(self.paths.scheduler_config, "r") as f:
            self.scheduler_config = json.load(f)

        # Tokenizer is safe to share (read-only)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

        # Worker RKNN configs
        if rknn_context_cfgs is None:
            rknn_context_cfgs = build_rknn_context_cfgs_for_rk3588(self.num_workers)
        if len(rknn_context_cfgs) != self.num_workers:
            raise ValueError("rknn_context_cfgs must match num_workers length")

        self.workers: List[PipelineWorker] = []
        self.threads: List[threading.Thread] = []
        self._stop = threading.Event()

        # Create worker pipelines
        for i in range(self.num_workers):
            w = PipelineWorker(
                worker_id=i,
                paths=self.paths,
                scheduler_config=self.scheduler_config,
                tokenizer=self.tokenizer,
                rknn_context_cfg=rknn_context_cfgs[i],
                use_rknn_context_cfgs=use_rknn_context_cfgs,
            )
            self.workers.append(w)

        # Start worker threads
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
        # Drain queue with errors
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
                if not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                self.q.task_done()


# -----------------------------
# FastAPI server
# -----------------------------
MODEL_ROOT = os.environ.get("MODEL_ROOT", "/models/lcm_rknn")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "1"))
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "64"))
PORT = int(os.environ.get("PORT", "4200"))
REQUEST_TIMEOUT = float(os.environ.get("DEFAULT_TIMEOUT", "120"))

# If your RKNN2Model does NOT accept multi_context/core_mask kwargs, set this to 0.
USE_RKNN_CONTEXT_CFGS = os.environ.get("USE_RKNN_CONTEXT_CFGS", "1") not in ("0", "false", "False")

paths = ModelPaths(root=MODEL_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create singleton service at startup
    app.state.service = PipelineService.get_instance(
        paths=paths,
        num_workers=NUM_WORKERS,
        queue_max=QUEUE_MAX,
        rknn_context_cfgs=build_rknn_context_cfgs_for_rk3588(NUM_WORKERS),
        use_rknn_context_cfgs=USE_RKNN_CONTEXT_CFGS,
    )
    yield
    # Shutdown on app stop
    app.state.service.shutdown()


app = FastAPI(lifespan=lifespan)


@app.post("/generate", responses={200: {"content": {"image/png": {}}}})
def generate(req: GenerateRequest):
    service: PipelineService = app.state.service

    fut = service.submit(req, timeout_s=0.25)
    try:
        png_bytes, seed = fut.result(timeout=REQUEST_TIMEOUT)
    except Exception as e:
        msg = str(e)
        if "Queue full" in msg:
            raise HTTPException(status_code=429, detail="Too many requests (queue full). Try again.")
        raise HTTPException(status_code=500, detail=f"Generation failed: {msg}")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Seed": str(seed),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=None,  # avoids logger dictConfig surprises
    )