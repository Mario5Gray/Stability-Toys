import io
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol

import numpy as np
from PIL import Image

import logging

try:
    from rknnlite.api import RKNNLite  # type: ignore[import-untyped]
    RKNNLITE_AVAILABLE = True
except ImportError:
    RKNNLITE_AVAILABLE = False
    RKNNLite = None


logger = logging.getLogger(__name__)

SuperResBackend = Literal["rknn", "cuda"]


class SuperResServiceProtocol(Protocol):
    def submit(
        self,
        image_bytes: bytes,
        *,
        out_format: str,
        quality: int,
        magnitude: int,
        timeout_s: float = 0.25,
    ) -> Future: ...

    def shutdown(self) -> None: ...


@dataclass
class SRJob:
    image_bytes: bytes
    out_format: str
    quality: int
    magnitude: int
    fut: Future
    submitted_at: float


def resolve_superres_backend(*, backend: str, use_cuda: bool) -> SuperResBackend:
    backend_norm = (backend or "auto").lower().strip()
    if backend_norm == "cuda":
        return "cuda"
    if backend_norm == "rknn":
        return "rknn"
    if backend_norm == "auto":
        return "cuda" if use_cuda else "rknn"
    raise ValueError(f"Unsupported backend: {backend}")


def create_superres_service(
    *,
    backend_kind: SuperResBackend,
    model_path: str,
    num_workers: int,
    queue_max: int,
    input_size: int,
    output_size: int,
    max_pixels: Optional[int] = None,
    rknn_factory: Optional[Callable[..., SuperResServiceProtocol]] = None,
    cuda_factory: Optional[Callable[..., SuperResServiceProtocol]] = None,
) -> SuperResServiceProtocol:
    kwargs = {
        "model_path": model_path,
        "num_workers": num_workers,
        "queue_max": queue_max,
        "input_size": input_size,
        "output_size": output_size,
    }
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels

    if backend_kind == "rknn":
        factory = rknn_factory or RknnSuperResService
        return factory(**kwargs)

    if backend_kind == "cuda":
        if cuda_factory is None:
            raise RuntimeError("CUDA super-resolution service is not configured")
        return cuda_factory(**kwargs)

    raise RuntimeError(f"Unsupported super-resolution backend: {backend_kind}")


class RknnSuperResWorker:
    def __init__(self, worker_id: int, model_path: str, input_size: int, output_size: int, max_pixels: int):
        if not RKNNLITE_AVAILABLE:
            raise RuntimeError("RKNNLite not available for SR - install rknnlite package")

        self.worker_id = worker_id
        self.model_path = model_path
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.max_pixels = int(max_pixels)

        self.rknn = RKNNLite()  # type: ignore[name-defined]
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

        if img.width * img.height > self.max_pixels:
            raise RuntimeError(
                f"Image too large: {img.width}x{img.height} exceeds SR_MAX_PIXELS={self.max_pixels}"
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
            inp = crop[np.newaxis, np.newaxis, :, :].astype(np.float32)

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
                img_cb.resize((out_w, out_h), Image.Resampling.BICUBIC),
                img_cr.resize((out_w, out_h), Image.Resampling.BICUBIC),
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


class RknnSuperResService:
    def __init__(
        self,
        model_path: str,
        num_workers: int,
        queue_max: int,
        input_size: int,
        output_size: int,
        max_pixels: int,
    ):
        self.model_path = model_path
        self.num_workers = max(1, int(num_workers))
        self.q: "queue.Queue[SRJob]" = queue.Queue(maxsize=int(queue_max))

        self.workers: list[RknnSuperResWorker] = []
        self.threads: list[threading.Thread] = []
        self._stop = threading.Event()

        for i in range(self.num_workers):
            worker = RknnSuperResWorker(
                worker_id=i,
                model_path=self.model_path,
                input_size=input_size,
                output_size=output_size,
                max_pixels=max_pixels,
            )
            self.workers.append(worker)

        for i in range(self.num_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def shutdown(self):
        self._stop.set()
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if not job.fut.done():
                job.fut.set_exception(RuntimeError("RknnSuperResService shutting down"))
            self.q.task_done()

        for worker in self.workers:
            try:
                worker.close()
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
            except Exception as exc:
                logger.error(f"SR Worker {worker_idx} job failed: {exc}", exc_info=True)
                if not job.fut.done():
                    job.fut.set_exception(exc)
            finally:
                self.q.task_done()
