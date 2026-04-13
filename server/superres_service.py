import io
import importlib
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional, Protocol

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
CudaSuperResLifecycle = Literal["sticky", "per_request"]


@dataclass(frozen=True)
class CudaSuperResConfig:
    model_path: str
    tile: int
    use_fp16: bool
    device: str
    lifecycle: CudaSuperResLifecycle


@dataclass(frozen=True)
class CudaSuperResModelSpec:
    scale: int
    num_block: int


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

    def unload(self) -> None: ...

    def shutdown(self) -> None: ...


@dataclass
class SRJob:
    image_bytes: bytes
    out_format: str
    quality: int
    magnitude: int
    fut: Future
    submitted_at: float


def _env_flag(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value not in ("0", "false", "False", "no", "No")


def load_cuda_superres_config(environ: Optional[Mapping[str, str]] = None) -> CudaSuperResConfig:
    env = environ or os.environ
    model_path = (env.get("CUDA_SR_MODEL") or "").strip()
    tile = int(env.get("CUDA_SR_TILE", "0"))
    use_fp16 = _env_flag(env.get("CUDA_SR_FP16"), True)
    device = (env.get("CUDA_DEVICE") or "cuda:0").strip() or "cuda:0"
    lifecycle = ((env.get("CUDA_SR_LIFECYCLE") or "sticky").strip().lower() or "sticky")
    if lifecycle not in ("sticky", "per_request"):
        raise RuntimeError("CUDA_SR_LIFECYCLE must be 'sticky' or 'per_request'")
    return CudaSuperResConfig(
        model_path=model_path,
        tile=tile,
        use_fp16=use_fp16,
        device=device,
        lifecycle=lifecycle,  # type: ignore[arg-type]
    )


def ensure_torchvision_functional_tensor_compat(
    *,
    import_module: Callable[[str], object] = importlib.import_module,
    sys_modules: Optional[dict[str, object]] = None,
) -> None:
    modules = sys_modules if sys_modules is not None else sys.modules
    try:
        import_module("torchvision.transforms.functional_tensor")
        return
    except ModuleNotFoundError:
        pass

    functional = import_module("torchvision.transforms.functional")
    shim = type("TorchvisionFunctionalTensorShim", (), {})()
    shim.rgb_to_grayscale = getattr(functional, "rgb_to_grayscale")
    modules["torchvision.transforms.functional_tensor"] = shim


def normalize_realesrgan_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "params_ema" in checkpoint or "params" in checkpoint:
        return checkpoint

    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return {"params": checkpoint}

    return checkpoint


def describe_cuda_sr_model(model_path: str) -> CudaSuperResModelSpec:
    model_name = os.path.basename(model_path).lower()
    scale = 2 if "x2" in model_name else 4
    num_block = 6 if "anime_6b" in model_name else 23
    return CudaSuperResModelSpec(scale=scale, num_block=num_block)


def resolve_superres_backend(*, backend: str, use_cuda: bool) -> SuperResBackend:
    backend_norm = (backend or "").lower().strip()
    if backend_norm == "cuda":
        return "cuda"
    if backend_norm == "rknn":
        return "rknn"
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
    cuda_config: Optional[CudaSuperResConfig] = None,
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
        factory = cuda_factory or CudaSuperResService
        if cuda_config is not None:
            kwargs["config"] = cuda_config
        return factory(**kwargs)

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
        self.scale_per_pass = (
            output_size // input_size if output_size % input_size == 0 else output_size / input_size
        )
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

    def unload(self) -> None:
        return None

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


class CudaSuperResWorker:
    def __init__(self, config: CudaSuperResConfig):
        if not config.model_path:
            raise RuntimeError("CUDA_SR_MODEL must be set for CUDA super-resolution")

        self.config = config
        self._torch = self._import_torch()
        self._upsampler = self._build_upsampler()

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for CUDA super-resolution") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for CUDA super-resolution")
        return torch

    def _build_upsampler(self):
        try:
            ensure_torchvision_functional_tensor_compat()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise RuntimeError(
                "CUDA super-resolution dependencies missing. Install realesrgan and basicsr."
            ) from exc

        model_spec = describe_cuda_sr_model(self.config.model_path)
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=model_spec.num_block,
            num_grow_ch=32,
            scale=model_spec.scale,
        )
        torch = self._torch
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            checkpoint = original_torch_load(*args, **kwargs)
            return normalize_realesrgan_checkpoint(checkpoint)

        torch.load = patched_torch_load
        try:
            return RealESRGANer(
                scale=model_spec.scale,
                model_path=self.config.model_path,
                model=net,
                tile=max(0, int(self.config.tile)),
                tile_pad=10,
                pre_pad=0,
                half=bool(self.config.use_fp16),
                device=self.config.device,
            )
        finally:
            torch.load = original_torch_load

    def close(self):
        torch = self._torch
        gc = __import__("gc")
        del self._upsampler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def upscale_bytes(self, image_bytes: bytes, *, magnitude: int, out_format: str, quality: int) -> bytes:
        mag = int(magnitude)
        if mag < 1 or mag > 3:
            raise RuntimeError("magnitude must be 1..3")

        out = image_bytes
        for _ in range(mag):
            img = Image.open(io.BytesIO(out)).convert("RGB")
            rgb = np.array(img, dtype=np.uint8)
            bgr = rgb[:, :, ::-1]
            pred_bgr, _ = self._upsampler.enhance(bgr, outscale=4)
            pred_rgb = pred_bgr[:, :, ::-1]
            out_img = Image.fromarray(pred_rgb.astype(np.uint8), mode="RGB")

            buf = io.BytesIO()
            if out_format == "jpeg":
                out_img.save(buf, format="JPEG", quality=int(quality))
            else:
                out_img.save(buf, format="PNG")
            out = buf.getvalue()
        return out


class CudaSuperResService:
    def __init__(
        self,
        model_path: str = "",
        num_workers: int = 1,
        queue_max: int = 32,
        input_size: int = 0,
        output_size: int = 0,
        max_pixels: Optional[int] = None,
        *,
        config: Optional[CudaSuperResConfig] = None,
        worker_factory: Optional[Callable[[CudaSuperResConfig], object]] = None,
    ):
        del num_workers, input_size, output_size, max_pixels
        self.config = config or CudaSuperResConfig(
            model_path=model_path,
            tile=0,
            use_fp16=True,
            device="cuda:0",
            lifecycle="sticky",
        )
        self.model_path = self.config.model_path
        self.scale_per_pass = 4
        self.worker_factory = worker_factory or CudaSuperResWorker
        self.q: "queue.Queue[SRJob]" = queue.Queue(maxsize=int(queue_max))
        self._stop = threading.Event()
        self._worker_lock = threading.Lock()
        self._worker: Optional[object] = None
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _ensure_worker(self):
        with self._worker_lock:
            if self._worker is None:
                self._worker = self.worker_factory(self.config)
            return self._worker

    def _unload_worker(self):
        with self._worker_lock:
            worker = self._worker
            self._worker = None
        if worker is not None:
            close = getattr(worker, "close", None)
            if callable(close):
                close()

    def shutdown(self):
        self._stop.set()
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if not job.fut.done():
                job.fut.set_exception(RuntimeError("CudaSuperResService shutting down"))
            self.q.task_done()
        self._unload_worker()

    def unload(self) -> None:
        self._unload_worker()

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

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if job.fut.cancelled():
                self.q.task_done()
                continue

            try:
                worker = self._ensure_worker()
                out_bytes = worker.upscale_bytes(
                    job.image_bytes,
                    magnitude=job.magnitude,
                    out_format=job.out_format,
                    quality=job.quality,
                )
                if not job.fut.done():
                    job.fut.set_result(out_bytes)
            except Exception as exc:
                if _is_cuda_oom(exc):
                    logger.error("CUDA SR worker hit OOM; unloading worker", exc_info=True)
                    self._unload_worker()
                else:
                    logger.error(f"CUDA SR worker job failed: {exc}", exc_info=True)
                if not job.fut.done():
                    job.fut.set_exception(exc)
            finally:
                if self.config.lifecycle == "per_request":
                    self._unload_worker()
                self.q.task_done()


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg and "cuda" in msg:
        return True

    try:
        import torch
    except Exception:
        return False

    return isinstance(exc, torch.cuda.OutOfMemoryError)
