# rknn_worker.py
from .base import PipelineWorker, GenSpec, ModelPaths, Job
from rknnlcm import RKNN2Model, RKNN2LatentConsistencyPipeline

from transformers import CLIPTokenizer
from diffusers import LCMScheduler
from PIL import Image

from typing import Optional, List, Dict, Tuple

import io
import numpy as np
from typing import Tuple

def parse_size(size_str: str) -> Tuple[int, int]:
    w_str, h_str = size_str.lower().split("x")
    w, h = int(w_str), int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")
    return w, h

def gen_seed_8_digits() -> int:
    return int(np.random.randint(0, 100_000_000))

# -----------------------------
# Pipeline Worker
# -----------------------------
class RKNNPipelineWorker(PipelineWorker):
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
        super().__init__(worker_id)
        self.worker_id = worker_id
        self.paths = paths
        self.scheduler_config = scheduler_config
        self.tokenizer = tokenizer
        self.rknn_context_cfg = rknn_context_cfg or {}
        self.use_rknn_context_cfgs = use_rknn_context_cfgs

        self.pipe = None
        self._init_pipeline()

    def _mk_model(self, model_path: str, *, data_format: str) -> RKNN2Model:
        if self.use_rknn_context_cfgs:
            return RKNN2Model(model_path, data_format=data_format, **self.rknn_context_cfg)
        return RKNN2Model(model_path, data_format=data_format)

    def _init_pipeline(self):
        scheduler = LCMScheduler.from_config(self.scheduler_config)
        self.pipe = RKNN2LatentConsistencyPipeline(
            text_encoder=self._mk_model(self.paths.text_encoder, data_format="nchw"),
            unet=self._mk_model(self.paths.unet, data_format="nhwc"),
            vae_decoder=self._mk_model(self.paths.vae_decoder, data_format="nhwc"),
            scheduler=scheduler,
            tokenizer=self.tokenizer,
        )

    def run_job(self, job: Job) -> Tuple[bytes, int]:
        width, height = parse_size(job.req.size)
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

    def run_job_with_latents(self, job: Job) -> Tuple[bytes, int, bytes]:
        """
        Returns:
          (png_bytes, seed_used, latents_bytes)

        latents_bytes:
          - raw tensor bytes for NCHW float16 with shape [1,4,8,8]
          - intended for hashing / similarity bookkeeping
        """
        png_bytes, seed = self.run_job(job)

        width, height = parse_size(job.req.size)
        rng = np.random.RandomState(seed)

        # Best-effort: try to obtain latents from the pipeline without changing run_job logic.
        latent = None

        # Attempt A: output_type="latent"
        try:
            res_lat = self.pipe(
                prompt=job.req.prompt,
                height=height,
                width=width,
                num_inference_steps=job.req.num_inference_steps,
                guidance_scale=job.req.guidance_scale,
                generator=rng,
                output_type="latent",
            )
            latent = _extract_latents(res_lat)
        except TypeError:
            latent = None

        # Attempt B: return_latents=True
        if latent is None:
            try:
                res_lat = self.pipe(
                    prompt=job.req.prompt,
                    height=height,
                    width=width,
                    num_inference_steps=job.req.num_inference_steps,
                    guidance_scale=job.req.guidance_scale,
                    generator=rng,
                    return_latents=True,
                )
                latent = _extract_latents(res_lat)
            except TypeError:
                latent = None

        # Attempt C: common key names in result dict
        if latent is None:
            try:
                res_lat = self.pipe(
                    prompt=job.req.prompt,
                    height=height,
                    width=width,
                    num_inference_steps=job.req.num_inference_steps,
                    guidance_scale=job.req.guidance_scale,
                    generator=rng,
                )
                latent = _extract_latents(res_lat)
            except Exception:
                latent = None

        if latent is None:
            # Keep behavior explicit: caller asked for latents; we must return something deterministic.
            # Use zeros so hashing remains stable and caller can detect "missing" by hashing metadata.
            latent_8 = np.zeros((1, 4, 8, 8), dtype=np.float16)
            return png_bytes, seed, latent_8.tobytes(order="C")

        latent_nchw = _latent_to_nchw(latent)
        latent_8 = _downsample_to_8x8_nchw(latent_nchw).astype(np.float16, copy=False)
        return png_bytes, seed, latent_8.tobytes(order="C")


def _extract_latents(res):
    """
    Best-effort extraction of latents from pipeline return object.
    Accepts dict-like objects or diffusers-like outputs.
    """
    if res is None:
        return None
    if isinstance(res, dict):
        for k in ("latents", "latent", "images"):
            if k in res and res[k] is not None:
                return res[k]
        return None
    # object with attributes
    for k in ("latents", "latent", "images"):
        if hasattr(res, k):
            v = getattr(res, k)
            if v is not None:
                return v
    return None


def _latent_to_nchw(x) -> np.ndarray:
    """
    Convert latent tensor to numpy NCHW.
    Supports:
      - numpy arrays
      - lists
      - objects with .numpy()
      - torch tensors (via .detach().cpu().numpy() if available)
    """
    if x is None:
        raise ValueError("latent is None")

    # Torch tensor?
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.ndim != 4:
        raise ValueError(f"latent must be 4D, got shape={x.shape}")

    # NCHW
    if x.shape[1] == 4:
        return x

    # NHWC -> NCHW
    if x.shape[-1] == 4:
        return np.transpose(x, (0, 3, 1, 2))

    # Unknown layout, best effort: if one dim equals 4, move it to C
    if 4 in x.shape:
        c_axis = list(x.shape).index(4)
        if c_axis != 1:
            axes = list(range(4))
            axes.pop(c_axis)
            axes.insert(1, c_axis)
            return np.transpose(x, axes)

    raise ValueError(f"cannot interpret latent layout, shape={x.shape}")


def _downsample_to_8x8_nchw(lat: np.ndarray) -> np.ndarray:
    """
    Downsample NCHW latent to [1,4,8,8] using block-mean if divisible.
    Falls back to nearest sampling if not divisible.
    """
    lat = np.asarray(lat)
    if lat.shape[0] != 1:
        lat = lat[:1]
    if lat.shape[1] != 4:
        raise ValueError(f"expected C=4, got shape={lat.shape}")

    _, _, h, w = lat.shape
    if h == 8 and w == 8:
        return lat

    # Block-average if divisible by 8
    if (h % 8 == 0) and (w % 8 == 0):
        bh = h // 8
        bw = w // 8
        # (1,4,8,bh,8,bw) -> mean over bh,bw
        return lat.reshape(1, 4, 8, bh, 8, bw).mean(axis=(3, 5))

    # Nearest sampling fallback
    ys = (np.linspace(0, h - 1, 8)).round().astype(np.int64)
    xs = (np.linspace(0, w - 1, 8)).round().astype(np.int64)
    return lat[:, :, ys][:, :, :, xs]