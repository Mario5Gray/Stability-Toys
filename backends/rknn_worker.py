# rknn_worker.py
from .base import PipelineWorker, GenSpec, ModelPaths, Job
from backends.rknnlcm import RKNN2Model, RKNN2LatentConsistencyPipeline
from backends.utils import parse_size, gen_seed_8_digits
from backends.latents import extract_latents, latent_to_nchw, downsample_to_8x8_nchw

from transformers import CLIPTokenizer
from diffusers import LCMScheduler
from PIL import Image

from typing import Optional, List, Dict, Tuple

import io
import numpy as np

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
            latent = extract_latents(res_lat)
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
                latent = extract_latents(res_lat)
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
                latent = extract_latents(res_lat)
            except Exception:
                latent = None

        if latent is None:
            # Keep behavior explicit: caller asked for latents; we must return something deterministic.
            # Use zeros so hashing remains stable and caller can detect "missing" by hashing metadata.
            latent_8 = np.zeros((1, 4, 8, 8), dtype=np.float16)
            return png_bytes, seed, latent_8.tobytes(order="C")

        latent_nchw = latent_to_nchw(latent)
        latent_8 = downsample_to_8x8_nchw(latent_nchw).astype(np.float16, copy=False)
        return png_bytes, seed, latent_8.tobytes(order="C")
