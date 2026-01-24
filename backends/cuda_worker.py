# cuda_worker.py
import os
import io
import torch
import numpy as np
from typing import Optional, Tuple

from diffusers import StableDiffusionPipeline, LCMScheduler
from PIL import Image

from .base import PipelineWorker, GenSpec


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


class DiffusersCudaWorker(PipelineWorker):
    """
    CUDA Diffusers worker for SD1.5 LCM models.

    Supports two formats:
      - Single file: .safetensors or .ckpt checkpoint
      - Diffusers layout: directory with model_index.json

    Env:
      MODEL_ROOT=/path/to/model.safetensors  (or /path/to/model_dir/)
      CUDA_DTYPE=fp16|bf16|fp32   (default fp16)
      CUDA_DEVICE=cuda:0         (default cuda:0)
      CUDA_ENABLE_XFORMERS=1     (default 0)
      CUDA_ATTENTION_SLICING=0/1 (default 0)
    """

    def __init__(self, worker_id: int):
        super().__init__(worker_id)
        import torch
        from diffusers import StableDiffusionPipeline, LCMScheduler

        self.worker_id = worker_id
        
        ckpt_path = os.path.join(os.environ.get('MODEL_ROOT'), os.environ.get('MODEL'))
        
        if not ckpt_path:
            raise RuntimeError("MODEL_ROOT is required for BACKEND=cuda")

        device = os.environ.get("CUDA_DEVICE", "cuda:0").strip()

        dtype_str = os.environ.get("CUDA_DTYPE", "fp16").lower().strip()
        if dtype_str == "bf16":
            dtype = torch.bfloat16
        elif dtype_str == "fp32":
            dtype = torch.float32
        else:
            dtype = torch.float16

        enable_xformers = _bool_env("CUDA_ENABLE_XFORMERS", "0")
        attention_slicing = _bool_env("CUDA_ATTENTION_SLICING", "0")
      
        is_diffusers_dir = os.path.isdir(ckpt_path) and os.path.exists(
            os.path.join(ckpt_path, "model_index.json")
        )

        if is_diffusers_dir:
            # Diffusers standard layout (directory with model_index.json)
            pipe = StableDiffusionPipeline.from_pretrained(
                ckpt_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "diffusers"
        else:
            # Single file (.safetensors or .ckpt)
            pipe = StableDiffusionPipeline.from_single_file(
                ckpt_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "safetensors"

        # LCM scheduler (applies to both formats)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

        pipe.enable_vae_tiling()

        if attention_slicing:
            pipe.enable_attention_slicing()

        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[cuda] worker {worker_id}: xformers enable failed: {e}")

        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        print(f"[cuda] worker {worker_id} loaded: {os.path.basename(ckpt_path)} ({format_name}) on {device} dtype={dtype_str}")

    def run_job(self, job) -> tuple[bytes, int]:
        """
        Run one generation job on this CUDA worker.

        Expects:
          job.req.prompt (str)
          job.req.size ("512x512")
          job.req.num_inference_steps (int)
          job.req.guidance_scale (float)
          job.req.seed (Optional[int])

        Returns:
          (png_bytes, seed_used)
        """
        req = job.req

        try:
            w_str, h_str = str(req.size).lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            raise RuntimeError(f"Invalid size '{req.size}', expected 'WIDTHxHEIGHT'")

        seed = int(req.seed) if req.seed is not None else int(torch.randint(0, 100_000_000, (1,)).item())

        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        with torch.inference_mode():
            out = self.pipe(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=int(req.num_inference_steps),
                guidance_scale=float(req.guidance_scale),
                generator=gen,
            )

        img: Image.Image = out.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), seed

    def run_job_with_latents(self, job) -> Tuple[bytes, int, bytes]:
        """
        Returns:
          (png_bytes, seed_used, latents_bytes)

        latents_bytes:
          - raw tensor bytes for NCHW float16 with shape [1,4,8,8]
          - intended for hashing / similarity bookkeeping

        Implementation:
          - preserves existing image-generation logic by calling run_job()
          - runs a second pass ONLY to obtain latents (output_type="latent")
        """
        req = job.req

        png_bytes, seed = self.run_job(job)

        try:
            w_str, h_str = str(req.size).lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            raise RuntimeError(f"Invalid size '{req.size}', expected 'WIDTHxHEIGHT'")

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))

        with torch.inference_mode():
            out_lat = self.pipe(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=int(req.num_inference_steps),
                guidance_scale=float(req.guidance_scale),
                generator=gen,
                output_type="latent",
                return_dict=True,
            )

        # Diffusers returns latents in out_lat.images when output_type="latent"
        lat = out_lat.images
        if isinstance(lat, (list, tuple)):
            lat = lat[0]

        # Ensure torch tensor
        if not torch.is_tensor(lat):
            lat = torch.as_tensor(lat)

        # lat expected: [1,4,H/8,W/8] (NCHW)
        if lat.ndim != 4:
            raise RuntimeError(f"Unexpected latent rank {lat.ndim}, shape={tuple(lat.shape)}")

        # Downsample to [1,4,8,8]
        lat = lat.to(dtype=torch.float32)
        lat_8 = torch.nn.functional.adaptive_avg_pool2d(lat, (8, 8))

        lat_8 = lat_8.to(dtype=torch.float16).contiguous()
        lat_np = lat_8.detach().cpu().numpy().astype(np.float16, copy=False)

        return png_bytes, seed, lat_np.tobytes(order="C")