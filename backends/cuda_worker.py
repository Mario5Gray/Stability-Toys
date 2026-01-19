# cuda_worker.py
import os
import io
import torch
from dataclasses import dataclass
from typing import Optional

from diffusers import StableDiffusionPipeline, LCMScheduler
from PIL import Image

from .base import PipelineWorker, GenSpec



def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


class DiffusersCudaWorker(PipelineWorker):
    """
    CUDA Diffusers worker for SD1.5 LCM checkpoints (.safetensors).

    Env:
      CUDA_CKPT_PATH=/path/to/DreamShaper-V88-LCM.safetensors
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

        ckpt_path = os.environ.get("CUDA_CKPT_PATH", "").strip()
        if not ckpt_path:
            raise RuntimeError("CUDA_CKPT_PATH is required for BACKEND=cuda")

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

        # ---- Load from single-file checkpoint (.safetensors) ----
        # This uses Diffusers' "from_single_file" path (recommended for ckpt/safetensors).
        # It will download/config the SD1.5 pipeline components as needed.
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,
            torch_dtype=dtype,
            safety_checker=None,          # local service: remove safety checker overhead
            requires_safety_checker=False,
        )

        # ---- Force LCM scheduler ----
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        # ---- Move to CUDA ----
        pipe = pipe.to(device)

        # ---- Optional performance toggles ----
        # NOTE: LCM is already fast; these just reduce memory / sometimes boost speed.
        pipe.enable_vae_tiling()  # helps large outputs without huge VRAM spikes

        if attention_slicing:
            pipe.enable_attention_slicing()

        if enable_xformers:
            # Only works if xformers is installed in the container
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[cuda] worker {worker_id}: xformers enable failed: {e}")

        # You can also optionally do:
        # pipe.unet.to(memory_format=torch.channels_last)

        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        print(f"[cuda] worker {worker_id} loaded: {os.path.basename(ckpt_path)} on {device} dtype={dtype_str}")

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

        # Parse WxH
        try:
            w_str, h_str = str(req.size).lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            raise RuntimeError(f"Invalid size '{req.size}', expected 'WIDTHxHEIGHT'")

        # Choose seed
        seed = int(req.seed) if req.seed is not None else int(torch.randint(0, 100_000_000, (1,)).item())

        # Torch generator for determinism
        # (LCM is fast, but still benefits from explicit generator)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        # Run pipeline
        # LCM typically likes low cfg (~1-2) and steps (~5-15)
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