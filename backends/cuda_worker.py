# backends/cuda_worker.py
from __future__ import annotations

import io
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

from .base import PipelineWorker
from backends.styles import STYLE_REGISTRY, parse_style_request


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")

class DiffusersCudaWorker(PipelineWorker):
    """
    CUDA Diffusers worker for SD1.5 LCM models.

    Supports two formats:
      - Single file: .safetensors or .ckpt checkpoint
      - Diffusers layout: directory with model_index.json

    Env:
      MODEL_ROOT=/basepath/to/
      MODEL=model.safetensors  (or MODEL=model_dir/)
      CUDA_DTYPE=fp16|bf16|fp32   (default fp16)
      CUDA_DEVICE=cuda:0         (default cuda:0)
      CUDA_ENABLE_XFORMERS=1     (default 0)
      CUDA_ATTENTION_SLICING=0/1 (default 0)
    """
    def __init__(self, worker_id: int):
        super().__init__(worker_id)

        self.worker_id = worker_id
        self._style_loaded: dict[str, bool] = {}  # adapter_name -> bool
        self._style_api: str = "unknown"  # "adapters" | "fuse" | "none"

        model_root = (os.environ.get("MODEL_ROOT") or "").strip()
        model_name = (os.environ.get("MODEL") or "").strip()
        if not model_root:
            raise RuntimeError("MODEL_ROOT is required for BACKEND=cuda")
        if not model_name:
            raise RuntimeError("MODEL is required for BACKEND=cuda")

        ckpt_path = os.path.join(model_root, model_name)
        print(f"[cuda] ckpt_path={ckpt_path}")

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
            print("loading diffusers")
            pipe = StableDiffusionPipeline.from_pretrained(
                ckpt_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "diffusers"
        else:
            print("loading safetensors")
            pipe = StableDiffusionPipeline.from_single_file(
                ckpt_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "single-file"

        # LCM scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

        pipe.vae.enable_tiling()

        if attention_slicing:
            pipe.enable_attention_slicing()

        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[cuda] worker {worker_id}: xformers enable failed: {e!r}")

        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        te_dim = getattr(getattr(self.pipe, "text_encoder", None), "config", None)
        te_dim = getattr(te_dim, "hidden_size", None)

        # SD1.5 styles: required_cross_attention_dim=768
        # SDXL styles: required_cross_attention_dim=2048,
        # Determine model compatibility info once
        cad = getattr(self.pipe.unet.config, "cross_attention_dim", None)
        if cad == 2048 and type(self.pipe).__name__ == "StableDiffusionPipeline":
            raise RuntimeError("Loaded SDXL UNet (cross_attention_dim=2048) but using StableDiffusionPipeline. Use StableDiffusionXLPipeline.")        
    
        cad = getattr(getattr(self.pipe, "unet", None), "config", None)
        cad = getattr(cad, "cross_attention_dim", None)


        for sid, sd in STYLE_REGISTRY.items():
            # --- Compatibility gate: cross-attention dim ---
            if sd.required_cross_attention_dim is not None and cad is not None:
                if int(cad) != int(sd.required_cross_attention_dim):
                    print(
                        f"[cuda] skip style '{sid}': incompatible cross_attention_dim "
                        f"(model={cad} style={sd.required_cross_attention_dim})"
                    )
                    self._style_loaded[sd.adapter_name] = False
                    continue
            try:
                try:
                    # Newer diffusers supports adapter_name
                    self.pipe.load_lora_weights(sd.lora_path, adapter_name=sd.adapter_name)
                    self._style_loaded[sd.adapter_name] = True
                    print(f"[cuda] loaded style LoRA: {sid} -> {sd.lora_path} (adapter={sd.adapter_name})")
                except TypeError:
                    # Older diffusers: no adapter_name kwarg
                    self.pipe.load_lora_weights(sd.lora_path)
                    self._style_loaded[sd.adapter_name] = True
                    print(f"[cuda] loaded style LoRA (no adapter_name API): {sid} -> {sd.lora_path}")
            except Exception as e:
                self._style_loaded[sd.adapter_name] = False
                print(f"[cuda] FAILED to load style LoRA {sid}: {e!r}")

        # Detect best available runtime API for toggling
        if hasattr(self.pipe, "set_adapters") and hasattr(self.pipe, "disable_lora"):
            self._style_api = "adapters"
        elif hasattr(self.pipe, "fuse_lora"):
            self._style_api = "fuse"
        else:
            self._style_api = "none"

        print(
            f"[cuda] worker {worker_id} loaded: {os.path.basename(ckpt_path)} "
            f"({format_name}) on {device} dtype={dtype_str} style_api={self._style_api}"
        )

    # ---------------------------
    # Style application (exclusive)
    # ---------------------------
    def _apply_style(self, style_id: str | None, level: int) -> None:
        # turn off
        if not style_id or int(level) <= 0:
            if hasattr(self.pipe, "disable_lora"):
                self.pipe.disable_lora()
            elif hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters([])
            return

        sd = STYLE_REGISTRY.get(style_id)
        if not sd:
            if hasattr(self.pipe, "disable_lora"):
                self.pipe.disable_lora()
            return

        # clamp level 1..N
        lvl = max(1, min(int(level), len(sd.levels)))
        weight = float(sd.levels[lvl - 1])

        if not self._style_loaded.get(sd.adapter_name, False):
            return

        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([sd.adapter_name], adapter_weights=[weight])
        elif hasattr(self.pipe, "fuse_lora"):
            # fallback: not ideal if concurrent, but your CUDA path is 1 worker
            if hasattr(self.pipe, "unfuse_lora"):
                try:
                    self.pipe.unfuse_lora()
                except Exception:
                    pass
            self.pipe.fuse_lora(lora_scale=weight)

    # ---------------------------
    # Job execution
    # ---------------------------
    def run_job(self, job) -> tuple[bytes, int]:
        req = job.req

        try:
            w_str, h_str = str(req.size).lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            raise RuntimeError(f"Invalid size '{req.size}', expected 'WIDTHxHEIGHT'")

        seed = int(req.seed) if req.seed is not None else int(torch.randint(0, 100_000_000, (1,)).item())

        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        sl = getattr(req, "style_lora", None)
        style_id = getattr(sl, "style", None) if sl else None
        level = int(getattr(sl, "level", 0)) if sl else 0

        self._apply_style(style_id, level)

        with torch.inference_mode():
            out = self.pipe(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=int(req.num_inference_steps),
                guidance_scale=float(req.guidance_scale),
                generator=gen,
            )

        # reset style to avoid state bleed
        self._apply_style(None, 0)

        img: Image.Image = out.images[0]

        # PERF NOTE: BytesIO reuse would be micro-optimization; not worth complexity here.
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

        # Apply the same style for latent pass so latents match the image pass.
        sl = getattr(req, "style_lora", None)
        style_id = getattr(sl, "style", None) if sl else None
        level = int(getattr(sl, "level", 0)) if sl else 0

        self._apply_style(style_id, level)

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

        self._apply_style(None, 0)

        lat = out_lat.images
        if isinstance(lat, (list, tuple)):
            lat = lat[0]

        if not torch.is_tensor(lat):
            lat = torch.as_tensor(lat)

        if lat.ndim != 4:
            raise RuntimeError(f"Unexpected latent rank {lat.ndim}, shape={tuple(lat.shape)}")

        # Downsample to [1,4,8,8]
        # PERF NOTE: do pooling in fp32 to reduce pooling artifacts; then cast to fp16.
        lat_8 = torch.nn.functional.adaptive_avg_pool2d(lat.to(dtype=torch.float32), (8, 8))
        lat_8 = lat_8.to(dtype=torch.float16).contiguous()

        # PERF NOTE: astype(copy=False) avoids extra copy when already float16.
        lat_np = lat_8.detach().cpu().numpy().astype(np.float16, copy=False)
        return png_bytes, seed, lat_np.tobytes(order="C")


class DiffusersSDXLCudaWorker(PipelineWorker):
    """
    CUDA Diffusers worker for SDXL (Stable Diffusion XL) models.

    Supports:
      - Single file: .safetensors or .ckpt checkpoint
      - Diffusers layout: directory with model_index.json
      - LCM-SDXL and regular SDXL with LCM scheduler
      - SDXL-compatible LoRAs (cross_attention_dim=2048)

    Env:
      MODEL_ROOT=/basepath/to/models/
      MODEL=model.safetensors               (or model_dir/)
      CUDA_DTYPE=fp16|bf16|fp32             (default fp16)
      CUDA_DEVICE=cuda:0                    (default cuda:0)
      CUDA_ENABLE_XFORMERS=1                (default 0)
      CUDA_ATTENTION_SLICING=0/1            (default 0)

    Notes:
      - SDXL has dual text encoders (CLIP-L and OpenCLIP-G)
      - Default resolution: 1024x1024 (vs 512x512 for SD1.5)
      - Latent space: 128x128 (vs 64x64 for SD1.5)
      - Cross-attention dim: 2048 (vs 768 for SD1.5)
    """

    def __init__(self, worker_id: int):
        super().__init__(worker_id)

        self.worker_id = worker_id
        self._style_loaded: dict[str, bool] = {}  # adapter_name -> bool
        self._style_api: str = "unknown"  # "adapters" | "fuse" | "none"

        # Use same MODEL_ROOT and MODEL as SD1.5 worker
        model_root = (os.environ.get("MODEL_ROOT") or "").strip()
        model_name = (os.environ.get("MODEL") or "").strip()

        if not model_root:
            raise RuntimeError("MODEL_ROOT is required for SDXL CUDA worker")
        if not model_name:
            raise RuntimeError("MODEL is required for SDXL CUDA worker")

        ckpt_path = os.path.join(model_root, model_name)
        print(f"[sdxl-cuda] ckpt_path={ckpt_path}")

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

        # Check if diffusers format
        is_diffusers_dir = os.path.isdir(ckpt_path) and os.path.exists(
            os.path.join(ckpt_path, "model_index.json")
        )

        # Load SDXL pipeline
        if is_diffusers_dir:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                ckpt_path,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if dtype == torch.float16 else None,
            )
            format_name = "diffusers"
        else:
            # Single-file SDXL checkpoint
            pipe = StableDiffusionXLPipeline.from_single_file(
                ckpt_path,
                torch_dtype=dtype,
            )
            format_name = "single-file"

        # Convert to LCM scheduler for fast inference
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

        # Enable VAE tiling for memory efficiency (important for SDXL)
        pipe.vae.enable_tiling()

        if attention_slicing:
            pipe.enable_attention_slicing()

        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print(f"[sdxl-cuda] worker {worker_id}: xformers enabled")
            except Exception as e:
                print(f"[sdxl-cuda] worker {worker_id}: xformers enable failed: {e!r}")

        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        # Get text encoder dimensions
        te_dim = getattr(getattr(self.pipe, "text_encoder", None), "config", None)
        te_dim = getattr(te_dim, "hidden_size", None)
        te2_dim = getattr(getattr(self.pipe, "text_encoder_2", None), "config", None)
        te2_dim = getattr(te2_dim, "hidden_size", None)

        print(f"[sdxl-cuda] text_encoder.hidden_size={te_dim}, text_encoder_2.hidden_size={te2_dim}")

        # Get UNet cross-attention dim (should be 2048 for SDXL)
        cad = getattr(getattr(self.pipe, "unet", None), "config", None)
        cad = getattr(cad, "cross_attention_dim", None)

        print(f"[sdxl-cuda] unet.cross_attention_dim={cad} pipeline={type(self.pipe).__name__}")

        # Load SDXL-compatible style LoRAs
        for sid, sd in STYLE_REGISTRY.items():
            # Filter by cross-attention dimension (SDXL requires 2048)
            if sd.required_cross_attention_dim is not None and cad is not None:
                if int(cad) != int(sd.required_cross_attention_dim):
                    print(
                        f"[sdxl-cuda] skip style '{sid}': incompatible cross_attention_dim "
                        f"(model={cad} style={sd.required_cross_attention_dim})"
                    )
                    self._style_loaded[sd.adapter_name] = False
                    continue

            try:
                try:
                    # Newer diffusers supports adapter_name
                    self.pipe.load_lora_weights(sd.lora_path, adapter_name=sd.adapter_name)
                    self._style_loaded[sd.adapter_name] = True
                    print(f"[sdxl-cuda] loaded style LoRA: {sid} -> {sd.lora_path} (adapter={sd.adapter_name})")
                except TypeError:
                    # Older diffusers: no adapter_name kwarg
                    self.pipe.load_lora_weights(sd.lora_path)
                    self._style_loaded[sd.adapter_name] = True
                    print(f"[sdxl-cuda] loaded style LoRA (no adapter_name API): {sid} -> {sd.lora_path}")
            except Exception as e:
                self._style_loaded[sd.adapter_name] = False
                print(f"[sdxl-cuda] FAILED to load style LoRA {sid}: {e!r}")

        # Detect best available runtime API for toggling
        if hasattr(self.pipe, "set_adapters") and hasattr(self.pipe, "disable_lora"):
            self._style_api = "adapters"
        elif hasattr(self.pipe, "fuse_lora"):
            self._style_api = "fuse"
        else:
            self._style_api = "none"

        print(
            f"[sdxl-cuda] worker {worker_id} loaded: {os.path.basename(ckpt_path)} "
            f"({format_name}) on {device} dtype={dtype_str} style_api={self._style_api}"
        )

    # ---------------------------
    # Style application (exclusive)
    # ---------------------------
    def _apply_style(self, style_id: str | None, level: int) -> None:
        """Apply or disable a style LoRA."""
        # turn off
        if not style_id or int(level) <= 0:
            if hasattr(self.pipe, "disable_lora"):
                self.pipe.disable_lora()
            elif hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters([])
            return

        sd = STYLE_REGISTRY.get(style_id)
        if not sd:
            if hasattr(self.pipe, "disable_lora"):
                self.pipe.disable_lora()
            return

        # clamp level 1..N
        lvl = max(1, min(int(level), len(sd.levels)))
        weight = float(sd.levels[lvl - 1])

        if not self._style_loaded.get(sd.adapter_name, False):
            return

        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([sd.adapter_name], adapter_weights=[weight])
        elif hasattr(self.pipe, "fuse_lora"):
            # fallback: not ideal if concurrent, but your CUDA path is 1 worker
            if hasattr(self.pipe, "unfuse_lora"):
                try:
                    self.pipe.unfuse_lora()
                except Exception:
                    pass
            self.pipe.fuse_lora(lora_scale=weight)

    # ---------------------------
    # Job execution
    # ---------------------------
    def run_job(self, job) -> tuple[bytes, int]:
        """
        Execute an SDXL generation job.

        Args:
            job: Job object with req (GenerateRequest)

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

        # Handle style LoRA
        sl = getattr(req, "style_lora", None)
        style_id = getattr(sl, "style", None) if sl else None
        level = int(getattr(sl, "level", 0)) if sl else 0

        self._apply_style(style_id, level)

        with torch.inference_mode():
            out = self.pipe(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=int(req.num_inference_steps),
                guidance_scale=float(req.guidance_scale),
                generator=gen,
            )

        # reset style to avoid state bleed
        self._apply_style(None, 0)

        img: Image.Image = out.images[0]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), seed

    def run_job_with_latents(self, job) -> Tuple[bytes, int, bytes]:
        """
        Execute SDXL generation and return latents.

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

        # Apply the same style for latent pass so latents match the image pass.
        sl = getattr(req, "style_lora", None)
        style_id = getattr(sl, "style", None) if sl else None
        level = int(getattr(sl, "level", 0)) if sl else 0

        self._apply_style(style_id, level)

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

        self._apply_style(None, 0)

        lat = out_lat.images
        if isinstance(lat, (list, tuple)):
            lat = lat[0]

        if not torch.is_tensor(lat):
            lat = torch.as_tensor(lat)

        if lat.ndim != 4:
            raise RuntimeError(f"Unexpected latent rank {lat.ndim}, shape={tuple(lat.shape)}")

        # Downsample to [1,4,8,8]
        # PERF NOTE: do pooling in fp32 to reduce pooling artifacts; then cast to fp16.
        lat_8 = torch.nn.functional.adaptive_avg_pool2d(lat.to(dtype=torch.float32), (8, 8))
        lat_8 = lat_8.to(dtype=torch.float16).contiguous()

        lat_np = lat_8.detach().cpu().numpy().astype(np.float16, copy=False)
        return png_bytes, seed, lat_np.tobytes(order="C")