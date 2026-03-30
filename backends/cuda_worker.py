# backends/cuda_worker.py
from __future__ import annotations

import io
import json
import os
from copy import deepcopy
from typing import Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline

from backends.styles import STYLE_REGISTRY
from backends.scheduler_registry import build_scheduler, normalize_scheduler_id


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


class CudaWorkerBase:
    """Shared base for CUDA diffusers workers.

    Centralises env-var parsing, device placement, and style-LoRA application
    so SD1.5 and SDXL workers stay in sync without code duplication.

    Subclass contract:
      - Call super().__init__(worker_id) first.
      - Load the pipeline, set the scheduler, then call:
            pipe = self._setup_pipe_memory_opts(pipe)
      - Store self.pipe after that call returns.
    """

    pipe: Any  # set by subclass __init__ after pipeline load

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self._style_loaded: dict[str, bool] = {}
        self._style_api: str = "unknown"
        self._img2img_pipe = None
        self._baseline_scheduler_class = None
        self._baseline_scheduler_config = None
        self._parse_env()

    def _parse_env(self) -> None:
        """Parse all CUDA_* env vars into instance attributes."""
        self.device = os.environ.get("CUDA_DEVICE", "cuda:0").strip()
        dtype_str = os.environ.get("CUDA_DTYPE", "fp16").lower().strip()
        if dtype_str == "bf16":
            self.dtype = torch.bfloat16
        elif dtype_str == "fp32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16
        self.dtype_str = dtype_str
        self._enable_xformers = _bool_env("CUDA_ENABLE_XFORMERS", "0")
        self._attention_slicing = _bool_env("CUDA_ATTENTION_SLICING", "0")
        self._quantize = os.environ.get("CUDA_QUANTIZE", "none").lower().strip()
        self._offload = os.environ.get("CUDA_OFFLOAD", "none").lower().strip()

    def _setup_pipe_memory_opts(self, pipe):
        """Apply device placement and memory optimizations to a loaded pipeline.

        Call after pipeline load and scheduler config, before storing self.pipe.
        xformers must be enabled before offload hooks are registered.
        Returns the (possibly modified) pipe.
        """
        if self._quantize == "fp8":
            # Checkpoint precision is storage metadata only. The current loaders
            # still materialize runtime modules as fp16/bf16/fp32 based on
            # CUDA_DTYPE, so CUDA_QUANTIZE must remain authoritative here.
            from optimum.quanto import freeze, quantize, qfloat8
            quantize(pipe.unet, weights=qfloat8)
            freeze(pipe.unet)
            if hasattr(pipe, "text_encoder_2"):  # SDXL only (~1.4 GB)
                quantize(pipe.text_encoder_2, weights=qfloat8)
                freeze(pipe.text_encoder_2)
            print(f"[cuda] worker {self.worker_id}: fp8 quantization applied")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        if self._attention_slicing:
            pipe.enable_attention_slicing(1)
        if self._enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print(f"[cuda] worker {self.worker_id}: xformers enabled")
            except Exception as e:
                print(f"[cuda] worker {self.worker_id}: xformers enable failed: {e!r}")
        gpu_id = self._device_index()
        if self._offload == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        elif self._offload == "model":
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe = pipe.to(self.device)
        return pipe

    def _device_index(self) -> int:
        """Parse the integer device index from self.device (e.g. 'cuda:1' → 1)."""
        if ":" in self.device:
            try:
                return int(self.device.split(":")[-1])
            except ValueError:
                return 0
        return 0

    def _normalize_img2img_modules(self) -> None:
        """
        Re-align shared img2img modules to the worker runtime dtype/device.

        Diffusers can upcast shared modules such as the VAE during prior runs.
        The next img2img encode then fails if the input tensor remains fp16
        while module bias tensors stayed fp32. Keep the correction narrow to
        the img2img path to avoid a broader VRAM penalty.
        """
        vae = getattr(getattr(self, "pipe", None), "vae", None)
        if vae is not None and hasattr(vae, "to"):
            vae.to(self.device, dtype=self.dtype)
        if self._img2img_pipe is not None and vae is not None:
            self._img2img_pipe.vae = vae

    def _capture_baseline_scheduler(self, pipe: Any) -> None:
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            self._baseline_scheduler_class = None
            self._baseline_scheduler_config = None
            return
        self._baseline_scheduler_class = scheduler.__class__
        self._baseline_scheduler_config = deepcopy(getattr(scheduler, "config", None))

    def _restore_baseline_scheduler(self) -> None:
        if self._baseline_scheduler_class is None or self._baseline_scheduler_config is None:
            return
        scheduler = self._baseline_scheduler_class.from_config(deepcopy(self._baseline_scheduler_config))
        self.pipe.scheduler = scheduler
        if self._img2img_pipe is not None:
            self._img2img_pipe.scheduler = scheduler

    def _allowed_scheduler_ids(self) -> Optional[set[str]]:
        allowed = getattr(self.model_info, "allowed_scheduler_ids", None)
        if not allowed:
            return None
        return {normalize_scheduler_id(scheduler_id) for scheduler_id in allowed}

    def _resolve_scheduler_id(self, req: Any) -> Optional[str]:
        requested = getattr(req, "scheduler_id", None)
        if requested:
            return normalize_scheduler_id(requested)
        default = getattr(self.model_info, "default_scheduler_id", None)
        if default:
            return normalize_scheduler_id(default)
        return None

    def _apply_request_scheduler(self, req: Any) -> Optional[str]:
        selected = self._resolve_scheduler_id(req)
        allowed = self._allowed_scheduler_ids()
        if selected is None:
            self._restore_baseline_scheduler()
            return None

        if allowed is not None and selected not in allowed:
            raise RuntimeError(f"scheduler_id '{selected}' is not allowed for the active mode")

        baseline_config = deepcopy(self._baseline_scheduler_config)
        scheduler = build_scheduler(selected, baseline_config)
        self.pipe.scheduler = scheduler
        if self._img2img_pipe is not None:
            self._img2img_pipe.scheduler = scheduler
        return selected

    # ---------------------------
    # Style application (exclusive)
    # ---------------------------
    def _apply_style(self, style_id: str | None, level: int) -> None:
        """Apply or disable a style LoRA."""
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


class DiffusersCudaWorker(CudaWorkerBase):
    """
    CUDA Diffusers worker for SD1.5 LCM models.

    Supports two formats:
      - Single file: .safetensors or .ckpt checkpoint
      - Diffusers layout: directory with model_index.json

    Env:
      CUDA_DTYPE=fp16|bf16|fp32   (default fp16)
      CUDA_DEVICE=cuda:0         (default cuda:0)
      CUDA_ENABLE_XFORMERS=1     (default 0)
      CUDA_ATTENTION_SLICING=0/1 (default 0)
    """
    def __init__(self, worker_id: int, model_path: str, model_info: Optional[Any] = None):
        super().__init__(worker_id)
        self.model_info = model_info

        ckpt_path = model_path
        print(f"[cuda] ckpt_path={ckpt_path}")

        format_hint = getattr(model_info, "loader_format", "unknown")
        is_diffusers_dir = format_hint == "diffusers_dir" or (
            format_hint == "unknown"
            and os.path.isdir(ckpt_path)
            and os.path.exists(os.path.join(ckpt_path, "model_index.json"))
        )

        if is_diffusers_dir:
            print("loading diffusers")
            pipe = StableDiffusionPipeline.from_pretrained(
                ckpt_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "diffusers"
        else:
            print("loading safetensors")
            pipe = StableDiffusionPipeline.from_single_file(
                ckpt_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "single-file"

        scheduler_profile = getattr(model_info, "scheduler_profile", "lcm")
        if scheduler_profile != "native":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = self._setup_pipe_memory_opts(pipe)

        self.pipe = pipe
        self._capture_baseline_scheduler(self.pipe)

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
            f"[cuda] worker {self.worker_id} loaded: {os.path.basename(ckpt_path)} "
            f"({format_name}) on {self.device} dtype={self.dtype_str} "
            f"quantize={self._quantize} offload={self._offload} style_api={self._style_api}"
        )

    # ---------------------------
    # Job execution
    # ---------------------------
    def run_job(self, job) -> tuple[bytes, int]:
        req = job.req
        init_image = getattr(job, 'init_image', None)

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
        scheduler_id = self._apply_request_scheduler(req)

        out = None
        try:
            if init_image is not None:
                # img2img path: reuse loaded weights at zero extra VRAM cost
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                if self._img2img_pipe is None:
                    self._img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipe.components)
                self._normalize_img2img_modules()
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                with torch.inference_mode():
                    out = self._img2img_pipe(
                        prompt=req.prompt,
                        negative_prompt=getattr(req, "negative_prompt", None),
                        image=init_pil,
                        strength=denoise_strength,
                        num_inference_steps=int(req.num_inference_steps),
                        guidance_scale=float(req.guidance_scale),
                        generator=gen,
                    )
            else:
                with torch.inference_mode():
                    out = self.pipe(
                        prompt=req.prompt,
                        negative_prompt=getattr(req, "negative_prompt", None),
                        width=width,
                        height=height,
                        num_inference_steps=int(req.num_inference_steps),
                        guidance_scale=float(req.guidance_scale),
                        generator=gen,
                    )

            img: Image.Image = out.images[0]  # type: ignore[union-attr]
            out = None  # release tensor reference before PNG encoding

            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("lcm", json.dumps({
                "prompt": req.prompt,
                "seed": seed,
                "size": req.size,
                "steps": int(req.num_inference_steps),
                "cfg": float(req.guidance_scale),
                "negative_prompt": getattr(req, "negative_prompt", None),
                "scheduler_id": scheduler_id,
            }))
            buf = io.BytesIO()
            img.save(buf, format="PNG", pnginfo=pnginfo)
            return buf.getvalue(), seed
        finally:
            out = None  # release on OOM/exception; no-op on success
            self._apply_style(None, 0)
            torch.cuda.empty_cache()

    def run_job_with_latents(self, job) -> Tuple[bytes, int, bytes]:
        """
        Returns:
          (png_bytes, seed_used, latents_bytes)

        latents_bytes:
          - raw tensor bytes for NCHW float16 with shape [1,4,8,8]
          - intended for hashing / similarity bookkeeping

        Single-pass: runs pipeline once with output_type="latent", then
        decodes via VAE. Eliminates the previous double-denoising approach.
        """
        from backends.latents import latent_to_nchw, downsample_to_8x8_nchw

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
        scheduler_id = self._apply_request_scheduler(req)

        with torch.inference_mode():
            out = self.pipe(
                prompt=req.prompt,
                negative_prompt=getattr(req, "negative_prompt", None),
                width=width,
                height=height,
                num_inference_steps=int(req.num_inference_steps),
                guidance_scale=float(req.guidance_scale),
                generator=gen,
                output_type="latent",
                return_dict=True,
            )

        self._apply_style(None, 0)

        lat = out.images  # type: ignore[union-attr]
        del out

        # Decode latents → pixel image
        with torch.inference_mode():
            decoded = self.pipe.vae.decode(lat / self.pipe.vae.config.scaling_factor).sample
        img = decoded.clamp(-1, 1).add(1).div(2)  # [-1,1] → [0,1]
        img = img[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
        img = Image.fromarray(img)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("lcm", json.dumps({
            "prompt": req.prompt,
            "seed": seed,
            "size": req.size,
            "steps": int(req.num_inference_steps),
            "cfg": float(req.guidance_scale),
            "negative_prompt": getattr(req, "negative_prompt", None),
            "scheduler_id": scheduler_id,
        }))
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=pnginfo)
        png_bytes = buf.getvalue()

        # Downsample latents to [1,4,8,8] float16 for similarity bookkeeping
        lat_nchw = latent_to_nchw(lat)
        lat_8 = downsample_to_8x8_nchw(lat_nchw).astype(np.float16)
        del lat, decoded
        torch.cuda.empty_cache()
        return png_bytes, seed, lat_8.tobytes(order="C")


class DiffusersSDXLCudaWorker(CudaWorkerBase):
    """
    CUDA Diffusers worker for SDXL (Stable Diffusion XL) models.

    Supports:
      - Single file: .safetensors or .ckpt checkpoint
      - Diffusers layout: directory with model_index.json
      - LCM-SDXL and regular SDXL with LCM scheduler
      - SDXL-compatible LoRAs (cross_attention_dim=2048)

    Env:
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

    def __init__(self, worker_id: int, model_path: str, model_info: Optional[Any] = None):
        super().__init__(worker_id)
        self.model_info = model_info

        ckpt_path = model_path
        print(f"[sdxl-cuda] ckpt_path={ckpt_path}")

        format_hint = getattr(model_info, "loader_format", "unknown")
        is_diffusers_dir = format_hint == "diffusers_dir" or (
            format_hint == "unknown"
            and os.path.isdir(ckpt_path)
            and os.path.exists(os.path.join(ckpt_path, "model_index.json"))
        )

        # Load SDXL pipeline
        if is_diffusers_dir:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                ckpt_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            format_name = "diffusers"
        else:
            # Single-file SDXL checkpoint
            pipe = StableDiffusionXLPipeline.from_single_file(
                ckpt_path,
                torch_dtype=self.dtype,
            )
            format_name = "single-file"

        scheduler_profile = getattr(model_info, "scheduler_profile", "native")
        if scheduler_profile == "lcm":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = self._setup_pipe_memory_opts(pipe)

        self.pipe = pipe
        self._capture_baseline_scheduler(self.pipe)

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
            f"[sdxl-cuda] worker {self.worker_id} loaded: {os.path.basename(ckpt_path)} "
            f"({format_name}) on {self.device} dtype={self.dtype_str} "
            f"quantize={self._quantize} offload={self._offload} style_api={self._style_api}"
        )

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
        init_image = getattr(job, 'init_image', None)

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
        scheduler_id = self._apply_request_scheduler(req)

        out = None
        try:
            if init_image is not None:
                # img2img path: reuse loaded weights at zero extra VRAM cost
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                if self._img2img_pipe is None:
                    self._img2img_pipe = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
                self._normalize_img2img_modules()
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                with torch.inference_mode():
                    out = self._img2img_pipe(
                        prompt=req.prompt,
                        negative_prompt=getattr(req, "negative_prompt", None),
                        image=init_pil,
                        strength=denoise_strength,
                        num_inference_steps=int(req.num_inference_steps),
                        guidance_scale=float(req.guidance_scale),
                        generator=gen,
                    )
            else:
                with torch.inference_mode():
                    out = self.pipe(
                        prompt=req.prompt,
                        negative_prompt=getattr(req, "negative_prompt", None),
                        width=width,
                        height=height,
                        num_inference_steps=int(req.num_inference_steps),
                        guidance_scale=float(req.guidance_scale),
                        generator=gen,
                    )

            img: Image.Image = out.images[0]  # type: ignore[union-attr]
            out = None  # release tensor reference before PNG encoding

            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("lcm", json.dumps({
                "prompt": req.prompt,
                "seed": seed,
                "size": req.size,
                "steps": int(req.num_inference_steps),
                "cfg": float(req.guidance_scale),
                "negative_prompt": getattr(req, "negative_prompt", None),
                "scheduler_id": scheduler_id,
            }))
            buf = io.BytesIO()
            img.save(buf, format="PNG", pnginfo=pnginfo)
            return buf.getvalue(), seed
        finally:
            out = None  # release on OOM/exception; no-op on success
            self._apply_style(None, 0)
            torch.cuda.empty_cache()

    def run_job_with_latents(self, job) -> Tuple[bytes, int, bytes]:
        """
        Execute SDXL generation and return latents.

        Returns:
          (png_bytes, seed_used, latents_bytes)

        latents_bytes:
          - raw tensor bytes for NCHW float16 with shape [1,4,8,8]
          - intended for hashing / similarity bookkeeping

        Single-pass: runs pipeline once with output_type="latent", then
        decodes via VAE. Eliminates the previous double-denoising approach.
        Note: SDXL VAE scaling_factor=0.13025 (from pipe.vae.config, no hardcode).
        """
        from backends.latents import latent_to_nchw, downsample_to_8x8_nchw

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
        scheduler_id = self._apply_request_scheduler(req)

        with torch.inference_mode():
            out = self.pipe(
                prompt=req.prompt,
                negative_prompt=getattr(req, "negative_prompt", None),
                width=width,
                height=height,
                num_inference_steps=int(req.num_inference_steps),
                guidance_scale=float(req.guidance_scale),
                generator=gen,
                output_type="latent",
                return_dict=True,
            )

        self._apply_style(None, 0)

        lat = out.images  # type: ignore[union-attr]
        del out

        # Decode latents → pixel image
        with torch.inference_mode():
            decoded = self.pipe.vae.decode(lat / self.pipe.vae.config.scaling_factor).sample
        img = decoded.clamp(-1, 1).add(1).div(2)  # [-1,1] → [0,1]
        img = img[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
        img = Image.fromarray(img)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("lcm", json.dumps({
            "prompt": req.prompt,
            "seed": seed,
            "size": req.size,
            "steps": int(req.num_inference_steps),
            "cfg": float(req.guidance_scale),
            "negative_prompt": getattr(req, "negative_prompt", None),
            "scheduler_id": scheduler_id,
        }))
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=pnginfo)
        png_bytes = buf.getvalue()

        # Downsample latents to [1,4,8,8] float16 for similarity bookkeeping
        lat_nchw = latent_to_nchw(lat)
        lat_8 = downsample_to_8x8_nchw(lat_nchw).astype(np.float16)
        del lat, decoded
        torch.cuda.empty_cache()
        return png_bytes, seed, lat_8.tobytes(order="C")
