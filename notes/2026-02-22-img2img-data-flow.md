# img2img Data Flow

**Date:** 2026-02-22
**Status:** Implemented

---

## Key Insight: Zero-VRAM Pipeline Sharing

diffusers allows creating an img2img pipeline directly from an existing txt2img pipeline's
`components` dict. This reuses the same loaded UNet/VAE/CLIP tensors in memory — no new
weights are loaded, no extra VRAM is consumed, just a different pipeline wrapper object.

```python
# SD 1.5
from diffusers import StableDiffusionImg2ImgPipeline
img2img = StableDiffusionImg2ImgPipeline(**self.pipe.components)

# SDXL
from diffusers import StableDiffusionXLImg2ImgPipeline
img2img = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
```

The `_img2img_pipe` is lazily created on the first img2img request and cached on the worker
instance. Subsequent img2img jobs reuse the same wrapper.

---

## Full Data Flow

```
User drops/uploads an image onto the chat area
  → ChatDropzone calls setInitImage({ file: File, objectUrl: string })
  → InitImagePreview thumbnail appears in OptionsPanel
  → StrengthSlider (0.01–1.0) becomes visible

User types prompt + adjusts denoise strength + clicks Send
  → App.onSend passes { initImageFile: File, denoiseStrength: 0.75 } to runGenerate

useImageGeneration.runGenerate:
  → Enqueues job with payload: { ...params, initImageFile, denoiseStrength }

Job runner (async, inside jobQueue):
  → If payload.initImageFile:
      POST /v1/upload  (FormData)  →  { fileRef: "abc123" }
  → generateViaWsWithRetry({ ...payload, initImageRef: "abc123", denoiseStrength })

generateRunnerWs.js:
  → job:submit with params: {
        prompt, size, steps, cfg, seed, superres,
        init_image_ref: "abc123",
        denoise_strength: 0.75
      }

ws_routes._run_generate (backend):
  → GenerateRequest(... denoise_strength=0.75)
  → resolve_file_ref("abc123") → bytes (raises KeyError if TTL expired)
  → GenerationJob(req=req, init_image=bytes)
  → pool.submit_job(job)

cuda_worker.run_job:
  → job.init_image is not None
  → PIL.Image.open(bytes).resize((width, height))
  → lazy-create self._img2img_pipe from self.pipe.components
  → self._img2img_pipe(prompt, image=pil, strength=req.denoise_strength, ...)
  → returns (png_bytes, seed)
```

---

## Strength Semantics

| Strength | Effect |
|----------|--------|
| 0.01     | Nearly identical to init image |
| 0.5      | Half-way blend |
| 0.75     | Default — significant creative change while preserving composition |
| 1.0      | Fully regenerated (equivalent to txt2img) |

---

## Upload TTL

Uploaded init images are stored in-memory for **5 minutes** (`TTL_S = 300` in `upload_routes.py`).
If a user uploads an image and waits more than 5 minutes before sending, the server returns
`job:error` with a "fileRef not found or expired" message.

---

## Cache Behavior

img2img results are **not cached** (unlike txt2img). The cache key is based on
`{ prompt, size, steps, cfg, seed, superres, superresLevel }` — it does not include the init
image content. Caching img2img results would risk serving stale results for a different init
image with the same params.

---

## Out of Scope (not implemented here)

- "Use generated image as init" (from history) — separate operator
- RKNN backend img2img — CUDA mode only
- `run_job_with_latents` img2img variant
