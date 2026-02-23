# RKNN img2img — Work Layout

**Date:** 2026-02-22
**Status:** Not implemented

---

## Current State

The CUDA worker (`backends/cuda_worker.py:228`) handles `job.init_image` by lazily creating a
`StableDiffusionImg2ImgPipeline` from the existing pipeline's `.components` dict — zero VRAM
cost, shares UNet/VAE/CLIP weights.

The RKNN worker (`backends/rknn_worker.py:68`) **ignores `job.init_image` entirely** — it runs
txt2img unconditionally. This is explicitly out of scope in `notes/2026-02-22-img2img-data-flow.md`.

---

## Root Blocker: No VAE Encoder

CUDA img2img works because diffusers' `StableDiffusionImg2ImgPipeline` encodes the init image to
latent space via the **VAE encoder**, adds noise at the appropriate timestep, then starts denoising
from there.

`RKNN2LatentConsistencyPipeline` only loads three RKNN models (see `backends/base.py:ModelPaths`):

| Component | Status |
|---|---|
| `text_encoder` | loaded |
| `unet` | loaded |
| `vae_decoder` | loaded |
| `vae_encoder` | **missing** |

No `.rknn` model file for the VAE encoder exists. No `vae_encoder` property on `ModelPaths`.

---

## Work Required (primary path)

### 1. VAE Encoder RKNN model *(biggest effort — offline)*
- Export the VAE encoder from the base SD1.5 diffusers checkpoint to ONNX
- Convert ONNX → `.rknn` using the RKNN Toolkit
- Validate I/O shapes: input `[1,3,H,W]` fp32 → output `[1,4,H/8,W/8]` (latent mean)
  - If encoder outputs `[1,8,H/8,W/8]` (mean+logvar), take the first 4 channels

### 2. `ModelPaths.vae_encoder` property (`backends/base.py`)
- Add property returning `os.path.join(self.root, "vae_encoder")`

### 3. Load VAE encoder in `RKNN2LatentConsistencyPipeline` (`backends/rknnlcm.py:180`)
- Accept `vae_encoder: Optional[RKNN2Model] = None` in `__init__`
- Store as `self.vae_encoder`

### 4. `encode_image` method (`backends/rknnlcm.py`)
- Preprocess PIL image → normalized `[1,3,H,W]` float32 (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
- Run through `self.vae_encoder(...)`
- Return latent (scale by `vae_decoder.config["scaling_factor"]`)

### 5. `prepare_latents_img2img` method (`backends/rknnlcm.py`)
- Encode init image via step 4
- Compute start timestep from `strength`: `t_start = int(num_inference_steps * (1 - strength))`
- Add scheduler noise at `timesteps[t_start]`

### 6. Extend `__call__` (`backends/rknnlcm.py:450`)
- Add `image: Optional[PIL.Image] = None` and `strength: float = 0.75` params
- If `image` is set: call `prepare_latents_img2img` instead of `prepare_latents`, slice
  `timesteps = timesteps[t_start:]`

### 7. Wire into `RKNNPipelineWorker.run_job` (`backends/rknn_worker.py:68`)
- Mirror the CUDA worker pattern:
  ```python
  init_image = getattr(job, 'init_image', None)
  if init_image is not None:
      init_pil = Image.open(io.BytesIO(init_image)).resize((width, height))
      result = self.pipe(..., image=init_pil, strength=req.denoise_strength)
  ```

### 8. Load VAE encoder in `RKNNPipelineWorker._init_pipeline` (`backends/rknn_worker.py:58`)
- Pass `vae_encoder=self._mk_model(self.paths.vae_encoder, data_format="nhwc")` into the
  pipeline constructor

---

## Alternative: CPU VAE Encoder (defer RKNN conversion)

If VAE encoder RKNN model conversion is deferred, the encode step can run on **CPU via diffusers**:

1. Load `AutoencoderKL` from the original SD1.5 checkpoint in float32 on CPU
2. Encode init image to latents using `vae.encode(image).latent_dist.sample()`
3. Scale latents: `latents *= vae.config.scaling_factor`
4. Convert to numpy, add noise via scheduler, pass into `RKNN2LatentConsistencyPipeline.__call__`
   via the existing `latents=` parameter

UNet + VAE decoder remain on RKNN. Trade-off: ~200–400ms CPU encode penalty per job; avoids the
RKNN Toolkit model conversion work entirely. Good for validation before committing to full
RKNN encoder export.

---

## Files to Touch

| File | Change |
|---|---|
| `backends/base.py` | add `ModelPaths.vae_encoder` property |
| `backends/rknnlcm.py` | add `vae_encoder` param, `encode_image`, `prepare_latents_img2img`, extend `__call__` |
| `backends/rknn_worker.py` | load vae_encoder, handle `job.init_image` in `run_job` |
