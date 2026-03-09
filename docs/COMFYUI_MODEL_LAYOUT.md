# ComfyUI-Compatible Model Directory Layout

## Background

The backend's model path handling evolved organically and ended up with two conflicting
conventions in use at the same time, making `modes.yml` brittle:

- **Old convention** (`conf/modes.yml`): `model_root` pointed at a type-specific subdirectory
  (`/models/diffusers`), so `model:` keys were bare filenames.
- **New convention** (`modes-test.yml`, `modes.yaml.example`): `model_root` pointed at the
  top-level `/models` dir, and `model:` keys included the type subdirectory
  (`diffusers/lcm-dreamshaper-v7`).

`scan_models()` in `model_routes.py` assumed the new convention (scanning
`models_root/checkpoints`, `models_root/diffusers`, etc.), so it was silently broken whenever
`conf/modes.yml` was active because `model_root` was `/models/diffusers` — one level too deep.

Additionally, the worker pool was passing model paths to workers via environment variables
(`MODEL_ROOT` + `MODEL`), which the workers then re-joined. This made the path flow implicit
and untestable.

---

## Directory Structure (canonical)

The `models/` directory must follow ComfyUI's layout. `model_root` in `modes.yml` always
points to the top-level `models/` directory:

```
models/
  checkpoints/       ← single-file .safetensors / .ckpt (SD1.5, SDXL, etc.)
  diffusers/         ← diffusers pipeline roots (dirs containing model_index.json)
  diffusion_models/  ← UNet-format models (FLUX, etc.)
  loras/             ← LoRA .safetensors files
  clip/
  vae/
  unet/
  text_encoders/
  controlnet/
  upscale_models/
```

---

## modes.yml Convention

`model_root` is always the top-level models directory. `model:` values are paths relative
to `model_root` and **must include the type subdirectory**:

```yaml
model_root: /models
lora_root: /models/loras

modes:
  lcm-dreamshaper:
    model: diffusers/lcm-dreamshaper-v7      # diffusers pipeline
    default_size: 768x768
    default_steps: 8
    default_guidance: 3.0

  sdxl-pony:
    model: checkpoints/cyberrealisticPony_v160.safetensors  # single-file
    default_size: 1024x1024
    default_steps: 8
    default_guidance: 2.0

  flux-dev:
    model: diffusion_models/flux1-dev.safetensors  # UNet format
    default_size: 1024x1024
    default_steps: 20
    default_guidance: 3.5
```

LoRA paths in `loras:` are relative to `lora_root` (just the filename, no subdirectory needed):

```yaml
  sdxl-portrait:
    model: checkpoints/sdxl-base-1.0.safetensors
    loras:
      - path: portrait-enhancer.safetensors
        strength: 0.8
```

---

## Changes Made

### 1. `conf/modes.yml`
`model_root` changed from `/models/diffusers` → `/models`. Each `model:` value prefixed
with `diffusers/`.

### 2. `server/model_routes.py`
- **Bug fix**: `get_inventory_loras()` was reading `config.model_root` instead of
  `config.lora_root`. Fixed.
- `scan_models()` now also scans `diffusion_models/` as a third category.
- `get_inventory_models()` includes `diffusion_models` in the returned list.

### 3. `backends/worker_pool.py` + `backends/worker_factory.py` + `backends/cuda_worker.py`
Removed the env-var-based path channel. Previously `_load_mode()` set `MODEL_ROOT` and
`MODEL` env vars, and workers re-joined them. Now:
- `_load_mode()` passes `mode.model_path` (already fully resolved) directly to the factory.
- `create_cuda_worker(worker_id, model_path)` accepts the resolved path.
- `detect_worker_type(model_path)` takes the path directly.
- Both `DiffusersCudaWorker` and `DiffusersSDXLCudaWorker` accept `model_path` as a
  constructor argument instead of reading env vars.

This makes the path flow explicit, testable, and removes the env-var side-channel.

### 4. `server/lcm_sr_server.py`
Module-level `os.environ.get('MODEL_ROOT')` calls (lines 518, 526) were passed directly
to `os.path.join`, which raises `TypeError` when the env var is absent. Added safe
defaults (`''`) so the server can import cleanly regardless of environment.

### 5. `conf/modes.yaml.example`
Updated `model:` paths from non-standard `sdxl/` and `sd15/` subdirectory names to
ComfyUI-standard `checkpoints/`.
