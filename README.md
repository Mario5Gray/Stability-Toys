---
base_model:
- TheyCallMeHex/LCM-Dreamshaper-V7-ONNX
tags:
- rknn
- LCM
- stable-diffusion
- super-resolution
- fastapi
- rk3588
---

# RKNN LCM-SD + Super-Resolution Server (Rockchip)

Run **Stable Diffusion 1.5 Latent Consistency Model (LCM-SD)** on **Rockchip RKNPU2** (RK3588 / RK356x) using **RKNN2**, with an integrated **RKNN-accelerated Super-Resolution (SR)** pipeline.

This repo provides:

- **LCM-SD generation** via **CLI** and a **production-ready FastAPI server**
- **Server-side Super-Resolution**:
  - ✅ **Post-process SR inside `/generate`**
  - ✅ **Standalone SR endpoint `/superres`** (multipart upload)
- **Multi-worker safe design**:
  - one pipeline/runtime per worker
  - no shared RKNN objects across threads
  - queue backpressure with clean lifecycle
- **UI (Vite)** served from the same FastAPI process:
  - `/` serves `ui-dist`
  - supports multi-inflight, cancel, and **click-image-to-load settings**
  - supports **upload + SR** via `/superres`

> ⚠️ The SR model is not included. You must supply `super-resolution-10.rknn`.

---

## What’s new (recent additions)

### Server features
- **Unified server** (`lcm_sr_server.py`) running:
  - `POST /generate` (LCM-SD, optional SR postprocess)
  - `POST /superres` (standalone SR upload)
  - `POST /v1/superres` (versioned alias)
  - static UI mount at `/` (Vite dist)
- **SR magnitude support**:
  - `superres_magnitude` (1..3) controls the **number of SR passes**
- **Better observability via headers**:
  - generation: `X-Seed`, `X-SuperRes`
  - SR: `X-SR-Model`, `X-SR-Passes`, `X-SR-Scale-Per-Pass`, etc.
- **Queue backpressure**:
  - returns **429** when queue is full (both SD and SR queues)
- **Deterministic generation**:
  - `seed -> numpy.RandomState(seed)` per request

### UI features
- **Cancel per-request** and **Cancel all**
- **Multi-inflight** requests supported
- **Click on an image** to load its settings into current controls:
  - size / steps / cfg / seed / SR toggle
- **Upload image + super-res** directly from UI (multipart to `/superres`)

---

## Performance (RK3588, single NPU core)

| Resolution | Text Encoder | U-Net (per step) | VAE Decoder |
|-----------:|-------------:|-----------------:|------------:|
| 384×384    | ~0.05s       | ~2.36s           | ~5.48s      |
| 512×512    | ~0.05s       | ~5.65s           | ~11–14s     |

> NOTE: VAE decode latency is a known RKNN limitation and is not caused by layout, server, or postprocessing overhead.

---

## LCM-SD Optimizations & Quirks (Specific to This Repo)

- Correct tensor layouts:
  - Text encoder: **NCHW**
  - U-Net: **NHWC**
  - VAE decoder: **NHWC**
- One RKNN runtime context per worker (safe multi-context usage)
- Deterministic generation via explicit `numpy.RandomState(seed)`
- Queue backpressure to avoid runaway memory use under load

---

## Super-Resolution (SR) overview

Super-resolution is RKNN-accelerated and implemented as:

1) **Standalone REST endpoint**  
   `POST /superres` (multipart upload)

2) **Optional postprocess in** `POST /generate`  
   Set `"superres": true` in the JSON body.

### What SR is based on

This work is based on the original standalone demo repo:

- `rknn-superresolution` (algorithm + conversion approach)
- Uses:
  - RKNNLite inference on Rockchip NPU
  - YCbCr color space conversion
  - SR applied to luminance (Y channel)
  - tiled inference for arbitrary image sizes
  - bicubic upscaling for chroma channels

The server wraps that approach with:
- FastAPI endpoints
- worker-owned RKNN runtimes
- queueing/backpressure
- optional integration inside SD generation

---

## Model requirements (SR model not included)

The server expects an RKNN-converted super-resolution model:

- Expected filename (default): `super-resolution-10.rknn`
- Default location: `${MODEL_ROOT}/super-resolution-10.rknn`

### Original source model
- `super-resolution-10.onnx` (Sub-Pixel CNN 2016)
- ONNX model zoo source:
  https://github.com/onnx/models/tree/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model

### Conversion (same approach as original demo)
1. Download `super-resolution-10.onnx`
2. Convert on a PC with RKNN Toolkit installed
3. Set `target_platform` to match your board (e.g., rk3588)
4. Copy resulting `super-resolution-10.rknn` to your device

> The server will fail-fast if `SR_ENABLED=1` and `SR_MODEL_PATH` does not exist.

---

## Server deployment

### Environment variables (LCM + SR)

```bash
# LCM-SD model root
PORT=4200

# LCM worker configuration
NUM_WORKERS=3
QUEUE_MAX=64
DEFAULT_SIZE=512x512
DEFAULT_STEPS=4
DEFAULT_GUIDANCE=1.0
DEFAULT_TIMEOUT=120

# RKNN multi-context (if supported by your RKNN2Model wrapper)
USE_RKNN_CONTEXT_CFGS=1

# --- Super-resolution ---
SR_ENABLED=1
SR_MODEL_PATH=/models/lcm_rknn/super-resolution-10.rknn
SR_INPUT_SIZE=224
SR_OUTPUT_SIZE=672

SR_NUM_WORKERS=1
SR_QUEUE_MAX=32
SR_MAX_PIXELS=24000000
SR_REQUEST_TIMEOUT=120

MODELS_HOST_PATH=/path_to_your_models_folder
FS_HOST_PATH=/path_to_a_empty_folder_for_cache
```

### Model Root

Since pretrained diffusers or safetensors can be used,
set MODEL_ROOT the name of your safetensor file, or the name of the 
diffusers basedir.

## Start the server

```bash
python lcm_sr_server.py
```

Or with uvicorn explicitly:

```bash
uvicorn lcm_sr_server:app \
  --host 0.0.0.0 \
  --port 4200
```

On startup you should see logs indicating:
  ** LCM workers created
  ** SR workers created (if enabled)
  ** each SR worker loads super-resolution-10.rknn

# Endpoints

## POST /generate (LCM-SD, optional SR postprocess)

Request body(JSON)
```json
{
  "prompt": "a cinematic forest at sunrise",
  "size": "512x512",
  "num_inference_steps": 4,
  "guidance_scale": 1.0,
  "seed": 12345678,

  "superres": true,
  "superres_format": "png",
  "superres_quality": 92,
  "superres_magnitude": 2
}
```

Response:
- 200 OK
- content-type image/png or image/jpeg depending on SR format
- image bytes

Useful headers:
- X-Seed: <seed>
- X-SuperRes: 1|0
- If SR ran:
- X-SR-Model
- X-SR-Passes
- X-SR-Scale-Per-Pass
- X-SR-Format

### curl example

```bash
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -o output.png \
  -d '{
    "prompt": "a cinematic forest at sunrise",
    "size": "512x512",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 12345678,
    "superres": true,
    "superres_format": "png",
    "superres_magnitude": 2
  }'
```

## POST /superres (standalone SR upload)

Multipart form fields:
  ** file: image upload (png/jpg/webp/etc)
  ** magnitude: 1..3 (default 2)
  ** out_format: png or jpeg (default png)
  ** quality: 1..100 (jpeg only)

### curl example

```bash
curl -F "file=@input.png" \
  -F "magnitude=2" \
  -F "out_format=jpeg" \
  -F "quality=92" \
  http://localhost:4200/superres \
  --output out.jpg
```

Also available:
  ** POST /v1/superres (alias)

# UI

The server mounts the UI build at:
  • GET / → serves /opt/lcm-sr-server/ui-dist

## UI features
  ** Send prompts to /generate
  ** Toggle SR postprocess for generated images
  ** Upload an image and send it to /superres
  ** Cancel in-flight requests
  ** Click any generated image to load its parameters (size/steps/cfg/seed/SR) into the current controls

## Command-line usage (LCM-SD only)

```bash
python ./run_rknn-lcm.py \
  -i ./model \
  -o ./images \
  --num-inference-steps 4 \
  -s 512x512 \
  --prompt "Majestic mountain landscape with snow-capped peaks, autumn foliage, turquoise river, ultra-realistic."
```

# Docker Usage (LCM-sd server)

## Build

```bash
docker build -t rknn-lcm-sd .
```
## Run


```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/rknpu \
  -v ./model:/models \
  -e MODEL_ROOT=/models \
  -e NUM_WORKERS=3 \
  -e SR_ENABLED=1 \
  -e SR_MODEL_PATH=/models/super-resolution-10.rknn \
  -p 4200:4200 \
  rknn-lcm-sd
```

A docker-compose.yml may also be provided depending on your branch.


# Known issues

1.  VAE decode is slower than other stages on RKNN (known behavior).
2.  Some RKNN toolkit versions historically caused precision loss; ensure you’re on a known-good toolkit version for your models.
3.  Multi-resolution conversion in one pass may fail depending on toolkit version (toolkit limitation).

# References
 - Base model: TheyCallMeHex/LCM-Dreamshaper-V7-ONNX
https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX
 - Optimum LCM pipeline reference
https://github.com/huggingface/optimum/blob/main/optimum/pipelines/diffusers/pipeline_latent_consistency.py
 - Prior art / inspiration
https://github.com/happyme531/RK3588-stable-diffusion-GPU
 - SR demo repo (original algorithm reference)
https://github.com/Mario5Gray/rknn-superresolution

