# CUDA Super-Resolution

This document covers the CUDA super-resolution path used by both:

- `POST /generate` with `"superres": true`
- `POST /superres`

The API shape is unchanged from RKNN deployments. Only the backend implementation changes.

## Model

Recommended starting model:

- `RealESRGAN_x4plus.pth`

Recommended path:

- `/models/sr/RealESRGAN_x4plus.pth`

Source:

- Real-ESRGAN project: <https://github.com/xinntao/Real-ESRGAN>

The server does not download weights automatically. Place the model on disk before startup.

## Environment

Use these variables on CUDA deployments:

```bash
BACKEND=cuda
SR_ENABLED=1
CUDA_SR_MODEL=/models/sr/RealESRGAN_x4plus.pth
CUDA_SR_TILE=0
CUDA_SR_FP16=1
SR_QUEUE_MAX=32
SR_REQUEST_TIMEOUT=120
```

Notes:

- `CUDA_SR_TILE=0` disables tiling. This is fastest but uses more VRAM.
- If VRAM is tight, increase `CUDA_SR_TILE` to values such as `256` or `512`.
- `CUDA_SR_FP16=1` is the default low-VRAM path.

## Behavior

- CUDA SR is lazy-loaded on the first SR request.
- The SR worker is separate from the txt2img/img2img CUDA workers.
- If CUDA SR OOMs, the SR worker unloads and the next SR request cold-reloads it.
- `superres_magnitude` still means repeated SR passes.

## Manual Acceptance

Run this on a CUDA machine with the model already present.

### 1. Startup

Start the server with CUDA SR enabled. Startup should succeed without requiring an SR request up front.

### 2. Standalone `/superres`

```bash
curl -sS -D /tmp/sr.headers \
  -o /tmp/sr.png \
  -F file=@input.png \
  -F magnitude=1 \
  -F out_format=png \
  -F quality=92 \
  http://localhost:4200/superres
```

Expected:

- HTTP 200
- image bytes returned
- `X-SR-Model` present and matches the CUDA SR model filename
- `X-SR-Passes` present
- `X-SR-Scale-Per-Pass` present

### 3. `/generate` with postprocess SR

```bash
curl -sS -D /tmp/generate-sr.headers \
  -o /tmp/generate-sr.png \
  -H 'Content-Type: application/json' \
  -X POST http://localhost:4200/generate \
  -d '{
    "prompt": "a studio photo of a ceramic owl",
    "size": "512x512",
    "num_inference_steps": 8,
    "guidance_scale": 2.5,
    "superres": true,
    "superres_magnitude": 1,
    "superres_format": "png",
    "superres_quality": 92
  }'
```

Expected:

- HTTP 200
- `X-SuperRes: 1`
- `X-SR-Model` present
- `X-SR-Passes` present
- `X-SR-Scale-Per-Pass` present

### 4. OOM Recovery

If an SR request fails with CUDA OOM:

1. Confirm the request fails cleanly.
2. Submit another SR request.
3. Confirm the later request succeeds after the SR worker cold-reloads.

If OOM occurs repeatedly:

- increase `CUDA_SR_TILE`
- keep `CUDA_SR_FP16=1`

### 5. CLI

```bash
python -m server.superres_cli \
  --input ./input.png \
  --output ./output.png \
  --magnitude 1 \
  --format png
```

## Verification Status

The code path and unit-level behavior are implemented and tested in the repository.
The manual CUDA acceptance steps above still require execution on real CUDA hardware.
