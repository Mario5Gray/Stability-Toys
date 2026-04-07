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
- sdxl
- lora
- dynamic-model-loading
---

# Dream Lab - Experimental Generative Workflows

A flexible inference server for Stable Diffusion and SDXL with dynamic model loading, LoRA support, and experimental NPU acceleration.

## Origin & Evolution

This project originated as a **proof-of-concept for running LCM/SD models on embedded IoT platforms** powered by RockChip NPUs (RK3588/RK356x). The goal was to demonstrate that efficient generative AI could run on edge devices using RKNN (Rockchip Neural Network) acceleration.

It has since evolved into a **playground for experimenting with various generative workflows**, including:
- Dynamic model hot-swapping
- Multi-model "mode" configurations
- LoRA composition and management
- Super-resolution pipelines
- Both GPU (CUDA) and NPU (RKNN) backends

The original RKNN-focused repository exists at the upstream project. **Going forward, you will see continual support for RKNN**, as it was the original motivation. Development continues toward making RKNN and IoT NPU rendering nearly as fast as full-blown GPU rendering.

**T:HIS project** is the experimental branch where new features are prototyped, tested, and refined before potentially being backported to the RKNN-optimized upstream.

---

## Architecture Overview

Dream Lab provides a **production-ready FastAPI backend** with:

- **Dynamic Model Loading** - Hot-swap models and LoRAs at runtime without restarting
- **Mode System** - Define model + LoRA combinations as reusable "modes" in YAML
- **Multi-Backend Support** - CUDA (GPU) and RKNN (NPU) workers with automatic detection
- **Extensible Job Queue** - Custom job types for preprocessing, generation, and post-processing
- **VRAM Management** - Real-time tracking with no artificial limits
- **Super-Resolution** - Optional post-processing pipeline
- **REST API** - Clean HTTP API for generation, mode switching, and management

---

## Dynamic Model Loading System

The core feature of Dream Lab is **hot-swappable models and LoRAs** configured via `modes.yaml`.

### Quick Start

1. **Create your mode configuration:**

```bash
cp modes.yaml.example modes.yaml
```

2. **Define your modes:**

```yaml
model_root: /models
lora_root: /models/loras
default_mode: sdxl-general

modes:
  sdxl-general:
    model: sdxl/sdxl-base-1.0.safetensors
    default_size: "1024x1024"
    default_steps: 30
    default_guidance: 7.5

  sdxl-portrait:
    model: sdxl/sdxl-base-1.0.safetensors
    loras:
      - path: portrait-enhancer.safetensors
        strength: 0.8
    default_size: "896x1152"
    default_steps: 35
    default_guidance: 8.0

  sd15-fast:
    model: sd15/dreamshaper-lcm.safetensors
    default_size: "512x512"
    default_steps: 4
    default_guidance: 1.0
```

3. **Start the server:**

```bash
./runner.sh
# or
python server/lcm_sr_server.py
```

The server automatically:
- Loads the default mode on startup
- Detects model type (SD1.5 vs SDXL) automatically
- Hot-reloads `modes.yaml` when you edit it
- Tracks VRAM usage in real-time
- Queues mode switches gracefully

### Features

#### Mode Management
- **Hot-reload**: Edit `modes.yaml` and changes apply automatically (inotify/FSEvents)
- **SIGHUP**: Send `kill -HUP <pid>` to manually reload configuration
- **API reload**: `POST /api/modes/reload`
- **No downtime**: In-flight generations continue during config changes

#### Job Control and Recovery
- Per-image cancel stops pending generation delivery.
- `Configuration -> Reload Active Model` reloads the active mode in place for same-mode recovery.
- `Configuration -> Free VRAM` cancels queued work, marks running generation jobs canceled on a best-effort basis, unloads the worker, and clears reclaimable CUDA cache.

#### VRAM Tracking
- **Real-time monitoring**: Uses `torch.cuda.memory_allocated()` for accuracy
- **No artificial limits**: Uses ALL available VRAM intelligently
- **Capacity checking**: Automatic validation before loading models
- **Statistics**: Full breakdown via `GET /api/vram`

#### Extensible Queue
- **Multiple job types**: Generation, mode switching, custom processing
- **Custom jobs**: Queue arbitrary functions from anywhere in your app
- **FIFO processing**: Jobs execute in order with automatic worker switching
- **Backpressure**: Returns 429 when queue is full

---

## LoRA Support

Load multiple LoRAs per mode with configurable strengths:

```yaml
modes:
  custom-style:
    model: sdxl/sdxl-base-1.0.safetensors
    loras:
      # Simple format
      - portrait-style.safetensors

      # Full format with strength and name
      - path: detail-enhancer.safetensors
        strength: 0.7
        adapter_name: detail

      - path: lighting-fix.safetensors
        strength: 1.2
        adapter_name: lighting
```

LoRAs are:
- Loaded automatically with the mode
- Applied in order with specified strengths
- Tracked in VRAM statistics
- Hot-swappable when changing modes

---

## Backend Deployment

### Required Environment Variables

```bash
# Server configuration
PORT=4200
HOST=0.0.0.0

# Worker configuration
NUM_WORKERS=3           # Number of generation workers (RKNN only)
QUEUE_MAX=64           # Maximum queue size

# Default generation parameters (used if modes.yaml not found)
DEFAULT_SIZE=512x512
DEFAULT_STEPS=4
DEFAULT_GUIDANCE=1.0
DEFAULT_TIMEOUT=120

# CUDA (GPU backend) — all optional, safe to omit
CUDA_DEVICE=cuda:0         # Which GPU to use
CUDA_DTYPE=fp16            # Weight dtype: fp16 | bf16 | fp32
CUDA_ENABLE_XFORMERS=0     # Enable xformers memory-efficient attention: 0 | 1
CUDA_ATTENTION_SLICING=0   # Enable attention slicing (lower VRAM, slower): 0 | 1
CUDA_QUANTIZE=none         # fp8 weight quantization via optimum-quanto: none | fp8
                           # Halves UNet VRAM footprint; requires optimum-quanto installed
                           # Combine with CUDA_OFFLOAD=model for maximum savings
CUDA_OFFLOAD=none          # CPU offload strategy: none | model | sequential
                           # model: moves each sub-model to GPU only when needed (recommended)
                           # sequential: layer-by-layer offload (lowest VRAM, slowest)
                           # CUDA_DEVICE is respected; set it to target a specific GPU

# RKNN-specific (NPU backend)
USE_RKNN_CONTEXT_CFGS=1    # Enable multi-context support
MODEL_ROOT=/models           # Path to models directory (legacy)
MODEL=model.safetensors      # Model filename (legacy)

# Super-resolution (optional)
SR_ENABLED=1
SR_QUEUE_MAX=32
SR_REQUEST_TIMEOUT=120

# RKNN super-resolution
SR_MODEL_PATH=/models/super-resolution-10.rknn
SR_INPUT_SIZE=224
SR_OUTPUT_SIZE=672
SR_NUM_WORKERS=1
SR_MAX_PIXELS=24000000

# CUDA super-resolution
CUDA_SR_MODEL=/models/sr/RealESRGAN_x4plus.pth
CUDA_SR_TILE=0            # 0 disables tiling; raise on lower-VRAM GPUs
CUDA_SR_FP16=1            # 1 uses fp16 when supported; 0 keeps fp32
```

### Super-Resolution CLI

You can run the shared SR pipeline directly from the shell without starting HTTP:

```bash
python -m server.superres_cli \
  --input ./input.png \
  --output ./output.png \
  --magnitude 2 \
  --format png
```

The CLI uses the same backend selection and SR environment variables as the server:

- `BACKEND`
- `SR_MODEL_PATH`
- `SR_INPUT_SIZE`
- `SR_OUTPUT_SIZE`
- `SR_QUEUE_MAX`
- `SR_REQUEST_TIMEOUT`
- `SR_MAX_PIXELS`
- `CUDA_SR_MODEL`
- `CUDA_SR_TILE`
- `CUDA_SR_FP16`

This is the quickest way to validate RKNN or CUDA super-resolution without going through `/superres`.

### Backend Selection

The server automatically detects which backend to use:

1. **CUDA (GPU) Backend**: Used if `torch.cuda.is_available()` returns True
   - Supports: SD1.5, SD2.x, SDXL, SDXL Refiner
   - Features: Dynamic model loading, LoRA, modes system
   - Auto-detection: Based on `cross_attention_dim` (768/1024/1280/2048)

2. **RKNN (NPU) Backend**: Used on RockChip devices with RKNN runtime
   - Supports: LCM-SD 1.5 (optimized for RKNN)
   - Features: Multi-worker, super-resolution
   - Optimized: Tensor layouts (NCHW/NHWC), deterministic generation

### Super-Resolution Backend Selection

Super-resolution follows the active deployment backend:

- `BACKEND=rknn` uses the RKNN SR service and `SR_MODEL_PATH`
- `BACKEND=cuda` uses the CUDA SR service and `CUDA_SR_MODEL`
- `BACKEND=auto` selects CUDA SR when generation is using CUDA, otherwise RKNN SR

The public API stays the same on both backends:

- `POST /generate` with `"superres": true`
- `POST /superres`

### Mode System vs Legacy Environment Variables

**NEW (Recommended): modes.yaml**
```yaml
model_root: /models
default_mode: my-mode

modes:
  my-mode:
    model: sdxl-base.safetensors
    loras: [portrait.safetensors]
```

**OLD (Deprecated): Environment variables**
```bash
MODEL_ROOT=/models
MODEL=sdxl-base.safetensors
LORA_ROOT=/models/loras
```

If `modes.yaml` exists, it takes precedence. Otherwise, server falls back to legacy env vars.

---

## REST API

### Model Management

#### List Available Modes
```bash
GET /api/modes
```

Returns all configured modes with their settings.

#### Get Current Status
```bash
GET /api/models/status
```

Returns current mode, queue size, and VRAM statistics.

#### Switch Mode
```bash
POST /api/modes/switch
Content-Type: application/json

{
  "mode": "sdxl-portrait"
}
```

Queues a mode switch. Returns immediately; switch happens after pending jobs.

#### Reload Configuration
```bash
POST /api/modes/reload
```

Manually reload `modes.yaml` from disk.

#### VRAM Statistics
```bash
GET /api/vram
```

Returns detailed VRAM usage, available space, and per-model breakdown.

### Generation

#### Generate Image
```bash
POST /generate
Content-Type: application/json

{
  "mode": "sdxl-portrait",        # Optional: switch to this mode
  "prompt": "a beautiful portrait",
  "size": "896x1152",             # Optional: override mode default
  "num_inference_steps": 35,      # Optional: override mode default
  "guidance_scale": 8.0,          # Optional: override mode default
  "seed": 42,                     # Optional: for reproducibility

  "superres": true,               # Optional: apply SR post-processing
  "superres_magnitude": 2,        # Optional: number of SR passes (1-3)
  "superres_format": "png",       # Optional: png or jpeg
  "superres_quality": 92          # Optional: jpeg quality (1-100)
}
```

**Response:**
- Returns PNG or JPEG image bytes
- Headers:
  - `X-Mode`: Mode used for generation
  - `X-Seed`: Seed used (generated if not provided)
  - `X-SuperRes`: Whether SR was applied (0 or 1)
  - `X-SR-Passes`: Number of SR passes (if SR enabled)

**Behavior:**
1. If `mode` specified and different from current → queue mode switch
2. Apply mode's default settings for any omitted parameters
3. Queue generation job
4. Return image with metadata headers

#### Standalone Super-Resolution
```bash
POST /superres
Content-Type: multipart/form-data

file=@input.png
magnitude=2
out_format=jpeg
quality=92
```

Upload an image for super-resolution without generation.

---

## Model Requirements

### GPU (CUDA) Backend

**Supported Models:**
- Stable Diffusion 1.5 (cross_attention_dim: 768)
- Stable Diffusion 2.x (cross_attention_dim: 1024)
- SDXL Base (cross_attention_dim: 2048)
- SDXL Refiner (cross_attention_dim: 1280)

**Format:**
- `.safetensors` files (recommended)
- Diffusers format directories (supported)

**Storage:**
- Place models in `model_root` directory
- Paths in `modes.yaml` are relative to `model_root`
- LoRAs in `lora_root` (defaults to `model_root`)

### NPU (RKNN) Backend

**Supported Models:**
- LCM-SD 1.5 (converted to RKNN format)

**Required Files:**
- Text encoder: `text_encoder.rknn`
- U-Net: `unet.rknn`
- VAE decoder: `vae_decoder.rknn`

**Conversion:**
Use RKNN-Toolkit2 to convert ONNX models to RKNN format for your target platform (rk3588, rk3566, etc.).

### Super-Resolution Models

#### RKNN / NPU

**File:** `super-resolution-10.rknn`

**Source:**
- ONNX Model Zoo: [Sub-Pixel CNN 2016](https://github.com/onnx/models/tree/main/validated/vision/super_resolution/sub_pixel_cnn_2016)
- Convert with RKNN-Toolkit2 for NPU acceleration

**Note:** RKNN SR model is not included. It must be converted and supplied separately.

#### CUDA / GPU

**Recommended file:** `RealESRGAN_x4plus.pth`

**Recommended location:** `/models/sr/RealESRGAN_x4plus.pth`

**Configuration:**
- `CUDA_SR_MODEL=/models/sr/RealESRGAN_x4plus.pth`
- `CUDA_SR_TILE=0` on larger GPUs, increase it when VRAM is tight
- `CUDA_SR_FP16=1` for the default low-VRAM path

**Source:**
- Real-ESRGAN project: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

**Notes:**
- CUDA SR is lazy-loaded on first SR request
- If CUDA SR OOMs, the SR worker unloads and the next SR request cold-reloads it
- `superres_magnitude` still means repeated upscale passes on both backends

---

## Performance

### GPU (CUDA)

Depends on GPU model and VRAM. Typical performance:
- **SD1.5**: ~2-5 seconds @ 512x512 (RTX 3090)
- **SDXL**: ~8-15 seconds @ 1024x1024 (RTX 3090)

### NPU (RKNN on RK3588)

| Component      | Resolution | Time per Step |
|----------------|------------|---------------|
| Text Encoder   | N/A        | ~0.05s        |
| U-Net          | 384×384    | ~2.36s        |
| U-Net          | 512×512    | ~5.65s        |
| VAE Decoder    | 384×384    | ~5.48s        |
| VAE Decoder    | 512×512    | ~11-14s       |

**Note:** VAE decode is slower on RKNN (known limitation). Not caused by server overhead.

### Mode Switching

- **Unload old model**: 1-2 seconds
- **Load new model**: 5-15 seconds (varies by model size)
- **Total**: ~10-20 seconds for SDXL ↔ SD1.5 switch

Mode switches are queued behind existing work. Jobs already in the queue continue before the switch completes unless you cancel them separately.

---

## Docker Deployment

### Build

```bash
docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t dream-lab .
```

For compose-based builds, pass the same value through the environment so the checked-in entrypoint can forward it:

```bash
GIT_SHA=$(git rev-parse --short HEAD) docker compose -f docker-cuda.yml build
```

If you prefer exporting it once for multiple compose commands, `export GIT_SHA=$(git rev-parse --short HEAD)` before running `docker compose`.

For the live-test workflow, build the UI dist with the same SHA first, then pass that same value into Compose:

```bash
GIT_SHA=$(git rev-parse --short HEAD)
cd lcm-sr-ui && VITE_APP_VERSION=$GIT_SHA yarn build
cd ..
GIT_SHA=$GIT_SHA docker compose -f docker-compose.live-test.yml up --build
```

If you use `--profile ui-dev`, Compose forwards the same `GIT_SHA` from the environment automatically, so the Vite dev server and backend stay aligned.

### Run (GPU)

```bash
docker run --rm -it \
  --gpus all \
  -v ./modes.yaml:/app/modes.yaml \
  -v ./models:/models \
  -e SR_ENABLED=1 \
  -e CUDA_SR_MODEL=/models/sr/RealESRGAN_x4plus.pth \
  -e CUDA_SR_TILE=0 \
  -e CUDA_SR_FP16=1 \
  -p 4200:4200 \
  dream-lab
```

### Run (RKNN/NPU)

```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/rknpu \
  -v ./modes.yaml:/app/modes.yaml \
  -v ./models:/models \
  -e MODEL_ROOT=/models \
  -e SR_ENABLED=1 \
  -e SR_MODEL_PATH=/models/super-resolution-10.rknn \
  -p 4200:4200 \
  dream-lab
```

See `docker-compose.yml` for additional configuration options.

### CUDA Super-Resolution Acceptance

Use these checks on a real CUDA machine after the service starts:

1. Confirm lazy-load at startup
   - Start with `BACKEND=cuda`
   - Verify startup succeeds without an immediate SR model load failure
   - Do not call `/superres` or `/generate` with `superres=true` yet

2. Validate standalone `/superres`

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
- image returned
- `X-SR-Model` matches the CUDA SR model filename
- `X-SR-Passes` and `X-SR-Scale-Per-Pass` are present

3. Validate `/generate` postprocess SR

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
- `X-SR-Model` matches the standalone `/superres` response
- `X-SR-Passes` and `X-SR-Scale-Per-Pass` are present

4. Validate OOM recovery
   - Raise `CUDA_SR_TILE` only if needed to reduce memory pressure
   - If an SR request OOMs, confirm the request fails cleanly
   - Submit another SR request afterward and confirm it succeeds after a cold reload

If VRAM is tight, increase `CUDA_SR_TILE` first. Start with `0`, then try values such as `256` or `512`.

See [docs/CUDA_SUPERRES.md](docs/CUDA_SUPERRES.md) for the focused operator guide.

---

## Development

### Project Structure

```
dream-lab/
├── server/
│   ├── lcm_sr_server.py      # Main FastAPI application
│   ├── mode_config.py         # Mode configuration manager
│   ├── model_routes.py        # Model management API
│   ├── file_watcher.py        # Hot-reload support
│   └── ...
├── backends/
│   ├── worker_pool.py         # Extensible job queue
│   ├── worker_factory.py      # Auto-detection & worker creation
│   ├── model_registry.py      # VRAM tracking
│   ├── cuda_worker.py         # GPU workers (SD1.5/SDXL)
│   └── rknn_worker.py         # NPU workers (LCM-SD)
├── utils/
│   └── model_detector.py      # Model inspection utilities
├── tests/
│   ├── test_model_registry.py
│   ├── test_worker_factory.py
│   └── test_worker_pool.py
├── modes.yaml.example         # Example mode configuration
└── docs/
    └── DYNAMIC_MODEL_LOADING.md
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific suite
pytest tests/test_worker_pool.py -v

# With coverage
pytest tests/ --cov=backends --cov=server --cov-report=html
```

See `tests/README.md` for detailed testing guide.

### Adding Custom Job Types

The job queue is extensible:

```python
from backends.worker_pool import get_worker_pool, CustomJob

def my_preprocessing(image_path):
    # Your logic here
    return processed_image

# Queue custom job
pool = get_worker_pool()
job = CustomJob(handler=my_preprocessing, args=("/path/to/image.png",))
future = pool.submit_job(job)
result = future.result()
```

See `docs/DYNAMIC_MODEL_LOADING.md` for complete documentation.

---

## Known Issues & Limitations

### RKNN Backend
1. **VAE decode latency**: Slower than other stages (known RKNN behavior)
2. **Toolkit version sensitivity**: Some versions cause precision loss
3. **Multi-resolution conversion**: May fail in single pass (toolkit limitation)
4. **Memory constraints**: Embedded devices have limited VRAM

### GPU Backend
1. **First load slow**: Initial model load takes longer (PyTorch/CUDA warmup)
2. **VRAM fragmentation**: Long-running servers may need occasional restart
3. **LoRA compatibility**: Not all LoRAs work with all models (user responsibility)

---

## Roadmap

- [x] Dynamic model loading system
- [x] LoRA support with per-mode configuration
- [x] VRAM tracking and management
- [x] Hot-reload configuration
- [x] Extensible job queue
- [x] Graceful mode switching
- [ ] ControlNet support (in progress)
- [ ] Multi-model batching
- [ ] Distributed inference (multiple GPUs/NPUs)
- [ ] Image-to-image pipelines
- [ ] Inpainting support
- [ ] Model caching strategies
- [ ] Enhanced RKNN optimizations

---

## References & Credits

### Base Models
- [LCM-Dreamshaper-V7-ONNX](https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

### Frameworks & Libraries
- [Optimum LCM Pipeline](https://github.com/huggingface/optimum/blob/main/optimum/pipelines/diffusers/pipeline_latent_consistency.py)
- [Diffusers](https://github.com/huggingface/diffusers)
- [FastAPI](https://fastapi.tiangolo.com/)

### RKNN Resources
- [RK3588 Stable Diffusion GPU](https://github.com/happyme531/RK3588-stable-diffusion-GPU)
- [RKNN Super-Resolution Demo](https://github.com/Mario5Gray/rknn-superresolution)
- [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)

### Super-Resolution
- [ONNX Model Zoo - Sub-Pixel CNN](https://github.com/onnx/models/tree/main/validated/vision/super_resolution/sub_pixel_cnn_2016)

---

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

Focus areas:
- RKNN performance optimizations
- New generative workflows
- Backend extensibility
- Documentation improvements

---

**Dream Lab** - Where edge AI meets creative experimentation.
