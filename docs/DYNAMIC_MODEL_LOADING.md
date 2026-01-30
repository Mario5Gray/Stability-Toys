# Dynamic Model Loading System

Complete guide to the extensible model loading system with hot-swappable modes.

## Overview

The dynamic model loading system allows you to:
- Define multiple model configurations (modes) in YAML
- Switch between models at runtime without restarting the server
- Load/unload models dynamically based on VRAM availability
- Queue generation requests that automatically switch modes
- Hot-reload configuration changes without downtime
- Extend the job queue with custom job types

## Quick Start

### 1. Create modes.yaml

Copy the example configuration:

```bash
cp modes.yaml.example modes.yaml
```

Edit to define your modes:

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

  sd15-fast:
    model: sd15/dreamshaper-lcm.safetensors
    default_size: "512x512"
    default_steps: 4
    default_guidance: 1.0
```

### 2. Start Server

The server automatically loads the default mode on startup:

```bash
./runner.sh
```

**Note:** `MODEL_ROOT` and `MODEL` environment variables are deprecated when using modes.yaml.

### 3. Generate with Mode

```bash
# Use default mode
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape"}'

# Switch to specific mode
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "sdxl-general", "prompt": "a portrait"}'
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Server                      │
│  ┌──────────────────────────────────────────────┐  │
│  │         ModeConfigManager                     │  │
│  │  - Loads modes.yaml                          │  │
│  │  - Validates configurations                  │  │
│  │  - Hot-reload support                        │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │         WorkerPool (Extensible Queue)        │  │
│  │  - Manages worker lifecycle                  │  │
│  │  - Queues: Generation, ModeSwitch, Custom    │  │
│  │  - Automatic mode switching                  │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │         ModelRegistry (VRAM Tracker)         │  │
│  │  - Tracks loaded models                      │  │
│  │  - Real-time VRAM monitoring                 │  │
│  │  - No artificial limits                      │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │         FileWatcher (Hot-reload)             │  │
│  │  - inotify/FSEvents (no polling)             │  │
│  │  - Auto-reload modes.yaml                    │  │
│  │  - SIGHUP signal support                     │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Job Queue (Extensible)

The worker pool uses an extensible job queue that supports:

**Built-in Job Types:**
- `GenerationJob` - Image generation
- `ModeSwitchJob` - Model switching
- `CustomJob` - Arbitrary functions

**Adding Custom Jobs:**

```python
from backends.worker_pool import get_worker_pool, CustomJob

# Define your function
def my_custom_task(arg1, arg2):
    # Your logic here
    return result

# Queue it
pool = get_worker_pool()
job = CustomJob(handler=my_custom_task, args=(val1, val2))
future = pool.submit_job(job)
result = future.result()
```

## API Reference

### Management Endpoints

#### GET /api/models/status

Get current model status and VRAM statistics.

**Response:**
```json
{
  "current_mode": "sdxl-general",
  "queue_size": 2,
  "vram": {
    "device": "NVIDIA GeForce RTX 3090",
    "total_gb": 24.0,
    "used_gb": 12.5,
    "available_gb": 11.5,
    "usage_percent": 52.1,
    "models_loaded": 1,
    "models": [
      {
        "name": "sdxl-general",
        "model_path": "/models/sdxl/sdxl-base-1.0.safetensors",
        "vram_gb": 12.5,
        "loras": []
      }
    ]
  }
}
```

#### GET /api/modes

List all available modes from configuration.

**Response:**
```json
{
  "default_mode": "sdxl-general",
  "modes": {
    "sdxl-general": {
      "model": "sdxl/sdxl-base-1.0.safetensors",
      "loras": [],
      "default_size": "1024x1024",
      "default_steps": 30,
      "default_guidance": 7.5
    }
  }
}
```

#### POST /api/modes/switch

Switch to a different mode.

**Request:**
```json
{
  "mode": "sdxl-portrait"
}
```

**Response:**
```json
{
  "status": "queued",
  "from_mode": "sdxl-general",
  "to_mode": "sdxl-portrait",
  "message": "Mode switch queued. Will switch after 2 pending jobs."
}
```

#### POST /api/modes/reload

Manually reload modes.yaml configuration.

**Response:**
```json
{
  "status": "reloaded",
  "modes_count": 5,
  "modes": ["sdxl-general", "sdxl-portrait", "sd15-fast"],
  "default_mode": "sdxl-general"
}
```

#### GET /api/vram

Get detailed VRAM statistics.

**Response:**
```json
{
  "device": "NVIDIA GeForce RTX 3090",
  "total_gb": 24.0,
  "used_gb": 12.5,
  "available_gb": 11.5,
  "usage_percent": 52.1,
  "models_loaded": 1,
  "models": [...]
}
```

### Generation Endpoint

#### POST /generate

Generate image with optional mode switching.

**Request:**
```json
{
  "mode": "sdxl-portrait",
  "prompt": "a beautiful portrait",
  "size": "896x1152",
  "num_inference_steps": 35,
  "guidance_scale": 8.0,
  "seed": 42
}
```

**Behavior:**
1. If `mode` specified and different from current → queue mode switch
2. Apply mode's default settings if not overridden in request
3. Queue generation job
4. Return image with `X-Mode` header showing which mode was used

**Response Headers:**
- `X-Mode`: Mode used for generation
- `X-Seed`: Seed used
- `X-SuperRes`: Whether super-resolution was applied

## Configuration Reference

### modes.yaml Structure

```yaml
# Global paths
model_root: /models              # Required
lora_root: /models/loras          # Optional, defaults to model_root

# Default mode on startup
default_mode: sdxl-general        # Required

# Mode definitions
modes:
  mode_name:
    model: path/to/model.safetensors   # Required, relative to model_root
    loras:                              # Optional
      - path: lora1.safetensors
        strength: 0.8
        adapter_name: custom_name       # Optional
    default_size: "1024x1024"           # Optional, defaults to 512x512
    default_steps: 30                   # Optional, defaults to 4
    default_guidance: 7.5               # Optional, defaults to 1.0
    metadata:                           # Optional custom metadata
      description: "..."
      any_key: "any_value"
```

### LoRA Formats

**Simple format (path only):**
```yaml
loras:
  - lora1.safetensors
  - lora2.safetensors
```

**Full format (all options):**
```yaml
loras:
  - path: lora1.safetensors
    strength: 0.8
    adapter_name: portrait
  - path: lora2.safetensors
    strength: 1.2
    adapter_name: detail
```

## Hot-Reload

The configuration is automatically reloaded when modes.yaml changes.

### Automatic (File Watcher)

The server uses `watchdog` library with native OS events:
- **Linux**: inotify (no polling)
- **macOS**: FSEvents (no polling)
- **Windows**: ReadDirectoryChangesW

Changes are detected within 1 second and validated before applying.

### Manual (SIGHUP)

```bash
# Find server PID
ps aux | grep "python.*server.run"

# Send SIGHUP signal
kill -HUP <PID>
```

### API Reload

```bash
curl -X POST http://localhost:4200/api/modes/reload
```

## VRAM Management

### No Artificial Limits

The system uses ALL available VRAM:
- No percentage-based budgets
- No artificial reservations
- Uses `torch.cuda.memory_allocated()` for accurate tracking
- Automatic garbage collection on model unload

### VRAM Estimation

File size × 1.2 = estimated VRAM usage

Actual usage measured after loading.

### Model Unloading

When switching modes:
1. Current worker is destroyed
2. `torch.cuda.empty_cache()` called
3. Garbage collection forced
4. New worker created with new model

## Migration from Legacy System

### Old Configuration (Deprecated)

```bash
export MODEL_ROOT=/models
export MODEL=sdxl-base.safetensors
export LORA_ROOT=/models/loras
```

### New Configuration (modes.yaml)

```yaml
model_root: /models
lora_root: /models/loras
default_mode: my-mode

modes:
  my-mode:
    model: sdxl-base.safetensors
    default_size: "1024x1024"
```

### Backward Compatibility

If `modes.yaml` not found:
- Server falls back to legacy `MODEL_ROOT`/`MODEL` behavior
- Mode system features disabled
- Warning logged on startup

## Examples

### Example 1: Multi-Model Workflow

```python
import requests

base_url = "http://localhost:4200"

# Generate with SDXL
response = requests.post(f"{base_url}/generate", json={
    "mode": "sdxl-general",
    "prompt": "a photorealistic landscape"
})

# Generate with SD1.5 (automatically switches)
response = requests.post(f"{base_url}/generate", json={
    "mode": "sd15-fast",
    "prompt": "anime character"
})

# Check which mode is currently loaded
status = requests.get(f"{base_url}/api/models/status").json()
print(f"Current mode: {status['current_mode']}")
print(f"VRAM used: {status['vram']['used_gb']} GB")
```

### Example 2: Custom Job Queue

```python
from backends.worker_pool import get_worker_pool, CustomJob

def my_preprocessing_task(image_path):
    # Your custom logic
    processed = do_something(image_path)
    return processed

# Queue custom job
pool = get_worker_pool()
job = CustomJob(
    handler=my_preprocessing_task,
    args=("/path/to/image.png",)
)
future = pool.submit_job(job)
result = future.result()
```

### Example 3: Mode-Specific Workflows

```yaml
# modes.yaml
modes:
  portraits:
    model: sdxl/sdxl-base.safetensors
    loras:
      - path: portrait-enhancer.safetensors
        strength: 0.8
    default_size: "896x1152"  # Portrait aspect
    default_steps: 40

  landscapes:
    model: sd15/realisticvision.safetensors
    loras:
      - path: landscape-enhancer.safetensors
        strength: 1.0
    default_size: "768x512"  # Landscape aspect
    default_steps: 30
```

## Troubleshooting

### modes.yaml not found

**Error:** `FileNotFoundError: modes.yaml not found`

**Solution:** Create modes.yaml in project root

```bash
cp modes.yaml.example modes.yaml
```

### Mode not found

**Error:** `Mode 'xyz' not found`

**Solution:** Check mode name in modes.yaml

```bash
curl http://localhost:4200/api/modes
```

### VRAM out of memory

**Error:** `CUDA out of memory`

**Solution:**
1. Use smaller models
2. Reduce batch size
3. Check VRAM usage: `GET /api/vram`
4. Switch to lighter mode

### Mode switch timeout

**Error:** `Mode switch timed out`

**Solution:**
- Wait for current jobs to complete
- Check queue size: `GET /api/models/status`
- Long queue = longer switch time

## Performance

### Mode Switch Time

- **Unload old model**: 1-2 seconds
- **Load new model**: 5-15 seconds (depends on model size)
- **Total**: ~10-20 seconds for SDXL ↔ SD1.5 switch

### Hot-Reload Time

- **File change detection**: <1 second (inotify/FSEvents)
- **Config validation**: <100ms
- **Does not interrupt**: In-flight generations continue

### Queue Throughput

- Single worker (CUDA): 1 generation at a time
- Queue size: Default 64, configurable via `QUEUE_MAX`
- Mode switches queued like regular jobs

## Best Practices

1. **Use modes.yaml for all model configurations**
   - Centralized configuration
   - Easy to manage multiple models
   - Hot-reload support

2. **Name modes descriptively**
   - `sdxl-portrait` not `mode1`
   - `sd15-anime-fast` not `xyz`

3. **Set appropriate defaults per mode**
   - Portrait: vertical aspect ratio
   - Landscape: horizontal aspect ratio
   - Fast: low steps, low guidance

4. **Monitor VRAM usage**
   - Check `/api/vram` periodically
   - Large models may need mode switching
   - Use smaller models for high-throughput

5. **Use custom jobs for extensibility**
   - Pre/post-processing
   - Batch operations
   - Integration with other systems

## See Also

- [modes.yaml.example](../modes.yaml.example) - Example configuration
- [Model Detection](MODEL_DETECTOR_EXTENSIBLE.md) - Auto-detection system
- [Worker Selection](WORKER_SELECTION.md) - SD1.5 vs SDXL selection
