# SDXL Integration Example

This document shows how to integrate the SDXL worker into the existing server infrastructure with minimal changes.

## Option 1: Environment Variable Toggle (Recommended)

This approach allows switching between SD1.5 and SDXL via environment variable.

### 1. Modify `server/lcm_sr_server.py`

```python
# In PipelineService.__init__() around line 190:

# 3) create exactly one worker for cuda, N for rknn
for i in range(self.num_workers):
    if use_cuda:
        # NEW: Check if SDXL mode is enabled
        use_sdxl = os.environ.get("USE_SDXL", "0").lower() in ("1", "true", "yes", "on")

        if use_sdxl:
            from backends.cuda_worker import DiffusersSDXLCudaWorker
            logger.info(f"Creating SDXL CUDA worker {i}")
            w = DiffusersSDXLCudaWorker(worker_id=i)
        else:
            from backends.cuda_worker import DiffusersCudaWorker
            logger.info(f"Creating SD1.5 CUDA worker {i}")
            w = DiffusersCudaWorker(worker_id=i)  # i will always be 0
    else:
        from backends.rknn_worker import RKNNPipelineWorker
        # ... existing RKNN code ...
```

### 2. Update Environment Files

Create `env.sdxl`:
```bash
# env.sdxl - SDXL Configuration
USE_SDXL=1
SDXL_MODEL_ROOT=/models/sdxl
SDXL_MODEL=sdxl-1.0-base.safetensors

# CUDA optimizations (highly recommended for SDXL)
CUDA_ENABLE_XFORMERS=1
CUDA_DTYPE=fp16
CUDA_DEVICE=cuda:0

# Server defaults - adjust for SDXL
DEFAULT_SIZE=1024x1024
DEFAULT_STEPS=4
DEFAULT_GUIDANCE=1.0
```

### 3. Run Server with SDXL

```bash
# In runner.sh or docker-compose, add env.sdxl:
docker run --rm -it \
  --env-file env.cuda \
  --env-file env.sdxl \
  --env-file env.custom \
  --gpus=all \
  -p 4200:4200 \
  mitraframe/stability-toys:latest
```

## Option 2: Separate Worker Class

Create a dedicated SDXL service that runs independently.

### 1. Create `sdxl_service.py`

```python
# sdxl_service.py
import os
import queue
import threading
from typing import List
from concurrent.futures import Future

from backends.cuda_worker import DiffusersSDXLCudaWorker
from backends.base import Job

class SDXLPipelineService:
    """Dedicated service for SDXL generation."""

    def __init__(self, num_workers: int = 1, queue_max: int = 32):
        self.num_workers = max(1, num_workers)
        self.q: queue.Queue[Job] = queue.Queue(maxsize=queue_max)
        self.workers: List[DiffusersSDXLCudaWorker] = []
        self.threads: List[threading.Thread] = []
        self._stop = threading.Event()

        # SDXL on CUDA always uses 1 worker (sweet spot)
        if self.num_workers != 1:
            print(f"[SDXL] Forcing num_workers {self.num_workers} -> 1 for optimal performance")
            self.num_workers = 1

        # Create workers
        for i in range(self.num_workers):
            w = DiffusersSDXLCudaWorker(worker_id=i)
            self.workers.append(w)

        # Start threads
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

        print(f"[SDXL] Service initialized with {self.num_workers} worker(s)")

    def submit(self, req, timeout_s: float = 0.25) -> Future:
        fut = Future()
        job = Job(req=req, fut=fut, submitted_at=time.time())
        try:
            self.q.put(job, timeout=timeout_s)
        except queue.Full:
            fut.set_exception(RuntimeError("Queue full"))
        return fut

    def _worker_loop(self, worker_idx: int):
        worker = self.workers[worker_idx]
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if job.fut.cancelled():
                self.q.task_done()
                continue

            try:
                png, seed = worker.run_job(job)
                if not job.fut.done():
                    job.fut.set_result((png, seed))
            except Exception as e:
                if not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                self.q.task_done()

    def shutdown(self):
        self._stop.set()
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if not job.fut.done():
                job.fut.set_exception(RuntimeError("Service shutting down"))
            self.q.task_done()
```

### 2. Add SDXL Endpoint

```python
# In server/lcm_sr_server.py, add:

@app.post("/generate_sdxl", responses={200: {"content": {"image/png": {}}}})
def generate_sdxl(req: GenerateRequest):
    """SDXL-specific generation endpoint."""
    sdxl_service = getattr(app.state, "sdxl_service", None)
    if sdxl_service is None:
        raise HTTPException(status_code=503, detail="SDXL service not enabled")

    fut = sdxl_service.submit(req, timeout_s=0.25)
    try:
        png_bytes, seed = fut.result(timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logger.error(f"SDXL generation failed: {e}", exc_info=True)
        msg = str(e)
        if "Queue full" in msg:
            raise HTTPException(status_code=429, detail="Too many requests (queue full). Try again.")
        raise HTTPException(status_code=500, detail=f"SDXL generation failed: {msg}")

    headers = {
        "Cache-Control": "no-store",
        "X-Seed": str(seed),
        "X-Model": "SDXL",
    }

    return Response(content=png_bytes, media_type="image/png", headers=headers)
```

### 3. Initialize in Lifespan

```python
# In lifespan():

SDXL_ENABLED = os.environ.get("SDXL_ENABLED", "0") in ("1", "true", "yes")

if SDXL_ENABLED:
    try:
        from sdxl_service import SDXLPipelineService
        app.state.sdxl_service = SDXLPipelineService(
            num_workers=1,
            queue_max=32,
        )
        logger.info("SDXL service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize SDXL service: {e}", exc_info=True)
        raise
```

## Option 3: Docker Multi-Service

Run SD1.5 and SDXL as separate containers.

### docker-compose.yml

```yaml
version: '3.8'

services:
  # SD1.5 Service (existing)
  lcm-sd15:
    image: mitraframe/stability-toys:latest
    ports:
      - "4200:4200"
    env_file:
      - env.cuda
      - env.custom
    environment:
      - MODEL_ROOT=/models/sd15
      - MODEL=lcm-sd15.safetensors
      - DEFAULT_SIZE=512x512
    volumes:
      - ./models/sd15:/models/sd15:ro
      - ./store:/store:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  # SDXL Service (new)
  lcm-sdxl:
    image: mitraframe/stability-toys:latest
    ports:
      - "4201:4200"  # Different port
    env_file:
      - env.cuda
      - env.sdxl
      - env.custom
    environment:
      - USE_SDXL=1
      - SDXL_MODEL_ROOT=/models/sdxl
      - SDXL_MODEL=sdxl-1.0-base.safetensors
      - DEFAULT_SIZE=1024x1024
      - CUDA_ENABLE_XFORMERS=1
    volumes:
      - ./models/sdxl:/models/sdxl:ro
      - ./store:/store:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # Same GPU, different process
              capabilities: [gpu]
```

### Usage

```bash
# Start both services
docker-compose up -d

# SD1.5 on port 4200
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat", "size": "512x512"}'

# SDXL on port 4201
curl -X POST http://localhost:4201/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat", "size": "1024x1024"}'
```

## Testing the Integration

### 1. Quick Test

```bash
# Set environment
export USE_SDXL=1
export SDXL_MODEL_ROOT=/path/to/models
export SDXL_MODEL=sdxl-model.safetensors
export CUDA_ENABLE_XFORMERS=1

# Start server
./runner.sh

# Test generation
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a majestic mountain landscape",
    "size": "1024x1024",
    "num_inference_steps": 4,
    "seed": 42
  }' \
  --output test_sdxl.png

# Check result
file test_sdxl.png  # Should show: PNG image data, 1024 x 1024
```

### 2. Browser Test

```javascript
// In browser console or UI
fetch('http://localhost:4200/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'a beautiful sunset over the ocean',
    size: '1024x1024',
    num_inference_steps: 4,
    guidance_scale: 1.0,
    seed: 12345
  })
})
.then(r => r.blob())
.then(blob => {
  const url = URL.createObjectURL(blob);
  const img = document.createElement('img');
  img.src = url;
  document.body.appendChild(img);
});
```

## Performance Benchmarks

Expected performance on different GPUs:

| GPU | VRAM | SD1.5 (512x512) | SDXL (1024x1024) |
|-----|------|-----------------|------------------|
| RTX 3060 | 12GB | ~0.5s | ~2.5s |
| RTX 3080 | 10GB | ~0.3s | ~1.8s |
| RTX 3090 | 24GB | ~0.25s | ~1.2s |
| RTX 4090 | 24GB | ~0.15s | ~0.7s |
| A100 | 40GB | ~0.12s | ~0.5s |

*With 4 LCM steps, xformers enabled, fp16*

## Troubleshooting

### Server Won't Start

```bash
# Check logs
docker logs stability-toys 2>&1 | tail -50

# Common issues:
# 1. Model not found -> Check SDXL_MODEL_ROOT and SDXL_MODEL
# 2. OOM -> Enable xformers: CUDA_ENABLE_XFORMERS=1
# 3. Wrong pipeline -> Ensure USE_SDXL=1 is set
```

### Generation Fails

```bash
# Check GPU memory
nvidia-smi

# If OOM:
export CUDA_ENABLE_XFORMERS=1
export CUDA_DTYPE=fp16
export CUDA_ATTENTION_SLICING=1

# Try smaller resolution
# "size": "768x768" instead of "1024x1024"
```

## Summary

Three integration options:

1. **Environment Toggle** (Recommended)
   - ✅ Simple, single codebase
   - ✅ Easy to switch
   - ❌ Can't run both simultaneously

2. **Separate Service**
   - ✅ Run both models
   - ✅ Independent endpoints
   - ❌ More complex code

3. **Docker Multi-Service**
   - ✅ Complete isolation
   - ✅ Scale independently
   - ✅ Production-ready
   - ❌ More resources needed

**Recommendation**: Start with Option 1 (environment toggle) for development, move to Option 3 (multi-service) for production.
