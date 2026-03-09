# VRAM Management

How GPU memory is allocated, held, and released during model load, generation,
and unload. Includes known failure modes and what the logging tells you.

---

## How PyTorch Uses VRAM

There are two distinct numbers to understand:

- **Allocated** (`torch.cuda.memory_allocated()`) — bytes actively held by live
  tensors. This is the true working set.
- **Reserved** (`torch.cuda.memory_reserved()`) — bytes the process has claimed
  from the CUDA driver, including the allocator's free-block cache. This is what
  `nvidia-smi` shows.

The gap between reserved and allocated is PyTorch's internal cache: blocks that
were freed but kept by the allocator for fast future reuse. This cache is never
returned to the OS until `torch.cuda.empty_cache()` is called explicitly.

On a model that occupies 6 GB, it is normal to see `nvidia-smi` reporting
10–14 GB after several generations. The extra is cache, not a leak.

---

## Model Lifecycle

### Load

`WorkerPool._load_mode(mode_name)` is the single entry point for loading.

1. Unloads the current worker if one is running (see Unload below)
2. Snapshots `get_used_vram()` (reserved bytes at driver level)
3. Calls the worker factory — `DiffusersCudaWorker` or `DiffusersSDXLCudaWorker`
   — which calls `from_pretrained` / `from_single_file` and moves the pipeline
   to the CUDA device
4. Snapshots VRAM again; logs the delta as `model_delta`
5. Registers the model in `ModelRegistry`
6. Starts the worker thread

**Log line on success:**
```
[WorkerPool] VRAM after load: allocated=6.21GB reserved=6.84GB total=24.00GB model_delta=6.21GB
```

**On failure:** the worker factory throws, partial GPU allocations are cleaned up
via `del self._worker`, `gc.collect()`, `torch.cuda.empty_cache()`, and
`_current_mode` is set to `None`. The server stays up.

### Unload

`WorkerPool._unload_current_worker()` is called on mode switch, idle eviction,
and shutdown.

1. Calls `ModelRegistry.unregister_model()`
2. `del self._worker` — drops the Python reference; CPython frees immediately
   because refcount hits zero
3. `gc.collect()` — catches any cycles the refcount missed
4. `torch.cuda.empty_cache()` — returns the allocator cache to the CUDA driver

**Log line on unload:**
```
[WorkerPool] Worker unloaded — allocated=0.01GB reserved=0.12GB ...
```

If `reserved` is significantly higher than `allocated` after unload, the cache
was not fully cleared. This is normal for small residuals but worth watching if
`reserved` stays above 1 GB.

---

## Generation

Each call to `run_job` runs a diffusion forward pass. The key VRAM consumers
during generation are:

| Component | Approximate cost (SDXL fp16, 1024×1024) |
|---|---|
| Model weights (resident) | ~6.5 GB |
| UNet activations per step | ~1–2 GB |
| VAE decode | ~0.5 GB |
| Peak (all live simultaneously) | ~9–10 GB |

### Cleanup after generation

After extracting the output image from the pipeline result:

```python
img = out.images[0]
del out                     # drop pipeline output — frees activation tensors
torch.cuda.empty_cache()    # return freed blocks to CUDA driver
```

Without `del out` the activation tensors stay alive until the next GC sweep.
Without `empty_cache()` the blocks stay in the allocator cache and show as
used in `nvidia-smi`.

### `run_job_with_latents` — double pipeline run

This method runs the full pipeline **twice**: once via `run_job()` for the image,
and once more for the latent pass. The first `run_job` call now calls
`empty_cache()` before returning, so the second pass starts with a clean
allocator. Each intermediate tensor from the latent pass is also explicitly
deleted:

```python
del out_lat   # pipeline output
del lat       # raw latent tensor
del lat_8     # downsampled float32 intermediate
torch.cuda.empty_cache()
```

Without these, peak VRAM during `run_job_with_latents` can reach
2× model_delta + 2× activation overhead.

---

## Idle Eviction

`WorkerPool` runs a background watchdog thread that evicts the loaded model
after `MODEL_IDLE_TIMEOUT_SECS` seconds of inactivity (default 300 s, 0 to
disable). The check runs every `MODEL_IDLE_CHECK_INTERVAL_SECS` seconds
(default 30 s).

When evicted, `_unload_current_worker()` runs on the worker thread (via a
queued `CustomJob`) so it is always serialised with generation. The mode name
is retained in `_current_mode` so the next incoming job can demand-reload the
same model.

**Demand reload:** if a job arrives and `_worker is None` but `_current_mode`
is set, the worker thread calls `_load_mode(_current_mode)` before executing
the job. The job blocks in the queue while the reload runs.

---

## What the Logs Tell You

| Log fragment | Meaning |
|---|---|
| `allocated=X reserved=Y` with Y >> X after load | Normal PyTorch cache overhead |
| `allocated=X reserved=Y` with Y >> X after unload | `empty_cache()` ran but some blocks still cached; non-zero X after unload means something holds a tensor reference |
| `model_delta=0.00GB` | VRAM measurement did not capture the load (possible if `get_used_vram` uses driver-level stats that update asynchronously) |
| `reserved>allocated` hint in unload log | Expected; call `empty_cache()` manually via `/api/models/unload` if you need to reclaim it immediately |

---

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `MODEL_IDLE_TIMEOUT_SECS` | `300` | Seconds idle before eviction; `0` disables |
| `MODEL_IDLE_CHECK_INTERVAL_SECS` | `30` | Watchdog poll interval |
| `VRAM_RESERVE_GB` | `1.0` | Headroom threshold for future evict-lru logic |
| `CUDA_DTYPE` | `fp16` | Model weight dtype (`fp16`, `bf16`, `fp32`) |
| `CUDA_DEVICE` | `cuda:0` | Target device |
| `CUDA_ENABLE_XFORMERS` | `0` | Memory-efficient attention (reduces activation VRAM ~30%) |
| `CUDA_ATTENTION_SLICING` | `0` | Further reduces peak activation VRAM at cost of speed |

---

## Known Issues / Future Work

- `run_job_with_latents` still runs the pipeline twice. The latent pass is a
  full generation at the same resolution. If only a fingerprint hash is needed,
  a lower-resolution or fewer-step pass would reduce the cost significantly.
- `_img2img_pipe` is lazily created and shares weights with `self.pipe` (no
  extra model VRAM), but is never explicitly deleted when the worker unloads.
  CPython's refcount handles this correctly in practice, but explicit teardown
  in a `__del__` or `cleanup()` method would be cleaner.
- No OpenTelemetry spans exist on the backend. Load/unload/generation durations
  are only visible in logs. Spans around `_load_mode`, `_unload_current_worker`,
  and `run_job` would make these visible in Grafana without log scraping.
