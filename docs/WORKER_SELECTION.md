# Worker Selection Guide

This guide covers CUDA worker selection after you have already chosen `BACKEND=cuda`.

Backend selection is explicit. The server does not infer CUDA, RKNN, MLX, CPU, or MPS from local hardware.

Once `BACKEND=cuda` is selected, the worker is chosen from
the model's **neutral family**, resolved from detector architecture facts.

## How It Works

Model selection is driven by the mode system (`conf/modes.yml`), not by ambient
`MODEL`/`MODEL_ROOT` environment variables. When the pool loads a mode:

1. `Governor._load_mode` calls `backends.model_resolution.resolve_model(model_path, mode)`,
   which detects the model **once**, resolves its neutral family with
   `backends.family_profiles.resolve_family(...)`, overlays the mode's capability
   fields, and emits a portable `ResolvedModel` plus a node-local
   `LocalModelBinding`.
2. The family is resolved from detector-owned architecture facts — `base_arch == "unet"`
   and `cross_attention_dim` (`{768, 1024}` → `sd15`, `{1280, 2048}` → `sdxl`).
   It is **not** taken from `checkpoint_variant`, which mode policy can overlay.
3. The backend provider supplies its family binding table to
   `backends.worker_factory.create_bound_worker`. The factory looks up
   `resolved.profile.family_id`, imports the selected cell's `worker_ref` lazily,
   and instantiates:
   - `DiffusersCudaWorker` for family `sd15` (SD1.5, SD2.0, SD2.1)
   - `DiffusersSDXLCudaWorker` for family `sdxl` (SDXL Base, Refiner)
   - `DiffusersHunyuanDiTCudaWorker` for family `hunyuandit`

A known family without a selected backend cell raises `UnsupportedFamilyError`.
An unresolvable model raises `FamilyResolutionError`. Binding reads and rejected
requests do not import CUDA worker code. `create_cuda_worker` remains a
compatibility wrapper for legacy CUDA callers.

Family resolution and worker selection are encapsulated in the `backends` package
(`family_profiles`, `model_resolution`, `platforms/cuda_bindings`, `worker_factory`).

`CUDAProvider` keeps the `CudaGenerationRuntime` compatibility class. Shared queue
behavior lives in `backends/platforms/pool_runtime.py::PoolGenerationRuntime`.

## MPS Framework

`BACKEND=mps` uses `MPSProvider`, the shared pool runtime, and MPS memory policy.
`MPS_FAMILY_BINDINGS` binds `sd15` to `backends.mps_worker.DiffusersMpsWorker`
for LCM text-to-image. Its cell claims no img2img, no ControlNet, and no
combined operation. `sdxl` and `hunyuandit` stay unbound.
The provider reports generation support as true after live acceptance.
It reports super-resolution and model-registry statistics as false.
Server startup continues without a model when the default mode has no MPS binding.
Mode switches return HTTP 422 with `family 'X' has no MPS platform binding`.
In-process workers use the provider factory. Subprocess workers use the module-level
`create_mps_worker` entry point. Both read the same binding table.
MPS super-resolution remains a placeholder. MLX behavior does not change.
For Mac setup, MPS limits, the host probe, and evidence, see [MPS_BACKEND.md](MPS_BACKEND.md).

## Configuration

Models are declared per mode in `conf/modes.yml` under `model_root` + each mode's
`model:` path. Select the backend and mode-config location via environment:

```bash
export BACKEND=cuda
export MODE_CONFIG_PATH=conf        # directory containing modes.yml
```

Switch modes at runtime through the mode API (`/api/modes/switch`); the pool
re-resolves and republishes one active snapshot per load.

## Example Configuration

### Docker with the mode config mounted

```bash
docker run --rm --gpus all --privileged \
  -v /models:/models:ro \
  -v $(pwd)/conf:/app/conf:ro \
  -e BACKEND=cuda \
  -p 4200:4200 \
  lcm-sd:latest
```

The server logs show the resolved family and selected worker on mode load:

```
[WorkerPool] Loading mode: SDXL
[WorkerFactory] Created backends.cuda_worker.DiffusersSDXLCudaWorker for family 'sdxl' (worker 0)
[WorkerPool] Mode 'SDXL' loaded successfully (VRAM: 6.10 GB, epoch=1)
```

## Troubleshooting

### `FamilyResolutionError` on load

**Problem**: the model's detector facts do not match exactly one neutral family
(e.g. a transformer/DiT checkpoint, or an ambiguous cross-attention dim).

**Solution**: confirm the model is a supported SD1.5/SD2.x/SDXL UNet checkpoint.
Inspect detector facts directly:

```bash
python -m utils.model_detector /models/your-model.safetensors
```

### `UnsupportedFamilyError` on load

**Problem**: the family resolved, but no CUDA cell is registered for it in
`CUDA_FAMILY_BINDINGS`.

**Solution**: verify the family is one the CUDA platform binds (`sd15`, `sdxl`, `hunyuandit`).
MPS binds `sd15` for LCM text-to-image only.

See [MODEL_DETECTOR_EXTENSIBLE.md](MODEL_DETECTOR_EXTENSIBLE.md) for the detection
system and `docs/superpowers/specs/2026-07-16-hunyuandit-family-profile-design.md`
for the neutral-family resolution and platform-binding design.
