# Worker Selection Guide

This guide covers CUDA worker selection after you have already chosen `BACKEND=cuda`.

Backend selection itself is explicit. The server does not infer CUDA vs RKNN vs MLX vs CPU from local hardware.

Once `BACKEND=cuda` is selected, the correct worker (SD1.5 or SDXL) is chosen from
the model's **neutral family**, resolved from detector architecture facts.

## How It Works

Model selection is driven by the mode system (`conf/modes.yml`), not by ambient
`MODEL`/`MODEL_ROOT` environment variables. When the pool loads a mode:

1. `WorkerPool._load_mode` calls `backends.model_resolution.resolve_model(model_path, mode)`,
   which detects the model **once**, resolves its neutral family with
   `backends.family_profiles.resolve_family(...)`, overlays the mode's capability
   fields, and emits a portable `ResolvedModel` plus a node-local
   `LocalModelBinding`.
2. The family is resolved from detector-owned architecture facts — `base_arch == "unet"`
   and `cross_attention_dim` (`{768, 1024}` → `sd15`, `{1280, 2048}` → `sdxl`).
   It is **not** taken from `checkpoint_variant`, which mode policy can overlay.
3. `backends.worker_factory.create_cuda_worker(worker_id, resolved, binding)` looks up
   the canonical CUDA cell for `resolved.profile.family_id` in
   `backends/platforms/cuda_bindings.py::CUDA_FAMILY_BINDINGS`, imports the cell's
   `worker_ref` lazily with `importlib` (only inside the factory), and instantiates:
   - `DiffusersCudaWorker` for family `sd15` (SD1.5, SD2.0, SD2.1)
   - `DiffusersSDXLCudaWorker` for family `sdxl` (SDXL Base, Refiner)

A known family with no CUDA cell raises `UnsupportedFamilyError`; an unresolvable
model raises `FamilyResolutionError`. Neither path imports Torch/Diffusers worker
code until `create_cuda_worker` runs, so server boot, status reads, and rejected
requests stay import-clean.

Family resolution and worker selection are encapsulated in the `backends` package
(`family_profiles`, `model_resolution`, `platforms/cuda_bindings`, `worker_factory`).

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

**Solution**: verify the family is one the CUDA platform binds (`sd15`, `sdxl`).

See [MODEL_DETECTOR_EXTENSIBLE.md](MODEL_DETECTOR_EXTENSIBLE.md) for the detection
system and `docs/superpowers/specs/2026-07-16-hunyuandit-family-profile-design.md`
for the neutral-family resolution and platform-binding design.
