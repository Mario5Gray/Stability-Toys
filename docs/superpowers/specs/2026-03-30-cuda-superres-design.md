# CUDA Super-Resolution Design

## Goal

Add CUDA-backed super-resolution that supports both existing entry points:

- postprocess on `/generate` when `superres=true`
- standalone `/superres`

The client-facing API must remain unchanged. Backend selection should stay transparent by deployment environment: RKNN deployments keep the current RKNN path, CUDA deployments use a CUDA SR implementation.

## Non-Goals

- No diffusion-based super-resolution in v1
- No frontend changes
- No public API changes for CUDA-specific tuning
- No sharing of lifecycle with txt2img/img2img CUDA workers

## Architecture

Introduce a backend-agnostic super-resolution service boundary between the HTTP layer and the concrete upscaler implementation.

`server/lcm_sr_server.py` should depend on a small interface shaped around the current needs, for example:

- `upscale_bytes(image_bytes, magnitude, out_format, quality) -> bytes`

Two concrete implementations sit behind that interface:

- `RknnSuperResService`
- `CudaSuperResService`

Backend selection remains transparent:

- RKNN deployments instantiate `RknnSuperResService`
- CUDA deployments instantiate `CudaSuperResService`

Both `/generate` postprocess and `/superres` call the same service interface.

This removes RKNN-specific knowledge from the request handlers and keeps CUDA SR isolated from the existing generation worker stack.

## CUDA SR Worker

CUDA super-resolution uses a separate dedicated CUDA worker/service with its own lifecycle.

Primary engine for v1:

- RealESRGAN-style pixel upscaler

Reasoning:

- practical first implementation
- lower integration risk than diffusion upscalers
- fits the existing `/generate` postprocess and `/superres` contracts

### Lifecycle

The CUDA SR worker is lazy-loaded on first SR request.

Reasons:

- super-resolution is optional
- CUDA deployments are already VRAM-constrained
- keeping SR unloaded until needed avoids reserving memory that generation may need

The worker should remain resident after first load until:

- explicit unload
- idle cleanup
- OOM recovery

### Configuration

CUDA SR tuning remains env/config-driven in v1, not request-driven.

Expected configuration shape:

- `CUDA_SR_MODEL`
- `CUDA_SR_TILE`
- `CUDA_SR_FP16`

Tiling is first-class because it is the main low-VRAM escape hatch.

### Failure Handling

If CUDA SR OOMs:

- fail the request cleanly
- unload the CUDA SR worker
- let the next SR request cold-reload it

This follows the same operational rule already used for CUDA generation recovery: do not trust a partially poisoned in-memory pipeline after OOM.

## Request Behavior

Client-facing behavior remains stable.

### `/generate`

Continue to support:

- `superres=true`
- `superres_format`
- `superres_quality`
- `superres_magnitude`

When enabled, generation output is passed through the selected SR service before response/storage handling continues.

### `/superres`

Continue to support the current multipart upload contract and output options.

### Response Semantics

Keep current headers and response behavior stable so:

- frontend code does not need CUDA-specific branching
- compat endpoints remain unchanged

`superres_magnitude` retains its existing meaning in v1: repeated upscale passes.

## Component Boundaries

### HTTP Layer

`server/lcm_sr_server.py` is responsible for:

- request validation
- selecting the active SR service
- calling the service
- preserving current response/header behavior

It should not contain CUDA- or RKNN-specific model lifecycle logic.

### RKNN Service

Wrap the existing RKNN SR implementation behind the new interface with no behavior change.

### CUDA Service

Responsible for:

- lazy model load
- CUDA device placement
- tiling/fp16 configuration
- byte-in / byte-out upscale execution
- unload-on-OOM behavior

## Testing Strategy

### Unit Tests

Add unit coverage for:

- SR service selection by backend
- shared interface behavior
- CUDA lazy-load behavior
- CUDA env/config parsing
- CUDA OOM unload behavior
- both `/generate superres=true` and `/superres` routing through the same service

### Integration/Manual Tests

Manual CUDA verification is required because:

- model weights are large
- image quality matters
- GPU hardware is required

Manual acceptance for CUDA:

1. `BACKEND=cuda` and no SR request: server starts without preloading CUDA SR
2. first `/superres` request loads the CUDA SR worker and returns an image
3. `/generate` with `superres=true` routes through the same CUDA SR service
4. an induced or real CUDA SR OOM unloads the SR worker and a later request reloads cleanly

## Rollout Boundaries

v1 is complete when:

- CUDA deployments support `/generate` postprocess SR
- CUDA deployments support standalone `/superres`
- API shape remains unchanged
- RKNN behavior remains unchanged
- CUDA SR is lazy-loaded and unloads on OOM

Future work, intentionally excluded from this design:

- diffusion-based SR
- frontend SR backend selection
- per-request CUDA SR tuning fields
- sharing CUDA SR model/process with txt2img/img2img workers
