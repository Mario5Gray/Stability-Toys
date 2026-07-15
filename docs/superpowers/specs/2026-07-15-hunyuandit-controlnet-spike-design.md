# HunyuanDiT ControlNet — Spike Design

**FP:** STABL-ichgkgno (Add HunyuanDiT ControlNet support — new model family)
**Date:** 2026-07-15
**Status:** design approved, pre-plan
**Kind:** throwaway spike (not production code)

---

## Purpose

De-risk the HunyuanDiT ControlNet family integration by proving the stack loads
and generates end-to-end **in the exact production load shape**, on real CUDA,
*before* writing the full-family spec. The spike answers the three open questions
on the FP issue that block confident planning:

1. Does the deployed dependency stack expose a **working** `HunyuanDiTControlNetPipeline`?
2. How much VRAM does the base + control model consume?
3. Does the `from_pipe` shared-base pattern behave (dtype/VAE) the way the SD/SDXL
   workers do?

This spec covers **only the spike**. The full family design (detector, registry,
worker, mode config, acceptance test) is a separate spec, opened only after the
spike passes.

## Background — why a spike, and what is already in place

### The copy-paste seam is already landed
The FP issue proposed a "prep option": introduce `_CONTROL_IMAGE_KWARG` + a shared
`_build_controlnet_kwargs` helper on a worker base class so adding HunyuanDiT is
purely additive. **This already exists** — it came free with the combined
img2img+ControlNet work (STABL-ztaxgbhv):

- `CudaWorkerBase._CONTROL_IMAGE_KWARG = "image"` — `backends/cuda_worker.py:153`
- shared `CudaWorkerBase._build_controlnet_kwargs(...)` — `backends/cuda_worker.py:652`,
  already supporting an `image_kwarg` override (used by the img2img+ControlNet path
  with `"control_image"`).

So the production add reduces to: a new `DiffusersHunyuanDiTCudaWorker(CudaWorkerBase)`
subclass with `_CONTROL_IMAGE_KWARG = "control_image"`, a family-specific
`_build_controlnet_pipe`, plus detector/registry/mode-config entries. No shared
assembly changes, no re-hardcoded string keys. The spike does not need to touch or
prove this seam — it is only validating the *upstream* pipeline stack.

### The one per-family divergence
The control map (canny/depth/pose PNG) is the same object across families; only the
kwarg name differs:

| Pipeline | control-map kwarg |
| --- | --- |
| `StableDiffusionControlNetPipeline` | `image=` |
| `StableDiffusionXLControlNetPipeline` | `image=` |
| `HunyuanDiTControlNetPipeline` | `control_image=` |

The spike passes the control map as `control_image=`.

### Dependency risk observed locally (mac CPU env — signal, not gate)
On the local Miniforge base env (`diffusers 0.37.0`, `transformers 5.10.2`):

- `HunyuanDiTControlNetPipeline` / `HunyuanDiT2DControlNetModel` **classes exist**
  (so open question #1's "may require a diffusers bump" is likely moot — the class
  shipped ~0.30, and 0.37 is installed).
- **BUT the import chain fails**: importing the pipeline raises
  `Could not import module 'BertModel'`, and `from transformers import BertModel`
  fails directly under transformers 5.10.2. HunyuanDiT's `text_encoder` is a BERT
  and `text_encoder_2` is an mT5/T5.

The mac CPU env is **not** the deployment target (ControlNet is CUDA-only in this
project), so this is a signal, not a verdict. Confirming the CUDA container's
dependency stack imports cleanly is the spike's **first** pass gate — and if it
fails there too, resolving the diffusers/transformers pin becomes explicit
in-scope work for the full-family plan.

## Decisions (from brainstorm)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Scope | Spike first, then design | Prove the dep stack before committing to a full-family spec |
| Execution host | Remote/CI NVIDIA host via `test-cuda` container | Needs real CUDA + VRAM; mac authors only |
| Load fidelity | **Mirror production** (`from_pipe` on shared base) | Proves the real integration + shared-base VRAM, not just that the class runs |
| Control types | **Canny only** | One type proves the whole stack; depth/pose reuse the identical path, no new signal |
| VRAM handling | **Record only** | Print peak VRAM; humans judge fit vs target GPUs in the full design. Spike passes regardless of the number |

## Design

### Location
- Spike script: `spikes/hunyuandit_controlnet_spike.py` (new `spikes/` dir; throwaway,
  not imported by any production path).
- A supplied Canny control-map PNG committed alongside, or a path arg to one.
- Run recipe captured in this spec (below).

### What the script does — in production load shape
1. **Import gate.** Import `HunyuanDiTControlNetPipeline`, `HunyuanDiT2DControlNetModel`
   from diffusers and the BERT/mT5 encoders from transformers. Print resolved
   `diffusers.__version__` and `transformers.__version__` up front so the run log
   states exactly which pins worked (or where it broke).
2. **Load base.** `HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
   torch_dtype=fp16)` on CUDA.
3. **Load control model.** `HunyuanDiT2DControlNetModel.from_pretrained(
   "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny", torch_dtype=fp16)`.
4. **Compose via `from_pipe`.** `HunyuanDiTControlNetPipeline.from_pipe(base,
   controlnet=control_model)` — the shared-base pattern the production worker will use.
5. **Generate once.** Short prompt, a Canny control-map PNG passed as `control_image=`,
   `1024x1024`, `use_resolution_binning=True`.
6. **Capture.** Save the output PNG for eyeball; print
   `torch.cuda.max_memory_allocated()` peak VRAM.

### Run recipe
```
# on the remote linux/amd64 + NVIDIA host, from repo root
docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
    python spikes/hunyuandit_controlnet_spike.py --control-map <canny.png>
```

## Pass criteria (the gate for opening the full-family spec)

The spike **passes** when all of:

1. **Imports resolve** in the CUDA container — records the working `diffusers` +
   `transformers` versions. (If they do not resolve, the spike has still delivered
   its highest-value finding: pin resolution is in-scope for the plan.)
2. **`from_pipe` builds** the pipeline without dtype/VAE explosions.
3. **Output PNG is a coherent, Canny-conditioned image** (human eyeball).
4. **Peak VRAM recorded** — a number, not a threshold. Feeds the VRAM / fp8-offload
   gating decision (STABL-ifymwtiv, open question #2).

The spike is **not** blocked by a VRAM ceiling and does not assert one.

## Explicit non-goals (deferred to the full-family spec)

- Depth and Pose control types (Pose revisits the deferred sd15 pose decision —
  open question #3 — but is out of the *spike's* scope).
- The `DiffusersHunyuanDiTCudaWorker` production worker, `_CONTROL_IMAGE_KWARG`
  override, detector, registry VRAM estimation, and mode config.
- Any change to the shared `_build_controlnet_kwargs` seam (already landed).
- fp8/offload gating (STABL-ifymwtiv) — informed by, not decided by, the spike.
- Multi-controlnet, img2img+HunyuanDiT combinations.

## Outcome recording

On completion, record on STABL-ichgkgno: the working (or broken) dep pins, peak
VRAM, the output image verdict, and a go/no-go on opening the full-family spec.
