# T5.4 Followup: Real ControlNet Pipeline Integration

**FP issue:** STABL-pfpvqfaf
**Predecessor task:** T5.4 (commits f1e2267 → 1caa628)
**Reviewer blocker:** posted on STABL-pfpvqfaf 2026-04-29 by theta
**Status:** LANDED 2026-06-28 — implemented in commit 69005d2 ("swap to ControlNet-capable pipeline when bindings present"), with test fixups in 768ee3d. Verified: 21 passing across tests/test_cuda_worker_controlnet.py + tests/test_cuda_worker_base.py. Reviewer blocker (theta, 2026-04-29) cleared — the runtime gap is closed.

---

## Why this exists

T5.4 threaded `controlnet`, `control_image`, `controlnet_conditioning_scale`,
`control_guidance_start`, and `control_guidance_end` into `run_job()` for both
SD1.5 and SDXL workers and proved the kwarg shape via
`tests/test_cuda_worker_controlnet.py`.

Reviewer (theta) flagged that the live runtime cannot honor those kwargs:

- [backends/cuda_worker.py:320](../../../backends/cuda_worker.py#L320) and
  [backends/cuda_worker.py:628](../../../backends/cuda_worker.py#L628) load
  `StableDiffusionPipeline.from_pretrained(...)` /
  `StableDiffusionXLPipeline.from_pretrained(...)`.
- [backends/cuda_worker.py:468](../../../backends/cuda_worker.py#L468) and
  [backends/cuda_worker.py:800](../../../backends/cuda_worker.py#L800) then
  invoke `self.pipe(**call_kwargs)` with the controlnet kwargs spread in.
- `StableDiffusionPipeline.__call__` and `StableDiffusionXLPipeline.__call__`
  in the installed diffusers do not accept `controlnet` / `control_image` —
  the kwargs are silently dropped.
- The unit tests pass only because `worker.pipe = MagicMock()` accepts any
  call shape.

T5.4 is therefore correct for the *kwarg contract* but inert at runtime.

## Scope of this followup

Make the CUDA workers actually run multi-attachment ControlNet generation
against live diffusers. Two parts must land together:

1. Pipeline-class swap when bindings are present.
2. Tests upgraded to lock the live pipeline class, not just the kwarg shape.

Out of scope (defer to later tasks):

- ControlNet + img2img. Spec [§controlnet-design:38](../specs/2026-04-18-controlnet-design.md)
  marks img2img-with-controlnet as V2. Reject (or ignore controlnet bindings)
  on the img2img path; do not attempt to integrate them in this followup.
- Caching the swapped pipeline across calls. v1 may construct fresh on each
  call with bindings; weights stay shared via `from_pipe`. Add caching only
  if a follow-up task explicitly calls for it.
- VRAM-budget enforcement of the controlnet pipeline (separate concern,
  already partially handled by `backends/controlnet_cache.py`).

## What to change

### backends/cuda_worker.py — SD1.5 path

When `bindings` is non-empty AND `init_image is None` (text-to-image only):

```python
from diffusers import StableDiffusionControlNetPipeline

cn_pipe = StableDiffusionControlNetPipeline.from_pipe(
    self.pipe,
    controlnet=call_kwargs["controlnet"],
)
# from_pipe shares unet/vae/text_encoder weights with self.pipe.
# Do NOT call .to(device) again — components already on device.
with torch.inference_mode():
    out = cn_pipe(**call_kwargs)
```

Considerations:

- `from_pipe` is the diffusers API for building a derived pipeline that shares
  weights with the source. It is the right tool here; do not use a fresh
  `from_pretrained` (would double VRAM).
- `controlnet` must be passed to `from_pipe`, not via the call kwargs alone.
  Diffusers needs the controlnet attached at pipeline construction so its
  `__call__` can find the conditioning hooks. Keep `control_image`,
  `controlnet_conditioning_scale`, `control_guidance_start`,
  `control_guidance_end` in the call kwargs.
- After `from_pipe`, **remove** `controlnet` from `call_kwargs` before calling
  `cn_pipe(**call_kwargs)`. The controlnet is now an attribute of the pipeline,
  not a per-call argument. The four other controlnet kwargs (`control_image`,
  `controlnet_conditioning_scale`, `control_guidance_start`,
  `control_guidance_end`) DO stay in call kwargs.

### backends/cuda_worker.py — SDXL path

Same pattern with the XL classes:

```python
from diffusers import StableDiffusionXLControlNetPipeline

cn_pipe = StableDiffusionXLControlNetPipeline.from_pipe(
    self.pipe,
    controlnet=call_kwargs["controlnet"],
)
with torch.inference_mode():
    out = cn_pipe(**{k: v for k, v in call_kwargs.items() if k != "controlnet"})
```

### backends/cuda_worker.py — img2img branch

When `init_image is not None` AND `bindings` is non-empty:

- Decision: reject. Raise a clear `NotImplementedError`
  ("ControlNet bindings on the img2img path are not supported in v1 — see
  spec §controlnet-design.md V1 vs V2 split.").
- Caller-side guard (`enforce_controlnet_policy` in
  `server/controlnet_constraints.py`) should already disallow this combo for
  v1 modes; this raise is a defense-in-depth check, not the primary gate.

When `init_image is not None` AND `bindings` is empty: existing behavior, no
change.

### Where to put the swap

Both `run_job()` methods are getting long. Extract a helper on
`CudaWorkerBase` so both subclasses share the swap logic:

```python
def _build_controlnet_pipe(self, controlnet_obj):
    """Swap self.pipe for a ControlNet-capable variant via from_pipe.

    Subclasses override to pick StableDiffusionControlNetPipeline vs
    StableDiffusionXLControlNetPipeline.
    """
    raise NotImplementedError
```

Override on `DiffusersCudaWorker` (SD1.5) and `DiffusersSDXLCudaWorker`. Call
from `run_job()` only when `bindings` is non-empty and `init_image is None`.

## Tests that must change

`tests/test_cuda_worker_controlnet.py` currently asserts the kwarg shape
against `worker.pipe.call_args.kwargs`. After the swap, the call goes to a
*different* pipeline object. Tests must:

1. Stop using `MagicMock` for `worker.pipe`. Use a real
   `StableDiffusionPipeline` / `StableDiffusionXLPipeline` instance, or a
   minimal stub that passes `isinstance` for the right base class.
2. Patch `StableDiffusionControlNetPipeline.from_pipe` /
   `StableDiffusionXLControlNetPipeline.from_pipe` to return a tracked mock
   that records its `__call__` kwargs, then assert on that mock instead of
   `worker.pipe.call_args`.
3. Add a positive assertion: when bindings are present,
   `from_pipe.assert_called_once_with(worker.pipe, controlnet=<expected>)`.
4. Add a negative assertion: when bindings are empty, `from_pipe` is NOT
   called and `worker.pipe(...)` IS called.
5. Add an img2img-rejection test: when `job.init_image` is bytes AND
   `job.controlnet_bindings` is non-empty, expect `NotImplementedError`
   (or whatever rejection contract gets adopted).

The four existing tests
(`test_sd15_worker_passes_single_controlnet_kwargs`,
`test_sdxl_worker_passes_controlnet_lists_in_request_order`,
`test_decode_control_image_converts_rgb_and_resizes`,
`test_load_controlnet_model_uses_process_cache`) should survive — but the
first two need to assert against the swapped `cn_pipe`, not `worker.pipe`.

## Verification commands

RED first (the followup tests should fail before the worker change lands):

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && \
  python -m pytest tests/test_cuda_worker_controlnet.py -q
```

GREEN after the worker change:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && \
  python -m pytest tests/test_cuda_worker_controlnet.py tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py -q
```

A real-CUDA smoke test (one attachment, manual canny map) is desirable but
gated on hardware. Document the exact command in
`docs/TESTING_CONTROLNET_TRACK3.md` (which T7 will create) rather than baking
it into this followup.

## Drift

`backends/cuda_worker.py` is bound by `docs/superpowers/specs/2026-04-18-controlnet-design.md`.
That spec, section 6 ("CUDA execution path"), already calls out
`backends/cuda_worker.py` as the right place to consume bindings — prose stays
accurate. Refresh provenance after the change with:

```bash
drift link docs/superpowers/specs/2026-04-18-controlnet-design.md backends/cuda_worker.py
drift check
```

## Commit + FP

Single commit recommended:

```
feat(controlnet): swap to ControlNet-capable pipeline when bindings present (STABL-pfpvqfaf)
```

After commit:

```bash
fp issue assign STABL-pfpvqfaf --rev <sha>
fp comment STABL-pfpvqfaf "T5.4-followup landed: ..."
```

The followup closes the live-runtime gap that theta flagged. T5.5 (re-run) and
T5.6 (commit step) on the original plan can then proceed cleanly.

## Pointers

- Reviewer blocker comment: STABL-pfpvqfaf, posted by theta 2026-04-29.
- Original T5.4 commits: `f1e2267` (initial threading), `1caa628`
  (image → control_image fix).
- Plan: [docs/superpowers/plans/2026-04-22-controlnet-track-3-backend.md](2026-04-22-controlnet-track-3-backend.md),
  Task 5 starts at line 605.
- Spec: [docs/superpowers/specs/2026-04-18-controlnet-design.md](../specs/2026-04-18-controlnet-design.md),
  section 6 covers CUDA execution.
