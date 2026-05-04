# ControlNet RKNN Feasibility Design

## Summary

This spec answers a narrower question than the CUDA and Apple specs:

Can ControlNet be implemented on the RKNN backend, or should ControlNet requests fall back to CPU?

Answer:

- `Yes, native RKNN ControlNet is possible in principle.`
- `No, it is not a cheap extension of the current RKNN path.`
- `A partial hybrid where CPU runs ControlNet and the current RKNN UNet stays unchanged is not a good v1 path.`
- `If we need correctness sooner than native RKNN delivery, the honest fallback is whole-request CPU execution, not mixed CPU-ControlNet + current-RKNN-UNet execution.`

Recommendation:

1. long-term: build native RKNN ControlNet for the existing RK3588-oriented SD1.5 LCM stack
2. short-term: if ControlNet must ship on Rockchip before that work is done, use request-level CPU fallback
3. do not invest in a mixed execution design that leaves the current RKNN UNet interface untouched

## Why This Is Different From CUDA

CUDA ControlNet in this repo works because Diffusers already supports:

- base UNet loading
- one or more ControlNet modules
- ordered multi-ControlNet binding
- residual injection into the UNet forward pass

The current RKNN path is not a diffusers-native graph. It is a hand-assembled pipeline with three compiled RKNN components:

- text encoder
- UNet
- VAE decoder

That matters because ControlNet is not just an image preprocessor. During denoising it adds learned residuals into the UNet path on every step. A backend that only knows how to call a compiled UNet with its current fixed input signature does not get ControlNet "for free".

## External Constraints

Current external facts used by this spec:

- Radxa's RKNN Stable Diffusion guide documents an SD1.5 LCM-style deployment with `text_encoder`, `unet`, and `vae_decoder` RKNN models, converted offline for a target resolution.
- RKNN-Toolkit2 requires offline model conversion before device inference.
- RKNN-Toolkit2 has broader operator support than older releases, including custom operators and improved transformer support, but Rockchip does not publish a turnkey ControlNet-for-RKNN path in the cited docs.

Design consequence:

- RKNN ControlNet should be treated as a custom backend project, not as a config-only extension of the existing LCM RKNN worker.

## Current Repo State

- [`backends/rknn_worker.py`](/home/hdd/workspace/Stability-Toys/backends/rknn_worker.py) runs text-to-image only and forwards prompt, size, steps, guidance, and seed into the RKNN LCM pipeline.
- [`backends/rknnlcm.py`](/home/hdd/workspace/Stability-Toys/backends/rknnlcm.py) exposes a compiled UNet call with the fixed inputs `sample`, `timestep`, `encoder_hidden_states`, and `timestep_cond`.
- [`backends/base.py`](/home/hdd/workspace/Stability-Toys/backends/base.py) exposes only `text_encoder`, `unet`, and `vae_decoder` model paths for RKNN.
- [`notes/rknn_ff.md`](/home/hdd/workspace/Stability-Toys/notes/rknn_ff.md) already shows the RKNN path is missing even a VAE encoder for img2img, which is a smaller architectural extension than ControlNet.
- The ControlNet request/policy/asset seam already exists at the server layer and remains backend-agnostic.

## Decision

### What is feasible

- `Native RKNN ControlNet`: feasible, high effort
- `Whole-request CPU fallback`: feasible, medium effort, poor performance on-device
- `CPU preprocessors only`: already compatible, but this is not ControlNet execution

### What is not a good path

- `CPU ControlNet residuals + existing RKNN UNet binary`: not recommended

Reason:

The current RKNN UNet export does not accept ControlNet residual tensors. CPU can compute edge maps or even ControlNet features, but unless the RKNN UNet is re-exported to accept those extra tensors, there is nowhere valid to inject them.

That means the "hybrid" idea does not actually avoid the hard part.

## Options

## Option A: Native RKNN ControlNet

### Description

Convert and run the ControlNet path on Rockchip, keeping denoising on the NPU-oriented pipeline.

### Required scope

- stay on `SD1.5 LCM` first
- stay on `txt2img` first
- one ControlNet attachment first
- fixed resolution first

### Required model work

At minimum, native RKNN ControlNet needs new compiled assets. There are two plausible technical shapes:

1. `Fused export`
   - export one combined graph that bakes ControlNet behavior into a ControlNet-aware UNet
   - likely one compiled model per control type and per resolution

2. `Split export`
   - export one or more ControlNet models separately
   - export a UNet that accepts additional residual inputs
   - feed residual tensors into the UNet on each denoising step

The split design is architecturally cleaner, but only if RKNN conversion and runtime behavior are stable for the added interfaces.

### Pros

- preserves the reason to use RKNN at all
- keeps inference on the NPU path
- avoids catastrophic latency from CPU denoising
- aligns with the repo's original edge-inference goal

### Cons

- requires substantial offline conversion work
- likely forces a custom RKNN graph/export pipeline per control type
- may require one compiled asset per resolution bucket
- multi-ControlNet would likely multiply memory and conversion complexity
- debugging numerical drift after conversion will be harder than on CUDA

### Expected v1 restrictions

- SD1.5 only
- canny and depth only
- single ControlNet only
- one or two fixed resolutions only
- no img2img
- no LoRA plus ControlNet guarantee until separately validated

## Option B: Mixed CPU ControlNet + Existing RKNN UNet

### Description

Run ControlNet on CPU, then keep the current RKNN text encoder, UNet, and VAE decoder exactly as they are.

### Recommendation

Do not build this.

### Why it fails architecturally

ControlNet is not just a one-time precompute. It contributes learned residuals into the UNet every denoising step. The current RKNN UNet call surface is fixed and does not expose ControlNet residual inputs.

Therefore this option ends up at one of two dead ends:

- either it computes information that the RKNN UNet cannot consume
- or it still requires re-exporting the UNet and pipeline, which collapses back into Option A

### Allowed exception

CPU preprocessing of the conditioning image remains fine:

- canny on CPU
- depth preprocessing on CPU
- emitted map reuse through the asset store

But that is preprocessing, not ControlNet inference.

## Option C: Whole-request CPU Fallback

### Description

If `BACKEND=rknn` receives a ControlNet request and no native RKNN ControlNet asset exists, route the entire generation to a CPU implementation instead of trying to splice CPU work into the current RKNN graph.

### Recommendation

This is the best short-term fallback if correctness matters more than performance.

### Pros

- simplest way to make the request contract honest
- easiest path to reuse a Python diffusers-style ControlNet implementation
- no RKNN graph surgery required for the fallback path
- preserves the existing mode policy and artifact contract

### Cons

- current `BACKEND=cpu` in this repo is scaffold-only, so CPU fallback itself is another implementation track
- performance on RK3588-class CPUs is likely poor
- mode switching and cache semantics get more complex if one backend delegates to another
- LoRA parity on CPU must be implemented and tested too

### When it is acceptable

- development validation
- correctness-first demos
- low-volume admin-only workflows
- explicit degraded mode behind configuration

### When it is not acceptable

- production interactive latency on SBC hardware
- any claim that Rockchip NPU now "supports ControlNet" if requests are really being done on CPU

## Recommended Architecture

## 1. Keep the external contract unchanged

Do not change:

- `GenerateRequest.controlnets`
- `controlnet_policy`
- `controlnet_artifacts`
- preprocessing and asset emission behavior

This preserves backend-agnostic API behavior.

## 2. Add backend-specific ControlNet capability resolution

Extend `conf/controlnets.yaml` so RKNN support is explicit per model:

```yaml
models:
  sd15-canny:
    control_types: [canny]
    compatible_with: [sd15]
    backends:
      cuda:
        path: /models/controlnets/sd15-canny
        format: diffusers
      rknn:
        path: /models/controlnets/rknn/sd15-canny-256
        format: rknn-controlnet-bundle
        resolutions: ["256x256"]
```

Rules:

- requests on `BACKEND=rknn` may execute natively only if the requested model resolves to an RKNN backend entry
- missing RKNN entries must either:
  - fail clearly, or
  - trigger whole-request CPU fallback if that mode is explicitly enabled

## 3. Add explicit fallback policy

Recommended environment variable:

```bash
RKNN_CONTROLNET_POLICY=reject   # reject | cpu_fallback
```

Rules:

- `reject`
  - default
  - if ControlNet is requested and no native RKNN support exists, fail with a clear error
- `cpu_fallback`
  - allow whole-request CPU execution
  - response metadata must disclose fallback backend used

Do not add a policy that implies mixed CPU-ControlNet plus unchanged RKNN UNet execution.

## 4. Add truthful result metadata

For any fallback path, include backend truth in success metadata and logs:

- requested backend: `rknn`
- execution backend: `cpu` or `rknn`
- fallback reason: missing native RKNN ControlNet support, unsupported resolution, unsupported multi-attachment, etc.

This avoids silently delivering non-NPU results on an NPU deployment.

## Native RKNN Design

## 1. Scope

Native v1 should be intentionally narrow:

- SD1.5 LCM only
- text-to-image only
- canny and depth only
- single attachment only
- one fixed resolution first, ideally `256x256`

Starting wider than that is likely to turn this into an open-ended research project.

## 2. Bundle format

Add an RKNN-specific model bundle type rather than pretending the current `path` points to a single model:

```yaml
format: rknn-controlnet-bundle
```

Suggested bundle contents:

```text
sd15-canny-256/
  manifest.json
  text_encoder/
  unet_controlled/
  vae_decoder/
  controlnet/
  scheduler/
```

or, for split export:

```text
sd15-canny-256/
  manifest.json
  text_encoder/
  unet_with_residual_inputs/
  vae_decoder/
  controlnet/
  scheduler/
```

The manifest should declare:

- base family
- control type
- fixed resolution
- expected latent shape
- guidance constraints
- whether multi-attachment is supported

## 3. Worker changes

Add a dedicated RKNN ControlNet worker instead of burying this in the existing worker:

- `backends/rknn_controlnet_worker.py`
- `backends/rknn_controlnet_runtime.py`

Responsibilities:

- resolve native RKNN control bundle
- decode control map from `AssetStore`
- resize control map deterministically
- run denoising with the ControlNet-aware compiled graph

Do not overload the current minimal LCM worker until the native path is proven.

## 4. Resolution policy

Native RKNN ControlNet should remain resolution-pinned in v1.

Rules:

- each bundle declares supported resolution(s)
- mismatched request sizes fail fast or are rejected by policy
- do not promise arbitrary-size ControlNet on RKNN v1

This follows the existing Radxa conversion model, where RKNN Stable Diffusion assets are converted for target resolutions.

## CPU Fallback Design

## 1. Triggering

CPU fallback should occur only when all of these are true:

- `BACKEND=rknn`
- request includes `controlnets`
- `RKNN_CONTROLNET_POLICY=cpu_fallback`
- no compatible native RKNN bundle exists for the request

## 2. Runtime shape

Use request-level delegation:

- validate request normally
- preprocess control maps normally
- resolve that RKNN cannot run this ControlNet request
- execute the whole generation in the CPU runtime
- return standard image and artifact response

The fallback path must not enqueue part of the denoising loop on RKNN and part on CPU.

## 3. CPU backend requirement

This repo currently has a placeholder CPU backend. CPU fallback therefore depends on a separate implementation track:

- real CPU generation runtime
- mode loading
- LoRA handling if required
- scheduler compatibility
- reasonable timeout and queue semantics

Until that exists, `reject` is the only honest policy.

## Testing Strategy

Automated tests should cover:

- RKNN backend rejects ControlNet when policy is `reject`
- RKNN backend chooses CPU fallback only when explicitly configured
- fallback responses disclose actual execution backend
- native RKNN bundle resolution validates family, control type, and resolution
- unsupported multi-attachment requests fail clearly on native RKNN v1

Manual validation for native RKNN should cover:

1. canny request at supported fixed resolution
2. depth request at supported fixed resolution
3. visible conditioning effect
4. repeated runs do not leak memory
5. mode switch still behaves correctly
6. unsupported resolution fails before generation

## Acceptance Criteria

This work is complete when one of these is true:

### Native milestone

- `BACKEND=rknn` can execute at least one real ControlNet type natively
- backend truth is explicit
- request/response contract matches CUDA

### Fallback milestone

- `BACKEND=rknn` can route ControlNet requests to CPU intentionally
- fallback is explicit in config, logs, and metadata
- there is no false claim of native RKNN ControlNet support

## Recommendation

For this repo, the most defensible order is:

1. keep default RKNN policy as `reject`
2. if near-term correctness is required, implement whole-request CPU fallback
3. if near-term NPU value is required, implement native RKNN ControlNet for SD1.5 LCM only
4. do not spend time on a mixed CPU-ControlNet plus unchanged-RKNN-UNet design

In short:

- `Can RKNN do ControlNet?` Yes, but as a dedicated native integration project.
- `Is CPU the way to go?` Only as a request-level fallback, not as a partial drop-in to the current RKNN denoising graph.

## References

- Radxa Stable Diffusion RKNN guide: https://docs.radxa.com/en/rock5/rock5b/app-development/ai/rknn-stable-diffusion
- RKNN-Toolkit2 maintained repository: https://github.com/airockchip/rknn-toolkit2
- RKNN-Toolkit2 legacy repository note and changelog: https://github.com/rockchip-linux/rknn-toolkit2
