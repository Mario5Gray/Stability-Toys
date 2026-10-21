# HunyuanDiT img2img + ControlNet design

**Status:** Reviewed for implementation planning; control-latent seeding decision
resolved after review.

## Goal and scope

Enable a HunyuanDiT CUDA request to use an init image and one Canny ControlNet
attachment together. The request uses the existing WebSocket fields:
`init_image_ref`, `denoise_strength`, `controlnets`, prompt, size, steps,
guidance, scheduler, and seed. A control attachment may supply an existing map
or request the existing Canny preprocessing path. The active mode continues to
limit Hunyuan to one attachment. Plain img2img, ControlNet text-to-image, and
other model families retain their existing routes.

This delivery does not enable Hunyuan Depth or Pose, multiple attachments,
an HTTP img2img route, or new UI controls. Depth and Pose registry entries do
not count as live-verified support; each needs a separate checkpoint and
preprocessing acceptance run before a capability claim.

## Chosen approach

Extend the Hunyuan-specific img2img component's denoising loop with an
optional ControlNet model and control image. Use the already-loaded Hunyuan
pipeline's transformer, VAE, two text encoders, tokenizers, scheduler, and
native attention processor. The worker acquires the Canny model through its
existing cache and releases the cache pin in `finally`. The component performs
the same ControlNet-to-transformer residual handoff as the installed Diffusers
0.39 `HunyuanDiTControlNetPipeline`, while retaining the img2img component's
strength-selected schedule and source-latent noising.

Do not pass a noisy source latent to `HunyuanDiTControlNetPipeline.__call__`:
that call sets the full timestep schedule, so a partial strength would not
represent the requested amount of denoising. Do not create a second pipeline
or duplicate the Hunyuan prompt, rotary embedding, and VAE decode loop. The
installed ControlNet pipeline has no init-image/strength call contract.

## Request and execution flow

1. Admission reads the active model snapshot and Hunyuan CUDA family binding.
   Enable `supports_img2img_and_controlnet` only with the worker route in the
   same change. Existing mode policy, preprocessing, registry family checks,
   and binding resolution run before the job is queued. Combined requests on
   unsupported families keep their existing rejection behavior. For Hunyuan
   combined requests, reject a non-Canny control type or nondefault control
   timing window at admission, before preprocessing writes an artifact.
2. The Hunyuan worker requires exactly one resolved Canny binding for a
   combined job. It validates the init image and map bytes, rejects a native
   aspect-ratio divergence greater than the existing 2% tolerance, then
   resizes both to the requested size. Invalid dimensions, bad bytes, missing
   bindings, or unsupported control types fail explicitly before generation.
   A request with no binding remains plain img2img.
3. The worker selects the scheduler by its current rules, acquires the
   ControlNet model through the existing cache, passes that model and the
   decoded control image to the Hunyuan img2img component, and runs under
   `torch.inference_mode()`. The component validates scheduler operations and
   image dimensions before VAE work. It encodes and scales the source latent,
   selects timesteps from `denoise_strength`, and seeds source sampling and
   noising from the request generator as plain img2img already does.
4. The component prepares the Canny image with the installed Hunyuan
   ControlNet pipeline's image normalization and CFG batch behavior, then
   encodes and scales it with the shared VAE. It passes the request generator
   to the control-image latent distribution's `sample(generator=generator)`.
   This deliberately differs from Diffusers 0.39's
   `HunyuanDiTControlNetPipeline`, which calls `sample()` without a generator;
   the difference makes the combined path repeatable for the same seed and
   inputs. Preserve the current source-sample then source-noise order; sample
   the control latent afterward from that same generator.
   It runs ControlNet on each selected timestep and passes the returned block
   samples to the Hunyuan transformer. Text conditioning, rotary embeddings,
   guidance, scheduler stepping, progress, cancellation, safety check, and
   decoding retain the plain img2img contract. With no ControlNet argument,
   the component follows its existing plain img2img path.
5. The output PNG records the existing `lcm` metadata including
   `denoise_strength` and the existing `controlnet` provenance chunk. Failures,
   cancellation, and successful completion release the ControlNet cache pin
   and transient tensors; model unload retains its current behavior. The
   opt-in Hunyuan debug dump may be extended for this path but is not required
   for correctness.

## Control timing and boundaries

`controlnet_conditioning_scale` uses the attachment's existing `strength`
value. The installed Hunyuan ControlNet pipeline has no
`control_guidance_start` or `control_guidance_end` argument. For combined
requests, require `start_percent=0.0` and `end_percent=1.0`; reject other
windows before generation rather than silently ignoring them. Existing
Hunyuan ControlNet text-to-image behavior is unchanged. The Canny control is
active for every timestep that img2img actually executes.

The baseline DDPMScheduler and any other scheduler claimed for Hunyuan plain
img2img must still satisfy the selected-timestep and `add_noise` contract.
Combined execution fails clearly if a selected scheduler cannot support it.
No resolution binning is applied to the init image or Canny map; both use the
requested size and the existing divisible-by-8 check. ControlNet must use the
already-loaded model dtype/device. No `from_pipe` fp32 recast is allowed.

## Verification

- CPU tests cover family admission and WebSocket routing through preprocessing
  and binding resolution; combined worker dispatch; Canny-only and
  one-attachment limits; image decode, resize, aspect alignment, scheduler
  incompatibility, unsupported timing windows, PNG metadata, cache release on
  success/failure/cancellation, and unaffected plain img2img behavior.
- Component tests assert the installed Hunyuan ControlNet call contract:
  normalized/CFG-batched map latent, seeded VAE sampling, per-selected-step
  ControlNet call, residual blocks passed to the transformer, conditioning
  scale, progress count, and no ControlNet call on the plain path. Two
  combined runs with the same seed and inputs must produce the same prepared
  source and control latents; a changed seed must change at least one. Use
  real tensor timesteps where relevant.
- A dedicated enigma CUDA acceptance run exercises the production `WorkerPool`
  with the mounted Hunyuan base and Canny checkpoints. Save the source, map,
  and outputs for low and higher `denoise_strength` with fixed seed. Check
  dimensions, seed, both metadata chunks, absence of blocking ignored-input
  warnings, cancellation/cleanup where practical, and report elapsed time and
  peak allocated VRAM. Visually review both source preservation and Canny edge
  influence. Re-run the existing Hunyuan text-to-image, ControlNet
  text-to-image, and plain img2img acceptance cases.
- Claim combined support as live-verified only after the CUDA run passes. CPU
  wiring tests alone do not establish generated-image quality or VRAM safety.
