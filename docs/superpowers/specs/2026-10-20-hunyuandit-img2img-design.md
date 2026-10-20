# HunyuanDiT img2img design

**Status:** Approved design; implementation plan pending written-spec review.

## Goal and scope

Add plain image-to-image generation for the CUDA HunyuanDiT family. A request
with `init_image_ref` uses the existing image bytes, `denoise_strength`, prompt,
size, steps, guidance, and seed contracts. This delivery does not combine an
init image with ControlNet; that is a separate delivery. Hunyuan text-to-image
and ControlNet text-to-image continue to use their current pipelines.

## Chosen approach

Use a Hunyuan-specific img2img execution component that shares the already
loaded Hunyuan transformer, VAE, text encoders, tokenizers, and scheduler. It
must not load a second copy of the model or cast shared components to fp32.
The component owns the img2img denoising path because Diffusers 0.39 does not
provide a HunyuanDiT image-to-image pipeline. It follows the installed
HunyuanDiT pipeline's prompt, rotary embedding, guidance, VAE, and scheduler
contracts rather than routing through an SD or SDXL pipeline.

Injecting a noisy source latent into the existing text-to-image pipeline was
rejected: that pipeline still executes its full timestep schedule, so the
requested strength would not correspond to the amount of source preservation.
Waiting for an upstream pipeline was rejected because it would leave this
delivery without a working Hunyuan img2img path.

## Request and execution flow

1. Existing WebSocket admission resolves the active family from its model
   snapshot. Set Hunyuan CUDA `supports_img2img=True` while keeping
   `supports_controlnet=True` and `supports_img2img_and_controlnet=False`.
   Admission still rejects a combined request before image preprocessing.
2. The Hunyuan worker decodes the init-image bytes as RGB and resizes to the
   requested dimensions, matching the SD workers' image input behavior. Bad
   image bytes fail clearly; the request schema already bounds strength to
   `0.01..1.0`.
3. The img2img component preprocesses the source, encodes it with the loaded
   VAE, and applies the VAE scaling factor. It sets the selected scheduler's
   timesteps, chooses the starting timestep from `denoise_strength`, adds
   generator-seeded noise to the encoded source, and runs only the remaining
   timesteps. A low strength retains more of the source; strength 1.0 runs the
   full denoising schedule.
4. The component uses Hunyuan's existing two-encoder conditioning contract,
   transformer inputs, rotary embeddings, guidance, and VAE decoding. The
   worker returns the usual PNG and seed, and records strength in the PNG
   generation metadata. Progress and cancellation operate on the actual
   number of executed denoising steps.
5. Model unload and failed jobs release transient latents and references.
   Scheduler selection, device and dtype behavior, and native attention
   processors retain the current Hunyuan worker rules.

## Boundaries and errors

- The active Hunyuan model must be a Diffusers directory as today. No new
  HTTP img2img endpoint or UI control is needed; the existing UI and CLI can
  send the input reference and strength through the WebSocket request path.
- A request containing both an init image and any ControlNet attachment still
  fails with the family-specific `img2img+controlnet` admission error.
- Invalid image bytes, an incompatible image size, or a scheduler that cannot
  perform the required noise and timestep operations fail explicitly. They
  must not fall back to text-to-image or produce a plausible but unrelated
  image.
- The baseline native DDPMScheduler must work. A request-selected scheduler
  works only when it supports the required timestep and `add_noise` operations;
  an incompatible selection is rejected before generation. A scheduler test
  covers the baseline and each other scheduler this delivery claims to support.

## Verification

- Unit tests cover family admission, plain and combined request routing,
  source image decoding, strength-to-timestep selection, VAE scaling and
  noising, seeded repeatability of the input latent, conditioning keys,
  progress/cancellation, metadata, and cleanup. Existing Hunyuan text-to-image
  and ControlNet tests remain green.
- A dedicated live CUDA acceptance run uses the mounted Hunyuan model and a
  known source image. It saves the source and output, verifies PNG dimensions,
  seed and metadata, checks for ignored-input warnings, and records runtime
  and peak allocated VRAM. It checks at least one moderate strength and one
  low strength with the same seed, with visual review of source influence.
- The capability is claimed as live-verified only after that CUDA run passes.
  CPU unit tests alone establish wiring, not generated-image quality.
