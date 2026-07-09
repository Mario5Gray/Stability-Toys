# img2img + ControlNet Combined Path — Design Decisions

**Date:** 2026-07-08
**Status:** Decided
**FP:** STABL-ztaxgbhv (parent), STABL-uiwneiqf, STABL-bwkjcbwc

Two decisions needed before the pipeline-wiring tasks (`STABL-vgbxamoz` SD1.5,
`STABL-umvdwgsm` SDXL) can implement the combined execution branch in
`backends/cuda_worker.py`. Both `StableDiffusionControlNetImg2ImgPipeline` and its
SDXL counterpart are thin wrappers diffusers exposes on top of the same base
components already loaded by this worker — this doc governs how we drive them, not
how diffusers implements them internally.

## Decision 1: denoise_strength × strength/start_percent/end_percent interaction

`start_percent`/`end_percent` on each ControlNet attachment are passed straight
through to the combined pipeline's `control_guidance_start`/`control_guidance_end`
kwargs, unmodified by `denoise_strength`. We do not attempt to renormalize them
against the nominal (pre-strength) step count — diffusers' combined pipeline already
applies `strength` to compute its own effective step schedule internally, and
`control_guidance_start`/`control_guidance_end` are diffusers' contract against
whatever schedule it derives. Re-deriving that math in our wrapper would duplicate
diffusers internals and drift the first time the installed diffusers version changes
its slicing behavior.

Concretely: `denoise_strength` flows into `strength=` (already the case for the
plain img2img path today), and each attachment's existing `strength` (ControlNet
conditioning scale — not to be confused with `denoise_strength`) flows into
`controlnet_conditioning_scale=` as it does on the txt2img ControlNet path. No new
plumbing beyond what the txt2img ControlNet branch already does for
`controlnet_conditioning_scale`/`control_guidance_start`/`control_guidance_end`.

**At strength=1.0** (full regenerate): behaves identically in spirit to the existing
txt2img + ControlNet path — the full nominal `num_inference_steps` schedule runs
and `control_guidance_start`/`end` apply across all of it. This must be covered by a
Group B test asserting the combined-path call captures the same
`controlnet_conditioning_scale`/`control_guidance_start`/`control_guidance_end`
values the txt2img branch would for equivalent attachment strength/start/end inputs.

**At low strength** (e.g. `denoise_strength=0.05`): diffusers computes very few
effective denoising steps from a low strength. If `end_percent` is also small (e.g.
`0.3`), the ControlNet conditioning may end up applied to zero of the few remaining
effective steps — the generation looks like an almost-unconditioned img2img pass.
This is **accepted v1 behavior, not a bug**: no auto-clamping, no validation error.
It is a documented operator caveat (added to `CONTROLNET.md`'s "Not supported in v1"
section update, see `STABL-dxaheihz`) so users understand very low `denoise_strength`
combined with a narrow `start_percent`/`end_percent` window may produce
ControlNet-invisible results. Group B does not need special-case code for this.

## Decision 2: control-map vs init-image size reconciliation

See the sizing-reconciliation section appended by the follow-up task (`STABL-bwkjcbwc`)
below.
