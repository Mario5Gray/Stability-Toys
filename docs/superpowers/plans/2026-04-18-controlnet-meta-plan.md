# ControlNet v1 — Meta Plan

**Spec:** [2026-04-18-controlnet-design.md](../specs/2026-04-18-controlnet-design.md)
**FP top-level parent:** STABL-utbuhifx — *Implement ControlNet v1 (CUDA + Diffusers)*
**Date:** 2026-04-18

## Purpose

The spec is large enough that a single implementation plan would be unwieldy and hard to review. This meta-plan decomposes the work into three self-contained tracks, each with its own FP parent issue and detailed implementation plan. It also pins the ordering constraints between them so track owners can see what they can do in parallel and what blocks them.

Each track ships something inspectable on its own. Track 1 can be merged and released behind a mode flag before Track 2 starts — requests that attempt to use ControlNet will return a clear "provider not yet implemented" error. Track 2 can be reviewed and merged before Track 3 delivers the execution path and UI.

## Tracks

### Track 1 — Request contract, mode policy, backend enforcement

**FP parent:** STABL-iajgqfqp
**Spec sections:** 1 (Request contract), 2 (Mode-owned policy), 7 (Backend enforcement)
**Depends on:** none — can start immediately
**Blocks:** Track 2 (request shape), Track 3 (policy)

Scope:

- Add `controlnets` field to `GenerateRequest` and matching WebSocket generation submission
- Add `ControlNetAttachment`, `ControlNetPreprocessRequest`, `ControlNetArtifactRef` Pydantic models
- Add `controlnet_policy` parsing to `ModeConfig`
- Serialize `controlnet_policy` through `/api/modes`
- Extend `finalize_mode_generate_request(...)` (or add a sibling) to reject/normalize ControlNet attachments before worker dispatch
- Cover HTTP `/generate` and WebSocket job-submit paths identically
- Stub the provider dispatch with a clear "ControlNet provider not yet implemented on this backend" error path so the feature can ship dormant

Out of scope for this track:

- Any preprocessing
- Any model loading or CUDA execution
- Any frontend UI — frontend may consume `/api/modes` `controlnet_policy` read-only for future readiness, but does not render a new panel

Acceptance:

- `ModeConfig` parses `controlnet_policy` round-trip
- `/api/modes` returns the policy shape defined in the spec
- A generation request with a valid-shape `controlnets` list and a mode that enables ControlNet validates all fields and then fails with the stub error
- A request with an invalid shape (unknown `control_type`, out-of-range strength, duplicate `attachment_id`, missing/both source refs, etc.) fails before worker dispatch with a specific message — enumerated in spec Failure Handling → Attachment-invalid classes

### Track 2 — Asset layer + preprocessors + response contract

**FP parent:** STABL-nsrpodvu
**Spec sections:** 3 (Derived control-map assets), 4 (Preprocessor seam), 9 (Response contract)
**Depends on:** Track 1 (request shape, policy types)
**Blocks:** Track 3 (preprocessors and artifact refs)

Scope:

- Extend `upload_routes.py` ref table with `kind` (`upload` | `control_map`) and optional metadata blob
- Add LRU eviction with configurable byte budget (default 512 MB) and per-generation pinning
- Define `ControlMapPreprocessor` protocol and `ControlMapResult`
- Implement `CannyPreprocessor` and `DepthPreprocessor` (library choice decided at plan time; Open Questions in spec)
- Emit derived maps as `control_map` entries in the ref table with metadata per spec section 3
- Add `controlnet_artifacts` to the HTTP `/generate` response and the WebSocket `job:complete` frame
- Wire the enforcement layer from Track 1 to invoke preprocessors before provider dispatch when `source_asset_ref + preprocess` is specified

Out of scope for this track:

- CUDA ControlNet execution (still stubbed)
- Frontend options-panel changes

Acceptance:

- Canny and depth preprocessors produce reusable `control_map` assets for a given source image
- A request with `source_asset_ref + preprocess.id = "canny"` completes preprocessing and surfaces the emitted artifact in the response, even though generation still errors with the Track 1 stub
- Eviction policy is test-covered: a new entry exceeding the byte budget evicts LRU entries except those pinned by an in-flight generation

### Track 3 — CUDA provider + frontend + reuse

**FP parent:** STABL-pfpvqfaf
**Spec sections:** 5 (Provider seam), 6 (CUDA execution path), 8 (Frontend integration)
**Depends on:** Track 1 (request + policy), Track 2 (preprocessors + asset layer + response)
**Blocks:** none; this is the delivery track

Scope:

- Add `conf/controlnets.yaml` registry with `model_id -> {path, compatible_with, control_types}`
- Startup validation: every policy-referenced `model_id` resolves to a registry entry compatible with the mode
- Implement `ControlNetProvider` protocol with CUDA implementation
- CUDA provider: multi-attachment Diffusers wiring, ordered bindings, resolution-aware map resize at generation time
- Model cache: LRU keyed by `model_id`, capped by model count (default 3) and VRAM budget (default 4 GB), pinning per generation
- Replace Track 1's stub dispatch with the real CUDA provider `run(...)` call
- Frontend: new `ControlNet` section in `OptionsPanel` gated on `controlnet_policy.enabled`
- Frontend: draft list management in `useGenerationParams`, attachment add/remove, source selection (upload / existing generation / emitted map), strength and percent controls
- Frontend: render emitted `controlnet_artifacts` on message results and allow reuse in subsequent attachments

Out of scope for this track:

- Drawing / sketch input (V2)
- img2img ControlNet (V2)
- RKNN provider (V2)
- Pose / normal / segmentation preprocessors (V2)

Acceptance:

- Manual validation steps 1–5 from the spec pass end-to-end
- Multi-ControlNet request with two attachments in specific order produces an image that visibly reflects both constraints
- VRAM cap triggers eviction before OOM under a stress scenario that exceeds the default budget
- Frontend renders emitted maps on the result message and allows selecting them as `map_asset_ref` for the next generation without re-preprocessing

## Ordering and parallelism

Tracks must merge in order 1 → 2 → 3 because each builds on the previous contract. Within a track, subissues can parallelize freely.

A cautious parallelization opportunity: once Track 1 has landed the request and policy types, Track 2's preprocessor protocol and Track 3's `controlnets.yaml` loader can be drafted in parallel; they share no code paths. Integration still gates on Tracks 2 and 3 having a merged Track 1 to build on.

## Cross-track risks

- **Contract drift.** If Track 2 or 3 discovers the Track 1 request shape is insufficient, the fix should land as a contract amendment in Track 1 (a small follow-up issue under STABL-iajgqfqp), not a local workaround. The meta-plan exists partly to protect that invariant.
- **Model registry scope creep.** `conf/controlnets.yaml` in Track 3 is tempting to expand into a generic model registry. Keep it ControlNet-scoped for v1.
- **Preprocessor library choice.** Deferred to Track 2 plan-time. Whichever is picked, keep it behind the `ControlMapPreprocessor` protocol so a swap later costs one module.
- **Frontend ambition in Track 3.** The UI scope can balloon (thumbnails, drag-reorder, per-attachment previews). Track 3 plan should hold the line on the spec's "render legal choices, add/remove, inspect, reuse" acceptance and defer polish.

## Deliverable from each track

Each track parent issue is assigned a detailed implementation plan authored via the writing-plans skill:

| Track | FP parent | Plan file |
| ----- | --------- | --------- |
| 1 | STABL-iajgqfqp | `docs/superpowers/plans/2026-04-18-controlnet-track-1-request-policy.md` |
| 2 | STABL-nsrpodvu | `docs/superpowers/plans/2026-04-XX-controlnet-track-2-assets-preprocessors.md` |
| 3 | STABL-pfpvqfaf | `docs/superpowers/plans/2026-04-XX-controlnet-track-3-cuda-provider-frontend.md` |

Track 1's plan is written next. Tracks 2 and 3 plans are authored once their predecessor is ready to merge, so that contract drift is captured before subissues are fanned out.

## Out-of-band concerns resolved in the spec amendment

For future readers: the spec was amended on 2026-04-18 to resolve ambiguities that would have otherwise produced contradictory implementation plans. Changes worth knowing before reading the track plans:

- `source_asset_ref` and `map_asset_ref` share a single ref namespace with existing `fileRef` values
- `attachment_id` is client-generated and echoed back in `controlnet_artifacts`
- Provider signature uses `run(mode, bindings, req) -> GenerateResult` (no `pipe` arg), keeping RKNN viable
- Control map resolution mismatch is handled by provider-side resize, not rejection
- Model cache policy and ref-table eviction policy are both specified; plan-time decisions are confined to library choices and file layout
