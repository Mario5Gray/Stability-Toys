# Gallery Advisor Design

## Summary

This design adds a gallery-backed advisor that analyzes curated image metadata, produces a reusable style digest, populates an editable advice object, and lets the user apply that advice to the draft prompt with deterministic `append` or `replace` actions.

The advisor is intentionally narrower than the earlier LLM expert proposal. In v1 it does not introduce loop orchestration, end-user expert personas, parameter mutation, or multimodal image understanding. It defines a smaller object model that can grow into those capabilities later:

- `Evidence`: typed analysis input
- `Digest`: machine-generated synthesis of evidence
- `Advice`: editable working object derived from the digest
- `AdvisorState`: persisted lifecycle state tied to a gallery

The source of truth is the gallery. Advisor identity is per `gallery_id`, not per mode. Stable Diffusion mode config still matters, but only as a runtime constraint when consuming advice. In particular, the mode defines `maximum_len`, which caps the advisor length slider.

## Goals

- Add an advisor section under `Negative Prompt Templates` in the existing options panel
- Let the user build a digest from gallery evidence without rescanning on every use
- Persist advisor state per gallery so it survives normal iteration
- Make advice user-editable
- Support `append` and `replace` application modes against the draft prompt
- Show advisor freshness, status, and recency in the UI
- Add `auto_advice`, `temperature`, and `length_limit` controls
- Define a typed `Evidence` shape that can later include multimodal image-analysis data

## Non-Goals

- Multi-step expert loops
- Automatic mutation of negative prompt, scheduler, CFG, steps, or other generation parameters
- Backend persistence of galleries or advisor state in v1
- Analysis of arbitrary non-gallery images
- LLM-based prompt rewrite during advice application
- Full multimodal image analysis in v1

## Current State

- Galleries are client-side only. `useGalleries()` stores gallery definitions in `localStorage` and gallery items in IndexedDB under the `lcm-galleries` database. There is no backend gallery model yet.
- The main options UI already has a `Negative Prompt Templates` section inside [OptionsPanel.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/options/OptionsPanel.jsx).
- The current app has no advisor object model, no digest generation flow, and no persistent analysis state linked to galleries.
- Mode config already carries Stable Diffusion-facing generation defaults and constraints, and `/api/modes` already exposes mode metadata to the frontend.
- The broader chat backend design introduces an OpenAI-compatible backend client. This advisor design should reuse that LLM transport rather than inventing a second LLM integration path.

## Proposed Approach

Use a gallery-backed advisor artifact owned by the frontend and a backend analysis endpoint owned by the server.

The split is:

- frontend owns gallery selection, advisor controls, advisor persistence, advice editing, and deterministic draft mutation
- backend owns analysis prompt execution, evidence normalization validation, and digest generation against the configured LLM endpoint

This is the smallest design that matches the current repo boundaries:

- galleries already live in the browser
- options UI already lives in the browser
- LLM transport should still remain backend-owned

The canonical v1 flow is:

`gallery metadata evidence -> build digest -> populate advice object -> apply advice to draft`

## Domain Model

### Evidence

`Evidence` is the structured input consumed by the advisor analysis prompt.

In v1, evidence is derived only from metadata belonging to one gallery. The gallery itself is treated as the representative set. The advisor does not analyze arbitrary chat history, live prompt text, or non-gallery generations.

The evidence shape must be typed and normalized before it is sent to the backend. It should be modeled around expected image metadata now, while reserving space for future multimodal enrichment.

Proposed v1 shape:

```json
{
  "version": 1,
  "gallery_id": "gal_123",
  "items": [
    {
      "cache_key": "abc",
      "added_at": 1712790000,
      "prompt": "painterly cyberpunk street vendor",
      "negative_prompt": "low quality, blurry",
      "size": "1024x1024",
      "steps": 28,
      "cfg": 6.5,
      "scheduler_id": "dpmpp_2m",
      "seed": 123456789,
      "superres_level": 0,
      "metadata": {}
    }
  ]
}
```

Notes:

- `metadata` is an escape hatch for already-stored cache metadata that does not fit the normalized top-level fields.
- Future versions may extend each item with multimodal analysis outputs, but the higher-level advisor flow should stay unchanged.

### Digest

`Digest` is the machine-generated baseline synthesized from evidence.

In v1, the digest is stored as text:

- `digest_text`: machine-authored baseline summary

Later versions may add a structured `digest_payload`, but v1 should not block on that.

### Advice

`Advice` is the consumable, user-editable working object derived from the digest.

In v1:

- `advice_text` is editable text
- `advice_text` is initially populated from `digest_text`
- the user can modify it directly
- the user can reset it back to the current digest

The advisor applies `advice_text` to the draft prompt only in v1.

### AdvisorState

`AdvisorState` is the persisted lifecycle container tied to one `gallery_id`.

Identity:

- persisted per `gallery_id`
- not per mode
- not per model
- future gallery subsets naturally become their own advisor states by appearing as distinct galleries

Proposed v1 stored shape:

```json
{
  "gallery_id": "gal_123",
  "evidence_version": 1,
  "evidence_fingerprint": "sha256:...",
  "digest_text": "Style digest...",
  "advice_text": "Editable advice...",
  "temperature": 0.4,
  "length_limit": 120,
  "auto_advice": true,
  "status": "fresh",
  "updated_at": 1712790500,
  "error_message": null
}
```

Reserved optional fields for later:

- `evidence_payload`
- `digest_payload`
- `advice_payload`

## Persistence Design

Advisor persistence should stay client-side in v1.

Rationale:

- galleries are already client-side only
- introducing backend gallery persistence would materially expand scope
- the backend only needs enough context to analyze evidence and return a digest

Recommended implementation:

- keep gallery definitions and items where they already live
- add a second IndexedDB object store for advisor states, keyed by `gallery_id`
- persist the full `AdvisorState` object in IndexedDB
- keep active gallery selection in localStorage as it already exists

Recommended IndexedDB evolution:

- existing DB: `lcm-galleries`
- bump schema version and add `advisor_states`
- keyPath: `gallery_id`

Why IndexedDB instead of localStorage:

- advice and digest text can grow beyond what is comfortable for localStorage-only state
- advisor state is structured and likely to expand
- this keeps the storage model aligned with existing gallery item persistence

Backend persistence of `AdvisorState` is explicitly deferred.

## Backend Design

### LLM Analysis Service

The backend should own digest generation.

Recommended shape:

- add `server/advisor_routes.py`
- add `server/advisor_service.py`
- reuse the planned OpenAI-compatible backend chat client rather than duplicating transport code

The service contract is:

1. receive typed `Evidence`
2. validate the payload shape
3. render the advisor analysis prompt
4. call the configured LLM endpoint
5. return a `Digest`

### Analysis Prompt

The analysis prompt is a first-class backend prompt asset. Its job is to tell the LLM how to read metadata and derive common themes in both draft wording and generation parameters.

The prompt should instruct the model to:

- look for repeated descriptive themes in prompt wording
- look for repeated parameter tendencies in the metadata
- summarize stable style traits, not one-off outliers
- produce concise, reusable guidance suitable for downstream draft refinement
- stay within the requested `length_limit`
- adjust strictness according to `temperature`

Even though v1 only applies advice to the draft prompt, the digest may still mention recurring parameter tendencies because that evidence is part of the source set and will matter later.

### Runtime Config

The current Stable Diffusion mode config gains:

- `maximum_len`

Purpose:

- caps the advisor length slider in the frontend
- reflects the target mode's practical prompt budget

Important boundary:

- `maximum_len` belongs to Stable Diffusion mode config
- it does not change advisor identity
- it does not cause `AdvisorState` fan-out per mode

The LLM analysis endpoint configuration should continue following the chat-backend direction. The advisor analysis service should resolve the currently active mode's LLM config, while consuming `maximum_len` as a separate Stable Diffusion-facing constraint.

### API Shape

Recommended v1 route:

- `POST /api/advisors/digest`

Request:

```json
{
  "gallery_id": "gal_123",
  "evidence": {
    "version": 1,
    "gallery_id": "gal_123",
    "items": []
  },
  "temperature": 0.4,
  "length_limit": 120
}
```

Response:

```json
{
  "digest_text": "Style digest...",
  "meta": {
    "evidence_fingerprint": "sha256:...",
    "model": "llama3.2"
  }
}
```

Why HTTP instead of a new WebSocket job in v1:

- the advisor rebuild is a single request/response analysis call
- the gallery source data already lives in the browser
- it avoids adding another WS job lifecycle before chat UI work is even landed

If future advisor analysis needs streaming progress or longer multi-step orchestration, it can move onto the WebSocket job protocol later.

## Frontend Design

### Advisor Panel Placement

Add a new `Advisor` section directly under `Negative Prompt Templates` in the existing options panel.

This keeps the feature next to the current prompt-shaping controls and avoids introducing a separate UI surface.

### Advisor Panel Controls

V1 panel contents:

- `Auto-Advice` toggle
- `Temperature` slider
- `Length` slider
- `Advisor Status` block
- `Rebuild Advisor` button
- `Advice Editor` text area
- `Reset To Digest` button
- `Apply Advice` button
- `Apply Mode` selector with `Append` and `Replace`

### Advisor Status

The status block must show both state and recency.

Supported statuses:

- `fresh`
- `stale`
- `building`
- `error`

Recency rules:

- less than 24 hours: relative text such as `12 min ago` or `3 hr ago`
- 24 hours or more: absolute date

Visual state:

- `fresh`: success color
- `stale`: warning color
- `building`: in-progress accent color
- `error`: red/error color

If the advisor is in `error` but still has usable prior `advice_text`, the user may continue using the existing advice.

### Auto-Advice Behavior

`Auto-Advice` controls refresh policy, not advisor identity.

Rules:

- gallery membership is the source boundary
- only gallery additions/removals affect advisor freshness
- arbitrary generations that are not added to the gallery do not affect the advisor

When gallery membership changes:

- update the gallery evidence fingerprint
- mark the advisor `stale`

If `auto_advice` is `off`:

- do not rebuild automatically
- keep current advice usable
- require explicit rebuild

If `auto_advice` is `on`:

- when a generation is added to the gallery, or another gallery membership change occurs, schedule an additive refresh when feasible
- additive refresh may update the digest from the changed evidence set without requiring the user to click rebuild
- full rebuild remains the canonical refresh path and repair path

Additive refresh must stay bounded:

- same `gallery_id`
- same evidence model
- no learning from non-gallery inputs

### Advice Editing

The `Advice Editor` is user-editable in v1.

Rules:

- initial populate comes from `digest_text`
- user edits affect only `advice_text`
- `Reset To Digest` replaces the current `advice_text` with the most recent `digest_text`
- rebuild updates `digest_text`
- rebuild should not clobber active user edits without explicit user intent

Recommended rebuild behavior:

- if the current `advice_text` exactly matches the old digest, replace it automatically with the new digest
- otherwise, preserve user-edited `advice_text` and surface that a newer digest is available

### Advice Application

Advice application is deterministic in v1 and affects only the draft prompt.

Supported apply modes:

- `Append`
- `Replace`

Semantics:

- `Append`: append `advice_text` to the existing draft prompt using one consistent separator rule
- `Replace`: overwrite the current draft prompt with `advice_text`

Not in scope for v1:

- LLM rewrite of the current draft
- automatic generation
- negative prompt mutation
- generation param overrides

Applying advice should:

- update only the current draft prompt state
- not trigger generation by itself

## Evidence Construction

Evidence should be built in the frontend from gallery item records and their cached generation metadata, then sent to the backend in normalized form.

The builder should:

- load all images for the selected gallery
- use all current gallery items as the v1 evidence set
- extract normalized fields from `params` and other stored metadata
- omit unavailable fields rather than fabricating them
- compute a stable `evidence_fingerprint`

V1 representation source is limited to data already available from the stored image cache and gallery records.

Future versions may expand evidence with:

- multimodal visual analysis
- image similarity clusters
- user annotations
- reference rankings

## Error Handling

### Empty Gallery

If the selected gallery has no images:

- disable rebuild
- show a clear empty-state message
- do not call the backend

### Missing Metadata

If some gallery items have partial metadata:

- include what can be normalized
- skip fields that are unavailable
- do not fail the entire evidence build unless nothing usable remains

### Build Failure

If digest generation fails:

- keep the last usable `digest_text` and `advice_text`
- set advisor status to `error`
- store `error_message`
- do not wipe timestamps from the last successful build

### Stale State

If gallery membership changes after a digest was built:

- mark the advisor `stale`
- keep `advice_text` usable
- let the user decide whether to rebuild immediately if `auto_advice` is off

## Testing Strategy

### Frontend Unit Tests

- advisor state store persists and reloads by `gallery_id`
- evidence builder normalizes gallery image metadata into the v1 shape
- `length_limit` slider clamps to `0..mode.maximum_len`
- gallery membership changes mark advisor state `stale`
- rebuild updates digest text, timestamp, and status
- user-edited advice survives rebuild when it diverges from the old digest
- `Reset To Digest` restores digest text
- `Append` and `Replace` mutate only the draft prompt

### Backend Unit Tests

- advisor digest route validates evidence payloads
- advisor service enforces `length_limit`
- advisor analysis prompt path returns `digest_text`
- backend error mapping returns structured failure details without leaking secrets

### Integration Tests

- frontend can build evidence from a gallery and send it to the backend
- backend returns a digest and frontend persists advisor state
- stale state is triggered after gallery mutation
- advisor apply updates draft prompt without triggering generation

### Manual Validation

1. Create a gallery and add representative generated images
2. Open the advisor section under `Negative Prompt Templates`
3. Set `temperature` and `length_limit`
4. Rebuild advisor and confirm `building -> fresh`
5. Confirm status shows relative time under 24 hours
6. Edit advice text manually
7. Apply via `Append`
8. Apply via `Replace`
9. Add another image to the gallery and confirm stale or additive refresh behavior according to `auto_advice`
10. Force an analysis failure and confirm prior advice remains usable while status becomes `error`

## Rollout

Implement in this order:

1. Extend mode config and `/api/modes` with `maximum_len`
2. Add advisor state persistence and evidence builder in the frontend
3. Add backend advisor digest route and service
4. Add advisor UI in `OptionsPanel`
5. Wire gallery mutation events to advisor freshness updates
6. Add deterministic apply behavior

## Acceptance

This work is complete when:

- a user can select a gallery and build a digest from gallery metadata evidence
- the system persists advisor state per `gallery_id`
- the advisor panel shows `fresh`, `stale`, `building`, and `error` plus recency
- the user can edit advice text directly
- `length_limit` is clamped by `mode.maximum_len`
- gallery membership changes affect advisor freshness
- `Append` and `Replace` update only the draft prompt
- the feature does not require backend gallery persistence
