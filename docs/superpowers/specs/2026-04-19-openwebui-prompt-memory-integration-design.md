# OpenWebUI Prompt And Memory Integration Design

## Summary

This design makes OpenWebUI the source of truth for prompt definitions and memory while keeping Stability Toys responsible for gallery evidence, advisor workflow, and image-generation policy.

The integration uses a single backend adapter boundary. Stability Toys should treat OpenWebUI as the only upstream LLM endpoint it knows about for this feature set. Prompt authoring, prompt lookup, and memory storage move out of Stability Toys. Stability Toys still assembles gallery-aware request envelopes and consumes the resulting completions.

This change is also the right place to harden the current advisor contract. The existing advisor path sends a weak prompt contract and is already producing unusable completions. Moving to OpenWebUI without tightening the advisor request shape would preserve the failure while only changing the endpoint.

The v1 object model is intentionally small:

- `OpenWebUIConfig`: upstream connection and defaults
- `GalleryPromptBinding`: per-gallery pointer to a canonical OpenWebUI prompt
- `AdvisorRequestEnvelope`: provider-neutral request assembled by Stability Toys

## Goals

- Make OpenWebUI the canonical home for prompts
- Make OpenWebUI the canonical home for memory
- Isolate memory by `gallery_id`
- Preserve one backend-owned LLM integration seam in Stability Toys
- Allow request-level overrides for:
  - `model`
  - `system_prompt`
  - `temperature`
  - `max_tokens`
  - gallery/advisor context payload
  - `memory_scope`
  - `session_id`
- Add per-gallery prompt binding without introducing real gallery prompt forks yet
- Fix the advisor request and response contract while introducing the OpenWebUI adapter

## Non-Goals

- Rebuilding OpenWebUI prompt management inside Stability Toys
- Rebuilding OpenWebUI memory storage inside Stability Toys
- Making prompt bodies editable per gallery in v1
- Supporting cross-gallery shared memory by default
- Turning OpenWebUI into the owner of image-generation policy
- Introducing a second non-OpenWebUI upstream chat provider in the same change
- Depending on autonomous model-managed memory behavior for correctness

## Current State

- [`server/advisor_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/advisor_service.py:31) builds the current advisor completion from a weak user prompt that embeds raw `evidence_json`.
- [`conf/modes.yml`](/Users/darkbit1001/workspace/Stability-Toys/conf/modes.yml:35) currently supplies a generic chat delegate prompt that is too weak for reliable advisor behavior.
- [`lcm-sr-ui/src/hooks/useGalleryAdvisor.js`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/hooks/useGalleryAdvisor.js:93) expects `response.meta.evidence_fingerprint`, but [`server/advisor_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/advisor_service.py:94) currently returns `evidence_fingerprint` at the top level.
- Chat transport is already backend-owned through [`backends/chat_client.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/chat_client.py:18), which talks to an OpenAI-compatible `/chat/completions` endpoint.
- Gallery state and advisor workflow are already Stability Toys concepts and should remain local to this repo.

The current boundary is close to correct, but the ownership is wrong. Stability Toys owns too much prompt behavior today while still not owning enough structure to make the advisor robust.

## Proposed Approach

Introduce an OpenWebUI adapter in the backend and move prompt and memory authority behind it.

The canonical flow becomes:

`gallery state -> gallery prompt binding + request overrides + normalized evidence -> OpenWebUI adapter -> OpenWebUI prompt/memory/model execution -> Stability Toys advisor state`

The split is:

- OpenWebUI owns:
  - prompt definitions
  - prompt lookup
  - memory storage and retrieval
  - upstream model execution
- Stability Toys owns:
  - gallery selection and identity
  - gallery evidence normalization
  - advisor controls and lifecycle
  - image-generation policy
  - per-gallery prompt bindings

This keeps one LLM seam while preventing Stability Toys from becoming a second prompt-management and memory-management system.

## Source Of Truth Model

### Prompts

Prompt bodies and templates live in OpenWebUI.

Stability Toys should not store copied prompt bodies in v1. Instead, it stores a small per-gallery binding object that points to an OpenWebUI-managed prompt.

This preserves a clean upgrade path:

- v1: each gallery points at a canonical global prompt
- later: each gallery may diverge into a real gallery-specific prompt if needed

### Memory

Memory lives in OpenWebUI and is isolated by gallery.

`gallery_id` is the primary memory namespace. Stability Toys should send it as the canonical `memory_scope` on every advisor or related chat-style request.

Default policy:

- no cross-gallery memory reuse
- no repo-wide shared prompt-advice memory
- no implicit fallback to global memory when gallery memory is empty

If a future feature wants broader recall, it should introduce that explicitly rather than widening scope silently.

## Prompt Binding Design

Each gallery should have a local prompt binding object that acts as a proxy to the canonical OpenWebUI prompt.

Proposed v1 shape:

```json
{
  "gallery_id": "gal_123",
  "prompt_ref": "advisor-default",
  "inherit_global": true,
  "system_append": null
}
```

Rules:

- `prompt_ref` points to an OpenWebUI-managed prompt
- `inherit_global: true` means use the referenced prompt as-is
- `system_append` is optional and request-scoped; it is not a prompt fork
- Stability Toys does not persist a full prompt body in v1

Effective resolution:

1. load the gallery binding
2. resolve the referenced OpenWebUI prompt
3. apply request-level overrides
4. send the final request envelope upstream

This gives the UI a per-gallery seam now without creating prompt drift.

## Request Envelope Design

Stability Toys should assemble a provider-neutral request envelope and then translate it to OpenWebUI at one backend seam.

Proposed v1 shape:

```json
{
  "gallery_id": "gal_123",
  "prompt_ref": "advisor-default",
  "memory_scope": "gal_123",
  "session_id": "adv_gal_123_1712790500",
  "model": "gemma3-1b",
  "system_prompt": "Optional request-specific override",
  "temperature": 0.4,
  "max_tokens": 120,
  "context_payload": {
    "version": 1,
    "gallery_id": "gal_123",
    "items": []
  }
}
```

Allowed request-level fields:

- `model`
- `system_prompt`
- `temperature`
- `max_tokens`
- `context_payload`
- `memory_scope`
- `session_id`

These fields are explicitly caller-controlled because the user wants them available per request instead of fully baked into OpenWebUI defaults.

The backend should remain responsible for validating and clamping these fields against Stability Toys runtime policy where needed.

`memory_scope` is a special case. The field is request-visible in v1, but Stability Toys should canonicalize it to the active `gallery_id` or reject the request if the caller tries to widen scope beyond the active gallery. This preserves the requested API seam without weakening gallery isolation.

## Advisor Prompt Contract

The advisor needs a stronger contract than it has today.

The new request path should not send a vague prompt plus raw JSON dump and hope the model infers the task. Instead, the adapter should build a stricter message set:

- a system layer derived from the resolved OpenWebUI prompt
- an optional request-level system append
- a structured user message that clearly states:
  - the task
  - the target output format
  - the length constraint
  - the gallery-scoped context payload
  - the instruction to produce style guidance, not code

The output contract should require a plain-text digest suitable for direct use as `digest_text`.

Minimum advisor prompt requirements:

- summarize stable themes rather than outliers
- produce reusable style guidance rather than code or explanations about code
- stay within `max_tokens`
- tolerate sparse evidence
- mention recurring generation tendencies only when they are stable enough to matter

This is not yet a structured `digest_payload` design. V1 still returns text, but the text contract must be explicit.

## OpenWebUI Adapter Design

Add a dedicated backend adapter layer rather than scattering OpenWebUI-specific logic across route handlers.

Recommended new responsibilities:

- resolve OpenWebUI config
- resolve the effective prompt reference
- attach gallery-scoped memory identity
- translate the Stability Toys request envelope into the OpenWebUI request format
- normalize the response back into Stability Toys shapes

Recommended boundaries:

- route handlers should not know OpenWebUI-specific prompt or memory details
- frontend should not call OpenWebUI directly
- `backends/chat_client.py` remains the transport primitive unless OpenWebUI-specific behavior forces a wrapper above it

The adapter should be the only place that knows:

- how prompt references are resolved
- how memory scope/session identity are encoded
- which OpenWebUI-specific endpoints beyond chat-completions are needed

## API Shape

The advisor route should continue to look like a Stability Toys route, not an OpenWebUI passthrough.

Recommended request:

```json
{
  "gallery_id": "gal_123",
  "evidence": {
    "version": 1,
    "gallery_id": "gal_123",
    "items": []
  },
  "prompt_ref": "advisor-default",
  "model": "gemma3-1b",
  "system_prompt": "Optional override",
  "temperature": 0.4,
  "length_limit": 120,
  "session_id": "adv_gal_123_1712790500"
}
```

Recommended response:

```json
{
  "gallery_id": "gal_123",
  "digest_text": "Style digest...",
  "meta": {
    "evidence_fingerprint": "sha256:...",
    "prompt_ref": "advisor-default",
    "memory_scope": "gal_123",
    "model": "gemma3-1b"
  }
}
```

This response shape fixes the current frontend/backend mismatch and leaves room for later metadata growth without breaking the main payload contract.

## Runtime Config

Introduce explicit OpenWebUI runtime config in Stability Toys.

Proposed shape:

```yaml
openwebui:
  base_url: "http://openwebui:3000/api"
  api_key_env: "OPENWEBUI_API_KEY"
  default_prompt_ref: "advisor-default"
  default_model: "gemma3-1b"
```

Purpose:

- one canonical upstream endpoint
- one canonical API key source
- one default prompt reference
- optional default model

Important boundary:

- this config selects the OpenWebUI control plane
- it does not replace mode-level image-generation config
- it should not be overloaded into a second generic provider registry in v1

## Persistence Design

Stability Toys should persist only the binding state and local advisor workflow state it truly owns.

Persist locally:

- `GalleryPromptBinding`
- `AdvisorState`

Do not persist locally:

- full prompt bodies fetched from OpenWebUI
- canonical memory contents

Caching prompt definitions locally for convenience should be deferred. Prompt caching introduces invalidation and conflict rules immediately, and the user explicitly wants OpenWebUI to be the source of truth.

## Error Handling

The integration should fail in predictable ways:

- if OpenWebUI prompt resolution fails, surface a request error and keep the last usable digest
- if memory scope cannot be attached, fail the request rather than silently widening scope
- if the upstream model rejects a request override, surface the backend error directly
- if OpenWebUI is unavailable, keep local gallery and advisor state intact and mark advisor status as error

For advisor rebuild failures:

- preserve previous `digest_text`
- preserve previous `advice_text`
- update local `status` to `error`
- store a user-visible `error_message`

## Risks

### OpenWebUI Memory Is Experimental

OpenWebUI memory is acceptable as the canonical persistence layer for this design, but autonomous memory behavior should not be treated as reliable enough to carry core product logic. The request path should work even if memory recall quality is uneven.

### Direct API Behavior Can Differ From Browser Behavior

OpenWebUI features such as filters, prompt expansion, or completion lifecycle hooks may behave differently when called by API instead of through the OpenWebUI browser. The adapter should isolate this risk so Stability Toys does not leak OpenWebUI-specific assumptions everywhere else.

### Too Many Request Knobs Can Create Ambiguity

Allowing request-level `model`, `system_prompt`, `temperature`, and `max_tokens` is intentional here, but it means the adapter must define strict precedence rules. Without that, debugging completions will become difficult.

## Precedence Rules

Recommended precedence:

`request override -> gallery binding -> OpenWebUI default -> hardcoded fallback`

Applied by field:

- `prompt_ref`: request value if present, else gallery binding, else `openwebui.default_prompt_ref`
- `model`: request value if present, else OpenWebUI default model
- `system_prompt`: request append after the resolved prompt, not a replacement for the canonical prompt body
- `temperature`: request value if present, else route default
- `max_tokens`: request value clamped by Stability Toys runtime constraints
- `memory_scope`: request value if present only when it resolves to the active `gallery_id`; otherwise reject the request; fallback is the active `gallery_id`

These rules should be implemented once in the adapter and reused by advisor and any future OpenWebUI-backed chat surfaces.

## Phased Rollout

### Phase 1

- add OpenWebUI runtime config
- add backend adapter
- harden advisor request contract
- fix advisor response metadata shape
- use one global OpenWebUI prompt with per-gallery bindings
- isolate memory by `gallery_id`

### Phase 2

- add explicit gallery-level `system_append` support in UI
- expose prompt binding controls in the advisor UI
- add richer upstream metadata capture for observability

### Phase 3

- allow true gallery prompt overrides when needed
- consider additional OpenWebUI-backed chat surfaces beyond the advisor

## Acceptance Criteria

- Stability Toys can rebuild an advisor digest through OpenWebUI without directly managing prompt bodies or memory storage
- prompt and memory source of truth are both OpenWebUI
- memory is isolated by `gallery_id`
- each gallery has a local prompt binding object that points at a canonical OpenWebUI prompt
- advisor requests can override `model`, `system_prompt`, `temperature`, `max_tokens`, `context_payload`, `memory_scope`, and `session_id`
- the advisor no longer emits code-like garbage under sparse default settings because the request contract is explicit
- backend responses return metadata under `meta`, and the frontend consumes that shape consistently

## Recommendation

Proceed with the OpenWebUI adapter as the next integration boundary and bundle advisor contract hardening into the same initiative.

This is the smallest design that:

- avoids building a second prompt/memory platform
- preserves Stability Toys' existing ownership boundaries
- supports many models in production without spreading prompt policy across repos
- leaves a clean upgrade path for true gallery-specific prompts later
