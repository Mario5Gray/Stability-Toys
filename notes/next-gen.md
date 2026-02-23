# Next-Gen Refactor Notes (From Dialogues)

## Goal
Build a smooth, consistent UI/UX for generation + refinement, with an arcade‑like menu clarity. Reduce unnecessary steps, avoid surprises, and support complex workflows (generate, refine, history, comfy) via a simple state machine.

## Core UX Principles
- **Predictable selection**: no auto‑selection hijacks on background jobs.
- **Non‑destructive history browsing**: browsing history should never mutate live params or overwrite current image state.
- **Stable display**: prefer server URLs; blobs are cache acceleration only and must be regenerated after refresh.
- **Explicit state transitions**: changes are driven by clear events, not incidental side effects.

## Proposed State Model (Two Orthogonal Axes)
- **View Mode**: `LIVE` vs `HISTORY`
- **Edit Mode**: `DRAFT` vs `LOCKED`

This yields 4 states:
- `LIVE + DRAFT`: edits update live params
- `LIVE + LOCKED`: view‑only
- `HISTORY + DRAFT`: browsing while preparing a refinement
- `HISTORY + LOCKED`: pure browsing

## Data Model (DAG / Tree)
**Root identity**:
- `RootKey = (seed, prompt)`

**Refinement coordinate**:
- `RefineKey = (steps, cfg)`

**Node**:
- `id`
- `root: { seed, prompt }`
- `refine: { steps, cfg }`
- `meta: { size, mode_name, ... }`
- `image: { serverImageUrl, serverImageKey }`
- `parentId`
- `createdAt`

**Rules**:
- New seed or prompt → new root.
- Descendants change `steps/cfg` (primary ordering).
- `mode_name/size` are secondary, queryable but not identity.
- Prefer tree (1 parent); only extend to DAG if explicit merging is needed.

## Parent Selection (for refinements)
- Choose nearest existing node with same RootKey via metric:
  - `d = w_steps * |Δsteps| + w_cfg * |Δcfg|`

## Immutable Data / No Mutable Assets
- Avoid storing mutable assets like `filename` in state.
- Derive filenames only at export/download time.

## History Semantics
- **Immutable history** by default (history entries are outputs only).
- **Live state** is separate and mutable (draft edits).
- If desired, last history entry can mirror live state, but should be explicit.

## URL & Cache Strategy
- Persist **serverImageUrl**; never persist `blob:` URLs.
- On refresh: replace blobs with server URL or regenerate blob from IndexedDB.
- Blob URLs are ephemeral acceleration; UI should always be able to fall back to server URL.

## Known Pain Points Observed
- Stale blob URLs after refresh causing broken images.
- Auto‑select on comfy completion hijacking user selection.
- History navigation mutating params (destructive browsing).
- Cache hydration overwriting newer image URL due to stale cacheKey associations.

## Implementation Ideas
- Centralize `resolveDisplayUrl(msg)`; use in all rendering paths.
- Maintain `msgId -> cacheKey` mapping to prevent stale hydration overwrites.
- Guard updates from background jobs if current selection or history index changed.
- Persist draft edits to message state on selection change.

## Open Decisions
- Whether to make last history entry always mirror live state.
- Whether to allow seed changes inside a lineage or treat as new root.
- Whether to support true DAG merges (multiple parents).
