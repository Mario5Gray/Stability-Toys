# Resolution Sets Design

## Summary

This design replaces the current global frontend size list with config-defined resolution sets that can be assigned per mode. The immediate use case is SDXL: it should surface a curated set of sizes it is actually good at generating, show aspect ratios alongside the pixel dimensions, reset to the mode default on mode switch, and reject unsupported sizes at the backend.

The source of truth is configuration, not inference and not runtime CSV parsing. Modes select a named `resolution_set`, the UI renders the active mode’s choices, and the backend enforces that submitted sizes are in the active mode’s resolved set.

## Goals

- Support named resolution sets in configuration
- Let each mode opt into a specific resolution set
- Fall back to a shared default set when a mode does not specify one
- Show resolution and aspect ratio together in the UI
- Limit the visible selector viewport to 5 rows while allowing scrolling
- Reset the current size to the mode’s `default_size` when switching modes
- Reject unsupported sizes at the backend with a clear error

## Non-Goals

- Infer resolution sets from model names or checkpoint metadata
- Parse `resolutions_sdxl.csv` at runtime in the app
- Allow arbitrary unsupported sizes through the UI when a mode declares a set
- Add a second independent size-policy system outside `modes.yml`
- Rework non-mode workflow sizing beyond the active mode generation path

## Current State

- The UI uses one global `SIZE_OPTIONS` list in [constants.js](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/utils/constants.js)
- Modes only expose `default_size` and optional `recommended_size`
- The backend applies `default_size` from the active mode when the request used the environment default
- The backend does not currently validate a requested size against a mode-specific allowed set
- `resolutions_sdxl.csv` exists as a useful seed dataset for SDXL-friendly resolutions and aspect ratios

## Proposed Approach

Add top-level `resolution_sets` to mode configuration and a per-mode `resolution_set` selector:

- `resolution_sets` defines named curated lists of `{ size, aspect_ratio }`
- modes can declare `resolution_set: sdxl`
- modes that omit it resolve to `default`
- `default_size` must be a member of the resolved set

The backend exposes the resolved set in `/api/modes` and rejects unsupported submitted sizes in `/generate`. The frontend switches from static `SIZE_OPTIONS` to the active mode’s resolved entries and renders `1024×1024 • 1:1` style labels in a scrollable selector with only 5 visible rows.

## Design

### 1. Config-owned resolution sets

Files in scope:

- `conf/modes.yml`
- `conf/modes.yaml.example`
- `server/mode_config.py`
- tests covering mode config parsing/validation

Design:

- Add a top-level `resolution_sets` mapping:

```yaml
resolution_sets:
  default:
    - size: "512x512"
      aspect_ratio: "1:1"
  sdxl:
    - size: "1024x1024"
      aspect_ratio: "1:1"
    - size: "896x1152"
      aspect_ratio: "7:9"
```

- Add `resolution_set` to each mode:

```yaml
modes:
  SDXL:
    resolution_set: sdxl
    default_size: "1024x1024"
```

- Resolve a mode’s effective set as:
  - declared set if present
  - otherwise `default`

- Validate at config load time:
  - referenced `resolution_set` must exist
  - `default` set must exist
  - `default_size` must be present in the resolved set for that mode

Expected outcome:

- resolution policy is explicit, centralized, and per mode
- SDXL-friendly sizes live in config rather than UI code

### 2. Seed SDXL set from CSV, but do not depend on CSV at runtime

Files in scope:

- `resolutions_sdxl.csv`
- `conf/modes.yml`
- optional documentation note if needed

Design:

- Use `resolutions_sdxl.csv` as a curation source for the initial `sdxl` entries.
- Copy the chosen rows into config as normalized values.
- Do not make frontend or backend parse the CSV during normal app execution.

Expected outcome:

- the repo keeps the CSV as a reference dataset
- runtime behavior remains config-driven and deterministic

### 3. Backend/API enforcement

Files in scope:

- `server/mode_config.py`
- `server/model_routes.py`
- `server/lcm_sr_server.py`
- tests for API serialization and request validation

Design:

- Extend the serialized `/api/modes` payload to include the mode’s resolved resolution entries and its `resolution_set` name.
- In `/generate`, after mode switching/default application, validate `req.size` against the active mode’s allowed sizes.
- Reject unsupported sizes with `400` and a clear message such as:
  - `size '768x768' is not allowed for mode 'SDXL'`

Expected outcome:

- manual API calls cannot bypass mode sizing policy
- frontend and backend enforce the same contract

### 4. Mode-aware frontend selector

Files in scope:

- `lcm-sr-ui/src/components/options/SizeSelector.jsx`
- `lcm-sr-ui/src/components/options/OptionsPanel.jsx`
- `lcm-sr-ui/src/hooks/useModeConfig.js`
- `lcm-sr-ui/src/utils/helpers.js`
- `lcm-sr-ui/src/utils/constants.js`
- frontend tests for rendering and mode switching

Design:

- Replace the global `SIZE_OPTIONS` usage in the generation UI with options derived from `modeState.activeMode`.
- Each entry renders as:
  - `1024×1024 • 1:1`
- The dropdown content should show only 5 rows of visible height and scroll for the rest.
- On mode switch:
  - size resets to the new mode’s `default_size`
- If a selected/remembered size is not present in the new set:
  - immediately replace it with the mode default

Expected outcome:

- SDXL users see the right sizes directly in the UI
- aspect ratio context is visible without extra clicks
- the selector stays compact even with a long SDXL set

## Testing Strategy

### Backend tests

- Add mode-config tests for:
  - parsing `resolution_sets`
  - resolving a mode’s `resolution_set`
  - failing on unknown set names
  - failing when `default_size` is not in the resolved set
- Add route/server tests for:
  - `/api/modes` including the resolved set metadata
  - `/generate` rejecting unsupported sizes for the active mode

### Frontend tests

- Add/adjust tests for:
  - rendering size labels with aspect ratio
  - using mode-provided size options instead of the global list
  - resetting to mode `default_size` on switch
  - keeping the selector visually constrained to 5 rows

### Manual validation

Validate the following:

1. Switching to SDXL resets the size control to the SDXL mode default.
2. The size dropdown shows SDXL-specific entries with aspect ratio labels.
3. Only 5 rows are visible before scrolling.
4. Submitting an unsupported size through the API returns `400`.

## Risks And Tradeoffs

### Config grows more detailed

That is intentional. The sizing policy is domain knowledge and belongs in configuration where it can be reviewed and edited explicitly.

### Existing saved or custom sizes may become invalid under stricter enforcement

That is also intentional. Invalid sizes should fail clearly or reset to a valid default instead of drifting through the system.

### One set may not fit all SDXL checkpoints forever

That is why the system is named-set based rather than hardcoded to one SDXL assumption. Additional sets can be added later without changing the runtime model.

## Rollout

Implement in this order:

1. Add config model support for `resolution_sets` and `resolution_set`
2. Seed and validate the SDXL/default sets in config
3. Expose resolved sets via `/api/modes`
4. Enforce size rejection in `/generate`
5. Update the frontend selector to use mode-aware entries with aspect ratios
6. Add tests and manual validation

## Acceptance

This work is complete when:

- modes can declare a named `resolution_set`
- SDXL uses a curated SDXL resolution set from config
- the UI shows `resolution • aspect ratio` labels with a 5-row visible selector
- switching modes resets size to the mode default
- the backend rejects unsupported sizes for the active mode
