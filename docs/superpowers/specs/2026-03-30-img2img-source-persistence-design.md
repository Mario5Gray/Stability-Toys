# Img2img Source Persistence — Design Spec

**Date:** 2026-03-30
**Status:** Proposed for implementation

---

## Goal

Persist the current img2img source image across page reloads and later visits so the Init Image UI and img2img workflow remain available after restart.

This must support both:

- chat-origin images already present in app history
- locally uploaded images dropped into the img2img slot

The persisted source is tied to the image that initiated it. It is not just transient page state.

---

## Scope

**In scope (v1):**

- Persist the active img2img source across reloads and later visits
- Support both chat-origin and uploaded source images
- Restore the Init Image preview UI on app startup
- Persist a per-source `defaultDenoiseStrength`
- Introduce a global fallback default denoise strength of `0.5`

**Out of scope (v1):**

- Multiple concurrent saved img2img sources
- User-facing management UI for old img2img sources
- Backend sync of img2img source state
- Cross-device sharing of img2img source state

---

## Data Model

### Global fallback

Add a frontend constant:

```js
DEFAULT_IMG2IMG_DENOISE_STRENGTH = 0.5
```

This is the final fallback when no image params, draft params, or source default are available.

### Active source pointer

Persist the current active source id in `localStorage`:

```text
key: lcm-active-img2img-source
value: "<source-id>" | null
```

This is just a pointer. The durable record lives in IndexedDB.

### Source storage

Create a dedicated IndexedDB database for img2img source persistence:

```text
DB name: lcm-img2img
DB version: 1
store: img2img_sources
```

Schema:

```text
img2img_sources {
  id:                     string   // keyPath, crypto.randomUUID()
  originType:             string   // "chat" | "upload"
  originMessageId:        string?  // set when source came from a chat image
  blob:                   Blob     // durable local source bytes
  mimeType:               string
  filename:               string
  cacheKey:               string?  // for chat-origin lookup/traceability
  serverImageUrl:         string?  // optional fallback/provenance
  defaultDenoiseStrength: number   // source-local fallback hint
  createdAt:              number
  updatedAt:              number
}
```

Indexes:

- `originMessageId` non-unique
- `originType` non-unique
- `updatedAt` non-unique

---

## Source Semantics

There is exactly one active img2img source at a time in v1.

Selecting a new source replaces the active pointer and makes that source current. Older persisted rows may remain unless explicitly deleted during replacement, but only the active source is restored automatically.

The source record owns:

- restoration of the image bytes
- provenance (`chat` vs `upload`)
- fallback default denoise value

The source record does **not** own the authoritative denoise strength for generated images. Generated and selected image params remain authoritative.

---

## Denoise Strength Precedence

When the UI needs an effective denoise strength:

1. selected image params `denoiseStrength`
2. draft params `denoiseStrength`
3. active img2img source `defaultDenoiseStrength`
4. global constant `DEFAULT_IMG2IMG_DENOISE_STRENGTH` (`0.5`)

This keeps image params authoritative while still allowing a restored source to supply a sensible fallback after reload.

---

## Lifecycle

### Upload-origin source

When the user drops or uploads a file for img2img:

1. Create a new source row in `img2img_sources`
2. Persist the uploaded `Blob`
3. Set `originType = "upload"`
4. Set `defaultDenoiseStrength = 0.5`
5. Write the new source id to `lcm-active-img2img-source`
6. Rebuild an `objectUrl` from the stored blob for UI display

### Chat-origin source

When the user promotes a chat image into the img2img source slot:

1. Resolve image bytes from the best available source:
   - existing blob/cache if available
   - otherwise fetch from `serverImageUrl`
2. Create a new source row in `img2img_sources`
3. Persist the resolved `Blob`
4. Set `originType = "chat"`
5. Record `originMessageId`, `cacheKey`, and `serverImageUrl` when available
6. Set `defaultDenoiseStrength = 0.5`
7. Write the new source id to `lcm-active-img2img-source`

### App startup restore

On app load:

1. Read `lcm-active-img2img-source`
2. If absent, do nothing
3. If present, load that row from `img2img_sources`
4. If found, rebuild an `objectUrl` from its blob and restore the Init Image UI
5. If missing/corrupt, clear the localStorage pointer and leave img2img inactive

### Clear behavior

When the user clears the Init Image:

1. Revoke the live `objectUrl`
2. Clear in-memory `initImage`
3. Remove `lcm-active-img2img-source`
4. Delete the active source row from `img2img_sources`

This keeps the user-visible clear action aligned with durable storage.

---

## State and Ownership

### `App.jsx`

`App.jsx` remains the owner of current img2img source state, but the source is now bootstrapped from durable storage instead of plain React state only.

In-memory `initImage` shape remains conceptually similar:

```js
{
  sourceId,
  originType,
  file,       // reconstructed File from persisted Blob where needed
  objectUrl,
  filename,
  cacheKey,
  serverImageUrl,
}
```

This lets existing UI consumers keep working with minimal surface change.

### Hook / helper boundary

Persistence should live in a dedicated helper module, not inside `App.jsx`:

- `saveSource(...)`
- `loadActiveSource()`
- `setActiveSourceId(...)`
- `clearActiveSource()`
- `deleteSource(id)`

`App.jsx` orchestrates; the storage helper persists and restores.

---

## File Changes

| File | Change |
|---|---|
| `lcm-sr-ui/src/utils/constants.js` | add `DEFAULT_IMG2IMG_DENOISE_STRENGTH = 0.5` |
| `lcm-sr-ui/src/utils/img2imgSourceStore.js` | **new** — IndexedDB + localStorage persistence helper |
| `lcm-sr-ui/src/App.jsx` | modify — restore, persist, clear, and thread active img2img source |
| `lcm-sr-ui/src/hooks/useGenerationParams.js` | modify — accept/use source default in denoise precedence |
| `lcm-sr-ui/src/components/chat/ChatDropzone.jsx` | modify — route upload-origin init images through persisted source creation |
| `lcm-sr-ui/src/components/options/OptionsPanel.jsx` | no structural change expected; consumes restored init image as before |

---

## Edge Cases

| Situation | Behaviour |
|---|---|
| Active source pointer exists but IndexedDB row is missing | Clear pointer and show no Init Image |
| Persisted blob is unreadable/corrupt | Clear pointer and show no Init Image |
| Chat-origin image has no cache blob but has `serverImageUrl` | Fetch bytes once at source creation time and persist them |
| Chat-origin image has neither cache blob nor `serverImageUrl` | Cannot promote durably; do not create source |
| Uploaded source survives reload | Yes, because bytes are stored in `img2img_sources.blob` |
| User replaces current source | New source becomes active; old active source is no longer restored automatically |
| Selected/generated image has explicit denoise | That value overrides source default |
| No explicit denoise anywhere | Fall back to global `0.5` |

---

## Acceptance

1. Upload an img2img source image, reload the page, and verify the Init Image preview still appears.
2. Use a chat-origin image as the img2img source, reload the page, and verify the Init Image preview still appears.
3. After reload, verify the denoise control resolves from image params first, otherwise from source default, otherwise from global `0.5`.
4. Clear the Init Image, reload the page, and verify the source does not return.
5. Restart the browser and verify later-visit restoration still works for both upload-origin and chat-origin sources.
