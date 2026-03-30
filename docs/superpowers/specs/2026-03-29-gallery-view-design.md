# Gallery View — Design Spec

**Date:** 2026-03-29
**Status:** Approved for implementation

---

## Goal

Let users capture generated images into named galleries. A gallery is opt-in: images are never added automatically. The feature lives entirely in the existing UI without replacing any current view — galleries are opened as a lightbox overlay.

---

## Scope

**In scope (v1):**

- Create and name galleries
- Select an active gallery from a dropdown in the options panel
- Send a generated image to the active gallery via a pill action
- View a gallery in a lightbox with a 5-column grid, pagination, and per-image metadata
- Open any gallery image in a new tab
- Persist gallery data durably in a dedicated browser IndexedDB that is never evicted or cleared by normal app operations

**Out of scope (v1):**

- Backend sync (shape described below for v2 reference)
- Deleting or renaming galleries
- Removing an image from a gallery
- Sharing galleries between devices

---

## Data Layer

### Gallery list — localStorage

Stored in `localStorage` under key `lcm-galleries`.

```json
[
  { "id": "gal_<uuid>", "name": "Nature", "createdAt": 1711670000000 },
  { "id": "gal_<uuid>", "name": "Portraits", "createdAt": 1711671000000 }
]
```

Gallery names are truncated to 16 characters at creation time. This truncation rule applies to all dropdowns in the UI that display gallery names.

The active gallery selection is persisted in `localStorage` under key `lcm-active-gallery` (string gallery id, or absent/`null` for "none"). This survives page refresh without re-selecting.

### Image assignments — separate `lcm-galleries` IndexedDB

Gallery item rows live in a **dedicated** `lcm-galleries` IndexedDB (DB version 1), completely separate from `lcm-image-cache`. This isolation means:

- Gallery contents are **never evicted** — the `_evictIfNeeded` logic in `cache.js` only touches `lcm-image-cache` and cannot reach this DB.
- Gallery contents **survive "Clear Messages"** — the `cache.clear()` call clears `lcm-image-cache`; it has no connection to `lcm-galleries`.
- No version coordination with the existing cache is needed. `cache.js` is **not modified**.

Schema of the `gallery_items` object store:

```text
gallery_items {
  id:             string   // keyPath — auto-generated UUID (crypto.randomUUID())
  galleryId:      string
  cacheKey:       string   // hint for local blob lookup; may become stale if blob evicted
  serverImageUrl: string?  // fully-resolved URL stored at assignment time
                           // e.g. `${apiBase}/storage/${serverImageKey}`
                           // derived using the same pattern as api.js:208 at call time
  params:         object   // snapshot: { prompt, seed, size, steps, cfg, ... }
  addedAt:        number   // Date.now() at time of assignment
}

indexes:
  "galleryId"  (non-unique) — per-gallery queries
  "cacheKey"   (non-unique) — detect if a cacheKey is already in a given gallery
```

Using a UUID keyPath means the same image can appear in multiple galleries without collision. The `cacheKey` index allows a pre-insert check for `(galleryId, cacheKey)` duplicates if desired.

### Image display fallback

When rendering a gallery thumbnail or viewer:

1. Try to resolve a blob URL from `lcm-image-cache` using `cacheKey`.
2. If the blob is absent (evicted or cleared), fall back to `serverImageUrl`.
3. If both are absent, show a broken-image placeholder.

This means images with no `serverImageUrl` (e.g. generated before server-side caching was enabled) degrade to placeholder under cache pressure. The spec accepts this as a known limitation of v1.

### Backend sync shape (v2, not implemented in v1)

```text
POST /api/galleries
{
  "galleries": [{ id, name, createdAt }],
  "items": [{ id, galleryId, cacheKey, serverImageUrl, params, addedAt }]
}
```

---

## State — `useGalleries` hook

**File:** `lcm-sr-ui/src/hooks/useGalleries.js`

Owned by `App.jsx` and passed to child components as `galleryState`. Manages its own `lcm-galleries` DB handle independently.

```js
useGalleries() → {
  galleries,              // [{ id, name, createdAt }]  — from localStorage
  activeGalleryId,        // string | null  — from localStorage (lcm-active-gallery)
  setActiveGalleryId,     // (id: string | null) => void  — also persists to localStorage
  createGallery,          // (name: string) => void  — truncates to 16 chars, auto-selects
  addToGallery,           // (cacheKey, { serverImageUrl, params }) => Promise<void>
  getGalleryImages,       // (galleryId) => Promise<GalleryItem[]>
}
```

- `galleries` and `activeGalleryId` are React state initialised from `localStorage` on mount.
- `createGallery` writes to `lcm-galleries` localStorage key synchronously, generates a UUID id, and calls `setActiveGalleryId`.
- `setActiveGalleryId` updates both React state and `localStorage` key `lcm-active-gallery`.
- `addToGallery` opens `lcm-galleries` IndexedDB and puts a new row with a fresh `crypto.randomUUID()` as the keyPath id.
- `getGalleryImages` queries the `galleryId` index and returns all matching rows.

---

## Components

### New — `src/components/gallery/`

#### `GalleryCreatePopover.jsx`

Triggered by the [+] button in the top tab bar. Renders as a small popover (not a modal):

- Background color matches the app (`bg-background`)
- Label: "Gallery name"
- `<Input>` with `maxLength={16}`, same style as existing inputs
- Confirm on Enter or a "Create" button
- On confirm: calls `createGallery(name)`, closes the popover

#### `GalleryLightbox.jsx`

Full-viewport fixed overlay. Contains the toolbar, grid, and viewer sub-components.

- `position: fixed`, `inset: 0`, `z-index` above everything
- Configurable background opacity via toolbar slider (range 0.7–1.0, default 0.95)
- Closes on **ESC** (keydown listener on mount) or clicking the **[X]** button (top-right)
- Closing the lightbox also closes any child windows opened via `window.open`; track those refs in a `useRef` array and call `.close()` on each
- Accepts `galleryId` prop; fetches images via `getGalleryImages(galleryId)` on open

**Top toolbar** (left to right):

- Gallery name (read-only label, truncated to 16 chars)
- Opacity slider (`<input type="range">`, same accent color as existing sliders)
- Reserved button slot area (empty in v1, comment-marked for future additions)
- [X] close button (top-right, lucide `X` icon)

#### `GalleryGrid.jsx`

Renders a 5-column CSS grid of image thumbnails.

- 20 images per page; pagination controls below the grid (Prev / Page N of M / Next)
- Each cell resolves its display URL using the fallback order: cache blob → `serverImageUrl` → placeholder
- Click on image → renders `GalleryImageViewer` inside the lightbox
- Keyboard: Space on a focused/hovered thumbnail → `window.open(resolvedUrl, '_blank')`
- Thumbnails use `object-fit: cover` in a fixed square cell

#### `GalleryImageViewer.jsx`

Single image view within the lightbox, opened by clicking a grid thumbnail.

- Full image centered in the available area, using the same blob → `serverImageUrl` → placeholder fallback
- **Metadata bar:** absolutely positioned at the bottom, hidden by default. Becomes visible when the pointer enters the lower 20% of the image container — detected by comparing `e.clientY` against the container's `getBoundingClientRect`. Fields: prompt, seed, size, steps, cfg, backend, addedAt.
- Back arrow (top-left within viewer) returns to the grid
- Spacebar → `window.open(resolvedUrl, '_blank')`

---

### Changes to existing files

#### `App.jsx`

1. Add `useGalleries()` and store result as `galleryState`.
2. In the tab bar, after the "Configuration" tab trigger, add:
   - A `<GalleryCreatePopover>` trigger: `FolderPlus` lucide icon + `[+]` label.
   - For each gallery in `galleryState.galleries`, render a clickable folder tab (`Folder` icon + truncated name). Clicking sets local state `openGalleryId`.
3. Render `<GalleryLightbox>` conditionally when `openGalleryId` is set.
4. Pass `galleryState` to `<OptionsPanel>` and `activeGalleryId` + `onAddToGallery` to `<ChatContainer>` → `<MessageBubble>`.

#### `OptionsPanel.jsx`

Add a `GallerySelector` block **below the draft-prompt textarea**.

- Uses the existing `<Select>` / `<SelectTrigger>` / `<SelectItem>` components (same style as Negative Prompt Template selector)
- First item: value `null`, label `None`
- Remaining items: one per gallery, label truncated to 16 chars
- `onValueChange` calls `galleryState.setActiveGalleryId`

#### `MessageBubble.jsx`

In the metadata pills row (~line 312), add a `→ Gallery` pill **next to the Download link**:

```jsx
{onAddToGallery && (
  <button
    className={`... ${activeGalleryId ? '' : 'opacity-40 cursor-not-allowed'}`}
    disabled={!activeGalleryId}
    onClick={(e) => {
      e.stopPropagation();
      onAddToGallery(msg.meta?.cacheKey, {
        serverImageUrl: msg.serverImageUrl ?? null,
        params: msg.params ?? {},
      });
    }}
    title={activeGalleryId ? 'Add to gallery' : 'Select a gallery first'}
  >
    <MoveRight className="h-3 w-3 inline mr-0.5" />
    Gallery
  </button>
)}
```

- Uses `MoveRight` from lucide-react (graphic arrow)
- Passes `serverImageUrl` directly (already resolved on the message object — no helper call needed at display time)
- Disabled and dimmed when `activeGalleryId` is null
- Receives `activeGalleryId` and `onAddToGallery` as new props

---

## Prop flow

```text
App
 ├─ useGalleries() → galleryState
 ├─ GalleryCreatePopover  (galleryState.createGallery)
 ├─ GalleryLightbox       (openGalleryId, galleryState.getGalleryImages, onClose)
 ├─ OptionsPanel
 │   └─ GallerySelector   (galleryState.galleries, activeGalleryId, setActiveGalleryId)
 └─ ChatContainer
     └─ MessageBubble     (activeGalleryId, onAddToGallery)
```

---

## Truncation rule

Gallery names are truncated to **16 characters** at creation (in `createGallery`). All UI surfaces that display gallery names — the tab bar, the options dropdown, the lightbox toolbar label — truncate at 16 chars with CSS `truncate` as a visual safety net. This rule applies to every dropdown in the app that shows gallery names.

---

## Error and edge cases

| Situation | Behaviour |
|---|---|
| Active gallery is "none" | `→ Gallery` pill is disabled and dimmed |
| `cacheKey` is absent on a message | `→ Gallery` pill is not rendered |
| Gallery has no images | `GalleryGrid` shows an empty state: "No images in this gallery yet" |
| Blob evicted, `serverImageUrl` present | Thumbnail and viewer use `serverImageUrl`; spacebar/click opens that URL |
| Blob evicted, no `serverImageUrl` | Thumbnail shows broken-image placeholder; click/spacebar disabled for that item |
| Two galleries created with the same name | Allowed — they get distinct UUIDs |
| Same image added to same gallery twice | `addToGallery` checks the `cacheKey` index before inserting; silently no-ops on duplicate |
| Same image added to two different galleries | Two rows with distinct UUIDs — allowed by design |
| ESC pressed while a child window is open | Lightbox closes and calls `.close()` on all tracked child window refs |

---

## Files changed / created

| File | Change |
|---|---|
| `src/hooks/useGalleries.js` | **new** — manages `lcm-galleries` DB + localStorage |
| `src/components/gallery/GalleryCreatePopover.jsx` | **new** |
| `src/components/gallery/GalleryLightbox.jsx` | **new** |
| `src/components/gallery/GalleryGrid.jsx` | **new** |
| `src/components/gallery/GalleryImageViewer.jsx` | **new** |
| `src/App.jsx` | modify — `useGalleries`, tab bar buttons, lightbox render, prop threading |
| `src/components/options/OptionsPanel.jsx` | modify — add `GallerySelector` |
| `src/components/chat/MessageBubble.jsx` | modify — add `→ Gallery` pill |

`src/utils/cache.js` is **not modified**. The gallery DB is fully independent.
