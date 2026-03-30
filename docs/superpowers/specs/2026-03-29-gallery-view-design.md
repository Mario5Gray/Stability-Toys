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
- Persist gallery data in the browser using the existing IndexedDB cache infrastructure

**Out of scope (v1):**

- Backend sync (shape described below for v2 reference)
- Deleting or renaming galleries
- Removing an image from a gallery
- Sharing galleries between devices

---

## Data Layer

### Gallery list

Stored in `localStorage` under key `lcm-galleries`.

```json
[
  { "id": "gal_<uuid>", "name": "Nature", "createdAt": 1711670000000 },
  { "id": "gal_<uuid>", "name": "Portraits", "createdAt": 1711671000000 }
]
```

Gallery names are truncated to 16 characters at creation time. This truncation rule applies to all dropdowns in the UI that display gallery names.

### Image assignments

New `gallery_items` object store added to the existing `lcm-image-cache` IndexedDB, bumping the DB version from 2 to **3**. The upgrade follows the existing `onupgradeneeded` pattern in `cache.js` and touches no existing data.

```text
gallery_items store {
  id:             string   // keyPath, same value as cacheKey
  galleryId:      string
  cacheKey:       string   // key into imageMeta / imageBlobs stores
  serverImageKey: string?  // backend image key if the image was server-saved
  params:         object   // snapshot: { prompt, seed, size, steps, cfg, ... }
  addedAt:        number   // Date.now() at time of assignment
}

index: "galleryId" (non-unique) — enables O(log n) per-gallery queries
```

### Backend sync shape (v2, not implemented in v1)

```text
POST /api/galleries
{
  "galleries": [{ id, name, createdAt }],
  "items": [{ galleryId, cacheKey, serverImageKey, params, addedAt }]
}
```

Keyed on `serverImageKey` for deduplication on the server.

---

## State — `useGalleries` hook

**File:** `lcm-sr-ui/src/hooks/useGalleries.js`

Owned by `App.jsx` and passed to child components as `galleryState`.

```js
useGalleries() → {
  galleries,              // [{ id, name, createdAt }]  — from localStorage
  activeGalleryId,        // string | null  (null = "none" / no active gallery)
  setActiveGalleryId,     // (id: string | null) => void
  createGallery,          // (name: string) => void  — truncates to 16 chars, auto-selects
  addToGallery,           // (cacheKey, { serverImageKey, params }) => Promise<void>
  getGalleryImages,       // (galleryId) => Promise<GalleryItem[]>
}
```

- `galleries` and `activeGalleryId` are React state initialised from `localStorage` on mount.
- `createGallery` writes to `localStorage` synchronously and calls `setActiveGalleryId` with the new gallery's id.
- `addToGallery` and `getGalleryImages` operate on `gallery_items` in the `lcm-image-cache` IndexedDB.
- **The hook must not open its own DB connection.** IndexedDB serialises version upgrades per origin — a second connection at v2 would block the v3 upgrade. Instead, `openDatabase()` in `cache.js` is the single opener for the DB. It is updated to v3 (adding `gallery_items`) and exported so `useGalleries` calls it directly. This guarantees one connection, one upgrade path.

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
- Images fetched from cache blobs via their `cacheKey`
- Click on image → renders `GalleryImageViewer` inside the lightbox
- Keyboard: Space on a focused/hovered thumbnail → `window.open(imageUrl, '_blank')`
- Thumbnails use `object-fit: cover` in a fixed square cell

#### `GalleryImageViewer.jsx`

Single image view within the lightbox, opened by clicking a grid thumbnail.

- Full image centered in the available area
- **Metadata bar:** absolutely positioned at the bottom, hidden by default. Becomes visible when the pointer enters the lower 20% of the image container — detected by comparing `e.clientY` against the container's `getBoundingClientRect`. Fields: prompt, seed, size, steps, cfg, backend, addedAt.
- Back arrow (top-left within viewer) returns to the grid
- Spacebar → `window.open(imageUrl, '_blank')`

---

### Changes to existing files

#### `App.jsx`

1. Add `useGalleries()` and spread `galleryState` into props.
2. In the tab bar, after the "Configuration" tab trigger, add:
   - A `<GalleryCreatePopover>` trigger: folder icon (`FolderPlus` from lucide) + `[+]` label.
   - For each gallery in `galleries`, render a clickable folder tab (`Folder` icon + truncated name). Clicking calls `setOpenGalleryId(gallery.id)`.
3. Render `<GalleryLightbox>` conditionally when `openGalleryId` is set.
4. Pass `galleryState` down to `<OptionsPanel>` and relevant props to `<ChatContainer>` → `<MessageBubble>`.

#### `OptionsPanel.jsx`

Add a `GallerySelector` block **below the draft-prompt textarea**, above whatever currently follows it.

- Uses the existing `<Select>` / `<SelectTrigger>` / `<SelectItem>` components (same style as Negative Prompt Template selector)
- First item: value `null`, label `None`
- Remaining items: one per gallery, label truncated to 16 chars
- `onValueChange` calls `setActiveGalleryId`
- Receives `galleryState` via props

#### `MessageBubble.jsx`

In the metadata pills row (line ~312), add a `→ Gallery` pill **next to the Download link**:

```jsx
{onAddToGallery && (
  <button
    className={`... ${activeGalleryId ? '' : 'opacity-40 cursor-not-allowed'}`}
    disabled={!activeGalleryId}
    onClick={(e) => {
      e.stopPropagation();
      onAddToGallery(msg.meta?.cacheKey, {
        serverImageKey: msg.serverImageKey ?? null,
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
| Image blob missing from cache (evicted) | Thumbnail shows a broken-image placeholder; spacebar/click still works via `serverImageKey` URL if present |
| Two galleries created with the same name | Allowed — they get distinct IDs |
| ESC pressed while a child window is open | Lightbox closes and calls `.close()` on all tracked child window refs |

---

## Files changed / created

| File | Change |
|---|---|
| `src/hooks/useGalleries.js` | **new** |
| `src/components/gallery/GalleryCreatePopover.jsx` | **new** |
| `src/components/gallery/GalleryLightbox.jsx` | **new** |
| `src/components/gallery/GalleryGrid.jsx` | **new** |
| `src/components/gallery/GalleryImageViewer.jsx` | **new** |
| `src/utils/cache.js` | modify — add `gallery_items` store in `onupgradeneeded`, bump DB to v3, export `openDatabase` |
| `src/App.jsx` | modify — `useGalleries`, tab bar buttons, lightbox render, prop threading |
| `src/components/options/OptionsPanel.jsx` | modify — add `GallerySelector` |
| `src/components/chat/MessageBubble.jsx` | modify — add `→ Gallery` pill |
