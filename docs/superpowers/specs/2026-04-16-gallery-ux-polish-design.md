# Gallery UX Polish Design

## Summary

Gallery is functional but not usable. This design adds the interaction layer: thumbnail selection with a universal floating action bar, a trash gallery with two-step delete (soft move then hard purge), a double-click zoom overlay, hover magnify, a discoverable open-in-new-tab control, arrow-key navigation, and a configurable keymap system backing every keyboard affordance.

Everything stays client-side because the gallery store is IndexedDB-local. The only server-side surface is a static keymap defaults file shipped under `conf/`.

## Goals

- Single floating action bar as the only action surface for selected images, whether one or many are selected
- Soft-delete via a dedicated trash gallery; hard-delete only available from trash
- Zoom overlay that is non-obscuring (50vw × 50vh max, AR-preserved) and ephemeral
- Keyboard parity with mouse: every action reachable via a key, defaults in `conf/keymap.yml`, user overrides in `localStorage`
- Arrow-key traversal inside the gallery grid and viewer
- Thumbnail hover magnify that does not reflow the grid

## Non-Goals

- Annotate action (deferred, separate issue)
- Drag-to-select, rubber-band selection, drag-to-reorder
- Key-hint overlays (HUD / legend) - tracked for a later pass
- Trash auto-purge policy
- Server-side gallery store
- Tiered permissions on delete
- Binding conflict detection UI

## Current State

- [`lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx) owns toolbar, grid, and viewer composition; opacity slider is the only toolbar control beyond Close.
- [`lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/gallery/GalleryGrid.jsx) has thumbnail rendering with Space-to-open-tab on a focused cell but no click-select, no hover state, and no action surface.
- [`lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx) has Space-to-open-tab and Back, nothing else.
- [`lcm-sr-ui/src/hooks/useGalleries.js`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/hooks/useGalleries.js) stores items in IndexedDB keyed by id with indices `galleryId` and `cacheKey`; exposes `removeFromGallery(galleryId, cacheKey)` but not a per-item id delete.
- No keymap abstraction exists. Hot keys are inlined in components.

## Architecture

Five coordinated layers. Each layer ships as its own FP sub-issue; the boundaries below are the contract between them.

1. **Keymap**: `useKeymap()` hook; reads defaults from `conf/keymap.yml` (bundled to the frontend), merges user overrides from `localStorage`, returns `matches(action, event)` and `bindingOf(action)`.
2. **Trash data layer**: new fields on gallery items, new APIs on `useGalleries` (`moveToTrash`, `restoreFromTrash`, `hardDelete`, `getTrashItems`, `removeGalleryItem`). A sentinel `TRASH_GALLERY_ID` constant.
3. **Selection**: `useSelection(items)` hook at the `GalleryLightbox` level, holds a Set of ids and anchor id, exposes toggle / range / clear / all.
4. **UI surfaces**: hover magnify, click-select, double-click zoom overlay, floating action bar, open-in-tab button, trash tab.
5. **Keyboard nav**: thin handlers in grid and viewer that map key events through the keymap to selection + trash + navigation APIs.

Layer 1 and 2 have no React UI; they are hooks and data. Layers 3-5 plug them into existing components.

## Keymap System

### Config File

`conf/keymap.yml` ships alongside `modes.yml`:

```yaml
keymap:
  next:          { code: ArrowRight, label: "Next" }
  prev:          { code: ArrowLeft,  label: "Previous" }
  up:            { code: ArrowUp,    label: "Up" }
  down:          { code: ArrowDown,  label: "Down" }
  delete:        { code: Backspace,  label: "Delete" }
  delete_alt:    { code: Delete,     label: "Delete" }
  select_all:    { code: KeyA, mod: mod, label: "Select all" }
  deselect_all:  { code: Escape,    label: "Deselect" }
  zoom:          { code: Enter,     label: "Zoom overlay" }
  open_new_tab:  { code: Space,     label: "Open in new tab" }
  close:         { code: Escape,    label: "Close" }
```

`code` uses `KeyboardEvent.code` values. `mod: mod` means cmd on mac, ctrl on others. Multiple actions may share a binding; the handler chooses by context.

### Loading

The frontend fetches defaults from `GET /api/keymap/defaults` served by a small static handler in `server/` that reads `conf/keymap.yml`. Result is merged at hook init with `localStorage["lcm-keymap-overrides"]`. Override shape is identical to defaults. `useKeymap().setBinding(action, code, mod?)` writes to localStorage.

### Matcher

```js
useKeymap().matches("delete", event)
```

compares `event.code` and modifier bit; returns boolean. Components call this in `keydown` handlers rather than hardcoding key names.

## Trash Data Layer

### Row Schema

Existing columns stay; add:

- `sourceGalleryId?: string` - set when an item is moved to trash; preserves origin for restore
- `trashedAt?: number` - epoch millis; set when moved to trash, cleared on restore

Untrashed items have neither field. No IndexedDB schema version bump required because object store is flexible; the index set stays the same.

### Sentinel

`TRASH_GALLERY_ID = "__trash__"` exported from `useGalleries`. Not a valid gallery id format (prefix `gal_` is required elsewhere), so no collision risk.

### New APIs On `useGalleries`

- `moveToTrash(itemIds: string[]): Promise<void>`
  For each id, load row, set `sourceGalleryId = row.galleryId`, `galleryId = TRASH_GALLERY_ID`, `trashedAt = Date.now()`. Bump revision on both the source gallery and trash.
- `restoreFromTrash(itemIds: string[]): Promise<void>`
  For each id, load row. If `sourceGalleryId` still points to a live gallery, move back; if origin is gone, move into a fallback (current active gallery, or first available). Clear trash fields.
- `hardDelete(itemIds: string[]): Promise<void>`
  Permanent removal. Only exposed to the trash UI path.
- `getTrashItems(): Promise<Item[]>`
  All rows with `galleryId === TRASH_GALLERY_ID`, sorted `trashedAt` descending.
- `removeGalleryItem(itemId: string): Promise<void>`
  Per-item delete that does not go through cacheKey semantics; replaces current `removeFromGallery(galleryId, cacheKey)` usage where callers have an id. `removeFromGallery` stays for now for any consumer that only has a cacheKey.

`getGalleryImages(galleryId)` must continue to return only non-trashed items from a given gallery. Since trash lives under a different `galleryId`, the existing index query already satisfies this.

## Selection

### Hook

`useSelection(items)` at `GalleryLightbox` level. Returns:

- `selectedIds: Set<string>`
- `toggle(id)` - flip membership; updates anchor
- `rangeTo(id)` - select contiguous range from anchor to id (using current grid order)
- `selectAll()` - all visible ids
- `clear()`
- `anchorId: string | null`

Selection resets on gallery change. Range uses the current sort order as presented (matches what the user sees).

### Visual

Selected thumbnails get a 2px primary ring and a small check badge in the corner. Anchor thumbnail gets a brighter ring so shift-click direction is obvious.

### Interaction Rules

- plain click on thumbnail: `toggle(id)`
- shift+click: `rangeTo(id)`
- cmd/ctrl+click: `toggle(id)` (platform idiom; same behavior as plain click for simplicity)
- double-click: opens zoom overlay; does not modify selection
- click on empty grid area: does nothing (explicit clear lives in the action bar and keymap)

### Floating Action Bar

Renders over the grid when `selectedIds.size >= 1`. Fixed position at bottom-center of the lightbox content area, z-indexed above thumbnails.

Structure:

- selected-count label
- `...` menu button that opens a small popover
  - in normal gallery context: `Delete` (calls `moveToTrash(selectedIds)` then `clear()`)
  - in trash context: `Restore`, `Delete permanently`
- deselect-all X button (calls `clear()`)

The bar is the universal action surface for one or many; there is no per-thumbnail action affordance.

## Hover Magnify

Thumbnail hover applies a CSS transform: `scale(1.08)` with a short `transition-transform`. Elevation shadow bumps one level. `transform-origin: center`. Grid container has `overflow: visible` so the scaled tile does not clip; neighboring tiles render over each other briefly by z-index which is acceptable.

Respects `prefers-reduced-motion: reduce` by disabling the transition (snap, no animation) and keeping scale at 1.0 (no magnify when motion-reduced).

## Zoom Overlay

Component `GalleryZoomOverlay`. Opens on double-click of a thumbnail. Props: `item`, `resolveImageUrl`, `onClose`.

### Sizing

Max width = `50vw`, max height = `50vh`. Image scales to fit those bounds while preserving aspect ratio via `object-fit: contain`. If the image's natural dimensions are smaller than both bounds, display at natural size.

### Positioning

Fixed, centered over the viewport for this pass. Position refinement (drag, corner snap, avoid main window center) is deferred.

### Lifecycle

- click outside the image or click the close X: closes
- `close` keymap action closes
- double-click somewhere else: closes current, opens on the new thumbnail
- opening the overlay does not mutate selection
- blob URLs reuse existing `resolveImageUrl` cache; no new URL allocation

## Open-In-New-Tab Button

Surface the existing Space-key behavior as a visible button. Renders in:

- `GalleryImageViewer` toolbar (top-right area alongside Back)
- `GalleryZoomOverlay` toolbar (top-right of the overlay frame)

Both call `window.open(resolvedUrl, '_blank')` and report through the existing `onWindowOpen` tracking so lightbox close still cleans up child windows.

The Space key binding remains, but both are wired through `useKeymap()` for the `open_new_tab` action so overrides work for both mouse and keyboard.

## Trash Surface

A fixed Trash tab renders in the gallery chrome (outside the normal gallery list). Location: the gallery selector area in `OptionsPanel`; render it visually separated (divider + muted styling). Clicking opens `GalleryLightbox` with `galleryId = TRASH_GALLERY_ID`.

Inside a trash-context lightbox:

- grid uses `getTrashItems()` instead of `getGalleryImages(galleryId)`
- action bar menu shows `Restore` and `Delete permanently` instead of `Delete`
- empty state copy: `Trash is empty.`

Trash tab label can optionally show a small count badge when trash is non-empty; this is styling and not a new interaction contract.

## Keyboard Navigation

All bindings route through `useKeymap().matches(action, event)`.

### Grid

- focus follows the last clicked or arrow-navigated cell
- arrow actions (`up`, `down`, `left`, `right`) move focus between cells using grid row/col math
- `select_all`: all visible items selected
- `deselect_all`: clears selection and leaves focus where it was
- `delete` / `delete_alt` on any cell: if selection is non-empty, `moveToTrash(selectedIds)`; else `moveToTrash([focusedId])`; in trash context, same pattern calls `hardDelete`
- `zoom`: opens zoom overlay on focused cell
- `open_new_tab`: opens focused cell in a new browser tab

### Viewer

- `next` / `prev`: cycle the viewer to the next/previous item in the current gallery's sort order; wrap at the ends (simpler than stop-at-ends for now)
- `delete` / `delete_alt`: context-aware, mirrors grid semantics
  - normal gallery context: `moveToTrash([viewerItemId])`
  - trash context: `hardDelete([viewerItemId])`
  - either case: advance to next item if one exists, else `onBack()`
- `close`: `onBack()` (viewer-level close) - the same key action closes the lightbox at the outer level if no viewer is open
- `open_new_tab`: same as existing Space behavior, now routed through keymap

### Zoom overlay

- `close`: closes overlay
- `open_new_tab`: opens current zoom item in new tab

## Testing Strategy

Add or update tests:

- `useKeymap`: merges defaults + overrides; `matches` returns true for defaults, false after override, true for override code
- trash data layer: `moveToTrash` preserves `sourceGalleryId`, flips `galleryId`; `restoreFromTrash` reverses; `hardDelete` removes rows; `getTrashItems` filters correctly; `getGalleryImages` never returns trashed rows
- selection hook: toggle, range (anchor-based), clear; range order matches visible order
- `GalleryGrid`: click toggles, shift+click range, cmd+click toggles; double-click opens zoom; hover class appears; reduced-motion disables animation
- `GalleryLightbox`: floating action bar appears when selection >= 1; menu items differ in trash context; deselect-all clears; delete action calls `moveToTrash` and clears
- zoom overlay: renders at correct bounds, closes on outside click / close key, does not mutate selection, calls open_new_tab through button and keymap
- keyboard nav: arrow keys move focus in grid row/col math; next/prev cycle in viewer; backspace triggers moveToTrash on selection or focused cell
- trash tab: renders outside gallery list; opens trash-context lightbox; action bar shows Restore / Delete permanently; empty state copy

Extend:

- [`lcm-sr-ui/src/hooks/useGalleries.test.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/hooks/useGalleries.test.jsx) - trash APIs
- [`lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx) - selection, hover, double-click
- [`lcm-sr-ui/src/components/gallery/GalleryLightbox.test.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/gallery/GalleryLightbox.test.jsx) - action bar, trash context
- [`lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx) - next/prev, delete behavior
- new files for `useKeymap`, `useSelection`, `GalleryZoomOverlay`, trash tab surface

## Validation And Edge Cases

- keymap loading: if `conf/keymap.yml` is missing or malformed, fallback to hardcoded defaults baked into the hook; log to console, do not crash
- localStorage quota: writing overrides is bounded to a few kilobytes; treat quota errors as non-fatal, keep in-memory state
- item id collisions: selection state uses row `id` (already UUID); no risk
- trash restore when origin gone: fallback path routes into the current active gallery or, if none, the first available; if no galleries exist, hard-delete the row instead (cannot leave an orphan)
- mixed selection across galleries: selection scope is per lightbox session and bound to the current gallery view; switching galleries clears selection
- reduced-motion: all animations (hover magnify, overlay fade) respect the media query
- double-click vs click: the single-click toggle fires on the first click; the second click within the browser double-click threshold triggers the zoom overlay without un-toggling (handle via ignoring the next `click` when a `dblclick` is observed)

## Risks And Tradeoffs

**Why a fixed trash tab rather than an entry in the gallery list.** Keeps trash distinct from user galleries, avoids accidental selection as a restore target, and avoids adding schema-level "system gallery" flags.

**Why a floating action bar instead of per-thumbnail `...`.** One action surface is less clutter on a grid full of thumbnails, keeps single and multi selection symmetric, and avoids duplicating action logic across thumbnails.

**Why soft-delete instead of confirm-then-delete.** Confirm dialogs are modal and fight keyboard-driven flows; trash gives reversibility without a dialog. Hard-delete lives only inside trash where the intent is explicit.

**Why keymap defaults come from the server.** Keeps defaults version-controlled and shared across clients; localStorage override keeps personalization client-side. The tradeoff is a startup fetch, which is fine given the lightbox is not the first surface loaded.

**Why no WASD / vim bindings by default.** Arrow-key defaults are the least-surprising choice; users who want modal bindings can override.

**Double-click vs single-click coupling.** The guard around single-click firing before dblclick has to be solid or selection flickers. Plan includes a dedicated test for that case.

## Migration

No schema migration needed; new fields on gallery rows are additive and only consulted via explicit trash-aware code paths. The existing `removeFromGallery(galleryId, cacheKey)` stays to avoid churn; new code paths use `removeGalleryItem(id)` or the trash APIs.

Existing tests that assume `removeFromGallery` is the only delete path will continue to pass. New tests cover the trash-aware paths separately.

## Tracking Plan

Sub-issues live under parent `STABL-osgiqtxv`. Recommended build order:

1. `STABL-gcqvmbpo` Keymap config system
2. `STABL-edzszkjk` Trash gallery data layer
3. `STABL-kiazzizg` Selection + floating action bar
4. `STABL-rxzigckx` Trash gallery UI surface
5. `STABL-jrediyzc` Thumbnail hover magnify
6. `STABL-rnwijffv` Double-click zoom overlay
7. `STABL-lgryhltt` Open-in-new-tab button
8. `STABL-uqwjxzvr` Keyboard navigation wired to keymap

1 and 2 are foundation. 3 unblocks 4. 5-7 are largely independent. 8 depends on 1 and 3.

## Success Criteria

- clicking a thumbnail selects it; the floating action bar appears with a `...` menu that moves the selection to trash
- shift+click selects a range; cmd/ctrl+click toggles
- the trash tab lives outside the gallery list and exposes Restore and Delete permanently on selection
- double-click opens a zoom overlay sized to 50vw/50vh, AR-preserved, and does not obscure the main window
- an open-in-new-tab button is visible in both the viewer and the zoom overlay
- backspace and arrow keys work in both grid and viewer via `useKeymap`; overrides written to localStorage take effect on reload
- no gallery item is ever hard-deleted outside the trash surface
- the existing Space-to-open-tab behavior continues to work via the `open_new_tab` keymap action
