# Gallery UX Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selection, trash-gallery two-step delete, hover magnify, double-click zoom overlay, open-in-new-tab button, configurable keymap, and keyboard navigation to the gallery surface.

**Architecture:** Client-side IndexedDB + React. Five layers: a `useKeymap` hook backed by `conf/keymap.yml` (server-delivered) + `localStorage` overrides; a trash data layer on `useGalleries` with `sourceGalleryId` + `trashedAt` soft-delete fields; a `useSelection` hook driving a floating action bar at the `GalleryLightbox` level; UI components for hover / zoom / open-tab / trash tab; keyboard handlers that route through the keymap.

**Tech Stack:** React 19, Vite, vitest + @testing-library/react, fake-indexeddb, FastAPI (one static endpoint), PyYAML. Everything else is existing UI.

---

## File Structure

### Foundation (server-side keymap delivery + frontend hook)

- Create: `conf/keymap.yml`
  Ships default action -> keycode bindings.
- Create: `server/keymap_routes.py`
  FastAPI router that exposes `GET /api/keymap/defaults` by reading `conf/keymap.yml`.
- Modify: `server/lcm_sr_server.py`
  Wire the new router alongside existing `include_router` calls.
- Create: `lcm-sr-ui/src/hooks/useKeymap.js`
  React hook: fetches defaults, merges localStorage overrides, exposes `matches(action, event)`, `bindingOf(action)`, `setBinding(action, code, mod?)`.
- Create: `lcm-sr-ui/src/hooks/useKeymap.test.jsx`

### Trash data layer

- Modify: `lcm-sr-ui/src/hooks/useGalleries.js`
  Adds `TRASH_GALLERY_ID`, `moveToTrash`, `restoreFromTrash`, `hardDelete`, `getTrashItems`, `removeGalleryItem`.
- Modify: `lcm-sr-ui/src/hooks/useGalleries.test.jsx`

### Selection + floating action bar

- Create: `lcm-sr-ui/src/hooks/useSelection.js`
  Selection state with toggle / rangeTo / clear / selectAll / anchor.
- Create: `lcm-sr-ui/src/hooks/useSelection.test.jsx`
- Create: `lcm-sr-ui/src/components/gallery/FloatingActionBar.jsx`
- Create: `lcm-sr-ui/src/components/gallery/FloatingActionBar.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`
  Add click / shift-click / cmd-click selection handling + visual ring + check badge.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`
  Integrate `useSelection`, render the action bar, pass selection down to grid.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.test.jsx`

### Trash UI surface

- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`
  Accept a `trashMode` prop; fetch via `getTrashItems()` when true; switch action bar menu items.
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.jsx`
  Add a Trash tab button in the gallery selector area.
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`

### Hover magnify

- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`
  Add hover scale class + reduced-motion guard.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`

### Zoom overlay

- Create: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.jsx`
- Create: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`
  Wire double-click to an `onZoom` callback.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`
  Render zoom overlay on top of grid, track `zoomItem` state.

### Open-in-new-tab button

- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx`
  Add visible button in the toolbar (top-right area), wired via `useKeymap` to `open_new_tab`.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.jsx`
  Add the same button inside the overlay.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.test.jsx`

### Keyboard navigation

- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`
  Arrow keys move focus; delete calls `onMoveToTrash(selectedOrFocused)`; select_all / deselect_all hooks into selection.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx`
  Left/right cycle, delete triggers move/hard delete (context-aware), open_new_tab and close wired through keymap.
- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx`

---

## Worktree Setup

- [ ] **Step 0: Create a feature worktree**

This plan is large enough to warrant isolation from the `chat-connection-decoupling` worktree currently unmerged.

Run:

```bash
git worktree add .worktrees/gallery-ux-polish -b gallery-ux-polish
cd .worktrees/gallery-ux-polish
```

All subsequent steps run inside `.worktrees/gallery-ux-polish`. Paths below are repo-relative.

---

### Task 1: Keymap Config System

**Files:**
- Create: `conf/keymap.yml`
- Create: `server/keymap_routes.py`
- Create: `tests/test_keymap_routes.py`
- Modify: `server/lcm_sr_server.py:797-809`
- Create: `lcm-sr-ui/src/hooks/useKeymap.js`
- Create: `lcm-sr-ui/src/hooks/useKeymap.test.jsx`

- [ ] **Step 1: Write the failing server-side test**

```python
# tests/test_keymap_routes.py
from fastapi.testclient import TestClient
from fastapi import FastAPI
from server.keymap_routes import router as keymap_router


def test_get_keymap_defaults_returns_mapping():
    app = FastAPI()
    app.include_router(keymap_router)
    client = TestClient(app)

    resp = client.get("/api/keymap/defaults")
    assert resp.status_code == 200
    body = resp.json()

    assert "keymap" in body
    assert body["keymap"]["delete"]["code"] == "Backspace"
    assert body["keymap"]["next"]["code"] == "ArrowRight"
    assert body["keymap"]["open_new_tab"]["code"] == "Space"
    assert body["keymap"]["zoom"]["code"] == "Enter"


def test_get_keymap_defaults_handles_missing_file(tmp_path, monkeypatch):
    from server import keymap_routes

    monkeypatch.setattr(keymap_routes, "KEYMAP_CONFIG_PATH", str(tmp_path / "missing.yml"))
    app = FastAPI()
    app.include_router(keymap_routes.router)
    client = TestClient(app)

    resp = client.get("/api/keymap/defaults")
    assert resp.status_code == 200
    assert resp.json()["keymap"] == {}
```

- [ ] **Step 2: Run server test to verify it fails**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_keymap_routes.py -q
```

Expected: ImportError / ModuleNotFoundError for `server.keymap_routes`.

- [ ] **Step 3: Implement server-side keymap delivery**

```yaml
# conf/keymap.yml
keymap:
  next:         { code: ArrowRight, label: "Next" }
  prev:         { code: ArrowLeft,  label: "Previous" }
  up:           { code: ArrowUp,    label: "Up" }
  down:         { code: ArrowDown,  label: "Down" }
  delete:       { code: Backspace,  label: "Delete" }
  delete_alt:   { code: Delete,     label: "Delete" }
  select_all:   { code: KeyA, mod: mod, label: "Select all" }
  deselect_all: { code: Escape,     label: "Deselect" }
  close:        { code: Escape,     label: "Close" }
  zoom:         { code: Enter,      label: "Zoom overlay" }
  open_new_tab: { code: Space,      label: "Open in new tab" }
```

```python
# server/keymap_routes.py
import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter

logger = logging.getLogger(__name__)

KEYMAP_CONFIG_PATH = os.environ.get("KEYMAP_CONFIG_PATH", "conf/keymap.yml")

router = APIRouter(prefix="/api", tags=["keymap"])


@router.get("/keymap/defaults")
def get_keymap_defaults() -> Dict[str, Any]:
    path = Path(KEYMAP_CONFIG_PATH)
    if not path.exists():
        logger.warning("[keymap] %s missing; returning empty defaults", path)
        return {"keymap": {}}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        logger.warning("[keymap] failed to parse %s: %s", path, exc)
        return {"keymap": {}}
    keymap = data.get("keymap") or {}
    if not isinstance(keymap, dict):
        logger.warning("[keymap] %s 'keymap' is not a mapping", path)
        return {"keymap": {}}
    return {"keymap": keymap}
```

```python
# server/lcm_sr_server.py  — add alongside existing include_router calls (around line 809)
from server.keymap_routes import router as keymap_router
app.include_router(keymap_router)
```

- [ ] **Step 4: Run server test to verify it passes**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_keymap_routes.py -q
```

Expected: both tests pass.

- [ ] **Step 5: Write the failing `useKeymap` tests**

```jsx
// lcm-sr-ui/src/hooks/useKeymap.test.jsx
// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useKeymap } from './useKeymap';

function mockFetchDefaults(keymap) {
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => ({ keymap }),
  }));
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  delete globalThis.fetch;
});

describe('useKeymap', () => {
  it('fetches defaults and matches keydown events by code', async () => {
    mockFetchDefaults({
      delete: { code: 'Backspace', label: 'Delete' },
      next: { code: 'ArrowRight', label: 'Next' },
    });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    const event = { code: 'Backspace', metaKey: false, ctrlKey: false, shiftKey: false, altKey: false };
    expect(result.current.matches('delete', event)).toBe(true);
    expect(result.current.matches('next', event)).toBe(false);
  });

  it('applies localStorage override over server defaults', async () => {
    mockFetchDefaults({ delete: { code: 'Backspace', label: 'Delete' } });
    localStorage.setItem(
      'lcm-keymap-overrides',
      JSON.stringify({ delete: { code: 'KeyD', label: 'Delete' } }),
    );

    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    expect(result.current.matches('delete', { code: 'KeyD', metaKey: false, ctrlKey: false })).toBe(true);
    expect(result.current.matches('delete', { code: 'Backspace', metaKey: false, ctrlKey: false })).toBe(false);
  });

  it('matches modifier bindings using platform modifier (meta on mac, ctrl elsewhere)', async () => {
    mockFetchDefaults({ select_all: { code: 'KeyA', mod: 'mod', label: 'Select all' } });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    expect(result.current.matches('select_all', { code: 'KeyA', metaKey: true, ctrlKey: false })).toBe(true);
    expect(result.current.matches('select_all', { code: 'KeyA', metaKey: false, ctrlKey: true })).toBe(true);
    expect(result.current.matches('select_all', { code: 'KeyA', metaKey: false, ctrlKey: false })).toBe(false);
  });

  it('setBinding writes to localStorage and updates matcher', async () => {
    mockFetchDefaults({ delete: { code: 'Backspace', label: 'Delete' } });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    await act(async () => {
      result.current.setBinding('delete', 'KeyX');
    });

    const stored = JSON.parse(localStorage.getItem('lcm-keymap-overrides'));
    expect(stored.delete.code).toBe('KeyX');
    expect(result.current.matches('delete', { code: 'KeyX', metaKey: false, ctrlKey: false })).toBe(true);
  });

  it('falls back to hardcoded defaults when fetch fails', async () => {
    globalThis.fetch = vi.fn(async () => { throw new Error('network'); });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    expect(result.current.matches('delete', { code: 'Backspace', metaKey: false, ctrlKey: false })).toBe(true);
    expect(result.current.matches('deselect_all', { code: 'Escape', metaKey: false, ctrlKey: false })).toBe(true);
  });
});
```

- [ ] **Step 6: Run frontend test to verify it fails**

Run:

```bash
cd lcm-sr-ui && npm test -- src/hooks/useKeymap.test.jsx
```

Expected: module-not-found for `./useKeymap`.

- [ ] **Step 7: Implement `useKeymap` hook**

```jsx
// lcm-sr-ui/src/hooks/useKeymap.js
import { useCallback, useEffect, useMemo, useState } from 'react';

const OVERRIDES_KEY = 'lcm-keymap-overrides';

const HARDCODED_FALLBACK = {
  next:         { code: 'ArrowRight', label: 'Next' },
  prev:         { code: 'ArrowLeft',  label: 'Previous' },
  up:           { code: 'ArrowUp',    label: 'Up' },
  down:         { code: 'ArrowDown',  label: 'Down' },
  delete:       { code: 'Backspace',  label: 'Delete' },
  delete_alt:   { code: 'Delete',     label: 'Delete' },
  select_all:   { code: 'KeyA', mod: 'mod', label: 'Select all' },
  deselect_all: { code: 'Escape',     label: 'Deselect' },
  close:        { code: 'Escape',     label: 'Close' },
  zoom:         { code: 'Enter',      label: 'Zoom overlay' },
  open_new_tab: { code: 'Space',      label: 'Open in new tab' },
};

function readOverrides() {
  try {
    const raw = localStorage.getItem(OVERRIDES_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeOverrides(overrides) {
  try {
    localStorage.setItem(OVERRIDES_KEY, JSON.stringify(overrides));
  } catch {
    // quota or storage disabled — keep in-memory state
  }
}

function modPressed(event) {
  return Boolean(event.metaKey || event.ctrlKey);
}

export function useKeymap() {
  const [defaults, setDefaults] = useState(HARDCODED_FALLBACK);
  const [overrides, setOverrides] = useState(() => readOverrides());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    fetch('/api/keymap/defaults')
      .then((res) => (res.ok ? res.json() : { keymap: {} }))
      .then((body) => {
        if (!active) return;
        const serverMap = body && typeof body.keymap === 'object' ? body.keymap : {};
        if (Object.keys(serverMap).length > 0) {
          setDefaults({ ...HARDCODED_FALLBACK, ...serverMap });
        }
      })
      .catch(() => {})
      .finally(() => { if (active) setReady(true); });
    return () => { active = false; };
  }, []);

  const merged = useMemo(() => ({ ...defaults, ...overrides }), [defaults, overrides]);

  const matches = useCallback((action, event) => {
    const binding = merged[action];
    if (!binding) return false;
    if (event.code !== binding.code) return false;
    if (binding.mod === 'mod') return modPressed(event);
    return !modPressed(event);
  }, [merged]);

  const bindingOf = useCallback((action) => merged[action] ?? null, [merged]);

  const setBinding = useCallback((action, code, mod) => {
    const next = { ...overrides, [action]: { code, ...(mod ? { mod } : {}), label: merged[action]?.label ?? action } };
    setOverrides(next);
    writeOverrides(next);
  }, [overrides, merged]);

  return { ready, matches, bindingOf, setBinding };
}
```

- [ ] **Step 8: Run frontend test to verify it passes**

Run:

```bash
cd lcm-sr-ui && npm test -- src/hooks/useKeymap.test.jsx
```

Expected: all five tests pass.

- [ ] **Step 9: Commit**

```bash
git add conf/keymap.yml server/keymap_routes.py server/lcm_sr_server.py tests/test_keymap_routes.py lcm-sr-ui/src/hooks/useKeymap.js lcm-sr-ui/src/hooks/useKeymap.test.jsx
git commit -m "feat: add keymap config system with server defaults and localStorage overrides

Refs STABL-gcqvmbpo"
```

---

### Task 2: Trash Data Layer

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useGalleries.js`
- Modify: `lcm-sr-ui/src/hooks/useGalleries.test.jsx`

- [ ] **Step 1: Write the failing trash-layer tests**

Append to `lcm-sr-ui/src/hooks/useGalleries.test.jsx`:

```jsx
describe('useGalleries — trash layer', () => {
  it('moveToTrash flips galleryId to TRASH_GALLERY_ID and preserves sourceGalleryId', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Alpha'); });
    const galleryId = result.current.activeGalleryId;
    let itemId;
    await act(async () => {
      await result.current.addToGallery('key_1', {
        serverImageUrl: 'x', params: {}, galleryId,
      });
      const items = await result.current.getGalleryImages(galleryId);
      itemId = items[0].id;
    });

    await act(async () => {
      await result.current.moveToTrash([itemId]);
    });

    const sourceItems = await result.current.getGalleryImages(galleryId);
    const trashItems = await result.current.getTrashItems();
    expect(sourceItems).toHaveLength(0);
    expect(trashItems).toHaveLength(1);
    expect(trashItems[0].sourceGalleryId).toBe(galleryId);
    expect(trashItems[0].trashedAt).toBeTypeOf('number');
  });

  it('restoreFromTrash returns items to their original gallery and clears trash fields', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Alpha'); });
    const galleryId = result.current.activeGalleryId;
    let itemId;
    await act(async () => {
      await result.current.addToGallery('key_1', { serverImageUrl: 'x', params: {}, galleryId });
      const items = await result.current.getGalleryImages(galleryId);
      itemId = items[0].id;
      await result.current.moveToTrash([itemId]);
    });

    await act(async () => {
      await result.current.restoreFromTrash([itemId]);
    });

    const restored = await result.current.getGalleryImages(galleryId);
    const trashItems = await result.current.getTrashItems();
    expect(restored).toHaveLength(1);
    expect(restored[0].sourceGalleryId).toBeUndefined();
    expect(restored[0].trashedAt).toBeUndefined();
    expect(trashItems).toHaveLength(0);
  });

  it('restoreFromTrash routes orphaned items into the active gallery if origin is gone', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Alpha'); });
    const origin = result.current.activeGalleryId;
    let itemId;
    await act(async () => {
      await result.current.addToGallery('key_1', { serverImageUrl: 'x', params: {}, galleryId: origin });
      const items = await result.current.getGalleryImages(origin);
      itemId = items[0].id;
      await result.current.moveToTrash([itemId]);
      result.current.createGallery('Beta');
    });
    const beta = result.current.activeGalleryId;

    // simulate origin removal by directly editing localStorage gallery list
    const filtered = JSON.parse(localStorage.getItem('lcm-galleries')).filter(g => g.id !== origin);
    localStorage.setItem('lcm-galleries', JSON.stringify(filtered));

    await act(async () => { await result.current.restoreFromTrash([itemId]); });

    const betaItems = await result.current.getGalleryImages(beta);
    expect(betaItems).toHaveLength(1);
    expect(betaItems[0].id).toBe(itemId);
  });

  it('hardDelete removes rows permanently', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Alpha'); });
    const galleryId = result.current.activeGalleryId;
    let itemId;
    await act(async () => {
      await result.current.addToGallery('key_1', { serverImageUrl: 'x', params: {}, galleryId });
      const items = await result.current.getGalleryImages(galleryId);
      itemId = items[0].id;
      await result.current.moveToTrash([itemId]);
      await result.current.hardDelete([itemId]);
    });
    expect(await result.current.getTrashItems()).toHaveLength(0);
  });

  it('getGalleryImages never returns trashed rows', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Alpha'); });
    const galleryId = result.current.activeGalleryId;
    await act(async () => {
      await result.current.addToGallery('a', { serverImageUrl: 'x', params: {}, galleryId });
      await result.current.addToGallery('b', { serverImageUrl: 'y', params: {}, galleryId });
      const items = await result.current.getGalleryImages(galleryId);
      await result.current.moveToTrash([items[0].id]);
    });
    const remaining = await result.current.getGalleryImages(galleryId);
    expect(remaining).toHaveLength(1);
  });

  it('removeGalleryItem deletes a single row by id', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Alpha'); });
    const galleryId = result.current.activeGalleryId;
    let itemId;
    await act(async () => {
      await result.current.addToGallery('key_1', { serverImageUrl: 'x', params: {}, galleryId });
      const items = await result.current.getGalleryImages(galleryId);
      itemId = items[0].id;
      await result.current.removeGalleryItem(itemId);
    });
    expect(await result.current.getGalleryImages(galleryId)).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd lcm-sr-ui && npm test -- src/hooks/useGalleries.test.jsx
```

Expected: fails because `moveToTrash`, `restoreFromTrash`, `hardDelete`, `getTrashItems`, `removeGalleryItem` do not exist.

- [ ] **Step 3: Implement the trash APIs**

Add exports and helpers at the top of [`lcm-sr-ui/src/hooks/useGalleries.js`](lcm-sr-ui/src/hooks/useGalleries.js):

```jsx
export const TRASH_GALLERY_ID = '__trash__';
```

Add new callbacks inside the `useGalleries()` body, immediately after `removeFromGallery`:

```jsx
const moveToTrash = useCallback(async (itemIds) => {
  if (!Array.isArray(itemIds) || itemIds.length === 0) return;
  try {
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const affectedGalleries = new Set();
    await Promise.all(itemIds.map(async (id) => {
      const row = await promisifyRequest(store.get(id));
      if (!row || row.galleryId === TRASH_GALLERY_ID) return;
      affectedGalleries.add(row.galleryId);
      const next = {
        ...row,
        sourceGalleryId: row.galleryId,
        trashedAt: Date.now(),
        galleryId: TRASH_GALLERY_ID,
      };
      await promisifyRequest(store.put(next));
    }));
    for (const galleryId of affectedGalleries) bumpGalleryRevision(galleryId);
    bumpGalleryRevision(TRASH_GALLERY_ID);
  } catch (err) {
    console.warn('[useGalleries] moveToTrash failed:', err);
  }
}, [getDb, bumpGalleryRevision]);

const restoreFromTrash = useCallback(async (itemIds) => {
  if (!Array.isArray(itemIds) || itemIds.length === 0) return;
  try {
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const galleries = JSON.parse(localStorage.getItem(LS_GALLERIES_KEY) || '[]');
    const knownIds = new Set(galleries.map((g) => g.id));
    const activeId = localStorage.getItem(LS_ACTIVE_KEY);
    const fallback = (activeId && knownIds.has(activeId)) ? activeId : galleries[0]?.id ?? null;
    const affectedGalleries = new Set();
    await Promise.all(itemIds.map(async (id) => {
      const row = await promisifyRequest(store.get(id));
      if (!row || row.galleryId !== TRASH_GALLERY_ID) return;
      const target = (row.sourceGalleryId && knownIds.has(row.sourceGalleryId)) ? row.sourceGalleryId : fallback;
      if (!target) {
        await promisifyRequest(store.delete(id));
        return;
      }
      const next = { ...row, galleryId: target };
      delete next.sourceGalleryId;
      delete next.trashedAt;
      await promisifyRequest(store.put(next));
      affectedGalleries.add(target);
    }));
    for (const galleryId of affectedGalleries) bumpGalleryRevision(galleryId);
    bumpGalleryRevision(TRASH_GALLERY_ID);
  } catch (err) {
    console.warn('[useGalleries] restoreFromTrash failed:', err);
  }
}, [getDb, bumpGalleryRevision]);

const hardDelete = useCallback(async (itemIds) => {
  if (!Array.isArray(itemIds) || itemIds.length === 0) return;
  try {
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    await Promise.all(itemIds.map((id) => promisifyRequest(store.delete(id))));
    bumpGalleryRevision(TRASH_GALLERY_ID);
  } catch (err) {
    console.warn('[useGalleries] hardDelete failed:', err);
  }
}, [getDb, bumpGalleryRevision]);

const getTrashItems = useCallback(async () => {
  try {
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const rows = await promisifyRequest(tx.objectStore(STORE_NAME).index('galleryId').getAll(TRASH_GALLERY_ID));
    return rows.slice().sort((a, b) => (b.trashedAt ?? 0) - (a.trashedAt ?? 0));
  } catch (err) {
    console.warn('[useGalleries] getTrashItems failed:', err);
    return [];
  }
}, [getDb]);

const removeGalleryItem = useCallback(async (itemId) => {
  if (!itemId) return;
  try {
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const row = await promisifyRequest(store.get(itemId));
    if (!row) return;
    await promisifyRequest(store.delete(itemId));
    bumpGalleryRevision(row.galleryId);
  } catch (err) {
    console.warn('[useGalleries] removeGalleryItem failed:', err);
  }
}, [getDb, bumpGalleryRevision]);
```

Extend the returned object:

```jsx
return {
  galleries,
  activeGalleryId,
  setActiveGalleryId,
  createGallery,
  addToGallery,
  removeFromGallery,
  getGalleryImages,
  getGalleryRevision: (galleryId) => galleryRevisions[galleryId] || 0,
  moveToTrash,
  restoreFromTrash,
  hardDelete,
  getTrashItems,
  removeGalleryItem,
  TRASH_GALLERY_ID,
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd lcm-sr-ui && npm test -- src/hooks/useGalleries.test.jsx
```

Expected: all existing + six new trash tests pass.

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useGalleries.js lcm-sr-ui/src/hooks/useGalleries.test.jsx
git commit -m "feat: add trash data layer with soft-delete and restore

Refs STABL-edzszkjk"
```

---

### Task 3: Selection Hook + Floating Action Bar

**Files:**
- Create: `lcm-sr-ui/src/hooks/useSelection.js`
- Create: `lcm-sr-ui/src/hooks/useSelection.test.jsx`
- Create: `lcm-sr-ui/src/components/gallery/FloatingActionBar.jsx`
- Create: `lcm-sr-ui/src/components/gallery/FloatingActionBar.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.test.jsx`

- [ ] **Step 1: Write the failing selection-hook tests**

```jsx
// lcm-sr-ui/src/hooks/useSelection.test.jsx
// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { useSelection } from './useSelection';

function items(n) {
  return Array.from({ length: n }, (_, i) => ({ id: `id_${i}` }));
}

describe('useSelection', () => {
  it('starts empty with null anchor', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    expect(result.current.selectedIds.size).toBe(0);
    expect(result.current.anchorId).toBeNull();
  });

  it('toggle adds then removes and updates anchor on add', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    act(() => { result.current.toggle('id_1'); });
    expect(result.current.selectedIds.has('id_1')).toBe(true);
    expect(result.current.anchorId).toBe('id_1');
    act(() => { result.current.toggle('id_1'); });
    expect(result.current.selectedIds.has('id_1')).toBe(false);
  });

  it('rangeTo selects contiguous ids from anchor to target in item order', () => {
    const { result } = renderHook(() => useSelection(items(5)));
    act(() => { result.current.toggle('id_1'); });
    act(() => { result.current.rangeTo('id_3'); });
    expect([...result.current.selectedIds].sort()).toEqual(['id_1', 'id_2', 'id_3']);
  });

  it('rangeTo works in reverse direction', () => {
    const { result } = renderHook(() => useSelection(items(5)));
    act(() => { result.current.toggle('id_3'); });
    act(() => { result.current.rangeTo('id_1'); });
    expect([...result.current.selectedIds].sort()).toEqual(['id_1', 'id_2', 'id_3']);
  });

  it('rangeTo with no anchor falls back to single-select', () => {
    const { result } = renderHook(() => useSelection(items(5)));
    act(() => { result.current.rangeTo('id_2'); });
    expect([...result.current.selectedIds]).toEqual(['id_2']);
  });

  it('selectAll selects every visible item', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    act(() => { result.current.selectAll(); });
    expect(result.current.selectedIds.size).toBe(3);
  });

  it('clear resets selection and anchor', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    act(() => { result.current.toggle('id_1'); });
    act(() => { result.current.clear(); });
    expect(result.current.selectedIds.size).toBe(0);
    expect(result.current.anchorId).toBeNull();
  });

  it('resets when items array identity changes to a different gallery', () => {
    const first = items(3);
    const { result, rerender } = renderHook(({ list }) => useSelection(list), {
      initialProps: { list: first },
    });
    act(() => { result.current.toggle('id_1'); });
    expect(result.current.selectedIds.size).toBe(1);
    rerender({ list: [{ id: 'other_1' }] });
    expect(result.current.selectedIds.size).toBe(0);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run:

```bash
cd lcm-sr-ui && npm test -- src/hooks/useSelection.test.jsx
```

Expected: module-not-found.

- [ ] **Step 3: Implement `useSelection`**

```jsx
// lcm-sr-ui/src/hooks/useSelection.js
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export function useSelection(items) {
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [anchorId, setAnchorId] = useState(null);
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const itemsKey = useMemo(
    () => items.map((it) => it.id).join('|'),
    [items],
  );

  useEffect(() => {
    setSelectedIds(new Set());
    setAnchorId(null);
  }, [itemsKey]);

  const toggle = useCallback((id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setAnchorId((prev) => (prev === id ? prev : id));
  }, []);

  const rangeTo = useCallback((id) => {
    const list = itemsRef.current;
    setAnchorId((prevAnchor) => {
      if (!prevAnchor) {
        setSelectedIds(new Set([id]));
        return id;
      }
      const a = list.findIndex((it) => it.id === prevAnchor);
      const b = list.findIndex((it) => it.id === id);
      if (a === -1 || b === -1) {
        setSelectedIds(new Set([id]));
        return id;
      }
      const [lo, hi] = a < b ? [a, b] : [b, a];
      const next = new Set();
      for (let i = lo; i <= hi; i++) next.add(list[i].id);
      setSelectedIds(next);
      return prevAnchor;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(itemsRef.current.map((it) => it.id)));
  }, []);

  const clear = useCallback(() => {
    setSelectedIds(new Set());
    setAnchorId(null);
  }, []);

  return { selectedIds, anchorId, toggle, rangeTo, selectAll, clear };
}
```

- [ ] **Step 4: Run to verify selection-hook tests pass**

Run:

```bash
cd lcm-sr-ui && npm test -- src/hooks/useSelection.test.jsx
```

Expected: all eight pass.

- [ ] **Step 5: Write failing `FloatingActionBar` tests**

```jsx
// lcm-sr-ui/src/components/gallery/FloatingActionBar.test.jsx
// @vitest-environment jsdom
import { render, screen, fireEvent } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { FloatingActionBar } from './FloatingActionBar';

describe('FloatingActionBar', () => {
  it('does not render when selection is empty', () => {
    const { container } = render(
      <FloatingActionBar selectedCount={0} trashMode={false} onDelete={vi.fn()} onClear={vi.fn()} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders count and delete button in normal context', () => {
    const onDelete = vi.fn();
    render(
      <FloatingActionBar selectedCount={2} trashMode={false} onDelete={onDelete} onClear={vi.fn()} />,
    );
    expect(screen.getByText('2 selected')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /delete/i }));
    expect(onDelete).toHaveBeenCalled();
  });

  it('renders Restore + Delete permanently in trash context', () => {
    const onRestore = vi.fn();
    const onHardDelete = vi.fn();
    render(
      <FloatingActionBar
        selectedCount={1}
        trashMode
        onRestore={onRestore}
        onHardDelete={onHardDelete}
        onClear={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /restore/i }));
    expect(onRestore).toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /delete permanently/i }));
    expect(onHardDelete).toHaveBeenCalled();
  });

  it('clear button calls onClear', () => {
    const onClear = vi.fn();
    render(
      <FloatingActionBar selectedCount={1} trashMode={false} onDelete={vi.fn()} onClear={onClear} />,
    );
    fireEvent.click(screen.getByRole('button', { name: /clear selection/i }));
    expect(onClear).toHaveBeenCalled();
  });
});
```

- [ ] **Step 6: Run to verify it fails**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/FloatingActionBar.test.jsx
```

Expected: module-not-found.

- [ ] **Step 7: Implement `FloatingActionBar`**

```jsx
// lcm-sr-ui/src/components/gallery/FloatingActionBar.jsx
import React, { useState, useRef, useEffect } from 'react';
import { MoreHorizontal, X } from 'lucide-react';

export function FloatingActionBar({
  selectedCount,
  trashMode,
  onDelete,
  onRestore,
  onHardDelete,
  onClear,
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    function onDoc(e) {
      if (!ref.current) return;
      if (!ref.current.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  if (selectedCount < 1) return null;

  const fire = (handler) => () => {
    setMenuOpen(false);
    handler?.();
  };

  return (
    <div
      ref={ref}
      role="toolbar"
      aria-label="Gallery selection actions"
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 rounded-full bg-background/95 border shadow-lg px-3 py-1.5"
    >
      <span className="text-sm font-medium">
        {selectedCount} selected
      </span>
      <div className="relative">
        <button
          type="button"
          aria-label="Open action menu"
          className="p-1.5 rounded hover:bg-muted"
          onClick={() => setMenuOpen((v) => !v)}
        >
          <MoreHorizontal className="h-4 w-4" />
        </button>
        {menuOpen && (
          <div
            role="menu"
            className="absolute bottom-full mb-2 left-0 min-w-[180px] rounded-md border bg-background shadow-md py-1 text-sm"
          >
            {trashMode ? (
              <>
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full text-left px-3 py-1.5 hover:bg-muted"
                  onClick={fire(onRestore)}
                >
                  Restore
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full text-left px-3 py-1.5 hover:bg-muted text-destructive"
                  onClick={fire(onHardDelete)}
                >
                  Delete permanently
                </button>
              </>
            ) : (
              <button
                type="button"
                role="menuitem"
                className="block w-full text-left px-3 py-1.5 hover:bg-muted"
                onClick={fire(onDelete)}
              >
                Delete
              </button>
            )}
          </div>
        )}
      </div>
      <button
        type="button"
        aria-label="Clear selection"
        className="p-1.5 rounded hover:bg-muted"
        onClick={onClear}
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
```

- [ ] **Step 8: Run to verify the bar tests pass**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/FloatingActionBar.test.jsx
```

Expected: four tests pass.

- [ ] **Step 9: Write failing `GalleryGrid` selection tests**

Append to [`lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`](lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx):

```jsx
describe('GalleryGrid — selection', () => {
  it('click on thumbnail calls onToggle with item id', async () => {
    const items = Array.from({ length: 3 }, (_, i) => makeItem(i));
    const onToggle = vi.fn();
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          resolveImageUrl={resolve}
          onOpenViewer={vi.fn()}
          onToggle={onToggle}
          onRange={vi.fn()}
          onZoom={vi.fn()}
          selectedIds={new Set()}
          anchorId={null}
        />,
      );
    });
    fireEvent.click(screen.getAllByRole('img')[1]);
    expect(onToggle).toHaveBeenCalledWith('id_1', { shift: false, mod: false });
  });

  it('shift+click calls onRange', async () => {
    const items = Array.from({ length: 3 }, (_, i) => makeItem(i));
    const onRange = vi.fn();
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          resolveImageUrl={resolve}
          onOpenViewer={vi.fn()}
          onToggle={vi.fn()}
          onRange={onRange}
          onZoom={vi.fn()}
          selectedIds={new Set()}
          anchorId={null}
        />,
      );
    });
    fireEvent.click(screen.getAllByRole('img')[2], { shiftKey: true });
    expect(onRange).toHaveBeenCalledWith('id_2');
  });

  it('double-click calls onZoom', async () => {
    const items = [makeItem(0)];
    const onZoom = vi.fn();
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          resolveImageUrl={resolve}
          onOpenViewer={vi.fn()}
          onToggle={vi.fn()}
          onRange={vi.fn()}
          onZoom={onZoom}
          selectedIds={new Set()}
          anchorId={null}
        />,
      );
    });
    fireEvent.doubleClick(screen.getAllByRole('img')[0]);
    expect(onZoom).toHaveBeenCalledWith(items[0]);
  });

  it('selected items get an aria-selected=true attribute', async () => {
    const items = Array.from({ length: 2 }, (_, i) => makeItem(i));
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          resolveImageUrl={resolve}
          onOpenViewer={vi.fn()}
          onToggle={vi.fn()}
          onRange={vi.fn()}
          onZoom={vi.fn()}
          selectedIds={new Set(['id_1'])}
          anchorId={'id_1'}
        />,
      );
    });
    const cells = screen.getAllByRole('gridcell');
    expect(cells[1].getAttribute('aria-selected')).toBe('true');
    expect(cells[0].getAttribute('aria-selected')).toBe('false');
  });
});
```

- [ ] **Step 10: Update `GalleryGrid.jsx` to support selection props**

Replace the body of [`lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`](lcm-sr-ui/src/components/gallery/GalleryGrid.jsx) with:

```jsx
// src/components/gallery/GalleryGrid.jsx
import React, { useState, useEffect, useRef } from 'react';

const PAGE_SIZE = 20;

const PLACEHOLDER_SRC =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

function GalleryThumbnail({
  item,
  resolveImageUrl,
  onOpenViewer,
  onToggle,
  onRange,
  onZoom,
  selected,
  isAnchor,
}) {
  const [url, setUrl] = useState(null);
  const urlRef = useRef(null);
  const clickTimerRef = useRef(null);

  useEffect(() => {
    let active = true;
    resolveImageUrl(item).then((resolved) => {
      if (active) {
        urlRef.current = resolved;
        setUrl(resolved);
      }
    });
    return () => { active = false; };
  }, [item.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleClick(e) {
    // Defer single-click so a following double-click can cancel it
    if (clickTimerRef.current) clearTimeout(clickTimerRef.current);
    const shift = e.shiftKey;
    const mod = e.metaKey || e.ctrlKey;
    clickTimerRef.current = setTimeout(() => {
      if (shift) onRange?.(item.id);
      else onToggle?.(item.id, { shift, mod });
    }, 180);
  }

  function handleDoubleClick() {
    if (clickTimerRef.current) clearTimeout(clickTimerRef.current);
    clickTimerRef.current = null;
    onZoom?.(item);
  }

  function handleKeyDown(e) {
    if (e.key === ' ' && urlRef.current) {
      e.preventDefault();
      window.open(urlRef.current, '_blank');
    }
  }

  const ringClass = selected
    ? 'ring-2 ring-primary'
    : isAnchor
      ? 'ring-2 ring-primary/40'
      : '';

  return (
    <div
      role="gridcell"
      aria-selected={selected ? 'true' : 'false'}
      data-gallery-cell
      tabIndex={0}
      className={`relative w-32 h-32 rounded-md overflow-hidden cursor-pointer bg-muted focus:outline-none focus:ring-2 focus:ring-primary transition-transform duration-150 motion-reduce:transition-none hover:scale-[1.08] motion-reduce:hover:scale-100 ${ringClass}`}
      onKeyDown={handleKeyDown}
    >
      <img
        src={url ?? PLACEHOLDER_SRC}
        alt={item.params?.prompt ?? ''}
        className={`w-full h-full object-cover${url ? '' : ' opacity-0'}`}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
      />
      {selected && (
        <div
          aria-hidden="true"
          className="absolute top-1 left-1 rounded-full bg-primary text-primary-foreground h-5 w-5 flex items-center justify-center text-xs"
        >
          ✓
        </div>
      )}
      {!url && (
        <div
          aria-hidden="true"
          className="absolute inset-0 w-full h-full bg-muted flex items-center justify-center text-xs text-muted-foreground pointer-events-none"
        >
          …
        </div>
      )}
    </div>
  );
}

export function GalleryGrid({
  items,
  resolveImageUrl,
  onOpenViewer,
  onToggle,
  onRange,
  onZoom,
  selectedIds,
  anchorId,
}) {
  const [page, setPage] = useState(0);

  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));

  useEffect(() => {
    setPage(0);
  }, [items.length]);

  const pageItems = items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
        No images in this gallery yet
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div role="grid" className="grid gap-2" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
        {pageItems.map((item) => (
          <GalleryThumbnail
            key={item.id}
            item={item}
            resolveImageUrl={resolveImageUrl}
            onOpenViewer={onOpenViewer}
            onToggle={onToggle}
            onRange={onRange}
            onZoom={onZoom}
            selected={selectedIds?.has(item.id) ?? false}
            isAnchor={anchorId === item.id}
          />
        ))}
      </div>

      <div className="flex items-center justify-center gap-4 text-sm">
        <button
          type="button"
          className="px-3 py-1 rounded border disabled:opacity-40 disabled:cursor-not-allowed hover:bg-muted"
          disabled={page === 0}
          onClick={() => setPage((p) => p - 1)}
          aria-label="Prev"
        >
          Prev
        </button>
        <span>Page {page + 1} of {totalPages}</span>
        <button
          type="button"
          className="px-3 py-1 rounded border disabled:opacity-40 disabled:cursor-not-allowed hover:bg-muted"
          disabled={page === totalPages - 1}
          onClick={() => setPage((p) => p + 1)}
          aria-label="Next"
        >
          Next
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 11: Update `GalleryLightbox` to use `useSelection` and render the action bar**

Replace the body of [`lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`](lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx):

```jsx
// src/components/gallery/GalleryLightbox.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X } from 'lucide-react';
import { createCache } from '../../utils/cache';
import { GalleryGrid } from './GalleryGrid';
import { GalleryImageViewer } from './GalleryImageViewer';
import { FloatingActionBar } from './FloatingActionBar';
import { useSelection } from '../../hooks/useSelection';

export function GalleryLightbox({
  galleryId,
  galleryName,
  getGalleryImages,
  onClose,
  trashMode = false,
  onMoveToTrash,
  onRestoreFromTrash,
  onHardDelete,
}) {
  const [items, setItems] = useState([]);
  const [viewerItem, setViewerItem] = useState(null);
  const [opacity, setOpacity] = useState(0.95);

  const selection = useSelection(items);

  const cacheRef = useRef(null);
  const blobUrlsRef = useRef(new Map());
  const childWindowsRef = useRef([]);

  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  function getCache() {
    if (!cacheRef.current) cacheRef.current = createCache();
    return cacheRef.current;
  }

  useEffect(() => {
    getGalleryImages(galleryId).then(setItems);
  }, [galleryId, getGalleryImages]);

  useEffect(() => {
    return () => {
      for (const url of blobUrlsRef.current.values()) {
        try { URL.revokeObjectURL(url); } catch {}
      }
    };
  }, []);

  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') closeAll();
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function closeAll() {
    for (const win of childWindowsRef.current) {
      try { if (win && !win.closed) win.close(); } catch {}
    }
    childWindowsRef.current = [];
    onCloseRef.current();
  }

  function handleWindowOpen(win) {
    if (win) childWindowsRef.current.push(win);
  }

  const resolveImageUrl = useCallback(async (item) => {
    const cache = getCache();
    if (item.cacheKey) {
      if (blobUrlsRef.current.has(item.cacheKey)) {
        return blobUrlsRef.current.get(item.cacheKey);
      }
      try {
        const entry = await cache.get(item.cacheKey);
        if (entry?.blob?.size > 0) {
          const blobUrl = URL.createObjectURL(entry.blob);
          blobUrlsRef.current.set(item.cacheKey, blobUrl);
          return blobUrl;
        }
      } catch {}
    }
    return item.serverImageUrl ?? null;
  }, []);

  async function handleDeleteAction() {
    if (selection.selectedIds.size === 0) return;
    const ids = [...selection.selectedIds];
    await onMoveToTrash?.(ids);
    selection.clear();
    setItems(await getGalleryImages(galleryId));
  }

  async function handleRestoreAction() {
    const ids = [...selection.selectedIds];
    await onRestoreFromTrash?.(ids);
    selection.clear();
    setItems(await getGalleryImages(galleryId));
  }

  async function handleHardDeleteAction() {
    const ids = [...selection.selectedIds];
    await onHardDelete?.(ids);
    selection.clear();
    setItems(await getGalleryImages(galleryId));
  }

  const displayName = (galleryName ?? (trashMode ? 'Trash' : '')).slice(0, 16);

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{ backgroundColor: `rgba(0,0,0,${opacity})` }}
    >
      <div className="flex items-center gap-4 px-4 py-2 border-b border-white/10 text-white">
        <span className="font-medium truncate max-w-[160px]">{displayName}</span>
        <input
          type="range"
          min="0.7"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="w-28 accent-primary"
          aria-label="Background opacity"
        />
        <div className="flex-1" />
        <button
          type="button"
          aria-label="Close gallery"
          onClick={closeAll}
          className="p-1.5 rounded-full hover:bg-white/10 transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4">
        {viewerItem ? (
          <GalleryImageViewer
            item={viewerItem}
            resolveImageUrl={resolveImageUrl}
            onBack={() => setViewerItem(null)}
            onWindowOpen={handleWindowOpen}
          />
        ) : (
          <GalleryGrid
            items={items}
            resolveImageUrl={resolveImageUrl}
            onOpenViewer={setViewerItem}
            onToggle={(id) => selection.toggle(id)}
            onRange={(id) => selection.rangeTo(id)}
            onZoom={() => {}}
            selectedIds={selection.selectedIds}
            anchorId={selection.anchorId}
          />
        )}
      </div>

      {!viewerItem && (
        <FloatingActionBar
          selectedCount={selection.selectedIds.size}
          trashMode={trashMode}
          onDelete={handleDeleteAction}
          onRestore={handleRestoreAction}
          onHardDelete={handleHardDeleteAction}
          onClear={selection.clear}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 12: Update `GalleryLightbox.test.jsx` to cover action bar**

Append to [`lcm-sr-ui/src/components/gallery/GalleryLightbox.test.jsx`](lcm-sr-ui/src/components/gallery/GalleryLightbox.test.jsx):

```jsx
describe('GalleryLightbox — selection action bar', () => {
  it('renders the action bar after selecting a thumbnail and fires onMoveToTrash on Delete', async () => {
    const items = [
      { id: 'id_1', galleryId: 'gal_1', cacheKey: 'k1', serverImageUrl: 'x', params: {}, addedAt: 1 },
      { id: 'id_2', galleryId: 'gal_1', cacheKey: 'k2', serverImageUrl: 'y', params: {}, addedAt: 2 },
    ];
    const getGalleryImages = vi.fn(async () => items);
    const onMoveToTrash = vi.fn(async () => {});

    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="gal_1"
          galleryName="Alpha"
          getGalleryImages={getGalleryImages}
          onClose={vi.fn()}
          onMoveToTrash={onMoveToTrash}
          onRestoreFromTrash={vi.fn()}
          onHardDelete={vi.fn()}
        />,
      );
    });

    const firstImg = screen.getAllByRole('img')[0];
    fireEvent.click(firstImg);
    await new Promise((r) => setTimeout(r, 200));
    expect(screen.getByText('1 selected')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /^delete$/i }));
    expect(onMoveToTrash).toHaveBeenCalledWith(['id_1']);
  });

  it('in trash mode, menu shows Restore and Delete permanently', async () => {
    const items = [
      { id: 'id_1', galleryId: '__trash__', cacheKey: 'k1', serverImageUrl: 'x', params: {}, addedAt: 1 },
    ];
    const getGalleryImages = vi.fn(async () => items);
    const onRestore = vi.fn(async () => {});
    const onHardDelete = vi.fn(async () => {});

    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="__trash__"
          galleryName="Trash"
          trashMode
          getGalleryImages={getGalleryImages}
          onClose={vi.fn()}
          onMoveToTrash={vi.fn()}
          onRestoreFromTrash={onRestore}
          onHardDelete={onHardDelete}
        />,
      );
    });

    fireEvent.click(screen.getAllByRole('img')[0]);
    await new Promise((r) => setTimeout(r, 200));
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    expect(screen.getByRole('menuitem', { name: /restore/i })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /delete permanently/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 13: Run the full gallery test subset**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery src/hooks/useSelection.test.jsx
```

Expected: all existing + new selection and action-bar tests pass.

- [ ] **Step 14: Commit**

```bash
git add lcm-sr-ui/src/hooks/useSelection.js lcm-sr-ui/src/hooks/useSelection.test.jsx lcm-sr-ui/src/components/gallery/FloatingActionBar.jsx lcm-sr-ui/src/components/gallery/FloatingActionBar.test.jsx lcm-sr-ui/src/components/gallery/GalleryGrid.jsx lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx lcm-sr-ui/src/components/gallery/GalleryLightbox.test.jsx
git commit -m "feat: add thumbnail selection and floating action bar

Refs STABL-kiazzizg"
```

---

### Task 4: Trash Gallery UI Surface

**Files:**
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.jsx`
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`
- Modify: `lcm-sr-ui/src/App.jsx` (wire trash-open state)

- [ ] **Step 1: Write the failing trash-tab test**

Append to [`lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`](lcm-sr-ui/src/components/options/OptionsPanel.test.jsx):

```jsx
describe('OptionsPanel — trash tab', () => {
  it('renders a Trash button outside the gallery list and calls onOpenTrash on click', () => {
    const onOpenTrash = vi.fn();
    render(
      <OptionsPanel
        {...baseProps}
        galleries={[]}
        onOpenTrash={onOpenTrash}
      />,
    );
    const btn = screen.getByRole('button', { name: /trash/i });
    fireEvent.click(btn);
    expect(onOpenTrash).toHaveBeenCalled();
  });
});
```

(Where `baseProps` reflects the test file's existing prop defaults; copy the nearest preceding test's prop shape.)

- [ ] **Step 2: Run to verify it fails**

```bash
cd lcm-sr-ui && npm test -- src/components/options/OptionsPanel.test.jsx
```

Expected: no Trash button rendered.

- [ ] **Step 3: Add the Trash button inside `OptionsPanel`**

Inside the gallery selector area of `OptionsPanel.jsx` (look for where `GallerySelector` or equivalent gallery chrome renders; add adjacent but visually separated):

```jsx
// near gallery controls
import { Trash } from 'lucide-react';

// ...

<div className="mt-2 pt-2 border-t">
  <button
    type="button"
    onClick={onOpenTrash}
    className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
    aria-label="Open trash"
  >
    <Trash className="h-4 w-4" />
    <span>Trash</span>
  </button>
</div>
```

Ensure `onOpenTrash` is declared in the prop list.

- [ ] **Step 4: Wire `App.jsx` to render a trash-mode lightbox**

In [`lcm-sr-ui/src/App.jsx`](lcm-sr-ui/src/App.jsx), add a `trashOpen` state and wire `onOpenTrash` down to `OptionsPanel`; render a second `GalleryLightbox` when `trashOpen` is true:

```jsx
const [trashOpen, setTrashOpen] = useState(false);
const {
  moveToTrash, restoreFromTrash, hardDelete, getTrashItems, TRASH_GALLERY_ID,
} = useGalleries();

// ... pass to OptionsPanel
<OptionsPanel
  /* existing props */
  onOpenTrash={() => setTrashOpen(true)}
/>

{trashOpen && (
  <GalleryLightbox
    galleryId={TRASH_GALLERY_ID}
    galleryName="Trash"
    trashMode
    getGalleryImages={getTrashItems}
    onClose={() => setTrashOpen(false)}
    onMoveToTrash={moveToTrash}
    onRestoreFromTrash={restoreFromTrash}
    onHardDelete={hardDelete}
  />
)}
```

Also pass `onMoveToTrash={moveToTrash}` to the existing non-trash GalleryLightbox render path.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd lcm-sr-ui && npm test -- src/components/options/OptionsPanel.test.jsx
```

Expected: all existing + trash-tab test pass.

- [ ] **Step 6: Commit**

```bash
git add lcm-sr-ui/src/components/options/OptionsPanel.jsx lcm-sr-ui/src/components/options/OptionsPanel.test.jsx lcm-sr-ui/src/App.jsx
git commit -m "feat: add trash tab that opens a trash-mode gallery lightbox

Refs STABL-rxzigckx"
```

---

### Task 5: Thumbnail Hover Magnify

Already partially applied in Task 3 via `hover:scale-[1.08]` on the thumbnail. This task adds the regression test and the reduced-motion verification.

**Files:**
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`

- [ ] **Step 1: Write the failing hover test**

```jsx
describe('GalleryGrid — hover magnify', () => {
  it('thumbnail has hover:scale-[1.08] and motion-reduce guard classes', async () => {
    const items = [makeItem(0)];
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          resolveImageUrl={resolve}
          onOpenViewer={vi.fn()}
          onToggle={vi.fn()}
          onRange={vi.fn()}
          onZoom={vi.fn()}
          selectedIds={new Set()}
          anchorId={null}
        />,
      );
    });
    const cell = screen.getAllByRole('gridcell')[0];
    expect(cell.className).toContain('hover:scale-[1.08]');
    expect(cell.className).toContain('motion-reduce:hover:scale-100');
    expect(cell.className).toContain('transition-transform');
    expect(cell.className).toContain('motion-reduce:transition-none');
  });
});
```

- [ ] **Step 2: Run to verify it passes already**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/GalleryGrid.test.jsx
```

Expected: pass (classes already applied in Task 3).

- [ ] **Step 3: Ensure grid `overflow-visible` so magnified tiles don't clip**

Update [`lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`](lcm-sr-ui/src/components/gallery/GalleryGrid.jsx) grid wrapper class from:

```jsx
<div role="grid" className="grid gap-2" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
```

to:

```jsx
<div role="grid" className="grid gap-2 overflow-visible" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
```

- [ ] **Step 4: Commit**

```bash
git add lcm-sr-ui/src/components/gallery/GalleryGrid.jsx lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx
git commit -m "feat: verify hover magnify and add overflow-visible guard

Refs STABL-jrediyzc"
```

---

### Task 6: Double-Click Zoom Overlay

**Files:**
- Create: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.jsx`
- Create: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`

- [ ] **Step 1: Write the failing zoom-overlay tests**

```jsx
// lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.test.jsx
// @vitest-environment jsdom
import { render, screen, fireEvent, act } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GalleryZoomOverlay } from './GalleryZoomOverlay';

function item() {
  return { id: 'id_1', cacheKey: 'k1', serverImageUrl: 'http://example.com/a.png', params: { prompt: 'x' }, addedAt: 1 };
}

describe('GalleryZoomOverlay', () => {
  it('renders image at 50vw/50vh bounds and closes on Close button', async () => {
    const onClose = vi.fn();
    await act(async () => {
      render(
        <GalleryZoomOverlay
          item={item()}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onClose={onClose}
        />,
      );
    });
    const img = await screen.findByAltText('x');
    expect(img.style.maxWidth).toBe('50vw');
    expect(img.style.maxHeight).toBe('50vh');
    fireEvent.click(screen.getByRole('button', { name: /close zoom/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it('closes on click outside the image frame', async () => {
    const onClose = vi.fn();
    await act(async () => {
      render(
        <GalleryZoomOverlay
          item={item()}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onClose={onClose}
        />,
      );
    });
    fireEvent.mouseDown(screen.getByTestId('zoom-backdrop'));
    expect(onClose).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/GalleryZoomOverlay.test.jsx
```

Expected: module-not-found.

- [ ] **Step 3: Implement the overlay**

```jsx
// lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.jsx
import React, { useEffect, useState, useRef } from 'react';
import { X } from 'lucide-react';

export function GalleryZoomOverlay({ item, resolveImageUrl, onClose }) {
  const [url, setUrl] = useState(null);
  const frameRef = useRef(null);

  useEffect(() => {
    let active = true;
    resolveImageUrl(item).then((resolved) => {
      if (active) setUrl(resolved);
    });
    return () => { active = false; };
  }, [item.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleBackdropMouseDown(e) {
    if (!frameRef.current) return;
    if (!frameRef.current.contains(e.target)) onClose?.();
  }

  return (
    <div
      data-testid="zoom-backdrop"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onMouseDown={handleBackdropMouseDown}
    >
      <div
        ref={frameRef}
        className="relative rounded-md shadow-2xl bg-background p-2"
      >
        <button
          type="button"
          aria-label="Close zoom"
          onClick={onClose}
          className="absolute -top-2 -right-2 rounded-full bg-background border p-1 hover:bg-muted"
        >
          <X className="h-4 w-4" />
        </button>
        {url ? (
          <img
            src={url}
            alt={item.params?.prompt ?? ''}
            style={{ maxWidth: '50vw', maxHeight: '50vh' }}
            className="object-contain block"
          />
        ) : (
          <div style={{ width: '25vw', height: '25vh' }} className="bg-muted rounded" />
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run to verify overlay tests pass**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/GalleryZoomOverlay.test.jsx
```

Expected: both tests pass.

- [ ] **Step 5: Wire zoom overlay into `GalleryLightbox`**

In [`lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`](lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx):

```jsx
import { GalleryZoomOverlay } from './GalleryZoomOverlay';

// add state
const [zoomItem, setZoomItem] = useState(null);

// replace the grid's onZoom prop
<GalleryGrid
  /* ... */
  onZoom={(item) => setZoomItem(item)}
  /* ... */
/>

// render overlay near the bottom of the component return
{zoomItem && (
  <GalleryZoomOverlay
    item={zoomItem}
    resolveImageUrl={resolveImageUrl}
    onClose={() => setZoomItem(null)}
  />
)}
```

- [ ] **Step 6: Run full gallery subset**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.jsx lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.test.jsx lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx
git commit -m "feat: add double-click zoom overlay (50vw/50vh AR-preserved)

Refs STABL-rnwijffv"
```

---

### Task 7: Open-In-New-Tab Button

**Files:**
- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.test.jsx`

- [ ] **Step 1: Write failing tests for the new button**

Append to `GalleryImageViewer.test.jsx`:

```jsx
it('renders an Open in new tab button that calls window.open with resolved url', async () => {
  const openSpy = vi.spyOn(window, 'open').mockReturnValue({ closed: false });
  const onWindowOpen = vi.fn();
  await act(async () => {
    render(
      <GalleryImageViewer
        item={{ id: 'id_1', serverImageUrl: 'http://example.com/a.png', params: { prompt: 'p' }, addedAt: 1 }}
        resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
        onBack={vi.fn()}
        onWindowOpen={onWindowOpen}
      />,
    );
  });
  await screen.findByAltText('p');
  fireEvent.click(screen.getByRole('button', { name: /open in new tab/i }));
  expect(openSpy).toHaveBeenCalledWith('http://example.com/a.png', '_blank');
  expect(onWindowOpen).toHaveBeenCalled();
  openSpy.mockRestore();
});
```

Append to `GalleryZoomOverlay.test.jsx`:

```jsx
it('renders an Open in new tab button', async () => {
  const openSpy = vi.spyOn(window, 'open').mockReturnValue({ closed: false });
  await act(async () => {
    render(
      <GalleryZoomOverlay
        item={{ id: 'id_1', serverImageUrl: 'http://example.com/a.png', params: { prompt: 'p' }, addedAt: 1 }}
        resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
        onClose={vi.fn()}
      />,
    );
  });
  await screen.findByAltText('p');
  fireEvent.click(screen.getByRole('button', { name: /open in new tab/i }));
  expect(openSpy).toHaveBeenCalledWith('http://example.com/a.png', '_blank');
  openSpy.mockRestore();
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/GalleryImageViewer.test.jsx src/components/gallery/GalleryZoomOverlay.test.jsx
```

Expected: button not found in either.

- [ ] **Step 3: Add the button to `GalleryImageViewer`**

In [`lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx`](lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx), add import and button inside the toolbar region (alongside Back):

```jsx
import { ChevronLeft, ExternalLink } from 'lucide-react';

// inside the return, next to the Back button
<button
  type="button"
  aria-label="Open in new tab"
  onClick={() => {
    if (!urlRef.current) return;
    const win = window.open(urlRef.current, '_blank');
    onWindowOpen?.(win);
  }}
  className="absolute top-2 right-2 z-10 p-1 rounded-full bg-background/80 hover:bg-background transition-colors"
>
  <ExternalLink className="h-5 w-5" />
</button>
```

- [ ] **Step 4: Add the button to `GalleryZoomOverlay`**

Inside the overlay frame, add the button alongside the close X:

```jsx
import { ExternalLink, X } from 'lucide-react';

// inside the frame div, next to close button
<button
  type="button"
  aria-label="Open in new tab"
  onClick={() => url && window.open(url, '_blank')}
  className="absolute -top-2 -left-2 rounded-full bg-background border p-1 hover:bg-muted"
>
  <ExternalLink className="h-4 w-4" />
</button>
```

- [ ] **Step 5: Run tests to verify pass**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/GalleryImageViewer.test.jsx src/components/gallery/GalleryZoomOverlay.test.jsx
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.jsx lcm-sr-ui/src/components/gallery/GalleryZoomOverlay.test.jsx
git commit -m "feat: surface open-in-new-tab as a visible button in viewer and zoom

Refs STABL-lgryhltt"
```

---

### Task 8: Keyboard Navigation Wired To Keymap

**Files:**
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx`
- Modify: `lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx`

- [ ] **Step 1: Write failing keyboard-nav tests**

Append to `GalleryGrid.test.jsx`:

```jsx
describe('GalleryGrid — keyboard navigation', () => {
  function renderGrid(overrides = {}) {
    const items = Array.from({ length: 10 }, (_, i) => makeItem(i));
    const props = {
      items, resolveImageUrl: resolve, onOpenViewer: vi.fn(),
      onToggle: vi.fn(), onRange: vi.fn(), onZoom: vi.fn(),
      onDeleteAction: vi.fn(), onSelectAll: vi.fn(), onDeselectAll: vi.fn(),
      selectedIds: new Set(), anchorId: null,
      keymap: {
        matches: (action, e) => {
          const map = {
            right: e.code === 'ArrowRight',
            left: e.code === 'ArrowLeft',
            down: e.code === 'ArrowDown',
            up: e.code === 'ArrowUp',
            delete: e.code === 'Backspace',
            delete_alt: e.code === 'Delete',
            select_all: e.code === 'KeyA' && (e.metaKey || e.ctrlKey),
            deselect_all: e.code === 'Escape',
            zoom: e.code === 'Enter',
            open_new_tab: e.code === 'Space',
          };
          return map[action] ?? false;
        },
      },
      ...overrides,
    };
    render(<GalleryGrid {...props} />);
    return props;
  }

  it('ArrowRight moves focus to the next cell', async () => {
    const props = renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'ArrowRight' });
    expect(document.activeElement).toBe(screen.getAllByRole('gridcell')[1]);
  });

  it('Backspace calls onDeleteAction with selection', async () => {
    const selected = new Set(['id_2']);
    const props = renderGrid({ selectedIds: selected });
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'Backspace' });
    expect(props.onDeleteAction).toHaveBeenCalledWith(['id_2']);
  });

  it('Backspace with empty selection deletes focused cell', async () => {
    const props = renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'Backspace' });
    expect(props.onDeleteAction).toHaveBeenCalledWith(['id_0']);
  });

  it('Cmd+A triggers select all', async () => {
    const props = renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'KeyA', metaKey: true });
    expect(props.onSelectAll).toHaveBeenCalled();
  });

  it('Escape triggers deselect all', async () => {
    const props = renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'Escape' });
    expect(props.onDeselectAll).toHaveBeenCalled();
  });
});
```

Append to `GalleryImageViewer.test.jsx`:

```jsx
describe('GalleryImageViewer — keyboard navigation', () => {
  const makeKeymap = () => ({
    matches: (action, e) => ({
      next: e.code === 'ArrowRight',
      prev: e.code === 'ArrowLeft',
      delete: e.code === 'Backspace',
      delete_alt: e.code === 'Delete',
      close: e.code === 'Escape',
      open_new_tab: e.code === 'Space',
    }[action] ?? false),
  });

  it('ArrowRight calls onNext', async () => {
    const onNext = vi.fn();
    await act(async () => {
      render(
        <GalleryImageViewer
          item={{ id: 'id_1', serverImageUrl: 'x', params: {}, addedAt: 1 }}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onBack={vi.fn()}
          onNext={onNext}
          onPrev={vi.fn()}
          onDelete={vi.fn()}
          keymap={makeKeymap()}
          onWindowOpen={vi.fn()}
        />,
      );
    });
    fireEvent.keyDown(document, { code: 'ArrowRight' });
    expect(onNext).toHaveBeenCalled();
  });

  it('Backspace calls onDelete', async () => {
    const onDelete = vi.fn();
    await act(async () => {
      render(
        <GalleryImageViewer
          item={{ id: 'id_1', serverImageUrl: 'x', params: {}, addedAt: 1 }}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onBack={vi.fn()}
          onNext={vi.fn()}
          onPrev={vi.fn()}
          onDelete={onDelete}
          keymap={makeKeymap()}
          onWindowOpen={vi.fn()}
        />,
      );
    });
    fireEvent.keyDown(document, { code: 'Backspace' });
    expect(onDelete).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery/GalleryGrid.test.jsx src/components/gallery/GalleryImageViewer.test.jsx
```

Expected: new keyboard-nav tests fail; others still pass.

- [ ] **Step 3: Add keyboard handlers to `GalleryGrid`**

Update `GalleryGrid` function signature and body:

```jsx
export function GalleryGrid({
  items,
  resolveImageUrl,
  onOpenViewer,
  onToggle,
  onRange,
  onZoom,
  onDeleteAction,
  onSelectAll,
  onDeselectAll,
  selectedIds,
  anchorId,
  keymap,
}) {
  const [page, setPage] = useState(0);
  const gridRef = useRef(null);
  const COLS = 5;

  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  useEffect(() => { setPage(0); }, [items.length]);
  const pageItems = items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function focusCellIndex(idx) {
    const cells = gridRef.current?.querySelectorAll('[data-gallery-cell]');
    if (!cells) return;
    const clamped = Math.max(0, Math.min(cells.length - 1, idx));
    cells[clamped]?.focus();
  }

  function currentFocusedIndex() {
    const cells = Array.from(gridRef.current?.querySelectorAll('[data-gallery-cell]') ?? []);
    return cells.indexOf(document.activeElement);
  }

  function handleKeyDown(e) {
    if (!keymap) return;
    const idx = currentFocusedIndex();
    if (idx < 0) return;
    if (keymap.matches('right', e) || keymap.matches('next', e)) { e.preventDefault(); focusCellIndex(idx + 1); return; }
    if (keymap.matches('left', e)  || keymap.matches('prev', e)) { e.preventDefault(); focusCellIndex(idx - 1); return; }
    if (keymap.matches('down', e)) { e.preventDefault(); focusCellIndex(idx + COLS); return; }
    if (keymap.matches('up', e))   { e.preventDefault(); focusCellIndex(idx - COLS); return; }
    if (keymap.matches('select_all', e)) { e.preventDefault(); onSelectAll?.(); return; }
    if (keymap.matches('deselect_all', e)) { e.preventDefault(); onDeselectAll?.(); return; }
    if (keymap.matches('delete', e) || keymap.matches('delete_alt', e)) {
      e.preventDefault();
      const selected = selectedIds && selectedIds.size > 0 ? [...selectedIds] : [pageItems[idx]?.id].filter(Boolean);
      if (selected.length > 0) onDeleteAction?.(selected);
      return;
    }
    if (keymap.matches('zoom', e)) { e.preventDefault(); if (pageItems[idx]) onZoom?.(pageItems[idx]); return; }
  }
```

Wrap the grid in a keydown listener:

```jsx
<div ref={gridRef} role="grid" tabIndex={-1} onKeyDown={handleKeyDown} className="grid gap-2 overflow-visible" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)` }}>
```

Note: `keymap` is optional; when omitted, grid renders without keyboard handlers (keeps existing unit tests passing without keymap).

- [ ] **Step 4: Wire `GalleryImageViewer` for prev/next/delete/open_new_tab/close**

Extend the component:

```jsx
export function GalleryImageViewer({ item, resolveImageUrl, onBack, onWindowOpen, onNext, onPrev, onDelete, keymap }) {
  // ... existing state ...

  useEffect(() => {
    if (!keymap) return;
    function onKeyDown(e) {
      if (keymap.matches('next', e)) { e.preventDefault(); onNext?.(); return; }
      if (keymap.matches('prev', e)) { e.preventDefault(); onPrev?.(); return; }
      if (keymap.matches('delete', e) || keymap.matches('delete_alt', e)) { e.preventDefault(); onDelete?.(); return; }
      if (keymap.matches('close', e)) { e.preventDefault(); onBack?.(); return; }
      if (keymap.matches('open_new_tab', e) && urlRef.current) {
        e.preventDefault();
        const win = window.open(urlRef.current, '_blank');
        onWindowOpen?.(win);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [keymap, onNext, onPrev, onDelete, onBack, onWindowOpen]);

  // keep existing Space-specific handler for backwards compat if keymap absent
```

Ensure the existing inline Space handler is guarded to not fire when `keymap` is provided (to avoid double-open):

```jsx
useEffect(() => {
  if (keymap) return;
  function onKeyDown(e) {
    if (e.key === ' ' && urlRef.current) {
      e.preventDefault();
      const win = window.open(urlRef.current, '_blank');
      onWindowOpen?.(win);
    }
  }
  document.addEventListener('keydown', onKeyDown);
  return () => document.removeEventListener('keydown', onKeyDown);
}, [keymap, onWindowOpen]);
```

- [ ] **Step 5: Wire `GalleryLightbox` to pass `keymap` + navigation callbacks**

In `GalleryLightbox.jsx`:

```jsx
import { useKeymap } from '../../hooks/useKeymap';

// inside the component
const keymap = useKeymap();

// new callbacks
const currentIndex = items.findIndex((it) => it.id === viewerItem?.id);
const nextItem = () => {
  if (items.length === 0) return;
  const next = (currentIndex + 1 + items.length) % items.length;
  setViewerItem(items[next]);
};
const prevItem = () => {
  if (items.length === 0) return;
  const next = (currentIndex - 1 + items.length) % items.length;
  setViewerItem(items[next]);
};
async function handleViewerDelete() {
  if (!viewerItem) return;
  const ids = [viewerItem.id];
  if (trashMode) await onHardDelete?.(ids);
  else await onMoveToTrash?.(ids);
  const refreshed = await getGalleryImages(galleryId);
  setItems(refreshed);
  if (refreshed.length === 0) setViewerItem(null);
  else setViewerItem(refreshed[Math.min(currentIndex, refreshed.length - 1)]);
}

// pass keymap + onDeleteAction / onSelectAll / onDeselectAll to grid
<GalleryGrid
  /* ... */
  keymap={keymap}
  onDeleteAction={async (ids) => {
    if (trashMode) await onHardDelete?.(ids);
    else await onMoveToTrash?.(ids);
    selection.clear();
    setItems(await getGalleryImages(galleryId));
  }}
  onSelectAll={() => selection.selectAll()}
  onDeselectAll={() => selection.clear()}
/>

// and to viewer
<GalleryImageViewer
  item={viewerItem}
  resolveImageUrl={resolveImageUrl}
  onBack={() => setViewerItem(null)}
  onWindowOpen={handleWindowOpen}
  onNext={nextItem}
  onPrev={prevItem}
  onDelete={handleViewerDelete}
  keymap={keymap}
/>
```

- [ ] **Step 6: Run full gallery subset to verify everything**

```bash
cd lcm-sr-ui && npm test -- src/components/gallery src/hooks/useKeymap.test.jsx src/hooks/useSelection.test.jsx src/hooks/useGalleries.test.jsx
```

Expected: all existing + all new tests pass.

- [ ] **Step 7: Run the backend test suite to confirm nothing regressed**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_keymap_routes.py tests/test_mode_config.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add lcm-sr-ui/src/components/gallery/GalleryGrid.jsx lcm-sr-ui/src/components/gallery/GalleryGrid.test.jsx lcm-sr-ui/src/components/gallery/GalleryImageViewer.jsx lcm-sr-ui/src/components/gallery/GalleryImageViewer.test.jsx lcm-sr-ui/src/components/gallery/GalleryLightbox.jsx
git commit -m "feat: wire keyboard navigation through the keymap system

Refs STABL-uqwjxzvr"
```

---

## Post-Tasks

- [ ] **Refresh drift for changed docs and code**

```bash
drift check --changed lcm-sr-ui --changed server --changed conf --changed docs/superpowers
```

Expected: `ok` or follow prompts to add/update anchors.

- [ ] **Run the full test suites one more time**

```bash
cd lcm-sr-ui && npm test
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest -q
```

Expected: all pass.

- [ ] **Assign commits to FP sub-issues**

Use `fp issue assign <id> --rev <commit>` for each task's commit, then close each sub-issue:

```bash
fp issue update --status done STABL-gcqvmbpo
fp issue update --status done STABL-edzszkjk
fp issue update --status done STABL-kiazzizg
fp issue update --status done STABL-rxzigckx
fp issue update --status done STABL-jrediyzc
fp issue update --status done STABL-rnwijffv
fp issue update --status done STABL-lgryhltt
fp issue update --status done STABL-uqwjxzvr
fp issue update --status done STABL-osgiqtxv
```

---

## Self-Review

**Spec coverage:**

- keymap system — Task 1
- trash data layer — Task 2
- selection + action bar — Task 3
- trash UI surface — Task 4
- hover magnify — Task 5 (plus class applied in Task 3)
- zoom overlay — Task 6
- open-in-new-tab button — Task 7
- keyboard navigation — Task 8
- additive system-prompt stacking, empty-string override — N/A (this plan is gallery UX, not chat)
- reduced-motion guard — covered in Task 5 class changes
- context-aware viewer delete — covered in Task 8 `handleViewerDelete` branches on `trashMode`
- double-click vs single-click guard — covered in Task 3 `GalleryThumbnail` with `setTimeout` gate

**Placeholder scan:**

- No "TODO", "TBD", "fill in", or "Similar to Task N" references remain
- All code blocks contain the actual code to apply
- `baseProps` reference in Task 4 Step 1 acknowledges that the test file already defines one — the engineer copies from the nearest existing test

**Type consistency:**

- `TRASH_GALLERY_ID` string constant exported from `useGalleries`, referenced by overlay / grid / lightbox
- `keymap.matches(action, event)` signature consistent across grid and viewer
- `useSelection()` returns `selectedIds`, `anchorId`, `toggle`, `rangeTo`, `selectAll`, `clear` — consumers reference only those
- `moveToTrash([ids])` / `restoreFromTrash([ids])` / `hardDelete([ids])` all take arrays — consumers always pass arrays
- `FloatingActionBar` props `selectedCount`, `trashMode`, `onDelete`, `onRestore`, `onHardDelete`, `onClear` match the lightbox caller

**Fresh-eyes notes:**

- Task 4 `OptionsPanel` change reads from the existing gallery selector area; if the current file layout is unexpected, the engineer should add the Trash button in whatever nearby location still sits outside the gallery list — the spec constraint is "outside the gallery list", not an exact location.
- The `setTimeout(180ms)` single-click gate in Task 3 is a pragmatic solution to double-click resolution; the regression test at Task 5 depends on the setup already in place.
- The `currentIndex` computation in Task 8 returns -1 when `viewerItem` is null; guards avoid that path and the math inside `nextItem`/`prevItem` never runs while viewerItem is null.
