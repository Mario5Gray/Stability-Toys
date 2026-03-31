# Img2img Source Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the active img2img source across reloads and later visits for both uploaded and chat-origin images, while restoring the Init Image UI and a sane denoise fallback.

**Architecture:** Add a dedicated frontend persistence helper backed by a separate IndexedDB database plus a small localStorage pointer for the active source id. Keep `App.jsx` as the owner of the live `initImage` UI state, but bootstrap that state from durable storage on startup and route both upload and chat-origin source selection through the same persistence layer. Denoise remains part of normal image/draft params, with the persisted source contributing only a fallback `defaultDenoiseStrength`.

**Tech Stack:** React, Vitest, jsdom, IndexedDB, localStorage, existing frontend hooks in `lcm-sr-ui`

---

## File Structure

- `lcm-sr-ui/src/utils/constants.js`
  - Add `DEFAULT_IMG2IMG_DENOISE_STRENGTH = 0.5`
- `lcm-sr-ui/src/utils/img2imgSourceStore.js`
  - New persistence helper for `lcm-img2img` IndexedDB and `lcm-active-img2img-source`
- `lcm-sr-ui/src/utils/img2imgSourceStore.test.mjs`
  - New targeted persistence tests
- `lcm-sr-ui/src/hooks/useGenerationParams.js`
  - Add source-default denoise fallback input and precedence logic
- `lcm-sr-ui/src/hooks/useGenerationParams.test.jsx`
  - Add denoise precedence coverage
- `lcm-sr-ui/src/App.jsx`
  - Restore persisted source on startup, persist new sources, clear active source
- `lcm-sr-ui/src/components/chat/ChatDropzone.jsx`
  - Stop creating purely ephemeral init-image state; route uploads through persisted source creation

---

### Task 1: Add Durable Img2img Source Store

**Files:**
- Create: `lcm-sr-ui/src/utils/img2imgSourceStore.js`
- Create: `lcm-sr-ui/src/utils/img2imgSourceStore.test.mjs`
- Modify: `lcm-sr-ui/src/utils/constants.js`

- [ ] **Step 1: Write the failing persistence tests**

```js
// lcm-sr-ui/src/utils/img2imgSourceStore.test.mjs
import { beforeEach, describe, expect, it } from 'vitest';
import {
  clearActiveSource,
  deleteSource,
  loadActiveSource,
  saveSource,
  setActiveSourceId,
} from './img2imgSourceStore';

describe('img2imgSourceStore', () => {
  beforeEach(async () => {
    localStorage.clear();
    indexedDB.deleteDatabase('lcm-img2img');
  });

  it('persists and restores an uploaded source', async () => {
    const blob = new Blob(['upload-bytes'], { type: 'image/png' });
    const saved = await saveSource({
      originType: 'upload',
      blob,
      mimeType: 'image/png',
      filename: 'init.png',
      defaultDenoiseStrength: 0.5,
    });

    await setActiveSourceId(saved.id);

    const restored = await loadActiveSource();
    expect(restored.id).toBe(saved.id);
    expect(restored.filename).toBe('init.png');
    expect(restored.defaultDenoiseStrength).toBe(0.5);
    expect(restored.blob.size).toBe(blob.size);
  });

  it('clears the active pointer when the source row is missing', async () => {
    await setActiveSourceId('missing-source');
    const restored = await loadActiveSource();

    expect(restored).toBeNull();
    expect(localStorage.getItem('lcm-active-img2img-source')).toBeNull();
  });

  it('removes the active pointer when clearing the active source', async () => {
    const saved = await saveSource({
      originType: 'upload',
      blob: new Blob(['x'], { type: 'image/png' }),
      mimeType: 'image/png',
      filename: 'x.png',
      defaultDenoiseStrength: 0.5,
    });
    await setActiveSourceId(saved.id);

    await clearActiveSource();

    expect(await loadActiveSource()).toBeNull();
    expect(localStorage.getItem('lcm-active-img2img-source')).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd lcm-sr-ui && npx vitest run src/utils/img2imgSourceStore.test.mjs`

Expected: FAIL with module-not-found for `img2imgSourceStore.js`

- [ ] **Step 3: Add the constant and minimal persistence helper**

```js
// lcm-sr-ui/src/utils/constants.js
export const DEFAULT_IMG2IMG_DENOISE_STRENGTH = 0.5;
```

```js
// lcm-sr-ui/src/utils/img2imgSourceStore.js
const DB_NAME = 'lcm-img2img';
const DB_VERSION = 1;
const STORE_NAME = 'img2img_sources';
const ACTIVE_KEY = 'lcm-active-img2img-source';

let dbPromise = null;

function openDatabase() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('originMessageId', 'originMessageId', { unique: false });
        store.createIndex('originType', 'originType', { unique: false });
        store.createIndex('updatedAt', 'updatedAt', { unique: false });
      }
    };
  });
  return dbPromise;
}

export async function saveSource(source) {
  const now = Date.now();
  const row = {
    id: source.id || crypto.randomUUID(),
    originType: source.originType,
    originMessageId: source.originMessageId || null,
    blob: source.blob,
    mimeType: source.mimeType || source.blob?.type || 'application/octet-stream',
    filename: source.filename || 'img2img-source',
    cacheKey: source.cacheKey || null,
    serverImageUrl: source.serverImageUrl || null,
    defaultDenoiseStrength: Number(source.defaultDenoiseStrength ?? 0.5),
    createdAt: source.createdAt || now,
    updatedAt: now,
  };

  const db = await openDatabase();
  await new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readwrite');
    tx.objectStore(STORE_NAME).put(row);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  return row;
}

export function setActiveSourceId(id) {
  if (!id) localStorage.removeItem(ACTIVE_KEY);
  else localStorage.setItem(ACTIVE_KEY, id);
}

export function getActiveSourceId() {
  return localStorage.getItem(ACTIVE_KEY);
}

export async function getSource(id) {
  if (!id) return null;
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readonly');
    const req = tx.objectStore(STORE_NAME).get(id);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function loadActiveSource() {
  const id = getActiveSourceId();
  const row = await getSource(id);
  if (!row && id) localStorage.removeItem(ACTIVE_KEY);
  return row;
}

export async function deleteSource(id) {
  if (!id) return;
  const db = await openDatabase();
  await new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readwrite');
    tx.objectStore(STORE_NAME).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearActiveSource() {
  const id = getActiveSourceId();
  localStorage.removeItem(ACTIVE_KEY);
  if (id) await deleteSource(id);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd lcm-sr-ui && npx vitest run src/utils/img2imgSourceStore.test.mjs`

Expected: PASS with 3 passing tests

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/utils/constants.js \
        lcm-sr-ui/src/utils/img2imgSourceStore.js \
        lcm-sr-ui/src/utils/img2imgSourceStore.test.mjs
git commit -m "Add durable img2img source store"
```

### Task 2: Add Denoise Fallback Precedence to Generation Params

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useGenerationParams.js`
- Modify: `lcm-sr-ui/src/hooks/useGenerationParams.test.jsx`

- [ ] **Step 1: Write the failing hook tests**

```jsx
// lcm-sr-ui/src/hooks/useGenerationParams.test.jsx
it('falls back to source default denoise when no selected or draft value is set', () => {
  const { result } = renderHook(() =>
    useGenerationParams(null, null, vi.fn(), null, null, 0.5)
  );

  expect(result.current.effective.denoiseStrength).toBe(0.5);
});

it('prefers selected image denoise over source default', () => {
  const { result } = renderHook(() =>
    useGenerationParams(
      { denoiseStrength: 0.82, prompt: '', size: '512x512', steps: 8, cfg: 1, seed: 1 },
      vi.fn(),
      vi.fn(),
      'msg-1',
      null,
      0.5
    )
  );

  expect(result.current.effective.denoiseStrength).toBe(0.82);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd lcm-sr-ui && npx vitest run src/hooks/useGenerationParams.test.jsx`

Expected: FAIL because the hook signature and/or precedence logic does not yet use the source default

- [ ] **Step 3: Implement the precedence input and logic**

```js
// lcm-sr-ui/src/hooks/useGenerationParams.js
import { DEFAULT_IMG2IMG_DENOISE_STRENGTH } from '../utils/constants';

export function useGenerationParams(
  selectedParams,
  patchSelectedParams,
  runGenerate,
  selectedMsgId,
  initImageFile = null,
  sourceDefaultDenoiseStrength = DEFAULT_IMG2IMG_DENOISE_STRENGTH
) {
  const [denoiseStrength, setDenoiseStrength] = useState(sourceDefaultDenoiseStrength);

  const effective = useMemo(() => {
    const src = selectedParams ?? DEFAULTS;
    const selectedDenoise = Number(selectedParams?.denoiseStrength);
    const draftDenoise = Number(DEFAULTS.denoiseStrength);
    const sourceDenoise = Number(sourceDefaultDenoiseStrength);
    const fallback = Number(DEFAULT_IMG2IMG_DENOISE_STRENGTH);

    const chosen =
      Number.isFinite(selectedDenoise) ? selectedDenoise
      : Number.isFinite(draftDenoise) ? draftDenoise
      : Number.isFinite(sourceDenoise) ? sourceDenoise
      : fallback;

    return {
      ...,
      denoiseStrength: Math.min(1.0, Math.max(0.01, chosen)),
    };
  }, [selectedParams, DEFAULTS, sourceDefaultDenoiseStrength]);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd lcm-sr-ui && npx vitest run src/hooks/useGenerationParams.test.jsx`

Expected: PASS with the new precedence tests and existing hook tests green

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useGenerationParams.js \
        lcm-sr-ui/src/hooks/useGenerationParams.test.jsx
git commit -m "Add img2img source denoise fallback"
```

### Task 3: Persist and Restore the Active Init Image in App Lifecycle

**Files:**
- Modify: `lcm-sr-ui/src/App.jsx`
- Modify: `lcm-sr-ui/src/components/chat/ChatDropzone.jsx`
- Test: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`

- [ ] **Step 1: Write the failing UI lifecycle tests**

```jsx
// lcm-sr-ui/src/components/options/OptionsPanel.test.jsx
it('renders the init-image controls when a restored init image is provided', () => {
  render(
    <OptionsPanel
      ...
      initImage={{
        sourceId: 'src-1',
        file: new File(['x'], 'restored.png', { type: 'image/png' }),
        objectUrl: 'blob:restored',
        filename: 'restored.png',
      }}
      denoiseStrength={0.5}
      onDenoiseStrengthChange={vi.fn()}
    />
  );

  expect(screen.getByText('Init Image')).toBeTruthy();
  expect(screen.getByText('restored.png')).toBeTruthy();
});
```

- [ ] **Step 2: Run test to verify current lifecycle gap**

Run: `cd lcm-sr-ui && npx vitest run src/components/options/OptionsPanel.test.jsx`

Expected: PASS for the render test, but no app-level persistence exists yet. This is the guardrail before wiring `App.jsx`.

- [ ] **Step 3: Add App-level restore/persist helpers and wire them into drop/upload flow**

```js
// lcm-sr-ui/src/App.jsx
import {
  clearActiveSource,
  loadActiveSource,
  saveSource,
  setActiveSourceId,
} from './utils/img2imgSourceStore';
import { DEFAULT_IMG2IMG_DENOISE_STRENGTH } from './utils/constants';

const [initImage, setInitImage] = useState(null);
const [sourceDefaultDenoiseStrength, setSourceDefaultDenoiseStrength] = useState(
  DEFAULT_IMG2IMG_DENOISE_STRENGTH
);

useEffect(() => {
  let cancelled = false;
  (async () => {
    const restored = await loadActiveSource();
    if (!restored || cancelled) return;
    const file = new File([restored.blob], restored.filename, { type: restored.mimeType });
    const objectUrl = URL.createObjectURL(restored.blob);
    setInitImage({
      sourceId: restored.id,
      originType: restored.originType,
      file,
      objectUrl,
      filename: restored.filename,
      cacheKey: restored.cacheKey,
      serverImageUrl: restored.serverImageUrl,
    });
    setSourceDefaultDenoiseStrength(
      restored.defaultDenoiseStrength ?? DEFAULT_IMG2IMG_DENOISE_STRENGTH
    );
  })();
  return () => { cancelled = true; };
}, []);

const persistUploadInitImage = useCallback(async (file) => {
  const row = await saveSource({
    originType: 'upload',
    blob: file,
    mimeType: file.type,
    filename: file.name,
    defaultDenoiseStrength: DEFAULT_IMG2IMG_DENOISE_STRENGTH,
  });
  setActiveSourceId(row.id);
  setSourceDefaultDenoiseStrength(row.defaultDenoiseStrength);
  setInitImage({
    sourceId: row.id,
    originType: row.originType,
    file,
    objectUrl: URL.createObjectURL(file),
    filename: row.filename,
    cacheKey: null,
    serverImageUrl: null,
  });
}, []);

const clearInitImage = useCallback(async () => {
  if (initImage?.objectUrl) URL.revokeObjectURL(initImage.objectUrl);
  setInitImage(null);
  setSourceDefaultDenoiseStrength(DEFAULT_IMG2IMG_DENOISE_STRENGTH);
  await clearActiveSource();
}, [initImage]);

const params = useGenerationParams(
  selectedParams,
  patchSelectedParams,
  runGenerate,
  selectedMsgId,
  initImage?.file || null,
  sourceDefaultDenoiseStrength
);
```

```jsx
// lcm-sr-ui/src/components/chat/ChatDropzone.jsx
export function ChatDropzone({ ..., setInitImageSource, ... }) {
  const onDrop = useMemo(
    () => async (acceptedFiles) => {
      try {
        if (acceptedFiles.length > 0 && setInitImageSource) {
          await setInitImageSource(acceptedFiles[0]);
        }
        await ingestFiles(acceptedFiles);
      } catch (e) {
        console.error('[ChatDropzone] ingest failed:', e);
      }
    },
    [ingestFiles, setInitImageSource]
  );
}
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
cd lcm-sr-ui && npx vitest run \
  src/components/options/OptionsPanel.test.jsx \
  src/hooks/useGenerationParams.test.jsx \
  src/utils/img2imgSourceStore.test.mjs
```

Expected: PASS with restored-init-image UI and denoise precedence tests green

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/App.jsx \
        lcm-sr-ui/src/components/chat/ChatDropzone.jsx \
        lcm-sr-ui/src/components/options/OptionsPanel.test.jsx
git commit -m "Restore persisted img2img source on startup"
```

### Task 4: Support Chat-Origin Source Persistence and Final Verification

**Files:**
- Modify: `lcm-sr-ui/src/App.jsx`
- Modify: `lcm-sr-ui/src/utils/img2imgSourceStore.js`
- Test: `lcm-sr-ui/src/utils/img2imgSourceStore.test.mjs`

- [ ] **Step 1: Write the failing chat-origin persistence test**

```js
// lcm-sr-ui/src/utils/img2imgSourceStore.test.mjs
it('persists a chat-origin source with provenance fields', async () => {
  const row = await saveSource({
    originType: 'chat',
    originMessageId: 'msg-123',
    blob: new Blob(['chat-bytes'], { type: 'image/png' }),
    mimeType: 'image/png',
    filename: 'chat_123.png',
    cacheKey: 'abc123',
    serverImageUrl: 'http://localhost:4200/storage/key123',
    defaultDenoiseStrength: 0.5,
  });

  expect(row.originType).toBe('chat');
  expect(row.originMessageId).toBe('msg-123');
  expect(row.cacheKey).toBe('abc123');
  expect(row.serverImageUrl).toContain('/storage/key123');
});
```

- [ ] **Step 2: Run test to verify it fails or is incomplete**

Run: `cd lcm-sr-ui && npx vitest run src/utils/img2imgSourceStore.test.mjs`

Expected: FAIL if provenance fields are not persisted correctly, or PASS if the store is already sufficient

- [ ] **Step 3: Wire chat-origin promotion through persisted source creation**

```js
// lcm-sr-ui/src/App.jsx
const persistChatInitImage = useCallback(async (selectedMsg) => {
  const candidateUrl = selectedMsg?.serverImageUrl || selectedMsg?.imageUrl;
  if (!candidateUrl) return;

  let blob = null;
  if (selectedMsg?.meta?.cacheKey) {
    const cached = await getImageFromCache(selectedMsg.params);
    if (cached?.blob) blob = cached.blob;
  }
  if (!blob) {
    const res = await fetch(candidateUrl);
    if (!res.ok) throw new Error(`Failed to fetch init image: ${res.status}`);
    blob = await res.blob();
  }

  const filename = `chat_${selectedMsg.id}.png`;
  const row = await saveSource({
    originType: 'chat',
    originMessageId: selectedMsg.id,
    blob,
    mimeType: blob.type || 'image/png',
    filename,
    cacheKey: selectedMsg.meta?.cacheKey || null,
    serverImageUrl: selectedMsg.serverImageUrl || null,
    defaultDenoiseStrength: DEFAULT_IMG2IMG_DENOISE_STRENGTH,
  });

  setActiveSourceId(row.id);
  const file = new File([blob], filename, { type: row.mimeType });
  const objectUrl = URL.createObjectURL(blob);
  setInitImage({
    sourceId: row.id,
    originType: 'chat',
    file,
    objectUrl,
    filename,
    cacheKey: row.cacheKey,
    serverImageUrl: row.serverImageUrl,
  });
}, [getImageFromCache]);
```

- [ ] **Step 4: Run final verification**

Run:

```bash
cd lcm-sr-ui && npx vitest run \
  src/utils/img2imgSourceStore.test.mjs \
  src/hooks/useGenerationParams.test.jsx \
  src/components/options/OptionsPanel.test.jsx
```

Expected: PASS

Manual verification:

```bash
# 1. Start the dev server
cd lcm-sr-ui && yarn dev

# 2. Upload an init image, reload the page, confirm the Init Image panel remains.
# 3. Promote a chat image to init image, reload the page, confirm the Init Image panel remains.
# 4. Clear the init image, reload, confirm the panel is gone.
```

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/App.jsx \
        lcm-sr-ui/src/utils/img2imgSourceStore.js \
        lcm-sr-ui/src/utils/img2imgSourceStore.test.mjs
git commit -m "Persist chat-origin img2img sources"
```

---

## Spec Coverage Check

- Persist active img2img source across reloads and later visits: Task 1, Task 3, Task 4
- Support both chat-origin and uploaded source images: Task 3, Task 4
- Restore Init Image preview UI on startup: Task 3
- Persist per-source `defaultDenoiseStrength`: Task 1
- Global fallback default denoise strength of `0.5`: Task 1, Task 2

No spec gaps found.
