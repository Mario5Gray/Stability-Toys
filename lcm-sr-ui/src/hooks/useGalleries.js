// src/hooks/useGalleries.js
import { useState, useCallback, useRef } from 'react';
import { uuidv4 } from '@/utils/uuid';

const LS_GALLERIES_KEY = 'lcm-galleries';
const LS_ACTIVE_KEY = 'lcm-active-gallery';
const DB_NAME = 'lcm-galleries';
const DB_VERSION = 2;
const STORE_NAME = 'gallery_items';
const ADVISOR_STORE = 'advisor_states';

export const TRASH_GALLERY_ID = '__trash__';

function openGalleryDb() {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('[useGalleries] IndexedDB is not available'));
  }
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('galleryId', 'galleryId', { unique: false });
        store.createIndex('cacheKey', 'cacheKey', { unique: false });
      }
      if (!db.objectStoreNames.contains(ADVISOR_STORE)) {
        db.createObjectStore(ADVISOR_STORE, { keyPath: 'gallery_id' });
      }
    };
  });
}

function promisifyRequest(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function loadGalleriesFromStorage() {
  try {
    const raw = localStorage.getItem(LS_GALLERIES_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function loadActiveFromStorage() {
  return localStorage.getItem(LS_ACTIVE_KEY) || null;
}

export function useGalleries() {
  const [galleries, setGalleries] = useState(() => loadGalleriesFromStorage());
  const [activeGalleryId, setActiveGalleryIdState] = useState(() => loadActiveFromStorage());
  const [galleryRevisions, setGalleryRevisions] = useState({});
  const dbRef = useRef(null);

  const getDb = useCallback(() => {
    if (!dbRef.current) dbRef.current = openGalleryDb();
    return dbRef.current;          // returns a Promise<IDBDatabase>
  }, []);

  const setActiveGalleryId = useCallback((id) => {
    setActiveGalleryIdState(id);
    if (id === null) {
      localStorage.removeItem(LS_ACTIVE_KEY);
    } else {
      localStorage.setItem(LS_ACTIVE_KEY, id);
    }
  }, []);

  const createGallery = useCallback((name) => {
    const truncated = String(name || '').slice(0, 16);
    const id = `gal_${uuidv4()}`;
    const entry = { id, name: truncated, createdAt: Date.now() };
    setGalleries((prev) => {
      const next = [...prev, entry];
      localStorage.setItem(LS_GALLERIES_KEY, JSON.stringify(next));
      return next;
    });
    setActiveGalleryId(id);
  }, [setActiveGalleryId]);

  const bumpGalleryRevision = useCallback((galleryId) => {
    if (!galleryId) return;
    setGalleryRevisions((prev) => ({
      ...prev,
      [galleryId]: (prev[galleryId] || 0) + 1,
    }));
  }, []);

  const addToGallery = useCallback(async (cacheKey, { serverImageUrl, params, galleryId, _addedAt }) => {
    if (!cacheKey || !galleryId) return;
    try {
      const db = await getDb();

      // Duplicate check — readonly tx (completes immediately)
      const roTx = db.transaction(STORE_NAME, 'readonly');
      const existing = await promisifyRequest(roTx.objectStore(STORE_NAME).index('cacheKey').getAll(cacheKey));
      if (existing.some((row) => row.galleryId === galleryId)) return;

      // Insert — separate readwrite tx
      const rwTx = db.transaction(STORE_NAME, 'readwrite');
      const row = {
        id: uuidv4(),
        galleryId,
        cacheKey,
        serverImageUrl: serverImageUrl ?? null,
        params: params ?? {},
        addedAt: _addedAt ?? Date.now(),
      };
      await promisifyRequest(rwTx.objectStore(STORE_NAME).put(row));
      bumpGalleryRevision(galleryId);
    } catch (err) {
      console.warn('[useGalleries] addToGallery failed:', err);
    }
  }, [getDb, bumpGalleryRevision]);

  const removeFromGallery = useCallback(async (galleryId, cacheKey) => {
    if (!galleryId || !cacheKey) return;
    try {
      const db = await getDb();
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const rows = await promisifyRequest(store.index('galleryId').getAll(galleryId));
      const matches = rows.filter((row) => row.cacheKey === cacheKey);
      await Promise.all(matches.map((row) => promisifyRequest(store.delete(row.id))));
      if (matches.length > 0) {
        bumpGalleryRevision(galleryId);
      }
    } catch (err) {
      console.warn('[useGalleries] removeFromGallery failed:', err);
    }
  }, [getDb, bumpGalleryRevision]);

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
      const galleriesList = JSON.parse(localStorage.getItem(LS_GALLERIES_KEY) || '[]');
      const knownIds = new Set(galleriesList.map((g) => g.id));
      const activeId = localStorage.getItem(LS_ACTIVE_KEY);
      const fallback = (activeId && knownIds.has(activeId)) ? activeId : galleriesList[0]?.id ?? null;
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

  const getGalleryImages = useCallback(async (galleryId) => {
    if (!galleryId) return [];
    try {
      const db = await getDb();
      const tx = db.transaction(STORE_NAME, 'readonly');
      const rows = await promisifyRequest(tx.objectStore(STORE_NAME).index('galleryId').getAll(galleryId));
      return rows.slice().sort((a, b) => b.addedAt - a.addedAt);
    } catch (err) {
      console.warn('[useGalleries] getGalleryImages failed:', err);
      return [];
    }
  }, [getDb]);

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
}
