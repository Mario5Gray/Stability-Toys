// src/hooks/useGalleries.js
import { useState, useCallback, useRef } from 'react';
import { uuidv4 } from '@/utils/uuid';

const LS_GALLERIES_KEY = 'lcm-galleries';
const LS_ACTIVE_KEY = 'lcm-active-gallery';
const DB_NAME = 'lcm-galleries';
const DB_VERSION = 1;
const STORE_NAME = 'gallery_items';

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
    } catch (err) {
      console.warn('[useGalleries] addToGallery failed:', err);
    }
  }, [getDb]);

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
    getGalleryImages,
  };
}
