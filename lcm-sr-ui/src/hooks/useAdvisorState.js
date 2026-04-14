import { useCallback, useEffect, useRef, useState } from 'react';

const DB_NAME = 'lcm-galleries';
const DB_VERSION = 2;
const ADVISOR_STORE = 'advisor_states';
const GALLERY_STORE = 'gallery_items';

function openAdvisorDb() {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('[useAdvisorState] IndexedDB is not available'));
  }
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(ADVISOR_STORE)) {
        db.createObjectStore(ADVISOR_STORE, { keyPath: 'gallery_id' });
      }
      if (!db.objectStoreNames.contains(GALLERY_STORE)) {
        const galleryStore = db.createObjectStore(GALLERY_STORE, { keyPath: 'id' });
        galleryStore.createIndex('galleryId', 'galleryId', { unique: false });
        galleryStore.createIndex('cacheKey', 'cacheKey', { unique: false });
      }
    };
  });
}

function requestToPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function useAdvisorState(galleryId) {
  const [state, setState] = useState(null);
  const dbRef = useRef(null);

  const getDb = useCallback(() => {
    if (!dbRef.current) dbRef.current = openAdvisorDb();
    return dbRef.current;
  }, []);

  const reload = useCallback(async () => {
    if (!galleryId) {
      setState(null);
      return null;
    }
    const db = await getDb();
    const tx = db.transaction(ADVISOR_STORE, 'readonly');
    const nextState = await requestToPromise(tx.objectStore(ADVISOR_STORE).get(galleryId));
    setState(nextState || null);
    return nextState || null;
  }, [galleryId, getDb]);

  const saveState = useCallback(async (nextState) => {
    const db = await getDb();
    const tx = db.transaction(ADVISOR_STORE, 'readwrite');
    await requestToPromise(tx.objectStore(ADVISOR_STORE).put(nextState));
    setState(nextState);
  }, [getDb]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { state, setState, saveState, reload };
}
