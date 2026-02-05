// src/utils/cache.js

/**
 * Cache abstraction layer - storage agnostic interface.
 * Implementations can use IndexedDB, memory, localStorage, etc.
 *
 * Added:
 *  - Optional EventTarget-based eventing (non-breaking)
 *  - emit("hydrated") etc. for background hydration workflows
 */

/* ============================================================================
 * CACHE KEY GENERATION
 * ========================================================================== */

export function generateCacheKey(params) {
  const { prompt, size, steps, cfg, seed, superresLevel = 0 } = params;

  const normalized = {
    p: String(prompt || "").trim().toLowerCase(),
    sz: String(size || "512x512"),
    st: Number(steps) || 0,
    cfg: Number(cfg) || 0,
    sd: Number(seed) || 0,
    sr: Number(superresLevel) || 0,
  };

  const str = JSON.stringify(normalized);
  return hashString(str);
}

function hashString(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/* ============================================================================
 * INDEXEDDB IMPLEMENTATION
 * ========================================================================== */

const DB_NAME = "lcm-image-cache";
const DB_VERSION = 2;
const META_STORE = "imageMeta";
const BLOB_STORE = "imageBlobs";
const LEGACY_STORE = "images";

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      const tx = event.target.transaction;

      let metaStore;
      if (!db.objectStoreNames.contains(META_STORE)) {
        metaStore = db.createObjectStore(META_STORE, { keyPath: "key" });
        metaStore.createIndex("createdAt", "createdAt", { unique: false });
        metaStore.createIndex("accessedAt", "accessedAt", { unique: false });
        metaStore.createIndex("size", "size", { unique: false });
      } else {
        metaStore = tx.objectStore(META_STORE);
      }

      let blobStore;
      if (!db.objectStoreNames.contains(BLOB_STORE)) {
        blobStore = db.createObjectStore(BLOB_STORE, { keyPath: "key" });
      } else {
        blobStore = tx.objectStore(BLOB_STORE);
      }

      if (db.objectStoreNames.contains(LEGACY_STORE)) {
        const legacy = tx.objectStore(LEGACY_STORE);
        legacy.openCursor().onsuccess = (e) => {
          const cursor = e.target.result;
          if (cursor) {
            const entry = cursor.value || {};
            const blob = entry.blob || new Blob([]);
            const size = entry.size ?? blob.size ?? 0;
            const meta = {
              key: entry.key,
              metadata: entry.metadata || {},
              createdAt: entry.createdAt || Date.now(),
              accessedAt: entry.accessedAt || Date.now(),
              size,
            };
            metaStore.put(meta);
            if (blob.size > 0) {
              blobStore.put({ key: entry.key, blob, size: blob.size });
            } else {
              blobStore.delete(entry.key);
            }
            cursor.continue();
          } else {
            db.deleteObjectStore(LEGACY_STORE);
          }
        };
      }
    };
  });
}

/**
 * Create an IndexedDB-backed cache.
 */
export function createIndexedDBCache(options = {}) {
  const { maxEntries = 500, maxBytes = 500 * 1024 * 1024 } = options;

  let dbPromise = null;

  // --- Eventing (optional, non-breaking) ---
  const events = new EventTarget();
  const emit = (type, detail) => {
    try {
      events.dispatchEvent(new CustomEvent(type, { detail }));
    } catch (e) {
      // CustomEvent can fail in some test envs; ignore
    }
  };

  const getDb = () => {
    if (!dbPromise) dbPromise = openDatabase();
    return dbPromise;
  };

  const withStores = async (mode, storeNames, callback) => {
    const db = await getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeNames, mode);
      const stores = storeNames.map((name) => tx.objectStore(name));

      let result;
      try {
        result = callback(stores, tx);
      } catch (err) {
        reject(err);
        return;
      }

      tx.oncomplete = () => resolve(result);
      tx.onerror = () => reject(tx.error);
    });
  };

  const promisify = (request) =>
    new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

  // Helper: fetch existing entry (readonly)
  const getMeta = async (key) => {
    const db = await getDb();
    const tx = db.transaction(META_STORE, "readonly");
    const store = tx.objectStore(META_STORE);
    return promisify(store.get(key));
  };

  // Helper: determine if a write is a "hydration" (empty -> non-empty)
  const isHydration = (prevEntry, nextBlob) => {
    const prevSize = prevEntry?.size ?? 0;
    const nextSize = nextBlob?.size ?? 0;
    return prevSize === 0 && nextSize > 0;
  };

  // Define api object first so methods can reference it safely
  const api = {
    // EventTarget passthroughs
    addEventListener: (...args) => events.addEventListener(...args),
    removeEventListener: (...args) => events.removeEventListener(...args),
    dispatchEvent: (...args) => events.dispatchEvent(...args),
    emit, // convenience

    async get(key) {
      try {
        const db = await getDb();
        const tx = db.transaction([META_STORE, BLOB_STORE], "readwrite");
        const metaStore = tx.objectStore(META_STORE);
        const blobStore = tx.objectStore(BLOB_STORE);

        const meta = await promisify(metaStore.get(key));
        const blobEntry = await promisify(blobStore.get(key));

        if (meta) {
          meta.accessedAt = Date.now();
          metaStore.put(meta);
        }

        if (!meta) return null;

        return {
          key,
          blob: blobEntry?.blob || new Blob([]),
          metadata: meta.metadata || {},
          createdAt: meta.createdAt,
          accessedAt: meta.accessedAt,
          size: meta.size ?? blobEntry?.size ?? blobEntry?.blob?.size ?? 0,
        };
      } catch (err) {
        console.warn("[Cache] get failed:", err);
        return null;
      }
    },

    /**
     * Store blob with metadata.
     * - Non-breaking: same signature.
     * - New behavior: emits events.
     * - New behavior: merges metadata with existing (shallow) when requested.
     *
     * Options (optional, non-breaking because 4th param is optional):
     *   { mergeMetadata: true }
     */
    async set(key, blob, metadata = {}, opts = {}) {
      const byteLen = blob?.size ?? null;
      console.log("[Cache] set() attempt", { key, byteLen, metadata });

      try {
        const prev = await getMeta(key);

        const mergeMetadata = !!opts.mergeMetadata;
        const nextMeta = mergeMetadata
          ? { ...(prev?.metadata || {}), ...(metadata || {}) }
          : (metadata || {});

        const entry = {
          key,
          metadata: nextMeta,
          createdAt: prev?.createdAt ?? Date.now(),
          accessedAt: Date.now(),
          size: (blob || new Blob([])).size,
        };

        await withStores("readwrite", [META_STORE, BLOB_STORE], ([metaStore, blobStore]) => {
          metaStore.put(entry);
          if (entry.size > 0) {
            blobStore.put({ key, blob: blob || new Blob([]), size: entry.size });
          } else {
            blobStore.delete(key);
          }
        });

        emit("set", { key, size: entry.size, metadata: entry.metadata });

        if (isHydration(prev, { size: entry.size })) {
          emit("hydrated", { key, size: entry.size, metadata: entry.metadata });
        }

        // Trigger eviction check (async, don't await)
        api._evictIfNeeded();
      } catch (err) {
        console.warn("[Cache] set failed:", err);
      }
    },

    /**
     * Convenience: store metadata-only pointer without bytes.
     * (Non-breaking add-on; you can keep doing cache.set(key, new Blob([]), meta))
     */
    async setMetaOnly(key, metadata = {}, opts = {}) {
      const meta = { ...(metadata || {}), __metaOnly: true };
      return api.set(key, new Blob([]), meta, { mergeMetadata: !!opts.mergeMetadata });
    },

    async has(key) {
      try {
        const db = await getDb();
        const tx = db.transaction(META_STORE, "readonly");
        const store = tx.objectStore(META_STORE);
        const count = await promisify(store.count(key));
        return count > 0;
      } catch (err) {
        console.warn("[Cache] has failed:", err);
        return false;
      }
    },

    async delete(key) {
      try {
        await withStores("readwrite", [META_STORE, BLOB_STORE], ([metaStore, blobStore]) => {
          metaStore.delete(key);
          blobStore.delete(key);
        });
        emit("delete", { key });
        return true;
      } catch (err) {
        console.warn("[Cache] delete failed:", err);
        return false;
      }
    },

    async clear() {
      try {
        await withStores("readwrite", [META_STORE, BLOB_STORE], ([metaStore, blobStore]) => {
          metaStore.clear();
          blobStore.clear();
        });
        emit("clear", {});
      } catch (err) {
        console.warn("[Cache] clear failed:", err);
      }
    },

    async size() {
      try {
        const db = await getDb();
        const tx = db.transaction(META_STORE, "readonly");
        const store = tx.objectStore(META_STORE);
        return await promisify(store.count());
      } catch (err) {
        console.warn("[Cache] size failed:", err);
        return 0;
      }
    },

    async totalBytes() {
      try {
        const db = await getDb();
        const tx = db.transaction(META_STORE, "readonly");
        const store = tx.objectStore(META_STORE);

        let total = 0;
        const cursor = store.openCursor();

        return new Promise((resolve, reject) => {
          cursor.onsuccess = (event) => {
            const c = event.target.result;
            if (c) {
              total += c.value.size || 0;
              c.continue();
            } else {
              resolve(total);
            }
          };
          cursor.onerror = () => reject(cursor.error);
        });
      } catch (err) {
        console.warn("[Cache] totalBytes failed:", err);
        return 0;
      }
    },

    async stats() {
      const [count, bytes] = await Promise.all([api.size(), api.totalBytes()]);
      return {
        entries: count,
        bytes,
        maxEntries,
        maxBytes,
        utilizationEntries: count / maxEntries,
        utilizationBytes: bytes / maxBytes,
      };
    },

    async _evictIfNeeded() {
      try {
        const [count, bytes] = await Promise.all([api.size(), api.totalBytes()]);
        const needsEviction = count > maxEntries || bytes > maxBytes;
        if (!needsEviction) return;

        const db = await getDb();
        const tx = db.transaction([META_STORE, BLOB_STORE], "readwrite");
        const metaStore = tx.objectStore(META_STORE);
        const blobStore = tx.objectStore(BLOB_STORE);
        const index = metaStore.index("accessedAt");

        const toDelete = Math.max(
          count - maxEntries + 10,
          Math.ceil((bytes - maxBytes) / (bytes / Math.max(count, 1))) + 5
        );

        let deleted = 0;
        const cursor = index.openCursor();

        return new Promise((resolve) => {
          cursor.onsuccess = (event) => {
            const c = event.target.result;
            if (c && deleted < toDelete) {
              metaStore.delete(c.primaryKey);
              blobStore.delete(c.primaryKey);
              deleted++;
              c.continue();
            } else {
              console.log(`[Cache] Evicted ${deleted} entries`);
              emit("evict", { deleted });
              resolve();
            }
          };
          cursor.onerror = () => resolve();
        });
      } catch (err) {
        console.warn("[Cache] eviction failed:", err);
      }
    },

    async close() {
      if (dbPromise) {
        const db = await dbPromise;
        db.close();
        dbPromise = null;
        emit("close", {});
      }
    },
  };

  return api;
}

/* ============================================================================
 * IN-MEMORY CACHE (Fallback / Testing)
 * ========================================================================== */

export function createMemoryCache(options = {}) {
  const { maxEntries = 100 } = options;
  const store = new Map();

  const events = new EventTarget();
  const emit = (type, detail) => {
    try {
      events.dispatchEvent(new CustomEvent(type, { detail }));
    } catch (e) {}
  };

  const api = {
    addEventListener: (...args) => events.addEventListener(...args),
    removeEventListener: (...args) => events.removeEventListener(...args),
    dispatchEvent: (...args) => events.dispatchEvent(...args),
    emit,

    async get(key) {
      const entry = store.get(key);
      if (entry) entry.accessedAt = Date.now();
      return entry || null;
    },

    async set(key, blob, metadata = {}, opts = {}) {
      const prev = store.get(key);
      const mergeMetadata = !!opts.mergeMetadata;
      const nextMeta = mergeMetadata
        ? { ...(prev?.metadata || {}), ...(metadata || {}) }
        : (metadata || {});

      const entry = {
        key,
        blob: blob || new Blob([]),
        metadata: nextMeta,
        createdAt: prev?.createdAt ?? Date.now(),
        accessedAt: Date.now(),
        size: (blob || new Blob([])).size,
      };

      const wasHydration = (prev?.blob?.size ?? 0) === 0 && entry.blob.size > 0;

      store.set(key, entry);
      emit("set", { key, size: entry.size, metadata: entry.metadata });
      if (wasHydration) emit("hydrated", { key, size: entry.size, metadata: entry.metadata });

      if (store.size > maxEntries) {
        const oldest = [...store.entries()].sort((a, b) => a[1].accessedAt - b[1].accessedAt)[0];
        if (oldest) store.delete(oldest[0]);
        emit("evict", { deleted: 1 });
      }
    },

    async setMetaOnly(key, metadata = {}, opts = {}) {
      const meta = { ...(metadata || {}), __metaOnly: true };
      return api.set(key, new Blob([]), meta, { mergeMetadata: !!opts.mergeMetadata });
    },

    async has(key) {
      return store.has(key);
    },

    async delete(key) {
      const ok = store.delete(key);
      if (ok) emit("delete", { key });
      return ok;
    },

    async clear() {
      store.clear();
      emit("clear", {});
    },

    async size() {
      return store.size;
    },

    async totalBytes() {
      let total = 0;
      for (const entry of store.values()) total += entry.size || 0;
      return total;
    },

    async stats() {
      const bytes = await api.totalBytes();
      return {
        entries: store.size,
        bytes,
        maxEntries,
        utilizationEntries: store.size / maxEntries,
      };
    },

    async close() {
      emit("close", {});
    },
  };

  return api;
}

/* ============================================================================
 * CACHE FACTORY
 * ========================================================================== */

export function createCache(options = {}) {
  if (typeof indexedDB !== "undefined") {
    try {
      return createIndexedDBCache(options);
    } catch (err) {
      console.warn("[Cache] IndexedDB unavailable, using memory cache:", err);
    }
  }
  return createMemoryCache(options);
}