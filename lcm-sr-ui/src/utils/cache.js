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
const DB_VERSION = 1;
const STORE_NAME = "images";

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;

      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "key" });
        store.createIndex("createdAt", "createdAt", { unique: false });
        store.createIndex("accessedAt", "accessedAt", { unique: false });
        store.createIndex("size", "size", { unique: false });
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

  const withStore = async (mode, callback) => {
    const db = await getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, mode);
      const store = tx.objectStore(STORE_NAME);

      let result;
      try {
        result = callback(store, tx);
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
  const getRaw = async (key) => {
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    return promisify(store.get(key));
  };

  // Helper: determine if a write is a "hydration" (empty -> non-empty)
  const isHydration = (prevEntry, nextBlob) => {
    const prevSize = prevEntry?.blob?.size ?? 0;
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
        const tx = db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);

        const entry = await promisify(store.get(key));

        if (entry) {
          entry.accessedAt = Date.now();
          store.put(entry);
        }

        return entry || null;
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
        const prev = await getRaw(key);

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

        await withStore("readwrite", (store) => {
          store.put(entry);
        });

        emit("set", { key, size: entry.size, metadata: entry.metadata });

        if (isHydration(prev, entry.blob)) {
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
        const tx = db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const count = await promisify(store.count(key));
        return count > 0;
      } catch (err) {
        console.warn("[Cache] has failed:", err);
        return false;
      }
    },

    async delete(key) {
      try {
        await withStore("readwrite", (store) => store.delete(key));
        emit("delete", { key });
        return true;
      } catch (err) {
        console.warn("[Cache] delete failed:", err);
        return false;
      }
    },

    async clear() {
      try {
        await withStore("readwrite", (store) => store.clear());
        emit("clear", {});
      } catch (err) {
        console.warn("[Cache] clear failed:", err);
      }
    },

    async size() {
      try {
        const db = await getDb();
        const tx = db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        return await promisify(store.count());
      } catch (err) {
        console.warn("[Cache] size failed:", err);
        return 0;
      }
    },

    async totalBytes() {
      try {
        const db = await getDb();
        const tx = db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);

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
        const tx = db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        const index = store.index("accessedAt");

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
              store.delete(c.primaryKey);
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