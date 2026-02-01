// src/lib/jobLedger.js — Append-Only IndexedDB Log

const DB_NAME = 'lcm-job-ledger';
const DB_VERSION = 1;
const STORE_NAME = 'entries';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('sessionId', 'sessionId', { unique: false });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('source', 'source', { unique: false });
        store.createIndex('parentId', 'parentId', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, mode = 'readonly') {
  return db.transaction(STORE_NAME, mode).objectStore(STORE_NAME);
}

function reqToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

class JobLedger {
  constructor() {
    this._dbPromise = null;
  }

  _db() {
    if (!this._dbPromise) this._dbPromise = openDB();
    return this._dbPromise;
  }

  async append(entry) {
    const db = await this._db();
    const store = tx(db, 'readwrite');
    await reqToPromise(store.put(entry));
  }

  async query({ source, sessionId, since, limit = 100 } = {}) {
    const db = await this._db();
    const store = tx(db);

    if (sessionId) {
      const idx = store.index('sessionId');
      const results = await reqToPromise(idx.getAll(sessionId));
      return since ? results.filter((e) => e.timestamp >= since).slice(0, limit) : results.slice(0, limit);
    }
    if (source) {
      const idx = store.index('source');
      const results = await reqToPromise(idx.getAll(source));
      return since ? results.filter((e) => e.timestamp >= since).slice(0, limit) : results.slice(0, limit);
    }
    // All entries, optionally filtered by time
    const all = await reqToPromise(store.getAll());
    const filtered = since ? all.filter((e) => e.timestamp >= since) : all;
    return filtered.slice(0, limit);
  }

  async getBySession(sessionId) {
    return this.query({ sessionId, limit: Infinity });
  }

  async getChain(id) {
    const db = await this._db();
    const store = tx(db);
    const chain = [];
    let current = id;
    while (current) {
      const entry = await reqToPromise(store.get(current));
      if (!entry) break;
      chain.unshift(entry);
      current = entry.parentId || null;
    }
    return chain;
  }

  async clear() {
    const db = await this._db();
    const store = tx(db, 'readwrite');
    await reqToPromise(store.clear());
  }
}

export const jobLedger = new JobLedger();
