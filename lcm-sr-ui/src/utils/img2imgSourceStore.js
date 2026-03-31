import { DEFAULT_IMG2IMG_DENOISE_STRENGTH } from "./constants.js";
import { uuidv4 } from "./uuid.js";

export const IMG2IMG_SOURCE_DB_NAME = "lcm-img2img";
export const IMG2IMG_SOURCE_STORE_NAME = "img2img_sources";
export const ACTIVE_IMG2IMG_SOURCE_KEY = "lcm-active-img2img-source";

const memoryState = {
  sources: new Map(),
  activeSourceId: null,
};

let dbPromise = null;

function hasIndexedDB() {
  return typeof indexedDB !== "undefined" && typeof indexedDB.open === "function";
}

function hasLocalStorage() {
  return typeof localStorage !== "undefined" && typeof localStorage.getItem === "function";
}

function normalizeDenoiseStrength(value) {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return numeric;
  }
  return DEFAULT_IMG2IMG_DENOISE_STRENGTH;
}

function openDatabase() {
  if (!hasIndexedDB()) {
    return Promise.resolve(null);
  }

  if (dbPromise) {
    return dbPromise;
  }

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(IMG2IMG_SOURCE_DB_NAME, 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(IMG2IMG_SOURCE_STORE_NAME)) {
        const store = db.createObjectStore(IMG2IMG_SOURCE_STORE_NAME, { keyPath: "id" });
        store.createIndex("originMessageId", "originMessageId", { unique: false });
        store.createIndex("originType", "originType", { unique: false });
        store.createIndex("updatedAt", "updatedAt", { unique: false });
      }
    };
  }).catch((error) => {
    dbPromise = null;
    throw error;
  });

  return dbPromise;
}

function toRecord(source, existing = null) {
  const now = Date.now();
  const blob = source.blob ?? existing?.blob ?? null;

  return {
    id: source.id || existing?.id || uuidv4(),
    originType: source.originType,
    originMessageId: source.originMessageId ?? existing?.originMessageId ?? null,
    blob,
    mimeType: source.mimeType || blob?.type || existing?.mimeType || "application/octet-stream",
    filename: source.filename || existing?.filename || "img2img-source",
    cacheKey: source.cacheKey ?? existing?.cacheKey ?? null,
    serverImageUrl: source.serverImageUrl ?? existing?.serverImageUrl ?? null,
    defaultDenoiseStrength: normalizeDenoiseStrength(
      source.defaultDenoiseStrength ?? existing?.defaultDenoiseStrength ?? DEFAULT_IMG2IMG_DENOISE_STRENGTH
    ),
    createdAt: source.createdAt ?? existing?.createdAt ?? now,
    updatedAt: now,
  };
}

async function getStoredSource(id) {
  if (!id) {
    return null;
  }

  if (!hasIndexedDB()) {
    return memoryState.sources.get(id) || null;
  }

  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IMG2IMG_SOURCE_STORE_NAME, "readonly");
    const store = tx.objectStore(IMG2IMG_SOURCE_STORE_NAME);
    const request = store.get(id);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

async function putStoredSource(record) {
  if (!hasIndexedDB()) {
    memoryState.sources.set(record.id, record);
    return record;
  }

  const db = await openDatabase();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(IMG2IMG_SOURCE_STORE_NAME, "readwrite");
    tx.objectStore(IMG2IMG_SOURCE_STORE_NAME).put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });

  return record;
}

async function deleteStoredSource(id) {
  if (!id) {
    return;
  }

  if (!hasIndexedDB()) {
    memoryState.sources.delete(id);
    return;
  }

  const db = await openDatabase();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(IMG2IMG_SOURCE_STORE_NAME, "readwrite");
    tx.objectStore(IMG2IMG_SOURCE_STORE_NAME).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function saveSource(source) {
  const existing = await getStoredSource(source?.id);
  const record = toRecord(source, existing);
  return putStoredSource(record);
}

export function setActiveSourceId(id) {
  if (!hasLocalStorage()) {
    memoryState.activeSourceId = id || null;
    return;
  }

  if (!id) {
    localStorage.removeItem(ACTIVE_IMG2IMG_SOURCE_KEY);
    return;
  }

  localStorage.setItem(ACTIVE_IMG2IMG_SOURCE_KEY, id);
}

export function getActiveSourceId() {
  if (!hasLocalStorage()) {
    return memoryState.activeSourceId;
  }

  return localStorage.getItem(ACTIVE_IMG2IMG_SOURCE_KEY);
}

export async function getSource(id) {
  return getStoredSource(id);
}

export async function loadActiveSource() {
  const id = getActiveSourceId();
  if (!id) {
    return null;
  }

  const source = await getStoredSource(id);
  if (!source) {
    setActiveSourceId(null);
    return null;
  }

  return source;
}

export async function deleteSource(id) {
  await deleteStoredSource(id);
}

export async function clearActiveSource() {
  const id = getActiveSourceId();
  setActiveSourceId(null);
  if (id) {
    await deleteStoredSource(id);
  }
}
