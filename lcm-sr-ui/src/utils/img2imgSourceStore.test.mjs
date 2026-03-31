// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearActiveSource,
  getSource,
  getActiveSourceId,
  loadActiveSource,
  saveSource,
  setActiveSourceId,
} from "./img2imgSourceStore.js";

describe("img2imgSourceStore", () => {
  let originalIndexedDB;

  beforeEach(() => {
    setActiveSourceId(null);
    originalIndexedDB = globalThis.indexedDB;
  });

  afterEach(() => {
    globalThis.indexedDB = originalIndexedDB;
  });

  it("persists and restores an uploaded source", async () => {
    const blob = new Blob(["upload-bytes"], { type: "image/png" });
    const saved = await saveSource({
      originType: "upload",
      blob,
      mimeType: "image/png",
      filename: "init.png",
      cacheKey: "cache-key",
      serverImageUrl: "http://example.test/image.png",
      defaultDenoiseStrength: 0.5,
    });

    setActiveSourceId(saved.id);

    const restored = await loadActiveSource();
    expect(restored.id).toBe(saved.id);
    expect(restored.originType).toBe("upload");
    expect(restored.filename).toBe("init.png");
    expect(restored.cacheKey).toBe("cache-key");
    expect(restored.serverImageUrl).toBe("http://example.test/image.png");
    expect(restored.defaultDenoiseStrength).toBe(0.5);
    expect(await restored.blob.text()).toBe("upload-bytes");
    expect(getActiveSourceId()).toBe(saved.id);
  });

  it("persists a chat-origin source with provenance fields", async () => {
    const blob = new Blob(["chat-bytes"], { type: "image/png" });
    const saved = await saveSource({
      originType: "chat",
      originMessageId: "msg-123",
      blob,
      mimeType: "image/png",
      filename: "chat_123.png",
      cacheKey: "abc123",
      serverImageUrl: "http://localhost:4200/storage/key123",
      defaultDenoiseStrength: 0.5,
    });

    setActiveSourceId(saved.id);

    const restored = await loadActiveSource();
    expect(restored.id).toBe(saved.id);
    expect(restored.originType).toBe("chat");
    expect(restored.originMessageId).toBe("msg-123");
    expect(restored.cacheKey).toBe("abc123");
    expect(restored.serverImageUrl).toBe("http://localhost:4200/storage/key123");
    expect(restored.defaultDenoiseStrength).toBe(0.5);
    expect(await restored.blob.text()).toBe("chat-bytes");
  });

  it("clears the active pointer when the source row is missing", async () => {
    setActiveSourceId("missing-source");

    const restored = await loadActiveSource();

    expect(restored).toBeNull();
    expect(getActiveSourceId()).toBeNull();
  });

  it("removes the active pointer and deletes the active source", async () => {
    const saved = await saveSource({
      originType: "chat",
      originMessageId: "message-123",
      blob: new Blob(["x"], { type: "image/png" }),
      mimeType: "image/png",
      filename: "x.png",
      defaultDenoiseStrength: 0.7,
    });

    setActiveSourceId(saved.id);

    await clearActiveSource();

    expect(getActiveSourceId()).toBeNull();
    expect(await loadActiveSource()).toBeNull();
    expect(await getSource(saved.id)).toBeNull();
  });

  it("recovers when indexedDB.open fails once and later succeeds", async () => {
    const stores = new Map();
    let openAttempts = 0;

    function getEntries(name) {
      if (!stores.has(name)) {
        stores.set(name, new Map());
      }
      return stores.get(name);
    }

    const db = {
      objectStoreNames: {
        contains(name) {
          return stores.has(name);
        },
      },
      createObjectStore(name) {
        getEntries(name);
        return {
          createIndex() {},
        };
      },
      transaction(name) {
        const entries = getEntries(name);
        const tx = {
          oncomplete: null,
          onerror: null,
          objectStore() {
            return {
              get(key) {
                const request = {
                  result: entries.get(key) || null,
                  onsuccess: null,
                  onerror: null,
                };
                queueMicrotask(() => request.onsuccess?.());
                return request;
              },
              put(record) {
                entries.set(record.id, record);
                queueMicrotask(() => tx.oncomplete?.());
              },
              delete(key) {
                entries.delete(key);
                queueMicrotask(() => tx.oncomplete?.());
              },
              createIndex() {},
            };
          },
        };
        return tx;
      },
    };

    globalThis.indexedDB = {
      open: vi.fn(() => {
        openAttempts += 1;
        const request = {
          error: null,
          result: null,
          onerror: null,
          onsuccess: null,
          onupgradeneeded: null,
        };

        queueMicrotask(() => {
          if (openAttempts === 1) {
            request.error = new Error("indexedDB open failed");
            request.onerror?.();
            return;
          }

          if (request.onupgradeneeded) {
            request.onupgradeneeded({ target: { result: db } });
          }

          request.result = db;
          request.onsuccess?.({ target: { result: db } });
        });

        return request;
      }),
    };

    await expect(
      saveSource({
        originType: "upload",
        blob: new Blob(["first"], { type: "image/png" }),
        filename: "first.png",
      })
    ).rejects.toThrow("indexedDB open failed");

    const saved = await saveSource({
      originType: "upload",
      blob: new Blob(["second"], { type: "image/png" }),
      filename: "second.png",
    });

    expect(saved.filename).toBe("second.png");
    expect((await getSource(saved.id)).filename).toBe("second.png");
    expect(openAttempts).toBe(2);
  });
});
