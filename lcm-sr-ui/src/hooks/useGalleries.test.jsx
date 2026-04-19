// @vitest-environment jsdom
import 'fake-indexeddb/auto';
import { IDBFactory } from 'fake-indexeddb';
import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { useGalleries } from './useGalleries';

beforeEach(() => {
  localStorage.clear();
  // Replace the global IndexedDB with a fresh factory so each test gets a
  // clean database state without needing to close/delete the previous one.
  globalThis.indexedDB = new IDBFactory();
});

describe('useGalleries — localStorage', () => {
  it('starts with no galleries and no active gallery', () => {
    const { result } = renderHook(() => useGalleries());
    expect(result.current.galleries).toEqual([]);
    expect(result.current.activeGalleryId).toBeNull();
  });

  it('createGallery adds a gallery with name truncated to 16 chars and auto-selects it', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => {
      result.current.createGallery('My Very Long Gallery Name');
    });
    expect(result.current.galleries).toHaveLength(1);
    expect(result.current.galleries[0].name.length).toBeLessThanOrEqual(16);
    expect(result.current.activeGalleryId).toBe(result.current.galleries[0].id);
  });

  it('createGallery persists the gallery list to localStorage', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => {
      result.current.createGallery('Nature');
    });
    const stored = JSON.parse(localStorage.getItem('lcm-galleries'));
    expect(stored).toHaveLength(1);
    expect(stored[0].name).toBe('Nature');
  });

  it('setActiveGalleryId persists to localStorage and updates state', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => {
      result.current.createGallery('Alpha');
      result.current.createGallery('Beta');
    });
    const alphaId = result.current.galleries.find(g => g.name === 'Alpha')?.id;
    await act(async () => {
      result.current.setActiveGalleryId(alphaId);
    });
    expect(result.current.activeGalleryId).toBe(alphaId);
    expect(localStorage.getItem('lcm-active-gallery')).toBe(alphaId);
  });

  it('setActiveGalleryId(null) clears to null', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => {
      result.current.createGallery('Test');
      result.current.setActiveGalleryId(null);
    });
    expect(result.current.activeGalleryId).toBeNull();
    expect(localStorage.getItem('lcm-active-gallery')).toBeNull();
  });
});

describe('useGalleries — IndexedDB addToGallery / getGalleryImages', () => {
  it('addToGallery inserts a row and getGalleryImages returns it', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Photos'); });
    const galleryId = result.current.activeGalleryId;
    await act(async () => {
      await result.current.addToGallery('key_abc', {
        serverImageUrl: 'http://example.com/img.png',
        params: { prompt: 'cat', seed: 42 },
        galleryId,
      });
    });
    let items;
    await act(async () => { items = await result.current.getGalleryImages(galleryId); });
    expect(items).toHaveLength(1);
    expect(items[0].cacheKey).toBe('key_abc');
    expect(items[0].serverImageUrl).toBe('http://example.com/img.png');
    expect(items[0].params.prompt).toBe('cat');
  });

  it('addToGallery is a no-op for duplicate (galleryId, cacheKey)', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Dupes'); });
    const galleryId = result.current.activeGalleryId;
    await act(async () => {
      await result.current.addToGallery('key_dup', { serverImageUrl: null, params: {}, galleryId });
      await result.current.addToGallery('key_dup', { serverImageUrl: null, params: {}, galleryId });
    });
    let items;
    await act(async () => { items = await result.current.getGalleryImages(galleryId); });
    expect(items).toHaveLength(1);
  });

  it('getGalleryImages returns items newest-first (addedAt DESC)', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Ordered'); });
    const galleryId = result.current.activeGalleryId;
    const t0 = Date.now();
    await act(async () => {
      await result.current.addToGallery('key_first',  { serverImageUrl: null, params: {}, galleryId, _addedAt: t0 });
      await result.current.addToGallery('key_second', { serverImageUrl: null, params: {}, galleryId, _addedAt: t0 + 1000 });
    });
    let items;
    await act(async () => { items = await result.current.getGalleryImages(galleryId); });
    expect(items[0].cacheKey).toBe('key_second');
    expect(items[1].cacheKey).toBe('key_first');
  });

  it('same image can belong to two different galleries', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => {
      result.current.createGallery('G1');
      result.current.createGallery('G2');
    });
    const g1Id = result.current.galleries.find(g => g.name === 'G1')?.id;
    const g2Id = result.current.galleries.find(g => g.name === 'G2')?.id;
    await act(async () => {
      await result.current.addToGallery('shared_key', { serverImageUrl: null, params: {}, galleryId: g1Id });
      await result.current.addToGallery('shared_key', { serverImageUrl: null, params: {}, galleryId: g2Id });
    });
    let g1Items, g2Items;
    await act(async () => {
      g1Items = await result.current.getGalleryImages(g1Id);
      g2Items = await result.current.getGalleryImages(g2Id);
    });
    expect(g1Items).toHaveLength(1);
    expect(g2Items).toHaveLength(1);
    expect(g1Items[0].id).not.toBe(g2Items[0].id);
  });

  it('removeFromGallery deletes a row by galleryId and cacheKey', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Advisor'); });
    const galleryId = result.current.activeGalleryId;
    await act(async () => {
      await result.current.addToGallery('key_a', { serverImageUrl: null, params: {}, galleryId });
      await result.current.removeFromGallery(galleryId, 'key_a');
    });
    let items;
    await act(async () => { items = await result.current.getGalleryImages(galleryId); });
    expect(items).toEqual([]);
  });

  it('addToGallery bumps the gallery revision', async () => {
    const { result } = renderHook(() => useGalleries());
    await act(async () => { result.current.createGallery('Advisor'); });
    const galleryId = result.current.activeGalleryId;
    const before = result.current.getGalleryRevision(galleryId);
    await act(async () => {
      await result.current.addToGallery('key_a', { serverImageUrl: null, params: {}, galleryId });
    });
    expect(result.current.getGalleryRevision(galleryId)).toBeGreaterThan(before);
  });
});

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
