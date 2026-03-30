// @vitest-environment jsdom
import 'fake-indexeddb/auto';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useGalleries } from './useGalleries';

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.useRealTimers();
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
});
