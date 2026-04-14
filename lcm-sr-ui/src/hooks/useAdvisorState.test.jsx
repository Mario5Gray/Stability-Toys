// @vitest-environment jsdom
import 'fake-indexeddb/auto';
import { IDBFactory } from 'fake-indexeddb';
import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { useAdvisorState } from './useAdvisorState';

beforeEach(() => {
  localStorage.clear();
  globalThis.indexedDB = new IDBFactory();
});

it('persists and reloads advisor state by gallery_id', async () => {
  const { result, rerender } = renderHook(({ galleryId }) => useAdvisorState(galleryId), {
    initialProps: { galleryId: 'gal_1' },
  });

  await act(async () => {
    await result.current.saveState({
      gallery_id: 'gal_1',
      digest_text: 'digest',
      advice_text: 'advice',
      status: 'fresh',
    });
  });

  rerender({ galleryId: 'gal_1' });
  await act(async () => { await result.current.reload(); });

  expect(result.current.state.digest_text).toBe('digest');
  expect(result.current.state.advice_text).toBe('advice');
});

it('saveState injects gallery_id from hook when omitted', async () => {
  const { result } = renderHook(({ galleryId }) => useAdvisorState(galleryId), {
    initialProps: { galleryId: 'gal_1' },
  });

  await act(async () => {
    await result.current.saveState({
      digest_text: 'digest',
      advice_text: 'advice',
      status: 'fresh',
    });
  });

  expect(result.current.state.gallery_id).toBe('gal_1');
});

it('saveState rejects mismatched gallery_id', async () => {
  const { result } = renderHook(({ galleryId }) => useAdvisorState(galleryId), {
    initialProps: { galleryId: 'gal_1' },
  });

  await expect(
    result.current.saveState({
      gallery_id: 'gal_2',
      digest_text: 'digest',
    }),
  ).rejects.toThrow('gallery_id mismatch');
});
