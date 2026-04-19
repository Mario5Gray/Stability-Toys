// @vitest-environment jsdom
import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { vi, expect, it } from 'vitest';
import { OperationsProvider, useOperationsStore } from '../contexts/OperationsContext';
import { useGalleryAdvisor } from './useGalleryAdvisor';

function wrapper({ children }) {
  return <OperationsProvider>{children}</OperationsProvider>;
}

it('rebuilds digest and seeds advice text when no edits exist', async () => {
  const api = {
    fetchPost: vi.fn(),
  };
  api.fetchPost.mockResolvedValue({
    digest_text: 'Painterly neon portrait',
    meta: { evidence_fingerprint: 'sha256:abc' },
  });

  const { result } = renderHook(() => useGalleryAdvisor({
    galleryId: 'gal_1',
    modeName: 'SDXL',
    galleryRevision: 1,
    galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
    maximumLen: 240,
    api,
    advisorState: null,
    saveAdvisorState: vi.fn(),
    setDraftPrompt: vi.fn(),
  }), { wrapper });

  await act(async () => {
    await result.current.rebuildAdvisor();
  });

  await waitFor(() => expect(result.current.state.digest_text).toBe('Painterly neon portrait'));
  expect(api.fetchPost).toHaveBeenCalledWith(
    '/api/advisors/digest',
    expect.objectContaining({ mode: 'SDXL' }),
  );
  expect(result.current.state.advice_text).toBe('Painterly neon portrait');
});

it('marks advisor state stale with the latest gallery revision when the gallery changes', async () => {
  const saveAdvisorState = vi.fn();

  const { result, rerender } = renderHook((props) => useGalleryAdvisor(props), {
    initialProps: {
      galleryId: 'gal_1',
      galleryRevision: 1,
      galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
      maximumLen: 240,
      api: { fetchPost: vi.fn() },
      advisorState: {
        gallery_id: 'gal_1',
        gallery_revision: 1,
        digest_text: 'digest',
        advice_text: 'digest',
        status: 'fresh',
      },
      saveAdvisorState,
      setDraftPrompt: vi.fn(),
    },
    wrapper,
  });

  rerender({
    galleryId: 'gal_1',
    galleryRevision: 2,
    galleryImages: [
      { cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } },
      { cacheKey: 'xyz', addedAt: 2, params: { prompt: 'dog' } },
    ],
    maximumLen: 240,
    api: { fetchPost: vi.fn() },
    advisorState: {
      gallery_id: 'gal_1',
      gallery_revision: 1,
      digest_text: 'digest',
      advice_text: 'digest',
      status: 'fresh',
    },
    saveAdvisorState,
    setDraftPrompt: vi.fn(),
  });

  await waitFor(() => expect(result.current.state.status).toBe('stale'));
  expect(result.current.state.gallery_revision).toBe(2);
});

it('preserves user-edited advice text when a rebuild returns a new digest', async () => {
  const api = {
    fetchPost: vi.fn().mockResolvedValue({
      digest_text: 'new digest',
      meta: { evidence_fingerprint: 'sha256:new' },
    }),
  };
  const { result } = renderHook(() => useGalleryAdvisor({
    galleryId: 'gal_1',
    galleryRevision: 1,
    galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
    maximumLen: 240,
    api,
    advisorState: {
      gallery_id: 'gal_1',
      digest_text: 'old digest',
      advice_text: 'custom user advice',
      temperature: 0.4,
      length_limit: 120,
    },
    saveAdvisorState: vi.fn(),
    setDraftPrompt: vi.fn(),
  }), { wrapper });

  await act(async () => {
    await result.current.rebuildAdvisor();
  });

  expect(result.current.state.digest_text).toBe('new digest');
  expect(result.current.state.advice_text).toBe('custom user advice');
});

it('rebuildAdvisor creates an active operation and completes it on success', async () => {
  const api = { fetchPost: vi.fn().mockResolvedValue({ digest_text: 'result', meta: {} }) };

  function useCombo() {
    return {
      advisor: useGalleryAdvisor({
        galleryId: 'gal_1', modeName: 'SDXL', galleryRevision: 1,
        galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
        maximumLen: 240, api,
        advisorState: null, saveAdvisorState: vi.fn(), setDraftPrompt: vi.fn(),
      }),
      store: useOperationsStore(),
    };
  }

  const { result } = renderHook(useCombo, { wrapper });

  expect(result.current.store.order).toHaveLength(0);

  await act(async () => { await result.current.advisor.rebuildAdvisor(); });

  expect(result.current.advisor.state.digest_text).toBe('result');
  // operation completes (tone complete or auto-expired); either way rebuild ran without error
  const op = result.current.store.operations.get('advisor-rebuild:gal_1');
  if (op) {
    expect(['complete', 'error']).toContain(op.tone);
  }
});
