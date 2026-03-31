// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  fetchBlobFromCandidates,
  getChatInitImageSuppressionKey,
  shouldPersistSelectedChatInitImage,
} from './App';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('App img2img source promotion helpers', () => {
  it('does not repersist when the active chat source matches the selected message origin', () => {
    const activeInitImage = {
      originType: 'chat',
      originMessageId: 'msg-123',
    };
    const selectedImage = {
      kind: 'image',
      id: 'msg-123',
    };

    expect(shouldPersistSelectedChatInitImage(activeInitImage, selectedImage)).toBe(false);
  });

  it('does not repersist while an upload init image is active', () => {
    const activeInitImage = {
      originType: 'upload',
      originMessageId: null,
    };
    const selectedImage = {
      kind: 'image',
      id: 'msg-456',
    };

    expect(shouldPersistSelectedChatInitImage(activeInitImage, selectedImage)).toBe(false);
  });

  it('does not repersist immediately after clearing the same chat image', () => {
    const activeInitImage = null;
    const selectedImage = {
      kind: 'image',
      id: 'msg-456',
    };

    expect(
      shouldPersistSelectedChatInitImage(activeInitImage, selectedImage, 'msg-456')
    ).toBe(false);
  });

  it('derives restored chat suppression from originMessageId, not source id', () => {
    const restoredSource = {
      originType: 'chat',
      id: 'persisted-row-id',
      originMessageId: 'msg-789',
    };

    expect(getChatInitImageSuppressionKey(restoredSource)).toBe('msg-789');
  });

  it('tries later chat init image candidates when the first fetch fails', async () => {
    const firstResponse = new Error('stale first candidate');
    const blob = new Blob(['chat-bytes'], { type: 'image/png' });
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockRejectedValueOnce(firstResponse)
      .mockResolvedValueOnce({
        ok: true,
        blob: async () => blob,
      });

    const result = await fetchBlobFromCandidates([
      'http://example.test/stale.png',
      'http://example.test/fresh.png',
    ]);

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(fetchSpy.mock.calls[0][0]).toBe('http://example.test/stale.png');
    expect(fetchSpy.mock.calls[1][0]).toBe('http://example.test/fresh.png');
    expect(result.resolvedUrl).toBe('http://example.test/fresh.png');
    expect(await result.blob.text()).toBe('chat-bytes');
  });
});
