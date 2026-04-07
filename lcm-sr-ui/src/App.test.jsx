// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  buildSelectedSeedDeltaPayload,
  fetchBlobFromCandidates,
  getModeDefaultsSyncPlan,
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

  it('does not promote an auto-selected generated image into the init image', () => {
    const activeInitImage = null;
    const selectedImage = {
      kind: 'image',
      id: 'msg-999',
    };

    expect(
      shouldPersistSelectedChatInitImage(activeInitImage, selectedImage, null, 'msg-999')
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

  it('keeps the active init image and denoise strength attached when applying a seed delta', () => {
    const initImageFile = new File(['init'], 'init.png', { type: 'image/png' });
    const payload = buildSelectedSeedDeltaPayload(
      {
        prompt: 'portrait',
        negativePrompt: 'blurry',
        schedulerId: 'ddim',
        size: '512x512',
        steps: 8,
        cfg: 2.8,
        seed: 100,
        superresLevel: 0,
        denoiseStrength: 0.42,
      },
      'msg-123',
      10,
      initImageFile,
      0.42
    );

    expect(payload.seed).toBe(110);
    expect(payload.seedMode).toBe('fixed');
    expect(payload.targetMessageId).toBe('msg-123');
    expect(payload.initImageFile).toBe(initImageFile);
    expect(payload.denoiseStrength).toBe(0.42);
  });
});

describe('App mode default sync helpers', () => {
  it('builds a fallback default-mode sync plan when runtime active mode is unavailable', () => {
    const plan = getModeDefaultsSyncPlan(
      {
        config: {
          default_mode: 'cinematic',
          modes: {
            cinematic: {
              default_size: '1024x1024',
              default_negative_prompt_template: 'clean',
              negative_prompt_templates: {
                clean: 'blurry, low quality',
              },
              allowed_scheduler_ids: ['euler', 'ddim'],
              default_scheduler_id: 'ddim',
            },
          },
        },
        activeModeName: null,
        activeMode: null,
      },
      {
        size: '512x512',
        negativePrompt: '',
        schedulerId: null,
      }
    );

    expect(plan).toMatchObject({
      mode: {
        default_size: '1024x1024',
      },
      draftDefaults: {
        size: '1024x1024',
        negativePrompt: 'blurry, low quality',
        schedulerId: 'ddim',
      },
    });
  });

  it('keeps syncing while the draft still matches the last auto-applied defaults', () => {
    const plan = getModeDefaultsSyncPlan(
      {
        config: {
          default_mode: 'cinematic',
          modes: {
            cinematic: {
              default_size: '1024x1024',
              default_scheduler_id: 'ddim',
            },
            portrait: {
              default_size: '832x1216',
              default_scheduler_id: 'euler',
            },
          },
        },
        activeModeName: 'portrait',
        activeMode: {
          default_size: '832x1216',
          default_scheduler_id: 'euler',
        },
      },
      {
        size: '1024x1024',
        negativePrompt: '',
        schedulerId: 'ddim',
      },
      {
        size: '1024x1024',
        negativePrompt: '',
        schedulerId: 'ddim',
      }
    );

    expect(plan?.draftDefaults).toEqual({
      size: '832x1216',
      negativePrompt: '',
      schedulerId: 'euler',
    });
  });

  it('does not build a later sync plan after the user changes the draft away from auto-applied defaults', () => {
    const plan = getModeDefaultsSyncPlan(
      {
        config: {
          default_mode: 'cinematic',
          modes: {
            cinematic: {
              default_size: '1024x1024',
              default_scheduler_id: 'ddim',
            },
          },
        },
        activeModeName: null,
        activeMode: null,
      },
      {
        size: '768x768',
        negativePrompt: '',
        schedulerId: 'ddim',
      },
      {
        size: '1024x1024',
        negativePrompt: '',
        schedulerId: 'ddim',
      }
    );

    expect(plan).toBeNull();
  });
});
