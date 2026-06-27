// @vitest-environment jsdom

import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useDropIngest } from './useDropIngest';

describe('useDropIngest', () => {
  beforeEach(() => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('arms the img2img init image with the dropped file', async () => {
    const addMessage = vi.fn();
    const setSelectedMsgId = vi.fn();
    const setUploadFile = vi.fn();
    const onInitImageSelect = vi.fn();

    const { result } = renderHook(() =>
      useDropIngest({ addMessage, setSelectedMsgId, setUploadFile, onInitImageSelect })
    );

    const file = new File([new Uint8Array([1, 2, 3])], 'drop.png', { type: 'image/png' });

    await act(async () => {
      await result.current.ingestFiles([file]);
    });

    // Image still imports into chat...
    expect(addMessage).toHaveBeenCalledTimes(1);
    // ...and the same file is promoted to the active img2img init image.
    expect(onInitImageSelect).toHaveBeenCalledTimes(1);
    expect(onInitImageSelect).toHaveBeenCalledWith(file);
  });
});
