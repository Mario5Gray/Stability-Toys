// @vitest-environment jsdom

import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useGenerationParams } from './useGenerationParams';

describe('useGenerationParams', () => {
  it('applies mode-driven negative prompt and scheduler defaults to the draft state', () => {
    const { result } = renderHook(() =>
      useGenerationParams(null, null, vi.fn(), null)
    );

    act(() => {
      result.current.applyModeControlDefaults({
        negative_prompt_templates: {
          clean: 'blurry, low quality',
        },
        default_negative_prompt_template: 'clean',
        allowed_scheduler_ids: ['euler', 'ddim'],
        default_scheduler_id: 'ddim',
      });
    });

    expect(result.current.draft.negativePrompt).toBe('blurry, low quality');
    expect(result.current.draft.schedulerId).toBe('ddim');
    expect(result.current.effective.negativePrompt).toBe('blurry, low quality');
    expect(result.current.effective.schedulerId).toBe('ddim');
  });

  it('lets the draft state override mode defaults for negative prompt and scheduler', () => {
    const { result } = renderHook(() =>
      useGenerationParams(null, null, vi.fn(), null)
    );

    act(() => {
      result.current.applyModeControlDefaults({
        negative_prompt_templates: {
          clean: 'blurry, low quality',
        },
        default_negative_prompt_template: 'clean',
        allowed_scheduler_ids: ['euler', 'ddim'],
        default_scheduler_id: 'ddim',
      });
    });

    act(() => {
      result.current.setNegativePrompt('washed out, flat lighting');
      result.current.setSchedulerId('euler');
    });

    expect(result.current.draft.negativePrompt).toBe('washed out, flat lighting');
    expect(result.current.draft.schedulerId).toBe('euler');
    expect(result.current.effective.negativePrompt).toBe('washed out, flat lighting');
    expect(result.current.effective.schedulerId).toBe('euler');
  });
});
