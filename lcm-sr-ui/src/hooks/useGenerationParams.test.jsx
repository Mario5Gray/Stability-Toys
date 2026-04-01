// @vitest-environment jsdom

import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useGenerationParams } from './useGenerationParams';

describe('useGenerationParams', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

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

  it('falls back to source default denoise when no selected or draft value is set', () => {
    const { result } = renderHook(() =>
      useGenerationParams(null, null, vi.fn(), null, null, 0.5)
    );

    expect(result.current.draft.denoiseStrength).toBe(0.5);
    expect(result.current.effective.denoiseStrength).toBe(0.5);
  });

  it('prefers selected image denoise over source default', () => {
    const { result } = renderHook(() =>
      useGenerationParams(
        {
          prompt: 'selected',
          size: '512x512',
          steps: 8,
          cfg: 1,
          seedMode: 'fixed',
          seed: 123,
          superresLevel: 0,
          denoiseStrength: 0.82,
        },
        vi.fn(),
        vi.fn(),
        'msg-1',
        null,
        0.5
      )
    );

    expect(result.current.effective.denoiseStrength).toBe(0.82);
  });

  it('preserves a user-edited draft denoise across source default and selection changes', () => {
    const { result, rerender } = renderHook(
      ({ selectedParams, sourceDefault }) =>
        useGenerationParams(
          selectedParams,
          vi.fn(),
          vi.fn(),
          selectedParams ? 'msg-1' : null,
          null,
          sourceDefault
        ),
      {
        initialProps: {
          selectedParams: null,
          sourceDefault: 0.5,
        },
      }
    );

    act(() => {
      result.current.setDenoiseStrength(0.33);
    });

    rerender({
      selectedParams: {
        prompt: 'selected',
        size: '512x512',
        steps: 8,
        cfg: 1,
        seedMode: 'fixed',
        seed: 123,
        superresLevel: 0,
        denoiseStrength: 0.91,
      },
      sourceDefault: 0.8,
    });

    rerender({
      selectedParams: null,
      sourceDefault: 0.8,
    });

    expect(result.current.draft.denoiseStrength).toBe(0.33);
    expect(result.current.effective.denoiseStrength).toBe(0.33);
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

  it('stages selected-image negative prompt edits without triggering regeneration', () => {
    vi.useFakeTimers();
    const runGenerate = vi.fn();
    const patchSelectedParams = vi.fn();
    const selectedParams = {
      prompt: 'portrait',
      negativePrompt: 'blurry',
      schedulerId: 'ddim',
      size: '512x512',
      steps: 8,
      cfg: 3,
      seed: 1234,
      superresLevel: 0,
      denoiseStrength: 0.75,
    };

    const { result } = renderHook(() =>
      useGenerationParams(
        selectedParams,
        patchSelectedParams,
        runGenerate,
        'msg-1'
      )
    );

    act(() => {
      result.current.setNegativePrompt('washed out');
    });

    expect(patchSelectedParams).toHaveBeenCalledWith({ negativePrompt: 'washed out' });
    act(() => {
      vi.runAllTimers();
    });

    expect(runGenerate).not.toHaveBeenCalled();
  });
});
