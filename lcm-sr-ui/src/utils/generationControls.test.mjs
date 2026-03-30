import { describe, expect, it } from 'vitest';

import {
  CUSTOM_NEGATIVE_PROMPT_ID,
  applyModeControlDefaultsToDraft,
  buildGenerateWsParams,
  getSchedulerOptions,
  resolveNegativePromptTemplateId,
} from './generationControls.js';

describe('generationControls', () => {
  it('applyModeControlDefaultsToDraft uses mode negative prompt and scheduler defaults', () => {
    const next = applyModeControlDefaultsToDraft(
      {
        negativePrompt: 'old negative',
        schedulerId: 'old_scheduler',
      },
      {
        negative_prompt_templates: {
          safe_photo: 'blurry, watermark',
        },
        default_negative_prompt_template: 'safe_photo',
        allowed_scheduler_ids: ['euler', 'dpmpp_2m'],
        default_scheduler_id: 'euler',
      }
    );

    expect(next.negativePrompt).toBe('blurry, watermark');
    expect(next.schedulerId).toBe('euler');
  });

  it('resolveNegativePromptTemplateId returns custom sentinel for custom prompt text', () => {
    const templateId = resolveNegativePromptTemplateId(
      {
        negative_prompt_templates: {
          base: 'blurry, watermark',
        },
        allow_custom_negative_prompt: true,
      },
      'extra fingers, blurry'
    );

    expect(templateId).toBe(CUSTOM_NEGATIVE_PROMPT_ID);
  });

  it('getSchedulerOptions only returns server-provided allowed schedulers', () => {
    const options = getSchedulerOptions({
      allowed_scheduler_ids: ['euler', 'dpmpp_2m'],
    });

    expect(options).toEqual([
      { value: 'euler', label: 'euler' },
      { value: 'dpmpp_2m', label: 'dpmpp_2m' },
    ]);
  });

  it('buildGenerateWsParams serializes negative prompt and scheduler id', () => {
    const params = buildGenerateWsParams({
      prompt: 'a castle',
      negativePrompt: 'blurry, watermark',
      schedulerId: 'euler',
      size: '512x512',
      steps: 8,
      cfg: 3,
      seed: 123,
      superres: false,
      superresLevel: 0,
      initImageRef: null,
      denoiseStrength: 0.75,
    });

    expect(params.negative_prompt).toBe('blurry, watermark');
    expect(params.scheduler_id).toBe('euler');
    expect(params.superres_magnitude).toBe(1);
  });
});
