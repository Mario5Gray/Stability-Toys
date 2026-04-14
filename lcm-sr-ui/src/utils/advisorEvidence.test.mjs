import { describe, expect, it } from 'vitest';
import { buildAdvisorEvidence } from './advisorEvidence';

describe('buildAdvisorEvidence', () => {
  it('normalizes gallery rows into versioned evidence', () => {
    const evidence = buildAdvisorEvidence('gal_1', [
      {
        cacheKey: 'abc',
        addedAt: 1712790000,
        params: {
          prompt: 'cat',
          negativePrompt: 'blur',
          size: '512x512',
          steps: 8,
          cfg: 2.5,
          schedulerId: 'euler',
          seed: 123,
        },
      },
    ]);

    expect(evidence).toEqual({
      version: 1,
      gallery_id: 'gal_1',
      items: [
        {
          cache_key: 'abc',
          added_at: 1712790000,
          prompt: 'cat',
          negative_prompt: 'blur',
          size: '512x512',
          steps: 8,
          cfg: 2.5,
          scheduler_id: 'euler',
          seed: 123,
          superres_level: null,
          metadata: {},
        },
      ],
    });
  });
});
