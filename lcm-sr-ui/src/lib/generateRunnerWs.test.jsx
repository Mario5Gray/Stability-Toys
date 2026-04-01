import { describe, expect, it, vi } from 'vitest';
import { generateViaWsWithRetry } from './generateRunnerWs';

describe('generateViaWsWithRetry', () => {
  it('does not retry deterministic validation failures', async () => {
    const mockGenerateViaWs = vi
      .fn()
      .mockRejectedValueOnce(new Error("scheduler_id 'dpmpp_2m' is not allowed for the active mode"));

    await expect(
      generateViaWsWithRetry(
        { prompt: 'portrait' },
        undefined,
        { generateViaWs: mockGenerateViaWs }
      )
    ).rejects.toThrow(/not allowed/);

    expect(mockGenerateViaWs).toHaveBeenCalledTimes(1);
  });
});
