import { describe, expect, it, vi } from 'vitest';
import { generateViaWsWithRetry, isRetryableGenerateError } from './generateRunnerWs';

describe('generateViaWsWithRetry', () => {
  it.each([
    "fileRef 'abc123' not found or expired",
    "Invalid size '1024x1024'",
    "Unknown scheduler_id 'dpmpp_2m'",
    "scheduler_id 'dpmpp_2m' is not allowed for the active mode",
  ])('classifies deterministic backend failures as non-retryable: %s', (message) => {
    expect(isRetryableGenerateError(new Error(message))).toBe(false);
  });

  it.each(['Generate timed out (no response)', 'WebSocket disconnected during generation', 'queue full'])(
    'classifies transient backend failures as retryable: %s',
    (message) => {
      expect(isRetryableGenerateError(new Error(message))).toBe(true);
    }
  );

  it('does not retry deterministic validation failures', async () => {
    const mockGenerateViaWs = vi.fn().mockRejectedValue(new Error("scheduler_id 'dpmpp_2m' is not allowed for the active mode"));

    await expect(
      generateViaWsWithRetry({ prompt: 'portrait' }, undefined, {
        generateViaWs: mockGenerateViaWs,
      })
    ).rejects.toThrow(/not allowed/);

    expect(mockGenerateViaWs).toHaveBeenCalledTimes(1);
  });

  it('retries transient queue full errors', async () => {
    vi.useFakeTimers();
    const mockGenerateViaWs = vi
      .fn()
      .mockRejectedValueOnce(new Error('queue full'))
      .mockResolvedValueOnce({ imageUrl: 'ok' });

    const promise = generateViaWsWithRetry({ prompt: 'portrait' }, undefined, {
      generateViaWs: mockGenerateViaWs,
    });

    await vi.runAllTimersAsync();

    await expect(promise).resolves.toEqual({ imageUrl: 'ok' });
    expect(mockGenerateViaWs).toHaveBeenCalledTimes(2);
  });
});
