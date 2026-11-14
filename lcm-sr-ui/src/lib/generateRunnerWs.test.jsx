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

vi.mock('./wsClient', () => {
  const target = new EventTarget();
  target.connected = true;
  target.sent = [];
  target.send = (m) => target.sent.push(m);
  return { wsClient: target, nextCorrId: () => 'corr-1' };
});

describe('generateViaWs', () => {
  it('rejects on a pre-ack error frame matching the correlation id', async () => {
    const { wsClient } = await import('./wsClient');
    const { generateViaWs } = await import('./generateRunnerWs');

    const promise = generateViaWs({ prompt: 'fox' });
    expect(wsClient.sent[0]).toMatchObject({ id: 'corr-1', type: 'job:submit' });

    wsClient.dispatchEvent(
      new CustomEvent('message', { detail: { type: 'error', id: 'corr-1', error: "fileRef 'x' not found or expired" } })
    );

    await expect(promise).rejects.toThrow(/not found or expired/);
  });

  it('ignores error frames for other correlation ids', async () => {
    vi.useFakeTimers();
    const { wsClient } = await import('./wsClient');
    const { generateViaWs } = await import('./generateRunnerWs');

    const promise = generateViaWs({ prompt: 'fox' });
    const settled = vi.fn();
    promise.then(settled, settled);

    wsClient.dispatchEvent(new CustomEvent('message', { detail: { type: 'error', id: 'other', error: 'nope' } }));
    await vi.advanceTimersByTimeAsync(1000);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(120_000);
    await expect(promise).rejects.toThrow(/timed out/);
    vi.useRealTimers();
  });
});
