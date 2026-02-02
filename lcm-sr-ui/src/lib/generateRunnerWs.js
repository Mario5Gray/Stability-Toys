// src/lib/generateRunnerWs.js — Generate images via WebSocket instead of HTTP POST
//
// Factory that returns an async runner with the same return shape as api.generate(),
// but uses the WS job:submit protocol.

import { wsClient, nextCorrId } from './wsClient';

const GENERATE_TIMEOUT_MS = 120_000; // 2 minutes

/**
 * Create a WS-based generate runner.
 * @param {object} payload - Generation parameters (prompt, size, steps, cfg, seed, superres, superresLevel)
 * @param {AbortSignal} [signal] - Optional abort signal
 * @returns {Promise<{imageUrl, serverImageUrl, serverImageKey, metadata}>}
 */
export function generateViaWs(payload, signal) {
  return new Promise((resolve, reject) => {
    if (!wsClient.connected) {
      return reject(new Error('WebSocket not connected'));
    }

    const corrId = nextCorrId();
    let settled = false;
    let cleanup;
    let jobId = null;

    const settle = (fn, value) => {
      if (settled) return;
      settled = true;
      cleanup?.();
      fn(value);
    };

    // Send job:submit
    wsClient.send({
      id: corrId,
      type: 'job:submit',
      jobType: 'generate',
      params: {
        prompt: payload.prompt,
        size: payload.size,
        steps: payload.steps,
        cfg: payload.cfg,
        seed: payload.seed,
        superres: payload.superres,
        superres_magnitude: payload.superresLevel || 1,
      },
    });

    // Timeout — reject if server never responds
    const timer = setTimeout(() => {
      settle(reject, new Error('Generate timed out (no response)'));
    }, GENERATE_TIMEOUT_MS);

    // Handle abort
    const onAbort = () => {
      const err = new Error('Aborted');
      err.name = 'AbortError';
      settle(reject, err);
    };

    if (signal) {
      if (signal.aborted) { onAbort(); return; }
      signal.addEventListener('abort', onAbort, { once: true });
    }

    // Reject on WS disconnect (server can't deliver results on a dead socket)
    const onDisconnect = (e) => {
      if (e.detail?.state === 'disconnected') {
        settle(reject, new Error('WebSocket disconnected during generation'));
      }
    };
    wsClient.addEventListener('statechange', onDisconnect);

    // Listen for matching response.
    // job:ack carries our corrId + a jobId; subsequent messages use jobId only.
    const handler = (e) => {
      const msg = e.detail;
      if (!msg) return;

      // Match ack by correlation id, capture jobId
      if (msg.type === 'job:ack' && msg.id === corrId) {
        jobId = msg.jobId;
        return;
      }

      // After ack, match by jobId
      if (!jobId || msg.jobId !== jobId) return;

      if (msg.type === 'job:error') {
        settle(reject, new Error(msg.error || 'Generation failed'));
        return;
      }

      if (msg.type === 'job:complete') {
        const out = msg.outputs?.[0] || {};
        const meta = msg.meta || {};
        settle(resolve, {
          imageUrl: out.url,
          serverImageUrl: out.url,
          serverImageKey: out.key,
          metadata: {
            seed: meta.seed,
            backend: meta.backend,
            superres: meta.sr,
            apiBase: 'ws',
          },
        });
        return;
      }
    };

    wsClient.addEventListener('message', handler);

    cleanup = () => {
      clearTimeout(timer);
      wsClient.removeEventListener('message', handler);
      wsClient.removeEventListener('statechange', onDisconnect);
      if (signal) signal.removeEventListener('abort', onAbort);
    };
  });
}
