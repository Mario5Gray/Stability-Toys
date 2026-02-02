// src/lib/generateRunnerWs.js — Generate images via WebSocket instead of HTTP POST
//
// Factory that returns an async runner with the same return shape as api.generate(),
// but uses the WS job:submit protocol.

import { wsClient, nextCorrId } from './wsClient';

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

    let cleanup;
    let jobId = null;

    // Handle abort
    const onAbort = () => {
      cleanup?.();
      const err = new Error('Aborted');
      err.name = 'AbortError';
      reject(err);
    };

    if (signal) {
      if (signal.aborted) { onAbort(); return; }
      signal.addEventListener('abort', onAbort, { once: true });
    }

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
        cleanup?.();
        reject(new Error(msg.error || 'Generation failed'));
        return;
      }

      if (msg.type === 'job:complete') {
        cleanup?.();
        const out = msg.outputs?.[0] || {};
        const meta = msg.meta || {};
        resolve({
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
      wsClient.removeEventListener('message', handler);
      if (signal) signal.removeEventListener('abort', onAbort);
    };
  });
}
