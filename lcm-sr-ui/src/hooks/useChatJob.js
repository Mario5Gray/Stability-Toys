// src/hooks/useChatJob.js — WS correlation hook for jobType=chat
//
// Mirrors useComfyJobWs shape but supports multiple in-flight chat jobs via
// per-handle corrId subscriptions. Socket disconnect mid-stream fails all
// active handles with "Connection lost".

import { useCallback, useEffect, useRef } from 'react';
import { wsClient, nextCorrId } from '../lib/wsClient';

export function useChatJob() {
  // Map corrId -> { cleanup, onError } for active (pre-ack) handles
  // Map jobId  -> { cleanup, onError } for acked (streaming) handles
  const byCorr = useRef(new Map());
  const byJob  = useRef(new Map());

  // Fail all active streaming handles when socket drops
  useEffect(() => {
    const handler = (e) => {
      if (e.detail?.state !== 'disconnected') return;
      const handles = [
        ...Array.from(byCorr.current.values()),
        ...Array.from(byJob.current.values()),
      ];
      byCorr.current.clear();
      byJob.current.clear();
      for (const h of handles) {
        h.cleanup();
        h.onError?.('Connection lost');
      }
    };
    wsClient.addEventListener('statechange', handler);
    return () => wsClient.removeEventListener('statechange', handler);
  }, []);

  const start = useCallback(({ prompt, onAck, onDelta, onComplete, onError }) => {
    if (!wsClient.connected) {
      onError?.('Not connected');
      return { cancel: () => {} };
    }

    const corrId = nextCorrId();
    const unsubs = [];
    let jobId = null;
    let cancelRequested = false;
    let closed = false;
    let terminated = false;

    const emitError = (message) => {
      if (terminated) return;
      terminated = true;
      onError?.(message);
    };

    const emitComplete = (text) => {
      if (terminated) return;
      terminated = true;
      onComplete?.({ text });
    };

    const cleanup = () => {
      if (closed) return;
      closed = true;
      for (const u of unsubs) u();
      byCorr.current.delete(corrId);
      if (jobId) byJob.current.delete(jobId);
    };

    // job:ack — matched by correlation id
    unsubs.push(wsClient.on('job:ack', (msg) => {
      if (msg.id !== corrId) return;
      jobId = msg.jobId;
      byCorr.current.delete(corrId);
      if (cancelRequested) {
        wsClient.send({ type: 'job:cancel', jobId });
        cleanup();
        return;
      }
      byJob.current.set(jobId, { cleanup, onError });
      onAck?.({ jobId });
    }));

    // job:progress — matched by jobId once acked
    unsubs.push(wsClient.on('job:progress', (msg) => {
      if (!jobId || msg.jobId !== jobId) return;
      if (msg.delta) onDelta?.(msg.delta);
    }));

    // job:complete — matched by jobId
    unsubs.push(wsClient.on('job:complete', (msg) => {
      if (!jobId || msg.jobId !== jobId) return;
      cleanup();
      emitComplete(msg.outputs?.[0]?.text ?? '');
    }));

    // job:error — matched by jobId
    unsubs.push(wsClient.on('job:error', (msg) => {
      if (!jobId || msg.jobId !== jobId) return;
      cleanup();
      emitError(msg.error || 'Chat job failed');
    }));

    byCorr.current.set(corrId, { cleanup, onError: emitError });

    wsClient.send({
      type: 'job:submit',
      id: corrId,
      jobType: 'chat',
      params: { prompt, stream: true },
    });

    return {
      cancel: () => {
        if (terminated) return;
        cancelRequested = true;
        emitError('Cancelled');
        if (jobId) wsClient.send({ type: 'job:cancel', jobId });
        if (jobId) cleanup();
      },
    };
  }, []);

  return { start };
}
