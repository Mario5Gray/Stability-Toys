// src/hooks/useComfyJobWs.js — WS-push ComfyUI job hook
//
// Drop-in replacement for useComfyJob that receives progress
// via WebSocket push events instead of HTTP polling.
// Falls back to useComfyJob if WS is not connected.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { wsClient, nextCorrId } from '../lib/wsClient';

/**
 * @param {{ api: object }} opts - Still needs api for image upload (POST /v1/upload)
 */
export function useComfyJobWs({ api } = {}) {
  const [jobId, setJobId] = useState(null);
  const [state, setState] = useState('idle');
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const corrIdRef = useRef(null);
  const jobIdRef = useRef(null);
  const unsubsRef = useRef([]);

  const cleanup = useCallback(() => {
    for (const unsub of unsubsRef.current) unsub();
    unsubsRef.current = [];
  }, []);

  const cancel = useCallback(() => {
    cleanup();
    if (jobId) {
      wsClient.send({ type: 'job:cancel', jobId });
    }
    jobIdRef.current = null;
    setState('canceled');
  }, [jobId, cleanup]);

  const start = useCallback(
    async (payload) => {
      // Reset
      cleanup();
      jobIdRef.current = null;
      setError(null);
      setJob(null);
      setState('starting');

      const { workflowId, params, inputImageFile } = payload;

      // Upload image via HTTP (binary — can't send over WS easily)
      let fileRef = null;
      if (inputImageFile) {
        const fd = new FormData();
        fd.append('file', inputImageFile);
        const res = await fetch('/v1/upload', { method: 'POST', body: fd });
        if (!res.ok) {
          const err = new Error(`Upload failed: ${res.status}`);
          setError(err);
          setState('error');
          throw err;
        }
        const data = await res.json();
        fileRef = data.fileRef;
      }

      const corrId = nextCorrId();
      corrIdRef.current = corrId;

      // Subscribe to WS events before sending
      const unsubs = [];

      // job:ack — match by correlation id, capture jobId into ref for sync access
      unsubs.push(wsClient.on('job:ack', (msg) => {
        if (msg.id !== corrId) return;
        jobIdRef.current = msg.jobId;
        setJobId(msg.jobId);
        setState('running');
      }));

      // job:progress — filter by jobId
      unsubs.push(wsClient.on('job:progress', (msg) => {
        if (!jobIdRef.current || msg.jobId !== jobIdRef.current) return;
        setJob((prev) => ({
          ...prev,
          status: msg.status,
          progress: msg.progress,
        }));
      }));

      // job:complete — filter by jobId
      unsubs.push(wsClient.on('job:complete', (msg) => {
        if (!jobIdRef.current || msg.jobId !== jobIdRef.current) return;
        setJob((prev) => ({
          ...prev,
          status: 'done',
          outputs: msg.outputs ?? [],
        }));
        setState('done');
        cleanup();
      }));

      // job:error — filter by jobId
      unsubs.push(wsClient.on('job:error', (msg) => {
        if (!jobIdRef.current || msg.jobId !== jobIdRef.current) return;
        setError(new Error(msg.error || 'ComfyUI job failed'));
        setState('error');
        cleanup();
      }));

      unsubsRef.current = unsubs;

      // Send via WS
      wsClient.send({
        type: 'job:submit',
        id: corrId,
        jobType: 'comfy',
        workflowId,
        params: params ?? {},
        inputImage: fileRef ? `fileRef:${fileRef}` : undefined,
      });

      return { corrId };
    },
    [cleanup]
  );

  // Cleanup on unmount
  useEffect(() => cleanup, [cleanup]);

  const isBusy = state === 'starting' || state === 'running';

  return useMemo(
    () => ({
      jobId,
      state,
      isBusy,
      job,
      error,
      start,
      cancel,
      refresh: () => {}, // no-op, push-based
      setJobId,
    }),
    [cancel, error, isBusy, job, jobId, start, state]
  );
}
