// src/lib/comfyRunnerWs.js — WS-native ComfyUI runner (replaces polling)
//
// Drop-in replacement for comfyRunner.js.
// Instead of polling GET /v1/comfy/jobs/:id every 750ms, this:
//   1. Uploads input image via POST /v1/upload → fileRef
//   2. Sends job:submit via WS → receives job:ack
//   3. Listens for job:complete / job:error push events
//
// Compatible with jobQueue runner signature: async (payload, signal) => result

import { wsClient, nextCorrId } from './wsClient';

/**
 * Upload a File to /v1/upload and get a fileRef.
 * @param {File} file
 * @returns {Promise<string>} fileRef
 */
async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/v1/upload', { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  const data = await res.json();
  return data.fileRef;
}

/**
 * Creates a WS-native async runner for ComfyUI jobs.
 * Same signature as createComfyRunner() — compatible with jobQueue.enqueue({ runner }).
 *
 * @returns {(payload: object, signal: AbortSignal) => Promise<object>}
 */
export function createComfyRunnerWs() {
  return async function comfyRunnerWs(payload, signal) {
    const { workflowId, params, inputImageFile } = payload;

    // Upload image if present → fileRef
    let fileRef = null;
    if (inputImageFile) {
      fileRef = await uploadFile(inputImageFile);
    }

    const corrId = nextCorrId();

    // Send job:submit via WS
    wsClient.send({
      type: 'job:submit',
      id: corrId,
      jobType: 'comfy',
      workflowId,
      params: params ?? {},
      inputImage: fileRef ? `fileRef:${fileRef}` : undefined,
    });

    // Wait for job:ack to get jobId
    const ack = await wsClient.waitFor(
      (msg) => msg.type === 'job:ack' && msg.id === corrId,
      10000
    );
    const jobId = ack.jobId;

    // Wait for job:complete or job:error
    return new Promise((resolve, reject) => {
      let settled = false;

      const settle = (fn, value) => {
        if (settled) return;
        settled = true;
        cleanup();
        fn(value);
      };

      if (signal.aborted) {
        wsClient.send({ type: 'job:cancel', jobId });
        reject(new DOMException('Job cancelled', 'AbortError'));
        return;
      }

      const onAbort = () => {
        wsClient.send({ type: 'job:cancel', jobId });
        settle(reject, new DOMException('Job cancelled', 'AbortError'));
      };
      signal.addEventListener('abort', onAbort, { once: true });

      const onDisconnect = (e) => {
        if (e.detail?.state === 'disconnected') {
          settle(reject, new Error('WebSocket disconnected during ComfyUI job'));
        }
      };
      wsClient.addEventListener('statechange', onDisconnect);

      const unsubComplete = wsClient.on('job:complete', (msg) => {
        if (msg.jobId !== jobId) return;
        settle(resolve, { jobId, outputs: msg.outputs ?? [], raw: msg });
      });

      const unsubError = wsClient.on('job:error', (msg) => {
        if (msg.jobId !== jobId) return;
        settle(reject, new Error(msg.error || 'ComfyUI job failed'));
      });

      function cleanup() {
        signal.removeEventListener('abort', onAbort);
        wsClient.removeEventListener('statechange', onDisconnect);
        unsubComplete();
        unsubError();
      }
    });
  };
}
