// src/lib/comfyRunner.js — Promise-Based ComfyUI Runner

/**
 * Creates an async runner function for ComfyUI jobs.
 * @param {object} api - comfyInvokerApi instance (startJob, getJob, cancelJob)
 * @param {number} [pollMs=750] - polling interval
 * @returns {(payload: object, signal: AbortSignal) => Promise<object>}
 */
export function createComfyRunner(api, pollMs = 750) {
  return async function comfyRunner(payload, signal) {
    const { jobId } = await api.startJob(payload, { signal });

    // Poll until done or aborted
    while (true) {
      if (signal.aborted) {
        api.cancelJob(jobId).catch(() => {});
        throw new DOMException('Job cancelled', 'AbortError');
      }

      await new Promise((r) => setTimeout(r, pollMs));

      if (signal.aborted) {
        api.cancelJob(jobId).catch(() => {});
        throw new DOMException('Job cancelled', 'AbortError');
      }

      const status = await api.getJob(jobId, { signal });

      if (status.status === 'done') {
        return { jobId, outputs: status.outputs ?? [], raw: status };
      }
      if (status.status === 'error') {
        throw new Error(status.error || 'ComfyUI job failed');
      }
    }
  };
}
