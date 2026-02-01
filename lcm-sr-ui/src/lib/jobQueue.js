// src/lib/jobQueue.js — Singleton Priority Queue

export const PRIORITY = Object.freeze({
  URGENT: 0,
  NORMAL: 1,
  BATCH: 2,
  BACKGROUND: 3,
});

let _nextId = 1;

class JobQueue extends EventTarget {
  constructor({ concurrency = 1 } = {}) {
    super();
    this._concurrency = concurrency;
    this._pending = [];    // sorted: priority ASC, enqueueTime ASC
    this._running = new Map(); // id → { job, controller }
    this._snapshot = null; // memoized for useSyncExternalStore
  }

  // ---- public API ----

  enqueue({ priority = PRIORITY.NORMAL, runner, payload, source = '', meta = {} } = {}) {
    const id = String(_nextId++);
    const job = Object.freeze({
      id,
      priority,
      runner,
      payload: Object.freeze({ ...payload }),
      source,
      meta: Object.freeze({ ...meta }),
      enqueuedAt: Date.now(),
    });

    // Insert sorted: priority ASC, then enqueuedAt ASC (FIFO within lane)
    let idx = this._pending.length;
    for (let i = 0; i < this._pending.length; i++) {
      if (
        priority < this._pending[i].priority ||
        (priority === this._pending[i].priority && job.enqueuedAt < this._pending[i].enqueuedAt)
      ) {
        idx = i;
        break;
      }
    }
    this._pending.splice(idx, 0, job);
    this._invalidate();
    this._emit('enqueue', { job });
    this._flush();
    return id;
  }

  cancel(id) {
    // Cancel from pending
    const pendingIdx = this._pending.findIndex((j) => j.id === id);
    if (pendingIdx !== -1) {
      const [job] = this._pending.splice(pendingIdx, 1);
      this._invalidate();
      this._emit('cancel', { job });
      return true;
    }
    // Cancel running
    const entry = this._running.get(id);
    if (entry) {
      entry.controller.abort();
      // removal happens in _runJob's finally
      return true;
    }
    return false;
  }

  cancelAll() {
    const cancelled = [...this._pending];
    this._pending.length = 0;
    for (const job of cancelled) {
      this._emit('cancel', { job });
    }
    for (const [, entry] of this._running) {
      entry.controller.abort();
    }
    this._invalidate();
  }

  // ---- getters ----

  get pending() { return this._pending.length; }
  get running() { return this._running.size; }
  get depth() { return this._pending.length + this._running.size; }

  /** Memoized snapshot for useSyncExternalStore */
  get state() {
    if (!this._snapshot) {
      this._snapshot = Object.freeze({
        pending: this._pending.length,
        running: this._running.size,
        depth: this._pending.length + this._running.size,
        jobs: Object.freeze(this._pending.map((j) => ({
          id: j.id,
          priority: j.priority,
          source: j.source,
          enqueuedAt: j.enqueuedAt,
        }))),
      });
    }
    return this._snapshot;
  }

  // ---- subscribe (for useSyncExternalStore) ----

  subscribe(callback) {
    const handler = () => callback();
    for (const evt of ['enqueue', 'start', 'complete', 'error', 'cancel', 'drain']) {
      this.addEventListener(evt, handler);
    }
    return () => {
      for (const evt of ['enqueue', 'start', 'complete', 'error', 'cancel', 'drain']) {
        this.removeEventListener(evt, handler);
      }
    };
  }

  getSnapshot() {
    return this.state;
  }

  // ---- internals ----

  _invalidate() {
    this._snapshot = null;
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  _flush() {
    while (this._running.size < this._concurrency && this._pending.length > 0) {
      const job = this._pending.shift();
      this._runJob(job);
    }
    this._invalidate();
  }

  async _runJob(job) {
    const controller = new AbortController();
    this._running.set(job.id, { job, controller });
    this._invalidate();
    this._emit('start', { job });

    try {
      const result = await job.runner(job.payload, controller.signal);
      this._running.delete(job.id);
      this._invalidate();
      this._emit('complete', { job, result });

      // Fire-and-forget ledger write
      try {
        const { jobLedger } = await import('./jobLedger.js');
        jobLedger.append({
          id: job.id,
          sessionId: job.meta.sessionId ?? null,
          timestamp: Date.now(),
          priority: job.priority,
          source: job.source,
          request: job.payload,
          result: result ?? null,
          parentId: job.meta.parentId ?? null,
          status: 'complete',
          error: null,
        }).catch(() => {});
      } catch { /* ledger unavailable */ }
    } catch (err) {
      this._running.delete(job.id);
      this._invalidate();

      if (controller.signal.aborted) {
        this._emit('cancel', { job, error: err });
      } else {
        this._emit('error', { job, error: err });

        // Ledger: record error
        try {
          const { jobLedger } = await import('./jobLedger.js');
          jobLedger.append({
            id: job.id,
            sessionId: job.meta.sessionId ?? null,
            timestamp: Date.now(),
            priority: job.priority,
            source: job.source,
            request: job.payload,
            result: null,
            parentId: job.meta.parentId ?? null,
            status: 'error',
            error: err?.message ?? String(err),
          }).catch(() => {});
        } catch { /* ledger unavailable */ }
      }
    } finally {
      if (this._pending.length > 0 || this._running.size === 0) {
        this._flush();
      }
      if (this.depth === 0) {
        this._emit('drain', {});
      }
    }
  }
}

export const jobQueue = new JobQueue();
