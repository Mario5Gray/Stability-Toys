// src/lib/jobQueue.js — Singleton Priority Queue

import { emitJobEvent } from '../utils/otelTelemetry.js';

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
    this._telemetry = new Map(); // id -> timing data
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
    this._telemetry.set(job.id, {
      enqueuedAt: job.enqueuedAt,
      startedAt: null,
    });
    emitJobEvent('queue.enqueue.job', {
      'job.id': job.id,
      'job.source': job.source,
      'job.priority': job.priority,
      'job.enqueued_at_ms': job.enqueuedAt,
      'job.queue.depth': this.depth,
    });
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
      emitJobEvent('queue.cancel.job', {
        'job.id': job.id,
        'job.source': job.source,
        'job.priority': job.priority,
        'job.enqueued_at_ms': job.enqueuedAt,
        'job.queue.depth': this.depth,
        'job.status': 'canceled',
      });
      this._telemetry.delete(job.id);
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
      emitJobEvent('queue.cancel.job', {
        'job.id': job.id,
        'job.source': job.source,
        'job.priority': job.priority,
        'job.enqueued_at_ms': job.enqueuedAt,
        'job.queue.depth': this.depth,
        'job.status': 'canceled',
      });
      this._telemetry.delete(job.id);
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
    const telem = this._telemetry.get(job.id);
    const startedAt = Date.now();
    if (telem) telem.startedAt = startedAt;
    emitJobEvent('queue.start.job', {
      'job.id': job.id,
      'job.source': job.source,
      'job.priority': job.priority,
      'job.enqueued_at_ms': job.enqueuedAt,
      'job.started_at_ms': startedAt,
      'job.queue.latency_ms': Math.max(0, startedAt - job.enqueuedAt),
      'job.queue.depth': this.depth,
    });

    try { 
      const f0 = performance.now();
      console.log("Sending JOB " + job.id);  
      const result = await job.runner(job.payload, controller.signal)
      const f1 = performance.now();
      console.log("await job runner time ", (f1 - f0).toFixed(1), "ms");

      this._running.delete(job.id);
      this._invalidate();
      this._emit('complete', { job, result });
      const finishedAt = Date.now();
      const telem2 = this._telemetry.get(job.id);
      const started = telem2?.startedAt ?? finishedAt;
      emitJobEvent('queue.complete.job', {
        'job.id': job.id,
        'job.source': job.source,
        'job.priority': job.priority,
        'job.enqueued_at_ms': job.enqueuedAt,
        'job.started_at_ms': started,
        'job.finished_at_ms': finishedAt,
        'job.queue.latency_ms': Math.max(0, started - job.enqueuedAt),
        'job.run_time_ms': Math.max(0, finishedAt - started),
        'job.total_time_ms': Math.max(0, finishedAt - job.enqueuedAt),
        'job.status': 'complete',
      });
      this._telemetry.delete(job.id);

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
      } catch { /* ledger unavailable */
        console.error("Ledger is not available!");
       }
    } catch (err) {
      console.error("Error in launching job " + job.id);
      this._running.delete(job.id);
      this._invalidate();

      if (controller.signal.aborted) {
        this._emit('cancel', { job, error: err });
        const finishedAt = Date.now();
        const telem2 = this._telemetry.get(job.id);
        const started = telem2?.startedAt ?? finishedAt;
        emitJobEvent('queue.cancel.job', {
          'job.id': job.id,
          'job.source': job.source,
          'job.priority': job.priority,
          'job.enqueued_at_ms': job.enqueuedAt,
          'job.started_at_ms': started,
          'job.finished_at_ms': finishedAt,
          'job.queue.latency_ms': Math.max(0, started - job.enqueuedAt),
          'job.run_time_ms': Math.max(0, finishedAt - started),
          'job.total_time_ms': Math.max(0, finishedAt - job.enqueuedAt),
          'job.status': 'canceled',
        });
        this._telemetry.delete(job.id);
      } else {
        this._emit('error', { job, error: err });
        const finishedAt = Date.now();
        const telem2 = this._telemetry.get(job.id);
        const started = telem2?.startedAt ?? finishedAt;
        emitJobEvent('queue.error.job', {
          'job.id': job.id,
          'job.source': job.source,
          'job.priority': job.priority,
          'job.enqueued_at_ms': job.enqueuedAt,
          'job.started_at_ms': started,
          'job.finished_at_ms': finishedAt,
          'job.queue.latency_ms': Math.max(0, started - job.enqueuedAt),
          'job.run_time_ms': Math.max(0, finishedAt - started),
          'job.total_time_ms': Math.max(0, finishedAt - job.enqueuedAt),
          'job.status': 'error',
          'error.message': err?.message ?? String(err),
        });
        this._telemetry.delete(job.id);

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
