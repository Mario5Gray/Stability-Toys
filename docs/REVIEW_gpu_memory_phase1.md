# Code Review: GPU Memory Management Phase 1

Covers changes in `backends/worker_pool.py` and `server/model_routes.py` from the
idle eviction, demand reload, and config-change reload sessions.

---

## Bugs / Correctness

### 1. `_last_activity` not updated on failed jobs

**Location:** `WorkerPool._worker_loop`

`_last_activity = time.monotonic()` sits after `job.execute(...)` inside the `try`
block. If execution throws, the timestamp is never updated. A user whose generations
are failing (OOM during inference, bad prompt, etc.) will have their model evicted
because the idle timer never resets — despite active traffic hitting the pool.

Should be moved to `finally` or placed before `job.execute(...)`.

---

### 2. Demand reload errors are opaque to the caller

**Location:** `WorkerPool._worker_loop`, demand reload block

If `_load_mode` fails during demand reload, the raw load exception (e.g.
`torch.cuda.OutOfMemoryError`) propagates directly onto the job's future. The caller
has no way to distinguish "my generation parameters were wrong" from "the server
couldn't reload the model after an idle eviction". Wrapping the load failure in a
descriptive exception (e.g. `ModelLoadError("demand reload failed: ...")`) would make
this diagnosable without reading server logs.

---

## Dead / Misleading Code

### 3. `self._lock` declared but never acquired

**Location:** `WorkerPool.__init__`

`self._lock = threading.Lock()` is initialised but never used in the class. The
current threading model is actually safe without it — all state mutations happen on
the worker thread via the queue. The unused lock creates a false expectation of
protection and will confuse anyone reading the code. Should either be used (guarding
`get_current_mode()` reads and `_last_activity` writes) or removed.

---

### 4. `get_current_mode()` is ambiguous after eviction

**Location:** `WorkerPool.get_current_mode`, `server/model_routes.py`

After idle eviction, `_current_mode` retains the mode name (intentionally, for demand
reload) but `_worker` is `None`. The `/api/models/status` endpoint reports
`current_mode: "sdxl-general"` with VRAM at 0 and no entries in the model registry.
The UI has no signal to distinguish "model loaded" from "model evicted, will reload on
next request".

A `get_model_status()` method or an `is_loaded: bool` field on the status response
would make the distinction explicit.

---

## Patterns Worth Extracting

### 5. VRAM cleanup logic is duplicated

**Locations:** `_unload_current_worker`, `_load_mode` (exception handler)

The same sequence appears in two places:
```python
del self._worker
self._worker = None
gc.collect()
torch.cuda.empty_cache()
```

Extractable to a `_free_worker(worker)` helper that accepts a worker reference,
deletes it, and flushes caches. Both call sites become one line, and any future
change to cleanup (e.g. calling `worker.teardown()` before deletion) only needs to
happen in one place.

---

### 6. "Queue a force-reload if this mode is currently running" duplicated in routes

**Locations:** `save_all_modes`, `create_or_update_mode` in `model_routes.py`

Both handlers end with the same block:
```python
reload_queued = False
if <mode is current>:
    try:
        pool.switch_mode(name, force=True)
        reload_queued = True
    except Exception as e:
        logger.warning(...)
```

Natural fit for a `WorkerPool.reload_if_current(mode_name) -> bool` method — returns
`True` if a reload was queued, `False` otherwise. Routes become one-liners, and the
pattern won't be copy-pasted a third time when Phase 4 (LPU) adds more config-change
handlers.

---

## Robustness / Minor

### 7. Watchdog thread has no exception guard

**Location:** `WorkerPool._idle_watchdog_loop`

The loop body has no `try/except`. An unexpected exception (however unlikely) would
kill the thread silently. Eviction would stop working for the lifetime of the process
with no log evidence. A bare `try/except Exception` around the loop body with a
`logger.error` call would make failure observable and keep the thread alive.

---

### 8. Watchdog can enqueue duplicate eviction jobs

**Location:** `WorkerPool._idle_watchdog_loop`

If the worker queue is backed up and the check interval fires again before the
eviction `CustomJob` has been processed, a second eviction job is enqueued. Both are
harmless (the second hits the `already_unloaded` early-return in `_evict_if_idle`),
but it is wasted queue churn. An `_eviction_pending: bool` flag — set before
`put_nowait`, cleared at the start of `_evict_if_idle` — would suppress duplicates.

---

## Summary

| # | Severity | Type | Location |
|---|----------|------|----------|
| 1 | Bug | `_last_activity` not updated on failed jobs | `_worker_loop` |
| 2 | Bug | Demand reload errors opaque to caller | `_worker_loop` |
| 3 | Misleading | `_lock` unused | `WorkerPool.__init__` |
| 4 | Misleading | `get_current_mode()` ambiguous after eviction | `WorkerPool`, routes |
| 5 | Reuse | VRAM cleanup duplicated | `_unload_current_worker`, `_load_mode` |
| 6 | Reuse | Force-reload pattern duplicated in routes | `model_routes.py` |
| 7 | Robustness | Watchdog thread unguarded against exceptions | `_idle_watchdog_loop` |
| 8 | Minor | Duplicate eviction jobs possible under queue pressure | `_idle_watchdog_loop` |
