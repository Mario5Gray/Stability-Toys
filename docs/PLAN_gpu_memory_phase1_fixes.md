# Implementation Plan: GPU Memory Phase 1 Review Fixes

Addresses all 8 items from `REVIEW_gpu_memory_phase1.md`.
Files touched: `backends/worker_pool.py`, `server/model_routes.py`.

Execution order: bugs first, then dead code, then extractions, then robustness.

---

## Step 1 — Bug: `_last_activity` not updated on failed jobs (Review #1)

**File:** `backends/worker_pool.py`, `_worker_loop`

**Current code (line 453):**
```python
            result = job.execute(self._worker)

            if not job.fut.done():
                job.fut.set_result(result)

            self._last_activity = time.monotonic()   # ← inside try, never reached on error

        except Exception as e:
            ...
        finally:
            self.q.task_done()
```

**Change:** Remove `self._last_activity = time.monotonic()` from inside `try`.
Add it to `finally`, before `self.q.task_done()`:

```python
        except Exception as e:
            ...
        finally:
            self._last_activity = time.monotonic()
            self.q.task_done()
```

This ensures the idle timer resets whether the job succeeds or fails, preventing
false eviction of an actively-used (but error-producing) model.

---

## Step 2 — Bug: Demand reload errors opaque (Review #2)

**File:** `backends/worker_pool.py`, `_worker_loop`, demand reload block

**Current code:**
```python
                if self._worker is None and self._current_mode is not None:
                    logger.info(...)
                    self._load_mode(self._current_mode)
```

**Change:** Wrap `_load_mode` in a try/except that re-raises as a descriptive
`RuntimeError`:

```python
                if self._worker is None and self._current_mode is not None:
                    logger.info(...)
                    try:
                        self._load_mode(self._current_mode)
                    except Exception as load_err:
                        raise RuntimeError(
                            f"Demand reload of '{self._current_mode}' failed: {load_err}"
                        ) from load_err
```

The outer `except Exception` in `_worker_loop` catches this and puts it on
`job.fut`. The caller now gets a message that clearly identifies a reload failure
vs. a generation failure.

---

## Step 3 — Dead code: Remove unused `_lock` (Review #3)

**File:** `backends/worker_pool.py`, `WorkerPool.__init__`

**Change:** Delete line:
```python
        self._lock = threading.Lock()
```

No acquisition sites exist. The threading model is safe without it (all state
mutations happen on the worker thread). Leaving it implies false protection.

---

## Step 4 — Misleading: Expose `is_model_loaded` (Review #4)

**File:** `backends/worker_pool.py`

**Change:** Add a method to `WorkerPool`:
```python
    def is_model_loaded(self) -> bool:
        """True if a worker is currently live in memory."""
        return self._worker is not None
```

`get_current_mode()` intentionally keeps the mode name after eviction (needed
for demand reload). This gives callers a separate boolean to distinguish
"model in VRAM" from "mode name retained for demand reload".

**File:** `server/model_routes.py`, `get_models_status`

**Change:** Add `is_loaded` to the response:
```python
    return {
        "current_mode": current_mode,
        "is_loaded": pool.is_model_loaded(),
        "queue_size": queue_size,
        "vram": vram_stats,
    }
```

---

## Step 5 — Extraction: `_free_worker()` helper (Review #5)

**File:** `backends/worker_pool.py`

The sequence `del self._worker / self._worker = None / gc.collect() / empty_cache()`
appears identically in `_unload_current_worker` and in `_load_mode`'s exception
handler.

**Change:** Add a private helper to `WorkerPool`:
```python
    def _free_worker(self):
        """Drop the worker reference and flush the GPU allocator cache."""
        if self._worker is not None:
            del self._worker
            self._worker = None
        gc.collect()
        torch.cuda.empty_cache()
```

Replace both sites:

In `_unload_current_worker`, replace:
```python
        del self._worker
        self._worker = None
        gc.collect()
        torch.cuda.empty_cache()
```
with:
```python
        self._free_worker()
```

In `_load_mode` exception handler, replace:
```python
            if self._worker is not None:
                try:
                    del self._worker
                except Exception:
                    pass
                self._worker = None
            gc.collect()
            torch.cuda.empty_cache()
```
with:
```python
            self._free_worker()
```

---

## Step 6 — Extraction: `reload_if_current()` helper (Review #6)

**File:** `backends/worker_pool.py`

**Change:** Add to `WorkerPool`:
```python
    def reload_if_current(self, mode_name: str) -> bool:
        """
        Queue a force-reload if mode_name is the currently loaded mode.

        Returns True if a reload was queued, False otherwise.
        """
        if self.get_current_mode() != mode_name:
            return False
        logger.info(f"[WorkerPool] Config changed for loaded mode '{mode_name}'; queuing reload")
        try:
            self.switch_mode(mode_name, force=True)
            return True
        except Exception as e:
            logger.warning(f"[WorkerPool] Could not queue reload for mode '{mode_name}': {e}")
            return False
```

**File:** `server/model_routes.py`

In `create_or_update_mode`, replace:
```python
    reload_queued = False
    if name == pool.get_current_mode():
        logger.info(f"[API] Updated config for loaded mode '{name}'; queuing reload")
        try:
            pool.switch_mode(name, force=True)
            reload_queued = True
        except Exception as e:
            logger.warning(f"[API] Could not queue reload for mode '{name}': {e}")
```
with:
```python
    reload_queued = pool.reload_if_current(name)
```

In `save_all_modes`, the `new_model != old_model or new_loras != old_loras` branch,
replace:
```python
                logger.info(f"[API] Config changed for loaded mode '{current_mode}'; queuing reload")
                try:
                    pool.switch_mode(current_mode, force=True)
                    reload_queued = True
                except Exception as e:
                    logger.warning(f"[API] Could not queue reload for mode '{current_mode}': {e}")
```
with:
```python
                reload_queued = pool.reload_if_current(current_mode)
```

---

## Step 7 — Robustness: Exception guard on watchdog loop (Review #7)

**File:** `backends/worker_pool.py`, `_idle_watchdog_loop`

**Change:** Wrap the loop body in a try/except so an unexpected error does not
silently kill the thread:

```python
    def _idle_watchdog_loop(self):
        logger.debug("[WorkerPool] Idle watchdog loop running")

        while not self._stop.wait(timeout=self._idle_check_interval):
            try:
                if self._worker is None:
                    continue

                idle_secs = time.monotonic() - self._last_activity
                if idle_secs < self._idle_timeout:
                    continue

                logger.info(
                    f"[WorkerPool] Model idle for {idle_secs:.0f}s "
                    f"(timeout={self._idle_timeout:.0f}s); queuing eviction"
                )
                try:
                    evict_job = CustomJob(handler=self._evict_if_idle)
                    self.q.put_nowait(evict_job)
                except queue.Full:
                    logger.warning("[WorkerPool] Queue full; skipping idle eviction this cycle")
            except Exception:
                logger.error("[WorkerPool] Idle watchdog error", exc_info=True)

        logger.debug("[WorkerPool] Idle watchdog loop stopped")
```

---

## Step 8 — Robustness: Deduplicate eviction jobs (Review #8)

**File:** `backends/worker_pool.py`

**Change 1:** Add flag to `WorkerPool.__init__`:
```python
        self._eviction_pending = False
```

**Change 2:** In `_idle_watchdog_loop`, skip enqueueing if one is already in queue:
```python
                if self._eviction_pending:
                    continue
                try:
                    evict_job = CustomJob(handler=self._evict_if_idle)
                    self._eviction_pending = True
                    self.q.put_nowait(evict_job)
                except queue.Full:
                    self._eviction_pending = False
                    logger.warning("[WorkerPool] Queue full; skipping idle eviction this cycle")
```

**Change 3:** In `_evict_if_idle`, clear the flag at entry:
```python
    def _evict_if_idle(self):
        self._eviction_pending = False
        idle_secs = time.monotonic() - self._last_activity
        ...
```

Note: `_eviction_pending` is written from the watchdog thread and read/cleared on
the worker thread. For the concurrency level here (boolean flag, worst case is an
extra eviction job that `_evict_if_idle` will harmlessly skip), a bare bool is
sufficient. If guarding with a lock is preferred, `self._lock` from Step 3 can be
retained for this purpose alone.

---

## Summary

| # | Item | File | Net change |
|---|------|------|-----------|
| 1 | `_last_activity` → `finally` | `worker_pool.py` | 2-line move |
| 2 | Wrap demand reload error | `worker_pool.py` | 4-line try/except |
| 3 | Delete `self._lock` | `worker_pool.py` | 1-line delete |
| 4 | Add `is_model_loaded()` + status field | `worker_pool.py`, `model_routes.py` | ~8 lines |
| 5 | Extract `_free_worker()` | `worker_pool.py` | +5 lines, -10 lines |
| 6 | Extract `reload_if_current()` | `worker_pool.py`, `model_routes.py` | +10 lines, -16 lines |
| 7 | Exception guard in watchdog loop | `worker_pool.py` | +2 lines |
| 8 | `_eviction_pending` flag | `worker_pool.py` | +6 lines |

All changes are contained to two files. No API contracts change except the
`/api/models/status` response gaining an `is_loaded` boolean field (additive,
non-breaking).
