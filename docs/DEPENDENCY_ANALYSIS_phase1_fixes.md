# Dependency Analysis: Phase 1 Review Fixes

Cross-checks every planned change against its call sites, callers, and existing
tests. Flags regressions before a single line is touched.

---

## Step 1 — `_last_activity` moved to `finally`

### Call sites of `_last_activity`
| Location | Use | Impact |
|---|---|---|
| `__init__` | Initialise to `time.monotonic()` | No change |
| `_idle_watchdog_loop` | Read: `time.monotonic() - self._last_activity` | No change |
| `_evict_if_idle` | Read: same | No change |

### Behaviour change
Moving the write from the end of `try` to `finally` means it also fires after:
- **`ModeSwitchJob`** — correct: mode switches are user activity, resetting the
  idle timer is right
- **`CustomJob` (including the eviction job itself)** — the eviction job runs
  `_evict_if_idle`, which unloads the worker. After that, `_worker is None`, so
  the watchdog loop body skips on every subsequent check anyway. Updating
  `_last_activity` here is inert.

### Test impact
`test_custom_job_with_exception` — exception still propagates to the future;
`finally` fires after the `except` block. ✓ No regression.

No test asserts on `_last_activity` directly. ✓

---

## Step 2 — Demand reload errors wrapped in `RuntimeError`

### Callers that receive the future exception
| Location | How it handles the exception | Impact |
|---|---|---|
| `ws_routes._run_generate` | `except Exception as e` → `job:error` with `str(e)` | Error message changes, handling unchanged |
| `lcm_sr_server.generate` | `except Exception as e` → HTTP 500 with `str(e)` | Same |
| Tests | See below | See below |

### Test regression — `test_model_lifecycle.test_generate_after_unload_fails`

```python
pool._unload_current_worker()  # clears _worker, but _current_mode stays "mode-a"
future = pool.submit_job(GenerationJob(...))
with pytest.raises(RuntimeError, match="No worker available"):
    future.result(timeout=5.0)
```

After `_unload_current_worker()`, `_current_mode` is still `"mode-a"` and
`_worker` is `None`. The worker loop sees `_worker is None and _current_mode is
not None` and fires demand reload.

**Pre-Step 2:** `_load_mode` may raise (e.g. factory signature mismatch in the
test fixture — see note below). The raw exception propagates. The test's
`match="No worker available"` is not guaranteed to match.

**Post-Step 2:** Demand reload failure is re-raised as
`RuntimeError("Demand reload of 'mode-a' failed: ...")`.
`match="No worker available"` **will not match**.

**Action required:** Verify whether this test currently passes. If it does, it
likely relies on `_current_mode` being cleared by a failed load (which lets the
job reach `execute(None)` and raise "No worker available"). The fix for this
test is to reset `_current_mode` to `None` explicitly in the test, or change
the `match` to cover both "No worker available" and "Demand reload.*failed".

**Note on fixture factory:** `test_model_lifecycle.py`'s `mock_worker_factory`
is `Mock(side_effect=factory)` where `factory(worker_id: int)` does not accept
`model_path`. `_load_mode` calls `_worker_factory(worker_id=0, model_path=...)`.
This would raise `TypeError` during demand reload. After Step 2 it becomes
`RuntimeError("Demand reload of 'mode-a' failed: factory() got an unexpected
keyword argument 'model_path'")`. The test_worker_pool.py fixtures use a plain
`Mock()` (accepts anything), so they are unaffected.

---

## Step 3 — Delete `self._lock`

### Usages
`find_usages` returns no acquisition site for `self._lock` anywhere in the
codebase. Safe to delete.

### Test impact
No test inspects `pool._lock`. ✓

---

## Step 4 — Add `is_model_loaded()` + status field

### New symbol, no conflicts
`find_usages` for `is_model_loaded` returns empty. No naming conflict.

### `/api/models/status` response shape change
All current callers of this endpoint:
- **Frontend config screen** — reads `current_mode`, `vram`, `queue_size`.
  Adding `is_loaded: bool` is additive; existing reads are unaffected.
- **`ws_routes._build_status`** — independent path, not affected.

### Test impact
`test_worker_pool.TestModeSwitching.test_get_current_mode` — tests
`get_current_mode()` only; `is_model_loaded()` is not tested there.

`test_model_lifecycle.TestBasicLifecycle` — calls `pool._worker is not None`
directly rather than using the new method. Will still work. The new method
should be added to the test suite but is not a regression risk.

---

## Step 5 — Extract `_free_worker()`

### Sites that call `_unload_current_worker`
| Caller | Effect of change |
|---|---|
| `_load_mode` (unload before reload) | Now calls `_unload_current_worker` → `_free_worker` internally. Same behaviour. |
| `_evict_if_idle` | Same. |
| `shutdown` | Same. |
| `test_model_lifecycle` direct calls: `pool._unload_current_worker()` | `_unload_current_worker` remains public; tests unchanged. |

### Behaviour equivalence check

**`_unload_current_worker` before vs after:**
- Before: `if self._worker is None: return` early; then `del / None / gc / empty_cache`.
- After: `if self._worker is None: return` early; then `self._free_worker()`.
- `_free_worker` does: `if self._worker is not None: del / None`; then `gc / empty_cache` unconditionally.
- Since `_unload_current_worker` only reaches `_free_worker` when `_worker is not None`, both branches are equivalent.

**`_load_mode` exception handler before vs after:**
- Before: `if self._worker is not None: del / try/except / None`; then `gc / empty_cache`.
- After: `_free_worker()` — same logic, same outcome, plus the `try/except`
  around `del` is replaced by the clean `if` guard in `_free_worker`.

### Test impact
`test_unload_triggers_gc_and_cache_clear` — patches `gc.collect` and
`torch.cuda.empty_cache` and calls `_unload_current_worker` directly. The patch
targets `gc.collect` via `patch.object(real_gc, 'collect')` and patches
`backends.worker_pool.torch.cuda.empty_cache`. Both still execute through the
same call chain. ✓

`test_mode_switch_clears_cuda_cache` — same patching approach. ✓

---

## Step 6 — Extract `reload_if_current()`

### New symbol, no conflicts
`find_usages` for `reload_if_current` returns empty. No naming conflict.

### Behaviour equivalence — `create_or_update_mode`

**Before:**
```python
reload_queued = False
if name == pool.get_current_mode():
    try:
        pool.switch_mode(name, force=True)
        reload_queued = True
    except Exception as e:
        logger.warning(...)
```

**After:**
```python
reload_queued = pool.reload_if_current(name)
```

`reload_if_current` does exactly the same check and call. ✓

### Behaviour equivalence — `save_all_modes`

The `reload_if_current` call replaces only the `new_model != old_model or
new_loras != old_loras` inner branch. The outer conditions (current mode
removed → switch to default; model/loras changed → force reload) remain. The
`new_mode_data is None` branch uses a plain `switch_mode` (no force), which is
correct and **not** replaced by `reload_if_current`. ✓

### `switch_mode` call sites unaffected
`lcm_sr_server.generate` and `model_routes.switch_mode` (the route handler)
call `pool.switch_mode` directly and are not touched.

---

## Step 7 — Exception guard on watchdog loop

### Behaviour change
`continue` statements inside the `try` block still continue the outer `while`
loop — Python resolves `continue` to the nearest enclosing loop, not the
nearest `try`. ✓

An unhandled exception now logs an error and keeps the thread alive rather than
dying silently. No functional change to the normal path.

### Test impact
No tests target `_idle_watchdog_loop` directly. ✓

---

## Step 8 — `_eviction_pending` flag

### Thread safety
`_eviction_pending` is a plain `bool` written from the watchdog thread and read
+cleared from the worker thread (and also written back to `False` in the
watchdog thread on `queue.Full`). CPython's GIL makes single `bool` reads/writes
atomic for primitive types. Worst case is a missed suppression (one extra
eviction job slips through), which `_evict_if_idle`'s `already_unloaded`
early-return handles safely. No data corruption possible.

### Initialisation order
`_eviction_pending = False` must be set in `__init__` **before**
`_start_watchdog_thread()`. The plan places it in the attributes block alongside
`_idle_timeout` and `_last_activity`, well before the watchdog starts. ✓

### Test impact
`_evict_if_idle` clears `_eviction_pending` at entry. The existing tests that
call `CustomJob(handler=self._evict_if_idle)` indirectly do not assert on
`_eviction_pending`, so they continue to pass. ✓

---

## Summary of Regression Risks

| Step | Risk | Severity | Action |
|---|---|---|---|
| 2 | `test_generate_after_unload_fails` — `match="No worker available"` may not match wrapped demand-reload error | Medium | Verify test currently passes; if so, update `match` to accept both messages, or reset `_current_mode` in test setup |
| 2 | `test_model_lifecycle.py` factory accepts `worker_id` only, not `model_path` — demand reload will raise `TypeError` which becomes wrapped `RuntimeError` | Medium | Pre-existing inconsistency; doesn't worsen with our change but surface it for test fix |
| 5 | `gc.collect` + `empty_cache` now called via `_free_worker` which may be patched differently in tests | Low | Existing test patches both `gc.collect` and `torch.cuda.empty_cache` at the module level; call chain unchanged |
| 1,8 | Eviction job and mode-switch jobs update `_last_activity` (Step 1) | None | Correct behaviour; no test broken |
| 3,4,6,7 | None identified | None | — |

**One test likely needs updating before Step 2 can land:**
`tests/test_model_lifecycle.py::TestBasicLifecycle::test_generate_after_unload_fails`
