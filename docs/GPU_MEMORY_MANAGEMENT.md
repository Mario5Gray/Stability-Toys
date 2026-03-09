# GPU Memory Management Plan

Strategy for keeping GPU memory under control: idle eviction, demand reloading,
VRAM pressure response, and load-process-unload pipelines.

---

## Current State (baseline)

- `ModelRegistry` — VRAM accounting only; no eviction logic
- `WorkerPool` — single worker; loads on startup, unloads only on mode switch or shutdown
- `_unload_current_worker()` — `del worker` + `gc.collect()` + `torch.cuda.empty_cache()`
- No idle timer, no VRAM pressure response, no multi-model sequencing

### Already done

- `_load_mode` cleans up partial GPU state and sets honest `_current_mode = None` on failure
- `WorkerPool.__init__` catches startup load failure — server starts in "no model" state
  instead of crashing

---

## Phase 1 — Idle Eviction + Demand Reload

**Problem:** Model sits in VRAM indefinitely after the last request.

### Idle eviction

- Add `_last_activity: float` timestamp to `WorkerPool`, updated on every job completion
- Background thread `_idle_watchdog` checks every `IDLE_CHECK_INTERVAL_SECS` (default 30 s)
- If `time.monotonic() - _last_activity > IDLE_TIMEOUT_SECS` (default 300 s), calls
  `_unload_current_worker()`
- Watchdog thread is daemon, started alongside the worker thread, stopped on `shutdown()`
- Config via env: `MODEL_IDLE_TIMEOUT_SECS`, `MODEL_IDLE_CHECK_INTERVAL_SECS`

### Demand reload

Required for eviction to be useful: pool must be able to recover without a restart.

- In `_worker_loop`, before executing a job, check `if self._worker is None and self._current_mode`
- If true, call `_load_mode(self._current_mode)` before executing
- Reuses existing mode name — the pool remembers what was last loaded even after eviction
- Generation job blocks in queue until reload completes (natural serialisation)

### State machine

```
LOADED ──── idle timeout ────> EVICTED
EVICTED ─── job arrives ─────> LOADING ──── success ──> LOADED
                                        └─── failure ──> EVICTED (error returned to caller)
```

### Files to change

- `backends/worker_pool.py` — `_last_activity`, watchdog thread, demand-reload in loop

---

## Phase 2 — VRAM Pressure Response

**Problem:** No policy for "we need VRAM before loading a new model."

- `ModelRegistry.evict_lru()` — evicts least-recently-used model when
  `available_vram < VRAM_RESERVE_BYTES`
- `VRAM_RESERVE_GB` env var (default 1.0 GB) sets the headroom threshold
- Called inside `_load_mode()` before the factory call, so it runs on every model switch
- For the current single-worker design this is mainly defensive; becomes important
  if multi-model support is added later

### Files to change

- `backends/model_registry.py` — `evict_lru()`, `update_activity(name)`
- `backends/worker_pool.py` — call `registry.evict_lru()` at top of `_load_mode`

---

## Phase 3 — Lazy Init

**Problem:** Server startup loads the model immediately; cold start is slow and
wastes VRAM if nobody connects right away.

- `WorkerPool.__init__` gains `lazy: bool = False` parameter
- If `lazy=True`, skip the initial `_load_mode(default_mode)` call entirely
- `_current_mode` is pre-set to `default_mode` so demand reload in Phase 1 knows
  what to load on first job
- Pairs naturally with idle eviction: evict → demand-reload on next job

### Files to change

- `backends/worker_pool.py` — `lazy` parameter, conditional init load
- `server/lcm_sr_server.py` — pass `lazy=True` if `LAZY_MODEL_LOAD=1` env var set

---

## Phase 4 — Load-Process-Unload (LPU) Job Type

**Problem:** Multi-model workflows (generate → upscale → postprocess) need to
sequence different models without keeping all of them resident simultaneously.

- New `SequentialJob` type: list of `(mode_name, spec)` steps
- `WorkerPool._worker_loop` handles it by iterating steps, switching mode between each
- Each step: switch mode (load), execute, optionally unload before next step
- Unload between steps controlled by `release_after: bool` per step (default False)
- Keeps memory at max(single model size) rather than sum(all model sizes)

### Files to change

- `backends/worker_pool.py` — `SequentialJob` dataclass, loop handling

---

## Implementation Order

| Phase | Feature | Value | Complexity |
|-------|---------|-------|------------|
| 1 | Idle eviction + demand reload | High — reclaims VRAM automatically | Low |
| 2 | VRAM pressure / evict_lru | Medium — prevents OOM on switch | Low |
| 3 | Lazy init | Low–medium — faster startup | Trivial |
| 4 | LPU / SequentialJob | High if multi-model workflows needed | Medium |

---

## Configuration Reference

| Env var | Default | Effect |
|---------|---------|--------|
| `MODEL_IDLE_TIMEOUT_SECS` | `300` | Seconds idle before eviction |
| `MODEL_IDLE_CHECK_INTERVAL_SECS` | `30` | Watchdog poll interval |
| `VRAM_RESERVE_GB` | `1.0` | Minimum free VRAM before evicting |
| `LAZY_MODEL_LOAD` | `0` | Skip model load at server startup |
