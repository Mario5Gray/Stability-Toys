# Governor Authority Reservation — mode-switch admission race

**FP:** `STABL-ltefhpkk` (StaleResolutionError window) + `STABL-iuiwzthc` (None-snapshot
ControlNet window). Same root, one fix.
**Date:** 2026-07-30
**Baseline:** `445eae3` (post PR #24, DeviceMemory)

---

## 1. Problem

A generate admitted concurrently with a mode switch resolves against *transient*
authority. Reproduced live on enigma three times (2026-07-22, 2026-07-27, 2026-07-30),
surviving the `WorkerPool` → `Governor` extraction.

### 1.1 Verified chain

1. `st gen --mode X` → CLI `client.SwitchMode` → `POST /api/modes/switch` →
   `Governor.switch_mode(X)` enqueues a `ModeSwitchJob` and the route returns
   `"queued"` **immediately** (`server/model_routes.py:248-256`). Fire-and-forget.
2. CLI then opens WS `job:submit`. Admission reads *live* authority
   (`server/ws_routes.py:186`). Two sub-cases, both stamp the pre-switch epoch N:
   - old snapshot still live → `snapshot.resolution_epoch` = N
     (`server/ws_routes.py:242-246`)
   - load in flight, `_active_snapshot is None` → `current_resolution_epoch()` falls
     back to `self._resolution_epoch`, still N (the bump is at
     `backends/governor.py:379`) — **and** the ControlNet branch at
     `server/ws_routes.py:220-225` fires, producing the spurious
     "ControlNet provider not yet implemented" of `STABL-iuiwzthc`.
3. Dispatch runs the `ModeSwitchJob` → `_load_mode` → epoch N+1, snapshot republished.
4. Dispatch runs the generate → barrier rejects: stamped N ≠ active N+1
   (`backends/governor.py:632-640`).

The model loads correctly. Only the racing generate is lost. Retry succeeds because
it is stamped after the switch settles.

### 1.2 Third symptom, not in either report

Because admission binds to the *live* mode, a generate that targets X also gets X's
work done against the **wrong mode's** configuration:

- `finalize_mode_generate_request` takes `size` / `num_inference_steps` /
  `guidance_scale` defaults from the outgoing mode.
- `resolve_controlnet_bindings` resolves against the outgoing
  `snapshot.resolved.profile.family_id`.

Today this is masked by the barrier rejecting the job. Any fix that relaxes the
barrier without moving admission would let it through silently. This is the decisive
argument against the order-aware-barrier approach (§2, option C).

### 1.3 The silent "unloaded" window (in scope)

`_load_mode` unregisters the outgoing mode from the registry and nulls
`_active_snapshot` (`backends/governor.py:338-340`), then re-registers only after the
load completes (`:370`) — tens of seconds for HunyuanDiT. For that whole window
`get_current_mode()`, `is_model_loaded()`, and `/api/models/status.vram` report
nothing loaded, **with no log line emitted**.

Every path that actually tears a worker down either logs (`_load_mode` prelude,
`_evict_if_idle`, `_load_mode` failure, the OOM branch of `_cleanup_vram`) or is
reachable only through API routes the CLI never calls during a generate
(`free_vram`, `unload_current_model`, `shutdown`). The CLI's per-generate surface is
`/api/models/status`, `/api/modes/switch`, the WS submit, and `/storage/<key>` — none
of which unload. `StaleResolutionError` does not match the OOM substring test at
`backends/governor.py:703`, so the OOM branch is out.

The registry gap is therefore the only *silent* "unloaded" state a bundled
`st gen --mode X` can produce, and it explains why manual `st modes switch` then
`st gen` never shows it: the load has already finished before the generate runs.
Same root — authority is unobservable while in transition — so it is fixed here.

---

## 2. Approaches considered

**A — Governor authority reservation (chosen).** Establish the switch's epoch and
resolved model atomically with its *enqueue*, and admit generates against that
reservation. Cures both windows and the wrong-mode-config symptom, leaves the barrier's
epoch comparison untouched, needs no new transport concept.

**B — Serialize the inline switch.** Have WS admission await the switch future before
capturing the snapshot, porting what `/generate` already does at
`server/lcm_sr_server.py:552-557`. Smallest diff and obviously correct, but delays
`job:ack` by a full model load, helps no other concurrent admitter, and leaves the
authority model reading live state. `STABL-ltefhpkk` ranks this third.

**C — Order-aware barrier (rejected).** Relax the barrier at execution instead of
moving admission. Does not fix `STABL-iuiwzthc` at all — that rejection happens at
admission, before the barrier ever runs — and it lets §1.2's wrong-mode defaults and
bindings through. This is the "lazy-stamp runs on the wrong model" hazard.

---

## 3. Design

### 3.1 The authority reservation

An **authority reservation** is the epoch and resolved model that a queued mode
switch *will* publish, established atomically with the switch's enqueue.

No new type. Reservations reuse `ActiveModelSnapshot` — same five fields — so every
existing admission consumer (`snapshot.mode`, `snapshot.resolved.profile.family_id`,
`snapshot.resolution_epoch`) works unchanged.

New Governor state, all guarded by `_job_lock`:

```text
_pending_authorities: list[ActiveModelSnapshot]   # FIFO, one per enqueued-not-yet-published switch
_dead_epochs: set[int]                            # epochs whose load failed (§3.5)
```

**Terminal authority** — the authority a job admitted *now* will actually execute
against:

```text
terminal = _pending_authorities[-1] if _pending_authorities else _active_snapshot
```

### 3.2 Epoch reservation replaces the in-load bump

`_reserve_authority(mode_name) -> ActiveModelSnapshot`:

1. `resolve_model(...)` **outside** `_job_lock` — it performs `detect_model` disk I/O
   and must not be held across the lock.
2. Under `_job_lock`: `self._resolution_epoch += 1`; build the reservation; append to
   `_pending_authorities`; return it.

Concurrent switches serialize on the lock, so epochs stay monotone and each switch
owns exactly one.

`ModeSwitchJob` gains a `reservation: Optional[ActiveModelSnapshot]` field.
`_load_mode(mode_name, reservation=None)` **publishes the reservation verbatim** — its
epoch, mode, resolved, and binding — instead of `self._resolution_epoch += 1` plus a
freshly minted snapshot, and pops it from `_pending_authorities`. It reuses
`reservation.resolved` / `.binding`, so `detect_model` moves off the dispatch thread.

**Reservation-less callers** — the `__init__` default load and any direct
`_load_mode` call — reserve inline and behave exactly as today.

**Demand reload is epoch-neutral and must stay that way.** `_reload_from_snapshot`
(`backends/governor.py:394-407`) does not call `_load_mode` and does not bump the
epoch: it reconstructs the worker from the retained snapshot, so queued generates
stamped at epoch N correctly survive an eviction/reload cycle. Introducing a reserve
there would bump to N+1 and spuriously reject every one of them. This path is
unchanged, byte for byte.

### 3.3 `switch_mode` short-circuits before reserving

`Governor.switch_mode(X, force=False)` must **not** reserve when X is already the
terminal mode. The dispatch fast-path at `backends/governor.py:606` returns
`{"status": "already_loaded"}` without calling `_load_mode`, so a reservation created
for such a switch is never published — and any generate bound to it is stamped N+1
against an active N, reproducing the exact bug being fixed, self-inflicted.

Required behavior, checked under `_job_lock` before reserving:

| Condition | Result |
| --- | --- |
| `X == terminal.mode_name` and terminal is `_active_snapshot` and not `force` | return a completed Future with `{"mode": X, "status": "already_loaded"}`; reserve nothing, enqueue nothing |
| `X == terminal.mode_name` and terminal is a pending reservation and not `force` | return a completed Future with `{"mode": X, "status": "already_queued"}`; reserve nothing, enqueue nothing |
| otherwise | reserve, enqueue `ModeSwitchJob(X, reservation=r)` |

`force=True` always reserves and enqueues (that is the `reload_current_mode` path).

The dispatch fast-path at `:606` stays as a defensive no-op; with this guard it
becomes unreachable for the non-`force` case.

### 3.4 Target-aware admission

One new Governor entry point, the single place both transports call:

```python
def admit_generation(self, target_mode: Optional[str]) -> Optional[ActiveModelSnapshot]:
    """Return the authority a job admitted NOW will execute against, creating the
    switch if the caller targets a mode that is neither active nor pending."""
```

Under `_job_lock`:

| `target_mode` | Returns |
| --- | --- |
| `None` | `_active_snapshot` — today's behavior, unchanged |
| `== terminal.mode_name` | the terminal authority (covers both "already active" and "a switch to X is already queued") |
| anything else | reserve + enqueue `ModeSwitchJob(target_mode, reservation=r)`; return `r` |

**`target_mode is None` deliberately returns the *active* snapshot, not the terminal
one.** A generate naming no mode means "the current mode"; if a switch supersedes it,
it must be rejected. Returning the terminal authority there would be precisely the
wrong-model hazard of §1.2.

The caller then runs *all* admission against the returned authority —
`admit_generation_operation`, `finalize_mode_generate_request`,
`enforce_controlnet_policy`, `resolve_controlnet_bindings` — and stamps
`GenerationJob(resolution_epoch=authority.resolution_epoch)`. ControlNet family,
generation defaults, and the epoch therefore all come from the mode that will actually
run the job.

**Two-phase is race-free.** Reserve-then-preprocess-then-submit does not hold a lock
across preprocessing. Once reserved, the generate's stamp matches whatever
`_load_mode` publishes. If another switch lands in between, it reserves N+2 and
becomes terminal — our generate is then *genuinely* superseded and the barrier
correctly rejects it.

**`queue.Full` unwind.** `admit_generation` holds `_job_lock` across
append-to-`_pending_authorities` and `q.put`. This cannot deadlock: the dispatch loop
never holds `_job_lock` while blocked on `q.get`. If the bounded put raises
`queue.Full`, the reservation is rolled back — popped from `_pending_authorities`, and
its epoch added to `_dead_epochs` in case a generate was already stamped against it —
before the exception propagates. A dangling reservation would poison terminal authority
for every later admission.

`queue.Full` on the *generate's* own `submit_job`, after `admit_generation` returned, is
a separate and benign case: the switch is legitimately queued and will load the target;
only the generate is lost, with today's "Queue full" error. No rollback, because the
reservation is still going to be published.

### 3.5 The barrier, and failed loads

The epoch-equality comparison and its `StaleResolutionError` at
`backends/governor.py:632-640` are **unchanged**; the fix is in what gets stamped, not
in what enforces it. It still rejects a generate for X after an unrelated switch to Y,
and a no-mode generate superseded by any switch.

The one structural edit is that the `snapshot is not None` conjunct moves *out* of the
equality guard and into a **dead-epoch check that runs immediately before it**:

```python
if generation_job is not None:
    with self._job_lock:
        snapshot = self._active_snapshot
        dead = generation_job.resolution_epoch in self._dead_epochs
    if dead or snapshot is None:
        raise ModeLoadFailedError(...)          # new exception type
    if snapshot.resolution_epoch != generation_job.resolution_epoch:
        raise StaleResolutionError(...)         # unchanged
```

This closes a confirmed latent bug. Today the barrier is guarded on
`snapshot is not None`, so after a failed load the barrier is **skipped entirely** and
the job falls through to the subprocess branch at `backends/governor.py:664`, where
`InProcessWorkerHandle.submit` fails it with a misleading
`RuntimeError("No worker available for generation")`
(`backends/worker_handle.py:150-151`). Reservations make that path more reachable, so
it is fixed here as a tested case rather than a side effect.

**On load failure**, `_load_mode`'s except path (`backends/governor.py:351-360`)
additionally drops its reservation from `_pending_authorities` and adds the reserved
epoch to `_dead_epochs`. `_resolution_epoch` stays at the reserved value — epochs are
monotone and never reused.

**Pruning.** On each successful publish, `_dead_epochs` discards entries below the
newly published epoch. Since epochs are monotone and `admit_generation` binds only to
terminal authority, no *newly* admitted job can carry a pruned epoch. A generate that
was already queued at a pruned dead epoch degrades to `StaleResolutionError` instead
of `ModeLoadFailedError` — correctly rejected either way, with a less precise message.
Accepted: the set holds one entry per consecutive failed load, which is naturally
tiny, and eager queue-scanning instead leaves a window where a job is dequeued between
the failure and the scan.

### 3.6 Observability — closing the silent window

`get_current_mode()` **keeps its present meaning** (the actually-loaded mode). It has
nine call sites, several of which require exactly that: `model_routes.py:561`
(`was_loaded` for mode-delete), `governor.py:819` (`reload_if_current`),
`backends/platforms/cuda.py:78`, `analysis_routes.py:70`. Overloading it to return
terminal authority would silently change all of them.

Instead:

- New `get_pending_mode() -> Optional[str]` returns the terminal reservation's
  `mode_name`, or `None` when no switch is outstanding.
- `_build_runtime_status` gains `pending_mode` and reports `loading` when a
  reservation is outstanding, so `/api/models/status` distinguishes "nothing loaded"
  from "loading X" during the §1.3 window.
- `_last_activity` is refreshed when a reservation is created, so the idle watchdog
  cannot evict a mode that was just requested.

`_resolve_chat_config` (`ws_routes.py:399`) therefore keeps today's behavior — no
behavioral decision is forced on the chat path.

---

## 4. Invariants

1. A generate's stamped epoch equals the epoch of the authority it was admitted
   against, and that authority is the one it executes against — or it is rejected.
2. Epochs are monotone and never reused, including across failed loads.
3. `admit_generation` binds only to terminal authority; no other authority is
   reachable from admission.
4. Demand reload never changes the epoch.
5. A reservation is either published by `_load_mode`, or dropped into `_dead_epochs`.
   It is never silently abandoned.
6. While a reservation is outstanding, the target mode is observable via
   `get_pending_mode()` and `/api/models/status`.

---

## 5. Wiring

| Site | Change |
| --- | --- |
| `server/ws_routes.py:366-387` | `_build_generate_request` passes `mode=params.get("mode")` — already on the wire, currently dropped |
| `server/ws_routes.py:186`, `:242-246` | `get_active_model_snapshot()` → `admit_generation(params.get("mode"))`; the `current_resolution_epoch()` fallback is removed |
| `server/lcm_sr_server.py:552-578` | the blocking `switch_fut.result(timeout=30)` is replaced by the same `admit_generation` call; both transports share one admission path |
| `cli/go/cmd/st/gen.go:670-677` | drop the pre-emptive `SwitchMode` and its `CurrentMode` round-trip |
| `backends/worker_pool.py` | delegating passthroughs for `admit_generation`, `get_pending_mode` |

`st modes switch` is untouched and stays fire-and-forget.

### 5.1 Dead code, not wired

`server/ws_routes.py:620-633` — the second `GenerationJob` construction site, and the
`resolution_epoch=pool.current_resolution_epoch()` stamp at `:626` — is
**unreachable**. `_run_generate` has exactly one caller (`:277`), in the `else` branch
of `if getattr(state, "use_mode_system", False)`, so the `use_mode_system` test at
`:620` is always false there. Left alone and noted; the live HTTP stamp site is the
`lcm_sr_server.py` one covered above.

### 5.2 CLI/server version coupling

Dropping `gen.go:670-677` means a CLI built from this change, run against an
**unfixed** server, silently generates on the current mode — because today's server
drops `params["mode"]`. Acceptable: CLI and server ship from one repo and one image.
Stated explicitly so it is a known coupling rather than a discovered one.

---

## 6. Costs

- `resolve_model` disk I/O (`detect_model`) moves onto the admitting request thread.
  Admission latency grows by detection time. The compensating win is fail-fast: a bad
  or unreadable target mode now fails at admission with a clear error instead of tens
  of seconds later on the dispatch thread. Detection also leaves the dispatch thread
  entirely, since `_load_mode` reuses the reservation's resolved value.
- `admit_generation` is side-effecting: it can enqueue a mode switch. This is the only
  shape that makes reserve-and-enqueue atomic; a split accessor-plus-switch API
  reintroduces the interleave window the fix exists to close.

---

## 7. Test matrix

All deterministic, no GPU. `tests/test_governor.py` already injects `mode_config`,
`registry`, and stub handles.

| # | Case | Today |
| --- | --- | --- |
| 1 | `ModeSwitchJob(X)` queued ahead of a generate admitted via `admit_generation("X")` → completes, barrier silent | `StaleResolutionError` (STABL-ltefhpkk) |
| 2 | ControlNet generate admitted mid-load → bindings resolve against the **target** family | `NotImplementedError` (STABL-iuiwzthc) |
| 3 | Generate admitted for X, then an unrelated switch to Y → `StaleResolutionError` | passes today, must keep passing |
| 4 | Generate with `mode=None` superseded by a switch → `StaleResolutionError` | passes today, must keep passing |
| 5 | `switch_mode(X)` when X is already terminal → no reservation, no enqueue; a generate bound to X still succeeds (§3.3 guard) | n/a — regression this design would otherwise introduce |
| 6 | Generate targeting X gets X's `size`/`steps`/`guidance`, not the active mode's | wrong-mode defaults (§1.2) |
| 7 | Failed load → reservation dropped, queued generates stamped at that epoch raise `ModeLoadFailedError` | falls through to `governor.py:664`, misleading `RuntimeError` |
| 8 | `snapshot is None` with a stamped generate → `ModeLoadFailedError` | barrier skipped, same fallthrough |
| 9 | Mid-switch: `get_pending_mode() == X`, status reports `loading`, `_last_activity` refreshed | reports nothing loaded, silently (§1.3) |
| 10 | Demand reload after idle eviction does not change the epoch; queued generates survive | passes today, must keep passing |
| 11 | `queue.Full` during `admit_generation` → reservation rolled back, terminal authority unpoisoned | n/a |

Then `drift check` on `backends/governor.py` and the spec.

---

## 8. Non-goals

- Re-shaping the barrier's comparison. Epoch equality carries the same information
  once stamp and publish come from one reserved authority; target-identity matching
  would be churn for no gain.
- Multi-GPU / UUID-keyed allocation (`STABL-cchxvuhs`).
- Facet-3 `SubprocessWorkerHandle`. The dead-epoch check touches the subprocess
  branch's reachability only; the branch itself is unchanged.
- Removing the `worker_pool.py` facade.
