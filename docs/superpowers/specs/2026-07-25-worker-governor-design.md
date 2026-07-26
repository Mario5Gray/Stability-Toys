# Worker Governor: control-plane extraction behind a WorkerHandle interface

**FP:** STABL-vdkdruox (child of umbrella STABL-nvmieaxh)
**Depends on:** STABL-yoauoqao (Backplane data-plane transport, merged `919a1d6`) · STABL-sqqlkmdl (VRAM accounting, merged `243455e`)
**Consumes (folded in):** STABL-qfjfflrx (seam inventory — contract + audit phase only; wire-form + CustomJob map deferred to facet-3)
**Enables:** facet-3 subprocess worker (`SubprocessWorkerHandle`) · mode-switch race fixes (`STABL-ltefhpkk`, `STABL-iuiwzthc`) · API status/VRAM routing · timed-out-job reap (`STABL-qvmdayhb`)
**Status:** design, pre-implementation
**Date:** 2026-07-25

---

## 1. Purpose

Extract the **control plane** out of `WorkerPool` behind a locality-agnostic
`WorkerHandle` interface, `InProcessWorkerHandle` first, with **zero observable
behavior change** — the same no-op-facade playbook that shipped the backplane
(0-byte `ws_routes.py` diff, 156 passed).

This is the control-plane counterpart to the backplane (data plane, merged). The
four-part worker-as-a-service model falls out cleanly:

| Layer | Owns | Issue |
|-------|------|-------|
| **Governor** (control plane) | queue, epoch/snapshot authority, admission barrier, dispatch, lifecycle (spawn/ready/kill/respawn) | **this** |
| **WorkerHandle** (locality seam) | uniform interface to ONE worker regardless of where it runs | **this** (interface) |
| **Backplane** (data plane) | carries job-in / status+result-out streams across transport | STABL-yoauoqao (done) |
| **Worker** (executor) | the resource-bound CudaWorker | existing |

The **queue is the Governor↔Handle boundary**; the **backplane is the Handle's
output contract**. `SubprocessWorkerHandle` / `RemoteWorkerHandle` plug in later
with no Governor or backplane change.

### Why now

The backplane landed as a provable no-op behind `submit_job()→Future`. But the
backplane is purely data-plane — it carries frames; it owns no authority. Today
`WorkerPool` is a **proto-governor fused to an in-thread worker**: it owns the
queue, resolution epoch, active snapshot, `_load_mode`/`_unload_current_worker`,
OOM recovery, idle eviction, AND the worker thread + factory (`worker_pool.py`,
1105 LOC). That fusion is what blocks facet-3 (subprocess): the threaded-worker
coupling has no seam to swap. Extracting the control plane into a Governor +
`InProcessWorkerHandle` creates that seam without changing any caller.

---

## 2. Decisions (settled in scoping, recorded on `fp context STABL-vdkdruox`)

| # | Axis | Decision |
|---|------|----------|
| D1 | Next umbrella step | **Governor** (control plane). Dependency-correct: Backplane → Governor → facet-3. |
| D2 | v1 scope | **Pure no-op extraction.** Race fixes + API status routing are follow-on issues, not v1. Same playbook as the backplane's 0-byte `ws_routes.py` diff. |
| D3 | `WorkerPool` fate | **Thin delegating facade** over the Governor (transitional shim; deleted later when routes migrate to the Governor directly — same pattern as the deferred progress→WS migration). `get_worker_pool()` singleton + every test stay put. |
| D4 | Seam inventory (`STABL-qfjfflrx`) | **Split, not folded whole.** In v1: `WorkerHandle` contract + CUDA-in-parent audit (recon that informs design; no code moves). **Defer to facet-3:** `ControlNetBinding` wire form (serialization is dead weight in-proc) + `CustomJob`→typed-control-message map (`CustomJob` powers `_evict_if_idle` in-proc via a callable; redesigning it touches working eviction code and jeopardizes the "existing suite green unmodified" proof). The `WorkerHandle` contract must **not preclude** serialization (pass the job object; the subprocess handle serializes later) but must **not implement** it. |

### What D2 excludes (follow-on issues, tracked in §7)

- **Mode-switch race fixes** (`STABL-ltefhpkk`, `STABL-iuiwzthc`): authority is
  now in one place, so the fix is possible — but it's a *behavior change*, not
  an extraction. Separate issue after v1 proves the no-op.
- **API status/VRAM routing**: remove inline `torch.cuda.mem_get_info()` at
  `ws_routes.py:853-854` + the `is_available` gates; route through the Governor.
  Separate issue; pulls in seam-inventory territory.
- **facet-3 subprocess worker**: `SubprocessWorkerHandle`; carries the
  backplane's facet-3 debts (`cancel_job→record.sink` subscription wiring,
  `STALE_EPOCH` consumer-injected reconstruction registry, IPC `request(n)`/
  `job_id`/`result()` hardening).

---

## 3. Architecture

### 3.1 Package layout

| Module | Owns | Status |
|--------|------|--------|
| `backends/governor.py` (new) | `Governor` **+ shared job types** (`Job`, `GenerationJob`, `ModeSwitchJob`, `CustomJob`, `JobType`, `JobRecord`, `ActiveModelSnapshot`, `StaleResolutionError`, `_FutureBridge`, `WorkerFactory`) — the Governor is the primary consumer of these types, so they live with it | **new** |
| `backends/worker_handle.py` (new) | `WorkerHandle` ABC + `WorkerHealth` + `InProcessWorkerHandle` impl | **new** |
| `backends/worker_pool.py` (existing) | `WorkerPool` = thin facade delegating to `Governor`; **re-exports** the shared types (`from backends.governor import GenerationJob, ...`) so `from backends.worker_pool import GenerationJob` (`ws_routes.py:621`) stays unbroken; `get_worker_pool`/`reset_worker_pool` stay | **reduced to facade** |
| `backends/backplane/` (existing) | data-plane transport — unchanged | done (PR #19) |

**Import graph (acyclic):** `governor.py` imports `backplane` + `model_resolution` + `base` + `worker_handle` (for the `WorkerHandle` type hint); `worker_handle.py` imports `backplane` + `base` (NOT `governor` — the handle is locality-agnostic and doesn't know the Governor); `worker_pool.py` imports `governor` and re-exports its types. No cycle. The public surface (`from backends.worker_pool import GenerationJob`) is preserved by re-export, not by ownership.

### 3.2 The WorkerHandle interface (locality-agnostic)

The contract the Governor programs to, regardless of where the worker runs.
Defined here; `InProcessWorkerHandle` implements it today.

```python
@dataclass
class WorkerHealth:
    """Liveness + readiness snapshot the Governor reads for admission/status."""
    state: str            # starting | ready | busy | draining | dead
    vram_bytes: int       # current allocated VRAM (0 if not applicable)
    mode: str | None      # loaded mode name, or None


class WorkerHandle(ABC):
    """Uniform interface to ONE worker, regardless of locality."""

    @abstractmethod
    def start(self, resolved_mode: ResolvedModel, binding: LocalModelBinding,
              mode: ModeConfig) -> None:
        """Provision + load the worker. Blocks until READY."""

    @abstractmethod
    def submit(self, job: Job) -> Publisher:
        """Execute a job. Returns a backplane Publisher[Frame] the Governor
        correlates by job_id. The handle drives the JobSink (ack/progress/
        result/complete/error) and honors inbound subscription.cancel()."""

    @abstractmethod
    def health(self) -> WorkerHealth:
        """Liveness + readiness for admission/status reads."""

    @abstractmethod
    def unload(self) -> None:
        """Graceful teardown: clear caches, unregister from registry, free the
        worker. Today's `_unload_current_worker` path. In-proc = del + gc +
        empty_cache (the best empty_cache can do)."""

    @abstractmethod
    def stop(self) -> None:
        """Hard terminate. In-proc v1 this is the same as unload() (no process
        to kill). Facet-3 makes this the real recovery primitive: kill the
        subprocess to drop a poisoned CUDA context — recovery where
        empty_cache/unload cannot."""
```

**`unload()` vs `stop()`:** in v1 (in-proc) both are the same operation
(`_unload_current_worker`). The distinction is defined now so facet-3 doesn't
reshape the interface: `unload()` is graceful (clear + free), `stop()` is hard
kill (subprocess termination — the durable OOM recovery).

**Serialization discipline (D4):** the interface passes the `Job` object
verbatim. In-proc, nothing serializes. The `SubprocessWorkerHandle` (facet-3)
will serialize at its boundary — but that's the handle's concern, not the
Governor's or the interface's. The contract must not preclude this (no
unpicklable types in the `Job` *interface*), but v1 does not enforce it.

**What the handle does NOT own:** the queue, the epoch, the snapshot, the
admission decision. Those are the Governor's. The handle is told "run this job"
and reports back via the backplane stream.

### 3.3 The Governor

Owns every governance concern currently fused into `WorkerPool`:

| Concern | Today (WorkerPool) | Governor |
|---------|-------------------|----------|
| Job queue | `self.q` (`:275`) | `self.q` — unchanged |
| Resolution epoch | `self._resolution_epoch` (`:283`) | `self._resolution_epoch` |
| Active snapshot | `self._active_snapshot` (`:282`) | `self._active_snapshot` |
| State lock | `self._job_lock` (`:285`) | `self._job_lock` |
| Job records | `self._job_records` (`:284`) | `self._job_records` |
| Mode config DI | `self._mode_config` (`:294`) | `self._mode_config` |
| Registry DI | `self._registry` (`:295`) | `self._registry` |
| Idle config | `self._idle_timeout`/`_idle_check_interval` (`:287-288`) | same |
| Mode load (resolve + publish) | `_load_mode` (`:328-433`) | `_load_mode` — delegates worker build to `handle.start()` |
| Demand reload | `_reload_from_snapshot` (`:447-483`) | `_reload_from_snapshot` — delegates to `handle.start()` |
| Snapshot accessors | `get_active_model_snapshot`/`current_resolution_epoch` (`:435-445`) | same |
| Cancel (queued + running) | `cancel_job`/`cancel_pending_generation_jobs`/`_mark_running_generation_jobs_cancel_requested` (`:609-706`) | same |
| VRAM cleanup | `_cleanup_vram` (`:662-670`) | `_cleanup_vram` — delegates worker teardown to `handle.unload()` |
| Runtime status | `_build_runtime_status` (`:672-691`) | same |
| Idle watchdog | `_idle_watchdog_loop`/`_start_watchdog_thread` (`:543-590`) | same |
| Dispatch | `_worker_loop` (`:728-868`) | dispatch loop — **see §5 open question (a)** |
| Submit | `submit_job` (`:870-916`) | `submit_job` — registers job, enqueues (channel creation is the Handle's per §5(a); `_FutureBridge` attachment timing — pre-enqueue vs. post-`handle.submit()` in the dispatch loop — is part of open question (a), softened by the in-proc `_Channel`'s buffer-on-no-subscriber semantics). Epoch is stamped **caller-side** via `current_resolution_epoch()` at job construction (`ws_routes.py:626`); the Governor owns the authority + accessor, not the stamping. |
| Mode switch | `switch_mode` (`:918-937`) | same |
| Reload/unload/free | `reload_current_mode`/`free_vram`/`unload_current_model` (`:939-977`) | same |
| Accessors | `get_current_mode`/`is_model_loaded`/`reload_if_current`/`get_queue_size` (`:979-1010`) | same |
| Shutdown | `shutdown` (`:1012-1036`) | same |

### 3.4 InProcessWorkerHandle

Owns the threaded-worker coupling that today lives inside `WorkerPool`:

| Concern | Today (WorkerPool) | InProcessWorkerHandle |
|---------|-------------------|----------------------|
| Worker reference | `self._worker` (`:277`) | `self._worker` |
| Worker thread | `self._worker_thread` (`:278`), `_start_worker_thread` (`:529-541`) | owns thread + start |
| Worker factory | `self._worker_factory` (`:293`), `_default_worker_factory` (`:314-326`) | owns factory |
| Worker free | `_free_worker` (`:484-490`) | `_free_worker` |
| Worker unload | `_unload_current_worker` (`:492-527`) | `_unload_current_worker` (controlnet cache clear + unregister + free) |
| Job execution | `job.execute(self._worker)` inside `_worker_loop` (`:796`) | `submit(job)` → drives `JobSink` from `job.execute(worker)` |
| Backplane driving | sink.result/complete/error in `_worker_loop` (`:798-863`) | opens the `JobSink` inside `submit()`, drives it from `job.execute(worker)` (channel creation is the Handle's — see §5(a)) |

The handle's `submit(job)` is where the backplane (done) plugs in: the handle
opens the `JobSink`, runs `job.execute(worker)`, and emits
`ack → progress* → result/complete` (or `error`) through the sink. The Governor
correlates the returned `Publisher` by `job_id`. Cancel propagates inbound via
`subscription.cancel()` → `sink.cancelled` (the cross-process cancel channel
proven in the backplane's Task 5).

### 3.5 WorkerPool = facade

Every public method becomes a one-line delegation:

```python
# Re-export shared types so `from backends.worker_pool import GenerationJob`
# (ws_routes.py:621) stays unbroken. The types live in governor.py; this
# module only re-exports them.
from backends.governor import (
    GenerationJob, ModeSwitchJob, CustomJob, Job, JobType, JobRecord,
    ActiveModelSnapshot, StaleResolutionError, WorkerFactory,
)


class WorkerPool:
    """Compatibility facade over Governor. Transitional — deleted when routes
    migrate to the Governor directly (same pattern as the deferred progress→WS
    migration). Preserved so every test and caller stays green unmodified."""

    def __init__(self, queue_max=64, queue_timeout_s=DEFAULT_QUEUE_TIMEOUT_S,
                 worker_factory=None, mode_config=None, registry=None):
        self._governor = Governor(queue_max, queue_timeout_s,
                                  worker_factory, mode_config, registry)

    def submit_job(self, job, *, timeout_s=None) -> Future:
        return self._governor.submit_job(job, timeout_s=timeout_s)

    def get_active_model_snapshot(self): return self._governor.get_active_model_snapshot()
    def current_resolution_epoch(self): return self._governor.current_resolution_epoch()
    # ... every public method delegates ...
```

`get_worker_pool()` / `reset_worker_pool()` (`:1043-1105`) are unchanged — they
construct/reset a `WorkerPool`, which now internally constructs a `Governor`.

---

## 4. Seam inventory (folded in per D4)

### 4.1 CUDA-in-parent audit (recon; no code moves in v1)

Three touchpoints where the parent process touches CUDA directly, blocking it
from ever going CUDA-free (which blocks facet-3's spawn). **v1 documents these;
it does not move them** (that's the API-status-routing follow-on, §7).

| # | Location | What | v1 action | Follow-on |
|---|----------|------|-----------|-----------|
| 1 | `server/ws_routes.py:853-854` | `torch.cuda.mem_get_info()` inline in a status route — bypasses `ModelRegistry` | document | route through Governor → registry |
| 2 | `server/lcm_sr_server.py:218` | `torch.cuda.is_available()` capability gate | document | route through Governor |
| 3 | `server/superres_service.py` | **entire superres path** — a second, independent in-parent GPU consumer (device `cuda:0`, `is_available`, `empty_cache`, `OutOfMemoryError`). NOT behind `WorkerPool`: competes for VRAM outside the pool's accounting/recovery. | document | decision needed: superres shares the child, gets its own worker, or its own service |

The worker/child-side CUDA surface (`backends/cuda_worker.py`,
`model_registry.py`, `controlnet_cache.py`, `worker_pool.py` ~39 `torch.cuda`
refs) is already in the correct home — it moves with the handle, not the
Governor.

### 4.2 WorkerHandle contract (defined in v1)

§3.2. The contract is the v1 deliverable from the seam inventory. It must:
- Pass the `Job` object verbatim (no serialization in v1).
- Not preclude serialization (no unpicklable types in the interface surface).
- Return a backplane `Publisher` from `submit()` (the data-plane contract).
- Own worker thread + factory + teardown; NOT own queue/epoch/snapshot.

### 4.3 Deferred to facet-3 (NOT v1)

- **`ControlNetBinding` wire form**: serializing a job payload to cross a
  process boundary. In-proc, nothing serializes — dead weight in a pure
  in-proc extraction. The `GenerationJob.controlnet_bindings` field
  (`worker_pool.py:129`) carries resolved objects that may hold PIL/bytes; the
  wire form is a facet-3 concern.
- **`CustomJob`→typed-control-message map**: `CustomJob` (`:213-232`) carries a
  `Callable` that cannot cross a spawn boundary. It powers `_evict_if_idle`
  (`:708-726`, enqueued by the idle watchdog at `:581`). In-proc it works as-is
  (a callable run in-thread). Redesigning it to typed messages is only needed
  when a bound method can't pickle across spawn — i.e. facet-3. Touching it in
  v1 modifies working eviction code and jeopardizes the no-op proof.

---

## 5. Open implementation-shape questions (deferred to plan / RED prototyping)

These two questions interact and are deliberately **not resolved in this spec**.
They are diff-shape decisions that want live RED-first prototyping, not scoping
guesses. Recorded here (and on the FP comment) so the plan starts warm.

### (a) Governor dispatch loop: replace vs. wrap `_worker_loop`?

Today `_worker_loop` (`worker_pool.py:728-868`) is one monolithic loop that:
dequeues → handles mode-switch → checks cancel-skip → demand-reloads → runs the
stale-epoch barrier (`:783-794`) → executes → drives the sink/future → handles
OOM recovery.

The question: does the Governor's dispatch loop **replace** this (calling
`handle.submit(job)` and correlating the stream), or **wrap** it (the handle
keeps a `_worker_loop`-shaped thread, the Governor just enqueues into it)?

**Channel ownership (resolved, folds into (a)):** the **Handle owns backplane
channel creation** — `handle.submit(job)` opens the `JobSink` and returns the
`Publisher`. This is uniform with facet-3's IPC channel (the subprocess handle
opens its own connection + returns a `Publisher`; the Governor must not
difference locality by opening the channel itself). The **Governor subscribes
`_FutureBridge` to the returned `Publisher`** — so `submit_job` becomes:
register job → `publisher = handle.submit(job)` → `publisher.subscribe(
_FutureBridge(job.fut))` → enqueue (or the handle enqueues internally; that's
the (a) part). The in-proc `_Channel` buffers must-deliver frames and drains
them when a subscriber attaches, so subscribe-after-emit still resolves the
`Future` — the backplane plan's "hangs forever" wording only applies if
`_FutureBridge` is *never* attached. No test requires synchronous resolution
through `submit_job`. Stating this now so the plan doesn't rediscover it.

**Coupling:** this is tied to the stale-epoch barrier. Today the barrier runs
at the "last safe boundary" inside `_worker_loop` (`:779-794`), just before
`run_job`. Moving authority to the Governor could pull the check earlier (at
dispatch, before `handle.submit`) — which is only safe because the Governor owns
the epoch and can check atomically under `_job_lock`. But that interacts with
(b): if the Governor controls reloads, the "last safe boundary" reasoning
changes (a job stamped before eviction stays valid because the snapshot is
retained — `:771-777`).

**Lean:** leave open. RED-first prototyping decides.

### (b) Where does demand-reload / idle-eviction land?

**Lean (firm): Governor.** Lifecycle is the control plane's job — the FP issue
lists load/unload/reload/evict under the Governor, and the Governor owns the
snapshot that demand-reload reconstructs from (`_reload_from_snapshot`,
`:447-483`, reads `self._active_snapshot`). The handle just starts/runs/kills.

The idle watchdog (`_idle_watchdog_loop`, `:560-590`) enqueues a
`CustomJob(_evict_if_idle)` onto the queue; `_evict_if_idle` (`:708-726`) runs
on the worker thread to serialize with generation. If the Governor owns
lifecycle, the watchdog stays Governor-side; the handle's `unload()` is the
teardown primitive the Governor calls. This is consistent with D4's deferral of
the `CustomJob` redesign (the callable works in-proc; the Governor calls it).

**Why (a) and (b) interact:** if the Governor owns reloads (b), then the
demand-reload check at `:771-777` moves from `_worker_loop` to the Governor's
dispatch — which changes where the stale-epoch barrier (a) can safely run.
That coupling is exactly why they're plan decisions, not spec decisions.

---

## 6. The no-op proof (acceptance)

Mirrors the backplane's acceptance structure. v1 is a **pure extraction** —
zero observable behavior change.

### Acceptance criteria

1. **Existing suite green unmodified.** `test_worker_pool` (66 tests) +
   `test_ws_routes` + `test_model_routes` pass with **no test file edits**.
   Every `WorkerPool` public method still callable as-is.
2. **Zero caller diff.** `git diff server/ws_routes.py server/model_routes.py`
   = 0 bytes. No caller migrates to the Governor yet; `WorkerPool` is the
   facade.
3. **Physical separation.** Epoch/snapshot authority, queue, and dispatch live
   in `backends/governor.py`; threaded-worker coupling lives in
   `backends/worker_handle.py` (`InProcessWorkerHandle`).
4. **Handle pluggability.** A second `WorkerHandle` impl (a stub/test double)
   requires no change to `Governor` or `backends/backplane/` code. Proven by a
   test that injects a stub handle into the Governor — not a real subprocess
   (that's facet-3).
5. **Backplane integration preserved.** The `submit_job()→Future` facade
   (backplane Task 4, `_FutureBridge`) still works through the new structure —
   the handle drives the `JobSink`, the bridge fulfils the `Future`.

### What "green unmodified" pins

- `submit_job` still returns a `Future` (`:870-909`).
- `cancel_job` still returns `bool` (`:693-706`).
- `get_active_model_snapshot` still returns `Optional[ActiveModelSnapshot]`
  (`:435-438`).
- `current_resolution_epoch` still returns `int` (`:440-445`).
- `switch_mode`/`reload_current_mode`/`free_vram`/`unload_current_model` still
  return their current shapes (`:918-977`).
- `get_current_mode`/`is_model_loaded`/`reload_if_current`/`get_queue_size`/
  `shutdown` unchanged (`:979-1036`).
- `get_worker_pool()`/`reset_worker_pool()` singleton behavior unchanged
  (`:1043-1105`).
- The stale-epoch barrier still fires at the same logical point (whether it
  moves physically is question (a), but the *behavior* — a stale job raises
  `StaleResolutionError` before `run_job` — is preserved).
- OOM recovery (`_cleanup_vram` on `OutOfMemoryError`, `:823-836`) still runs.
- Idle eviction (watchdog → `CustomJob` → `_evict_if_idle`) still works.

---

## 7. Scope

### In scope

- `backends/governor.py`: the `Governor` class (queue, authority, dispatch,
  lifecycle, recovery).
- `backends/worker_handle.py`: `WorkerHandle` ABC + `InProcessWorkerHandle`.
- `backends/worker_pool.py`: reduced to a thin delegating facade; `Job` types,
  `ActiveModelSnapshot`, `StaleResolutionError`, `_FutureBridge`,
  `get_worker_pool`/`reset_worker_pool` stay (they're either pure data or
  singleton accessors).
- CUDA-in-parent audit (§4.1) documented in this spec — no code moves.
- `WorkerHandle` contract (§3.2/§4.2) defined.
- New unit tests for `Governor` + `InProcessWorkerHandle` isolation (handle
  pluggability, contract conformance). Existing suite green unmodified.

### Explicit non-goals (follow-on issues)

| Track | Why deferred |
|-------|-------------|
| Mode-switch race fixes (`STABL-ltefhpkk`, `STABL-iuiwzthc`) | Authority is now in one place, making the fix possible — but it's a *behavior change*, not an extraction. Separate issue after v1 proves the no-op. |
| API status/VRAM routing (remove inline `torch.cuda` in routes) | Pulls in seam-inventory territory (§4.1 touchpoints 1-2). Separate issue. |
| `superres` second-GPU-consumer decision (§4.1 touchpoint 3) | Needs its own design: share the child, own worker, or own service. Not blocking the Governor extraction. |
| facet-3 subprocess worker (`SubprocessWorkerHandle`) | Depends on Governor (this) + carries backplane facet-3 debts: `cancel_job→record.sink` subscription wiring, `STALE_EPOCH` consumer-injected reconstruction registry, IPC `request(n)`/`job_id`/`result()` hardening. |
| `ControlNetBinding` wire form | Dead weight in-proc (D4). Facet-3. |
| `CustomJob`→typed-control-message redesign | Touches working eviction code; jeopardizes no-op (D4). Facet-3. |
| Timed-out-job reap (`STABL-qvmdayhb`) | Needs `handle.stop()` = real kill (facet-3's durable recovery). Governor v1 has `stop()` but doesn't wire a reap timer. |
| Global GPU identity (`STABL-cchxvuhs`) | UUID-keyed allocation; not blocking single-GPU Governor path. |

### Follow-ups seeded

Mode-switch race fixes · API status routing · `superres` decision · facet-3
subprocess (`SubprocessWorkerHandle`) · `RemoteWorkerHandle` (microservice) ·
timed-out-job reap · global GPU identity.

---

## 8. Risks / mitigations

| Risk | Mitigation |
|------|------------|
| **Extraction introduces a behavior change** (the no-op breaks). | The backplane's no-op proof is the template: every public method signature preserved, `git diff` on routes = 0 bytes, existing suite green unmodified. TDD: RED on existing tests first (they must pass unchanged), then extract. |
| **`_worker_loop` is 140 LOC of intertwined logic** (dispatch + cancel + demand-reload + stale barrier + sink driving + OOM recovery). Splitting it across Governor/Handle risks reordering. | Question (a) is left open precisely because this is the diff-shape risk. The spec defines the *ownership boundary* (Governor owns dispatch+authority; Handle owns execution+sink-driving) without pre-resolving the loop's internal structure. RED-first prototyping validates before committing. |
| **Idle eviction via `CustomJob` callable** is in-proc-only and breaks if the Handle boundary hardens prematurely. | D4 defers the `CustomJob` redesign to facet-3. v1 keeps the callable working as-is; the Governor owns the watchdog + `_evict_if_idle`, the Handle's `unload()` is the teardown primitive. |
| **`_job_lock` acquisition across Governor/Handle boundary** could deadlock if the Handle acquires it (the backplane's `Subscriber↔lock` invariant forbids this). | The Handle drives the `JobSink` only — it never touches `_job_lock`. The Governor owns the lock and all record mutations. Same invariant as the backplane's `_FutureBridge` (which "touches ONLY the Future — never pool state or `_job_lock`"). |
| **Test isolation: `autouse` mock fixtures** may not transfer across the new module boundary (the backplane hit this — dropped a pool-level facade test for an isolated bridge test). | Expect the same: prefer isolated `Governor`/`Handle` unit tests over pool-level integration tests that depend on cross-module fixture imports. The 66 existing `test_worker_pool` tests are the integration proof. |

---

## 9. Build path

When `STABL-vdkdruox` opens for implementation:

1. **Fresh brainstorming session** (per the same discipline as the backplane) to
   resolve open questions (a) and (b) with RED-first prototyping.
2. **Implementation plan** via `writing-plans` →
   `docs/superpowers/plans/2026-07-25-worker-governor.md`.
3. **TDD tasks**, dependency-ordered, each reviewed green — same cadence as the
   backplane's 5-task plan.
4. **No-op landing** as a named task (like backplane Task 4): the moment
   `WorkerPool` becomes the facade and `git diff server/ws_routes.py` = 0 bytes.

**Not this spec.** This spec is the design; the plan is the next step after
user review.
