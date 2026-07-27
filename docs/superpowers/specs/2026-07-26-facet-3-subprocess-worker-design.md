# Facet-3: SubprocessWorkerHandle — Design

**FP:** `STABL-rgvxuedo` (child of umbrella `STABL-nvmieaxh`; seam work overlaps `STABL-qfjfflrx`).
**Depends on:** Backplane (`STABL-yoauoqao`, merged PR #19) — IPC data-plane transport. Worker Governor (`STABL-vdkdruox`, merged PR #20) — control plane + `WorkerHandle` interface.
**Status:** design, pre-plan.

---

## 1. Context & goal

The worker-as-a-service refactor exists for one reason — failure #3 in the forward
register: **OOM poisons the CUDA context; in-process recovery cannot fix it.**
`empty_cache()`/`del` cannot drop a poisoned context, and the per-process context
memory (~0.5–1.5 GB) is unreclaimable except by process exit. The durable fix is
**subprocess isolation: kill the poisoned process and respawn a clean one.**

This facet delivers that. It adds `SubprocessWorkerHandle` — a new `WorkerHandle`
implementation that hosts the real `CudaWorker` in a **spawn** subprocess (not fork —
CUDA contexts do not survive fork). The Governor flips its dispatch from reaching
into the in-proc worker to calling `handle.submit()`, and gains a kill+respawn
recovery path. This is the payoff the Backplane (transport) and Governor (control
plane) were built to enable.

---

## 2. Scope

### In (facet-3 v1)

- `SubprocessWorkerHandle` hosting a **real `CudaWorker`** in a spawn subprocess on a
  single GPU.
- Governor **dispatch-loop flip** to `handle.submit(job) → Publisher`, subscribing
  `_FutureBridge` to the returned Publisher (finding B's reserved "dispatch
  pluggability").
- **Durable OOM recovery**: in-band OOM error frame → kill → respawn → reload from
  the retained snapshot → next job succeeds.
- A **minimal frameless-death guard** (connection EOF / dead process) so the Governor
  never hangs on `fut.result()`.
- A **`LivenessSource` abstraction** behind `WorkerHandle.health()`, so heartbeat/liveness
  is relocatable (microservice / rsocket) with zero Governor change.
- **Minimal, versioned `GenerationJob` wire-form** for txt2img: `{req, job_id,
  resolution_epoch}`, `init_image=None`, `controlnet_bindings=[]`.
- `handle.stop()` = **real process kill**; `handle.health().state` state-driven.
- VRAM reported **from the child** (driver truth), consumed by the parent registry.

### Out (deferred — named siblings)

| Deferred | Owner / rationale |
|---|---|
| Multi-GPU / UUID identity | `STABL-cchxvuhs` — single-GPU path first |
| `superres_service.py` migration | Second in-parent GPU consumer; parent keeps its context **for superres only** until a later facet |
| Full `ControlNetBinding` wire-form | D4 defer; additive to the versioned wire-form |
| `init_image` transport | Additive (`BlobRef`) to the versioned wire-form |
| `CustomJob` → typed-control-message redesign | D4 — touches working eviction code |
| Timed-out-job **reap policy** | `STABL-qvmdayhb` — v1 delivers the `stop()`=kill primitive it needs; the policy is a follow-on |
| Mode-switch race fixes | `STABL-ltefhpkk` / `STABL-iuiwzthc` — authority is in the Governor now; independent follow-on |
| Full heartbeat-based liveness watcher | v1 ships the `LivenessSource` seam + EOF guard; periodic-heartbeat hardening is follow-on |

### Carried forward from the Backplane's facet-3 debts (Backplane plan Deferred)

- `cancel_job → record.sink` subscription wiring.
- `STALE_EPOCH` **consumer-injected reconstruction registry** (`code→factory`) — keeps
  control-plane types out of `frames.py`.
- IPC `request(n)` backpressure + `IpcJobSink(conn, job_id)` + `result()` signature
  hardening.

---

## 3. Architecture — what runs where

```
Parent (Governor process)                    Subprocess (spawn)
─────────────────────────                    ──────────────────
Governor                                      CudaWorker (owns the ONLY CUDA context)
  queue, epoch/snapshot authority             model load (from resolved+binding+mode)
  admission, dispatch → handle.submit()       run_job → drives JobSink
ModelRegistry + register/unregister           reports driver-truth VRAM (mem_get_info)
SubprocessWorkerHandle                        heartbeat / liveness producer
  spawn/kill/respawn, LivenessSource
  _FutureBridge on returned Publisher
NO CUDA context on the single-GPU path
```

- **Registry authority stays parent-side.** `register_model` / `unregister_model` is
  bookkeeping, not GPU work, and must survive a child kill. The kill→respawn cycle
  routes through the Governor's existing `_unload_current_worker` seam
  (`governor.py:_unload_current_worker`) — **unregister is Governor authority** — so
  kill → unregister → respawn → re-register reuses the seam Task 4 already wired, not
  a new one.
- **The parent holds no CUDA context** on the single-GPU facet-3 path. This is
  load-bearing: a parent-side context would reintroduce failure #2 (persistent
  unreclaimable context) and make the parent a *second* in-parent GPU consumer — the
  exact category being deferred for `superres`. See §8.

---

## 4. The boundary contracts

### 4.1 `start(resolved, binding, mode)` — spawn + load

Resolution/detection stays parent-side; the child receives an **already-resolved**
model and loads it (no re-detection). `ResolvedModel` + `LocalModelBinding` +
`ModeConfig` cross via the spawn pickle. The child calls the existing worker factory,
loads the model, and signals READY. `start()` blocks until READY (or fails).

### 4.2 `submit(job) → Publisher` — the versioned job wire-form

M1 wire-form is minimal: `{req, job_id, resolution_epoch}`, with `init_image=None`
and `controlnet_bindings=[]`. `fut` **never crosses** — it stays parent-side and is
fulfilled by `_FutureBridge` subscribed to the returned `Publisher`.

**Requirement — versioned from the start.** The wire-form reuses the Backplane's
leading `schema_version` byte discipline. Post-M1 additions (`init_image` as
`BlobRef`, `controlnet_bindings`) MUST be **additive**, never breaking. Do not ship an
unversioned M1 wire-form and retrofit versioning — that forces a migration.

**Requirement — `GenerateRequest` serialization is a M1 *prerequisite*, RED-first.**
The Backplane proved *result* transport (opaque bytes via `BlobRef`); the **request**
side has never crossed a boundary. `GenerateRequest` may contain fields that do not
pickle (file handles, tensors, lazy refs). **Before M1 implementation**, a RED-first
prototype MUST confirm `GenerateRequest` round-trips across the spawn boundary. If it
does not pickle cleanly, M1 adds an explicit serialization boundary
(`model_dump()` → dict → reconstruct). **This is the single biggest M1 schedule
risk** — it is pinned here so it is surfaced early, not discovered mid-M1.

---

## 5. Dispatch-loop flip (Milestone 1 — the biggest delta from Governor v1)

The Governor stops reaching into `self._handle.worker` and calls `handle.submit(job)`,
subscribing `_FutureBridge` to the returned `Publisher`. This dissolves the
reconciliation-#2 problem from Governor v1: the subprocess owns execution and cannot
share the parent's `_job_lock`, so the post-execute cancel-discard moves to the proven
Backplane path — `subscription.cancel()` → reverse control frame → `sink.cancelled` —
not a `_job_lock`-guarded read. This is exactly what Governor v1's finding B reserved
as "dispatch pluggability = facet-3."

**Cancel is eventually-consistent, by construction — state it.** In Governor v1 the
Governor controlled both `job.execute` return *and* the result emit, so it could read
`cancel_requested` under `_job_lock` between them and guarantee discard. In facet-3
the **child** emits the result frame; the parent cannot interpose a post-execute,
pre-emit check across the boundary. Therefore:

- cancel that **loses** the race to the result frame → the result lands (job completed
  before cancel arrived);
- cancel that **wins** → `sink.cancelled` (reverse control frame) suppresses the
  result.

This is *better* than v1 (no cross-boundary lock), but it is **eventually-consistent,
not point-in-time**. The spec states this so no future agent recreates the
`_job_lock`-guaranteed-discard illusion across a process boundary. These are the
Backplane's existing `subscription.cancel()` semantics; facet-3 inherits them.

---

## 6. OOM detection & recovery (Milestone 2)

Two paths, per decision: **(a) primary in-band + (b) minimal frameless-death guard.**

### (a) In-band OOM — the thesis path

GPU VRAM exhaustion raises `torch.cuda.OutOfMemoryError`, which is catchable. The
child's IPC connection is CPU-side and unpoisoned, so the child catches it and emits
`sink.error(BackplaneError.from_exc(e))` (the `OOM` code already exists in
`frames.py`). The handle's recovery path kills the child (its poisoned CUDA context
dies with the process), respawns, reloads from the retained snapshot, and the next job
succeeds. After emitting OOM the child MUST NOT run another job — but this is a
**contract enforced by the parent**, not the child: the parent kills the child before
dispatching the next job. That kill is **racy with the dispatch loop** — if the parent
is slow and a second job is already in the child, the poisoned context OOMs again
immediately. The design **tolerates double-OOM** (idempotent: the second OOM → the same
kill), so the "MUST NOT" is a statement of who enforces it, not a guarantee the child
self-polices.

### (b) Minimal frameless-death guard — the safety net

A child can die **without** sending a frame: a context so poisoned the next CUDA call
segfaults, a hard crash, or (rarely) the Linux OOM-killer's SIGKILL on *system-RAM*
exhaustion (a different failure than VRAM OOM — the full watcher for it is deferred).
The parent must never deadlock on `fut.result()`. The EOF/dead-process guard splits
into **two cases**:

| Child dies… | Outstanding `fut`? | Guard action |
|---|---|---|
| **mid-job** | yes | synthesize a failure terminal for the in-flight `fut`; kill (already dead) → respawn |
| **between jobs (idle)** | no | mark `dead`; respawn on demand-reload / next dispatch |

**During the dead window** (post-OOM/crash, pre-respawn) the registry MUST report
**conservative "0 free"** for that worker, so the Governor does not admit a job against
the dead child's *stale last heartbeat*.

### Recovery routes through the existing seam

Kill → respawn reuses `Governor._unload_current_worker` (registry `unregister_model` =
Governor authority) → `start()` re-loads → `register_model`. No new teardown seam.

---

## 7. Liveness / heartbeat — abstracted for relocation

**Requirement (operator):** heartbeat logic must be abstracted so it can be relocated
later — to a microservice, or backed by rsocket — without touching the Governor.

Design: a **`LivenessSource`** protocol, **owned by the handle, consumed by the
Governor only through `WorkerHandle.health()`**. The Governor never knows *how*
liveness is determined; it reads `health().state`.

- **Subprocess impl:** periodic heartbeat frame on the IPC channel + EOF detection.
  Stale heartbeat or EOF ⇒ `state = "dead"`.
- **Rsocket-ready seam:** when the transport becomes rsocket, its built-in KEEPALIVE
  frames back the same `LivenessSource` with **zero Governor change** — the heartbeat
  becomes a no-brainer. A future `RemoteWorkerHandle` (microservice) reuses the
  identical contract.

The invariant: heartbeat/liveness lives behind the `LivenessSource` seam — never
inline in the Governor, never hardcoded to IPC.

---

## 8. VRAM accounting moves to the child

With no CUDA context in the parent, `mem_get_info` (STABL-sqqlkmdl) cannot run
parent-side. VRAM is reported **from the child** and consumed by the parent registry.

**Rejected alternative — a thin parent-side CUDA context "just for accounting."** It
reintroduces failure #2 (persistent unreclaimable context) *and* makes the parent a
second in-parent GPU consumer — the very category deferred for `superres`. Deferring
superres-as-second-consumer while adding a new one is a contradiction. Rejected firmly.

### 8.1 `vram_bytes` semantics — resolve the v1 dataclass mismatch now

Governor v1's `WorkerHealth.vram_bytes` is documented "current allocated VRAM" and
implemented as `int(torch.cuda.memory_allocated())` — the **torch allocator**, *not*
the `mem_get_info` **driver truth** that STABL-sqqlkmdl established. Left as-is, the
registry's admission math would diverge from the accounting fix.

**Resolution:** `WorkerHealth` reports **driver-truth** VRAM via `mem_get_info`,
replacing the ambiguous `vram_bytes` with:

- `vram_free_bytes` — driver-truth free (what admission needs);
- `vram_total_bytes` — driver-truth total.

`InProcessWorkerHandle.health()` and the test `StubHandle` are updated to match (small,
low-risk — `WorkerHealth` VRAM fields are consumed only by tests today). This aligns
the health contract with STABL-sqqlkmdl.

**Sequencing note:** this touches *merged, shipped* Governor v1 code (`worker_handle.py`),
not new facet-3 code. The plan MUST land the `WorkerHealth` field change as an **early
task, before M1**, so the health contract is settled before `SubprocessWorkerHandle`
implements it — not discovered when the subprocess handle tries to fill the new fields.

### 8.2 Advisory for admission, authoritative for status

Heartbeat-piggybacked VRAM is **eventually-consistent**, not point-in-time. The
Governor may admit a job against slightly stale VRAM and the child may OOM anyway —
**which is fine, because M2's kill+respawn is the recovery.** The spec states this so a
future agent does not try to make admission "perfect" with a synchronous VRAM
round-trip that defeats the async heartbeat. **Stale-admit → OOM → respawn is the
designed path, not a bug.** (Exception: the dead-window "0 free" of §6 is authoritative
— never admit against a dead child.)

---

## 9. Governor changes — finding B, realized

Facet-3 is where Governor v1's finding B ("v1 proves lifecycle pluggability, not
dispatch pluggability") is discharged. The plan MUST enumerate these explicitly so
they are not discovered mid-M1:

1. **Dispatch flip** (§5): `handle.submit()` + `_FutureBridge` subscription, replacing
   the in-proc `job.execute(self._handle.worker)` path.
2. **Liveness-read flip:** there are exactly **three** `self._handle.worker is None`
   reads (`governor.py:529`, `:551`, `:594`). `SubprocessWorkerHandle.worker` returns
   `None` (no in-proc worker), so these flip to `health().state == "dead"` — the
   cleanup Governor v1's docstring reserved. This *is* the "Governor dispatch changes"
   finding B named. **Preserve the semantic split — do not flatten all three to one
   expression:** `:529` (idle watchdog, "is there a worker to evict?") and `:551`
   (`_evict_if_idle`, "already unloaded?") mean *dead ⇒ nothing to evict / skip*;
   `:594` is a **demand-reload trigger**, not a liveness check — *dead ⇒ respawn, then
   run this job*. The plan must keep `:594`'s reload semantics distinct when flipping
   the predicate.
3. **Kill path via the existing seam** (§6): kill → `_unload_current_worker`
   (unregister) → respawn → re-register. Reuse, do not re-invent.

---

## 10. Milestones → plan tasks

- **M0 (prerequisite, RED-first):** confirm `GenerateRequest` round-trips across the
  spawn boundary (§4.2). Gates M1 scope.
- **M1 — dispatch flip + real subprocess, succeeding job:** `SubprocessWorkerHandle`
  hosts a real `CudaWorker`; Governor flips to `handle.submit()`; one txt2img job
  (`init_image=None`, `controlnet_bindings=[]`) succeeds end-to-end. Proves the
  contract change against the real worker **before** OOM is in the picture — a wrong
  contract surfaces as a clean failing job, not a poisoned context.
- **M2 — OOM kill + respawn:** a job that OOMs → child killed → respawned → reload from
  snapshot → next job succeeds. Plus the frameless-death guard (§6, both cases). The
  thesis proof.

---

## 11. Testing strategy

- Real spawn-boundary tests, reusing the Backplane's `shared_memory` /
  `resource_tracker`-unregister patterns.
- A **fault-injecting worker**: raises `torch.cuda.OutOfMemoryError` on command (M2
  path a), and a variant that **dies frameless** (M2 path b, both mid-job and idle).
- Assert **next-job-succeeds after respawn** — the durable-recovery invariant.
- Assert the dead-window registry reports "0 free" and blocks admission.
- Assert cancel both ways: wins the race (suppressed) and loses (result lands) — §5
  eventual consistency.
- `GenerateRequest` round-trip test (M0).

---

## 12. Acceptance criteria

1. A txt2img job runs end-to-end through `SubprocessWorkerHandle` on a spawn
   subprocess (M1).
2. A job that OOMs is followed by a **successful** job after automatic kill+respawn —
   no manual intervention, no persistent poisoned context (M2).
3. A frameless child death (mid-job and idle) does not hang the Governor; the in-flight
   `fut` fails and the worker respawns (M2).
4. `health().state` drives all Governor liveness reads; no Governor code reaches into a
   subprocess worker; the `LivenessSource` seam is transport-agnostic (§7).
5. VRAM is driver-truth (`mem_get_info`) reported from the child; the parent holds no
   CUDA context on the single-GPU path (§8).
6. The existing suite stays green; `server/` route diffs remain 0 bytes unless a
   caller intentionally migrates (the dispatch flip is Governor-internal).

---

## 13. Forward / non-goals recap

Multi-GPU (`cchxvuhs`), `superres` migration, full `ControlNetBinding` + `init_image`
wire-forms, `CustomJob` redesign, reap policy (`STABL-qvmdayhb`), and the mode-switch
race fixes (`ltefhpkk`/`iuiwzthc`) are explicit non-goals here, each tracked to its
sibling. Facet-3 v1 delivers the durable-OOM-recovery thesis on the single-GPU txt2img
path and the `SubprocessWorkerHandle` contract the rest build on.
