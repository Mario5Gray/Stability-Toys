# Backplane: data-plane transport for the worker output/progress stream

**FP:** STABL-yoauoqao (child of umbrella STABL-nvmieaxh)
**Depends on:** STABL-sqqlkmdl (VRAM accounting flip, merged `243455e`)
**Consumed by:** STABL-vdkdruox (Worker Governor, control plane) · seam inventory STABL-qfjfflrx
**Status:** design accepted, pre-implementation
**Date:** 2026-07-24

---

## 1. Purpose

Abstract the worker's progress/output stream behind a **backplane** seam so the
worker boundary — today the in-process job queue — can carry streaming progress
**and** the final result across `same-process → subprocess → microservice`
**without changing callers**.

This is the **data plane**. Lifecycle, state, and authority (resolution epoch,
active snapshot, admission barrier) are the **control plane** and stay parent-side
in the Governor (STABL-vdkdruox). The backplane carries data only.

### The reframe that drives the design

Today's data-plane transport **is a `concurrent.futures.Future`**:

- Parent builds `GenerationJob` (carries `job.fut`) → worker loop runs
  `job.execute(worker)` → `job.fut.set_result((png_bytes, seed))` /
  `set_exception(e)` (`backends/worker_pool.py:688-809`).
- Parent consumes via `fut.result(timeout)` in a thread executor
  (`server/ws_routes.py:535`).
- Cancel is inbound and out-of-band: `cancel_job(job_id)` flips
  `cancel_requested`; the worker loop checks it at boundaries
  (`worker_pool.py:718`, `:758`).
- **No gen progress stream exists.** The `Future` is one terminal value.
  (`job:progress` at `ws_routes.py:476` is the CHAT path, not generation.)

The `Future` is a **degenerate stream**: zero progress frames, one terminal. The
backplane generalizes it to `ack → progress* → terminal[result|error]` plus an
inbound cancel. So the backplane must (a) carry result/error for generation **now**,
and (b) **newly enable** per-step generation progress — progress is net-new, *not a
current behavior to preserve*.

---

## 2. Decisions (settled in brainstorm)

| # | Axis | Decision |
|---|------|----------|
| D1 | Interface shape | **Full reactive-streams from day one, in-proc included.** Program both sides to rsocket-py's `reactivestreams` `Publisher`/`Subscriber`/`Subscription` ABCs so the future rsocket transport is a literal drop-in. |
| D2 | Result payload | **Transport-resolved `BlobRef`.** `Result{seed, image: BlobRef}`; the wire schema never changes across transports. In-proc → bytes; IPC → shared-mem; rsocket → fragment reassembly (later). |
| D3 | Backpressure | **Split policy.** `Progress` conflates (onBackpressureLatest — GPU worker never blocks on a slow client); `Result` + `onComplete`/`onError` are must-deliver. |
| D4 | No-op landing | **Backplane lands under a preserved `submit_job() -> Future` facade.** `ws_routes.py` gets zero diff; existing suite green unmodified = the no-op proof. Progress→WS is a deliberate follow-up. |
| D5 | Cross-process transport | **Build the stdlib IPC/shared-mem transport now** (`multiprocessing.Connection` frames + `shared_memory` payload) and test it across a real process boundary. **rsocket transport is a follow-up child** — the interface already fits it. |

---

## 3. Architecture

### 3.1 Package layout — `backends/backplane/`

| Module | Owns |
|--------|------|
| `frames.py` | Frame dataclasses (`Ack`, `Progress`, `Result`), `BackplaneError` + code enum, `BlobRef` ABC |
| `blob.py` | `BlobRef` impls (`InProcBlob`, `SharedMemBlob`) + payload/frame codec |
| `interface.py` | `Backplane` protocol + producer-side `JobSink`; typed on rsocket-py `reactivestreams` ABCs |
| `inproc.py` | In-proc transport: anyio-backed `Publisher`, thread→loop bridge, conflating operator |
| `ipc.py` | Stdlib cross-process transport: `Connection` frames + `shared_memory` payload |

### 3.2 Two-sided interface

The worker (producer) programs to a `JobSink`; the parent (consumer) programs to a
reactive `Publisher[Frame]`. Neither imports a transport — a `Backplane` factory
returns a `(sink, publisher)` pair whose wiring is transport-specific.

```
worker thread ──JobSink──▶ [transport] ──Publisher[Frame]──▶ parent (asyncio)
   sink.ack()                                                   onNext(Ack)
   sink.progress(step,total)   conflate ─────────────────────▶ onNext(Progress)   (dropped by facade for now)
   sink.result(seed, BlobRef)  must-deliver ─────────────────▶ onNext(Result)
   sink.complete()/error(e)                                     onComplete / onError
   sink.cancelled ◀──────────── subscription.cancel() ◀──────  (inbound)
```

### 3.3 Substrate (from D1)

- Add **`rsocket` (rsocket-py)** as a pinned dependency; program to its
  `reactivestreams` ABCs. (`anyio` is already present via Starlette.)
- In-proc `Publisher` is backed by an **`anyio` memory-object stream** with an
  **`anyio.from_thread`** bridge — the worker-thread→asyncio-consumer crux, the
  reactive generalization of today's `run_in_executor(fut.result)`.

---

## 4. Wire contract

### 4.1 Frames (onNext payloads)

```python
@dataclass(frozen=True)
class Ack:       job_id: str; queued_position: int          # job entered the queue
@dataclass(frozen=True)
class Progress:  job_id: str; step: int; total: int; stage: str = "denoise"   # net-new
@dataclass(frozen=True)
class Result:    job_id: str; seed: int; image: BlobRef
```

`total` unknown at emit time is allowed as `total = -1`; the consumer treats a
negative total as indeterminate.

### 4.2 Terminals come from the reactive protocol, not a frame type

- `onComplete()` — emitted immediately after `Result` on a clean run.
- `onError(BackplaneError)` — carries a code enum plus message + optional
  original-exception repr:

  ```
  BackplaneErrorCode = OOM | STALE_EPOCH | CANCELLED | TIMEOUT | GENERIC
  ```

  In-proc wraps the live exception; IPC reconstructs from the code. Maps 1:1 onto
  today's `fut.set_result((png,seed))` / `fut.set_exception(e)`; the
  `StaleResolutionError`, `CancelledError`, and OOM paths at
  `worker_pool.py:756-804` each get a code.

### 4.3 Ordering, correlation, cancel

- One job = one `Publisher` = one stream. `job_id` on every frame. Ordered per
  stream by construction (maps to an rsocket requestStream id later).
- **Inbound cancel** = `subscription.cancel()` → sets `sink.cancelled`. The worker
  loop checks it at today's exact boundaries (`worker_pool.py:718`, `:758`), so the
  two pinned cancel semantics — queued job ⇒ future cancelled, running job ⇒ late
  result discarded — are preserved verbatim.

### 4.4 BlobRef

Transport-resolved payload handle. Consumer always does `png = await
frame.image.read()`.

- `InProcBlob` → returns the held `bytes` (no-op unwrap).
- `SharedMemBlob(name, size)` → maps + reads the `shared_memory` block.

Schema identical on every transport.

### 4.5 Backpressure operator (D3)

Sits between the worker-producer and the subscriber:

- `Progress` → **conflate** (onBackpressureLatest): demand exhausted ⇒ keep newest,
  drop stale. The GPU worker thread never blocks on a slow client.
- `Result` + `onComplete`/`onError` → **must-deliver**: buffered until `request(n)`,
  never dropped.
- `Ack` → must-deliver (single cheap frame).

---

## 5. In-proc transport + Future facade (the no-op landing, D4)

`WorkerPool.submit_job(job) -> Future` is **kept**. Internally it now:

1. Registers the job (unchanged).
2. Creates a backplane channel via the in-proc `Backplane` → `(sink, publisher)`.
3. Attaches a **compat Subscriber** to `publisher` that fulfills `job.fut`:
   - `Result` → `fut.set_result((png_bytes, seed))` (BlobRef read is an in-proc
     unwrap)
   - `onError` → `fut.set_exception(mapped_exc)` (code enum → concrete exception)
   - `onComplete` → no-op
   - `Progress` → **dropped** (subscriber requests zero progress demand)
4. Returns `job.fut`.

**Worker loop change is purely internal** (`worker_pool.py:688-809`): where it now
calls `job.fut.set_result/set_exception`, it drives the `sink` instead —
`sink.ack()` at run start; `sink.result(seed, blob); sink.complete()` on success;
`sink.error(BackplaneError.from_exc(e))` on failure — checking `sink.cancelled` at
the two existing cancel boundaries. The compat Subscriber turns those frames back
into identical Future outcomes.

`cancel_job(job_id)` maps onto `subscription.cancel()`; the queued-vs-running branch
logic (`worker_pool.py:653`) is retained.

### 5.1 No-op proof

These pass **with zero edits** (the empty `ws_routes.py` diff *is* the proof):

- `tests/test_ws_routes.py` — `pool.submit_job.return_value = fut; fut.result...`
  stubs (`:303`, `:400`, `:492`, `:590`, `:1090`) still describe the real facade.
- `tests/test_worker_pool.py` — `TestJobSubmission`, `TestErrorHandling`,
  `TestOomRecovery`, cancel tests (`:514`, `:546`), `TestConcurrency`.
- `tests/test_model_routes.py` — routes untouched.

**New backplane tests (added, not modified):** in-proc Publisher delivers
`ack→result→complete` in order; `onError` code round-trips to the right exception;
conflating operator drops stale `Progress` under zero demand while `Result` is never
dropped; `subscription.cancel()` sets `sink.cancelled`.

---

## 6. Stdlib IPC/shared-mem transport (D5)

Same interface, different wiring. The `ipc` `Backplane` hands the producer a
`JobSink` and the consumer a `Publisher[Frame]` — identical types to in-proc.
Worker and parent code are byte-for-byte unchanged; only the factory differs.

**Two channels:**
- **Frames** (small: `Ack`, `Progress`, `Result`-metadata, error, terminal markers)
  ride a `multiprocessing.Connection` (Pipe). The codec serializes each frame to a
  tagged record via explicit `to_wire`/`from_wire` — **not** blanket pickle — so the
  wire schema is auditable and version-stable.
- **Payload** (the PNG) rides `multiprocessing.shared_memory`. `sink.result(seed,
  blob)` allocates a `SharedMemory` block, writes the bytes; the `Result` frame
  carries `SharedMemBlob(name, size)`. Consumer `await blob.read()` maps, copies out,
  then unlinks.

### 6.1 BlobRef lifecycle (the sharp edge)

- Producer creates + writes; frame carries `(name, size)`.
- Consumer reads once, then `close()` + `unlink()` — single-consumer, read-once.
- On `onError` / `subscription.cancel()` before the consumer reads: a **reaper** on
  the producer side unlinks the orphan when the stream terminates. No segment
  outlives its stream.

### 6.2 Backpressure across the pipe

`request(n)` demand rides a tiny control frame on a reverse `Connection`. The
conflating operator lives on the **producer** side, so stale `Progress` is dropped
before it crosses the pipe (progress is metadata-only; never allocate shared-mem for
it). `Result`/terminal block on demand.

### 6.3 Boundary test (satisfies acceptance)

Spawn a dummy worker **process** (`multiprocessing`, **`spawn`** start method —
matching the CUDA-safe posture) that drives a `JobSink` through a scripted
`ack → progress×N → result(bytes) → complete`. The parent subscribes to the
`Publisher` and asserts:

- frames arrive ordered,
- PNG bytes round-trip through shared-mem intact,
- a mid-stream `subscription.cancel()` reaches the child, it stops, and the segment
  is unlinked.

This proves the seam across a real process boundary **without** moving CudaWorker to
a subprocess (that is facet 3).

---

## 7. Scope

### In scope
- `backends/backplane/` package: frames, BlobRef, two-sided interface on rsocket-py
  reactivestreams, in-proc transport, stdlib IPC transport.
- `WorkerPool.submit_job`/`cancel_job` reworked to drive the backplane behind the
  preserved Future facade — zero `ws_routes.py` diff.
- New backplane unit tests + the process-boundary IPC test. Existing suite green
  unmodified.
- `rsocket` pinned in `requirements.txt` (`anyio` already present).

### Explicit non-goals (belong to named siblings)
- Moving CudaWorker to a subprocess — **facet 3** (depends on this).
- The **Governor** (STABL-vdkdruox): lifecycle, epoch/snapshot authority, admission
  barrier stay parent-side, untouched here.
- **rsocket transport impl** — follow-up child; interface already fits (drop-in).
- **Progress → WS** wiring + real per-step emission from CudaWorker
  (`callback_on_step_end`) — deliberate follow-up; capability lands dormant here.
- Inbound **job-payload** serialization (`ControlNetBinding` wire form, `CustomJob`
  callable → typed control messages) — shared with the seam inventory / Governor.
  This issue serializes only a **minimal job** for the boundary test and the
  **outbound** frame stream. `superres` second GPU consumer untouched.

### Follow-ups seeded
rsocket-over-UDS transport child · progress-to-WS child · facet-3 subprocess
(consumes both transports) · Governor extraction.

---

## 8. Risks / mitigations

| Risk | Mitigation |
|------|-----------|
| rsocket-py transitive weight | Verify footprint before pinning (Compel `--no-deps` precedent). If heavy, **vendor just the `reactivestreams` ABCs** (3 tiny files) instead of the whole package. **Plan step 1.** |
| rsocket-py `reactivestreams` API is asyncio-shaped and its exact module surface is unverified (not installed) | **Plan step 1 confirms** real import paths before building on them; fallback is the vendored ABCs. |
| thread→asyncio bridge correctness | In-proc no-op hinges on `anyio.from_thread` delivering frames without reorder/deadlock under the worker lock; covered by ordering + concurrency tests. |
| shared-mem orphans on macOS/dev | `resource_tracker` warnings + leaked segments; reaper + read-once-unlink discipline tested explicitly; boundary test asserts no surviving segment. |

---

## 9. Acceptance criteria (from the FP issue)

1. A backplane interface with at least the in-proc transport, wired so WS
   progress/result/error are delivered through it with **no observable change** to
   current clients. → §5 (Future facade, empty `ws_routes.py` diff).
2. A shared-mem/IPC transport behind the **same interface**, exercised across a real
   process boundary in a test. → §6.
3. Worker and WS-route code reference **only** the backplane interface (no direct
   callback wiring, no transport imports). → §3.2, §5 (factory returns
   `(sink, publisher)`; transport selected by config).
