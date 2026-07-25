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
| D1 | Interface shape | **Full reactive-streams from day one, in-proc included.** Program both sides to **vendored** `Publisher`/`Subscriber`/`Subscription` ABCs (~150 LOC under `backends/backplane/reactivestreams/`) so the interface is rsocket-*shaped* without a runtime dependency on the pre-1.0 rsocket-py package. We control method naming (Python snake_case: `on_next`/`on_complete`/`on_error`, `request`, `cancel`). The rsocket transport follow-up pins rsocket-py and adapts to its surface then. *(Revised from "pin rsocket-py": the interface must not couple to a pre-1.0 ABC surface; vendoring 3 tiny files kills the "unverified module surface" risk. See §9.)* |
| D2 | Result payload | **Transport-resolved `BlobRef`.** `Result{seed, image: BlobRef}`; the wire schema never changes across transports. In-proc → bytes; IPC → shared-mem; rsocket → fragment reassembly (later). |
| D3 | Backpressure | **Split policy.** `Progress` conflates (onBackpressureLatest — GPU worker never blocks on a slow client); `Result` + `on_complete`/`on_error` are must-deliver. The conflating operator is **dormant in-proc** (compat Subscriber requests unbounded) and **active for IPC** (demand crosses a pipe). |
| D4 | No-op landing | **Backplane lands under a preserved `submit_job() -> Future` facade.** `ws_routes.py` gets zero diff; existing suite green unmodified = the no-op proof. Progress→WS is a deliberate follow-up. |
| D5 | Cross-process transport | **Build the stdlib IPC/shared-mem transport now** (`multiprocessing.Connection` frames + `shared_memory` payload) and test it across a real process boundary. **rsocket transport is a follow-up child** — the interface already fits it. |

---

## 3. Architecture

### 3.1 Package layout — `backends/backplane/`

| Module | Owns |
|--------|------|
| `reactivestreams/` | **Vendored** `Publisher`/`Subscriber`/`Subscription` ABCs (~150 LOC, snake_case surface). No runtime rsocket-py dep. |
| `frames.py` | Frame dataclasses (`Ack`, `Progress`, `Result`), `BackplaneError` + code enum + `code → exception_factory` table, `BlobRef` ABC |
| `blob.py` | `BlobRef` impls (`InProcBlob`, `SharedMemBlob`) + payload/frame codec (`schema_version`-tagged) |
| `interface.py` | `Backplane` protocol + producer-side `JobSink`; typed on the vendored ABCs |
| `inproc.py` | In-proc transport: anyio-backed `Publisher`, thread→loop bridge, conflating operator |
| `ipc.py` | Stdlib cross-process transport: `Connection` frames + `shared_memory` payload |

### 3.2 Two-sided interface

The worker (producer) programs to a `JobSink`; the parent (consumer) programs to a
reactive `Publisher[Frame]`. Neither imports a transport — a `Backplane` factory
returns a `(sink, publisher)` pair whose wiring is transport-specific.

```
worker thread ──JobSink──▶ [transport] ──Publisher[Frame]──▶ parent (asyncio)
   sink.ack()          (non-blocking)                           on_next(Ack)
   sink.progress(...)  (non-blocking, conflatable) ───────────▶ on_next(Progress)  (dropped by facade for now)
   sink.result(seed, BlobRef)  (sync, must-deliver) ─────────▶  on_next(Result)
   sink.complete()/error(e)    (sync, must-deliver)             on_complete / on_error
   sink.cancelled ◀──────────── subscription.cancel() ◀──────  (inbound)
```

### 3.3 Substrate (from D1)

- **Vendor** the `Publisher`/`Subscriber`/`Subscription` ABCs (~150 LOC) under
  `backends/backplane/reactivestreams/`; program both sides to them. **No runtime
  rsocket-py dependency** — the interface is rsocket-*shaped*, not rsocket-*coupled*.
  Method surface is Python snake_case (`on_next`/`on_complete`/`on_error`,
  `request(n)`, `cancel()`). (`anyio` is already present via Starlette.)
- In-proc `Publisher` is backed by an **`anyio` memory-object stream** with an
  **`anyio.from_thread`** bridge — the worker-thread→asyncio-consumer crux, the
  reactive generalization of today's `run_in_executor(fut.result)`.

**Sink call semantics (pinned — the no-op depends on it):**

- `sink.ack(...)` / `sink.progress(...)` — **fire-and-forget, non-blocking**. The
  worker thread never parks on these; they are conflatable telemetry. A slow or
  stalled consumer cannot back-pressure into the GPU worker through them.
- `sink.result(...)` / `sink.complete()` / `sink.error(...)` — **synchronous,
  must-deliver**: the worker blocks until the frame is accepted by the transport
  (handed to the consumer side), preserving today's ordering where `set_result`
  hands off before the worker moves on. Never dropped.

**Compat Subscriber demand protocol (in-proc):** requests **unbounded** up front.
Backpressure is therefore off in-proc (correct — the facade's `Future` is
single-valued), so the conflating operator is **dormant** in-proc and only engages
for the IPC transport where `request(n)` demand crosses a pipe (§6.2).

**Lock invariant:** the compat Subscriber touches **only `job.fut`** — it must never
acquire `_job_lock` or call pool methods. `cancel_job` keeps its direct `fut.cancel()`
**under `_job_lock`**; `subscription.cancel()` sets `sink.cancelled`, a plain flag
requiring no lock. This forecloses a loop↔worker lock inversion.

---

## 4. Wire contract

### 4.1 Frames (`on_next` payloads)

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

- `on_complete()` — emitted immediately after `Result` on a clean run.
- `on_error(BackplaneError)` — carries a code enum plus message, **and, in-proc, the
  live exception instance itself**:

  ```
  BackplaneErrorCode = OOM | STALE_EPOCH | CANCELLED | GENERIC
                       | TIMEOUT   # RESERVED — not emitted by this issue (see below)
  ```

**Exception preservation is a hard invariant (not implied).** The in-proc transport
carries the **original exception instance** unwrapped; the compat Subscriber calls
`fut.set_exception(err.original)`, never `fut.set_exception(BackplaneError(...))`.
This is load-bearing: [test_worker_pool.py:474](../../../tests/test_worker_pool.py)
does `pytest.raises(fake_oom)` on the concrete `torch.cuda.OutOfMemoryError`, and
[:1423] does `pytest.raises(StaleResolutionError)` — a `BackplaneError` substitution
fails both. The **code enum is IPC-reconstruction metadata only**.

**`code → exception_factory` table** (used by IPC, where the live instance cannot
cross the boundary; in-proc prefers the carried instance):

| code | reconstructs |
| --- | --- |
| `OOM` | `torch.cuda.OutOfMemoryError` |
| `STALE_EPOCH` | `StaleResolutionError` |
| `CANCELLED` | `concurrent.futures.CancelledError` |
| `GENERIC` | `RuntimeError` (message-only) |

This maps 1:1 onto today's `fut.set_exception(e)` paths at `worker_pool.py:756-804`.

**`TIMEOUT` is reserved, not emitted this issue.** Timeout lives consumer-side today
(`ws_routes.py:531`, `fut.result(timeout=120)`); the worker has no knowledge of the
consumer's deadline and no code path produces it. It is reserved in the enum for the
Governor watchdog / abandoned-job reap (STABL-qvmdayhb), which will emit it when that
lands.

### 4.3 Ordering, correlation, cancel

- One job = one `Publisher` = one stream. `job_id` on every frame. Ordered per
  stream by construction (maps to an rsocket requestStream id later).

**Cancel has two distinct sides, and every entry point keeps its direct
`fut.cancel()`.** `subscription.cancel()` is *additive* — it arms the worker thread;
it does **not** replace the synchronous `fut.cancel()` the consumer contract depends
on. "Cancel maps onto `subscription.cancel()`" (the original phrasing) was too coarse
and would break synchronous `fut.cancelled()` assertions.

| Entry point | Consumer side (synchronous, kept) | Worker side (added) |
| --- | --- | --- |
| `cancel_job` queued branch (`worker_pool.py:661`) | `record.job.fut.cancel()` **under `_job_lock`** — `fut.cancelled()` true on return ([test:537-538](../../../tests/test_worker_pool.py)) | — (job never runs) |
| `cancel_job` running branch (`:665`) | sets `cancel_requested` | `sink.cancelled` armed; worker discards **producer-side** (§5) |
| `cancel_pending_generation_jobs` (`:598`) | `job.fut.cancel()` directly | — ([test:477](../../../tests/test_worker_pool.py), [:504]) |
| `_mark_running_generation_jobs_cancel_requested` (`:560+`) | `cancel_requested` → later `CancelledError` | `sink.cancelled` armed |

The two pinned semantics — queued ⇒ future synchronously cancelled, running ⇒ late
result discarded — are preserved because the direct `fut.cancel()` calls stay exactly
where they are; the backplane only *adds* the worker-thread signal.

### 4.4 BlobRef

Transport-resolved payload handle. The ABC is **two methods**:

```python
class BlobRef(ABC):
    async def read(self) -> bytes: ...   # materialize the payload
    def close(self) -> None: ...          # release the handle
```

- `InProcBlob` → `read()` returns the held `bytes`; `close()` is a no-op.
- `SharedMemBlob(name, size)` → `read()` maps + copies out of the `shared_memory`
  block; `close()` does `SharedMemory.close()` + `unlink()`.

**The consumer MUST `close()` after `read()`** (a context manager is the intended
form: `async with frame.image as png: ...`). Skipping `close()` leaks a `shared_memory`
segment under IPC — read-once-unlink is the lifecycle (§6.1). Schema identical on
every transport.

### 4.5 Backpressure operator (D3)

Sits between the worker-producer and the subscriber:

- `Progress` → **conflate** (onBackpressureLatest): demand exhausted ⇒ keep newest,
  drop stale. The GPU worker thread never blocks on a slow client.
- `Result` + `on_complete`/`on_error` → **must-deliver**: buffered until `request(n)`,
  never dropped.
- `Ack` → must-deliver (single cheap frame).

---

## 5. In-proc transport + Future facade (the no-op landing, D4)

`WorkerPool.submit_job(job) -> Future` is **kept**. Internally it now:

1. Registers the job (unchanged).
2. Creates a backplane channel via the in-proc `Backplane` → `(sink, publisher)`.
3. Attaches a **compat Subscriber** to `publisher` that fulfills `job.fut`, and
   `request(unbounded)` (§3.3 demand protocol). On each signal:
   - `Result` → `fut.set_result((png_bytes, seed))` (`await blob.read()` then
     `blob.close()`; in-proc read is an unwrap, close a no-op)
   - `on_error(err)` → `fut.set_exception(err.original)` — the **live exception
     instance**, unwrapped (§4.2 invariant). Never a `BackplaneError`.
   - `on_complete` → no-op
   - `Progress` → **ignored** (delivered under unbounded demand, dropped by the
     Subscriber; the worker does not even emit real per-step progress this issue)
   - **Touches only `job.fut`** — never `_job_lock`, never pool methods (§3.3 lock
     invariant).
4. Returns `job.fut`.

**Worker loop change is purely internal** (`worker_pool.py:688-809`): where it now
calls `job.fut.set_result/set_exception`, it drives the `sink` instead. The
**producer-side cancel discard is explicit** — a cancelled job must not emit `Result`:

```python
sink.ack()
result = job.execute(worker)          # (png_bytes, seed)
if sink.cancelled:                     # running-cancel: discard producer-side
    sink.error(BackplaneError(CANCELLED))   # -> Subscriber: fut.set_exception(CancelledError)
else:
    sink.result(seed, InProcBlob(png_bytes))
    sink.complete()
# on exception e:
#   sink.error(BackplaneError.from_exc(e))   # carries the live instance
#   OOM / StaleResolutionError paths keep their existing pre-checks (worker_pool.py:743-804)
```

The discard happens by **not emitting `Result`**, not by the Subscriber suppressing
one — reproducing `worker_pool.py:758-763` where the running-cancel branch calls
`set_exception(CancelledError())` in place of `set_result`.

**Cancel entry points are unchanged on the consumer side** — every direct
`fut.cancel()` stays exactly where §4.3 tabulates it; the backplane only *adds*
`sink.cancelled`.

### 5.1 No-op proof

These pass **with zero edits** (the empty `ws_routes.py` diff *is* the proof):

- `tests/test_ws_routes.py` — `pool.submit_job.return_value = fut; fut.result...`
  stubs (`:303`, `:400`, `:492`, `:590`, `:1090`) still describe the real facade.
- `tests/test_worker_pool.py` — `TestJobSubmission`, `TestErrorHandling`,
  `TestOomRecovery`, cancel tests (`:514`, `:546`), `TestConcurrency`.
- `tests/test_model_routes.py` — routes untouched.

**New backplane tests (added, not modified):** in-proc Publisher delivers
`ack→result→complete` in order; `on_error` carries the **live exception instance**
(a `pytest.raises(OutOfMemoryError)` / `raises(StaleResolutionError)` passes through
the facade); a running-cancel discards producer-side (no `Result` emitted, `fut` gets
`CancelledError`); conflating operator drops stale `Progress` under bounded demand
while `Result` is never dropped; `subscription.cancel()` sets `sink.cancelled`.

---

## 6. Stdlib IPC/shared-mem transport (D5)

Same interface, different wiring. The `ipc` `Backplane` hands the producer a
`JobSink` and the consumer a `Publisher[Frame]` — identical types to in-proc.
Worker and parent code are byte-for-byte unchanged; only the factory differs.

**Two channels:**
- **Frames** (small: `Ack`, `Progress`, `Result`-metadata, error, terminal markers)
  ride a `multiprocessing.Connection` (Pipe). The codec serializes each frame to a
  tagged record via explicit `to_wire`/`from_wire` — **not** blanket pickle — so the
  wire schema is auditable. Every record carries a **`schema_version: int`** tag as
  its first field. This is what actually makes the schema forward-stable: the video
  extensions (§8) add fields (`media_type`, `Output`/`Chunk`) as higher-version
  records the decoder branches on, rather than forcing a codec rewrite. Costs one int
  now; without it, "additive field with a default" is a lie at the byte level (the
  decoder can't know whether to expect the field).
- **Payload** (the PNG) rides `multiprocessing.shared_memory`. `sink.result(seed,
  blob)` allocates a `SharedMemory` block, writes the bytes; the `Result` frame
  carries `SharedMemBlob(name, size)`. Consumer `await blob.read()` maps, copies out,
  then unlinks.

### 6.1 BlobRef lifecycle (the sharp edge)

- Producer creates + writes; frame carries `(name, size)`.
- Consumer reads once, then `blob.close()` (= `SharedMemory.close()` + `unlink()`,
  §4.4) — single-consumer, read-once. The consumer's obligation to `close()` is the
  ABC contract, not IPC-specific etiquette.
- On `on_error` / `subscription.cancel()` before the consumer reads: a **reaper** on
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

- `backends/backplane/` package: **vendored reactivestreams ABCs**, frames, BlobRef,
  two-sided interface, in-proc transport, stdlib IPC transport.
- `WorkerPool.submit_job`/`cancel_job`/`cancel_pending_generation_jobs` reworked to
  drive the backplane behind the preserved Future facade — every direct `fut.cancel()`
  retained (§4.3) — zero `ws_routes.py` diff.
- New backplane unit tests + the process-boundary IPC test. Existing suite green
  unmodified.
- **No new runtime dependency** — reactivestreams ABCs are vendored; `anyio` already
  present via Starlette.

### Data-plane / control-plane boundary note

`BackplaneErrorCode.STALE_EPOCH` names a control-plane concept (resolution epoch)
inside a data-plane taxonomy. This is deliberate and does not breach "data only": the
backplane **carries** the code as an opaque label — it never computes staleness. The
epoch check and the decision to fail stays in the worker-loop barrier today
(`worker_pool.py:743`) and moves to the Governor later; the backplane only relays the
resulting terminal.

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

## 8. Forward compatibility: video (non-goal, extension points only)

Video is **out of scope** for this issue. It is documented here because it is the
forcing function that validates the interface choices — the expensive, hard-to-reverse
decisions (reactive-streams interface, per-job `Publisher`, `BlobRef`, `request(n)`
backpressure) are precisely what video needs. Video enters as **new frame types + one
additive field + graduating video off the Future facade**, never as a reshape of the
interface, BlobRef, or transports.

**What holds unchanged (video-ready by construction):**

- **Reactive-streams interface (D1).** A `Future` is single-valued and could have
  carried image generation; video **cannot** — decoded frames/chunks stream out before
  the clip finishes (inherently multi-output). The per-job `Publisher[Frame]` expresses
  that natively. Video vindicates D1.
- **BlobRef (D2).** Media-agnostic; `read() -> bytes` holds for mp4/webm/raw-frames.
  Video makes shared-mem **non-optional** (tens–hundreds of MB) and possibly
  multi-segment. One additive, non-breaking field: `media_type` on the blob
  (today `image/png` is implicit).
- **Split backpressure (D3).** Generalizes: telemetry conflates, output must-deliver.
  `request(n)` — near-pointless for a single image `Result` — becomes the real throttle
  for a client consuming a video stream.
- **IPC/shared-mem transport (D5).** Same seam; segment sizing + BlobRef reaper just
  get stressed harder.

**What video adds (follow-ups, not reshapes):**

- **A third frame class — streamed output.** `Output`/`Chunk{seq, blob, media_type,
  final}`: must-deliver, **ordered** (unlike conflatable `Progress`), large (rides
  BlobRef). Today's `Result` becomes the degenerate single-chunk case (`seq=0,
  final=True`).
- **Stage-aware `Progress`.** The `stage` field (§4.1) is the forward hook; video makes
  it load-bearing: `denoise → decode` (VAE latents→frames, often the dominant cost)
  `→ mux`. Likely `frames_done/frames_total` alongside `step/total`.
- **Video does not use the Future facade (D4) — by design.** A `Future` is
  single-valued, so video generation cannot use the compat shim; it is *born* consuming
  the `Publisher` directly — which is exactly the **progress→WS "consume the stream
  directly" follow-up already deferred** in §7. Video does not create that work; it
  justifies it. The image path stays on the facade untouched.
- **Cancel + subprocess isolation go from nice-to-have to necessary.** Video jobs run
  minutes holding VRAM the whole time, so the inbound `subscription.cancel()` path and
  facet-3 subprocess reap (STABL-qvmdayhb) become load-bearing — but they are already
  named siblings, not new scope.

**Design tell to preserve now (costs nothing):** do **not** bake "exactly one `Result`
then `on_complete`" into the transport or the codec. Keep the output side a stream of
**N≥1** output frames terminated by `on_complete`, even though the image path always
sends N=1. That keeps the video door open for free.

---

## 9. Risks / mitigations

| Risk | Mitigation |
| --- | --- |
| ~~rsocket-py dependency weight / unverified ABC surface~~ **Resolved by D1 revision.** | reactivestreams ABCs are **vendored** (~150 LOC, snake_case surface we control). No runtime dep, no pre-1.0 coupling, nothing to verify at install time. Footprint of rsocket-py was confirmed trivial (91 KB, zero transitive deps) — but the coupling, not the weight, was the risk, and vendoring removes it. |
| thread→asyncio bridge correctness — **the under-weighted risk.** | Now pinned at the **design** level, not deferred to tests: sink call semantics are fixed (ack/progress non-blocking; result/complete/error synchronous must-deliver), demand is `request(unbounded)` in-proc (conflation dormant), and the Subscriber↔lock invariant forbids `_job_lock` acquisition (§3.3). Tests then confirm ordering/no-deadlock — they do not substitute for the pinned protocol. |
| shared-mem orphans on macOS/dev | `resource_tracker` warnings + leaked segments; `BlobRef.close()` read-once-unlink (ABC contract, §4.4) + producer-side reaper on error/cancel; boundary test asserts no surviving segment. |

---

## 10. Acceptance criteria (from the FP issue)

1. A backplane interface with at least the in-proc transport, wired so WS
   progress/result/error are delivered through it with **no observable change** to
   current clients. → §5 (Future facade, empty `ws_routes.py` diff).
2. A shared-mem/IPC transport behind the **same interface**, exercised across a real
   process boundary in a test. → §6.
3. Worker and WS-route code reference **only** the backplane interface (no direct
   callback wiring, no transport imports). → §3.2, §5 (factory returns
   `(sink, publisher)`; transport selected by config).
