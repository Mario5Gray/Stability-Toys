# Reap the abandoned job — cooperative step-granularity cancellation

**Issue:** STABL-jredufxb (deliberately unparented; tracked in `project-forward-notes.md`)
**Date:** 2026-08-03
**Status:** approved

## Problem

A generation that exceeds its execution budget runs to completion anyway. The waiter
raises `TimeoutError`, the caller gets an error, and the worker keeps denoising — holding
VRAM for the full run and producing a result nobody reads.

`cancel_requested` is read only at job boundaries — pre-execute (`governor.py:835`),
post-execute (`:917`), and in the dispatch loop's failure branch (`:987`) — and `run_job`
never checks it. `wait_for_result`'s own docstring says so (`:638`). (The issue text cites
`:718`, `:772`, `:841` from 2026-07-31; the file has moved since.)

This is the last open facet of `STABL-nvmieaxh`, and the only remaining one that is about
VRAM pressure rather than architecture.

## What changed since the issue was filed

The issue (2026-07-31) asserts that Python cannot interrupt a running worker thread, so
`handle.stop()` — kill+respawn — is the only real mechanism, and asks whether the in-proc
path therefore has no reap at all.

`STABL-zueslhah` landed after it, installing `inject_step_progress`
(`backends/step_progress.py`) into **every family's denoise loop** —
`callback_on_step_end` where the pipeline supports it, the legacy
`callback` + `callback_steps=1` pair otherwise. That is a per-step re-entry point into a
running generation, in both isolation modes.

So the reap is cooperative, at step granularity: **no kill, no respawn, no model reload**,
and in-proc reaps just as well as subprocess.

## Mechanism

`inject_step_progress` grows a `should_cancel: Optional[Callable[[], bool]]` parameter.
The `_modern` / `_legacy` wrapper consults it and raises **before** emitting progress:

```python
def _modern(_pipe, step, _timestep, callback_kwargs):
    _check_cancel()          # raises; NOT inside _emit
    _emit(step)
    return callback_kwargs
```

**The check must not go through `_emit`.** `_emit` swallows every exception on purpose —
*"a bad consumer must never break generation"* — so a cancel raised inside it is silently
eaten. That swallow is correct and stays; the cancel check simply sits outside it.

### Coverage bound, stated rather than hidden

Cancellation is observed **between steps**. A single long step, VAE decode, safety
checking, and any wedged CUDA call are not interruptible by this mechanism and never will
be. What the reap guarantees is that a timed-out generation stops within roughly one step
of its budget expiring, not instantly.

### The exception type: `concurrent.futures.CancelledError`, not a bespoke class

Raise the stdlib type. Three reasons, in order of how expensive the alternative is:

1. **The child's terminal is classified by type.** `_worker_main` wraps job exceptions in
   `BackplaneError.from_exc(e)`, and `classify_exception()` maps to
   `BackplaneErrorCode.CANCELLED` only for `concurrent.futures.CancelledError` or a class
   literally named `CancelledError`. A bespoke `JobCancelled` would arrive at the parent
   as `GENERIC` — a failure, not a cancel. The subprocess parent path finalizes after
   `job.fut.result()` **without** the in-proc `cancel_requested` remap, so nothing
   downstream would correct it.
2. **It is already the codebase's spelling for this meaning** — the in-proc cancel branch
   does `job.fut.set_exception(CancelledError())` today.
3. Teaching `classify_exception()` a new type adds classification surface for no gain.

**Verified, because the obvious version of this is a trap:** `asyncio.CancelledError`
derives from `BaseException`, and had `concurrent.futures.CancelledError` been an alias of
it, the raise would pass straight through both `except Exception` handlers — the child
would die frameless, `_worker_available()` would go false, and the Governor would
kill+respawn, which is the expensive path this design exists to avoid. It is not an alias:

```text
python 3.12.13
concurrent.futures.CancelledError is asyncio.CancelledError: False
MRO: ['CancelledError', 'Error', 'Exception', 'BaseException', 'object']
caught by 'except Exception'? YES
```

**Constraint:** the exception message must not contain the substring `"out of memory"`.
The dispatch loop's `_oom` test is a substring match on `str(e)` and runs **before** the
cancel branch; a message containing that phrase would route a reap into OOM recovery.

## In-proc wiring

`GenerationJob.execute()` passes `should_cancel` alongside the existing `progress`
emitter. The predicate reads `record.cancel_requested`.

**Read lock-free, by design.** `cancel_job` writes the flag under `_job_lock`, but the
worker must never acquire `_job_lock` — the standing backplane `Subscriber`↔lock
invariant. A bool read is atomic under the GIL, and a one-step-late read is harmless: the
next step catches it.

**No change to the terminal path.** `_dispatch_loop`'s `except Exception` already has a
`job_record.cancel_requested` branch that emits `BackplaneError(CANCELLED)` and sets
`state = "cancelled"` (`governor.py:987`). The raised `CancelledError` lands there
correctly as-is.

## Subprocess wiring

Cancel travels on the **dedicated control pipe**, not the data pipe.

**Why not the data pipe**, which already carries `_CANCEL` in the backplane's own IPC
transport: the child blocks on `conn.recv_bytes()` on the data pipe waiting for its *next
job*, so a cancel arriving between jobs is handed to `decode_job()` as though it were a
job frame. And while a job is running the child is not reading that pipe at all, so a
cancel sent there would not be seen until after the work it was meant to stop. This is the
same hazard that forced a separate pipe for stats in `STABL-xtkhoidu`, in a different
shape.

`IpcJobSink.cancelled` and `_IpcSubscription.cancel()` therefore stay **unused on this
path**. They remain the backplane's own contract and its tests; they are not the production
cancel channel. Recorded explicitly so the next reader does not wire them and reintroduce
the framing race.

### Changes

- `_serve_stats` becomes `_serve_control`, handling `_STATS` (unchanged request/reply) and
  a new `_CANCEL` (no reply). It already runs on its own daemon thread reading its own
  pipe — the exact property an out-of-band signal during a job requires.
- The child holds a `threading.Event`, set by `_CANCEL`, **cleared at the start of each
  job**. `should_cancel` is `event.is_set`.
- `SubprocessWorkerHandle.cancel_current()` sends `_CANCEL`.

### Locking

Cancel sends take the existing `_control_lock`. That lock exists precisely to serialize
traffic on this pipe against the `_STATS` request/reply; a fire-and-forget write outside it
is the interleaving hazard the lock was introduced to prevent.

**Lock ordering — send outside `_job_lock`.** `cancel_job` holds `_job_lock`, and
`_control_lock` can be held for up to `_STATS_REPLY_TIMEOUT_S` (0.25s) by an in-flight
stats reply. Mutate the record under `_job_lock`, release it, *then* send. Otherwise a
`/api/models/status` fan-out can stall the dispatch loop behind a lock it has no business
waiting on.

### Recovery is not triggered

`CANCELLED` is not `OOM`, and the child stays alive, so `oom or not _worker_available()`
is false. No kill, no respawn, the model stays resident, and the next request pays no
reload. That is the whole point of choosing cooperative over kill.

## What actually gets freed

Unwinding frees the generation's intermediates back to **torch's caching allocator**, not
to the driver.

- **In-proc:** finish the reap with the existing `DeviceMemory.reclaim()` seam, so the pool
  is trimmed and driver-truth free VRAM actually rises.
- **Subprocess:** `_SubprocessMemoryConsumer.reclaim()` is a deliberate no-op, so the bytes
  return to the child's pool and are reused by the next job rather than handed back to the
  driver. `nvidia-smi` will not drop.

Both satisfy the acceptance criterion that matters — the job stops consuming VRAM and
leaves no ratchet — but only the in-proc path returns bytes to the device. A `RECLAIM`
control verb for the child is the natural follow-on and is **not** built here.

## Tests

TDD, RED before GREEN. The spawn-boundary cases must be real, not mocked — mocked
transport is what produced `STABL-spxwqlan`.

1. **The predicate stops the loop.** A fake pipe whose `__call__` drives N callbacks halts
   early once the predicate flips, and raises `CancelledError`.
2. **The swallow still holds.** A progress consumer that raises does not break generation —
   the regression guard on `_emit`, proving the cancel check sits outside it.
3. **In-proc reap end to end.** A job that exceeds its execution budget: the waiter raises
   `TimeoutError`, the record ends `cancelled`, the terminal is `CANCELLED`, and the worker
   stopped before the last step.
4. **Subprocess reap across a real spawn boundary.** The child stops mid-job, the terminal
   is `CANCELLED` (not `GENERIC` — this is the test that pins the exception-type decision),
   the **child pid is unchanged**, and no respawn occurred.
5. **A cancel between jobs does not corrupt the job stream.** The test that justifies the
   control pipe: cancel while the child sits in `recv_bytes()` awaiting its next job, then
   submit a job and assert its result arrives intact.
6. **VRAM returns in-proc.** Reserved bytes after a reaped job return to roughly the
   pre-job level.

## Non-goals

- **No kill fallback.** A generation wedged inside a single step is not reaped. Adding
  `handle.stop()` as an escalation needs an owner — the dispatch thread is blocked in
  `fut.result()` and the waiter thread races it — and costs a full model reload. Out of
  scope by decision, not oversight.
- **No reap for a job nobody is waiting on.** The trigger is `_expire`, which runs on the
  waiting thread. A client that disconnects mid-generation leaves the job running. This is
  a known uncovered case; a Governor-side deadline monitor would close it and is not built
  here.
- **No new WS frame.** The caller already receives `TimeoutError`; the sink already emits
  `CANCELLED`.
- **No `RECLAIM` control verb** for the subprocess child (see "What actually gets freed").

## Deadline ownership — already resolved

The issue asks whether the deadline belongs to the transport that timed out or to the
Governor. `wait_for_result` → `_expire` → `cancel_job` is already Governor code;
`STABL-atzqpcte` moved it there. The transport only calls in. Nothing to relocate.
