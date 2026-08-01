# Per-process VRAM attribution — design

**Issue:** STABL-xtkhoidu (child of STABL-nvmieaxh)
**Date:** 2026-07-31
**Status:** approved (per-process granularity; child attribution folded in — 2026-07-31)

## Problem

DeviceMemory's attribution is blind in both directions, and the filed issue only saw one:

1. **superres is unregistered.** `server/superres_service.py` is an independent in-parent GPU
   consumer (own `cuda:0`, `is_available`, `empty_cache`, OOM classification, queue). Its
   bytes land in `unattributed_bytes`.
2. **Under `WORKER_ISOLATION=subprocess`, nothing is registered at all.**
   `InProcessWorkerHandle.start()` registers a `WorkerMemoryConsumer`;
   `SubprocessWorkerHandle` registers nothing. On the production isolation path,
   `consumers` is empty and **100% of worker VRAM is unattributed**. DeviceMemory
   (`STABL-hjldxurg`) predates the facet-3 wiring and was never re-pointed at it.

And the obvious fix makes things worse: `WorkerMemoryConsumer.pool_stats()` reads
`torch.cuda.memory_allocated()` / `memory_reserved()`, which are **process-global**.
Registering superres as a second consumer in the same process doubles
`sum(reserved_bytes)`, so `unattributed_bytes = max(0, used - sum(reserved))` clamps to
zero while the residual goes negative — the registry already logs
`consumer over-report` for exactly this.

## Approach: the process is the unit of attribution

Coarse by choice for this iteration. The driver attributes per-pid, and `ConsumerMemory`
already carries a `pid` field, so the design anticipated it.

**Invariant: exactly one registered consumer per process.**

| mode | parent process | child process |
| --- | --- | --- |
| `inproc` | one consumer, label `worker` — holds the worker **and** superres | — |
| `subprocess` | one consumer, label `server` — holds superres | one consumer, label `worker` |

A useful property falls out: **switching to subprocess isolation improves attribution
fidelity.** In-proc, `worker` necessarily includes superres because the driver cannot
attribute below process granularity — that is a limit of the measurement, not a modelling
choice, and it is stated rather than hidden. Under subprocess the two separate cleanly.

### The `worker` label is load-bearing

`ModelRegistry._worker_entry()` selects `c.label == "worker"`, and `get_reserved_vram()`,
`get_used_vram()` and the `stale` flag in `/api/models/status` all depend on it. The label
keeps its exact spelling; its *meaning* becomes "the process hosting the worker".

## Components

**`ProcessMemoryConsumer(label)`** — reports the **current process's** torch counters with
`pid=os.getpid()`. Generalises `WorkerMemoryConsumer`, which reported the same numbers but
was named as though they were the worker's alone.

**Child attribution over a dedicated control pipe.** `SubprocessWorkerHandle` gains a
second `Pipe()` used only for control, and the child serves it from a daemon thread. A
dedicated channel is required, not a preference: the data pipe is being read concurrently
by `drain_to_subscriber` during a job, so a stats request/reply interleaved there would
corrupt the frame stream.

`pool_stats()` on the parent side does a bounded request/reply. It needs no new failure
handling — `_ConsumerRegistry._read_consumer` already bounds every consumer at
`POOL_STATS_TIMEOUT_S = 0.5` and substitutes last-known with `stale=True`, which
`/api/models/status` already surfaces (`STABL-hjldxurg`). A busy or dead child degrades
along the path built for a wedged worker.

**Parent registration in subprocess mode only.** Registered where the isolation decision is
already made (`get_worker_pool`), because registering it unconditionally would reintroduce
the double-count in `inproc`.

## Tests

1. **The double-count guard.** Two `ProcessMemoryConsumer`s registered in one process drive
   `unattributed_bytes` to zero with a negative residual — asserted so the invariant has a
   failing test behind it rather than a comment.
2. **Child attribution across a real spawn boundary.** A `SubprocessWorkerHandle` consumer
   reports the *child's* pid, not the parent's. Mocked transport is what missed
   `STABL-spxwqlan`; the same applies here.
3. **Stats do not corrupt the job stream.** A stats request issued while a job is in flight
   leaves the job's result intact — the reason the control pipe is separate.
4. **Timeout degrades to stale.** An unresponsive child yields `stale=True` rather than a
   raised exception or a zero that reads as truth.
5. **The `worker` label still resolves.** `get_reserved_vram()` remains non-zero in both
   isolation modes, guarding the `_worker_entry()` lookup.

## Non-goals

- Per-subsystem attribution inside one process. It needs load-time delta measurement per
  consumer and is materially more work; per-process is the chosen resolution for this
  iteration.
- Moving superres out of the parent. That decision stays open in `STABL-xtkhoidu`'s
  options 2 and 3; this makes its cost *visible*, which is the prerequisite for deciding.
