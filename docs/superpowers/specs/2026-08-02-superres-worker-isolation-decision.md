# Superres worker isolation — decision

**Issue:** STABL-xtkhoidu acceptance 2 (under STABL-nvmieaxh)
**Date:** 2026-08-02
**Status:** decided, deliberately unbuilt — see Triggers

## Decision

**Superres gets its own long-lived subprocess child (option 2, sticky).** Option 3 —
sharing the generation child — is rejected on the record, not merely unchosen.

Nothing today forces the parent to go CUDA-free. This spec exists so the decision is made
once, with the measurement in hand, rather than re-argued under pressure later.

## What the measurement changed

`STABL-xtkhoidu` was filed with three options ordered by effort, on the assumption that
moving superres out costs a CUDA context. The live acceptance
(`spikes/xtkhoidu_attribution_acceptance.py`, enigma RTX 3090, PR #41) shows otherwise:

```text
- label='server'  pid=1    reserved=0.12 GiB    nvidia-smi 426 MiB   -> ~300 MiB context
- label='worker'  pid=154  reserved=2.01 GiB    nvidia-smi 2374 MiB  -> ~316 MiB context
```

**The box already runs two CUDA contexts.** Option 2 does not add one — it *moves* the
parent's into a child. Option 3 removes one. So option 3's entire material win is roughly
**300 MiB of 24 GB, 1.2%**, and everything below is what that 1.2% would cost.

## Why option 3 is rejected

**It pollutes the authority model.** The generation child is reached through the Governor:
resolution epoch, active snapshot, admission barrier, mode switch. An upscale has no mode,
no epoch, and no ControlNet. Multiplexing it into that child means teaching the admission
barrier about a job class exempt from everything it enforces — reopening precisely what
`STABL-ltefhpkk` / `STABL-iuiwzthc` closed in PR #26.

**Its lifecycle coupling is fatal, not awkward.** Facet-3 recovery works by *killing the
child*. Under option 3, every generation OOM destroys the resident upscaler; so does every
mode switch, and so does idle eviction. Superres would inherit a failure and churn model
built for a completely different workload.

**Head-of-line blocking** is the smallest of the three problems: one serial worker means a
HunyuanDiT denoise delays every queued upscale.

## Architecture

`SubprocessSuperResService` implements the **existing** `SuperResServiceProtocol`
(`submit(image_bytes, *, out_format, quality, magnitude, timeout_s) -> Future`, plus
`unload()` / `shutdown()`), injected at `create_superres_runtime` in
`backends/platforms/cuda.py`. This is a new implementation of a protocol that already
exists, not a new contract.

The protocol is bytes in, bytes out, with no callables — it crosses a spawn boundary
cleanly, unlike `CustomJob`, whose callable payload is why `STABL-govweiat` is still open.

Callers do not change. All four hold the protocol, not the implementation: `generate()`'s
superres branch, the `/superres` route (and `/v1/superres`, which delegates to it), the
compat `_run_generate_from_dict` branch — all in `server/lcm_sr_server.py` — and
`server/superres_cli.py`. RKNN and CPU providers are untouched.

### Shared child-process core

`SubprocessWorkerHandle` already owns spawn-not-fork, the `_READY` handshake with the
liveness-plus-deadline guards from `STABL-wotsqcjb`, the dedicated stats control pipe,
`DeviceMemory` registration, and kill+respawn. An SR child needs all of that and none of
the job or epoch semantics.

Extract that plumbing into a reusable base; both handles sit on it. Every guard in it was
paid for by a specific failure — a start that hung the parent forever with VRAM held, a
stats reply that would have been eaten as a job frame — and re-deriving them in a second
implementation is how they silently drift apart.

**Proof obligation:** the generation child's behavior must not change. A re-run of
`spikes/facet3_oom_acceptance.py` with the same verdict lines is the evidence.

### Wire form — new work, not a reuse

Images, not job descriptions, and in both directions: the acceptance run produced 3.4 MB
out of a 0.4 MB input at magnitude 1, and magnitude scales that hard.

**The backplane has no request-side blob lane.** `backends/job_envelope.py` says so
outright — `SharedMemBlob` exists for the **result** path, and `init_image` rides the pipe
with the pickled body, with a note to "revisit if init images become multi-MB". Superres
outputs already are. `encode_frame` asserts a `Result` carries a `SharedMemBlob`; nothing
else may. And the frame vocabulary is generation-shaped: `Result` carries a `seed`, which
an upscale has no concept of, and `Progress` carries a denoise `stage`.

So this is a **new request-side contract**, not a lane that already exists:

- an envelope shape for an SR request,
- segment ownership, which **inverts** the result path — there the child creates and the
  parent consumes; here the parent creates and the child consumes, so the unlink-on-error
  reaper has to be re-thought rather than reused,
- frame types for a job with no seed and no denoise progress.

What is genuinely reused is the `SharedMemBlob` primitive and the versioned-frame
discipline (leading `schema_version` byte). **Do not introduce a second blob primitive.**

Two things deliberately left open, because a codec written months before it is built is
speculation: the concrete frame set, and whether shared memory is warranted at all at
these sizes. A few MB over a pipe may well be cheaper than segment setup plus unlink
bookkeeping — measure before assuming the result path's answer transfers.

### Attribution

The SR child registers its own consumer, labelled `"superres"`. It must **not** use
`"worker"`: `ModelRegistry._worker_entry()` selects on that exact spelling, and
`get_reserved_vram()`, `get_used_vram()` and the `/api/models/status` stale flag all hang
off that lookup.

The parent keeps its `"server"` registration after it goes CUDA-free. A live consumer
reporting zero is *evidence* that the parent holds nothing; an absent consumer is only
silence. The one-consumer-per-process invariant is unaffected — three processes, three
consumers.

**This needs no new consumer contract, and the reason is measured rather than assumed.**
`ProcessMemoryConsumer.pool_stats()` reads `torch.cuda.memory_allocated()` /
`memory_reserved()`, and the concern worth checking is whether that read would itself
initialise CUDA — which would pin a ~300 MiB context in the very process the move exists
to empty, the measurement defeating the thing it measures. It does not. In the CUDA
container on enigma, in a process that has never touched the GPU:

```text
is_initialized BEFORE read : False
pool_stats() -> label='server' pid=1 allocated=0 reserved=0 stale=False
is_initialized AFTER read  : False
smi AFTER read             : <no compute apps>
```

Zeros, CUDA still uninitialised, no context, across repeated reads. The existing adapter
already *is* the passive zero-byte entry.

## Non-goals

- **No change to Governor admission, epochs, or the barrier.** The SR path never touches
  them, and that separation is the point of the decision.
- **RKNN and CPU superres stay in-process.** The problem is CUDA-context-specific.
- **Exposing the consumer list over HTTP.** Nothing today surfaces consumer labels, pids,
  or `unattributed_bytes` — which is why the acceptance had to be a spike rather than a
  `curl`. Real gap, separate issue.
- **On-demand or per-request child.** Considered and set aside, and the trade is not the
  obvious one. `CUDA_SR_LIFECYCLE=per_request` does **not** mean what its name suggests
  for context purposes: the job loop's `finally` calls `_unload_worker()`, which drops the
  upsampler and `empty_cache()`s, but the process — and therefore its CUDA context —
  survives. So a per-request *child* is not "today's cost plus a spawn". It would
  additionally pay **context creation per upscale**, and it would gain something today's
  `per_request` never delivers: the context actually reclaimed between requests, since only
  process exit frees it. Sticky is chosen because it is easier to reason about, on a thin
  margin; recorded here so the alternative is not rediscovered from scratch.

## Triggers

Build it when any one of these holds. Until then it stays recorded and unbuilt.

1. **A parent-side CUDA OOM is observed in the wild.** The parent cannot be
   kill+respawned — it is the server — so facet-3's recovery does not cover it. This is
   the trigger most likely to fire.
2. **The microservice step of the scale path is scheduled.** A parent holding a CUDA
   context cannot move to a GPU-less host.
3. **Superres VRAM measurably contends with generation.** ~300 MiB of context plus a
   resident upscaler is noise at 3.6 GiB used; it is not noise at the 24 GB ceiling.

## Open, not decided here

Whether the parent should *also* shed its remaining GPU touchpoints once superres moves.
The `STABL-qfjfflrx` audit found superres to be the last one, but that audit predates the
progress-frame work (`STABL-zueslhah`); re-verify before claiming a CUDA-free parent
rather than inheriting the claim.
