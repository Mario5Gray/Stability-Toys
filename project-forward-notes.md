# Project Forward Notes

Live register of current structural shifts and active boundary guidance.
Stable policy lives in `AGENTS.md`. This file is operational and will drift.

---

## Current objectives

The VRAM umbrella surfaced running HunyuanDiT + ControlNet on enigma (RTX 3090,
24 GB) and has since become a deliberate **worker-as-a-service** refactor, not just
bug fixes. Human-driven, no waveplan, kept close.

**Status as of 2026-07-31:** umbrella `STABL-nvmieaxh` is `in-progress`. **Four children
done** — `STABL-sqqlkmdl` (accounting), `STABL-yoauoqao` (backplane, PR #19),
`STABL-vdkdruox` (Governor, PR #20), `STABL-hjldxurg` (DeviceMemory, PR #24). Two
in-progress with code landed (`STABL-kfekehhc`, `STABL-rgvxuedo`). Three `todo`, one of
which is the load-bearing gap below.

### Worker-as-a-service — umbrella `STABL-nvmieaxh`

Four-part model (control plane vs data plane): **Governor** (parent-side control) +
**Worker Handle** (locality-agnostic) + **Backplane** (data plane) + **Worker**
(executor). Boundary = the job queue; scale path in-proc thread → subprocess (spawn,
NOT fork — CUDA contexts don't survive fork) → microservice, one contract throughout.
Authority (resolution epoch, active snapshot, admission barrier) stays PARENT-side in
the Governor.

The enigma logs separated one apparent "leak" into three distinct failures — status:

1. **Accounting was fiction → FIXED.** `get_available_vram()` used
   `total - memory_reserved()` (torch's pool vs nameplate, ignoring the CUDA context /
   library workspaces / other processes), over-committing → OOM. **`STABL-sqqlkmdl`
   (done, merged `243455e`)** flipped it to driver truth via
   `torch.cuda.mem_get_info()`.
2. **Post-free residual is the CUDA context, not a torch leak.** After free-vram torch
   reports fully freed; the ~0.5–1.5 GB left in `nvidia-smi` is the per-process context,
   unreclaimable by `empty_cache()` — only process exit frees it. Not fragmentation.
   → drives the subprocess direction (facet-3).
3. **OOM poisons the context; in-process recovery can't fix it.** `_cleanup_vram` runs
   on the worker thread but `empty_cache`/`del` cannot drop a poisoned context. Durable
   fix = subprocess isolation (kill + respawn), which the backplane now makes possible.

**Merged children:**

- **`STABL-sqqlkmdl` (done)** — driver-truth VRAM accounting (`mem_get_info`).
- **`STABL-yoauoqao` (done, PR #19, merge `919a1d6`)** — the **Backplane** data-plane
  transport. `backends/backplane/`: vendored reactive-streams ABCs (no rsocket-py dep),
  frames + `BlobRef` + `schema_version` codec, synchronous in-proc transport behind a
  preserved `submit_job()→Future` facade (0-byte `ws_routes.py` diff = the no-op proof),
  and a stdlib IPC transport proven across a real spawn boundary incl. the cross-process
  cancel channel. See "Recently landed" for the carry-forwards.

**Remaining children:**

- **`STABL-vdkdruox` — Worker Governor — DONE** (PR #20, `2768802`). Authority
  (resolution epoch, active snapshot, admission barrier) now lives in
  `backends/governor.py`. The mode-switch races it was expected to own were fixed as a
  follow-on in PR #26 — see "Authority reservation" under Recently landed.
- **`STABL-qfjfflrx` — parent↔worker seam inventory**: the CUDA-in-parent audit + the
  map of every touchpoint the service split must cover (per-job payload wire form,
  `CustomJob` callable that can't cross a boundary, `superres` as a 2nd in-parent GPU
  consumer, authority placement). Feeds the Governor + facet-3.
- **`STABL-cchxvuhs` — global GPU identity** (UUID-keyed, not local index): governor
  allocates by UUID; `CUDA_VISIBLE_DEVICES` per worker. Not blocking the Governor's
  single-GPU path.
- **Facet-3 — LANDED AND PROVEN ON HARDWARE (2026-07-31).** See "Facet-3 subprocess
  worker" under Recently landed. `STABL-rgvxuedo` M1/M2 merged as PR #23; the
  `WORKER_ISOLATION=subprocess` wiring (`STABL-ptoicrho`) merged as PR #28, which is what
  finally made the code reachable; Task 8's live acceptance passed on enigma the same
  day. The backplane's facet-3 carry-forwards (`STALE_EPOCH` reconstruction registry,
  IPC `request(n)` backpressure) remain in the backplane plan's Deferred section.
- **`STABL-xtkhoidu` — superres, the 2nd in-parent GPU consumer** (split out of
  `STABL-qfjfflrx`). **Accounting half PROVEN ON HARDWARE (2026-08-01)** — see
  "Per-process VRAM attribution" under Recently landed; the parent and the child are now
  separately attributed consumers. **The CUDA-free-parent half is DECIDED but UNBUILT
  (2026-08-02)** — superres gets its own long-lived subprocess child (option 2, sticky);
  sharing the generation child is rejected on the record. Spec:
  `docs/superpowers/specs/2026-08-02-superres-worker-isolation-decision.md`; trigger-gated
  build filed as **`STABL-jylvadvb`**. **The measurement inverted the filed framing:** the
  box already runs two CUDA contexts, so option 2 *moves* the parent's rather than adding
  one, and option 3's entire material win is ~300 MiB of 24 GB. Do not start the build
  without one of the spec's three triggers.

**Timeout ↔ VRAM interaction — clock half FIXED 2026-07-31, reap half open.** The
umbrella's 2026-07-22 comment notes the flat WS result timeout abandons a long job
*without stopping the backend*, so the worker keeps denoising and holds VRAM with the
result discarded. The `STABL-ltefhpkk` acceptance required `DEFAULT_TIMEOUT=600`, making
an abandoned job hold VRAM for **ten** minutes rather than two — correct for the admission
race, actively worse for this umbrella's goal.

The clock is fixed (`STABL-atzqpcte`, `4646005`, PR #32): two budgets split at
`JobRecord.executing_since`, so waiting is no longer charged to a generation budget.
**`DEFAULT_TIMEOUT=600` is therefore obsolete and should be removed wherever it was
applied** — it now buys nothing and still costs the ten-minute VRAM hold.

The reap is still open and is now tracked properly as **`STABL-jredufxb`**: a timed-out
generation runs to completion regardless, because `cancel_requested` is read only at job
boundaries and `run_job` never checks it. Python cannot interrupt a running worker thread,
so `handle.stop()` — facet-3's kill+respawn, now a production path — is the only real
mechanism. (The 2026-07-22 comment cites `STABL-qvmdayhb`, which does not resolve; that is
why the concern needed re-filing.)

`STABL-xdsdhmov` (ControlNet cache freed on unload/free-vram) is the merged
predecessor (`a3c1c64`): fixed retained ControlNet weights but not the accounting or
recovery facets.

### Mode-switch concurrency — RESOLVED, merged (PR #26)

Both windows are fixed. See "Authority reservation" under Recently landed.

Open, unowned (pre-existing):

| Issue | What |
|---|---|
| `STABL-vwcwmiku` | `.github/workflows/ci.yml` has never run — `.gitignore:21`'s bare `workflows` pattern means it was never committed. The Concourse pipeline in `../continuous` is already fully configured for this repo and is one `fly login` away. |

---

## Recently landed

### Per-process VRAM attribution — PROVEN (PR #36 + #41)

**FP:** STABL-xtkhoidu (in-progress — acceptance 2 open by design)
**Merges:** `287674f` (PR #36, implementation), `c0a2f65` (PR #41, live acceptance)
**Spec:** `docs/superpowers/specs/2026-07-31-per-process-vram-attribution-design.md`
**Acceptance:** `spikes/xtkhoidu_attribution_acceptance.py`

The unit of attribution is the **PROCESS**, because `torch.cuda.memory_allocated()` /
`memory_reserved()` are process-global and cannot attribute below one.
`ProcessMemoryConsumer(label)` reports the current process's counters with its pid;
`SubprocessWorkerHandle` registers a child-backed consumer over a **dedicated** control
pipe; `get_worker_pool()` registers a parent consumer labelled `"server"` **in subprocess
mode only**.

**The filed fix (register superres as a second consumer in the same process) would have
made accounting WORSE** — two consumers in one process double `sum(reserved)`, so
`unattributed_bytes` clamps to zero over a negative residual, silent apart from a debug
log. **Exactly one registered consumer per process**, with a test pinning the hazard.
The larger hole was elsewhere: `SubprocessWorkerHandle` registered **nothing**, so on the
production isolation path 100% of worker VRAM was unattributed — DeviceMemory predates the
facet-3 wiring.

Live acceptance, enigma RTX 3090, `WORKER_ISOLATION=subprocess`:

```text
- label='server'  pid=1    reserved=0.12 GiB  stale=False   <- superres, in the parent
- label='worker'  pid=154  reserved=2.01 GiB  stale=False   <- the model, in the child
unattributed : 1.93 GiB          (24.00 total / 4.06 used)
nvidia-smi   : pid=1 -> 426 MiB, pid=154 -> 2374 MiB

unattributed WITH the child       : 1.93 GiB
unattributed WITHOUT it (pre-#36) : 3.93 GiB
delta                             : 2.01 GiB == the child's reserved pool
```

**The delta is the proof, not the consumer count.** Two entries only show that something
registered; deregistering the child reproduces the pre-#36 state and shows exactly how
many bytes the fix explains. Each pool sits ~300 MiB under its `nvidia-smi` per-pid figure
— that process's CUDA context, correctly left unattributed.

**Subprocess isolation therefore IMPROVES attribution fidelity**: in-proc, `"worker"`
necessarily includes superres, because the driver cannot attribute below process
granularity. A limit of the measurement, stated rather than hidden.

Traps worth carrying:

- **`/api/models/status` cannot verify any of this.** It surfaces the `"worker"` entry's
  reserved bytes via `ModelRegistry._worker_entry()` and nothing else — no consumer list,
  no pids, no `unattributed_bytes`. The snapshot must be read inside the server's own
  process; hence a spike that owns its process and builds the pool through the production
  `get_worker_pool()` path.
- **`env.cuda` ships `CUDA_SR_LIFECYCLE=per_request`**, which frees the upscaler before a
  snapshot can see it. Run as deployed, the parent reads 0.00 GiB and the run looks like
  broken attribution when it only shows the model is not resident. Force `sticky` when
  measuring superres.
- **The control channel is a separate pipe by necessity, not preference.** The data pipe
  is read concurrently by `drain_to_subscriber` during a job, so an interleaved stats
  request/reply would be consumed as a job frame.
- **The label `"worker"` keeps its exact spelling** — `ModelRegistry._worker_entry()`
  selects on it, and `get_reserved_vram`/`get_used_vram`/the `/status` stale flag all hang
  off that lookup. Renaming it silently zeroes those. Its *meaning* widens to "the process
  hosting the worker".

**Still open, by design:** acceptance 2 — the explicit choice between option 2 (superres
gets its own child) and option 3 (share the generation child) for a CUDA-free parent. The
input it was waiting on is now measured: the parent holds a **~300 MiB CUDA context of its
own purely for superres**, on top of the 0.12 GiB upscaler.

### Facet-3 subprocess worker — durable OOM recovery — PROVEN (PR #23 + #28)

**FP:** STABL-rgvxuedo, STABL-ptoicrho | **Merges:** `18a6bdb` (PR #23), `4e4673e` (PR #28)
**Plan:** `docs/superpowers/plans/2026-07-26-facet-3-subprocess-worker.md`

The umbrella's central claim is now demonstrated on hardware rather than argued. The CUDA
context lives in a child process, so a poisoned context is dropped by **killing the
process** — which in-process `empty_cache()`/`del` categorically cannot do.

**Live acceptance, enigma RTX 3090, 2026-07-31** (`spikes/facet3_oom_acceptance.py`):

```text
handle = SubprocessWorkerHandle          <- via get_worker_pool(), the production path
child pid after load: 153                   nvidia-smi: 153, 7652 MiB
job 1: OK                                   hog pid 231 holds 15616 MiB
OOM: 'Tried to allocate 96.00 MiB. GPU 0 has 98.81 MiB free'
[Governor] Subprocess needs recovery (oom=True, alive=True); kill+respawn
child pid before OOM: 153  ->  after: 261
nvidia-smi after recovery: EMPTY         <- pid 153's 7652 MiB reclaimed
job 3: OK on the fresh process
```

**Proven:** the per-process CUDA context reclaim. `nvidia-smi` went from `153, 7652 MiB`
to nothing — that memory includes the ~0.5–1.5 GB residual this register has always said
only process exit frees. And `oom=True, alive=True` shows the in-band-OOM branch firing:
the child *survived* and was killed deliberately, not tidied up after a crash.

**NOT proven, deliberately:** that the context was genuinely *poisoned*. The OOM was
induced by external VRAM pressure, which makes allocation **fail** but does not make the
context **sticky**. The next job might have succeeded without the kill. The recovery
*policy* is proven correct; its *necessity* for that particular OOM is not. Real poisoning
came from workloads at the ceiling and may not be synthesisable.

Traps worth carrying:

- **An oversized request cannot provoke an OOM in HunyuanDiT.** `use_resolution_binning=True`
  bins everything to 1280×1280, so `size` is normalised away before it can exhaust
  anything. Use `spikes/vram_hog.py` — pressure from a *separate* process, because
  allocating in the parent would give the parent its own CUDA context and change the very
  topology under test. **Leave a window:** enough free VRAM for the 1024×1024 baseline job,
  not enough for the binned 1280×1280. On a 24GB card 14 GiB works and 15 does not — 15
  OOMs the baseline job itself, which proves nothing, since a failure then cannot be
  distinguished from "there was never enough VRAM".
- **A merged PR is not the same as every commit pushed to its branch.** `vram_hog.py` was
  authored in `e50acf8` but PR #28 merged only up to `530b785`, so the one tool that makes
  Task 8 reproducible was absent from `main` while three documents referenced it. Restored
  in `1dc2dfa`.
- **`test_hunyuandit_acceptance.py` is not a vehicle for subprocess work.** It builds
  `WorkerPool(...)` directly, while the env switch lives in `get_worker_pool()` — so
  `WORKER_ISOLATION=subprocess` leaves it running in-proc, passing, and appearing to prove
  something it never exercised.
- **`ResolvedModel` cannot be pickled** (`MappingProxyType`). It crosses the spawn boundary
  as its JSON dict via `resolved_model_to_json_dict` / `resolved_model_from_json_dict`.
  Re-resolving in the child instead has no working configuration and silently defeats
  parent-side `resolve_model` patching, since patches do not cross a spawn boundary.

Both children are now resolved:

- **`STABL-wotsqcjb` — FIXED** (`7b8a46b`, PR #30). `start()` blocked on the `_READY`
  handshake with no timeout or liveness check, so any child-side failure hung the parent
  indefinitely with VRAM held, 0% util and no error. Two guards, because neither covers
  the other's cases: a parent-side `poll()` loop with an `is_alive()` check and a deadline
  (covers `SIGKILL` and the OOM-killer), plus a child-side `_FAILED` frame carrying the
  real traceback (covers ordinary startup exceptions). `WORKER_START_TIMEOUT_S`, default
  300s. On a death detected mid-poll the loop re-checks for a buffered frame first, so the
  traceback is never discarded in favour of a bare exit code.
- **`STABL-nstyyrhh` — CLOSED as accepted risk**, and the filed mechanism was wrong.
  Measured on enigma: **one POSIX semaphore per MODEL LOAD**, linear, never reclaimed
  (4 = one default load + three forced switches). **Not** kill+respawn — six spawn/kill
  cycles with no model load leak zero, clearing `stop()` and `SIGKILL` entirely. Not
  facet-3-specific either: it is on the model-load path and would occur identically
  in-proc. The ecosystem treats this class of warning as noise, and every standard
  mitigation is worse for us — `resource_tracker.unregister` would unlink another
  library's live lock, and changing the start method would destroy facet-3, since spawn
  is what gives the child its own CUDA context. **Residual risk accepted:** growth is
  unbounded, not fixed — filed as **`STABL-cxbwwgly`** for the observability work: surface
  leaked resource counts (`/dev/shm/sem.*`, shm segments, fds) so the trend is visible
  rather than discovered as a mystery failure. Sample **inside** the container —
  `/dev/shm` is per-mount-namespace, and a host-side check reads a different one.
  `spikes/sem_creator_trace.py` names the owning library in ~1 minute if this is reopened
  — it was never identified.

Method note worth carrying: the count was stable at 2 across two runs, which read as a
fixed cost until the variable actually moving turned out to be **load count**, not
respawn count. Two runs agreeing is not a controlled comparison.


### Timeout semantics — bound execution, not queue wait — merged (PR #32)

**FP:** STABL-atzqpcte (done) | **Merge:** `4646005` (PR #32, `fix/execution-timeout-semantics` → `main`)
**Spec:** `docs/superpowers/specs/2026-07-31-execution-timeout-semantics-design.md`
**Plan:** `docs/superpowers/plans/2026-07-31-execution-timeout-semantics.md`

One clock became two, split at the moment the job actually starts executing:

| budget    | env                   | default               | bounds                                       |
| --------- | --------------------- | --------------------- | -------------------------------------------- |
| execution | `DEFAULT_TIMEOUT`     | `120` (**unchanged**) | the generation itself                        |
| admission | `ADMISSION_TIMEOUT_S` | `900`                 | queue wait, incl. a mode switch's model load |

`Governor.wait_for_result(fut)` polls the future and picks the budget from
`JobRecord.executing_since`. Admission is **bounded** rather than unbounded so a job
wedged behind a hung `ModeSwitchJob` still fails instead of pinning a connection forever.

**`DEFAULT_TIMEOUT=600` is now obsolete** — remove it wherever it was applied.

Three things worth carrying:

- **`state == "running"` is the wrong clock signal, and moving it is also wrong.** It is
  set *before* the demand reload and the stale-epoch barrier, so an execution clock there
  charges a model reload to the execution budget. And `cancel_job` branches on
  `state == "queued"` to cancel the future outright, so moving the transition later
  widens a cancel/fulfilment race — trading a timeout bug for a cancel bug. Hence the
  separate `executing_since` timestamp, which leaves cancellation untouched.
- **The waiter keys on FUTURE IDENTITY, not job id.** `runtime.submit_generate()` returns
  only a future; an id-keyed API fixes WebSocket and silently leaves HTTP broken.
- **There were THREE wait sites, not two.** `_run_generate_from_dict` (external compat
  endpoints) had its own `fut.result(timeout=REQUEST_TIMEOUT)`. It was found by a
  module-wide assertion, not by reading — scope a call-site test to the module, not to
  the function you already decided to change.

Cancel-on-timeout takes a **queued** job off the queue entirely. It does **not** stop a
running generation — that is `STABL-jredufxb`.

### Authority reservation — mode-switch admission race — merged (PR #26)

**FP:** STABL-ltefhpkk (done), STABL-iuiwzthc (done) | **Merge:** `ff1a300` (PR #26, `fix/governor-authority-reservation` → `main`)
**Spec:** `docs/superpowers/specs/2026-07-30-governor-authority-reservation-design.md`
**Plan:** `docs/superpowers/plans/2026-07-30-governor-authority-reservation.md`

The Governor reserves a mode switch's **resolution epoch and resolved model at
enqueue time** rather than at load time, and admission binds a targeted generate to
that reservation. The generate's stamp therefore equals the epoch `_load_mode` will
publish. **The barrier's epoch-equality comparison is unchanged** — the fix is in what
gets stamped, not what enforces it. Seven TDD tasks; Python 1085 passed / 9 skipped /
1 xfailed, Go 9/9; live acceptance on enigma across two sequences.

Design decisions worth carrying:

- **`mode=None` binds the ACTIVE snapshot, not terminal authority.** A generate naming
  no mode means "the current mode"; binding it to a pending switch would silently run
  it on the wrong model — worse than the bug. A bare `st gen` racing someone else's
  switch is *correctly* still rejected.
- **`switch_mode` short-circuits BEFORE reserving** when the target is already
  terminal. The dispatch fast-path (`governor.py:606`) returns `already_loaded` without
  calling `_load_mode`, so a reservation minted there is never published — the same bug,
  self-inflicted. Reports `already_queued` when the match is a pending reservation, and
  still falls through to a reload when the active mode's worker was idle-evicted.
- **Demand reload is epoch-neutral, byte-for-byte.** `_reload_from_snapshot` never
  bumps, so generates stamped at epoch N survive an eviction/reload cycle. Do not
  "fix" this by reserving there.
- **`get_current_mode()` still means "the actually-loaded mode"** — nine call sites
  depend on it. Observability went into a new `get_pending_mode()` + `pending_mode` in
  `/api/models/status`, which closes the previously **silent** window where `_load_mode`
  unregisters the outgoing mode (`:338`) and re-registers only after the load (`:370`).
- **`_reserve_and_enqueue_switch` is one critical section.** Resolve outside the lock
  (disk I/O), then re-check/bump/append/put under it. Splitting them lets a concurrent
  admitter invert queue order against `_pending_authorities`. `queue.Full` rolls the
  reservation back.
- **The CLI no longer pre-switches.** `gen.go`'s `CurrentMode` + `SwitchMode` returned
  as soon as the switch was *queued*, so the generate behind it was admitted against
  pre-switch authority — the direct cause of the race. `params["mode"]` already shipped
  in the WS frame and is now read server-side. Replaced by an intentionally empty
  `preSubmitModeSideEffects()` seam so "the CLI does not pre-switch" stays testable.

Two findings beyond the filed issues:

- **Wrong-mode configuration.** Admission bound to the live mode, so a generate
  targeting X took `size`/`steps`/`guidance` from the **outgoing** mode and resolved
  ControlNet bindings against the outgoing `family_id`. Masked only because the barrier
  rejected the job — the decisive argument against relaxing the barrier.
- **A correctness hole at the barrier.** The epoch check was conjoined with
  `snapshot is not None`, so after a failed load it was skipped entirely. With a handle
  whose `submit()` succeeds the no-authority job did not error — it **ran**. A
  dead-epoch / no-authority guard now runs first, raising `ModeLoadFailedError`.

Test-infrastructure invariants for anyone writing Governor tests: `gov._stop.set()`
does **not** stop the dispatch loop (it is blocked in `q.get(timeout=1.0)` and will
dequeue and run one more job) — use `_freeze_dispatch`, which also joins the thread.
And `shutdown()` begins with `q.join()`, so anything left queued must be cleared with
`_drain_queue` or shutdown blocks.

Deferred / filed, NOT fixed here:

- **`STABL-atzqpcte` — FIXED** (`4646005`, PR #32). See the timeout-semantics entry below.
- **`STABL-anxqlxkm`** (under `STABL-sgdavnvz`) — cross-file test isolation; two
  `gc`/`empty_cache` mock assertions fail when `test_governor.py` runs first.
- **Unload-after-gen cause not isolated.** The symptom is verified gone (inline `--mode`
  is now sticky; the model stays resident and the next generate reuses it), but the
  cause was not discriminated between the registry-gap reporting artifact and a genuine
  idle eviction from stale `_last_activity` — both were addressed in the same task.

### DeviceMemory — backend-neutral device-memory accounting — merged (PR #24)

**FP:** STABL-hjldxurg (done) | **Merge:** `445eae3` (PR #24, `feat/device-memory` → `main`)
**Spec:** `docs/superpowers/specs/2026-07-28-device-memory-design.md`
**Plan:** `docs/superpowers/plans/2026-07-28-device-memory.md`

The single source of truth for VRAM. Torch-free, backend-neutral: driver truth
(NVML-by-UUID, no CUDA-context burn) ⊕ per-consumer torch pools via a consumer
registry. Replaces the three ad-hoc `torch.cuda.mem_get_info()`/`empty_cache()`
readers (ModelRegistry, WorkerHandle, Governor). New `backends/device_memory.py`;
`model_registry.py` is now a pure DeviceMemory view, `worker_handle.py` injects it
and owns a crash-safe `Registration`, `governor.py` measures load + builds status
through it and `reclaim()` replaces inline `empty_cache`. 10 TDD tasks + follow-ons.

Design decisions worth carrying:

- **Provider by topology:** `CudaDeviceMemory` (NVML/DISCRETE), `UnifiedDeviceMemory`
  (psutil/UNIFIED), `NullDeviceMemory` (UNKNOWN — degrade, never borrow). Singleton
  selection; unusable-NVML (wheel present, no driver — mac dev) falls through to
  psutil→Unified, Null only when psutil is also absent.
- **`cached_snapshot()` (no fan-out) vs `snapshot()` (fresh, bounded fan-out).** The
  registry view is pure `cached_snapshot` — it NEVER fans out, so a wedged worker
  cannot hang `/status`. Load-time measurement is the one fresh-snapshot exception.
- **`stale`** is snapshot-authoritative: consumer pool reads never self-declare
  staleness; only a fan-out timeout substitutes last-known with `stale=True`.
  Surfaced at `/api/models/status` via the registry cached view (not the Governor's
  bytes-shape builder — full Governor-status HTTP wiring stays deferred).
- **Behavioral no-op** on `/status`: swapping `mem_get_info`→NVML is invisible
  (both driver truth).

Live T10 acceptance on enigma (RTX 3090): branch self-identifying via
`backend_version` (GIT_SHA capture); NVML-vs-`nvidia-smi` free delta **median
−0.88 MiB, dead constant** at steady state; **zero ratchet** across HunyuanDiT↔SDXL
free cycles (every free returns to the same ~0.8 GB CUDA-context floor). Suite 1052 passed.

Carried follow-ons in the same PR: `nvidia-ml-py==13.610.43` pin; dev container
now applies `LOGGING_CONFIG` via `--log-config` (the dev CMD imports the app, so
app INFO — every `[ModelRegistry]` line — was silently dropped to WARNING; feeds
`STABL-oxbwjwvu` observability); "dispatch thread already running" demoted to debug.

Deferred (tracked, NOT done): route `/status` through the Governor bytes-shape
builder; mode-switch epoch race (`STABL-ltefhpkk`/`STABL-iuiwzthc`, see above);
facet-3 `SubprocessWorkerHandle` wiring (`STABL-ptoicrho`); runtime `LOG_LEVEL`
into the build-time-generated dev log-config.

### Worker Governor — control plane — merged (PR #20)

**FP:** STABL-vdkdruox (v1 merged) | **Merge:** `2768802` (PR #20, `feat/worker-governor` → `main`)
**Spec:** `docs/superpowers/specs/2026-07-25-worker-governor-design.md`
**Plan:** `docs/superpowers/plans/2026-07-25-worker-governor.md`

The control-plane counterpart to the backplane. The four-part model is now concrete
in code: **`backends/governor.py`** (new, ~779 LOC) owns the queue, resolution
epoch/snapshot authority, admission barrier, dispatch loop, lifecycle
(load/reload/evict), cancel, and recovery; **`backends/worker_handle.py`** (new,
~193 LOC) owns the `WorkerHandle` ABC + `WorkerHealth` + `InProcessWorkerHandle`
(the threaded-worker coupling); **`backends/worker_pool.py`** is reduced to a
~260-LOC delegating facade (transitional — deleted when routes migrate to the
Governor directly). Five TDD tasks, each reviewed green; **0-byte `server/` diff**
is the no-op proof; 1008 passed (1 pre-existing `test_mode_config` hunyuandit
failure on baseline, unrelated).

Design decisions worth carrying:

- **Dispatch loop is `_worker_loop` behavior-verbatim** (open question (a) resolved
  → wrap, not replace) with one substitution: `self._worker` → `self._handle.worker`.
  The post-execute cancel-discard reads `record.cancel_requested` under `_job_lock`
  and drives `record.sink` — the handle cannot acquire `_job_lock` (backplane
  `Subscriber↔lock` invariant) or touch `record.sink`, so the dispatch body stays
  Governor-side. **`handle.submit()` is the facet-3 contract, NOT called in v1's
  in-proc dispatch loop.** (Behavior-verbatim, not literal: a few log lines
  condensed; `CustomJob` split into its own `else` branch — both equivalent, no
  test asserts on logs.)
- **`submit_job` keeps channel-opening in the Governor** (open question (b)
  resolved → Governor owns lifecycle): opens `InProcBackplane(job.job_id)`,
  stores `record.sink`, subscribes `_FutureBridge(job.fut)` BEFORE enqueueing —
  verbatim from the former `worker_pool.py`. The dispatch loop drives `record.sink`
  directly.
- **Acyclic import graph:** `worker_handle.py` imports `governor` only under
  `TYPE_CHECKING` (the `Job` hint is a string at runtime); `governor.py` imports
  `worker_handle` at module top to construct `InProcessWorkerHandle`. `InProcessWorkerHandle`
  is hoisted to `governor.py` module-top (was lazy during parallel Task 2/3 work;
  moot once Task 2 landed).
- **Authority split:** registry `unregister_model` is Governor authority
  (`_unload_current_worker` seam); worker + ControlNet-cache teardown delegates to
  the handle. `unload_current_model` returns `status:"unloaded"`; `_build_runtime_status`
  takes a `status` kwarg so both paths share one builder.
- **Pluggability proof (acceptance #4, lifecycle):** a stub `WorkerHandle` plugs in
  via `handle=` with no Governor or backplane change. v1 proves **lifecycle**
  pluggability, **not** dispatch pluggability — the dispatch loop reaches into
  `self._handle.worker` directly, so a real `SubprocessWorkerHandle` (no in-proc
  worker) would still require Governor dispatch changes (facet-3).

Carry-forwards for the next agent: the plan's Task 3/5 tests had test-environment
mocking gaps (missing `resolve_model` patches + `mode_config`/`registry` injection
that `test_worker_pool.py` provides via fixtures) — fixed minimally without changing
test intent; worth feeding back to the plan author. The frozen suite's patch targets
were repointed mechanically (`resolve_model` → `governor` namespace; `_load_mode`/
`_start_worker_thread`/`_start_watchdog_thread` → `Governor.*`) — zero assertion
changes. `torch`/`gc` patches stay on `backends.worker_pool.*` (shared module
objects — the patch reaches the Governor's code via the facade's kept imports).

Deferred to facet-3 / follow-ons (tracked in the plan's Deferred, NOT done):
mode-switch race fixes (`STABL-ltefhpkk`/`STABL-iuiwzthc` — authority now in one
place, fix is a follow-on issue); API status/VRAM routing (remove inline
`torch.cuda` in routes, route through Governor); facet-3 `SubprocessWorkerHandle`
(carries backplane facet-3 debts: `cancel_job`→`record.sink` wiring, `STALE_EPOCH`
reconstruction registry, IPC hardening); `ControlNetBinding` wire form;
`CustomJob`→typed-message redesign; timed-out-job reap (`STABL-qvmdayhb`).

### Backplane — data-plane transport — merged (PR #19)

**FP:** STABL-yoauoqao (done) | **Merge:** `919a1d6`
**Spec:** `docs/superpowers/specs/2026-07-24-backplane-data-plane-transport-design.md`
**Plan:** `docs/superpowers/plans/2026-07-24-backplane-data-plane-transport.md`

The first concrete piece of the worker-as-a-service split. New package
`backends/backplane/`. Five TDD tasks, each reviewed green; 156 passed; **0-byte
`server/ws_routes.py` diff** is the no-op proof. Design decisions worth carrying:

- **Vendored** reactive-streams `Publisher`/`Subscriber`/`Subscription` ABCs under
  `backends/backplane/reactivestreams/` — no runtime rsocket-py dep. The interface is
  rsocket-*shaped*; the rsocket transport is a follow-up child.
- `WorkerPool.submit_job` still returns a `Future`; a `_FutureBridge` Subscriber
  (attached with unbounded demand **before** enqueue) fulfils it, and the worker loop
  drives a `JobSink`. **No `anyio` / no event loop** — the worker loop is a plain
  thread, so the facade delivers synchronously in-thread. The `anyio.from_thread`
  bridge is only for the deferred async progress→WS consumer.
- **Result is carried opaquely**, not decomposed into `(png, seed)` — existing tests
  mock `run_job`→`"test_result"`; the bridge does `fut.set_result(image.read_sync())`
  verbatim. Typed seed/PNG split is deferred to the streaming consumers.
- **`BlobRef`** is transport-resolved (in-proc bytes / IPC `shared_memory`, read-once
  `close()`+unlink; `SharedMemBlob.create` unregisters the producer's `resource_tracker`
  to dodge the spawn handoff race). Frame codec carries a leading `schema_version` byte.
- **IPC** = duplex `multiprocessing.Connection` frames + `shared_memory` payload;
  inbound cancel = `subscription.cancel()` → reverse control frame → `sink.cancelled`
  (a subprocess worker can't read `cancel_requested`), proven across a real spawn
  boundary.

Deferred to facet-3 / Governor (tracked in the plan's Deferred, NOT done): wire
production `cancel_job → record.sink` subscription; `STALE_EPOCH` IPC reconstruction
via a consumer-injected `code→factory` registry (keep control-plane types out of
`frames.py`); IPC `request(n)` backpressure + `IpcJobSink(conn, job_id)` + `result()`
signature. The Governor (`STABL-vdkdruox`) is now complete and in PR review (#20) —
see "Worker Governor" above.

### HunyuanDiT family profile — merged (PR #17)

**FP:** STABL-ichgkgno | **Spec:** `docs/superpowers/specs/2026-07-16-hunyuandit-family-profile-design.md`
**Plan:** `docs/superpowers/plans/2026-07-17-hunyuandit-family-profile.md`
**Merge:** `a62bfb1` — also carried `STABL-fdurqnnn` (`drift check` now exits 0,
down from 12 stale anchors) and `STABL-svpnjbjh` (`make drift` targets).

Family dispatch is now a neutral registry (`FamilyProfile` + exact-one
`resolve_family`) resolved before mode policy, with CUDA workers selected from
one family-by-platform binding table by lazy dotted reference. HunyuanDiT runs
txt2img with zero or one Canny ControlNet through the production `WorkerPool`:
`(supports_img2img=False, supports_controlnet=True, combined=False)`, native
BERT+mT5 conditioning, `control_image`, `use_resolution_binning=True`, native
DDPMScheduler. **Live acceptance at 1024x1024 peaks at 9.88 GiB** (re-measured
2026-07-31 on enigma, `peak_allocated_bytes=10604451328`, torch 2.10.0+cu128, reproduced
identically twice). The previously recorded 18.80 GiB — and the "2.57 GiB under the spike
observation, 5.2 GiB under the 24 GiB operator floor" arithmetic built on it — understated
headroom by roughly half. The drop is consistent with `STABL-kfekehhc` stopping the fp32
upcast of the ControlNet composition, though it is single-configuration evidence on a newer
torch, not a controlled A/B against the pre-fix code.

Three family-specific traps worth carrying forward to the next family:

- **Attention processor swaps are not universally safe.** `HunyuanDiT2DModel`
  passes rotary positional embeddings through `cross_attention_kwargs`, which
  `XFormersAttnProcessor` and `SlicedAttnProcessor` warn about and drop, so the
  transformer denoises without positional information and returns noise.
  `CudaWorkerBase.supports_attention_processor_swap` gates both; it costs ~10%
  per iteration and, measurably, no VRAM at all.
- **Shared ControlNet kwargs are not universally accepted.**
  `HunyuanDiTControlNetPipeline.__call__` takes `controlnet_conditioning_scale`
  but has no `control_guidance_start`/`end`, so the SD/SDXL-shaped kwargs are
  filtered per family.
- **Control-map fixtures are family-sensitive.** A border-to-border edge map
  drives this Canny checkpoint into noise while an inset one is fine.
  `tests/hunyuan_control_map.py` is the single fixture shared by the acceptance
  and `scripts/hunyuan_cn_probe.py` — they previously held separate maps, and
  the probe validating its own map while the acceptance ran a different one cost
  a long investigation into worker code that was correct throughout.

Diagnostics: `HUNYUAN_DEBUG_DUMP=1` dumps the exact control image, call kwargs,
conditioning keys, and pipe state per job under `HUNYUAN_DEBUG_ROOT`, read-only
and inert when unset. `scripts/hunyuan_cn_probe.py` runs the family with no app
plumbing and replays a dumped control image via `CONTROL_IMAGE`. Together they
split an output-quality failure into image-bytes versus pipe-state causes in one
run — worth reaching for before reading worker code.

Depth and Pose are registered and user-reachable but only Canny is live-verified.
Hunyuan img2img, combined img2img+ControlNet, materialized Hunyuan conditioning,
and `/models/status` family exposure remain deferred.

### Pluggable prompt conditioning + Compel long prompts — merged

**FP:** STABL-hvalobvn (done, incl. docs/container/live closeout `STABL-dxxgoevd`)
**Spec:** `docs/superpowers/specs/2026-07-09-long-prompt-compel-design.md`
**Plan:** `docs/superpowers/plans/2026-07-10-pluggable-prompt-conditioning.md`

CUDA workers now use a Stability-Toys-owned prompt-conditioning seam. Native
prompt delegation remains the empty-configuration default; per-mode
`conditioning.service: compel` opts CUDA modes into local Compel materialization
for SD1.5 and SDXL. Compel is pinned in `requirements-conditioning.txt` and
installed with `--no-deps` in CUDA-capable images to avoid Notebook/Jupyter
dependency creep.

The consumer boundary is intentionally CUDA-local and live: every SD1.5/SDXL
generation branch, including txt2img, img2img, ControlNet, combined
img2img+ControlNet, and both latent entry points, invokes one chain per job and
then validates the artifact against the exact target pipeline immediately before
calling Diffusers. Compatibility failures are structural consumer failures and
never enter native fallback; `native_on_failure` only covers configured-service
invocation failure and can restore native truncation.

Direct/proxy conditioning, Redis/Qdrant artifact storage, non-CUDA materialized
consumers, frontend changes, and new CLI flags remain deferred. Operators should
enable Compel only in CUDA deployment config, not in shared repo defaults.

### Combined img2img + ControlNet — merged (PR #6)

**FP:** STABL-ztaxgbhv (parent, 10 children) | **Spec:** `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`
**Plans:** `docs/superpowers/plans/2026-07-08-img2img-controlnet-{groundwork,pipeline-wiring,followups}.md`

img2img + ControlNet in one request now executes end-to-end on CUDA (SD1.5 and
SDXL), WS/CLI only. Both workers run
`StableDiffusion(XL)ControlNetImg2ImgPipeline.from_pipe(self.pipe, ...)` — zero
extra base-model VRAM — with `image=` (init) and `control_image=` (map) kept
distinct via an `image_kwarg` override on `_build_controlnet_kwargs`, shared-VAE
dtype normalization, and a 2%-tolerance aspect-ratio gate that rejects naming the
offending `attachment_id`. Requests are capability-gated: the WS guard
(`reject_combined_img2img_controlnet`) reads
`BackendCapabilities.supports_img2img_and_controlnet` (also surfaced in
`GET /models/status`) and rejects fail-fast **before preprocessing** on non-capable
backends. Design decisions: `denoise_strength` × `start/end_percent` pass through
without renormalization (low strength + narrow window can yield no visible
conditioning — documented caveat, not a bug); combined results stay uncached.
HTTP `/generate` intentionally cannot express img2img (no `init_image_ref`) —
adding it would be a separate API decision. This track also recorded a
cross-file `sys.modules`/`lru_cache` diffusers-stub pollution failure between
`test_cuda_worker_controlnet.py` and `test_worker_controlnet_metadata.py`; that
no longer reproduces (2026-07-20: 34 passed in one session, and both files are
clean in the full suite), most likely resolved when `STABL-ichgkgno` removed the
family-string branching those stubs interacted with. No FP issue was filed.

The combined-track test-hygiene follow-ups (`STABL-bclnlnzd` torch stubbing,
`STABL-zisphapv` Miniforge pin) are both now **done**.

### Earlier landed (settled; forward-relevant detail folded into boundary decisions)

- **AssetStore bucketed interface** — `STABL-hvkybzlg` (PR #3). Protocol +
  `InMemoryAssetStore`; flat `upload`/`control_map`/`ref_image` buckets, per-bucket
  fail-closed byte budgets, `promote(ref, target_bucket)`.
- **Tiered AssetStore persistence** — `STABL-slsbyhga` (PR #4). `TieredAssetStore` =
  bucketed hot cache + optional `StorageProvider` via `server/asset_codec.py`; strict
  write-through; `ASSET_STORE_PROVIDER` env (`DISABLED`/`MEMORY`/`FILESYSTEM`, Redis
  out of scope).
- **st read: ControlNet metadata** — `STABL-teiotvmc` (PR #5). Detects `lcm`,
  `controlnet`, `controlnet_map` PNG tEXt chunks; output wrapped by chunk keyword.
- **st CLI v1.x point release** — `STABL-csqqcjmo`. `st modes switch/show/reload`,
  `Generate()` `--stream`/`--quiet`, `--controlnet-file`, upload bucket arg,
  ControlNet presets.

---

## Active boundary decisions

### CLI-first, always
Frontend has no scope until CLI surface is complete and stable. This is not
a temporary freeze — it reflects the project's delivery philosophy. Any agent
suggesting a "quick UI" for a new capability is out of bounds.

### `st gen --reset` was removed — use `conflate off` / `on`

`gen --reset` (STABL-ykdsormc) added a per-run clean slate, then was reverted
(`0779f06`). With no explicit prompt it resolved to an **empty** prompt — the
conflation baseline was the only prompt source, and the WS handler defaults a
missing prompt to `""` (`ws_routes.py:370`), rendering noise. The clean-slate
path is `st conflate off; st gen ...; st conflate on`. Do not re-add `--reset`
without first solving that empty-prompt resolution.

### `--json` output contract is frozen
`st gen --json` emits exactly `{"output","seed","storage_key","storage_url"}` —
indented, terminal (single object, not stream). Do not add fields, do not
change to NDJSON. Scripts depend on this shape. The new NDJSON surface is
`--stream`.

### `pkg/stclient` is a shared surface — design accordingly
It was always intended as the shared layer between CLI and a future MCP server.
Changes to `stclient` must be clean enough to serve both. Do not add CLI-specific
concerns (flag state, stderr, cobra) into `stclient`.

### Backend WS re-attach is deferred
The backend does not support re-attaching to an in-flight job by `jobId` from
a second connection. `st watch --job <id>` (the other half of the canonical
pipeline) is blocked on this. Do not attempt an IPC workaround. Leave the
`--stream` output contract stable so `st watch` can be added non-breakingly
when backend support lands.

### Upload `type` routes to a store bucket (STABL-kcjkrpry)
`POST /v1/upload` reads the `type` form field and routes the file to a store
bucket: `canny`/`depth`/`pose` → the durable `control_map` bucket,
`image`/`ref` → `ref_image`, and any other or missing type → the ephemeral
`upload` bucket (5-minute TTL). Routed buckets are validated as decodable
images (400 otherwise); the `upload` bucket stays lenient. The response is
`{fileRef, bucket, width?, height?}` and `st upload --json` surfaces the
server-resolved bucket. The mapping is a local constant in
`server/upload_routes.py`, intentionally decoupled from the ControlNet
registry. (Supersedes the earlier "intent-only" note — the server now routes.)

---

## Deferred tracks (explicit, with rationale)

| Track | Why deferred |
|-------|-------------|
| `st watch --job` | Backend has no WS re-attach; IPC ruled out |
| `st watch --all` | Needs a backend queue-state endpoint that doesn't exist yet |
| MCP server (`st serve mcp`) | Second consumer of `pkg/stclient`; right after CLI surface stabilizes |
| Batch (`--batch`, `--variations`, `--concurrency`) | Requires goroutine pool + N WS connections; non-trivial concurrency model |
| Config management (`st config get/set/edit`) | Nice-to-have; unblocked but not urgent |
| `st modes set-default` | No `POST /api/modes/default` endpoint in backend |
| `--dry-run` | Deferred — scope (params only vs WS mock) not decided |
| `st doctor` | Deferred post-point-release |
| Non-CUDA img2img+ControlNet execution | Compounds onto the existing non-CUDA ControlNet deferral; explicit non-goal even after CUDA combined path (`STABL-ztaxgbhv`) ships |

---

## v2 brainstorm
**FP brainstorm:** `fp://brainstorm?id=ifnwzfkdyysvlweulcigrnubknzswavj`

Eight clusters (A–H). MCP server (F) is the highest-value deferred item —
it reuses `stclient` with minimal new logic. Batch (B) is highest effort.
No v2 plan exists yet. v1.x must ship first.

---

## Structural notes

- `modesCmd` in `modes.go` retains `RunE: runModes` even after subcommands are
  added. Cobra routes child invocations to child commands; the parent `RunE`
  fires for `st modes` with no args (list behavior preserved).
- `multipartFile(filename, data, fields map[string]string)` in `http.go` is
  the existing extension point for extra form fields. SuperRes uses it for
  `magnitude`. Upload bucket uses it for `type`. No new mechanism needed.
- `buildGenParams` in `gen.go` is the layering point: config → baked PNG →
  flags. Preset expansion (`@name`) and `--controlnet-file` both hook here,
  after the existing `--controlnet` JSON block.
