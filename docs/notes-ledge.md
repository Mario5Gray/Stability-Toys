# Notes Ledge

Archived "Recently landed" notes from `project-forward-notes.md`. Moved
2026-09-26 to keep the forward notes at the live tip. Newest first.

---

### Structured Loki-ready logging — third observability pillar — merged (PR #54)

**FP:** STABL-bpsfmoke (done) — child of **STABL-oxbwjwvu**
**Merge:** `872d92d` (PR #54, `feat/bpsfmoke-structured-logging` → `main`)
**Plan:** `docs/superpowers/plans/2026-08-06-structured-logging.md` (50/50 steps)
**Contract:** `docs/observability-contract.md` — now covers metrics **and** logs

`LOG_FORMAT=json` emits one JSON object per line carrying `job_id` correlation, from
**both** processes that write to the container's stdout — the server, and the spawned
worker child where generation actually happens. Default `text`, byte-for-byte
unchanged. Suite 1466 passed (baseline 1323, +143 tests).

**THE TIMING IS THE ENTIRE DESIGN: `LOG_FORMAT` is resolved in
`StabilityFormatter.__init__`, not at module import.** `docker/runtime/live-test.Dockerfile:34`
materialises `LOGGING_CONFIG` to `/app/logging_config.json` at **build** time, so
anything `logging_config.py` reads from the environment at import is baked into the
image — which is exactly why `LOG_LEVEL` is *still* not runtime-settable on the dev
path. A formatter **object** is constructed when `dictConfig` runs, i.e. container
start, on both entry paths. The formatter is therefore a `"()"` dotted reference, not
an imperatively attached instance.

The wiring tests configure from `json.loads(json.dumps(LOGGING_CONFIG))`, never the
Python object, because the dev path never sees the object. Confirmed RED without the
change: 3 of 4 fail.

**`job_id` spans two threads and the reset is the load-bearing half.** The WS handler
runs on the event loop; generation runs on the dispatch thread, which inherits nothing
from the submitter.

- **Dispatch loop:** token set after `q.get()`, reset in the existing `finally` next to
  `task_done()` — the only path that runs on every exit including the cancel-check
  `continue`. 44 lines added to `governor.py`, **zero removals, zero indentation
  change**; wrapping ~170 lines in a `with` would have made the diff unreviewable.
- **WS:** binds `None` around the message loop's handler invocation (`ws_routes.py:902`).
  One place, covers handlers added later. `asyncio.create_task` **copies the context**,
  so the four `_run_*` tasks inherit the id for free — pinned by two tests, because if
  that property lapsed every generation line would silently lose its `job_id`.

Set-without-reset is the failure that matters: a stale id reads as *real* correlation
and survives review. Observed via a log `Handler` — `emit()` runs on the **emitting**
thread, the only way to read the dispatch loop's context from a test — asserting the
loop's own `[Governor] Dispatch loop stopped` line carries no id.

**Known gap, documented rather than hidden: no HTTP handler line carries `job_id`** —
both `POST /generate` and the compat runner. `submit_generate()` returns only a future,
deliberately (`STABL-atzqpcte`: an id-keyed waiter API fixes WebSocket and silently
leaves HTTP broken). Dispatch-thread lines still carry the id. Closing it is a runtime
API change, not a formatter change.

**Six plan defects surfaced in execution** — the running theme since `STABL-asawxgvp`,
now with a sixth instance. All patched back into the plan:

- **`mode` needed a THIRD publication site.** `_reload_from_snapshot` restores the
  worker after idle eviction *without* going through `_load_mode`, and the eviction
  already cleared the field. **Rule: anywhere `_publish_mode_active` is republished, the
  log field must be too.**
- **A spawned-child test must report failures ON THE QUEUE.** A crashed child put
  nothing, so RED took **122 seconds** to say `ImportError`. A test whose failure mode is
  a two-minute timeout is a test people stop running.
- **Calling the bootstrap directly does not test the WIRING** — that test passes if
  `_worker_main` stops calling it. An `ast` test pins it as the first statement, and was
  itself verified non-vacuous by parsing a copy with the call removed.
- **The three `debug dump ... failed` lines were specced at DEBUG on false reasoning.**
  `_hunyuan_debug_dir` returns `None` unless `HUNYUAN_DEBUG_DUMP=1` and both writers
  return early on `None`, so they are *unreachable* with the dump off. DEBUG hides a
  diagnostic's failure from the one person who enabled it. Now `warning`.
- Two named test files do not exist (`test_cuda_worker.py`, `test_ws_metrics.py`).
- The drift step was too coarse — below.

**DRIFT: `drift link <doc>` relinks EVERY anchor in that doc, not just yours.** The
eight bound docs also carried anchors already stale on `main` from `STABL-zueslhah` and
`STABL-atzqpcte`, so the natural per-doc relink put fresh provenance on ten bindings
nobody reviewed — the exact failure `AGENTS.md` warns about. Caught, `drift.lock`
reverted, redone as `drift link <doc> <anchor>` one at a time.

**And `drift check` is NOT clean on `main` — 18 stale, exit 1.** Without a baseline the
number is unreadable. The procedure that works:

```bash
git worktree add /tmp/baseline main
(cd /tmp/baseline && drift check) | awk '/^docs\//{d=$1} /STALE/{print d" -> "$2}' | sort > /tmp/main.txt
drift check | awk '/^docs\//{d=$1} /STALE/{print d" -> "$2}' | sort > /tmp/branch.txt
comm -13 /tmp/main.txt /tmp/branch.txt      # exactly the anchors YOU made stale
```

Relink only those, then assert **both** diff directions empty. Result here: 18 = 18,
both empty, `drift.lock` churn 13 lines. Filed **`STABL-qjbqzwpe`** for the pre-existing
condition (18 stale from five merged branches; `STABL-fdurqnnn` had it at 0 and it
drifted back, unnoticed because the gate is prose in `AGENTS.md` and CI has never run).

**Evidence-measurement trap, second instance.** `drift check | tail -20; echo $?` reports
**`tail`'s** exit status, not `drift`'s — reported as `EXIT=0` in review when the real
code is 1. Conclusions were unaffected because the load-bearing evidence was the *set*
comparison, but the number was wrong. Same lesson as the suite-count correction on
`STABL-cxbwwgly`: **measure what you claim to be measuring; a pipeline's exit status is
the last command's.**

**Two follow-ups filed, both real, neither fixed here:**

- **`STABL-xqqqqvse`** — `LOG_FORMAT` is set in **no** env file, compose file or
  Dockerfile. The feature ships dark: implemented, tested, documented, enabled nowhere.
  Has a precondition — confirm with `../continuous` that something is scraping container
  stdout, or turning on JSON only makes logs harder for a human with no gain.
- **`STABL-ataigkdk`** — `LOG_LEVEL` and `LOG_FORMAT` now have **opposite semantics in
  the same config file**: `LOG_FORMAT` is runtime on both paths, `LOG_LEVEL` is frozen at
  image build on the dev path. Surprising beats broken for cost of diagnosis.
  **Decided 2026-08-07: precedence is runtime `LOG_LEVEL` > baked value > `INFO`, applied
  at the FastAPI `lifespan`** (`lcm_sr_server.py:390`, which runs on *both* entry paths)
  plus `_configure_child_logging` for the child. The baked value stays — it is the
  default, not the problem. **Rebaking the config file at build/entrypoint is rejected on
  the record as a kludge.** Trap for the implementer: setting the root logger is not
  enough and *will look like it worked* — `LOGGING_CONFIG` assigns explicit levels to
  `comfy`, `comfy.jobs`, `uvicorn`, `uvicorn.error`, `uvicorn.access`, and an explicit
  child level shadows root.

Umbrella `STABL-oxbwjwvu` now has **four of five children done**; `STABL-qnlaclof`
(tracing) is the last.

### Leaked OS resource gauges — the semaphore watch — merged (PR #52)

**FP:** STABL-cxbwwgly (done) — child of **STABL-oxbwjwvu**, built on STABL-asawxgvp
**Merge:** `ec6075a` (PR #52, `feat/cxbwwgly-leaked-resource-gauges` → `main`)
**Plan:** `docs/superpowers/plans/2026-08-06-leaked-resource-gauges.md` (21/21 steps)
**Contract:** `docs/observability-contract.md` — the three families + the leak ratio

`STABL-nstyyrhh` accepted the one-semaphore-per-model-load leak **specifically because
it is cheap to watch**. This is the watch, so that half of the bargain is no longer
outstanding. `server/resource_probe.py` (stdlib + psutil) feeds three gauges through the
existing sampler: `st_process_leaked_semaphores`, `st_process_shm_segments`,
`st_process_open_fds`, all labelled `process="server"`. Suite 1323 passed (baseline 1302).

**ABSENT, NEVER ZERO — and the label is what makes it possible.** A `0` for leaked
semaphores on a host with no `/dev/shm` is indistinguishable from a healthy Linux box.
The plan specified **unlabelled** gauges and was wrong: an unlabelled prometheus `Gauge`
renders its default `0.0` from declaration, so the rule is unachievable that way. A
labelled family emits nothing until a child is created. Measured:

```text
before any set:  bare_g 0.0                    <- renders immediately
                 (labelled)                    <- nothing
after set:       lab_g{process="server"} 7.0
```

The `process` label also answers what the contract would otherwise answer only in prose
(whose counts these are), matches the DeviceMemory `consumer` vocabulary, and leaves room
for the deferred worker-side probe. Live on macOS, where `/dev/shm` does not exist, the
scrape shows `st_process_open_fds{process="server"}` **and nothing else**.

**Both of the issue's stated requirements are satisfied structurally, not by discipline:**

- *"Sample inside the container"* — putting the probe in `MetricsSampler` satisfies it
  **by construction**. The sampler thread runs in the server process inside the container;
  there is no configuration to get wrong and no way to read the host's mount namespace,
  which is the mistake that cost time in the original investigation.
- *"Report the trend, not the instant"* — needed no new counter. The contract ships
  `increase(st_process_leaked_semaphores[1h]) / increase(st_governor_mode_load_seconds_count[1h])`;
  near 1 reproduces the original finding.

Traps worth carrying:

- **An early `return` in a multi-reader function orphans every reader added after it.**
  `sample_once`'s `if self._runtime_stats_fn is None: return` would have put the resource
  block behind "was a runtime stats reader injected?" — true in production, false in most
  tests. Dead code that looks healthy in the app. Now a guarded branch.
- **The contract test had to be widened to accept histogram children.**
  `st_governor_mode_load_seconds_count` failed its reverse direction; `_count` is the
  normal way to query a histogram, and a contract that cannot name it cannot explain how
  to use its own histograms. `_created` stays excluded as a client-library artifact.
- **Import concretely when the layer allows it.** `resource_probe` is imported as a real
  type rather than structurally mirrored like `_SnapshotLike` — same layer, no
  import-direction reason to mirror, and a real type cannot drift out of step.
- **An observability component that hides its own failure is the worst kind.** The probe's
  outer guards log with `exc_info`, pinned by a test asserting both message and traceback
  presence — without it, an absent series reads identically whether the source is
  unavailable or the probe broke.

**Process note worth more than the code.** The suite figure in one commit message and in
the FP closeout was measured on a tree that predated that very commit's change — the run
finished, then the code changed, then both were committed together. Corrected to 1323 on
the real tip. The FP copy mattered more than the commit copy: a stale number in immutable
history can be spoken to by a later comment, but the issue's audit trail is what a future
agent reconstructs state from *without re-deriving it*. **Run the suite on the tree you are
about to summarise, not the tree you had when you started typing.**

**NOT done: fixing the leak.** This is the watch, not the cure. `spikes/sem_creator_trace.py`
names the owning library in about a minute by patching
`multiprocessing.synchronize.SemLock.__init__` — the creator was never identified.

### HTTP + WebSocket metrics — second observability pillar — merged (PR #50)

**FP:** STABL-xmsrxvto (done) — child of **STABL-oxbwjwvu**, built on STABL-asawxgvp
**Merge:** `2205b74` (PR #50, `feat/xmsrxvto-http-ws-metrics` → `main`)
**Plan:** `docs/superpowers/plans/2026-08-05-http-ws-metrics.md` (27/27 steps)
**Contract:** `docs/observability-contract.md` — extended with the five new families

Five families on the existing facade: `st_http_requests_total`,
`st_http_request_duration_seconds`, `st_ws_connections_active`, `st_ws_sessions_total`,
`st_ws_messages_total`. One pure ASGI middleware for HTTP; WS split between `ws_hub.py`
(connections + every outbound message) and the `ws_routes.py` message loop (inbound).
Still inert by default. Suite 1302 passed (baseline 1265, +45 tests).

**Two places the approved spec's hook points were incomplete** — both worth remembering
before instrumenting anything else in this codebase:

- **Outbound WS goes in `ws_hub`, NOT `websocket_endpoint`.** `_status_broadcaster` calls
  `hub.broadcast()` every 5s entirely outside the endpoint's loop. Hooking only the
  endpoint would have made the single most frequent outbound message invisible.
- **`ws_connections_active` is SET from `hub.client_count`, never inc/dec.**
  `disconnect()` is idempotent, and BOTH `send()` and `broadcast()` reap dead clients
  through it, so a paired counter drifts negative.

Traps worth carrying:

- **`route` must be the matched route TEMPLATE** (`scope["route"].path`, read AFTER the
  downstream app runs), never `request.url.path` — the latter makes `/api/models/{name}`
  one series per model. An unmatched request has `route=None` → `__unmatched__`, which is
  load-bearing: that is the unbounded set a scanner probes.
- **`/metrics` counts itself ONE SCRAPE BEHIND.** The counter increments after the body is
  rendered, so scrape N reports N−1 and the first scrape after a restart shows no
  `/metrics` series. Not a lost request.
- **A client-controlled string must map through a bounded registry before it can be a
  label.** Inbound WS `type` goes through `HANDLERS`; unrecognised → `unknown`,
  unparseable → `invalid_json`, deliberately distinct.
- **`raw in HANDLERS` and `HANDLERS.get(raw)` both HASH their argument.** A client sending
  `{"type": {}}` raises `TypeError`. See `STABL-gzfzzsdq` below.
- **FastAPI needs a WS handler parameter ANNOTATED as `WebSocket`** to inject it; without
  the annotation it is treated as a required query param and the connection closes with
  1008 before the handler runs. Presents as `WebSocketDisconnect`, which looks like
  disconnect handling and is not — read the close frame's `reason`.
- **`prometheus_client` strips `_total` from Counter names in `collect()`** and emits a
  `_created` companion gauge per counter. `PROMETHEUS_DISABLE_CREATED_SERIES=1` suppresses
  those (verified on 0.21.1).
- **Pure ASGI middleware over `BaseHTTPMiddleware`** — no anyio task group per request, no
  interaction with streaming or background tasks, explicit WebSocket/lifespan skip.

**The bidirectional contract test earned its keep twice.** It caught this issue's own prose
naming `st_`-prefixed example series that were not real families, and it is what forces the
doc and the code to move together. Do not weaken it to accommodate wording.

**Adjacent bug found and deliberately NOT fixed here: `STABL-gzfzzsdq`.** A client can drop
its own WebSocket connection with `{"type": {}}` or `{"type": []}` — `HANDLERS.get()`
(`ws_routes.py`) hashes a client-controlled value, the `TypeError` escapes to the loop's
outer `except Exception`, and `finally: await hub.disconnect()` closes the socket. Verified
end to end with controls (`ping` and an unknown STRING type both survive). Suggested fix is
an `isinstance(msg_type, str)` guard. The `_inbound_type` guard added by this PR does NOT
fix it — it only stops the metrics code being the first thing to raise.

**Drift note worth carrying.** Relinking after review is correct even when the doc's prose
does not change — `AGENTS.md` says "never relink without review", not "never relink without
a prose edit". A review challenged this and was withdrawn on evidence: the bound-file diff
was 48 lines with zero removals, none of the seven bound docs mentioned the middleware
stack or message loop at all, and none of the eight symbols they assert about was touched.
The lesson is that the evidence has to be produced, not asserted — grepping for filename
mentions is weaker than the file-level claim a `drift.lock` entry makes.

### Prometheus substrate — first observability pillar — merged (PR #48)

**FP:** STABL-asawxgvp (done) — first child of umbrella **STABL-oxbwjwvu**
**Merge:** `dd633e6` (PR #48, `feat/asawxgvp-prometheus-substrate` → `main`)
**Spec:** `docs/superpowers/specs/2026-08-03-server-observability-seams-design.md`
**Plan:** `docs/superpowers/plans/2026-08-03-prometheus-substrate.md`
**Contract:** `docs/observability-contract.md` ← what `../continuous` consumes

The server now exports metrics. `server/metrics.py` (facade + gate + 20 families),
`server/metrics_sampler.py` (the only thing that fans out), `server/metrics_routes.py`
(`/metrics` + runtime-stats adapter), Governor instrumentation, 103 new tests.
**Inert by default:** `METRICS_ENABLED` unset → 404, no sampler thread, no-op metric
objects at every call site. Suite 1265 passed (baseline 1183).

Three decisions worth carrying:

- **The gate lives INSIDE the facade.** Disabled returns `_NoopMetric` whose `labels()`
  returns self, so ~30 instrumentation sites carry no branches. `if enabled:` at each
  site puts a live boolean in hot paths that drift apart the first time someone adds one.
- **The scrape path NEVER fans out.** `DeviceMemory.snapshot()` round-trips every
  consumer, and under subprocess isolation that is a control-pipe request/reply. A
  sampler thread owns the cadence; `/metrics` renders memory only. Ten scrapers cost
  the same as one, and no blocking round-trip sits in an ASGI handler — the starvation
  shape that already makes `/status` time out during a job.
- **Job terminals are derived at ONE choke point.** Every dispatch-loop terminal branch
  funnels through `_finalize_job_record` with the future already resolved, so the
  outcome is read there rather than passed from five sites inside the loop's `try`.
  New `JobRecord.enqueued_at` — `executing_since` existed, the enqueue stamp did not,
  so queue wait was not derivable at all.

Traps worth carrying to the next pillar:

- **`/metrics` MUST be registered before `app.mount("/", StaticFiles)`**
  (`lcm_sr_server.py:982` vs `:1008`). That mount matches every path and Starlette
  routes in registration order. **Invisible on a dev box** — no `ui-dist`, mount
  skipped — and 404s only in the deployed image.
- **`fut.exception()` RAISES on a cancelled future**, so `_terminal_outcome` checks
  `cancelled()` first. Not hypothetical: `cancel_pending_generation_jobs` calls
  `fut.cancel()` on queued jobs (`governor.py:683`).
- **`pid` looks like a bounded label and is not** — the subprocess handle mints a new
  one per kill+respawn, leaking a dead series per recovery. Excluded alongside
  `job_id` (unbounded) and `hostname` (Prometheus supplies `instance` from the target).
- **Prometheus default buckets stop at 10s**, which bins every real generation and
  mode load into `+Inf`. Explicit buckets per histogram.
- **`prometheus_client` strips `_total` from Counter family names** in `collect()`.
- **`Mock` is the wrong double wherever production branches on `getattr(x, name, default)`** —
  it auto-creates the attribute, so a negative test can pass without exercising anything.
  Cost a test that was green for the wrong reason.
- **`get_worker_pool()` CONSTRUCTS a pool** (loading a model). Never call it to read
  state; read `app.state.worker_pool`.
- **`_publish_mode_active` iterates `mode_config.list_modes()`**, and most Governor
  tests supply a `Mock` whose `list_modes()` is not iterable. Production survives only
  because the call sits inside `Governor._metric`. That guard is load-bearing — anyone
  adding a metrics call to `governor.py` outside `_metric` will find out the hard way.

**Two spec amendments, ratified at review.** `timeout` is NOT a terminal outcome — it is
a waiter-side budget breach counted by `st_governor_wait_expired_total{budget}`, and the
job that follows reaches the dispatch loop as a *cancel*; counting both double-counts.
And no `hostname` label, per above.

**Method note.** Four of the seven planned tasks contained defects that only surfaced in
execution: a disabled sampler that still fanned out, a non-positive interval that spun at
~62k control-pipe round-trips/sec (`Event.wait(0)` returns immediately — measured 12,472
samples in 0.2s), a sampler wired to the model-loading singleton accessor, and a contract
test that checked 3 of 20 families because it read a *rendered body* rather than the
registry. Each was fixed with a test and the plan doc patched so a re-run does not
reproduce them. **A plan is a hypothesis, not a specification.**

Remaining umbrella children, all unblocked by this: **`STABL-xmsrxvto`** (HTTP/WS
metrics — depends on this facade; label on the matched route template, never
`request.url.path`), **`STABL-cxbwwgly`** (leaked sem/shm/fd gauges — sibling, sample
INSIDE the container), **`STABL-bpsfmoke`** (structured logging — must also cover
`_worker_main` in the spawned child and the materialised `/app/logging_config.json` dev
path; `job_id` needs a contextvar RESET per dispatch-loop iteration), **`STABL-qnlaclof`**
(tracing — `OTEL_PROXY_ENDPOINT` is a full `/v1/traces` path for the browser proxy and
cannot be reused by an SDK exporter).

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
- **`STABL-anxqlxkm` — FIXED** (`9da1c19`, PR #45), and it exposed a second bug,
  **`STABL-hdzggeir` — FIXED** (`8af2f59`, PR #46). Both were the same root cause:
  **every backend module binds whatever `sys.modules['torch']` held when IT was first
  imported**, so two backend modules can hold DIFFERENT torch objects in one pytest
  session depending on file order. Consequences worth carrying:
  - **Patch where the call happens, not on a facade.** `empty_cache()` is called by
    `InProcessWorkerHandle.unload()`, so patching `backends.worker_pool.torch...` landed
    on a MagicMock while the guard read the real `is_available() == False` and the work
    never ran. The assertion failed because the work did not happen, not because the
    mock was bypassed.
  - **It reaches PRODUCTION code.** `isinstance(e, torch.cuda.OutOfMemoryError)` raises
    `TypeError` against a Mock, and that ran inside the dispatch loop's own `except`
    handler — so the handler died before delivering the error, the future never
    resolved, the dispatch thread ended, the queue went permanently dead, and
    `shutdown()` blocked forever on `q.join()`. An error handler that can itself raise
    is a wedge waiting to happen; `_deliver_job_failure` is now wrapped and `_is_oom()`
    cannot raise.
  - **A green full suite does not clear this class.** Both instances were masked at
    full-suite scope (1150+ passed, 0 failed) and reproduced only on a narrower file
    selection. Reproduce with the two-file command, not with CI.
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

