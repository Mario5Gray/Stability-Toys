# Server-side observability seams — Prometheus, Loki, Tempo

**Issue:** STABL-oxbwjwvu (umbrella)
**Children:** STABL-asawxgvp (Prometheus substrate), STABL-xmsrxvto (HTTP/WS metrics),
STABL-bpsfmoke (structured logging), STABL-qnlaclof (traces)
**Date:** 2026-08-03
**Status:** approved (Sigma, 2026-08-03)

## Problem

The server emits nothing an operator can scrape, query, or correlate. `/api/models/status`
is a point-in-time JSON blob a human has to poll by hand; there is no queue-pressure
history, no failure-rate series, no VRAM trend, no structured log a query language can
filter, and no trace. Every diagnosis in this repo's recent history — the VRAM umbrella,
the mode-switch race, the timeout split, the attribution work — was reconstructed after
the fact from ad-hoc spikes and `nvidia-smi` transcripts, because there was no standing
signal to read.

The one OTLP path that exists (`server/telemetry_routes.py`, plus the `telemetry:otlp` WS
handler) is a **browser telemetry proxy**: it forwards UI-originated spans to
`OTEL_PROXY_ENDPOINT`. It emits nothing about the server itself. That distinction matters
later — see §7.

## Scope boundary

Per `AGENTS.md`, shared build/platform architecture lives in `../continuous/docs`. This
repo owns **emission seams only**. Scrape jobs, collectors, promtail/alloy, Tempo/Loki/
Grafana deployment, retention, dashboards and alert rules are all out of scope here.

The consequence, which is easy to miss: **the metric names and label sets are the
cross-repo interface.** If this repo does not publish them, `../continuous` guesses. §8
makes that a deliverable rather than an accident.

## 1. The facade, and where the gate lives

One repo-local module — `server/metrics.py` — is the only thing that imports
`prometheus_client`. Runtime code (`backends/governor.py`, `backends/device_memory.py`,
routes, WS handlers) imports the facade and nothing else.

**The gate lives inside the facade, not at the call sites.** When metrics are disabled the
facade hands back no-op metric objects whose `.inc()` / `.observe()` / `.set()` do nothing.
Instrumentation code is therefore unconditional:

```python
JOB_TERMINAL.labels(mode=mode, outcome="ok").inc()   # always safe, never branched
```

The alternative — `if metrics_enabled:` at each site — puts a live boolean in ~30 hot
paths, and the branches drift out of sync the first time someone adds a site.

**Gate:** `METRICS_ENABLED` (default **off**). The umbrella's own success criterion is that
default/shared repo behaviour stays inert unless configured; the gate is what discharges it.
Disabled means: no `/metrics` route, no sampler thread, no-op metric objects.

**Dependency:** `prometheus_client` goes in `requirements.txt` (pure Python, no transitive
deps — unlike the Compel situation that forced `--no-deps` in
`requirements-conditioning.txt`). The facade nonetheless tolerates `ImportError` by
degrading to the no-op objects, so a dev env or test run without it does not break the
import graph. This mirrors `device_memory.py`'s `_import_pynvml()` / `NullDeviceMemory`
degradation pattern — degrade, never borrow.

## 2. The scrape path must never fan out

This is the highest-risk decision in the substrate.

`DeviceMemory` exposes two reads with deliberately different costs
(`backends/device_memory.py:86-88`):

- `snapshot()` — fresh; **fans out to every registered consumer**
- `cached_snapshot()` — last computed; **no fan-out**

In subprocess isolation the worker consumer's `pool_stats()` is a request/reply over the
child's control pipe (`worker_handle_subprocess.py:435`, `request_stats`). A naive
`/metrics` handler calling `snapshot()` would put that round-trip on the scrape path: every
15s from Prometheus, plus every human `curl`, plus every duplicate scraper. And a
synchronous collector inside the ASGI app is an event-loop starvation vector — `/status`
already times out during a job for exactly this reason.

**Decision: a background sampler owns the fan-out; the scrape path reads memory only.**

- A `MetricsSampler` daemon thread calls `device_memory.snapshot()` every
  `METRICS_SAMPLE_INTERVAL_S` (default `15`) and writes the results into gauges.
- `/metrics` renders the registry. It touches no pipe, no lock held by the dispatch loop,
  and no device driver.
- Fan-out cadence is therefore decoupled from scrape cadence. Ten scrapers cost the same
  as one.
- The sampler wraps its body in a bare `except Exception: logger.exception(...)`. A
  sampler that dies silently is worse than no sampler, and a sampler that kills its thread
  on one bad read stops all device metrics permanently.

**Sampling during a live job is safe, by prior design.** The control channel is a
*separate* pipe from the data pipe precisely because `drain_to_subscriber` reads the data
pipe concurrently during a job — an interleaved stats request on the data pipe would be
consumed as a job frame. That separation was built for this. A stats timeout is not an
error path: `DeviceMemorySnapshot.stale` goes `True` and is itself exported (§5).

## 3. `/metrics` must be registered before the UI static mount

`server/lcm_sr_server.py:971-976`:

```python
if UI_DIST and os.path.isdir(_ui_dist):
    app.mount("/", StaticFiles(directory=_ui_dist, html=True), name="ui")
```

Starlette matches routes in registration order and `Mount("/")` matches everything. **Any
route registered after that mount is shadowed** when the UI dist is present.

The trap is that it is invisible where it will be tested: on a dev box with no
`/opt/lcm-sr-server/ui-dist` the mount is skipped and a late-registered `/metrics` works
fine. It 404s (or serves `index.html`) only in the deployed image.

`/metrics` is registered alongside `/health` (`:946`), well above the mount.

## 4. Single-process assumption, stated and pinned

`prometheus_client`'s default registry is **process-local**.

Production runs `start.sh` → `python -m server.run` → `uvicorn.run(app, ...)` with no
`workers` argument: one process. (`NUM_WORKERS` in `lcm_sr_server.py` is *pipeline*
workers, and is forced to 1 under CUDA anyway — unrelated, and easy to confuse.)

**Decision:** single-process is a stated precondition, not an accident. The facade logs a
loud `WARNING` at init if `WEB_CONCURRENCY` is set above 1, because in that world
`/metrics` silently reports whichever worker answered the scrape — counters that appear to
go backwards. Multiprocess mode (`prometheus_client.multiprocess`) is an explicit non-goal
until someone actually needs multiple uvicorn workers; it costs a shared directory, a
custom registry, and loses gauge semantics.

**The worker child owns no registry.** Anything the child knows reaches Prometheus through
the existing parent-side control pipe. A `prometheus_client` import in `_worker_main` would
build a registry nothing ever scrapes.

## 5. Metric families

Namespace prefix `st_`. Label policy, first pass: **`device_uuid` and `mode` only**, plus
`outcome` / `reason` / `consumer` / `budget` where a family declares them.

**Amended 2026-08-03 (ratified by Sigma at plan review): no `hostname` label.** It was in
the original allowed list. Prometheus already supplies target identity as the `instance`
label from the scrape target, so an in-app host label duplicates it and goes stale when the
container moves. `hostname` stays in the structured-log field set (§7.4), where it is
useful for correlation and costs no cardinality.

Two further labels are excluded on cardinality grounds and the reasons are not symmetric:

- **`job_id`** — unbounded and monotonically growing. It belongs in logs (§6), which is
  where the per-job question is actually answerable.
- **`pid`** — this one looks bounded and is not. `ConsumerMemory` carries a pid, and the
  subprocess handle mints a **new pid on every kill+respawn** (facet-3 recovery: `153 → 261`
  in the OOM acceptance). Exporting pid as a label leaks a dead series per recovery. The
  consumer `label` (`"server"` / `"worker"`) is the bounded key; pid stays in logs.

### Governor

| Metric | Type | Labels |
|---|---|---|
| `st_governor_queue_depth` | gauge | — |
| `st_governor_jobs_in_flight` | gauge | — |
| `st_governor_job_queue_wait_seconds` | histogram | `mode` |
| `st_governor_job_execution_seconds` | histogram | `mode` |
| `st_governor_job_terminal_total` | counter | `mode`, `outcome` |
| `st_governor_wait_expired_total` | counter | `budget` (`admission` / `execution`) |
| `st_governor_mode_load_seconds` | histogram | `mode` |
| `st_governor_mode_switch_total` | counter | `mode` (target) |
| `st_governor_demand_reload_total` | counter | `mode` |
| `st_governor_unload_total` | counter | `mode`, `reason` |
| `st_governor_worker_recovery_total` | counter | `reason` (`oom` / `dead`) |
| `st_governor_mode_active` | gauge | `mode` |
| `st_governor_resolution_epoch` | gauge | — |

`outcome` ∈ `ok` / `cancelled` / `oom` / `error` — the existing terminal classification,
not a new taxonomy.

**Amended 2026-08-03 (ratified by Sigma at plan review): `timeout` is NOT a terminal
outcome.** The original enum included it. It does not belong there: a timeout is a
*waiter-side* budget breach raised at `_expire` (`governor.py:716`), and what subsequently
reaches the dispatch loop for that job is a **cancel**. Counting it as a terminal would
either double-count the same job (once as `timeout`, once as `cancelled`) or misreport
which of the two actually happened. It gets its own series instead:

- `st_governor_wait_expired_total{budget}` answers "how often do requests time out"
- `st_governor_job_terminal_total{mode,outcome}` answers "what happened to the job"

This also keeps the terminal counter derivable from a single choke point — see the
implementation plan's note on `_finalize_job_record`.

`st_governor_mode_active` reports **0 or 1 for every configured mode**, rather than a
single gauge labelled with the current mode. The latter leaves a stale `1` on the previous
mode's series forever; the set of modes is bounded by `mode_config`, so reporting all of
them is cheap and correct.

`st_governor_mode_switch_total` labels the **target only**. A `{from, to}` pair squares the
cardinality to buy a transition matrix nobody has asked for.

#### `JobRecord` needs an `enqueued_at`

Queue-wait is not currently derivable. `JobRecord` (`governor.py:164-178`) carries
`executing_since` — added by `STABL-atzqpcte`, stamped after the demand reload and after
the stale-epoch barrier, which is exactly the boundary the two histograms want — but there
is **no enqueue timestamp**. `_register_job` (`:585`) constructs the record with
`state="queued"` and nothing else.

Add `enqueued_at: Optional[float] = None`, stamped monotonic in `_register_job`. Then:

- queue wait = `executing_since - enqueued_at`
- execution = `now - executing_since`

**Both must be observed before the record is popped.** `_finalize_job_record` (`:592`)
deletes it, and both the success path and `_deliver_job_failure` (`:1055`) call it. The
observation goes immediately before finalize, on every terminal branch — the cancel branch
in the dispatch loop (`:895`) included, which is easy to miss because it `continue`s.

**Nothing added here may be able to kill the dispatch loop.** `STABL-hdzggeir` landed on
this branch for precisely that failure mode — the error handler itself throwing. Metric
calls are no-op-safe by construction (§1), but the timestamp arithmetic is not: a `None`
`enqueued_at` on a record built by an older path would raise. The observation helper
tolerates `None` by skipping, and the dispatch-loop call sites do not grow a new
`try`/`except` — the helper cannot raise.

### DeviceMemory

| Metric | Type | Labels |
|---|---|---|
| `st_device_total_bytes` | gauge | `device_uuid` |
| `st_device_free_bytes` | gauge | `device_uuid` |
| `st_device_used_bytes` | gauge | `device_uuid` |
| `st_device_unattributed_bytes` | gauge | `device_uuid` |
| `st_consumer_reserved_bytes` | gauge | `device_uuid`, `consumer` |
| `st_consumer_allocated_bytes` | gauge | `device_uuid`, `consumer` |
| `st_device_snapshot_stale` | gauge | `device_uuid` |

All written by the sampler (§2), all straight off `DeviceMemorySnapshot` /
`ConsumerMemory` — no new accounting. `unattributed_bytes` is the one that made the
attribution work legible (1.93 GiB with the child registered, 3.93 without) and it is the
single most useful VRAM series here.

`device_uuid` is available today from `CudaDeviceMemory`. `STABL-cchxvuhs` (global
UUID-keyed GPU identity) is still `todo`, so in practice this is one series per family
until multi-GPU lands — the label is forward-compatible, not speculative.

`st_device_snapshot_stale` is a first-class series, not a footnote. `stale=True` means a
consumer fan-out timed out, i.e. **the child is not answering** — an early wedged-worker
signal that nothing else surfaces.

### Adopt `STABL-cxbwwgly`

Leaked-resource counts (`/dev/shm/sem.*`, shm segments, fds) were filed standalone as the
observability follow-on to the accepted semaphore-leak risk. They are gauges on this
substrate — `st_process_leaked_semaphores`, `st_process_open_fds` — and the reason that
issue exists is that the growth is *unbounded and currently invisible*. Now re-parented
under `STABL-oxbwjwvu` and made dependent on `STABL-asawxgvp`, since without the facade
there is nothing for it to emit through. **Sample inside the container:** `/dev/shm` is
per-mount-namespace and a host-side read measures a different one.

## 6. HTTP and WebSocket metrics — STABL-xmsrxvto

Reuses the §1 facade; adds no infrastructure.

**Label on the matched route template, never the raw path.** `request.url.path` makes
`/api/models/{name}` one series per model name and `/v1/storage/{key}` one series per
object — unbounded. Starlette exposes the matched route via
`request.scope["route"].path`; a request that matched nothing is labelled `__unmatched__`
rather than by its literal path (that is exactly the unbounded set a scanner probes).

| Metric | Type | Labels |
|---|---|---|
| `st_http_requests_total` | counter | `method`, `route`, `status` |
| `st_http_request_duration_seconds` | histogram | `method`, `route` |
| `st_ws_connections_active` | gauge | — |
| `st_ws_sessions_total` | counter | — |
| `st_ws_messages_total` | counter | `type`, `direction` |

WS `type` is drawn from the bounded `_handler` registry (`ws_routes.py:167`) —
`ping`, `job:submit`, `job:cancel`, `job:priority`, `telemetry:otlp`. An unrecognised
inbound type is labelled `unknown`; the client controls that string, so it must not reach
a label value.

HTTP instrumentation is one ASGI middleware, added with the CORS middleware
(`lcm_sr_server.py:982`). WS instrumentation hooks `websocket_endpoint` (`:836`) at
accept/disconnect and the dispatch point in the message loop.

## 7. Structured logging — STABL-bpsfmoke

`server/logging_config.py` stays the single source of truth. Add a JSON formatter and a
`LOG_FORMAT` env (`text` | `json`, default **`text`** — inert by default, same rationale as
§1). The existing text formatter is unchanged.

Three things stand between "add a formatter" and logs that are actually queryable, and the
FP issue names none of them.

### 7.1 Both entry paths must get the config

There are two, and they are configured differently:

- **prod:** `start.sh` → `python -m server.run` → `logging.config.dictConfig(LOGGING_CONFIG)`
  *and* `uvicorn.run(..., log_config=LOGGING_CONFIG)`.
- **dev:** `docker/runtime/live-test.Dockerfile:35` materialises `LOGGING_CONFIG` to
  `/app/logging_config.json` at build time and passes `--log-config`, because the dev CMD
  imports the app so the `__main__` block never runs. That is `92d09fc`, the sub-fix
  already logged on this umbrella.

A formatter reachable only through the Python dict works in prod and silently does nothing
in dev. The JSON formatter must be expressible in the materialised dict — i.e. referenced
by dotted path in `LOGGING_CONFIG`, resolvable inside the image — not attached
imperatively in `run.py`.

### 7.2 The subprocess child logs into the same stdout, configured by nobody

`_worker_main` (`worker_handle_subprocess.py:180`) runs in a spawned child that inherits
the parent's stdout and never sees `--log-config` or `run.py`'s `dictConfig`. Under
`WORKER_ISOLATION=subprocess` — the production path — the container's log stream is
therefore JSON from the parent interleaved with default-formatted lines from the child, and
the child is where generation actually happens.

`_worker_main` applies the same config early, before `import torch`. In scope for this
issue; a JSON logging story that omits the worker is not a logging story.

### 7.3 `print()` — 43 calls, and they are not all the same

`grep -c` across `server/` and `backends/` finds 43, in six files. They bypass `logging`
entirely: no level, no formatter, no structure, straight to the same stdout a JSON parser
is reading.

- **In scope:** `server/lcm_sr_server.py`, `server/superres_service.py`,
  `server/advisor_service.py`, `backends/cuda_worker.py`, `backends/rknnlcm.py` — server
  runtime, all of it lifecycle or job-path output.
- **Explicitly out:** `server/superres_cli.py`. It is a CLI; `print` *is* its output
  contract. Converting it would be a regression.

Stated as a decision rather than left to whoever runs the grep.

### 7.4 `job_id` correlation is plumbing, not a field

The FP issue lists `job_id` among the fields to include. Getting it there is the work.

Governor logging happens on the **dispatch thread**; WS and HTTP logging happen on the
**event loop**. There is no shared frame to pass it through, and threading it as an
explicit argument through every call site is the thing structured logging exists to avoid.

**Decision: a `contextvars.ContextVar` read by the JSON formatter**, set at two points —
WS/HTTP handler entry (request/job scope) and the top of each dispatch-loop iteration.

The non-obvious half: **contextvars do not inherit into a long-lived thread.** The dispatch
loop is one thread processing many jobs; it copies nothing from the submitter. It must
**set the var per job and reset it in the `finally`** alongside `q.task_done()` (`:1049`),
or job N's id leaks onto job N+1's logs — worse than an absent field, because it is
plausible.

Field set: `job_id`, `mode`, `device_uuid`, `hostname`, `level`, `logger`, `thread`,
`timestamp`, `message`. `pid` is included here — logs are where per-process identity
belongs (§5).

## 8. The metric-name contract is a deliverable

Dashboards and alerts live in `../continuous`. If the names live only in this repo's source,
that repo either guesses or reverse-engineers a scrape.

`docs/observability-contract.md` (repo-local, per `AGENTS.md`'s allowance for "how
Stability-Toys adopts or consumes that shared platform contract") lists every metric
family, its type, its label set, and its stability expectation, and points at the
authoritative platform docs in `../continuous/docs`. It ships with `STABL-asawxgvp` and is
extended by `STABL-xmsrxvto` — a success criterion on both, not a nice-to-have.

## 9. Tracing — STABL-qnlaclof

Last pillar. The FP text's ordering claim ("once metrics and structured logs are in place")
is now backed by real dependencies on `STABL-asawxgvp` and `STABL-bpsfmoke` rather than
living only in prose.

**`OTEL_PROXY_ENDPOINT` cannot be reused as written.** It is
`http://otel-collector:4318/v1/traces` (`env.dev:4`, `env.prod:4`) — a full *signal path*,
correct for the browser proxy that POSTs to it directly. OTel SDK exporters take a **base**
endpoint and append the signal path themselves; handing them this value yields
`/v1/traces/v1/traces`. Use the standard `OTEL_EXPORTER_OTLP_ENDPOINT`, and leave the
browser proxy untouched — it is a different concern that happens to share a collector.

First pass: request-path spans plus Governor lifecycle spans (admit → queue → load →
execute → terminal). Gated the same way, default off. No OTel *metrics* — Prometheus owns
that pillar and dual-exporting the same numbers is how the two disagree.

## 10. Test strategy

TDD is mandatory here; each child states RED/GREEN, and all of this tests without hardware:

- **Facade (`asawxgvp`)** — disabled facade returns no-op objects and instrumented code
  runs unchanged; enabled facade registers families; `ImportError` degrades rather than
  raises.
- **Sampler** — asserts `cached_snapshot` is never called from the render path and
  `snapshot` is never called from it either; a raising `snapshot()` does not kill the
  thread. Both are behavioural assertions on a fake `DeviceMemory`, no GPU.
- **Route ordering** — a test that `/metrics` resolves with a UI dist present. This is the
  §3 trap and it is the one bug in this spec that testing on a dev box will not find; the
  test builds an app with a stub static dir.
- **Cardinality** — assert `job_id` and `pid` appear in no label set, and that the HTTP
  middleware labels a parameterised route by template.
- **Governor timing** — `enqueued_at` stamped at register; both durations observed before
  finalize on every terminal branch including cancel; the observation helper cannot raise
  on a `None` timestamp.
- **Logging** — JSON formatter emits the declared field set; `job_id` contextvar is reset
  between two sequential jobs on one dispatch thread (the §7.4 leak, asserted directly).

Governor tests inherit the existing invariants: `gov._stop.set()` does **not** stop the
dispatch loop — use `_freeze_dispatch`; and `shutdown()` begins with `q.join()`, so leftover
queued jobs need `_drain_queue`.

## Non-goals

- No collector, promtail/alloy, Grafana, Tempo or Loki deployment in this repo.
- No dashboards, alert rules, scrape config or retention policy here.
- No OTel metrics.
- No multiprocess `prometheus_client` registry (§4).
- No change to the existing browser OTLP proxy.
- No conversion of `server/superres_cli.py`'s `print` output (§7.3).

## Resolved questions (Sigma, 2026-08-03)

1. **`METRICS_ENABLED` defaults to `off`.** It matches the umbrella boundary that default
   repo behaviour stays inert unless configured, and the facade's no-op objects (§1) remove
   the usual ergonomics penalty for a default-off gate — nothing at the call sites changes.
2. **Sampler interval defaults to 15s, env-overridable via `METRICS_SAMPLE_INTERVAL_S`.**
   No shared scrape interval is declared under `../continuous/docs`, so there is nothing
   authoritative to align to yet. 15s is the local default; tighten or relax it once
   `../continuous` declares a scrape cadence. The env override is what keeps that a config
   change rather than a code change.
3. **`st_governor_mode_active` reports all configured modes.** Verified against
   `conf/modes.yml`: four modes — `lcm-general`, `lcm-runwayml`, `SDXL`, `HunyuanDiT`.
   Comfortably bounded, so 0/1 per configured mode is correct and the stale-series problem
   of a single mode-labelled gauge is avoided.
