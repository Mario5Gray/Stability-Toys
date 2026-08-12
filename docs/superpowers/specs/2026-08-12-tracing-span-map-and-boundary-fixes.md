# Tracing span map and boundary-observation fixes — STABL-qnlaclof

**Status:** design, not yet planned
**Issue:** `STABL-qnlaclof` (last child of umbrella `STABL-oxbwjwvu`)
**Depends on:** `STABL-zuhuxwvf` (job_id does not cross the spawn boundary) — see §4
**Supersedes nothing.** Extends §9 of
[`2026-08-03-server-observability-seams-design.md`](2026-08-03-server-observability-seams-design.md),
which named the pillar and the `OTEL_PROXY_ENDPOINT` trap but did not map the spans.

---

## 1. What this document is for

Two questions, answered concretely:

1. **Where do the spans go?** §3 is the map, with file and line for every seam.
2. **What has to change for a span to survive the boundaries it crosses?** §4. This is
   the larger half, and it is where the work actually is. The instrumentation is
   mechanical; the boundaries are not.

The deployment precondition is already met. An otel-collector runs on enigma on
`observ-net`, `stability-toys` is on the same network, and a span pushed from inside
the application container was read back from node1's Tempo on 2026-08-12. That
proof is recorded on the FP issue. Nothing in this document is gated on
infrastructure.

---

## 2. Layer decision: interface, not hardware

**Spans go on the framework and interface seams. No span is placed in
`cuda_worker.py`, `rknn_worker.py`, or any other device-specific module.**

This is not a style preference. Three measurements decide it.

**`run_job` has four implementations and they do not agree on a signature.**

| implementation | signature |
|---|---|
| `backends/cuda_worker.py:1072` | `run_job(self, job, progress=None, should_cancel=None)` |
| `backends/cuda_worker.py:1455` | same |
| `backends/cuda_worker.py:1808` | same |
| `backends/rknn_worker.py:58` | `run_job(self, job)` — **no `progress`, no `should_cancel`** |

Instrumenting there means four sites that must not drift, and the RKNN one cannot
even be wrapped identically. Contrast the interface seams: `WorkerHandle.submit()`
has **two** implementations (`worker_handle.py:137`, `worker_handle_subprocess.py:514`)
and the Governor dispatch loop has **one** (`governor.py:1025`).

**The metrics pillar already proved the layering.** `STABL-asawxgvp` shipped 20
families across ~30 call sites with **zero** instrumentation in `cuda_worker.py`.
Everything it needed was reachable from the Governor, the handle, and DeviceMemory.
Tracing needs strictly less than metrics did — it needs the *shape* of a job's
progress, and the shape is entirely defined by the control plane.

**There is already a hardware-neutral seam inside generation, and it is not in the
workers.** `backends/step_progress.py:24` `inject_step_progress` is installed into
*every* family's denoise loop — `callback_on_step_end` where the pipeline supports
it, the legacy `callback`/`callback_steps=1` pair otherwise. `STABL-jredufxb` already
used it for cooperative cancellation at step granularity. If sub-job spans are ever
wanted (per-stage, per-N-steps), that is where they go. Reaching into a family's
pipeline call to get inside a generation is unnecessary.

> **Trap, inherited from `STABL-jredufxb`.** `_emit` (`step_progress.py:40`) swallows
> every exception on purpose — *"a bad consumer must never break generation"*. Any
> span work placed inside `_emit` therefore fails **silently**. It belongs in the
> `_modern`/`_legacy` wrapper, outside that swallow, exactly where the cancel check
> was put for the same reason.

---

## 3. The span map

Names use the OTel convention `<component>.<operation>`. All spans are `INTERNAL`
except the HTTP/WS entry points (`SERVER`) and the handle boundary (`PRODUCER`) with
its child-side counterpart (`CONSUMER`).

### 3.1 Entry — where a trace begins

| span | site | notes |
|---|---|---|
| `http.server.request` | `server/metrics_middleware.py:31` `MetricsMiddleware.__call__` | Reuse the existing pure-ASGI middleware rather than adding a second one. |
| `ws.message` | `server/ws_routes.py:915` (`with log_context.bind_job_id(None)`) | One span per inbound message. Same braces as the `job_id` bind — the two correlate the same unit and must not diverge. |

`route` must be the matched route **template**, read *after* the downstream app runs
— `server/metrics_middleware.py:17` `route_label()` already does exactly this and
returns `__unmatched__` for a request that matched nothing. Use that function; do not
re-derive from `request.url.path`, which mints one span name per model id.

Inbound WS `type` is client-controlled. It must map through the bounded `HANDLERS`
registry before it can become a span name, the same way it does for the metrics
label — `unknown` for unrecognised, `invalid_json` for unparseable.

> `raw in HANDLERS` and `HANDLERS.get(raw)` both **hash** their argument, so a client
> sending `{"type": {}}` raises `TypeError` (`STABL-gzfzzsdq`, still open). Guard with
> `isinstance(msg_type, str)` before using it for anything, span names included.

### 3.2 Governor — the control plane

| span | site |
|---|---|
| `governor.submit` | `governor.py:1280` `submit_job` — opens the backplane channel, subscribes the bridge, enqueues |
| `governor.admit` | admission barrier: epoch/authority check |
| `governor.mode_switch` | `governor.py:1319` `switch_mode` → `governor.py:1334` `_reserve_and_enqueue_switch` |
| `governor.dispatch` | `governor.py:1025` `_dispatch_loop` — **the job span**, opened at `:1048` and closed at `:1220` |
| `governor.mode_load` | `governor.py:522` `_load_mode` |
| `governor.reload` | `governor.py:605` `_reload_from_snapshot` (demand reload after idle eviction) |
| `governor.unload` | `governor.py:916` `_unload_current_worker` |
| `governor.wait` | `governor.py:785` `wait_for_result` — the waiter side; attribute the budget (`execution` / `admission`) |
| `governor.cancel` | `governor.py:864` `cancel_job` |

**`governor.dispatch` shares its lifetime exactly with the `job_id` contextvar bind**
— set at `governor.py:1048`, reset in the `finally` at `governor.py:1220`. That
`finally` is the only path that runs on every exit including the cancel-check
`continue`. Put the span's `end()` in the same block, for the same reason.

**Terminal status is derived at one choke point.** `governor.py:652`
`_finalize_job_record` already receives the resolved future and derives the metrics
outcome there rather than at five branches inside the loop's `try`. The span status
is set from the same place, from the same value. Two derivations of "how did this job
end" is how a trace and a counter start disagreeing.

> `fut.exception()` **raises** on a cancelled future. `_terminal_outcome` checks
> `cancelled()` first. Not hypothetical: `cancel_pending_generation_jobs`
> (`governor.py:743`) calls `fut.cancel()` on queued jobs at `governor.py:753`, and
> `cancel_job` does the same at `governor.py:878`.

### 3.3 The worker boundary

| span | site | kind |
|---|---|---|
| `worker.submit` | `WorkerHandle.submit()` — `worker_handle.py:61` (ABC) | PRODUCER |
| `worker.execute` | in-proc: `worker_handle.py:137` `_run()` thread | INTERNAL |
| `worker.execute` | subprocess: `worker_handle_subprocess.py:211` `_worker_main` job loop | CONSUMER |

This is the only place in the map where the same logical operation has two physically
different implementations, and §4 is entirely about making that seam observable.

Attribute `isolation` (`inproc` / `subprocess`) on `worker.submit`. Without it a trace
cannot be read without separately knowing how the deployment was configured, and the
two shapes are legitimately different — not a regression to be alarmed at.

### 3.4 Device memory

| span | site |
|---|---|
| `device.snapshot` | `device_memory.py:138` / `:212` `snapshot()` — **fresh, fans out to consumers** |

`cached_snapshot()` (`:172`, `:215`) gets **no span**. It performs no I/O by
definition, and the registry view is pure `cached_snapshot` precisely so a wedged
worker cannot hang `/status`. Spanning it adds cost to the path that was made cheap
on purpose.

`snapshot()` is worth a span because under subprocess isolation each consumer read is
a **control-pipe request/reply** — a real round trip, on a dedicated pipe, with a
timeout that substitutes last-known values and sets `stale=True`. That timeout is
currently visible only as a gauge (`st_device_snapshot_stale`); a span shows *which*
consumer was slow and *how* slow.

### 3.5 Deliberately not instrumented

- **`cuda_worker.py`, `rknn_worker.py`** — §2.
- **`cached_snapshot()`** — §3.4.
- **Per-denoise-step spans.** One span per step at up to 50 steps per job is a
  cardinality decision, not an observability one. The seam exists
  (`step_progress.py`) if it is ever wanted; the first pass does not take it.
- **OTel metrics.** Prometheus owns that pillar. Dual-exporting the same numbers is
  how the two silently disagree. Stated in the spec §9 and restated here because it
  is the most tempting thing an OTel SDK offers for free.

---

## 4. Boundary observation — the fixes this needs

Three boundaries. One is already broken in a measurable way, one needs a wire change,
one needs a config change.

### 4.1 The spawn boundary — `job_id` already fails here

**Measured on live Loki, 2026-08-12, 24h window:**

```
lines by pid (all)     lines WITH job_id     backends.cuda_worker lines
  49   -> 2405           49 -> 4               76   -> 8
  76   -> 10             (nothing else)        241  -> 8
  241  -> 10                                   377  -> 4
  377  -> 5                                    2062 -> 4
  2062 -> 5                                    2406 -> 4
  2406 -> 5
```

pid 49 is the server. The rest are spawned worker children, one pid per respawn
generation. **`backends.cuda_worker` lines come only from child pids, and not one
carries `job_id`.** Correlation works where generation does not happen and is absent
where it does.

Cause: `governor.py:1048` binds a `ContextVar` in the **parent**. Under
`WORKER_ISOLATION=subprocess` the work runs in `_worker_main`
(`worker_handle_subprocess.py:211`) in a **different process**. A `ContextVar` cannot
cross a spawn boundary. `_configure_child_logging` (`:180`) calls
`refresh_process_fields()`, so child lines correctly carry their own `pid` and
`hostname` — nothing binds `job_id`, even though the child holds `d.job_id` from
`decode_job`.

It is correct in-proc, which is what every test covers. It fails only in the deployed
topology.

**Fix — `STABL-zuhuxwvf`, do this first.** Bind `job_id` around the job body in the
child's loop with the same set/reset discipline the parent uses. `job_id` is already
a `CARRIED_JOB_FIELDS` member (`job_envelope.py:20`), so nothing changes on the wire.
RED must be observed under subprocess isolation; an in-proc test cannot fail.

**Why it gates tracing.** Trace context hits the identical wall one layer up: a span
started in the parent cannot be continued in the child without crossing the same
boundary. Fixing `job_id` first establishes both the pattern and the test shape that
trace propagation reuses. Doing them in the other order means designing the envelope
change before the simpler case has proven where the seams are.

> Set-without-reset is the failure that matters here, in both cases. A stale
> correlation id reads as *real* correlation and survives review. `STABL-bpsfmoke`
> pinned this with a test that observes the dispatch loop's own
> `[Governor] Dispatch loop stopped` line carrying no id, via a log `Handler` —
> `emit()` runs on the **emitting** thread, which is the only way to read another
> thread's context from a test. The same technique applies to the child.

### 4.2 Trace context must ride the job envelope

Once §4.1 lands, the same boundary needs `traceparent` (and `tracestate` if
`OTEL_PROPAGATORS` is ever configured beyond W3C default).

`backends/job_envelope.py` is versioned for exactly this: `JOB_SCHEMA_VERSION = 2`, a
leading version byte, and `decode_job` **rejects** unknown versions rather than
default-filling. Adding a carried field is therefore a **schema version bump to 3**,
not an additive change — and the rejection is the feature. `STABL-spxwqlan` was
caused by a field silently taking its dataclass default across the boundary; a
mixed-version pair would reproduce it exactly, which is why the codec refuses.

Required changes:

1. `JOB_SCHEMA_VERSION = 3`.
2. `traceparent` (and optional `tracestate`) added to `CARRIED_JOB_FIELDS`, injected
   parent-side in `SubprocessWorkerHandle.submit()` (`worker_handle_subprocess.py:514`).
3. Child extracts in `_worker_main`'s loop and starts `worker.execute` as a CONSUMER
   span linked to the parent's PRODUCER span.
4. The child must initialise its own tracer provider in `_configure_child_logging`'s
   sibling position — **before** heavy imports, wrapped, degrading to no-op on
   failure. It inherits stdout but no SDK state, precisely as it inherits no logging
   config.

**Absent context must produce a root span, never a dropped one.** A child that
receives no `traceparent` (tracing disabled parent-side, or a v2 parent against a v3
child during a rolling change) must still emit. Silence there is indistinguishable
from a healthy idle worker.

**The sampling decision crosses with the context.** `traceparent` carries the sampled
flag, so parent-based sampling works for free — but only if the child's provider is
configured `ParentBased`, not a fresh independent sampler. An independent sampler in
the child produces traces whose parent span was never recorded.

### 4.3 The exporter endpoint — do not reuse `OTEL_PROXY_ENDPOINT`

`env.dev:4` and `env.prod:4` set:

```
OTEL_PROXY_ENDPOINT=http://otel-collector:4318/v1/traces
```

That is a full **signal path**, and it is correct for what uses it: the browser
telemetry proxy, which POSTs to it directly (`server/telemetry_routes.py:26`,
`server/ws_routes.py:372`). OTel SDK exporters take a **base** endpoint and append the
signal path themselves. Handing them this value yields `/v1/traces/v1/traces`.

Use the standard `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318` and leave
the proxy untouched. Two concerns that happen to share a collector.

> The 2026-08-12 live proof deliberately POSTed the **full** path, because it was
> imitating the browser proxy. It does **not** validate the base-endpoint form.
> Re-run the equivalent probe against `:4318` with the SDK doing the append, as part
> of the acceptance.

---

## 5. Gate and failure posture

**The gate lives inside the facade**, exactly as `server/metrics.py` does it: disabled
returns a no-op tracer whose `start_as_current_span()` is a null context manager, so
call sites carry no branches. `if enabled:` at ~15 sites puts a live boolean in hot
paths that drift apart the first time someone adds one.

**Default off.** `STABL-xqqqqvse` and `STABL-xolucarj` are both the record of a pillar
shipping dark because it was implemented and enabled nowhere. Whatever variable gates
this, set it in `env.prod` **in the same change** and confirm with a real trace, not
by reading config.

**Governor instrumentation must be guarded like `_metric`.** `governor.py:689`
`_metric` exists because `_publish_mode_active` iterates `mode_config.list_modes()`
and most Governor tests supply a `Mock` whose `list_modes()` is not iterable —
production survives only because the call sits inside that guard. Add a `_span`
helper mirroring it. Anyone adding a tracing call to `governor.py` outside a guard
will find out the hard way.

**An observability component that hides its own failure is the worst kind.** Outer
guards log with `exc_info`, pinned by a test asserting both message and traceback
presence — `STABL-cxbwwgly`'s lesson. Without it, an absent trace reads identically
whether the collector is down or the instrumentation broke.

---

## 6. Acceptance

Not "the code is written". Three things, in order:

1. **In-proc:** a generate produces one trace spanning entry → dispatch → execute →
   terminal, read back from `node1.lan:3200`.
2. **Subprocess:** the same generate under `WORKER_ISOLATION=subprocess` produces
   **one** trace, not two. This is the acceptance that matters — it is the only one
   that proves §4.1 and §4.2, and it is the deployed configuration.
3. **Correlation:** the `job_id` on the log lines equals the `job_id` attribute on the
   spans, for **both** pids. Verifiable with the recipes in
   [`../../observability.md`](../../observability.md).

Acceptance 2 fails today for `job_id` and would fail identically for traces. That is
the measurement this document exists to prevent repeating.

---

## 7. Sequencing

| # | work | issue |
|---|---|---|
| 1 | Bind `job_id` in the worker child | `STABL-zuhuxwvf` |
| 2 | Tracer facade + gate, default off | `STABL-qnlaclof` |
| 3 | Entry spans (HTTP, WS) | `STABL-qnlaclof` |
| 4 | Governor spans | `STABL-qnlaclof` |
| 5 | Job envelope v3 + child provider + propagation | `STABL-qnlaclof` |
| 6 | `OTEL_EXPORTER_OTLP_ENDPOINT` + enable in `env.prod` + live acceptance | `STABL-qnlaclof` |

Steps 2–4 are testable without hardware. Step 5 needs a real spawn boundary, which
the existing subprocess tests already provide. Step 6 needs enigma.
