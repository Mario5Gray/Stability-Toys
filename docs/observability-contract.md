# Observability contract — metrics and logs exported by Stability-Toys

**Issues:** STABL-asawxgvp (metrics), STABL-bpsfmoke (logs) — umbrella STABL-oxbwjwvu
**Spec:** `docs/superpowers/specs/2026-08-03-server-observability-seams-design.md`

This repo owns **emission only**. Scrape config, collectors, log shipping,
dashboards, alert rules and retention live in `../continuous/docs` — see
`AGENTS.md`. This document is the interface between the two: it is what
`../continuous` reads instead of guessing or reverse-engineering a scrape.

Every family listed here is checked against the running facade by
`tests/test_metrics.py::test_every_family_is_documented_in_the_contract`, in both
directions — a metric that ships without an entry fails, and an entry for a
metric that no longer exists fails too. The log field set in
[Structured logs](#structured-logs) is checked the same way, by
`tests/test_log_format.py`.

## Endpoint

`GET /metrics` — Prometheus text format, `text/plain; version=0.0.4`,
`Cache-Control: no-store`.

- Gated by `METRICS_ENABLED` (default **off**). Disabled returns **404**.
- Refreshed by a background sampler every `METRICS_SAMPLE_INTERVAL_S`
  (default **15**, must be positive). The scrape path itself performs no device
  or worker round-trip, so scrape cadence and sampling cadence are independent:
  ten scrapers cost the same as one.
- **Scrape interval is not yet aligned across repos.** When `../continuous`
  declares one, revisit the sampler default.
- **Single process only.** `prometheus_client`'s registry is process-local and
  the server runs one uvicorn worker (`server/run.py`). With `WEB_CONCURRENCY > 1`
  the endpoint reports whichever worker answered the scrape and counters appear
  to go backwards; the facade logs a WARNING in that case.

## Label policy

Allowed: `device_uuid`, `mode`, plus `outcome`, `reason`, `consumer` and `budget`
where a family declares them.

Three labels are deliberately absent, for three different reasons:

| Label | Why not |
|---|---|
| `job_id` | unbounded. Available in structured logs (`STABL-bpsfmoke`). |
| `pid` | *looks* bounded and is not — the subprocess worker mints a new pid on every kill+respawn, so a pid label leaks a dead series per recovery. |
| `hostname` | duplicates the `instance` label Prometheus attaches from the scrape target, and goes stale when the container moves. Host identity is the scraper's job. |

Do not add them, and do not go looking for them.

## Governor families

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_governor_queue_depth` | gauge | — | jobs currently queued |
| `st_governor_jobs_in_flight` | gauge | — | jobs past the admission barrier |
| `st_governor_job_queue_wait_seconds` | histogram | `mode` | enqueue → execution start |
| `st_governor_job_execution_seconds` | histogram | `mode` | execution start → terminal |
| `st_governor_job_terminal_total` | counter | `mode`, `outcome` | `ok` / `cancelled` / `oom` / `error` |
| `st_governor_wait_expired_total` | counter | `budget` | `admission` / `execution` budget blown |
| `st_governor_mode_load_seconds` | histogram | `mode` | successful loads only |
| `st_governor_mode_switch_total` | counter | `mode` | **target** mode |
| `st_governor_demand_reload_total` | counter | `mode` | reload after idle eviction |
| `st_governor_unload_total` | counter | `mode`, `reason` | model unloads |
| `st_governor_worker_recovery_total` | counter | `reason` | `oom` / `dead` kill+respawn |
| `st_governor_mode_active` | gauge | `mode` | 1 for the loaded mode, 0 for every other configured mode |
| `st_governor_resolution_epoch` | gauge | — | current authority epoch |

### Things that will mislead you if you assume otherwise

**`wait_expired_total` is not `job_terminal_total{outcome="cancelled"}`.** A timeout
is a *waiter-side* budget breach; the job that follows it is reaped as a cancel.
"How often do requests time out" is the first series. "What happened to the job"
is the second. `timeout` is deliberately **not** an outcome value — counting it in
both places would double-count the same job.

**`mode_load_seconds` counts successful loads only.** A load that raises has no
duration to report. A failed load is visible as `mode_active` being 0 for every
mode.

**`mode_active` reports every configured mode**, not just the loaded one. A single
gauge labelled with the current mode would leave a stale `1` on the previous
mode's series forever. `conf/modes.yml` currently holds four modes.

**`unload_total{reason}` includes routine churn.** `_load_mode` unloads the
outgoing worker before every load, so a mode switch emits `reason="switch"`.
Reasons: `switch`, `idle_evict`, `explicit`, `shutdown`, and the reason passed to
a VRAM cleanup (e.g. `oom`). The very first load emits nothing — there is no
outgoing mode.

**`mode_switch_total` labels the target only.** A `{from,to}` pair would square the
cardinality to buy a transition matrix nobody asked for.

## DeviceMemory families

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_device_total_bytes` | gauge | `device_uuid` | device total |
| `st_device_free_bytes` | gauge | `device_uuid` | driver-truth free |
| `st_device_used_bytes` | gauge | `device_uuid` | driver-truth used |
| `st_device_unattributed_bytes` | gauge | `device_uuid` | used minus every consumer's reserved pool |
| `st_consumer_reserved_bytes` | gauge | `device_uuid`, `consumer` | per-consumer pool, reserved |
| `st_consumer_allocated_bytes` | gauge | `device_uuid`, `consumer` | per-consumer pool, allocated |
| `st_device_snapshot_stale` | gauge | `device_uuid` | 1 when a consumer fan-out timed out |

`consumer` takes the values `server` (the parent process, which hosts superres)
and `worker` (the process hosting the generation worker). The spelling `worker`
is load-bearing elsewhere in the codebase and will not change.

**`st_device_unattributed_bytes` means different things per topology.** On a
DISCRETE (CUDA) device it is the per-process CUDA context plus non-torch
workspaces plus anything unregistered — roughly 300 MiB–1.5 GiB per process, and
a useful VRAM-pressure signal. On **UNIFIED** topology (`device_uuid="host"`,
Apple silicon / CPU / RKNN) it is host RAM including the OS and every unrelated
process, and equals `used_bytes` whenever no torch consumer is registered.
**Informational only on UNIFIED — never alert on it.**

**`st_device_snapshot_stale = 1` means a consumer stopped answering** its control
pipe. It is an early wedged-worker signal with no other surface, and it is worth
alerting on where `topology=DISCRETE`.

**Absent device series are not zero.** Gauges appear only after the sampler's
first pass; a scrape in the first moments of startup, or one taken while metrics
were just enabled, legitimately shows none.

## HTTP and WebSocket families

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_http_requests_total` | counter | `method`, `route`, `status` | HTTP requests |
| `st_http_request_duration_seconds` | histogram | `method`, `route` | request duration |
| `st_ws_connections_active` | gauge | — | currently connected WS clients |
| `st_ws_sessions_total` | counter | — | WS sessions accepted |
| `st_ws_messages_total` | counter | `type`, `direction` | WS messages, `in` / `out` |

**`route` is the matched route TEMPLATE, never the raw path.** `/api/models/SDXL`
and `/api/models/HunyuanDiT` are one series, `/api/models/{name}`. A request that
matched no route is `route="__unmatched__"` — do not expect literal paths there,
and do not add them: that is the unbounded set a scanner probes.

**`/metrics` counts itself, one scrape behind.** The counter increments after the
response body has been rendered, so scrape N reports N−1 scrapes and the first
scrape after a restart shows no `route="/metrics"` series at all. That is not a
lost request. Expect a baseline request rate equal to the scrape interval once
steady.

**An unhandled server exception is counted as `status="500"`.** The request
happened; dropping it would hide exactly the traffic being investigated.

**WebSocket upgrades are not HTTP requests here.** They have no status, so they
appear only in the WebSocket families above — a WS-heavy deployment will show a
low HTTP request rate and that is correct.

**`st_ws_messages_total{direction="in"}` has a bounded `type`.** Values are the
server's own handler names plus exactly two fallbacks: `unknown` (a type the
server has no handler for) and `invalid_json` (a payload that did not parse). The
client controls that field, so its raw value never reaches a label. The two
fallbacks are deliberately distinct — a broken client and a client asking for
something that does not exist are different operational events.

**`st_ws_messages_total{direction="out"}` counts per RECIPIENT**, not per
broadcast call: the status broadcaster fans one message out to every client every
5s, and the per-recipient number is the one that reflects actual socket writes.
Both directions count *delivered* messages — a send that raised is not counted.

**`st_ws_connections_active` is set from the hub's own client count**, so it
cannot drift out of step with `st_ws_sessions_total` minus disconnects.

## OS resource families

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_process_leaked_semaphores` | gauge | `process` | POSIX named semaphores visible to this process |
| `st_process_shm_segments` | gauge | `process` | shared-memory segments, excluding semaphores |
| `st_process_open_fds` | gauge | `process` | open file descriptors for this process |

`process` currently takes one value, `server` — the parent process, matching the
`consumer` vocabulary used by the DeviceMemory families. A worker-side probe
would add `worker`; it is deferred because it needs a control-pipe round trip per
sample.

**These watch an ACCEPTED risk. They are not a fault indicator.** `STABL-nstyyrhh`
established that the server leaks one POSIX named semaphore per **model load** —
linear, never reclaimed — and accepted that risk precisely because it is cheap to
watch. A count of 4 means nothing. The signal is growth per model load:

```promql
increase(st_process_leaked_semaphores[1h])
  / increase(st_governor_mode_load_seconds_count[1h])
```

A value near 1 reproduces the original finding. Alert on that ratio, or on
approaching the host's semaphore limit — **not** on the raw count.

**An ABSENT series is not zero.** Where a source cannot be read, the series is not
emitted at all — which is why these families are labelled: an unlabelled gauge
would render `0` from process start and make "no `/dev/shm` here" indistinguishable
from "no leak". `/dev/shm` is per-mount-namespace and absent on macOS, so a
developer machine emits `st_process_open_fds` and neither of the other two. Treat
absence as *not measurable here*, never as *no leak* — a host-side check reporting
0 while the container held several is the exact mistake that cost time in the
original investigation.

**Scope: this process, this namespace.** The fd count is the server's own and does
not cover the subprocess worker, which has its own fd table. `/dev/shm` is shared
within the container's mount namespace, so the semaphore and segment counts **do**
include anything the worker child created there.

## A note on `_created` series

`prometheus_client` emits a companion gauge for every counter in this document,
named by replacing the counter's `_total` suffix with `_created` and carrying the
Unix timestamp at which that child series was first observed.

These are a client-library artifact, not part of this contract: they are not
listed above, they carry no operational meaning here, and dashboards should
ignore them. Setting `PROMETHEUS_DISABLE_CREATED_SERIES=1` suppresses them
globally — verified against `prometheus_client==0.21.1`, where it took a sample
render from three such series to zero.

(This section deliberately names no example series. The contract test checks both
directions, so any `st_`-prefixed token appearing in this file must be a real
metric family — naming one of these as an illustration would fail the build.)

## Structured logs

Emitted to **stdout**, one JSON object per line, when `LOG_FORMAT=json`. The
default is `text` — the unchanged human format. `LOG_FORMAT` is read when the
logging config is *applied* (container start), not when it is built, so it is a
runtime switch on both the prod (`server/run.py`) and dev (`--log-config`) paths.

Checked-in env defaults currently do this:

| env file | `LOG_FORMAT` | effect |
|---|---|---|
| `env.prod` | `json` | prod-family container runs emit JSON |
| `env.live-test` | unset | live-test stays on the human text format |
| `env.dev` | unset | dev stays on the human text format |

**Two processes write to this stream.** The server, and — under
`WORKER_ISOLATION=subprocess`, which is the production path — the spawned worker
child, where generation actually happens. Both emit this shape; `pid`
distinguishes them.

| Field | Always? | Meaning |
|---|---|---|
| `timestamp` | yes | ISO 8601, UTC, milliseconds, `Z`-suffixed |
| `level` | yes | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `logger` | yes | Python logger name — the emitting module |
| `thread` | yes | Thread name. The dispatch loop and the event loop are different threads, and this is how you tell them apart |
| `message` | yes | Interpolated message text |
| `pid` | yes | Emitting process. Server and worker child both write here |
| `hostname` | yes | Deliberately a log field and **not** a metric label — see the label policy above |
| `mode` | while a mode is resident | Active mode name. Absent when nothing is loaded, and republished after a demand reload |
| `device_uuid` | when a device is resolved | Stable GPU identity. Same vocabulary as the metric label |
| `job_id` | during a job | Correlation id. Present on the WebSocket handler, on the dispatch-thread lines, **and** — under `WORKER_ISOLATION=subprocess` — on the spawn child's lines for the same job. **Known gap:** no HTTP handler line carries it — see below |
| `exception` | on a failing record | Formatted traceback |
| `stack` | on `stack_info=True` | Formatted stack |

Any field a caller attaches through `logging`'s `extra=` appears alongside these.
Fields that cannot be determined are **omitted**, never emitted as `null` — the
same absent-never-zero rule the OS resource gauges follow.

### Log levels

The baked config in `server/logging_config.py` ships **declared defaults**; the
environment overrides them at startup. Precedence, highest first:

| source | scope |
|---|---|
| `LOG_LEVELS` | named loggers |
| declared level in `logging_config.py` | that logger (currently only `comfy.jobs`, at `DEBUG`) |
| `LOG_LEVEL` | root and the `uvicorn*` loggers |
| `INFO` | fallback |

```bash
LOG_LEVEL=WARNING
LOG_LEVELS="comfy.jobs=WARNING,backends.governor=DEBUG"
```

`LOG_LEVELS` takes the **logger name verbatim** — no name mangling, so
`server.ws_routes` is written exactly that way. A name that no module has imported
yet is valid: the level is waiting when it does.

Bad input never fails startup, but the two variables recover differently, and the
difference is deliberate:

| bad input | result |
|---|---|
| `LOG_LEVEL` absent or unrecognised | tracking loggers resolve to **`INFO`**, with a warning |
| a `LOG_LEVELS` entry malformed or unrecognised | **that entry is skipped**; the logger keeps its declared or tracking level |

An override that cannot be read simply does not apply. `LOG_LEVEL` is different
because the loggers it governs are *defined* as "whatever `LOG_LEVEL` says" — they
have to resolve to something, and falling back to the value baked at image-build
time would just reinstate a stale build environment.

Applied at FastAPI startup and in the spawned worker child. Records emitted before
that point — import-time logging and uvicorn's own startup lines — carry the baked
level.

### Every line is JSON, including the ones logging cannot reach

Under `LOG_FORMAT=json` the stream is JSON only — a parser does not need to
tolerate stray lines.

That is not automatic. The diffusers **tqdm progress bar**
(`Loading pipeline components...: 45%|████ | ...`) writes to the stream directly
rather than through `logging`, so no logging configuration can capture it. It was
the last non-JSON writer, found by a live container run rather than by the suite.
It is now **disabled when `LOG_FORMAT=json`** and left alone otherwise — the bar
is useful to a human watching a dev container, and is only corruption when
something is parsing.

The gate reads the same `resolve_log_format()` the formatter uses, so it cannot
disagree with the format actually being emitted.

**If a non-JSON line ever appears, it is a bug — report it rather than working
around it.** The two known classes are both closed: bare `print()` calls
(guarded by `tests/test_no_print_in_server_runtime.py`) and direct stream writers
like tqdm.

### Env file syntax

**Values may be quoted or bare.** `utils/env.py` strips one layer of matching
outer quotes, because the two loaders disagree and `env.dev` is loaded by both:

| value form | `docker run --env-file` (`runner.sh`) | `docker compose env_file` |
|---|---|---|
| `LOG_LEVELS=a=1,b=2` | works | works |
| `LOG_LEVELS="a=1,b=2"` | quotes kept literally | quotes stripped |
| `export LOG_LEVEL=DEBUG` | **whole file rejected** | works |

Measured against docker 29.6.2. Only a *matching outer pair* is removed, and only
one layer — `"a` and `a"b` are left alone, because an unmatched or interior quote
is data.

**`export ` prefixes must not appear** in any file `runner.sh` passes to
`docker run --env-file`. That loader rejects the **entire file**
(`variable 'export X' contains whitespaces`) and the container never starts — no
application-side tolerance can help, because Python never sees the file.
`env.prod` is the exception: it carries `export` lines and is loaded only by
compose. `tests/test_env_file_contract.py` guards both rules, including the
condition that keeps `env.prod`'s exclusion safe.

### `job_id` correlation spans two threads — and, under isolation, two processes

The WebSocket handler runs on the event loop and the generation runs on the
Governor's dispatch thread. Contextvars do not cross that boundary on their own,
so the dispatch loop sets and **resets** the id per iteration. A missing `job_id`
on a dispatch line is a bug; a *wrong* one would be worse, which is why the reset
sits in the loop's `finally` next to `task_done()`.

Under `WORKER_ISOLATION=subprocess` — the production path — there is a **third**
boundary, and it is a process boundary. Generation runs in a spawn child, which
inherits stdout but no contextvar, so the child binds the id itself around its
job body from the `job_id` the envelope already carries
(`worker_handle_subprocess.py`). Same set/reset discipline, same reason: it is a
`with`, so a `run_job` that raises still resets rather than leaking a stale id
into whatever the child logs next.

This was measured broken before it was fixed (`STABL-zuhuxwvf`): over a 24h
window every `backends.cuda_worker` line came from a child pid and **none**
carried a `job_id`, while the server pid's lines did. Correlation worked
everywhere except where the work happens. The failure is invisible in-proc —
same process, same context — which is why the whole suite was green.

To check it is still true, split by process and require both to be correlated:

```logql
sum by (pid) (count_over_time({container="stability-toys"} | json | job_id != "" [1h]))
```

More than one `pid` in that result is the healthy reading under isolation.

Correlate a whole job with:

```logql
{container="stability-toys"} | json | job_id = "<id>"
```

### No HTTP handler line carries `job_id` — a documented limitation, not a regression

This applies to **both** HTTP generation paths — `POST /generate` and the compat
endpoints' runner — and for the same reason: `submit_generate()` returns only a
future, deliberately (`STABL-atzqpcte` — an id-keyed waiter API fixes WebSocket
and silently leaves HTTP broken). So a `/generate` failure line has no `job_id`
by design.

The generation's own **dispatch-thread** lines still carry the id, so the work is
correlatable; what is not correlatable is the HTTP request that asked for it.
Closing this needs a runtime API change, not a formatter change. For an HTTP
request, correlate by time and `mode` against the dispatch-thread lines.

## Stability

Metric names and label sets are a stable interface, and so is the log field set
above. Additions are compatible; renames, label changes and field removals are
breaking and must be announced to `../continuous` before landing. The
bidirectional tests named at the top of this document are what keep this file and
the code from drifting apart.
