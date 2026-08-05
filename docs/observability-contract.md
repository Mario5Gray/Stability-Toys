# Observability contract — metrics exported by Stability-Toys

**Issue:** STABL-asawxgvp (umbrella STABL-oxbwjwvu)
**Spec:** `docs/superpowers/specs/2026-08-03-server-observability-seams-design.md`

This repo owns **emission only**. Scrape config, collectors, dashboards, alert
rules and retention live in `../continuous/docs` — see `AGENTS.md`. This document
is the interface between the two: it is what `../continuous` reads instead of
guessing or reverse-engineering a scrape.

Every family listed here is checked against the running facade by
`tests/test_metrics.py::test_every_family_is_documented_in_the_contract`, in both
directions — a metric that ships without an entry fails, and an entry for a
metric that no longer exists fails too.

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

## Stability

Metric names and label sets are a stable interface. Additions are compatible;
renames and label changes are breaking and must be announced to `../continuous`
before landing. The bidirectional test named at the top of this document is what
keeps this file and the code from drifting apart.
