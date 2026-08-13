# Observability recipes — queries you can paste into Grafana

Companion to [`observability-contract.md`](observability-contract.md). The contract
defines **what is emitted**; this file is **what to ask it**.

Everything here was executed against the live Loki on `node1.lan` and the running
`stability-toys` container on `enigma`, not composed from documentation. Where a
query is unverified, it says so.

A ready-made dashboard is at
[`grafana/stability-toys-overview.json`](grafana/stability-toys-overview.json) —
import it and pick your Loki and Prometheus datasources when prompted.

---

## Read this first: three things that will silently return nothing

### 1. `level` is Loki structured metadata, and it SHADOWS our JSON field

Our log lines carry `"level": "INFO"` in **upper** case. Loki auto-detects a level
and stores it as structured metadata in **lower** case, and that wins. Measured:

| query | result |
|---|---|
| `{container="stability-toys"} \| json \| level="INFO"` | **empty** |
| `{container="stability-toys"} \| json \| level_extracted="INFO"` | **empty** |
| `{container="stability-toys", level="info"}` | **empty** — not a stream label |
| `{container="stability-toys"} \| level="info"` | **hits** |
| `{container="stability-toys"} \| json \| level="info"` | **hits** |

**Use lower case, and you do not need `| json` to filter by level.** Copying
`level="ERROR"` straight out of the contract's field table gets you nothing.

Confirmed values: `info`, `error`, `debug`. `warn` vs `warning` could not be
distinguished — no WARNING lines existed in the query window — so the recipes below
use `level=~"error|warn.*"` rather than guess.

### 2. Every other field DOES need `| json`

`logger`, `job_id`, `mode`, `pid`, `hostname`, `thread`, `device_uuid`, `message`
are inside the line. `{container="stability-toys"} | json | logger="backends.governor"`
works; without `| json` it does not.

### 3. Prometheus is live — but its own platform is not scraped

`STABL-xolucarj` is closed: `METRICS_ENABLED=true` is in `env.prod` on enigma, and
node1's `prometheus.yml` carries a `stability-toys` job. Verified 2026-08-12 —
target UP, 43 `st_` families in the TSDB.

What is **not** scraped is the observability platform itself. Loki serves 2817
`loki_` series on `node1.lan:3100/metrics` and Prometheus knows none of them; same
for promtail and Tempo. The "Platform health" section at the bottom of this file is
correct PromQL against a target that does not exist yet.

---

## Stream labels

Verified present on the `stability-toys` stream:

| label | value | use |
|---|---|---|
| `container` | `stability-toys` | the portable selector |
| `service_name` | `stability-toys` | Loki-derived, equivalent here |
| `node` | `enigma` | which host |
| `job` | `docker-enigma` | promtail scrape job |
| `container_id` | 64-hex | distinguishes container generations |

`image` is configured but absent in practice — an empty label value is dropped by
Loki, so do not build a query that depends on it.

---

## LogQL — works today

### The one that justifies the whole `job_id` design

A single job's life, across **both** threads — the WebSocket handler on the event
loop and the generation on the Governor's dispatch thread:

```logql
{container="stability-toys"} | json | job_id = "1607887a263c"
```

Add `| line_format "{{.thread}} {{.logger}} {{.message}}"` to see the thread hand-off
directly. That correlation is the entire point of `STABL-bpsfmoke`; without it these
are two unrelated log streams.

### What is going wrong, right now

```logql
{container="stability-toys"} | level=~"error|warn.*"
```

Errors only, with the module that raised them:

```logql
{container="stability-toys"} | level="error" | json
  | line_format "{{.logger}} — {{.message}}"
```

### Error rate by module — the fastest regression detector

```logql
sum by (logger) (
  count_over_time({container="stability-toys"} | json | __error__="" [5m])
)
```

Restrict to failures:

```logql
sum by (logger) (
  count_over_time({container="stability-toys"} | level="error" | json [5m])
)
```

> `| __error__=""` drops lines the JSON parser could not read. Without it a parse
> failure is silently counted as a matching line.

### Model lifecycle — loads, evictions, demand reloads

The idle watchdog evicts after `MODEL_IDLE_TIMEOUT_SECS` (default 300), and the next
generate triggers a demand reload. This is what that looks like:

```logql
{container="stability-toys"} | json | logger="backends.governor"
  | message =~ "(?i)evict|load|unload|reload|dispatch"
```

Count evictions per hour — if this is high, the idle timeout is fighting your usage
pattern:

```logql
sum(count_over_time(
  {container="stability-toys"} | json | message =~ "(?i)evicting idle model" [1h]
))
```

### The spawned worker child, separately from the server

Under `WORKER_ISOLATION=subprocess` generation happens in a child process with its
own pid. Both write to the same stdout:

```logql
{container="stability-toys"} | json | logger =~ "backends\\..*"
```

Split by process:

```logql
sum by (pid) (count_over_time({container="stability-toys"} | json [5m]))
```

A **new `pid` appearing** is a worker respawn — the facet-3 kill-and-restart after an
OOM. Worth an alert.

### Non-JSON lines — this should always be empty

The contract promises every line is JSON. This is the detector for that promise:

```logql
{container="stability-toys"} != `{"timestamp"`
```

Both known sources are closed in `main` — bare `print()` (guarded by
`tests/test_no_print_in_server_runtime.py`) and direct stream writers like the
diffusers tqdm bar (disabled under `LOG_FORMAT=json`). **A hit here is a bug to
report, not a query to work around.**

> **It fired on first use.** Run against the live deployment while writing this
> file, it returned `Loading pipeline components...:   0%|` — the tqdm bar. That is
> not a false positive: the deployed image predates the merge that disables it. The
> query correctly reports that *the running build* still has the defect. Rebuild
> the image and it clears.

Related, and a different cause with the same symptom:

```logql
sum(count_over_time({container="stability-toys"} | json | __error__="JSONParserErr" [$__auto]))
```

Measured 1 failure against 119 clean lines in the same hour. Beyond a writer that
bypasses `logging`, the other producer is **Docker's `json-file` driver splitting a
very long line into chunks** — each chunk is emitted separately and neither is valid
JSON. A large traceback or a `[REQ]` line carrying a big body can trigger it. Watch
the rate; a step change usually means someone started logging something large.

### Generation throughput, from logs alone

Until metrics land, this approximates completions:

```logql
sum(count_over_time(
  {container="stability-toys"} | json | logger="backends.cuda_worker"
    | message =~ "(?i)worker .* loaded" [1h]
))
```

### Request volume without drowning in health checks

`/health` is dropped at the promtail pipeline, so what remains is real traffic:

```logql
sum by (node) (count_over_time(
  {container="stability-toys"} | json | logger="uvicorn.access" [5m]
))
```

### Anything on this host, not just us

```logql
{node="enigma"} | level=~"error|warn.*"
```

---

## PromQL — live

Family names come from [`observability-contract.md`](observability-contract.md),
which is bidirectionally tested against the code, so these cannot drift silently.

Every query below was run against node1's Prometheus on 2026-08-12. Where one
returned no data the reason is recorded — **a counter family does not exist until
its first increment**, so an empty result for a `_total` is usually "that has never
happened", not "that is broken".

Two things about the current scrape config worth knowing before you read a graph:

- **`scrape_interval` is 60s** for this job, so `rate(...[5m])` has five points.
  Enough to be correct, coarse enough that quantiles look stepped.
- **The job sets no `node` label**, unlike `glances`. Today there is one live
  target so it does not matter; the moment a second one comes up, `sum by (...)`
  silently merges two hosts. `instance` is the only thing separating them.

### Saturation

```promql
st_governor_queue_depth
st_governor_jobs_in_flight
```

### Latency, split the way the code splits it

`STABL-atzqpcte` deliberately separated queue wait from execution — conflating them
is what hid a timeout bug. Keep them apart:

```promql
histogram_quantile(0.95,
  sum by (le, mode) (rate(st_governor_job_execution_seconds_bucket[5m])))

histogram_quantile(0.95,
  sum by (le) (rate(st_governor_job_queue_wait_seconds_bucket[5m])))
```

### Outcomes

`ok` / `cancelled` / `oom` / `error`. Note `timeout` is deliberately **not** an
outcome — it is a waiter-side budget breach counted separately, and counting it in
both places would double-count one job:

```promql
sum by (outcome) (rate(st_governor_job_terminal_total[5m]))
sum by (budget)  (rate(st_governor_wait_expired_total[5m]))
```

> Measured: `job_terminal_total` has data; `wait_expired_total` is **empty**, and
> that is the healthy reading — no job has breached either budget, so the family
> has no children yet.

### VRAM

`unattributed` is the per-process CUDA context plus non-torch workspaces — the
VRAM-pressure signal:

```promql
st_device_unattributed_bytes / st_device_total_bytes
sum by (consumer) (st_consumer_reserved_bytes)
```

> **Only meaningful on DISCRETE topology.** On UNIFIED (`device_uuid="host"`,
> Apple silicon / CPU / RKNN) `unattributed` is host RAM including unrelated
> processes. Never alert on it there.

### Worker health

```promql
sum by (reason) (rate(st_governor_worker_recovery_total[1h]))
st_device_snapshot_stale == 1
```

`snapshot_stale == 1` means a consumer stopped answering its control pipe — an early
wedged-worker signal with no other surface. Worth alerting on.

> Measured: both empty. `worker_recovery_total` has never incremented (no OOM
> recovery since the container started), and `snapshot_stale == 1` filters away a
> healthy `0`. Graph `st_device_snapshot_stale` without the comparison if you want
> to see the healthy line rather than a blank panel.

### The semaphore leak ratio

Straight from the contract. Near 1 reproduces the original `STABL-nstyyrhh` finding —
one leaked semaphore per model load:

```promql
increase(st_process_leaked_semaphores[1h])
  / increase(st_governor_mode_load_seconds_count[1h])
```

> Measured: the ratio is **empty**, because a `0 / 0` over a quiet hour is NaN and
> Prometheus drops it. The raw gauges are live and were
> `st_process_leaked_semaphores{process="server"} 2` against
> `st_governor_mode_load_seconds_count{mode="lcm-general"} 1`. Read the two
> separately until there is enough load history for the ratio to mean anything —
> and note the ratio needs the *counters to move*, so an idle box tells you nothing
> either way.

### Mode churn

`mode_active` reports **every** configured mode as 0/1, not just the loaded one — a
single gauge labelled with the current mode would leave a stale `1` behind forever:

```promql
st_governor_mode_active
sum by (mode, reason) (rate(st_governor_unload_total[1h]))
```

`unload_total{reason="switch"}` is routine churn: `_load_mode` unloads the outgoing
worker before every load. `idle_evict` is the one to watch.

---

## TraceQL — the third pillar

Tempo query API on `node1.lan:3200`. Everything here was run against it.

### Read this first: a `limit` plus a chatty endpoint hides your data

The trap that cost a debugging session, and it is the LogQL health-check problem
one signal over:

```traceql
{ resource.service.name = "stability-toys" }
```

returned **25 traces, 24 of them `GET /health`**, with the generation crowded out
entirely — and the obvious reading of that result is "tracing is not working".
It was working. Search returns the most recent N and a 2-second health check
wins every time.

**Query by what you want, never by service alone:**

```traceql
{ name = "governor.dispatch" }        # every generation, immune to noise
{ .job.id = "090f9c65c115" }          # one specific job
```

Measured on the same window: `service.name` alone → 24/25 health;
`name = "governor.dispatch"` → 3 traces, all generations; `.job.id = "<id>"` → 1.

### The log → trace join

`job_id` in a log line is `.job.id` on a span. That is the whole join — no trace
id in the log line is required:

```logql
{container="stability-toys"} | json | job_id = "090f9c65c115"
```
```traceql
{ .job.id = "090f9c65c115" }
```

Both directions work, and under `WORKER_ISOLATION=subprocess` both sides span
two processes — the log field by riding the job envelope (`STABL-zuhuxwvf`), the
span by riding the same envelope as a W3C carrier (`STABL-qnlaclof`).

### What a real generation looks like

Trace `c9f1a579d673a671ddbc7b7dfc881cc2`, enigma, 2026-08-13:

```text
span                kind          ms   job.id        mode         outcome
governor.dispatch   INTERNAL   5635.9  090f9c65c115  lcm-general  ok
governor.reload     INTERNAL   3861.3
worker.submit       PRODUCER      6.4  090f9c65c115
worker.execute      CONSUMER   1528.8  090f9c65c115
```

**Read the second row.** 3861 ms of a 5636 ms job is `governor.reload` — the
demand reload after idle eviction — and only 1529 ms is generation. Two thirds
of that request was model loading, and the trace says so without anyone adding
a timer. Metrics could only aggregate that and logs could only imply it.

`worker.execute` ran in the **spawn child**, a different process, carrying the
same `job.id` under a PRODUCER → CONSUMER pair.

### Useful queries

```traceql
{ name = "governor.dispatch" && .job.outcome != "ok" }   # jobs that did not succeed
{ name = "governor.mode_load" }                          # model loads, with durations
{ name = "governor.reload" }                             # demand reloads after eviction
{ name = "worker.execute" }                              # work in the child process
{ .http.route = "__unmatched__" }                        # what is probing us
{ name = "ws.message" && .messaging.type = "job:submit" }
```

> **`job.outcome` is derived once**, at the same choke point that feeds
> `st_governor_job_terminal_total{outcome}`. A trace and that counter cannot
> disagree about a job — by construction, not by discipline.

### Is tracing actually exporting?

Not a Tempo question. The exporter says so in the container's own log:

```bash
docker logs stability-toys-dev 2>&1 | grep "POST /v1/traces"
# DEBUG [urllib3.connectionpool] http://otel-collector:4318 "POST /v1/traces HTTP/1.1" 200
```

A 200 there is the collector accepting a batch. It is **not** proof of delivery
to Tempo — the same first-hop-versus-delivery distinction that applies to logs.
For that, read a trace back by id.

---

## Platform health — is the pipeline itself working?

Query these before concluding "the app is quiet". Every one of them was a real
failure during setup.

> **The PromQL here does not run yet.** Prometheus scrapes the *application* but not
> the *platform*: Loki, promtail and Tempo all export self-metrics and none of them
> is a scrape target. Verified 2026-08-12 — `loki_discarded_samples_total` is absent
> from Prometheus while `node1.lan:3100/metrics` serves 2817 `loki_` series. Adding
> those jobs is `../continuous` work. Until then, read the counter straight off the
> component:
>
> ```bash
> curl -s http://node1.lan:3100/metrics | grep '^loki_discarded_samples_total'
> ```

**Is anything being discarded?** Check the *rate*, not the total — these counters are
cumulative and carry historical scars:

```promql
sum by (reason) (rate(loki_discarded_samples_total[5m]))
```

`missing_labels` means a promtail job shipped a stream with zero labels, which
rejects the **entire batch**, not just that stream. Cause is almost always a
`docker_sd_configs` job with no `job` label — unlike `static_configs`, docker SD does
not synthesise one.

**Is the app's own scrape target up?** This one *does* run, and is the fastest check
that the metrics half is alive at all:

```promql
up{job="stability-toys"}
```

A permanent `0` for a target is not an outage — it is a target for a host that does
not run the container. `mindgate.lan:4200` is currently in that state.

**Is this host shipping at all?**

```logql
sum by (node) (count_over_time({node=~".+"} [5m]))
```

A host missing from that result is not quiet — it is not shipping.
