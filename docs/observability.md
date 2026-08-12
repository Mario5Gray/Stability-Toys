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

### 3. Prometheus queries return nothing yet

`METRICS_ENABLED` is unset **and** nothing scrapes `:4200` — `STABL-xolucarj`. Both
halves are required; neither alone is enough. The PromQL section below is correct
and inert until that lands.

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

## PromQL — correct, and inert until `STABL-xolucarj`

Family names come from [`observability-contract.md`](observability-contract.md),
which is bidirectionally tested against the code, so these cannot drift silently.

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

### The semaphore leak ratio

Straight from the contract. Near 1 reproduces the original `STABL-nstyyrhh` finding —
one leaked semaphore per model load:

```promql
increase(st_process_leaked_semaphores[1h])
  / increase(st_governor_mode_load_seconds_count[1h])
```

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

## Platform health — is the pipeline itself working?

Query these before concluding "the app is quiet". Every one of them was a real
failure during setup.

**Is anything being discarded?** Check the *rate*, not the total — these counters are
cumulative and carry historical scars:

```promql
sum by (reason) (rate(loki_discarded_samples_total[5m]))
```

`missing_labels` means a promtail job shipped a stream with zero labels, which
rejects the **entire batch**, not just that stream. Cause is almost always a
`docker_sd_configs` job with no `job` label — unlike `static_configs`, docker SD does
not synthesise one.

**Is this host shipping at all?**

```logql
sum by (node) (count_over_time({node=~".+"} [5m]))
```

A host missing from that result is not quiet — it is not shipping.
