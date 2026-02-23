# Grafana notes

## Data sources
- Prometheus: `Prometheus`
- Loki: `Loki`

## Suggested dashboard panels (v1)
- Job throughput: rate of job completions over time.
- Queue latency: histogram heatmap (p50/p95/p99).
- Job runtime: histogram heatmap (p50/p95/p99).
- Errors: rate of job failures by source.
- UI render time: `ui.render.app` first render metric.
- Dream usage: counts of `dream.toggle` + `dream.*` events.
- Mode popularity: counts of `mode.select` by mode name.

## Query placeholders
- Prometheus: replace with actual metric names once OTEL metrics are finalized.
- Loki: filter by `service` or `job.source` once log pipeline is enabled.

## Next steps
- Add Tempo datasource for traces when ready.
- Add dashboard variables for `mode`, `job.source`, and `status`.
