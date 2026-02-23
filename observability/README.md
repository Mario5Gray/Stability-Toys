OTel Collector (UI Telemetry)

Purpose
- Accept OTLP/HTTP from the UI and export to Prometheus + Loki (and Tempo if available).

Environment
- OTEL_PROXY_ENDPOINT (backend): e.g. http://otel-collector:4318/v1/traces
- VITE_OTEL_PROXY_ENDPOINT (UI): e.g. /api/telemetry (used by fallback HTTP)
- WebSocket telemetry: UI sends `telemetry:otlp` over `/v1/ws`, backend forwards to OTEL_PROXY_ENDPOINT
 - VITE_OTEL_ENABLED=true|false
 - VITE_OTEL_SAMPLE_RATE=1.0 (0.0–1.0, sampled per session)
 - VITE_OTEL_NAME_PREFIX=ui
 - VITE_OTEL_BATCH_MS=3000
 - VITE_OTEL_BATCH_MAX=50
 - VITE_OTEL_QUEUE_MAX=500

Collector config
- File: observability/otel-collector-config.yml
- OTLP HTTP receiver: :4318
- Prometheus scrape endpoint: :9464

Example run (docker)
```
docker compose -f observability/docker-compose.otel.yml up -d
```

Prometheus scrape target
```
  - job_name: "otel-collector"
    static_configs:
      - targets: ["otel-collector:9464"]
```
