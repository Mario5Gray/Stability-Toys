# Observability notes (scratch pad)

## Deeper performance telemetry ideas
- Add PerformanceObserver metrics for LCP, FCP, CLS, and long tasks.
- Track React render durations for key components (App, ChatContainer, OptionsPanel) using React Profiler.
- Emit UI navigation timing (tab switches) and action-to-result latency (e.g., time from generate click -> first image render).
- Capture resource timing for heavy assets (model lists, workflows) to spot slow endpoints.

## UI telemetry enhancements
- Standardize event names with `ui.<component>.<action>` and `queue.<action>.job`.
- Include session sampling + user role (if available) to segment data.
- Track error boundaries with component stack for recoverable errors.

## Backend telemetry alignment
- Add trace correlation IDs from UI -> backend -> worker for end-to-end latency.
- Emit job lifecycle metrics server-side (enqueue/dequeue/start/finish) to compare against UI timings.
- Tag telemetry with mode/workflow names for popularity + performance heatmaps.

## OTEL/Loki/Prometheus pipeline notes
- Consider a local OTEL Collector sidecar for UI telemetry in dev to reduce noise.
- Loki labels: keep labels low-cardinality (service, job type, status, mode); put high-cardinality fields in log body.
- Prometheus: prefer histograms for queue latency + run time; counters for completions/errors.
