# Journal — 2026-01-31 — WebSocket Backend Implementation

Woke up to a clear plan. The ws-migration-map was well-defined, the architecture made sense — additive WS layer on top of existing HTTP endpoints, no breaking changes. My kind of work.

## What I built

Five-file implementation following the build order:
1. **ws_hub.py** — Singleton connection manager. Clean async lock, tolerant of dead clients. ~55 lines.
2. **invokers/jobs.py** — Added `set_on_update` callback hook. When comfy jobs mutate state, the hook fires and we push `job:progress` through the hub. Elegant — no polling loops needed.
3. **upload_routes.py** — Ephemeral file upload store for WS clients. 5-min TTL, background cleanup. Simple and disposable.
4. **ws_routes.py** — The main dispatcher. Single `/v1/ws` endpoint, dispatch table pattern. Handlers for generate/comfy/sr jobs, dream control, ping/pong. The generate handler reuses the exact same `GenerateRequest` + `PipelineService.submit()` / `WorkerPool.submit_job()` code paths as the HTTP endpoint.
5. **lcm_sr_server.py** — Mounted routers, added status broadcaster (5s interval), wired up the job hook in lifespan.

## Design decisions I liked

- The `_on_job_update` callback handles both "called from async context" and "called from worker thread" cases. Thread-safe scheduling via `call_soon_threadsafe`.
- Status broadcaster only fires when clients are connected (saves CPU when idle).
- Upload store is intentionally ephemeral — no persistence needed, it's just a bridge for WS→binary→job.

## What the next me should know

- Frontend migration is the next phase. The WS is ready, HTTP untouched.
- `dream:guide` is a new capability — lets you steer a running dream session's prompt/temperature without stopping it. The worker needs `base_prompt` and `temperature` attributes exposed for this to work fully.
- The `_build_status` function tries to grab VRAM info opportunistically. If torch isn't available it gracefully degrades.

## Vibes

This was satisfying. Clean additive architecture, no regressions. The garden grows another layer. 🌱
