# WebSocket Channel Map

> **Scope: Backend-only.** All existing HTTP endpoints remain untouched.
> The WebSocket is an **additional** channel layered alongside HTTP — not a replacement.
> Frontend migration to prefer WS happens in a later phase.

## Legend

| Symbol | Meaning |
|--------|---------|
| `→` | Client-to-server |
| `←` | Server-to-client |
| `↔` | Request/response pair (client sends, server replies) |
| `←⋯` | Server pushes unprompted (stream/event) |
| **V1** | Include in first production WebSocket release |
| **V2** | Defer to second iteration |
| **HTTP-only** | No WS equivalent — stays HTTP forever |

---

## 1. Image Generation

| # | HTTP Endpoint (unchanged) | WS Equivalent | Direction | Style | Phase | Notes |
|---|--------------------------|---------------|-----------|-------|-------|-------|
| 1 | `POST /generate` | `job:submit {jobType:"generate", ...}` → `job:ack` → `job:progress` → `job:complete` | ↔ then ←⋯ | Fire → stream | **V1** | WS response: `{jobId, outputs:[{url, key}], meta:{seed, backend, sr}}`. HTTP endpoint unchanged. |
| 2 | `POST /superres` | `job:submit {jobType:"sr", fileRef, magnitude, ...}` → same flow | ↔ then ←⋯ | Fire → stream | **V1** | File uploaded via `POST /v1/upload` → `fileRef`, then referenced in WS message. HTTP endpoint unchanged. |
| 3 | `POST /v1/superres` | — | — | — | **HTTP-only** | Alias of #2. Stays as-is. |

---

## 2. ComfyUI

| # | HTTP Endpoint (unchanged) | WS Equivalent | Direction | Style | Phase | Notes |
|---|--------------------------|---------------|-----------|-------|-------|-------|
| 4 | `POST /v1/comfy/jobs` | `job:submit {jobType:"comfy", workflowId, params, inputImage:"fileRef:x"}` → `job:ack {jobId}` | ↔ | Req/Res | **V1** | HTTP endpoint unchanged. WS alternative avoids multipart. |
| 5 | `GET /v1/comfy/jobs/{id}` | `job:progress` / `job:complete` / `job:error` | ←⋯ | Server push | **V1** | HTTP polling still works. WS clients get pushed instead. |
| 6 | `DELETE /v1/comfy/jobs/{id}` | `job:cancel {jobId}` | → | Fire | **V1** | HTTP DELETE still works. WS is an alternative path. |

---

## 3. Dream System

| # | HTTP Endpoint (unchanged) | WS Equivalent | Direction | Style | Phase | Notes |
|---|--------------------------|---------------|-----------|-------|-------|-------|
| 7 | `POST /dreams/start` | `dream:start {prompt, duration, temperature, ...}` → `dream:started {sessionId}` | ↔ | Req/Res | **V1** | |
| 8 | `POST /dreams/stop` | `dream:stop` → `dream:stopped {stats}` | ↔ | Req/Res | **V1** | |
| 9 | `GET /dreams/status` | `dream:status` | ←⋯ | Push (periodic, e.g. every 2s while dreaming) | **V1** | HTTP polling still works. WS clients get pushed. |
| 10 | `GET /dreams/top` | `dream:top {limit, min_score}` → `dream:top:result` | ↔ | Req/Res | **V1** | |
| 11 | `GET /dreams/recent` | `dream:recent {limit}` → `dream:recent:result` | ↔ | Req/Res | **V2** | |
| 12 | `GET /dreams/stats` | Included in `dream:status` push payload | ←⋯ | Push | **V1** | Folded into #9. HTTP endpoint unchanged. |
| 13 | — (new) | `dream:guide {params}` | → | Fire | **V1** | **New.** Live param steering. No HTTP predecessor. |
| 14 | — (new) | `dream:candidate {seed, score, thumbnailUrl}` | ←⋯ | Push | **V2** | **New.** Server pushes promising candidates as discovered. |

---

## 4. Model Management — HTTP-only

> All model management stays HTTP. WS receives read-only state observation via `system:status` pushes.

| # | HTTP Endpoint (unchanged) | WS | Notes |
|---|--------------------------|-----|-------|
| 15 | `GET /api/models/status` | WS clients observe state via `system:status` push. | Commands stay HTTP; WS only *observes*. |
| 16 | `GET /api/modes` | No WS equivalent. | One-time fetch. |
| 17 | `POST /api/modes/switch` | WS clients get `system:status` push when switch completes. | Infrequent admin action. |
| 18 | `POST /api/modes/reload` | No WS equivalent. | Admin/dev action. |
| 19 | `GET /api/vram` | Included in `system:status` push. | HTTP for on-demand, WS for passive. |
| 20 | `POST /api/models/unload` | — | 501 Not Implemented. |
| 21 | `POST /api/models/load` | — | 501 Not Implemented. |

---

## 5. Storage

| # | HTTP Endpoint (unchanged) | WS Equivalent | Direction | Style | Phase | Notes |
|---|--------------------------|---------------|-----------|-------|-------|-------|
| 22 | `PUT /storage/{key}` | `storage:put {key, payload}` → `storage:put:ok` | ↔ | Req/Res | **V2** | Small payloads only. HTTP endpoint unchanged. |
| 23 | `GET /storage/{key}` | — | — | — | **HTTP-only** | `<img src>` needs a URL. |
| 24 | `GET /storage/health` | Included in `system:status` push. | ←⋯ | Push | **V1** | HTTP endpoint unchanged. |

---

## 6. Health & Compatibility — HTTP-only

| # | HTTP Endpoint (unchanged) | WS | Notes |
|---|--------------------------|-----|-------|
| 25 | `GET /health` | WS clients use `ping`/`pong` keepalive. | HTTP stays for load balancers. |
| 26 | `GET /sdapi/v1/sd-models` | No WS equivalent. | External compat. |
| 27 | `GET /sdapi/v1/options` | No WS equivalent. | External compat. |
| 28 | `GET /sdapi/v1/samplers` | No WS equivalent. | External compat. |
| 29 | `POST /sdapi/v1/txt2img` | No WS equivalent. | External compat. |
| 30 | `POST /v1/images/generations` | No WS equivalent. | External compat. |

---

## 7. New WS-Only Messages (no HTTP predecessor)

| # | WS Message `type` | Direction | Style | Phase | Purpose |
|---|-------------------|-----------|-------|-------|---------|
| 31 | `job:priority {jobId, priority}` | → | Fire | **V1** | Reprioritize queued job |
| 32 | `system:status {mode, vram, storage}` | ←⋯ | Push (on change) | **V1** | Read-only observation of model/VRAM/storage state. HTTP model mgmt endpoints trigger these pushes as a side-effect. |
| 33 | `system:backends {available:[...]}` | ←⋯ | Push (on change) | **V2** | Backend node discovery (multi-GPU) |
| 34 | `queue:state {pending, running, jobs:[]}` | ←⋯ | Push (on change) | **V1** | Server-side queue state mirror |

---

## V1 Summary

```
WS messages (V1):     13 total
  Job lifecycle:        6  (submit, ack, progress, complete, error, cancel)
  Dream control:        5  (start, stop, status push, top, guide)
  System observation:   2  (system:status push, queue:state push)

HTTP endpoints:        ALL UNCHANGED (30 endpoints stay as-is)

New capabilities:
  - Server push replaces polling for comfy progress + dream status
  - dream:guide enables live param steering (new)
  - job:priority enables queue reordering (new)
  - system:status gives passive model/VRAM/storage observation (new)
```

---

## Connection Lifecycle

```
1. Client opens  ws://host:port/v1/ws
2. Server sends  ← system:status {mode, vram, storage, queueState}
3. Client ready — can submit jobs via WS or HTTP (both work)
4. Keepalive     → ping  ← pong  (every 30s)
5. Reconnect     Client auto-reconnects with exp. backoff
                 Server replays current state on reconnect
```

---

## Dual-Path Principle

Every WS job message has an HTTP equivalent that continues to work:

```
                 ┌─── WS: job:submit ───┐
  Frontend ──────┤                      ├──── Backend Job Runner
                 └─── HTTP: POST /gen ──┘

                 ┌─── WS: job:progress ◄──┐
  Frontend ◄─────┤                        ├── Backend Job Runner
                 └─── HTTP: GET /status ──┘
                        (poll)
```

WS and HTTP are parallel paths to the same backend. The frontend can migrate
one endpoint at a time without breaking anything.
