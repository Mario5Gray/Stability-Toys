# Asynchronous Events — User-Facing Inventory

A catalogue of every asynchronous process that affects what the user sees or
waits on. Organised by layer: backend threads and tasks, WebSocket protocol,
frontend job queue, and UI-level timers.

---

## 1. Backend — Threads and Background Tasks

### 1.1 Worker Thread (`WorkerPool`)

**What it is:** A single `threading.Thread` that processes all GPU work.

**Jobs it runs:**
| Job type | Triggered by |
|---|---|
| `GenerationJob` | `job:submit` over WS (generate path) |
| `ModeSwitchJob` | Config screen save, `/api/models/switch`, idle eviction |
| `CustomJob` | Internal: demand-reload, idle eviction |

**User-visible effects:**
- New images appear in chat
- Model loading delay on the first request after eviction (demand-reload)
- Model switching delay after config save

**Queue:** `queue.Queue(maxsize=4)`. If full, the WS handler returns
`job:error "Queue full"` immediately.

---

### 1.2 Idle Watchdog Thread (`WorkerPool._watchdog_thread`)

**What it is:** A daemon thread that wakes every `MODEL_IDLE_CHECK_INTERVAL_SECS`
(default 30 s) and checks `time.monotonic()` against `_last_activity`.

**Trigger condition:** `MODEL_IDLE_TIMEOUT_SECS` (default 300 s) of no
generation activity.

**User-visible effects:**
- VRAM freed silently in the background — user does not see this directly.
- Next generation after eviction stalls while the model is reloaded (demand
  reload via `CustomJob`). The spinner remains visible throughout.

---

### 1.3 Jobs Reaper Thread (`startup_hooks._jobs_reaper_loop`)

**What it is:** A daemon thread that wakes every 5 s and scans the in-memory
ComfyUI job store.

**Trigger conditions:**
| Condition | Action |
|---|---|
| `created_at` > 15 min ago | Mark `error("Job timed out (hard timeout)")` |
| `status == running` and no heartbeat for > 60 s | Mark `error("Job stalled")` |

**User-visible effects:**
- A stuck Comfy job eventually resolves to an error state instead of spinning
  forever. The client receives `job:error` via the WS push path.

---

### 1.4 Status Broadcaster (`lcm_sr_server._status_broadcaster`)

**What it is:** An `asyncio.Task` that wakes every 5 s and broadcasts
`system:status` to all connected WS clients.

**Payload includes:** current mode, VRAM free/total, storage health,
connected client count.

**User-visible effects:**
- The UI's mode indicator and VRAM display update passively without polling.

---

### 1.5 Upload Cleanup Loop (`upload_routes.cleanup_uploads_loop`)

**What it is:** An `asyncio.Task` that periodically deletes stale temp upload
files from `/v1/upload`.

**User-visible effects:** None directly. Prevents disk exhaustion that would
break init-image uploads.

---

### 1.6 Config File Watcher (`file_watcher.ConfigFileWatcher`)

**What it is:** A `watchdog` `Observer` thread watching the directory
containing `modes.yaml` via inotify (Linux) or FSEvents (macOS).

**Trigger:** Any write to `modes.yaml` (debounced to 1 s).

**User-visible effects:**
- Mode configuration reloads in the background.
- Does **not** automatically reload the running worker; the config change
  only takes effect when a mode switch is explicitly triggered (e.g. via
  the config screen save which queues `ModeSwitchJob(force=True)`).

---

### 1.7 SIGHUP Handler

**What it is:** A Python signal handler registered at startup that calls
`reload_mode_config()`.

**Trigger:** `kill -HUP <pid>` from the operator.

**User-visible effects:** Same as file watcher — config reloads but running
worker is not disrupted unless a mode switch follows.

---

## 2. WebSocket Protocol — Message Flow

The single endpoint is `/v1/ws`. All messages are JSON with a `type` field.

### 2.1 Connection Lifecycle

| Event | Direction | When |
|---|---|---|
| WS `connect` | client→server | On page load / reconnect |
| `system:status` | server→client | Immediately on connect, then every 5 s |
| `ping` | client→server | Every 25 s (keepalive) |
| `pong` | server→client | In response to `ping` |

---

### 2.2 Generation Job (jobType: `generate`)

```
client                              server
  │── job:submit (jobType=generate) ──►│
  │◄─ job:ack (jobId) ────────────────│  immediate
  │                                    │  ... GPU runs in thread ...
  │◄─ job:complete (outputs, meta) ───│  on success
  │◄─ job:error (error) ──────────────│  on failure / cancel / timeout
```

**Timeout:** Client-side 120 s timer. On expiry, sends `job:cancel` then
rejects the Promise. Server-side task is cancelled via `asyncio.CancelledError`.

**Retry:** `generateViaWsWithRetry` retries up to 3 times with 1 s / 2 s
backoff on any error except `AbortError`.

---

### 2.3 ComfyUI Job (jobType: `comfy`)

```
client                              server
  │── job:submit (jobType=comfy) ────►│
  │◄─ job:ack (jobId) ────────────────│  immediate
  │◄─ job:progress (fraction, node) ──│  on each ComfyUI node execution
  │◄─ job:complete / job:error ────────│  on finish
```

Progress is pushed from `invokers/jobs.py` via `set_on_update` → `_on_job_update`
→ `asyncio.ensure_future(hub.broadcast(...))` — crosses from the ComfyUI WS
thread to the asyncio event loop via `call_soon_threadsafe`.

---

### 2.4 Super-Resolution Job (jobType: `sr`)

```
client                              server
  │── job:submit (jobType=sr) ───────►│
  │◄─ job:ack (jobId) ────────────────│  immediate
  │                                    │  ... SR runs in thread ...
  │◄─ job:complete / job:error ────────│
```

No progress events. The SR model runs synchronously in a thread pool via
`run_in_executor`.

---

### 2.5 Cancellation

```
client                              server
  │── job:cancel (jobId) ────────────►│
  │◄─ job:cancel:ack ─────────────────│
```

Server calls `task.cancel()` on the tracked `asyncio.Task` for that `jobId`.
If the task has already finished, returns `"no running task found"`.

---

### 2.6 Error Envelope

All error paths send:
```json
{ "type": "job:error", "jobId": "...", "error": "human-readable message" }
```

or (for protocol errors not tied to a job):
```json
{ "type": "error", "id": "corrId", "error": "..." }
```

---

## 3. Frontend — Job Queue (`jobQueue`)

A singleton `JobQueue` with `concurrency = 1` (one job runs at a time).
Priority lanes:

| Priority | Value | Used for |
|---|---|---|
| URGENT | 0 | (reserved, unused) |
| NORMAL | 1 | `generate`, `comfy` |
| BATCH | 2 | (reserved, unused) |
| BACKGROUND | 3 | `cache-hydrate` |

**Events emitted** (all via `CustomEvent` on the `JobQueue` instance):

| Event | When |
|---|---|
| `enqueue` | Job added |
| `start` | Job begins executing |
| `complete` | Job runner resolves |
| `error` | Job runner rejects (non-abort) |
| `cancel` | Job cancelled from pending, or runner aborted |
| `drain` | Queue depth reaches 0 |

**User-visible effects:**
- While a generation job is running, a second submission goes into the
  pending queue and shows as waiting.
- `cancelAll()` drains pending and aborts the running job; all in-flight
  WS generate requests are also cancelled.

---

## 4. Frontend — Image URL Lifecycle

A generated image passes through three URL forms before it is fully cached:

```
1. job:complete arrives
   → message.imageUrl = serverImageUrl  (/storage/<key>)
   → new window.Image() preloads before Promise resolves
   → spinner disappears, image is visible (served from server)

2. background (BACKGROUND priority in jobQueue)
   → fetch(serverImageUrl) → Blob → IndexedDB via cache.set()
   → cache emits "hydrated" event

3. "hydrated" handler
   → URL.createObjectURL(blob)
   → message.imageUrl swapped to blobUrl
   → subsequent renders come from browser memory (no network round-trip)
```

If the user reloads the page, step 1 is skipped; cached entries go straight
to step 3 or trigger step 2 hydration if only metadata was stored.

---

## 5. Frontend — Timers and Debounces

| Source | Timer | Effect |
|---|---|---|
| `wsClient` ping | 25 s `setInterval` | Sends `ping` to keep connection alive |
| `wsClient` reconnect | Exponential backoff: 1 s → 15 s max | Auto-reconnects after disconnect |
| `generateRunnerWs` timeout | 120 s `setTimeout` | Rejects and cancels server job if no `job:complete` arrives |
| `generateViaWsWithRetry` backoff | 1 s / 2 s fixed | Re-attempts failed generation (up to 3 total) |
| Dream mode interval | `dreamInterval` ms (default 5000) | Fires `runDreamCycle` repeatedly, painting into same message |
| Denoise strength slider debounce | 120 ms | Delays `onDenoiseStrengthChange` call until slider settles |
| `scheduleRegenSelected` | (debounced in `useGenerationParams`) | Batches param changes before re-generating the selected image |

---

## 6. Frontend — Dream Mode

Dream mode is a `setInterval`-driven loop that fires every `dreamInterval`
milliseconds and enqueues a `BACKGROUND`-priority generation job with a
stochastically mutated prompt and parameters.

**Concurrency note:** Dream jobs share the single-concurrency `jobQueue` with
normal generation. A user-initiated generation at `PRIORITY.NORMAL` preempts
pending dream jobs in the queue (it gets inserted ahead of them).

**Stopping:** `stopDreaming()` calls `clearInterval` and cancels nothing in the
queue — in-flight dream jobs complete normally.

---

## 7. Summary Table

| Layer | Process | Trigger | User sees |
|---|---|---|---|
| Backend thread | Worker thread | job:submit / ModeSwitchJob | Image appears / loading delay |
| Backend thread | Idle watchdog | 30 s poll | Nothing (silent eviction); delay on next generate |
| Backend thread | Jobs reaper | 5 s poll | Stuck Comfy job eventually errors |
| Backend asyncio | Status broadcaster | 5 s timer | Mode / VRAM indicator updates |
| Backend asyncio | Upload cleanup | Periodic timer | (invisible) |
| Backend thread | File watcher | inotify event on modes.yaml | Config reloads (worker unchanged) |
| Backend signal | SIGHUP handler | kill -HUP | Config reloads (worker unchanged) |
| WS | job:ack | Immediate on submit | Spinner appears |
| WS | job:progress | ComfyUI node events | Progress bar / node name |
| WS | job:complete | Generation done | Image rendered |
| WS | job:error | Failure / cancel / timeout | Error state in chat |
| WS | system:status | Every 5 s + on connect | Status bar updates |
| Frontend | Job queue flush | enqueue / complete | Queue depth badge |
| Frontend | Cache hydrate | After first display | Image moves to blob URL (transparent) |
| Frontend | Dream interval | setInterval | New image in dream bubble |
| Frontend | WS reconnect | Disconnect event | "Reconnecting…" banner |
| Frontend | Generate timeout | 120 s | Error + job:cancel sent to server |
