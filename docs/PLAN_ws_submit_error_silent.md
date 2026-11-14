# PLAN: WS job:submit rejections are silent (server + UI)

## Problem (observed 2026-11-13)

The UI sent `job:submit` (generate) and the spinner hung. Server metrics showed
6 `job:submit` in, 2 `job:ack` out (both from a manual probe), and 4 bare
`error` frames out. The model loaded, but no UI job reached the Governor.

Two defects combine:

1. **Server**: `handle_job_submit` sends some rejections as
   `{"type": "error", "id": corrId, "error": ...}` before any `job:ack`, and does
   not log them. The paths are `init_image_ref` resolve failure
   (`server/ws_routes.py` ~L298) and unknown `jobType` (~L363). Nothing in the
   container log explains the failure.
2. **UI**: `generateViaWs` (`lcm-sr-ui/src/lib/generateRunnerWs.js`) matches
   `job:ack` by corrId and every other frame by `jobId` only. A pre-ack `error`
   has no `jobId`, so the UI drops it and waits for the 120 s timeout. That
   timeout is then retried, so the user sees "nothing happens" for up to about 6 min.

## Changes

### 1. `server/ws_routes.py::handle_job_submit`
- L298 `except KeyError as e:` add
  `logger.warning("job:submit rejected (init_image_ref): %s", e)` before send.
- L363 `else:` add
  `logger.warning("job:submit rejected: unknown jobType %r", job_type)` before send.

### 2. `lcm-sr-ui/src/lib/generateRunnerWs.js::generateViaWs` handler
Before the `if (!jobId || msg.jobId !== jobId) return;` guard, add:
```js
if (msg.type === 'error' && msg.id === corrId) {
  settle(reject, new Error(msg.error || 'Generate request rejected'));
  return;
}
```

### 3. Test: `lcm-sr-ui/src/lib/generateRunnerWs.test.jsx`
Mock `./wsClient`. Assert that a pre-ack `{type:'error', id: corrId}` rejects
`generateViaWs` with the server message.

## Order
1 → 2 → 3. Independent; no shared API shape changes (frame shapes unchanged).

## Dependencies
- `_error` frame shape unchanged; other WS clients unaffected.
- `generateViaWs` only gains an earlier reject path; `generateViaWsWithRetry`
  classifies the message as before (not retryable unless it matches
  timeout/disconnect/queue-full).

## Net lines
About +4 server lines, +4 UI lines, and about +30 test lines.

## Follow-up (not in this plan)
- After idle eviction, an untargeted submit admits against `default_mode`
  (`lcm-general`) rather than the last-used mode. Confirm this is intended.
