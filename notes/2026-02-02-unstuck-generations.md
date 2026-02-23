# Journal: Unstuck Generations — 2026-02-02

Implemented the timeout/retry/cancel system today. The garden needed some resilience — plants die sometimes, and you need to be able to replant.

## What was done

Five files touched to create a full retry loop:

1. **generateRunnerWs.js** — Added `generateViaWsWithRetry` wrapping the existing WS generator with 3 attempts and exponential-ish backoff (1s, 2s). On timeout, we now send `job:cancel` to the server so it stops wasting cycles on abandoned work.

2. **useImageGeneration.js** — Swapped to the retry wrapper. On error, we now stash the original `retryParams` on the message object so the UI can re-trigger the exact same generation.

3. **ws_routes.py** — The cancel stub is now real. `_running_tasks` dict tracks every asyncio.Task by job_id with done-callbacks for cleanup. Cancel looks up the task and calls `.cancel()`. `_run_generate` catches `CancelledError` gracefully.

4. **MessageBubble.jsx** — Added a retry button (RotateCcw icon) that appears in the error overlay on images and inline in text-only error messages. Only shows when `retryParams` exist.

5. **ChatContainer.jsx** + **App.jsx** — Plumbed `onRetry` through to call `runGenerate` with the saved params.

## Vibes

This felt clean. The pattern of storing retry params on the message object is elegant — it's like leaving seeds in the pot so you can regrow if the first attempt fails. The server-side cancel is important too; no point in the garden hose running if nobody's there to collect the water.

The backoff times (1s, 2s) are conservative but reasonable for a local/LAN setup. If the backend is truly down, 3 attempts won't waste much time before showing the user the error with a manual retry button.

## Future thoughts

Could add exponential backoff with jitter for distributed scenarios. Could also track retry count on the message for UI feedback ("Retrying 2/3..."). But that's gold-plating for now.
