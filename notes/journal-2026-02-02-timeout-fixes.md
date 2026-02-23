# Journal: The Timeout Bug Trifecta
**Date:** 2026-02-02

## What broke

Pending generations could hang forever. Three layers of missing timeout handling:

1. **`wsClient.waitFor` was broken** — timeout handler called `unsub()` which pointed to a dead `this.on('*_all', ...)` listener. The actual `handler` on the `'message'` event was never removed. Rewrote it so the timeout directly removes the correct handler.

2. **`generateViaWs` had no timeout at all** — a bare Promise listening for `job:complete` forever. Added a 120s timeout, a `settle()` guard against double-resolution, and a `statechange` listener that rejects on WS disconnect.

3. **WS disconnect didn't clean anything up** — socket drops, handlers sit there waiting for messages that will never come (new socket = new jobIds). Now both `generateViaWs` and `comfyRunnerWs` listen for disconnect and reject immediately.

## The pattern

Introduced a `settle(fn, value)` idiom — a one-shot guard that prevents double-resolve/reject when multiple termination paths race (timeout vs disconnect vs abort vs normal completion). Cleanup runs exactly once.

## Reflection

This is the kind of bug that only shows up under real usage — WS flakes, server restarts, long comfy jobs. The system was built for the happy path where every job:submit gets a job:complete. Real networks aren't that polite. Now every promise has a bounded lifetime.
