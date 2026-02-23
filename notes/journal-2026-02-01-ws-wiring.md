# Journal: WS Wiring Complete - Feb 1, 2026

Finally plugged the WebSocket pipe into the ComfyUI UI. The pieces were all sitting there — `useComfyJobWs`, `createComfyRunnerWs`, `useWs`, `wsClient` — like perfectly machined gears in a box, waiting for someone to just... assemble the clock.

The old code had a funny double-fire pattern: `jobQueue.enqueue(...)` AND `comfy.start(...)` both running on the same button click. Two execution paths racing each other. Like ordering the same pizza from two different apps and hoping one arrives. Cleaned that right up — now it's just `comfy.start()` through the WS hook. One path, push-based progress, no polling.

What strikes me is how much cleaner push is than pull. The polling version was making GET requests every 750ms to check "are we there yet?" like a kid on a road trip. Now the server just tells us when something happens. Progress events flow in naturally.

The `useWs()` call in App.jsx is elegant in its simplicity — two lines to ensure the singleton connects before any component tries to use it. The hook pattern with `useSyncExternalStore` underneath is solid React.

Next time I wake up: verify the `/v1/upload` endpoint works with the WS flow, and check that `job:ack` / `job:progress` / `job:complete` events actually fire from the backend. The frontend wiring is done but the proof is in the pudding.

Feeling: satisfied. Like snapping the last piece into a puzzle.
