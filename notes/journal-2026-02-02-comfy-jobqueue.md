# Journal: Wiring ComfyUI through jobQueue

**Date**: 2026-02-02

The old `useComfyJobWs` hook was a single-job tracker — calling `start()` again would nuke the previous job's WS listeners. Classic race condition when you fire multiple comfy jobs.

The fix was clean: `createComfyRunnerWs` already existed with the perfect `(payload, signal) => Promise` signature. Just needed to:

1. Add `runComfy` to `useImageGeneration` — mirrors `runGenerate` pattern exactly. Creates pending message, enqueues with the comfy runner, handles error/cancel/complete via jobQueue events.
2. Strip `useComfyJobWs` out of `ComfyOptions` — now it just calls `onRunComfy(payload)` and listens to WS `job:progress` events for the progress bar.
3. Thread `runComfy` through App -> OptionsPanel -> ComfyOptions as `onRunComfy` prop.
4. Deleted the old `pendingIdRef` / `onComfyStart` / `onComfyOutputs` ceremony from App.jsx.

The `useComfyJobWs` hook is now dead code. Could delete the file but left it for now.

What I like about this: the jobQueue is now the single chokepoint for ALL generation work (LCM, comfy, cache hydration). No more parallel state machines fighting over WS listeners. Multiple comfy jobs will just queue up and execute in order, each with their own abort controller.

Progress bar in ComfyOptions listens to raw WS events filtered by `source === 'comfy'` — slightly loose matching but good enough for now. Could tighten by tracking the specific jobId if needed.

Build passes clean. Next step: browser verification with multiple rapid comfy submissions.
