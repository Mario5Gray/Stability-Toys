# Stability Toys Job Control And Recovery Design

## Summary

This design adds three related capabilities to Stability Toys:

1. Explicit per-job cancellation from the UI and backend
2. Safer retry and recovery behavior after worker failures, especially OOM
3. Removal of implicit regeneration when editing the negative prompt on an existing image

The core issue is that generation work currently spans three loosely-coupled systems:

- the frontend queue and WebSocket retry wrapper
- the backend WebSocket request task
- the backend worker pool and live model state

Today, canceling a job only cancels the WebSocket task, not the queued or executing backend generation. Deterministic request failures can be retried even though they will never succeed. OOM recovery unloads the worker, but pending jobs may continue to fail against stale state. The UI also treats edits to `negativePrompt` on a selected image as an implicit regenerate action, which is not the desired editing model.

This design makes job lifecycle and model lifecycle explicit so those failure modes stop compounding.

## Goals

- Allow users to cancel a queued or running generation from the UI
- Expose backend cancellation over WebSocket and HTTP
- Retry only transient generation failures
- Automatically perform cleanup after OOM so stale VRAM does not block subsequent work
- Expose a manual maintenance endpoint to cancel work, unload the worker, and free reclaimable VRAM
- Prevent `negativePrompt` edits from auto-triggering regeneration on selected images
- Add explicit reload and VRAM cleanup controls in the Configuration tab, not the main generation view
- Fix same-mode reload behavior so a user can recover the current model without switching to another mode first

## Non-Goals

- Preemptively interrupt CUDA kernels already executing inside the model pipeline
- Implement direct arbitrary model loading outside the existing mode system
- Redesign the main chat/generation layout
- Guarantee that absolutely all VRAM is returned without restarting the process

## Current Problems

### Cancellation is incomplete

`server/ws_routes.py` currently handles `job:cancel` by canceling the `asyncio.Task` created for the socket request. That does not cancel:

- a queued `GenerationJob` inside `WorkerPool`
- a job already executing in `worker.run_job(...)`
- stale result delivery after the user thinks the job is gone

### Retry policy is too blunt

`lcm-sr-ui/src/lib/generateRunnerWs.js` retries all non-abort failures. That means deterministic request errors such as disallowed schedulers get retried even though the same request will fail again.

### OOM leaves the system in a confusing intermediate state

`WorkerPool` already unloads the worker on OOM, but follow-on jobs can still remain queued and continue failing because:

- the model is no longer loaded
- the queued requests may no longer be valid for the runtime state
- the UI has no operational recovery control besides switching modes

### Mode state in the UI is conflated with configuration defaults

`useModeConfig()` uses `default_mode` as if it were the active runtime mode. That causes incorrect UI state and makes explicit unload/reload behavior awkward, especially when recovering the currently selected mode after failure.

### Negative prompt edits regenerate selected images

`useGenerationParams()` routes `negativePrompt` changes through the same debounced regeneration path as prompt, steps, CFG, and size when an existing image is selected. The desired behavior is to stage that edit and require explicit `Shift+Enter` regeneration.

## Proposed Approach

Use a single job-control model across the frontend queue, WebSocket protocol, and backend worker pool:

- generation jobs get stable backend job identifiers
- the backend tracks queued, running, canceled, failed, and completed jobs
- cancellation can target a queued or running backend job
- running generation cannot always be interrupted inside the model code, so cancellation for active jobs is best-effort and suppresses result delivery
- deterministic request errors fail fast without retry
- OOM triggers the same recovery path as manual VRAM cleanup

This approach keeps the current architecture but closes the lifecycle gaps between its layers.

## Backend Design

### WorkerPool job registry

Extend `backends/worker_pool.py` to track generation jobs explicitly.

Additions:

- a stable `job_id` on `GenerationJob`
- a lightweight job record map keyed by `job_id`
- job states such as `queued`, `running`, `cancel_requested`, `canceled`, `failed`, `completed`
- helper methods:
  - `submit_generation_job(...)`
  - `cancel_job(job_id)`
  - `cancel_pending_generation_jobs(reason)`
  - `free_vram(reason)`
  - `reload_current_mode()`

Behavior:

- If a queued job is canceled before execution, remove it from the registry, mark its future canceled, and skip execution.
- If a running job is canceled, mark it `cancel_requested`. The worker may continue running, but the result is discarded and the future resolves as canceled when control returns to the pool.
- The pool remains the source of truth for whether a backend generation is still live.

### OOM cleanup path

When the worker loop encounters OOM:

- mark the triggering job failed
- cancel queued generation jobs because they are likely targeting a poisoned or unloaded runtime
- unload the current worker
- run `gc.collect()` and `torch.cuda.empty_cache()`
- retain the current mode name so the mode can be reloaded explicitly later

This cleanup logic should be shared with a manual maintenance API so automatic and manual recovery use the same code path.

### Mode switch semantics

Before applying a mode switch:

- cancel queued generation jobs
- enqueue the switch as the next control operation
- allow new generation jobs submitted after the switch request to run only after the switch completes, so they execute against the new active mode

Rationale:

- generation requests queued for the old model should not silently execute against the new one
- this is especially important after OOM or operator-initiated recovery

Running jobs remain best-effort:

- if cancellation is requested and the worker returns late, discard the result

### Manual reload and free-VRAM operations

Add two explicit worker-pool maintenance operations:

`reload_current_mode()`
- force-reload the currently selected mode even if the mode name has not changed

`free_vram(reason)`
- cancel queued generation work
- mark running generation work canceled
- unload the worker
- run `gc.collect()` and `torch.cuda.empty_cache()`
- return post-cleanup runtime and VRAM status

This does not guarantee full process-level VRAM release because CUDA context and library allocations may persist, but it reclaims the PyTorch-managed memory that is currently causing most repeated failure loops.

## API Design

### WebSocket

Retain the existing `job:submit` and `job:cancel` messages, but connect them to real backend generation job identifiers.

Expected flow:

1. Frontend submits `job:submit`
2. WS route creates a backend generation job with a pool `job_id`
3. `job:ack` returns that same backend job id
4. `job:cancel` targets the backend job id
5. completion/error/cancel responses map to the backend job lifecycle, not only the socket task lifecycle

### HTTP

Add maintenance-oriented endpoints under `/api`:

- `POST /api/jobs/{job_id}/cancel`
- `POST /api/models/reload`
- `POST /api/models/free-vram`

`/api/models/unload` may remain available as a narrower primitive, but the user-facing operational control should be `free-vram`, because it performs the full cleanup sequence needed after OOM.

Suggested `free-vram` response shape:

```json
{
  "status": "ok",
  "canceled_jobs": 3,
  "running_jobs_marked_canceled": 1,
  "current_mode": "sdxl-general",
  "is_loaded": false,
  "vram": {
    "allocated_bytes": 0,
    "reserved_bytes": 0,
    "total_bytes": 25769803776
  }
}
```

Exact field names can follow the existing `/api/models/status` and `/api/vram` conventions.

## Frontend Design

### Negative prompt editing

Change `lcm-sr-ui/src/hooks/useGenerationParams.js` so `setNegativePromptEffective()` no longer calls `scheduleRegenSelected(...)` when a selected image is active.

New behavior:

- update the selected message params locally
- do not enqueue generation
- apply the edit only when the user explicitly reruns with `Shift+Enter`

This matches the editing model already expected for the draft prompt flow.

### Retry classifier

Change `lcm-sr-ui/src/lib/generateRunnerWs.js` to retry only transient failures.

Retry candidates:

- WebSocket disconnect during generation
- timeout waiting for completion
- maybe queue-pressure failures such as `429` or `"Queue full"` if desired

Do not retry:

- invalid scheduler for active mode
- malformed request errors
- missing init image references
- validation/configuration errors from the backend

This should be implemented with an explicit classifier instead of string matching scattered through the UI. String matching may still be required initially because the WS protocol currently sends generic `job:error` text, but the classifier should live in one place and be easy to replace with structured error codes later.

### Per-message cancel

The existing cancel affordance should cancel both:

- the frontend queue entry or abort controller
- the backend generation job if one has already been acknowledged by the server

This requires the WS generate runner to expose or retain the acknowledged backend `jobId` so a canceled local request can send `job:cancel`.

### Configuration tab controls

Place explicit operational controls in the Configuration tab, not the main generation view:

- `Reload Active Model`
- `Free VRAM`

These controls should:

- call the new backend maintenance endpoints
- refresh runtime model status after completion
- surface any failure details inline

The main page should keep only per-job cancel.

### Runtime status vs configuration state

Update `useModeConfig()` so the UI distinguishes:

- configuration default mode from `/api/modes`
- runtime active mode and loaded state from `/api/models/status`

This fixes the current bug where `default_mode` is treated as the active runtime mode and enables:

- same-mode reload
- correct display after OOM cleanup
- correct display after explicit unload or free-VRAM operations

## Error Handling

### Best-effort cancellation

Active CUDA work may not stop immediately. The contract should be:

- user cancel means the request is no longer expected to deliver a result
- if the worker finishes later, the pool drops the output instead of mutating user-visible state

This is good enough for correctness even without true kernel interruption.

### Recovery after OOM

After automatic or manual cleanup:

- the active mode name may still be known
- the worker may be unloaded
- the UI must display that distinction clearly

The user can then explicitly:

- reload the active model, or
- switch to another mode

### Recovery endpoint caveat

`free-vram` should state that it frees reclaimable VRAM, not guarantee total zero VRAM usage. Some memory may remain held by the CUDA context or native libraries until process restart.

## Testing

### Backend tests

Add or update tests covering:

- cancel queued generation job before execution
- cancel running generation job and discard late result
- OOM triggers cleanup and queued job cancellation
- mode switch cancels queued generation jobs
- `reload_current_mode()` works without switching through another mode
- `free-vram` endpoint unloads worker and reports status

### Frontend tests

Add or update tests covering:

- editing `negativePrompt` on a selected image does not trigger regeneration
- explicit rerun still applies the edited negative prompt
- deterministic WS errors do not retry
- transient WS errors still retry within budget
- Configuration tab controls call reload/free-VRAM actions and refresh runtime state

## Rollout Plan

1. Add backend job registry and cancellation support in `WorkerPool`
2. Wire WS job IDs and cancel handling to backend jobs
3. Add manual reload and free-VRAM HTTP endpoints
4. Update UI retry classifier and negative prompt behavior
5. Add Configuration tab operational controls
6. Fix runtime mode status handling in the UI
7. Add regression tests for backend and frontend behavior

## Risks

- Best-effort cancel may still allow GPU time to be spent on a request whose result will be discarded
- Canceling queued jobs on mode switch is a behavioral change and may surprise users who expected the queue to survive model transitions
- VRAM cleanup may not fully reclaim memory if third-party libraries retain allocations outside PyTorch

## Recommendation

Proceed with the full job-control approach rather than patching retries and UI behavior in isolation.

The three user-reported problems are coupled:

- retries need accurate error classification
- cancellation needs backend job awareness
- OOM recovery needs explicit model lifecycle controls

Solving them together produces a cleaner and more predictable system than incremental partial fixes.
