# Stability Toys Job Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real generation-job cancellation, safer retry and OOM recovery behavior, explicit reload/free-VRAM controls, and stop auto-regenerating selected images when only the negative prompt changes.

**Architecture:** Keep the current queue-based design but make backend generation jobs first-class objects with stable IDs and cancel state in `WorkerPool`. Reuse that job state from WebSocket and HTTP control surfaces, and keep frontend changes narrow: negative-prompt edits become staged-only, retry logic becomes error-classified, and operational model controls live in the Configuration tab.

**Tech Stack:** Python, FastAPI, concurrent.futures, pytest, React, Vitest, existing `WorkerPool`, WS job protocol, and UI config components

**FP Parent Issue:** `STABL-qcjviryo`

**FP Child Issues:**
- `STABL-hllaktvj` WorkerPool job IDs and cancellation state
- `STABL-nusinvli` OOM, reload, and free-VRAM recovery controls
- `STABL-lsplqhqs` WebSocket generate/cancel backend job wiring
- `STABL-qjqynfan` Negative-prompt staging and WS retry classification
- `STABL-iawyoaqc` Configuration-tab operational controls and runtime mode status
- `STABL-wzadhont` Acceptance verification and operator docs

---

### Task 1: Add WorkerPool generation job lifecycle and cancellation primitives

**Files:**
- Modify: `backends/worker_pool.py`
- Modify: `tests/test_worker_pool.py`

- [ ] **Step 1: Write the failing worker-pool tests**

Add focused tests for:

```python
def test_cancel_queued_generation_job_marks_future_cancelled(worker_pool):
    req = Mock()
    job = GenerationJob(req=req, job_id="job-1")
    fut = worker_pool.submit_job(job)
    assert worker_pool.cancel_job("job-1") is True
    assert fut.cancelled()

def test_cancel_running_generation_job_discards_late_result(worker_pool, mock_worker_factory):
    release = threading.Event()
    worker = mock_worker_factory.return_value
    worker.run_job.side_effect = lambda job: (release.wait(), ("png", 123))[1]
    req = Mock()
    fut = worker_pool.submit_job(GenerationJob(req=req, job_id="job-2"))
    assert worker_pool.cancel_job("job-2") is True
    release.set()
    with pytest.raises(concurrent.futures.CancelledError):
        fut.result(timeout=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_worker_pool.py -k 'cancel_queued_generation_job or cancel_running_generation_job' -q`
Expected: FAIL because `GenerationJob` does not expose a stable `job_id` and `WorkerPool` has no `cancel_job(...)` support yet.

- [ ] **Step 3: Implement the minimal job-state tracking**

Add stable backend job IDs and generation-job bookkeeping in `backends/worker_pool.py`:

```python
@dataclass
class GenerationJob(Job):
    req: Any
    init_image: Optional[bytes] = None
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class JobRecord:
    job_id: str
    state: str
    job: GenerationJob
    cancel_requested: bool = False
```

Add `WorkerPool.cancel_job(job_id)` and make the worker loop skip queued canceled jobs and discard late results from running canceled jobs.

- [ ] **Step 4: Run worker-pool tests to verify they pass**

Run: `python3 -m pytest tests/test_worker_pool.py -k 'cancel_queued_generation_job or cancel_running_generation_job' -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backends/worker_pool.py tests/test_worker_pool.py
git commit -m "feat: add worker pool job cancellation state"
```

### Task 2: Add OOM cleanup, reload, and free-VRAM worker-pool behavior

**Files:**
- Modify: `backends/worker_pool.py`
- Modify: `tests/test_worker_pool.py`
- Modify: `tests/test_model_routes.py`

- [ ] **Step 1: Write the failing recovery tests**

Add tests for OOM cleanup and explicit reload/free-VRAM behavior:

```python
def test_oom_cancels_pending_generation_jobs_and_unloads_worker(worker_pool, mock_worker_factory):
    fake_oom = torch.cuda.OutOfMemoryError
    worker = mock_worker_factory.return_value
    worker.run_job.side_effect = fake_oom("CUDA out of memory")
    first_future = worker_pool.submit_job(GenerationJob(req=Mock(), job_id="job-1"))
    queued_future = worker_pool.submit_job(GenerationJob(req=Mock(), job_id="job-2"))
    with pytest.raises(fake_oom):
        first_future.result(timeout=1.0)
    assert queued_future.cancelled()
    assert worker_pool.is_model_loaded() is False

async def test_reload_and_free_vram_routes_call_pool_methods():
    pool.reload_current_mode.return_value = {"status": "reloaded", "mode": "sdxl-general"}
    pool.free_vram.return_value = {
        "status": "ok",
        "is_loaded": False,
        "current_mode": "sdxl-general",
        "vram": {"allocated_bytes": 0, "reserved_bytes": 0, "total_bytes": 8 * 1024**3},
    }
    with patch("server.model_routes.get_worker_pool", return_value=pool):
        assert (await model_routes.reload_current_model())["status"] == "reloaded"
        assert (await model_routes.free_vram())["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_worker_pool.py -k 'oom_cancels_pending_generation_jobs' tests/test_model_routes.py -q`
Expected: FAIL because `WorkerPool` does not expose shared recovery helpers and `server.model_routes` does not yet provide reload/free-VRAM endpoints.

- [ ] **Step 3: Implement shared recovery helpers**

Add shared worker-pool operations:

```python
def reload_current_mode(self) -> dict:
    if self._current_mode is None:
        raise RuntimeError("No active mode to reload")
    self.cancel_pending_generation_jobs(reason="reload_current_mode")
    self.switch_mode(self._current_mode, force=True).result(timeout=30.0)
    return {"status": "reloaded", "mode": self._current_mode}

def free_vram(self, reason: str) -> dict:
    cancelled = self.cancel_pending_generation_jobs(reason=reason)
    self._mark_running_generation_jobs_cancel_requested(reason=reason)
    self._unload_current_worker()
    gc.collect()
    torch.cuda.empty_cache()
    return self._build_runtime_status(cancelled_jobs=cancelled)
```

Hook OOM handling to call the same cleanup path before returning control to callers.

- [ ] **Step 4: Run focused recovery tests**

Run: `python3 -m pytest tests/test_worker_pool.py -k 'oom_cancels_pending_generation_jobs or reload_current_mode' tests/test_model_routes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backends/worker_pool.py tests/test_worker_pool.py tests/test_model_routes.py
git commit -m "feat: add worker pool recovery controls"
```

### Task 3: Wire WebSocket and HTTP job-control surfaces

**Files:**
- Modify: `server/ws_routes.py`
- Modify: `server/model_routes.py`
- Modify: `tests/test_ws_routes.py`
- Modify: `tests/test_model_routes.py`

- [ ] **Step 1: Write the failing route tests**

Add tests that assert:

```python
def test_job_cancel_ack_reports_backend_cancel_result():
    pool.cancel_job.return_value = {"status": "canceled", "job_id": "abc123"}
    app.state.use_mode_system = True
    app.state.worker_pool = pool
    assert msg["detail"] == "canceled"

def test_generate_ack_uses_backend_generation_job_id():
    generation_job = pool.submit_job.call_args.args[0]
    assert ack["jobId"] == generation_job.job_id

async def test_cancel_job_route_calls_worker_pool():
    result = await model_routes.cancel_job("abc123")
    assert result["job_id"] == "abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ws_routes.py tests/test_model_routes.py -q`
Expected: FAIL because the WS layer still acks with its own local ID flow and `server.model_routes` does not yet expose job cancel/reload/free-VRAM operations.

- [ ] **Step 3: Implement backend job-control routes**

Update `server/ws_routes.py` so generate jobs ack with the actual `GenerationJob.job_id` and `job:cancel` delegates to `state.worker_pool.cancel_job(...)` when the mode system is active.

Add HTTP routes in `server/model_routes.py`:

```python
@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    pool = get_worker_pool()
    return pool.cancel_job(job_id)

@router.post("/models/reload")
async def reload_current_model():
    return get_worker_pool().reload_current_mode()

@router.post("/models/free-vram")
async def free_vram():
    return get_worker_pool().free_vram(reason="api")
```

- [ ] **Step 4: Run route tests to verify they pass**

Run: `python3 -m pytest tests/test_ws_routes.py tests/test_model_routes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/ws_routes.py server/model_routes.py tests/test_ws_routes.py tests/test_model_routes.py
git commit -m "feat: expose job cancel and model recovery routes"
```

### Task 4: Stop negative-prompt auto-regeneration and classify retryable WS failures

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useGenerationParams.js`
- Modify: `lcm-sr-ui/src/hooks/useGenerationParams.test.jsx`
- Modify: `lcm-sr-ui/src/lib/generateRunnerWs.js`
- Create: `lcm-sr-ui/src/lib/generateRunnerWs.test.jsx`

- [ ] **Step 1: Write the failing frontend tests**

Update the hook test and add retry-classifier coverage:

```jsx
it('stages selected-image negative prompt edits without triggering regeneration', () => {
  result.current.setNegativePrompt('washed out');
  expect(patchSelectedParams).toHaveBeenCalledWith({ negativePrompt: 'washed out' });
  vi.runAllTimers();
  expect(runGenerate).not.toHaveBeenCalled();
});

it('does not retry deterministic validation failures', async () => {
  mockGenerateViaWs
    .mockRejectedValueOnce(new Error("scheduler_id 'dpmpp_2m' is not allowed for the active mode"));
  await expect(generateViaWsWithRetry(payload)).rejects.toThrow(/not allowed/);
  expect(mockGenerateViaWs).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd lcm-sr-ui && npm test -- useGenerationParams.test.jsx generateRunnerWs.test.jsx`
Expected: FAIL because negative prompt edits still schedule regeneration and the WS retry wrapper retries all non-abort failures.

- [ ] **Step 3: Implement staged negative-prompt edits and retry classification**

Change `setNegativePromptEffective()` to patch selected params without calling `scheduleRegenSelected(...)`.

Add a retry classifier in `generateRunnerWs.js`:

```js
function isRetryableGenerateError(err) {
  const message = String(err?.message || err);
  if (err?.name === 'AbortError') return false;
  if (/not allowed for the active mode|missing init image|validation/i.test(message)) return false;
  return /timed out|disconnected|queue full/i.test(message);
}
```

Use the classifier inside `generateViaWsWithRetry(...)` so only transient failures spend retry budget.

- [ ] **Step 4: Run frontend tests to verify they pass**

Run: `cd lcm-sr-ui && npm test -- useGenerationParams.test.jsx generateRunnerWs.test.jsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useGenerationParams.js lcm-sr-ui/src/hooks/useGenerationParams.test.jsx lcm-sr-ui/src/lib/generateRunnerWs.js lcm-sr-ui/src/lib/generateRunnerWs.test.jsx
git commit -m "fix: stage negative prompt edits and tighten ws retries"
```

### Task 5: Add Configuration-tab operational controls and correct runtime mode status

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useModeConfig.js`
- Modify: `lcm-sr-ui/src/components/config/ModeEditor.jsx`
- Modify: `lcm-sr-ui/src/utils/api.js`
- Create: `lcm-sr-ui/src/hooks/useModeConfig.test.jsx`
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`

- [ ] **Step 1: Write the failing UI-state tests**

Add tests for runtime mode state and configuration-tab controls:

```jsx
it('keeps runtime active mode separate from config default mode', async () => {
  fetchGet
    .mockResolvedValueOnce({
      default_mode: 'sd15-fast',
      modes: {
        'sd15-fast': { model: 'sd15.safetensors' },
        'sdxl-general': { model: 'sdxl.safetensors' },
      },
    })
    .mockResolvedValueOnce({ current_mode: 'sdxl-general', is_loaded: false, queue_size: 0, vram: {} });
  expect(result.current.activeModeName).toBe('sdxl-general');
  expect(result.current.defaultModeName).toBe('sd15-fast');
});

it('shows reload and free-vram controls in mode configuration', async () => {
  expect(screen.getByRole('button', { name: /reload active model/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /free vram/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd lcm-sr-ui && npm test -- useModeConfig.test.jsx OptionsPanel.test.jsx`
Expected: FAIL because `useModeConfig()` only mirrors `/api/modes` defaults and the Configuration tab does not expose operational controls yet.

- [ ] **Step 3: Implement runtime status polling and controls**

Extend `useModeConfig()` to load both `/api/modes` and `/api/models/status`, and add operational methods:

```js
const refreshStatus = useCallback(async () => {
  const status = await api.fetchGet('/api/models/status');
  setRuntimeStatus(status);
}, [api]);

const reloadActiveModel = useCallback(async () => {
  await api.fetchPost('/api/models/reload', {});
  await refreshStatus();
}, [api, refreshStatus]);

const freeVram = useCallback(async () => {
  await api.fetchPost('/api/models/free-vram', {});
  await refreshStatus();
}, [api, refreshStatus]);
```

Render `Reload Active Model` and `Free VRAM` inside `ModeEditor.jsx`, along with runtime status text showing active mode and whether it is currently loaded.

- [ ] **Step 4: Run UI tests to verify they pass**

Run: `cd lcm-sr-ui && npm test -- useModeConfig.test.jsx OptionsPanel.test.jsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useModeConfig.js lcm-sr-ui/src/hooks/useModeConfig.test.jsx lcm-sr-ui/src/components/config/ModeEditor.jsx lcm-sr-ui/src/utils/api.js lcm-sr-ui/src/components/options/OptionsPanel.test.jsx
git commit -m "feat: add configuration recovery controls"
```

### Task 6: Run cross-surface verification and document operator workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-01-stability-toys-job-control-design.md`

- [ ] **Step 1: Document the operator-facing recovery flow**

Add a short README section covering:

```md
- Per-image cancel stops pending generation delivery
- Configuration → Reload Active Model forces same-mode recovery
- Configuration → Free VRAM cancels queued work, unloads the worker, and clears reclaimable CUDA cache
```

- [ ] **Step 2: Run full targeted verification**

Run: `python3 -m pytest tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py -q`
Expected: PASS

Run: `cd lcm-sr-ui && npm test -- useGenerationParams.test.jsx generateRunnerWs.test.jsx useModeConfig.test.jsx OptionsPanel.test.jsx`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-04-01-stability-toys-job-control-design.md tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py lcm-sr-ui/src/hooks/useGenerationParams.test.jsx lcm-sr-ui/src/lib/generateRunnerWs.test.jsx lcm-sr-ui/src/hooks/useModeConfig.test.jsx lcm-sr-ui/src/components/options/OptionsPanel.test.jsx
git commit -m "docs: describe job control and recovery flow"
```
