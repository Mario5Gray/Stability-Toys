# WS job:progress delta string Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Read before writing any code:**
>
> 1. [`.superpowers/sdd/project-context.md`](../../../.superpowers/sdd/project-context.md) — protocol facts, patterns
> 2. This file — task requirements and constraints

**Goal:** Add a human-readable `delta` string to `job:progress` WS frames for image generation jobs so that `st gen --stream` shows per-node progress lines.

**Architecture:** Single-file backend change in `server/ws_routes.py`. Extract a pure `_progress_delta(progress: dict) -> str | None` helper (unit-testable without async) and call it in `_on_job_update`. The existing `progress` dict field is preserved for forward compatibility. The existing `_fire_update` → `_on_job_update` path already fires on every `jobs_update_path` call — no wiring change needed.

**Tech Stack:** Python, FastAPI, pytest, `unittest.mock`

**FP issue:** STABL-dakbipff — tag every commit with it.

## Global Constraints

- Do not remove or rename the `progress` dict field from the `job:progress` frame — other consumers may read it
- `delta` is omitted (key absent) when `progress.fraction == 0` — no noise for setup/preamble broadcasts
- `delta` format: `"node {nodes_seen}/{nodes_total} ({pct}%)"` when `nodes_total > 0`, else `"{pct}%"`
- Run tests from repo root: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_ws_routes.py -v`
- Do not modify `invokers/jobs.py` — the callback wiring is already correct

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `server/ws_routes.py` | Modify | Extract `_progress_delta` helper; add `delta` to `job:progress` frame in `_on_job_update` |
| `tests/test_ws_routes.py` | Modify | Add `TestProgressDelta` class testing the helper; add test for delta presence in frame |

---

## Task 1: Add `_progress_delta` helper and wire into `_on_job_update`

**Files:**
- Modify: `server/ws_routes.py`
- Modify: `tests/test_ws_routes.py`

**Interfaces:**
- Produces: `_progress_delta(progress: dict) -> str | None` — pure function, no side effects

- [ ] **Step 1: Write failing tests** — add `TestProgressDelta` class to `tests/test_ws_routes.py`

Add the following import at the top of `tests/test_ws_routes.py` (after the existing `from server.ws_routes import ws_router` import):

```python
from server.ws_routes import _progress_delta
```

Add the following class at the bottom of the file (before any existing test classes or at end of file):

```python
class TestProgressDelta:
    def test_zero_fraction_returns_none(self):
        assert _progress_delta({"fraction": 0.0, "nodes_seen": 0, "nodes_total": 4}) is None

    def test_empty_dict_returns_none(self):
        assert _progress_delta({}) is None

    def test_fraction_with_nodes(self):
        delta = _progress_delta({"fraction": 0.5, "nodes_seen": 3, "nodes_total": 6})
        assert delta == "node 3/6 (50%)"

    def test_fraction_without_nodes_total(self):
        # nodes_total absent or zero — fall back to percent only
        delta = _progress_delta({"fraction": 0.75})
        assert delta == "75%"

    def test_fraction_100_percent(self):
        delta = _progress_delta({"fraction": 1.0, "nodes_seen": 6, "nodes_total": 6})
        assert delta == "node 6/6 (100%)"

    def test_delta_in_job_progress_frame(self):
        """_on_job_update builds a msg dict; verify delta is included when fraction > 0."""
        import asyncio
        from unittest.mock import patch, AsyncMock

        broadcast_calls = []

        async def _run():
            with patch("server.ws_routes.hub") as mock_hub:
                mock_hub.broadcast = AsyncMock(side_effect=lambda msg: broadcast_calls.append(msg))
                ws_routes._on_job_update(
                    "job-delta-test",
                    {
                        "status": "running",
                        "progress": {
                            "fraction": 0.5,
                            "nodes_seen": 3,
                            "nodes_total": 6,
                            "current_node": "KSampler",
                            "node_progression": ["KSampler"],
                        },
                    },
                )
                # _on_job_update uses asyncio.get_running_loop() in an async context
                await asyncio.sleep(0)  # let create_task run

        asyncio.run(_run())
        assert len(broadcast_calls) == 1
        msg = broadcast_calls[0]
        assert msg["type"] == "job:progress"
        assert msg["delta"] == "node 3/6 (50%)"
        assert "progress" in msg  # existing field preserved

    def test_no_delta_in_frame_when_fraction_zero(self):
        """delta key must be absent (not empty string) when fraction is 0."""
        import asyncio
        from unittest.mock import patch, AsyncMock

        broadcast_calls = []

        async def _run():
            with patch("server.ws_routes.hub") as mock_hub:
                mock_hub.broadcast = AsyncMock(side_effect=lambda msg: broadcast_calls.append(msg))
                ws_routes._on_job_update(
                    "job-no-delta-test",
                    {
                        "status": "running",
                        "progress": {"fraction": 0.0, "nodes_seen": 0, "nodes_total": 4},
                    },
                )
                await asyncio.sleep(0)

        asyncio.run(_run())
        assert len(broadcast_calls) == 1
        assert "delta" not in broadcast_calls[0]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_ws_routes.py::TestProgressDelta -v
```

Expected: `ImportError` — `cannot import name '_progress_delta' from 'server.ws_routes'`

- [ ] **Step 3: Add `_progress_delta` helper to `server/ws_routes.py`**

Add after the `_on_job_update` function (around line 65, before `register_job_hook`):

```python
def _progress_delta(progress: dict) -> "str | None":
    """Format a human-readable progress string from the job progress snapshot dict.

    Returns None when fraction is zero so callers can omit the 'delta' key
    entirely — the client guards against missing/empty delta.
    """
    fraction = progress.get("fraction") or 0.0
    if not fraction:
        return None
    pct = int(fraction * 100)
    nodes_total = progress.get("nodes_total") or 0
    if nodes_total > 0:
        nodes_seen = progress.get("nodes_seen") or 0
        return f"node {nodes_seen}/{nodes_total} ({pct}%)"
    return f"{pct}%"
```

- [ ] **Step 4: Update `_on_job_update` to include `delta` in the frame**

Replace the `msg = {...}` block in `_on_job_update` (currently lines 50–55):

```python
    progress = snapshot.get("progress") or {}
    delta = _progress_delta(progress)
    msg = {
        "type": "job:progress",
        "jobId": job_id,
        "status": snapshot.get("status"),
        "progress": snapshot.get("progress"),
    }
    if delta is not None:
        msg["delta"] = delta
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_ws_routes.py::TestProgressDelta -v
```

Expected: all 7 tests `PASS`

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add server/ws_routes.py tests/test_ws_routes.py
git commit -m "feat(ws): add delta string to job:progress frames for image generation (STABL-dakbipff) — next: client empty-delta guard (STABL-ykcbssxk)"
```

---

## Self-Review

**Spec coverage:**
- `_on_job_update` emits `delta` string when `fraction > 0` ✅ Task 1 Step 4
- `delta` absent (not empty string) when no progress ✅ `if delta is not None: msg["delta"] = delta`
- `progress` dict preserved in frame ✅ field kept in `msg`
- No change to `invokers/jobs.py` ✅ wiring already correct
- `delta` format includes nodes and percent ✅ `_progress_delta` helper
- Tests cover zero-fraction, with-nodes, without-nodes, and frame-level presence/absence ✅

**Placeholder scan:** None found.

**Type consistency:** `_progress_delta` returns `str | None`. `msg["delta"]` only set when not None. Import of `_progress_delta` in test matches function name in implementation.
