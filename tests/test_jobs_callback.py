"""
Tests for invokers/jobs.py — on_update callback hook.
"""

import pytest
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from invokers.jobs import (
    jobs_put, jobs_get, jobs_update, jobs_update_path,
    set_on_update, JOBS, JOBS_LOCK,
)


@pytest.fixture(autouse=True)
def _clean_jobs():
    """Clean JOBS dict and callback before/after each test."""
    with JOBS_LOCK:
        JOBS.clear()
    set_on_update(None)
    yield
    with JOBS_LOCK:
        JOBS.clear()
    set_on_update(None)


class TestOnUpdateCallback:
    def test_callback_fires_on_jobs_update(self):
        cb = MagicMock()
        set_on_update(cb)

        jobs_put("j1", {"id": "j1", "status": "queued"})
        jobs_update("j1", {"status": "running"})

        assert cb.call_count == 1
        job_id, snapshot = cb.call_args[0]
        assert job_id == "j1"
        assert snapshot["status"] == "running"

    def test_callback_fires_on_jobs_update_path(self):
        cb = MagicMock()
        set_on_update(cb)

        jobs_put("j2", {"id": "j2", "progress": {"fraction": 0.0}})
        jobs_update_path("j2", "progress.fraction", 0.5)

        assert cb.call_count == 1
        _, snapshot = cb.call_args[0]
        assert snapshot["progress"]["fraction"] == 0.5

    def test_callback_receives_snapshot_not_reference(self):
        """Callback should get a deepcopy, not a reference to the live dict."""
        received = []
        def cb(job_id, snap):
            received.append(snap)
        set_on_update(cb)

        jobs_put("j3", {"id": "j3", "status": "queued"})
        jobs_update("j3", {"status": "running"})
        jobs_update("j3", {"status": "done"})

        assert len(received) == 2
        assert received[0]["status"] == "running"
        assert received[1]["status"] == "done"
        # Verify they're independent copies
        received[0]["status"] = "MUTATED"
        live = jobs_get("j3")
        assert live["status"] == "done"  # not mutated

    def test_no_callback_is_noop(self):
        """No callback registered — should not raise."""
        jobs_put("j4", {"id": "j4", "status": "queued"})
        jobs_update("j4", {"status": "running"})  # should not raise
        jobs_update_path("j4", "progress.fraction", 0.5)  # should not raise

    def test_callback_error_does_not_propagate(self):
        """A broken callback should not break job mutations."""
        def bad_cb(job_id, snap):
            raise RuntimeError("callback exploded")
        set_on_update(bad_cb)

        jobs_put("j5", {"id": "j5", "status": "queued"})
        # Should not raise even though callback throws
        jobs_update("j5", {"status": "running"})
        assert jobs_get("j5")["status"] == "running"

    def test_callback_not_fired_for_missing_job(self):
        cb = MagicMock()
        set_on_update(cb)

        jobs_update("nonexistent", {"status": "running"})
        cb.assert_not_called()

        jobs_update_path("nonexistent", "progress.fraction", 0.5)
        cb.assert_not_called()

    def test_set_on_update_to_none_disables(self):
        cb = MagicMock()
        set_on_update(cb)

        jobs_put("j6", {"id": "j6", "status": "queued"})
        jobs_update("j6", {"status": "running"})
        assert cb.call_count == 1

        set_on_update(None)
        jobs_update("j6", {"status": "done"})
        assert cb.call_count == 1  # unchanged
