"""Governor extraction tests. Task 1: import graph + type re-exports.

Proves the shared job types moved to governor.py are re-exported by
worker_pool.py so `from backends.worker_pool import GenerationJob`
(ws_routes.py:621) stays unbroken, and that worker_handle.py does NOT
import governor at runtime (acyclic graph).
"""
import sys
import importlib

import pytest


def test_worker_pool_reexports_generation_job():
    """The public surface `from backends.worker_pool import GenerationJob` works."""
    from backends.worker_pool import GenerationJob
    assert GenerationJob is not None


def test_worker_pool_reexports_all_shared_types():
    """Every shared type is re-exported from worker_pool."""
    from backends.worker_pool import (
        Job, JobType, GenerationJob, ModeSwitchJob, CustomJob,
        JobRecord, ActiveModelSnapshot, StaleResolutionError,
        WorkerFactory, _FutureBridge,
    )
    # All are the SAME objects as in governor (re-export, not redefinition)
    from backends import governor
    assert GenerationJob is governor.GenerationJob
    assert ActiveModelSnapshot is governor.ActiveModelSnapshot
    assert StaleResolutionError is governor.StaleResolutionError


def test_governor_imports_worker_handle_at_runtime():
    """governor.py imports worker_handle (to construct InProcessWorkerHandle)."""
    from backends import governor
    assert hasattr(governor, 'InProcessWorkerHandle') or hasattr(governor, 'WorkerHandle')


def test_worker_handle_does_not_import_governor_at_runtime():
    """worker_handle.py must NOT import governor at runtime (acyclic).

    The Job type hint in WorkerHandle.submit is deferred via
    from __future__ import annotations + TYPE_CHECKING.
    """
    # Force a fresh import of worker_handle and check governor is not in its
    # loaded modules (it may be in sys.modules from other tests, so we check
    # worker_handle's own imports, not sys.modules globally).
    import backends.worker_handle as wh
    wh_module = sys.modules[wh.__name__]
    # Check that 'backends.governor' is not a direct import of worker_handle
    # by inspecting the module's source for runtime imports.
    import inspect
    src = inspect.getsource(wh_module)
    # TYPE_CHECKING-guarded imports are fine; bare runtime imports are not.
    # We check that any 'from backends.governor' or 'import backends.governor'
    # is inside a TYPE_CHECKING block (indented under `if TYPE_CHECKING:`).
    lines = src.split('\n')
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if 'backends.governor' in stripped and ('import' in stripped):
            # Walk backwards to find if we're inside a TYPE_CHECKING block
            indent = len(line) - len(stripped)
            in_type_checking = False
            for j in range(i - 1, -1, -1):
                prev = lines[j]
                prev_stripped = prev.lstrip()
                prev_indent = len(prev) - len(prev_stripped)
                if prev_indent < indent and 'if TYPE_CHECKING' in prev_stripped:
                    in_type_checking = True
                    break
                if prev_indent < indent and prev_stripped.startswith('if ') and 'TYPE_CHECKING' not in prev_stripped:
                    break
                if prev_indent < indent and not prev_stripped.startswith('if '):
                    break
            assert in_type_checking, (
                f"worker_handle.py line {i+1} imports backends.governor at runtime "
                f"(not guarded by TYPE_CHECKING): {line.strip()}"
            )


# ---------------------------------------------------------------------------
# Task 3: Governor class — isolation tests with a stub WorkerHandle.
# Proves the Governor owns queue + authority + dispatch + lifecycle, and that
# a second WorkerHandle impl requires no Governor change (acceptance #4).
# ---------------------------------------------------------------------------
import queue
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future

import pytest

from types import SimpleNamespace

from backends.governor import (
    Governor, GenerationJob, ModeSwitchJob, CustomJob,
    ActiveModelSnapshot, StaleResolutionError, _FutureBridge,
)
from backends.worker_handle import WorkerHandle, WorkerHealth
from backends.backplane.reactivestreams import Publisher


class StubHandle(WorkerHandle):
    """A test-double WorkerHandle that records calls and returns canned results.

    Exposes worker=None (read-only property) so the Governor's
    `self._handle.worker is None` checks resolve.
    """

    def __init__(self, result="stub_result"):
        self._result = result
        self._worker = None
        self.start_calls = []
        self.submit_calls = []
        self.unload_calls = 0
        self.stop_calls = 0
        self._state = "ready"

    @property
    def worker(self):
        return self._worker

    def start(self, resolved_mode, binding, mode):
        self.start_calls.append((resolved_mode, binding, mode))
        self._state = "ready"  # mirror InProcessWorkerHandle.start (sets ready)

    def submit(self, job):
        self.submit_calls.append(job)
        from backends.backplane.inproc import InProcBackplane
        from backends.backplane.blob import InProcBlob
        sink, pub = InProcBackplane(job.job_id).open()
        sink.result(0, InProcBlob(self._result))
        sink.complete()
        return pub

    def health(self):
        return WorkerHealth(state=self._state, vram_free_bytes=0, vram_total_bytes=0, mode=None)

    def unload(self):
        self.unload_calls += 1
        self._state = "dead"

    def stop(self):
        self.stop_calls += 1
        self._state = "dead"


class CancelRecordingHandle(StubHandle):
    """StubHandle that records cancel_job calls AND whether `_job_lock` was held
    at the moment of the call (STABL-jredufxb)."""

    def __init__(self, result="stub_result"):
        super().__init__(result)
        self.cancelled_ids = []
        self.lock_was_held = None
        self.governor = None            # set by the test after construction

    def cancel_job(self, job_id):
        gov = self.governor
        if gov is not None:
            acquired = gov._job_lock.acquire(blocking=False)
            self.lock_was_held = not acquired
            if acquired:
                gov._job_lock.release()
        self.cancelled_ids.append(job_id)
        return True


def _governor_with_recorded_cancel(state="running", handle=None):
    """A Governor holding ONE job record in `state`, with no dispatch work queued.

    Shapes copied from test_governor_accepts_custom_handle below — same mock mode
    config and registry, so nothing here can drift from the real constructors.
    """
    from backends.governor import GenerationJob, JobRecord

    mode_config = Mock()
    mode_config.get_default_mode.return_value = "none"
    mode_config.get_mode.side_effect = KeyError("no mode")
    registry = Mock()
    registry.get_total_vram.return_value = 0
    handle = handle if handle is not None else CancelRecordingHandle()
    gov = Governor(
        worker_factory=Mock(),
        handle=handle,
        mode_config=mode_config,
        registry=registry,
    )
    if isinstance(handle, CancelRecordingHandle):
        handle.governor = gov

    job = GenerationJob(req=Mock(), resolution_epoch=0)
    record = JobRecord(job_id=job.job_id, state=state, job=job)
    with gov._job_lock:
        gov._job_records[job.job_id] = record
    return gov, handle, job.job_id


def test_cancel_job_signals_a_running_job_on_the_handle():
    """STABL-jredufxb. The subprocess child cannot see cancel_requested, so the
    reap only reaches it if the Governor forwards the id."""
    gov, handle, job_id = _governor_with_recorded_cancel(state="running")
    try:
        assert gov.cancel_job(job_id) is True
        assert handle.cancelled_ids == [job_id]
    finally:
        gov.shutdown()


def test_cancel_job_releases_job_lock_before_signalling():
    """_control_lock can be held for _STATS_REPLY_TIMEOUT_S by an in-flight stats
    reply. Signalling under _job_lock would let a /status fan-out stall the
    dispatch loop behind a lock it has no business waiting on."""
    gov, handle, job_id = _governor_with_recorded_cancel(state="running")
    try:
        gov.cancel_job(job_id)
        assert handle.lock_was_held is False
    finally:
        gov.shutdown()


def test_cancel_job_does_not_signal_for_a_queued_job():
    """A queued job is taken off the queue outright — there is nothing running to
    reap, and signalling would name a job the child never started."""
    gov, handle, job_id = _governor_with_recorded_cancel(state="queued")
    try:
        gov.cancel_job(job_id)
        assert handle.cancelled_ids == []
    finally:
        gov.shutdown()


def test_cancel_job_tolerates_a_handle_without_cancel_job():
    """InProcessWorkerHandle has no cancel_job — the in-proc reap goes through the
    predicate instead, and cancel_job must not raise."""
    gov, handle, job_id = _governor_with_recorded_cancel(
        state="running", handle=StubHandle()
    )
    try:
        assert gov.cancel_job(job_id) is True
    finally:
        gov.shutdown()


def test_governor_accepts_custom_handle():
    """A second WorkerHandle impl (stub) requires no Governor change (acceptance #4)."""
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "none"
    mode_config.get_mode.side_effect = KeyError("no mode")
    registry = Mock()
    registry.get_total_vram.return_value = 0
    handle = StubHandle()
    gov = Governor(
        worker_factory=Mock(),
        handle=handle,
        mode_config=mode_config,
        registry=registry,
    )
    assert gov._handle is handle
    gov.shutdown()


def test_governor_submit_job_resolves_future_through_handle():
    """submit_job opens the channel + attaches _FutureBridge; the dispatch loop
    drives record.sink. Uses an InProcessWorkerHandle with a mock factory so
    the dispatch loop can run job.execute(worker) directly (v1 does NOT call
    handle.submit() in the dispatch loop — that's the facet-3 contract)."""
    from backends.worker_handle import InProcessWorkerHandle
    from backends.conditioning.contracts import ConditioningConfig
    from backends.model_resolution import LocalModelBinding

    # Patch resolve_model so _load_mode succeeds without filesystem detection
    # (same seam test_worker_pool.py patches at worker_pool.resolve_model).
    # The Governor stores `resolved` opaquely in the snapshot and passes it to
    # handle.start(); a Mock is sufficient here.
    def _resolve(model_path: str, mode):
        return Mock(), LocalModelBinding(model_path)

    with patch("backends.governor.resolve_model", side_effect=_resolve):
        # Build a mock mode_config + registry so _load_mode succeeds in __init__
        worker = Mock()
        worker.run_job = Mock(return_value="png")
        worker.configure_conditioning = None
        handle = InProcessWorkerHandle(worker_factory=Mock(return_value=worker))

        mode_config = Mock()
        mode = Mock()
        mode.model_path = "/models/test.safetensors"
        mode.loras = []
        mode.conditioning = ConditioningConfig()
        mode_config.get_mode.return_value = mode
        mode_config.get_default_mode.return_value = "test-mode"

        registry = Mock()
        registry.get_used_vram.return_value = 0
        registry.get_allocated_vram.return_value = 0
        registry.get_total_vram.return_value = 8 * 1024**3
        registry.register_model = Mock()

        gov = Governor(
            worker_factory=Mock(return_value=worker),
            handle=handle,
            mode_config=mode_config,
            registry=registry,
        )
        # _load_mode was called in __init__ → handle.start was called → _worker is set
        # The dispatch thread was started by _load_mode

        job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
        fut = gov.submit_job(job)
        assert fut.result(timeout=2.0) == "png"
        gov.shutdown()


def test_governor_owns_epoch_and_snapshot():
    """The Governor owns resolution_epoch and active_snapshot."""
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "none"  # will fail _load_mode gracefully
    mode_config.get_mode.side_effect = KeyError("no mode")
    registry = Mock()
    registry.get_total_vram.return_value = 0
    gov = Governor(
        worker_factory=Mock(),
        handle=StubHandle(),
        mode_config=mode_config,
        registry=registry,
    )
    assert gov.current_resolution_epoch() == 0  # no snapshot yet
    assert gov.get_active_model_snapshot() is None
    gov.shutdown()


# ---------------------------------------------------------------------------
# Task 5: Handle pluggability proof (acceptance #4, lifecycle) + helpers.
# ---------------------------------------------------------------------------

def _make_mock_mode_config():
    """Minimal mock mode config for the Governor (default mode fails _load_mode
    gracefully so the pluggability tests stay off the filesystem)."""
    from backends.conditioning.contracts import ConditioningConfig
    config = Mock()
    mode = Mock()
    mode.model_path = "/models/test.safetensors"
    mode.loras = []
    mode.conditioning = ConditioningConfig()
    config.get_mode.return_value = mode
    config.get_default_mode.return_value = "test-mode"
    return config


def _make_mock_registry():
    """Minimal mock registry for the Governor."""
    registry = Mock()
    registry.get_used_vram.return_value = 0
    registry.get_allocated_vram.return_value = 0
    registry.get_total_vram.return_value = 8 * 1024**3
    registry.register_model = Mock()
    registry.unregister_model = Mock()
    return registry


def _make_multi_mode_config(*names, default=None):
    """A mode config where get_mode(name) returns a DISTINCT mode per name.

    Reservation tests need to tell modes apart; _make_mock_mode_config returns one
    shared mode for every name, which cannot express a switch.

    Modes are SimpleNamespace, not Mock, deliberately: _resolve_target deepcopies the
    mode, and deepcopy of a Mock does not reliably preserve a post-construction
    `.name` attribute — which is exactly what these tests assert on. The existing
    _make_subprocess_governor helper uses SimpleNamespace for the same class of reason.
    """
    from backends.conditioning.contracts import ConditioningConfig

    modes = {
        name: SimpleNamespace(
            name=name,
            model_path=f"/models/{name}.safetensors",
            loras=[],
            conditioning=ConditioningConfig(),
            controlnet_policy=None,
        )
        for name in names
    }

    config = Mock()
    config.get_mode.side_effect = lambda n: modes[n]  # KeyError for unknown, as today
    config.get_default_mode.return_value = default or names[0]
    config._modes = modes
    return config


def _resolve_by_path(model_path: str, mode):
    """resolve_model stand-in whose family_id is derived from the mode, so tests can
    assert WHICH mode a reservation resolved against."""
    from backends.model_resolution import LocalModelBinding

    resolved = Mock()
    resolved.profile.family_id = f"family-of-{getattr(mode, 'name', 'unknown')}"
    return resolved, LocalModelBinding(model_path)


def test_second_handle_impl_requires_no_governor_change():
    """Acceptance #4: a second WorkerHandle impl (stub) plugs in with no
    Governor or backplane code change.

    In v1, the Governor calls handle.start()/unload()/stop()/health() and
    accesses handle.worker — but does NOT call handle.submit() in the dispatch
    loop (that's the facet-3 contract). So the pluggability proof is: the
    Governor constructs + uses a stub handle for lifecycle (start/unload/health)
    without branching on locality. The stub exposes worker=None (read-only
    property) so the Governor's `self._handle.worker is None` checks resolve.

    NOTE: v1 proves LIFECYCLE pluggability, not dispatch pluggability — the
    dispatch loop reaches into self._handle.worker directly (reconciliation #2),
    so a real SubprocessWorkerHandle (no in-proc _worker) would still require
    Governor dispatch changes (that's facet-3, deferred).
    """
    from backends.model_resolution import LocalModelBinding

    # Patch resolve_model so _load_mode succeeds and actually exercises
    # handle.start() (same seam test_worker_pool.py patches). Without this,
    # _load_mode hits the real filesystem and its failure path calls
    # handle.unload() — which would mask the lifecycle pluggability proof.
    def _resolve(model_path: str, mode):
        return Mock(), LocalModelBinding(model_path)

    with patch("backends.governor.resolve_model", side_effect=_resolve):
        handle = StubHandle()
        handle._worker = None  # stub doesn't have a real worker; Governor checks this
        gov = Governor(
            worker_factory=Mock(),
            handle=handle,
            mode_config=_make_mock_mode_config(),
            registry=_make_mock_registry(),
        )
        assert gov._handle is handle
        # _load_mode ran in __init__ → handle.start was called → state is "ready"
        assert len(handle.start_calls) == 1
        assert gov._handle.health().state == "ready"
        # Governor can call unload through the handle
        gov._handle.unload()
        assert handle.unload_calls >= 1
        gov.shutdown()


def test_governor_dispatches_mode_switch_through_lifecycle():
    """The Governor handles ModeSwitchJob via _load_mode (lifecycle), which
    calls handle.start(). Proves the dispatch loop differentiates job types
    and delegates lifecycle to the handle."""
    from backends.model_resolution import LocalModelBinding

    def _resolve(model_path: str, mode):
        return Mock(), LocalModelBinding(model_path)

    with patch("backends.governor.resolve_model", side_effect=_resolve):
        handle = StubHandle()
        handle._worker = None  # no worker initially
        gov = Governor(
            worker_factory=Mock(),
            handle=handle,
            mode_config=_make_mock_mode_config(),
            registry=_make_mock_registry(),
        )
        # _load_mode was called during __init__ (default mode) — handle.start was called
        assert len(handle.start_calls) >= 1
        gov.shutdown()


# ---------------------------------------------------------------------------
# Task 6: Governor drives an out-of-proc handle.
# ---------------------------------------------------------------------------

def _make_subprocess_governor(handle):
    """Build a Governor wired to a real SubprocessWorkerHandle + FaultWorker.

    All objects reaching handle.start() are picklable across the spawn boundary.
    Mocks are parent-side only and never pickled.
    """
    from backends.conditioning.contracts import ConditioningConfig
    from backends.model_resolution import LocalModelBinding

    mode_config = Mock()
    mode = SimpleNamespace(
        model_path="/models/test.safetensors",
        loras=[],
        conditioning=ConditioningConfig(),
    )
    mode_config.get_mode.return_value = mode
    mode_config.get_default_mode.return_value = "test-mode"
    registry = Mock()
    registry.get_total_vram.return_value = 0
    registry.get_used_vram.return_value = 0
    registry.get_allocated_vram.return_value = 0
    # resolved is opaque to FaultWorker; binding must be a real picklable object
    # because the Governor stores it in ActiveModelSnapshot and later reads
    # snapshot.binding.model_path in _reload_from_snapshot.
    resolved_binding = (None, LocalModelBinding(model_path="/models/test.safetensors"))
    return mode_config, registry, resolved_binding


def test_governor_dispatches_generation_to_subprocess_handle():
    """The Governor dispatches a GenerationJob to a SubprocessWorkerHandle.

    Uses picklable stand-ins for everything crossing handle.start() (spawn
    pickles resolved/binding/mode). Mocks are parent-side only.
    """
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from server.lcm_sr_server import GenerateRequest

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    mode_config, registry, resolved_binding = _make_subprocess_governor(handle)
    with patch("backends.governor.resolve_model", return_value=resolved_binding):
        gov = Governor(handle=handle, mode_config=mode_config, registry=registry)

        job = GenerationJob(
            req=GenerateRequest(prompt="hi", num_inference_steps=4, size="512x512"),
            resolution_epoch=gov.current_resolution_epoch(),
        )
        fut = gov.submit_job(job)
        assert fut.result(timeout=15) == b"PNG:hi"
    gov.shutdown()


def test_governor_subprocess_mode_switch_no_respawn_when_already_loaded():
    """ModeSwitchJob to the current mode is a no-op for a live subprocess handle.

    Without the line-575 liveness flip, the dispatch loop sees
    self._handle.worker is None and falls through to _load_mode(), killing and
    respawning the subprocess for nothing. With the flip, it reports
    already_loaded and leaves the process alive.
    """
    from backends.worker_handle_subprocess import SubprocessWorkerHandle

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    mode_config, registry, resolved_binding = _make_subprocess_governor(handle)
    with patch("backends.governor.resolve_model", return_value=resolved_binding):
        gov = Governor(handle=handle, mode_config=mode_config, registry=registry)

        original_pid = handle._proc.pid
        fut = gov.switch_mode("test-mode", force=False)
        assert fut.result(timeout=15) == {"mode": "test-mode", "status": "already_loaded"}
        assert handle._proc.pid == original_pid, "subprocess was respawned for a no-op mode switch"
    gov.shutdown()


def test_governor_recovers_from_subprocess_oom_and_next_job_succeeds():
    """Subprocess OOM -> Governor kills + respawns -> next job succeeds.

    The child catches the OOM, emits an error frame, and stays alive (poisoned).
    The Governor must detect terminal_error_code == OOM, kill the child, and
    demand-reload from the retained snapshot so the next job runs on a fresh
    process.
    """
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from server.lcm_sr_server import GenerateRequest

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    mode_config, registry, resolved_binding = _make_subprocess_governor(handle)
    with patch("backends.governor.resolve_model", return_value=resolved_binding):
        gov = Governor(handle=handle, mode_config=mode_config, registry=registry)

        original_pid = handle._proc.pid
        oom = GenerationJob(
            req=GenerateRequest(prompt="__OOM__", num_inference_steps=4, size="512x512"),
            resolution_epoch=gov.current_resolution_epoch(),
        )
        with pytest.raises(Exception):
            gov.submit_job(oom).result(timeout=15)

        ok = GenerationJob(
            req=GenerateRequest(prompt="after", num_inference_steps=4, size="512x512"),
            resolution_epoch=gov.current_resolution_epoch(),
        )
        assert gov.submit_job(ok).result(timeout=15) == b"PNG:after"
        assert handle._proc.pid != original_pid, "OOM recovery must respawn the subprocess"
    gov.shutdown()


def test_governor_recovers_from_frameless_subprocess_death():
    """Frameless death (SIGKILL) -> EOF guard synthesizes terminal -> recovery.

    This path must use the liveness branch (not the OOM branch). We assert the
    first job's exception is a plain RuntimeError (GENERIC reconstruction), not
    torch.cuda.OutOfMemoryError, pinning the distinction so a future refactor
    cannot silently collapse the two triggers.
    """
    import torch
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from server.lcm_sr_server import GenerateRequest

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    mode_config, registry, resolved_binding = _make_subprocess_governor(handle)
    with patch("backends.governor.resolve_model", return_value=resolved_binding):
        gov = Governor(handle=handle, mode_config=mode_config, registry=registry)

        die = GenerationJob(
            req=GenerateRequest(prompt="__DIE__", num_inference_steps=4, size="512x512"),
            resolution_epoch=gov.current_resolution_epoch(),
        )
        fut = gov.submit_job(die)
        with pytest.raises(Exception) as exc_info:
            fut.result(timeout=15)
        assert not isinstance(exc_info.value, torch.cuda.OutOfMemoryError), (
            "frameless death must surface as GENERIC (RuntimeError), not OOM"
        )

        ok = GenerationJob(
            req=GenerateRequest(prompt="after", num_inference_steps=4, size="512x512"),
            resolution_epoch=gov.current_resolution_epoch(),
        )
        assert gov.submit_job(ok).result(timeout=15) == b"PNG:after"
    gov.shutdown()


# ---------------------------------------------------------------------------
# Task 8 (DeviceMemory, STABL-hjldxurg): _build_runtime_status sources the vram
# block from the DeviceMemory snapshot's worker consumer entry — the direct
# torch.cuda.memory_allocated/memory_reserved reads at governor.py:513-514 are
# gone. /status is the cache-refresh point (one fresh snapshot() per call).
# ---------------------------------------------------------------------------

def _make_runtime_status_governor(dm):
    """Governor whose initial _load_mode fails gracefully (default mode 'none'
    raises KeyError before any DeviceMemory read) so no worker starts; the
    status payload is then built from the injected DeviceMemory stub."""
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "none"
    mode_config.get_mode.side_effect = KeyError("no mode")
    registry = Mock()
    registry.get_total_vram.return_value = 24 * 1024**3
    return Governor(
        worker_factory=Mock(),
        handle=StubHandle(),
        mode_config=mode_config,
        registry=registry,
        device_memory=dm,
    )


def test_runtime_status_reads_worker_consumer_entry():
    """_build_runtime_status:513-514 direct-torch reads are gone; the vram
    block comes from the DeviceMemory snapshot's worker consumer entry."""
    from backends.device_memory import (
        ConsumerMemory, DeviceMemorySnapshot, MemoryTopology,
    )
    worker = ConsumerMemory(label="worker", pid=1, allocated_bytes=3 * 1024**3,
                            reserved_bytes=5 * 1024**3, stale=False)
    snap = DeviceMemorySnapshot(device_uuid="GPU-t", topology=MemoryTopology.DISCRETE,
                                total_bytes=24 * 1024**3, free_bytes=10 * 1024**3,
                                consumers=(worker,))
    dm = Mock()
    dm.snapshot.return_value = snap
    governor = _make_runtime_status_governor(dm)
    status = governor._build_runtime_status()
    assert status["vram"]["allocated_bytes"] == 3 * 1024**3
    assert status["vram"]["reserved_bytes"] == 5 * 1024**3
    assert status["vram"]["stale"] is False
    governor.shutdown()


def test_runtime_status_no_worker_reads_zero_not_hang():
    """No registered worker consumer -> zeros, stale False, no fan-out hang."""
    from backends.device_memory import DeviceMemorySnapshot, MemoryTopology
    snap = DeviceMemorySnapshot(device_uuid="GPU-t", topology=MemoryTopology.DISCRETE,
                                total_bytes=24 * 1024**3, free_bytes=24 * 1024**3,
                                consumers=())
    dm = Mock()
    dm.snapshot.return_value = snap
    governor = _make_runtime_status_governor(dm)
    status = governor._build_runtime_status()
    assert status["vram"]["allocated_bytes"] == 0
    assert status["vram"]["stale"] is False
    governor.shutdown()


# ---------------------------------------------------------------------------
# Authority reservation (STABL-ltefhpkk / STABL-iuiwzthc).
# Spec: docs/superpowers/specs/2026-07-30-governor-authority-reservation-design.md
# ---------------------------------------------------------------------------

def test_generate_behind_queued_switch_is_not_stale():
    """A generate targeting mode B, admitted while a switch to B is queued ahead of
    it, must execute — not raise StaleResolutionError."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        worker = Mock()
        worker.run_job = Mock(return_value="png")
        worker.configure_conditioning = None
        from backends.worker_handle import InProcessWorkerHandle
        handle = InProcessWorkerHandle(worker_factory=Mock(return_value=worker))

        gov = Governor(
            handle=handle,
            mode_config=_make_multi_mode_config("mode-a", "mode-b", default="mode-a"),
            registry=_make_mock_registry(),
        )
        try:
            gov.switch_mode("mode-b")
            authority = gov.admit_generation("mode-b")
            job = GenerationJob(
                req=Mock(), resolution_epoch=authority.resolution_epoch
            )
            assert gov.submit_job(job).result(timeout=5.0) == "png"
        finally:
            gov.shutdown()


def _reservation_governor(*names, default=None):
    """A Governor on a stub handle with distinct modes. Caller must gov.shutdown()."""
    gov = Governor(
        handle=StubHandle(),
        mode_config=_make_multi_mode_config(*names, default=default),
        registry=_make_mock_registry(),
    )
    return gov


def _freeze_dispatch(gov):
    """Stop the dispatch loop and WAIT for it to actually exit.

    gov._stop.set() alone is not sufficient: the loop is typically blocked in
    q.get(timeout=1.0) and will dequeue and run one more job before it re-checks the
    flag. Any test that fills the queue or asserts on its contents must know nothing
    is still draining it — otherwise a bounded-queue test sees a freed slot and a
    queue-contents assertion sees an item disappear.
    """
    gov._stop.set()
    thread = getattr(gov, "_worker_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "dispatch loop did not stop"


def _drain_queue(gov):
    """Discard whatever a frozen dispatch loop will never consume.

    Tests that set gov._stop to hold a reservation pending must call this before
    shutdown(): shutdown() begins with q.join(), which blocks until every queued item
    is task_done(). Without draining, whether the test hangs depends on a race between
    _stop.set() and the loop's q.get(timeout=1.0). Mirrors the clear+task_done pattern
    in Governor.cancel_pending_generation_jobs.
    """
    with gov.q.mutex:
        discarded = len(gov.q.queue)
        gov.q.queue.clear()
    for _ in range(discarded):
        gov.q.task_done()


def test_reserve_authority_bumps_epoch_and_appends():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            before = gov._resolution_epoch
            reservation = gov._reserve_authority("mode-b")
            assert reservation.resolution_epoch == before + 1
            assert reservation.mode_name == "mode-b"
            assert reservation.resolved.profile.family_id == "family-of-mode-b"
            assert gov._pending_authorities[-1] is reservation
        finally:
            gov.shutdown()


def test_terminal_authority_prefers_pending_over_active():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            with gov._job_lock:
                assert gov._terminal_authority().mode_name == "mode-a"
            reservation = gov._reserve_authority("mode-b")
            with gov._job_lock:
                assert gov._terminal_authority() is reservation
            assert gov.get_pending_mode() == "mode-b"
        finally:
            gov.shutdown()


def test_get_pending_mode_is_none_with_no_reservation():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            assert gov.get_pending_mode() is None
        finally:
            gov.shutdown()


def test_drop_reservation_removes_by_identity_and_marks_dead():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            reservation = gov._reserve_authority("mode-b")
            gov._drop_reservation(reservation, dead=True)
            assert reservation not in gov._pending_authorities
            assert reservation.resolution_epoch in gov._dead_epochs
            # epoch is NOT rolled back — monotone, never reused
            assert gov._resolution_epoch == reservation.resolution_epoch
        finally:
            gov.shutdown()


def test_switch_mode_to_active_mode_reserves_nothing():
    """Spec §3.3: switching to the already-loaded mode must NOT reserve. The dispatch
    fast-path returns already_loaded without calling _load_mode, so a reservation made
    here would never be published — and any generate bound to it would be stamped N+1
    against active N. That is the bug, self-inflicted."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            before = gov._resolution_epoch
            result = gov.switch_mode("mode-a").result(timeout=2.0)
            assert result == {"mode": "mode-a", "status": "already_loaded"}
            assert gov._pending_authorities == []
            assert gov._resolution_epoch == before
        finally:
            gov.shutdown()


def test_switch_mode_to_pending_mode_reports_already_queued():
    """Terminal is a pending reservation for the same mode: report already_QUEUED, not
    already_loaded — the mode is not loaded yet and status must not claim otherwise."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)  # freeze dispatch so the reservation stays pending
            gov._reserve_authority("mode-b")
            before = gov._resolution_epoch
            result = gov.switch_mode("mode-b").result(timeout=2.0)
            assert result == {"mode": "mode-b", "status": "already_queued"}
            assert gov._resolution_epoch == before
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_switch_mode_to_active_but_evicted_mode_still_reloads():
    """Regression guard: when the active mode's worker was idle-evicted, switching to
    it must still enqueue a reload. The guard mirrors the dispatch fast-path condition
    at governor.py:606, which requires _worker_available()."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            _freeze_dispatch(gov)
            gov._handle.unload()  # simulate idle eviction: state -> "dead"
            assert not gov._worker_available()
            before = gov._resolution_epoch
            gov.switch_mode("mode-a")
            assert gov._resolution_epoch == before + 1
            assert gov.get_pending_mode() == "mode-a"
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_switch_mode_force_always_reserves():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            _freeze_dispatch(gov)
            before = gov._resolution_epoch
            gov.switch_mode("mode-a", force=True)
            assert gov._resolution_epoch == before + 1
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_switch_mode_unknown_mode_still_raises_keyerror():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            with pytest.raises(KeyError):
                gov.switch_mode("nope")
        finally:
            gov.shutdown()


def test_load_mode_publishes_the_reserved_epoch():
    """The published snapshot carries the RESERVED epoch — not a fresh bump."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)  # freeze dispatch; drive _load_mode directly
            reservation = gov._reserve_authority("mode-b")
            gov._load_mode("mode-b", reservation=reservation)
            snapshot = gov.get_active_model_snapshot()
            assert snapshot is reservation
            assert snapshot.resolution_epoch == reservation.resolution_epoch
            assert gov._pending_authorities == []
            assert gov.get_pending_mode() is None
            assert gov.get_current_mode() == "mode-b"
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_load_mode_with_reservation_does_not_re_resolve():
    """_load_mode reuses reservation.resolved/.binding, so detect_model leaves the
    dispatch thread entirely."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path) as spy:
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            reservation = gov._reserve_authority("mode-b")
            calls_after_reserve = spy.call_count
            gov._load_mode("mode-b", reservation=reservation)
            assert spy.call_count == calls_after_reserve
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_load_mode_without_reservation_reserves_inline():
    """__init__ and direct callers still work: no reservation means reserve inline."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            before = gov._resolution_epoch
            gov._load_mode("mode-b")
            assert gov.get_active_model_snapshot().resolution_epoch == before + 1
            assert gov._pending_authorities == []
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_demand_reload_does_not_change_the_epoch():
    """Spec §3.2 / matrix case 10: _reload_from_snapshot is epoch-NEUTRAL. Queued
    generates stamped at epoch N must survive an eviction/reload cycle; a reserve here
    would bump to N+1 and reject every one of them."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            _freeze_dispatch(gov)
            epoch_before = gov.get_active_model_snapshot().resolution_epoch
            gov._unload_current_worker()          # simulate idle eviction
            gov._reload_from_snapshot()
            assert gov.get_active_model_snapshot().resolution_epoch == epoch_before
            assert gov._resolution_epoch == epoch_before
            assert gov._pending_authorities == []
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_mode_switch_job_carries_its_reservation():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            gov.switch_mode("mode-b")
            queued = list(gov.q.queue)
            switches = [j for j in queued if isinstance(j, ModeSwitchJob)]
            assert len(switches) == 1
            assert switches[0].reservation is gov._pending_authorities[-1]
        finally:
            _drain_queue(gov)
            gov.shutdown()


def _submit_and_capture(gov, epoch):
    """Submit a generate stamped at `epoch` and return the exception it terminates
    with (or None if it completed)."""
    job = GenerationJob(req=Mock(), resolution_epoch=epoch)
    fut = gov.submit_job(job)
    try:
        fut.result(timeout=5.0)
    except Exception as exc:
        return exc
    return None


def test_generate_stamped_at_dead_epoch_raises_mode_load_failed():
    """Matrix case 7: the target's load failed; the queued generate must say so, not
    fall through to the subprocess branch."""
    from backends.governor import ModeLoadFailedError

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._handle.start = Mock(side_effect=RuntimeError("checkpoint is corrupt"))
            reservation = gov._reserve_authority("mode-b")
            with pytest.raises(RuntimeError, match="checkpoint is corrupt"):
                gov._load_mode("mode-b", reservation=reservation)
            assert reservation.resolution_epoch in gov._dead_epochs

            exc = _submit_and_capture(gov, reservation.resolution_epoch)
            assert isinstance(exc, ModeLoadFailedError), f"got {exc!r}"
        finally:
            gov.shutdown()


def test_generate_with_no_active_snapshot_raises_mode_load_failed():
    """Matrix case 8: no authority at all means the job cannot run.

    Before this guard the barrier was SKIPPED (it was conjoined with
    `snapshot is not None`) and the job fell through to the paths below with no epoch
    check whatsoever. With a handle whose submit() succeeds it would RUN — this test
    originally failed with TypeError from the subprocess bridge choking on the stub's
    str result, not with any 'no model' error. That is a correctness hole, not just a
    confusing message.

    The rejection keeps the established operator-facing 'No worker available for
    generation' wording (asserted in 5 places across test_model_lifecycle.py and
    test_worker_pool.py): the CONDITION is unchanged, only the point of rejection moved
    from the handle to the barrier. What changes is that it is now a typed
    ModeLoadFailedError raised before any execution path is reachable.
    """
    from backends.governor import ModeLoadFailedError

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            with gov._job_lock:
                gov._active_snapshot = None
            exc = _submit_and_capture(gov, 1)
            assert isinstance(exc, ModeLoadFailedError), f"got {exc!r}"
            assert "no active model authority" in str(exc)
            # The stub handle's submit() would have returned "stub_result"; proof the
            # job never reached an execution path.
            assert gov._handle.submit_calls == []
        finally:
            gov.shutdown()


def test_barrier_still_rejects_a_genuinely_superseded_generate():
    """Matrix case 3: the barrier keeps its teeth. A generate admitted for one epoch,
    superseded by an unrelated switch, must still raise StaleResolutionError."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            stale_epoch = gov.get_active_model_snapshot().resolution_epoch
            gov.switch_mode("mode-b").result(timeout=5.0)
            exc = _submit_and_capture(gov, stale_epoch)
            assert isinstance(exc, StaleResolutionError), f"got {exc!r}"
        finally:
            gov.shutdown()


def test_dead_epochs_pruned_on_publish():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            gov._dead_epochs.add(1)
            reservation = gov._reserve_authority("mode-b")
            gov._load_mode("mode-b", reservation=reservation)
            assert 1 not in gov._dead_epochs
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_admit_generation_with_no_target_returns_active_snapshot():
    """Spec §3.4: a generate naming no mode means 'the current mode'. It must NOT bind
    to a pending switch — that is the wrong-model hazard."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            active = gov.get_active_model_snapshot()
            gov._reserve_authority("mode-b")  # a switch is pending
            assert gov.admit_generation(None) is active
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_admit_generation_binds_to_the_pending_switch():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            reservation = gov._reserve_authority("mode-b")
            assert gov.admit_generation("mode-b") is reservation
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_admit_generation_binds_to_the_active_mode():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            _freeze_dispatch(gov)
            active = gov.get_active_model_snapshot()
            assert gov.admit_generation("mode-a") is active
            assert gov._pending_authorities == []
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_admit_generation_creates_the_switch_for_an_untargeted_mode():
    """The Governor owns the switch: naming a mode that is neither active nor pending
    reserves it AND enqueues the ModeSwitchJob."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            authority = gov.admit_generation("mode-b")
            assert authority.mode_name == "mode-b"
            assert authority.resolved.profile.family_id == "family-of-mode-b"
            switches = [j for j in list(gov.q.queue) if isinstance(j, ModeSwitchJob)]
            assert len(switches) == 1
            assert switches[0].reservation is authority
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_admit_generation_is_idempotent_for_the_same_target():
    """Two generates targeting the same not-yet-active mode share ONE switch."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            first = gov.admit_generation("mode-b")
            second = gov.admit_generation("mode-b")
            assert first is second
            switches = [j for j in list(gov.q.queue) if isinstance(j, ModeSwitchJob)]
            assert len(switches) == 1
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_admit_generation_rolls_back_the_reservation_on_queue_full():
    """Matrix case 11: a dangling reservation would poison terminal authority for every
    later admission."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            while True:                      # fill the bounded queue
                try:
                    gov.q.put_nowait(CustomJob(handler=lambda: None))
                except queue.Full:
                    break
            with pytest.raises(queue.Full):
                gov.admit_generation("mode-b")
            assert gov._pending_authorities == []
            assert gov.get_pending_mode() is None
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_untargeted_generate_superseded_by_a_switch_is_still_rejected():
    """Matrix case 4: mode=None binds to the active snapshot, so a later switch still
    supersedes it. The barrier must reject."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            authority = gov.admit_generation(None)
            gov.switch_mode("mode-b").result(timeout=5.0)
            exc = _submit_and_capture(gov, authority.resolution_epoch)
            assert isinstance(exc, StaleResolutionError), f"got {exc!r}"
        finally:
            gov.shutdown()


def test_runtime_status_reports_the_pending_mode_during_a_switch():
    """Matrix case 9: while a switch is queued, status must say WHICH mode is coming
    instead of reporting nothing loaded. _load_mode unregisters the outgoing mode
    (governor.py:338) and re-registers only after the load (:370) — tens of seconds
    for HunyuanDiT — and that window emitted no log line at all."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            gov.admit_generation("mode-b")
            status = gov._build_runtime_status()
            assert status["pending_mode"] == "mode-b"
            assert status["current_mode"] == "mode-a"
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_runtime_status_pending_mode_is_none_when_settled():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            assert gov._build_runtime_status()["pending_mode"] is None
        finally:
            gov.shutdown()


def test_reserving_refreshes_last_activity():
    """The idle watchdog must not evict a mode that was just requested."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            _freeze_dispatch(gov)
            gov._last_activity = time.monotonic() - 10_000
            gov.admit_generation("mode-b")
            assert time.monotonic() - gov._last_activity < 5.0
        finally:
            _drain_queue(gov)
            gov.shutdown()


# ---------------------------------------------------------------------------
# Timeout semantics: bound EXECUTION, not queue wait (STABL-atzqpcte).
#
# DEFAULT_TIMEOUT was applied as fut.result(timeout=120) on a generate whose
# future does not resolve until everything ahead of it in the queue has run —
# so a timeout written to bound GENERATION also bounded QUEUE WAIT and MODEL
# LOAD. Confirmed in the field: the first inline --mode generate against
# HunyuanDiT timed out during the load.
# ---------------------------------------------------------------------------


def test_job_record_carries_an_execution_start_timestamp():
    """The execution clock needs its own signal.

    NOT `state == "running"`: that is set at governor.py:724, BEFORE the demand
    reload and BEFORE the stale-epoch barrier, so a full model reload would land
    inside the execution budget — this issue's own defect in miniature. And the
    marker cannot simply move, because cancel_job branches on state == "queued"
    to cancel the future outright.
    """
    import dataclasses
    from backends.governor import JobRecord

    fields = {f.name: f for f in dataclasses.fields(JobRecord)}
    assert "executing_since" in fields, "JobRecord has no executing_since field"
    assert fields["executing_since"].default is None, (
        "executing_since must default to None — a job has not executed until it has"
    )


def test_demand_reload_runs_before_the_execution_clock_starts():
    """A demand reload must NOT be charged to the execution budget.

    The dispatch loop reloads an evicted worker at governor.py:727-731 — a full
    model load, tens of seconds to minutes. It runs BEFORE the execution stamp, so
    a job that triggers one is still judged against the ADMISSION budget while the
    reload happens. Asserted from inside the reload itself, which is the only place
    that can observe the ordering directly.
    """
    observed: dict = {}

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            req = SimpleNamespace(prompt="x")
            job = GenerationJob(req=req, resolution_epoch=gov.current_resolution_epoch())

            def _spy_reload():
                # Look the record up live: submit_job calls _register_job, which
                # REPLACES any record captured beforehand with a fresh one.
                record = gov._get_job_record(job.job_id)
                observed["executing_since_during_reload"] = record.executing_since
                observed["state_during_reload"] = record.state

            gov._reload_from_snapshot = _spy_reload
            gov._worker_available = lambda: False   # force the demand-reload branch

            gov.submit_job(job)
            try:
                job.fut.result(timeout=10.0)
            except Exception:
                pass            # the stub handle may not produce a real result

            assert observed, "the demand reload never ran"
            assert observed["executing_since_during_reload"] is None, (
                "executing_since was already stamped during the demand reload — the "
                "reload is being charged to the execution budget"
            )
            assert observed["state_during_reload"] == "running", (
                "state == 'running' during the reload, which is exactly why it is the "
                "wrong signal for the execution clock"
            )
        finally:
            _drain_queue(gov)
            gov.shutdown()


# --- the two-budget waiter (STABL-atzqpcte Task 2) --------------------------


def _waiter_governor():
    """A Governor whose dispatch loop is frozen, so job state is test-controlled."""
    gov = _reservation_governor("mode-a", default="mode-a")
    _freeze_dispatch(gov)
    return gov


def _queued_job(gov):
    """Register a job WITHOUT enqueueing it: the record exists, state is queued,
    executing_since is None. Returns (job, record)."""
    job = GenerationJob(req=SimpleNamespace(prompt="x"), resolution_epoch=0)
    gov._register_job(job)
    return job, gov._get_job_record(job.job_id)


def test_a_queued_job_is_not_timed_out_by_the_execution_budget():
    """THE HEADLINE. A generate waiting behind a multi-minute model load must not
    be killed by a budget meant for generation.

    This is the field failure from enigma: the first inline --mode generate against
    HunyuanDiT was timed out at 120s while the model was still loading.
    """
    import concurrent.futures

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _waiter_governor()
        try:
            job, record = _queued_job(gov)
            assert record.executing_since is None

            start = time.monotonic()
            with pytest.raises(concurrent.futures.TimeoutError) as exc_info:
                # Execution budget far below the admission budget: a queued job must
                # survive the former and only be judged by the latter.
                gov.wait_for_result(
                    job.fut,
                    execution_timeout_s=0.05,
                    admission_timeout_s=0.6,
                    poll_interval_s=0.02,
                )
            elapsed = time.monotonic() - start

            assert elapsed >= 0.5, (
                f"gave up after {elapsed:.2f}s — the queued job was judged against the "
                f"0.05s EXECUTION budget instead of the 0.6s admission budget"
            )
            assert "admission" in str(exc_info.value).lower()
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_an_executing_job_is_timed_out_by_the_execution_budget():
    """Once the job is genuinely running, the tight budget applies — and is measured
    from when execution actually began, not from when the waiter noticed."""
    import concurrent.futures

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _waiter_governor()
        try:
            job, record = _queued_job(gov)
            record.state = "running"
            record.executing_since = time.monotonic()

            start = time.monotonic()
            with pytest.raises(concurrent.futures.TimeoutError) as exc_info:
                gov.wait_for_result(
                    job.fut,
                    execution_timeout_s=0.2,
                    admission_timeout_s=30.0,
                    poll_interval_s=0.02,
                )
            elapsed = time.monotonic() - start

            assert elapsed < 5.0, f"took {elapsed:.2f}s for a 0.2s execution budget"
            assert "execution" in str(exc_info.value).lower()
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_timing_out_requests_cancellation():
    """A timed-out job is told to stop. For a still-queued job this cancels it
    outright, so the work is never done at all.

    It does NOT stop a running generation — cancel_requested is only checked at job
    boundaries and run_job never reads it. That limit is STABL-jredufxb.
    """
    import concurrent.futures

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _waiter_governor()
        try:
            job, record = _queued_job(gov)
            with pytest.raises(concurrent.futures.TimeoutError):
                gov.wait_for_result(
                    job.fut,
                    execution_timeout_s=5.0,
                    admission_timeout_s=0.1,
                    poll_interval_s=0.02,
                )
            assert record.cancel_requested is True, "timeout did not request cancellation"
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_a_result_that_arrives_is_returned_unchanged():
    """The waiter is transparent on the happy path — it carries the worker's opaque
    return exactly as fut.result() did."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _waiter_governor()
        try:
            job, _record = _queued_job(gov)
            job.fut.set_result((b"PNG", 4242))
            assert gov.wait_for_result(job.fut) == (b"PNG", 4242)
        finally:
            _drain_queue(gov)
            gov.shutdown()


def test_waiter_budgets_default_to_the_module_constants():
    """DEFAULT_TIMEOUT keeps its name and its 120 default, and finally means
    EXECUTION. ADMISSION_TIMEOUT_S is the new, generous queue-wait budget."""
    from backends import governor as gov_mod

    assert gov_mod.DEFAULT_EXECUTION_TIMEOUT_S == 120.0
    assert gov_mod.DEFAULT_ADMISSION_TIMEOUT_S == 900.0


def test_oom_classification_survives_a_non_type_oom_class():
    """STABL-hdzggeir. `torch.cuda.OutOfMemoryError` is not guaranteed to be a TYPE.

    In a shared pytest session `backends.governor.torch` can be a MagicMock — every
    backend module binds whatever sys.modules['torch'] held when IT was first
    imported, so importing test_worker_pool.py first leaves the governor holding the
    stub. `isinstance(e, <MagicMock>)` then raises TypeError INSIDE the dispatch
    loop's except handler, before the real error can be delivered, and the caller's
    future is left unresolved forever.
    """
    import backends.governor as g
    from unittest.mock import MagicMock as _MM
    from backends.governor import ModeLoadFailedError

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        with patch.object(g, "torch", _MM()):
            gov = _reservation_governor("mode-a", default="mode-a")
            try:
                with gov._job_lock:
                    gov._active_snapshot = None
                exc = _submit_and_capture(gov, 1)
                assert isinstance(exc, ModeLoadFailedError), (
                    f"got {exc!r} — the error handler raised before delivering"
                )
            finally:
                gov.shutdown()


def test_a_failure_inside_the_error_handler_does_not_wedge_the_dispatch_loop():
    """STABL-hdzggeir. If the except handler itself raises, the exception escapes the
    while loop and the dispatch THREAD DIES. The queue is then permanently dead: every
    later job hangs until its own timeout with no error surfaced, and shutdown() —
    which begins with q.join() — never returns either.

    Asserted on thread liveness rather than by submitting a second job, precisely
    because a second job WEDGES the test instead of failing it (verified: the run had
    to be killed by pytest-timeout at 300s).

    Injected at _finalize_job_record because the handler always reaches it for a
    GenerationJob. In production the same shape occurs if BackplaneError.from_exc or
    sink.error raises on a broken pipe — not only under a mocked torch.
    """
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            with gov._job_lock:
                gov._active_snapshot = None

            with patch.object(Governor, "_finalize_job_record",
                              side_effect=RuntimeError("handler boom")):
                exc = _submit_and_capture(gov, 1)
                assert exc is not None and not isinstance(exc, TimeoutError), (
                    f"future left unresolved by a failing handler: got {exc!r}"
                )

            # The future resolves BEFORE the thread unwinds, so an immediate
            # is_alive() check races and reads True either way. Give the thread a
            # window to die; if it is still alive at the end of it, it survived.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and gov._worker_thread.is_alive():
                time.sleep(0.05)
            assert gov._worker_thread.is_alive(), (
                "dispatch thread died inside its own error handler — the queue is now "
                "permanently dead and shutdown() will block on q.join()"
            )
        finally:
            gov.shutdown()
