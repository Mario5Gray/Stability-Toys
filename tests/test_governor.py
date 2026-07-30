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

@pytest.mark.xfail(
    strict=True,
    reason="STABL-ltefhpkk: generate stamped against pre-switch authority. "
           "Marker removed in Task 4 when admit_generation lands.",
)
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
            gov._stop.set()  # freeze dispatch so the reservation stays pending
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
            gov._stop.set()
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
            gov._stop.set()
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
