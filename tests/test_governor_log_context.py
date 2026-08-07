"""STABL-bpsfmoke: job_id correlation across the dispatch thread."""
import logging
from unittest.mock import Mock, patch

from backends.governor import Governor, GenerationJob
from backends.model_resolution import LocalModelBinding
from server import log_context


class _JobIdProbe(logging.Handler):
    """Records the contextvar AS SEEN BY THE EMITTING THREAD.

    emit() runs on whichever thread logged, so this is the only way to observe the
    dispatch loop's context from a test.
    """

    def __init__(self):
        super().__init__()
        self.seen = []

    def emit(self, record):
        self.seen.append((record.getMessage(), log_context.current_job_id()))


def _governor(run_job, device_memory=None):
    """Verbatim from tests/test_governor.py:265, with an injectable run_job."""
    from backends.conditioning.contracts import ConditioningConfig
    from backends.worker_handle import InProcessWorkerHandle

    worker = Mock()
    worker.run_job = run_job
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

    kwargs = {} if device_memory is None else {"device_memory": device_memory}
    return Governor(
        worker_factory=Mock(return_value=worker),
        handle=handle,
        mode_config=mode_config,
        registry=registry,
        **kwargs,
    )


def _resolve(model_path, mode):
    return Mock(), LocalModelBinding(model_path)


def _recording_run_job(seen):
    def _run(*args, **kwargs):
        seen.append(log_context.current_job_id())
        return "png"
    return _run


def test_a_job_sees_its_own_id():
    seen = []
    with patch("backends.governor.resolve_model", side_effect=_resolve):
        gov = _governor(_recording_run_job(seen))
        try:
            job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
            assert gov.submit_job(job).result(timeout=5.0) == "png"
        finally:
            gov.shutdown()
            log_context.set_static_field("mode", None)
    assert seen == [job.job_id]


def test_each_job_sees_its_OWN_id_not_the_previous_one():
    seen = []
    with patch("backends.governor.resolve_model", side_effect=_resolve):
        gov = _governor(_recording_run_job(seen))
        try:
            first = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
            gov.submit_job(first).result(timeout=5.0)
            second = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
            gov.submit_job(second).result(timeout=5.0)
        finally:
            gov.shutdown()
            log_context.set_static_field("mode", None)
    assert seen == [first.job_id, second.job_id]


def test_the_loop_does_NOT_carry_a_finished_jobs_id_into_its_OWN_lines():
    """The failure this test exists for. Set-without-reset leaves the last job's id
    on every line the dispatch thread emits afterwards — which reads as a real
    correlation, survives review, and misleads whoever is reading the incident."""
    probe = _JobIdProbe()
    gov_logger = logging.getLogger("backends.governor")
    gov_logger.addHandler(probe)
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"))
            try:
                job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
                gov.submit_job(job).result(timeout=5.0)
            finally:
                gov.shutdown()      # "[Governor] Dispatch loop stopped" — same thread
                log_context.set_static_field("mode", None)
    finally:
        gov_logger.removeHandler(probe)

    stopped = [jid for msg, jid in probe.seen if "Dispatch loop stopped" in msg]
    assert stopped, "the loop never logged its stop line — test cannot conclude"
    assert stopped == [None] * len(stopped)


def test_mode_is_published_as_a_process_field_on_load():
    """`mode` is process-global, not per-job: every line the process writes while a
    mode is resident belongs to that mode, including lines from threads that never
    saw a job."""
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"))
            assert log_context.static_fields()["mode"] == "test-mode"
            gov.shutdown()
    finally:
        log_context.set_static_field("mode", None)


def test_mode_is_REMOVED_on_unload_not_set_to_empty():
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"))
            gov.unload_current_model()
            assert "mode" not in log_context.static_fields()
            gov.shutdown()
    finally:
        log_context.set_static_field("mode", None)


def test_a_DEMAND_RELOAD_republishes_mode():
    """Idle eviction clears the field (correctly — nothing is resident). The demand
    reload that follows brings the worker back WITHOUT going through _load_mode, so
    it has to republish, or every line after an eviction/reload cycle claims no mode
    is loaded while one is. Found by reading the reload path, not by the plan."""
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"))
            try:
                gov._unload_current_worker(reason="idle_evict")
                assert "mode" not in log_context.static_fields()
                gov._reload_from_snapshot()
                assert log_context.static_fields()["mode"] == "test-mode"
            finally:
                gov.shutdown()
    finally:
        log_context.set_static_field("mode", None)


def test_device_uuid_is_published_from_the_selected_provider():
    dm = Mock()
    dm.device_uuid = "GPU-abc123"
    dm.snapshot.return_value = Mock(consumers=())
    dm.cached_snapshot.return_value = Mock(consumers=())
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"), device_memory=dm)
            assert log_context.static_fields()["device_uuid"] == "GPU-abc123"
            gov.shutdown()
    finally:
        log_context.set_static_field("device_uuid", None)
        log_context.set_static_field("mode", None)


def test_a_provider_without_a_uuid_leaves_the_field_ABSENT():
    """Mock(spec=[...]) is deliberate: a bare Mock() auto-creates device_uuid and
    returns a child Mock, so this test would pass without exercising anything. That
    exact failure cost a green-for-the-wrong-reason test in STABL-asawxgvp."""
    dm = Mock(spec=["snapshot", "cached_snapshot", "reclaim"])
    dm.snapshot.return_value = Mock(consumers=())
    dm.cached_snapshot.return_value = Mock(consumers=())
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"), device_memory=dm)
            assert "device_uuid" not in log_context.static_fields()
            gov.shutdown()
    finally:
        log_context.set_static_field("mode", None)
