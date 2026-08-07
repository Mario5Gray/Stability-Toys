"""STABL-bpsfmoke: the correlation fields a structured log line carries."""
import os
import threading

import pytest

from server import log_context


@pytest.fixture(autouse=True)
def _isolate():
    saved = log_context.static_fields()
    yield
    for key in list(log_context.static_fields()):
        log_context.set_static_field(key, None)
    for key, value in saved.items():
        log_context.set_static_field(key, value)


def test_job_id_is_absent_until_bound():
    assert log_context.current_job_id() is None


def test_bind_job_id_sets_and_restores():
    with log_context.bind_job_id("abc123"):
        assert log_context.current_job_id() == "abc123"
    assert log_context.current_job_id() is None


def test_bind_job_id_restores_even_when_the_body_raises():
    with pytest.raises(ValueError):
        with log_context.bind_job_id("abc123"):
            raise ValueError("boom")
    assert log_context.current_job_id() is None


def test_nested_binds_restore_the_outer_value():
    with log_context.bind_job_id("outer"):
        with log_context.bind_job_id("inner"):
            assert log_context.current_job_id() == "inner"
        assert log_context.current_job_id() == "outer"


def test_a_bind_does_NOT_leak_into_a_thread_started_from_it():
    """The dispatch loop is a LONG-LIVED thread that copies nothing from the
    submitter. This test pins that fact, because it is the reason the loop has to
    set the var itself rather than inherit it (spec 7.4)."""
    seen = []
    with log_context.bind_job_id("submitter-job"):
        t = threading.Thread(target=lambda: seen.append(log_context.current_job_id()))
        t.start()
        t.join()
    assert seen == [None]


def test_process_fields_carry_pid_and_hostname():
    fields = log_context.static_fields()
    assert fields["pid"] == os.getpid()
    assert isinstance(fields["hostname"], str) and fields["hostname"]


def test_static_fields_returns_a_COPY():
    fields = log_context.static_fields()
    fields["mode"] = "tampered"
    assert "mode" not in log_context.static_fields()


def test_set_static_field_with_None_REMOVES_the_field():
    log_context.set_static_field("mode", "sdxl")
    assert log_context.static_fields()["mode"] == "sdxl"
    log_context.set_static_field("mode", None)
    assert "mode" not in log_context.static_fields()
