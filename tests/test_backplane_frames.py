import concurrent.futures
import pytest
from backends.backplane.frames import (
    Ack, Progress, Result, BlobRef,
    BackplaneError, BackplaneErrorCode, classify_exception,
)
from backends.backplane.reactivestreams import Publisher, Subscriber, Subscription


def test_frames_carry_job_id_and_defaults():
    assert Ack("j1").queued_position == 0
    assert Progress("j1", step=3, total=20).stage == "denoise"
    assert Progress("j1", step=0, total=-1).total == -1  # indeterminate allowed


def test_classify_exception_by_ducktype():
    assert classify_exception(concurrent.futures.CancelledError()) is BackplaneErrorCode.CANCELLED
    assert classify_exception(RuntimeError("CUDA out of memory")) is BackplaneErrorCode.OOM

    class StaleResolutionError(RuntimeError):
        pass
    assert classify_exception(StaleResolutionError("x")) is BackplaneErrorCode.STALE_EPOCH
    assert classify_exception(ValueError("boom")) is BackplaneErrorCode.GENERIC


def test_backplane_error_preserves_live_instance():
    orig = RuntimeError("CUDA out of memory")
    err = BackplaneError.from_exc(orig)
    assert err.code is BackplaneErrorCode.OOM
    assert err.to_exception() is orig  # in-proc: the live instance, not a rebuild


def test_backplane_error_reconstructs_from_code_when_no_original():
    err = BackplaneError(BackplaneErrorCode.CANCELLED, "gone")
    rebuilt = err.to_exception()
    assert isinstance(rebuilt, concurrent.futures.CancelledError)


def test_abcs_are_abstract():
    for abc in (Publisher, Subscriber, Subscription):
        with pytest.raises(TypeError):
            abc()
