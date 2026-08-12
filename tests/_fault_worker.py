"""Fault-injecting fake PipelineWorker for facet-3 spawn-boundary tests.

Module-level (not a fixture) so the spawn child can import it by dotted ref. Its
run_job echoes the prompt as opaque bytes, or on a sentinel prompt injects an OOM
(alive-but-poisoned) or a frameless death (SIGKILL) — the two failure modes the
Governor's durable recovery (Task 7) must survive.
"""


class FaultWorker:
    """Fake PipelineWorker for spawn-boundary tests. run_job echoes; can be told to
    OOM or die frameless via the request payload's prompt sentinel."""

    def __init__(self, *args, **kwargs):
        pass

    def run_job(self, job, progress=None, should_cancel=None):
        prompt = getattr(job.req, "prompt", "")
        if prompt == "__OOM__":
            import torch
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")
        if prompt == "__DIE__":
            import os
            import signal
            os.kill(os.getpid(), signal.SIGKILL)   # frameless death
        return b"PNG:" + prompt.encode()


def make_fault_worker(worker_id, resolved, binding):
    return FaultWorker()


class RecordingWorker:
    """Records constructor args and conditioning configuration for wiring tests."""

    def __init__(self, worker_id, resolved, binding):
        self.worker_id = worker_id
        self.resolved = resolved
        self.binding = binding
        self.conditioning_config = None

    def configure_conditioning(self, config):
        self.conditioning_config = config

    def run_job(self, job, progress=None, should_cancel=None):
        if self.conditioning_config is None:
            raise RuntimeError("configure_conditioning was not called in child")
        return b"PNG:" + getattr(job.req, "prompt", "").encode()


def make_recording_worker(worker_id, resolved, binding):
    return RecordingWorker(worker_id, resolved, binding)


class PayloadEchoWorker:
    """Reports back what actually arrived on the job (STABL-spxwqlan).

    The real CudaWorker reads these defensively —
    `getattr(job, 'init_image', None)` / `getattr(job, 'controlnet_bindings', []) or []`
    — and None/[] is the legitimate txt2img shape, so a payload lost in transit
    produces a plausible image instead of an error. This worker makes the loss
    observable by returning what it saw.
    """

    def __init__(self, *args, **kwargs):
        pass

    def run_job(self, job, progress=None, should_cancel=None):
        bindings = getattr(job, "controlnet_bindings", []) or []
        init_image = getattr(job, "init_image", None)
        return {
            "prompt": getattr(job.req, "prompt", None),
            "init_image": init_image,
            "binding_ids": [b.attachment_id for b in bindings],
            "control_image_bytes": [b.control_image_bytes for b in bindings],
            "strengths": [b.strength for b in bindings],
        }


def make_payload_echo_worker(worker_id, resolved, binding):
    return PayloadEchoWorker()


# --- startup-handshake fault factories (STABL-wotsqcjb) --------------------
#
# These fail INSIDE the spawn child, before _worker_main can signal READY —
# the window that used to hang the parent forever. Each covers a different
# guard: the child-side failure frame, the parent-side liveness check, and
# the parent-side deadline.


def make_exploding_worker(worker_id, resolved, binding):
    """Raises during construction: an ordinary startup exception, so the child
    can catch it and report a traceback before it dies."""
    raise RuntimeError("worker construction failed (injected)")


def make_suiciding_worker(worker_id, resolved, binding):
    """SIGKILLs the child mid-construction. No failure frame is possible — only
    the parent's is_alive() check can notice this one."""
    import os
    import signal

    os.kill(os.getpid(), signal.SIGKILL)


def make_hanging_worker(worker_id, resolved, binding):
    """Blocks far longer than any injected timeout, staying ALIVE throughout, so
    only the parent's deadline can end the wait."""
    import time

    time.sleep(3600)


class CancellableWorker:
    """Polls should_cancel the way a denoise loop polls it at each step, so a
    spawn-boundary test can prove the child stops MID-JOB rather than at a job
    boundary (STABL-jredufxb).

    Raises concurrent.futures.CancelledError deliberately: classify_exception()
    maps only that (or a class literally named CancelledError) to
    BackplaneErrorCode.CANCELLED, and the subprocess parent path does no
    cancel_requested remap — a bespoke type would arrive as GENERIC.
    """

    def __init__(self, *args, **kwargs):
        pass

    def run_job(self, job, progress=None, should_cancel=None):
        import time
        from concurrent.futures import CancelledError

        for _ in range(500):
            if should_cancel is not None and should_cancel():
                raise CancelledError("job cancelled at step boundary")
            time.sleep(0.01)
        return b"PNG:finished"


def make_cancellable_worker(worker_id, resolved, binding):
    return CancellableWorker()


class LogLineWorker:
    """Emits a real log line from inside the job body and returns it FORMATTED
    (STABL-zuhuxwvf).

    Returning the formatted line rather than the contextvar is deliberate: the
    thing that was measured missing in Loki is a log line without a `job_id`
    field, and StabilityFormatter reading the contextvar is the only reason a
    line ever has one. Asserting on the rendered payload tests what an operator
    actually queries.

    `backends.cuda_worker` as the logger name is not decoration — that is the
    exact logger whose child-pid lines carried no job_id in the field data.
    """

    def __init__(self, *args, **kwargs):
        pass

    def run_job(self, job, progress=None, should_cancel=None):
        import logging

        from server.log_format import StabilityFormatter

        record = logging.LogRecord(
            "backends.cuda_worker", logging.INFO, "/x.py", 1,
            "generating in the child", (), None,
        )
        return StabilityFormatter().format(record).encode()


def make_log_line_worker(worker_id, resolved, binding):
    return LogLineWorker()
