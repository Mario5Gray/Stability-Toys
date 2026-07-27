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

    def run_job(self, job):
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
