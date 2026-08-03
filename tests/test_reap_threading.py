"""STABL-jredufxb: should_cancel reaches the worker from the job."""
from backends.governor import GenerationJob


class SpyWorker:
    def __init__(self):
        self.seen = None

    def run_job(self, job, progress=None, should_cancel=None):
        self.seen = {"progress": progress, "should_cancel": should_cancel}
        return b"PNG"


def _job():
    class Req:
        prompt = "x"
        num_inference_steps = 4
    return GenerationJob(req=Req(), resolution_epoch=1)


def test_execute_threads_should_cancel_to_run_job():
    worker = SpyWorker()

    def predicate():
        return False

    _job().execute(worker, progress=None, should_cancel=predicate)
    assert worker.seen["should_cancel"] is predicate


def test_execute_still_threads_progress():
    worker = SpyWorker()

    def emitter(step, total, stage):
        return None

    _job().execute(worker, progress=emitter)
    assert worker.seen["progress"] is emitter
    assert worker.seen["should_cancel"] is None
