import time
from backends.liveness import SubprocessLiveness


class _FakeProc:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def test_live_when_process_alive_and_heartbeat_fresh():
    liv = SubprocessLiveness(_FakeProc(alive=True), stale_after_s=1.0)
    liv.note_heartbeat()
    assert liv.state() == "live"


def test_dead_when_process_exited():
    liv = SubprocessLiveness(_FakeProc(alive=False), stale_after_s=1.0)
    liv.note_heartbeat()
    assert liv.state() == "dead"


def test_dead_when_heartbeat_stale():
    liv = SubprocessLiveness(_FakeProc(alive=True), stale_after_s=0.05)
    liv.note_heartbeat()
    time.sleep(0.1)
    assert liv.state() == "dead"
