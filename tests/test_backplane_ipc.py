"""Task 5: the stdlib IPC transport exercised across a REAL process boundary (spawn).

Proves acceptance #2: the same JobSink/Subscriber interface carries frames + the PNG
payload (via shared_memory) and the inbound cancel (subscription.cancel() -> a reverse
control frame -> sink.cancelled) across a genuine subprocess — without moving the
production CudaWorker to a subprocess (that is facet-3).
"""
import asyncio
import multiprocessing as mp
import time
from multiprocessing import shared_memory

import pytest

from backends.backplane.ipc import IpcJobSink, drain_to_subscriber
from backends.backplane.frames import Progress, Result, BackplaneError, BackplaneErrorCode
from backends.backplane.reactivestreams import Subscriber


# --- worker bodies run in the spawned child process -------------------------

def _forward_worker(conn):
    sink = IpcJobSink(conn)
    sink.ack()
    sink.progress(1, 2)
    sink.progress(2, 2)
    sink.result(99, b"PNGBYTES")
    sink.complete()
    conn.recv_bytes()  # block until parent signals DONE — keeps the segment alive
    conn.close()


def _cancellable_worker(conn):
    sink = IpcJobSink(conn)
    sink.ack()
    for i in range(200):
        if sink.cancelled:  # polls the reverse control channel
            sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
            conn.close()
            return
        sink.progress(i, 200)
        time.sleep(0.005)
    sink.result(0, b"done")
    sink.complete()
    conn.recv_bytes()
    conn.close()


# --- parent-side subscribers ------------------------------------------------

class Collector(Subscriber):
    def __init__(self):
        self.frames, self.done, self.error = [], False, None

    def on_subscribe(self, s):
        s.request(1 << 62)

    def on_next(self, v):
        self.frames.append(v)

    def on_error(self, e):
        self.error = e

    def on_complete(self):
        self.done = True


class CancelAfter(Subscriber):
    def __init__(self, after=3):
        self.after, self.progress, self.error, self.done = after, 0, None, False
        self._sub = None

    def on_subscribe(self, s):
        self._sub = s

    def on_next(self, v):
        if isinstance(v, Progress):
            self.progress += 1
            if self.progress == self.after:
                self._sub.cancel()  # send cancel across the boundary

    def on_error(self, e):
        self.error = e

    def on_complete(self):
        self.done = True


@pytest.mark.timeout(30)
def test_frames_and_bytes_cross_a_real_process_boundary():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_forward_worker, args=(child_conn,))
    proc.start()
    child_conn.close()

    col = Collector()
    result_name = drain_to_subscriber(parent_conn, col)

    result = next(f for f in col.frames if isinstance(f, Result))
    assert asyncio.run(result.image.read()) == b"PNGBYTES"  # bytes round-trip via shared_memory
    result.image.close()                                    # consumer owns the unlink
    parent_conn.send_bytes(b"\x00DONE")                     # release the child
    proc.join(timeout=10)

    assert [type(f).__name__ for f in col.frames] == ["Ack", "Progress", "Progress", "Result"]
    assert col.done is True
    with pytest.raises(FileNotFoundError):                  # segment unlinked
        shared_memory.SharedMemory(name=result_name)


@pytest.mark.timeout(30)
def test_inbound_cancel_reaches_child_across_boundary():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_cancellable_worker, args=(child_conn,))
    proc.start()
    child_conn.close()

    col = CancelAfter(after=3)
    drain_to_subscriber(parent_conn, col)
    proc.join(timeout=10)

    assert col.error is not None and col.error.code is BackplaneErrorCode.CANCELLED
    assert col.progress < 200   # child observed the cancel and stopped early
    assert col.done is False
