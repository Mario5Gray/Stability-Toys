from __future__ import annotations

from .blob import encode_frame, decode_frame, SharedMemBlob
from .frames import Result, BackplaneError, BackplaneErrorCode
from .interface import JobSink
from .reactivestreams import Subscriber, Subscription

# Control markers (not schema-versioned frames — they carry no payload).
_CANCEL = b"\x00CANCEL"
_COMPLETE = b"\x00COMPLETE"


class IpcJobSink(JobSink):
    """Producer-side sink over a duplex `multiprocessing.Connection`. Frames go out;
    an inbound cancel control frame arrives on the same connection and surfaces via
    `cancelled` (the cross-process cancel channel — in-proc keys off cancel_requested,
    but a subprocess worker cannot see that, so it reads this)."""

    def __init__(self, conn, job_id: str = "job"):
        self._conn = conn
        self._job_id = job_id
        self._cancelled = False
        self._live_blob: SharedMemBlob | None = None

    def _poll_cancel(self) -> None:
        while self._conn.poll():
            if self._conn.recv_bytes() == _CANCEL:
                self._cancelled = True

    def ack(self, queued_position: int = 0) -> None:
        from .frames import Ack
        self._conn.send_bytes(encode_frame(Ack(self._job_id, queued_position)))

    def progress(self, step: int, total: int, stage: str = "denoise") -> None:
        from .frames import Progress
        self._conn.send_bytes(encode_frame(Progress(self._job_id, step, total, stage)))

    def result(self, seed: int, blob) -> None:
        data = bytes(blob) if isinstance(blob, (bytes, bytearray)) else None
        shm_blob = SharedMemBlob.create(data) if data is not None else blob
        self._live_blob = shm_blob                      # armed for the reaper
        self._conn.send_bytes(encode_frame(Result(self._job_id, seed, shm_blob)))
        self._live_blob = None                          # ownership handed to consumer

    def complete(self) -> None:
        self._conn.send_bytes(_COMPLETE)

    def error(self, err: BackplaneError) -> None:
        self._reap()
        self._conn.send_bytes(encode_frame(err))

    def _reap(self) -> None:
        # Unlink an allocated-but-unsent segment if the stream ends on error/cancel.
        if self._live_blob is not None:
            self._live_blob.close()
            self._live_blob = None

    @property
    def cancelled(self) -> bool:
        self._poll_cancel()
        return self._cancelled


class _IpcSubscription(Subscription):
    """Consumer-side handle. `cancel()` sends the reverse control frame to the producer."""

    def __init__(self, conn):
        self._conn = conn

    def request(self, n: int) -> None:
        # Bounded demand across the pipe is not needed for the boundary proof; the
        # producer is not rate-limited by request(n) in this transport yet.
        pass

    def cancel(self) -> None:
        self._conn.send_bytes(_CANCEL)


def drain_to_subscriber(conn, subscriber: Subscriber):
    """Synchronously pump frames from `conn` into `subscriber` until a terminal.
    Returns the Result blob name (for lifecycle assertions), or None."""
    subscriber.on_subscribe(_IpcSubscription(conn))
    result_name = None
    while True:
        try:
            raw = conn.recv_bytes()
        except EOFError:
            # Frameless death (spec §6b): the producer end closed before a terminal.
            # Synthesize a failure terminal so a waiting Future never hangs.
            subscriber.on_error(BackplaneError(BackplaneErrorCode.GENERIC, "worker connection closed"))
            break
        if raw == _COMPLETE:
            subscriber.on_complete()
            break
        frame = decode_frame(raw)
        if isinstance(frame, BackplaneError):
            subscriber.on_error(frame)
            break
        if isinstance(frame, Result):
            result_name = frame.image.name
        subscriber.on_next(frame)
    return result_name
