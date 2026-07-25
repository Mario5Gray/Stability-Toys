from __future__ import annotations

import json
from multiprocessing import shared_memory

from .frames import (
    Ack, Progress, Result, BlobRef, BackplaneError, BackplaneErrorCode,
)

SCHEMA_VERSION = 1


class InProcBlob(BlobRef):
    """In-process payload carrier. Holds either PNG bytes (the streaming/IPC shape)
    or, on the no-op facade path, the worker's OPAQUE run_job result passed straight
    through (e.g. a `(png, seed)` tuple in production, or a test's `"test_result"`).
    The async `read()` is for the byte-payload case; the facade uses `read_sync()`."""

    def __init__(self, data):
        self._data = data

    async def read(self) -> bytes:
        return self._data

    def read_sync(self):
        """Synchronous read for the one loop-less consumer: the compat Subscriber
        (_FutureBridge in worker_pool, Task 4). It runs on the worker thread with no
        asyncio loop, so it cannot `await read()`. Returns the held payload opaquely
        (facade carries the raw worker result here). Intentionally NOT on the BlobRef
        ABC — only the in-proc facade adapter, which knows it holds an InProcBlob,
        calls it; general consumers use the async `read()`."""
        return self._data

    def close(self) -> None:
        # No-op (spec §4.4): in-proc holds no OS resource. Kept for ABC symmetry so
        # `read()` after `close()` still returns the held bytes.
        pass


class SharedMemBlob(BlobRef):
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size

    @classmethod
    def create(cls, data: bytes) -> "SharedMemBlob":
        shm = shared_memory.SharedMemory(create=True, size=max(len(data), 1))
        assert shm.buf is not None  # live segment always has a buffer
        shm.buf[: len(data)] = data
        blob = cls(shm.name, len(data))
        # Consumer owns the unlink (read-once). Detach the PRODUCER process's
        # resource_tracker so it does not unlink the segment when the producer exits —
        # across a spawn boundary that would race the consumer's read. Semi-private
        # API; best-effort. Correctness also relies on the consumer reading before the
        # producer is released (handshake in the boundary test).
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        shm.close()  # keep the segment alive (not unlinked); consumer re-attaches by name
        return blob

    async def read(self) -> bytes:
        shm = shared_memory.SharedMemory(name=self.name)
        try:
            assert shm.buf is not None  # live segment always has a buffer
            return bytes(shm.buf[: self.size])
        finally:
            shm.close()

    def close(self) -> None:
        try:
            shm = shared_memory.SharedMemory(name=self.name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def encode_frame(frame) -> bytes:
    if isinstance(frame, Ack):
        body = {"t": "ack", "job_id": frame.job_id, "pos": frame.queued_position}
    elif isinstance(frame, Progress):
        body = {"t": "progress", "job_id": frame.job_id, "step": frame.step,
                "total": frame.total, "stage": frame.stage}
    elif isinstance(frame, Result):
        assert isinstance(frame.image, SharedMemBlob), "IPC Result carries a SharedMemBlob"
        body = {"t": "result", "job_id": frame.job_id, "seed": frame.seed,
                "blob": {"name": frame.image.name, "size": frame.image.size}}
    elif isinstance(frame, BackplaneError):
        body = {"t": "error", "code": frame.code.value, "msg": frame.message}
    else:
        raise TypeError(f"un-encodable frame: {type(frame)!r}")
    return bytes([SCHEMA_VERSION]) + json.dumps(body).encode("utf-8")


def decode_frame(raw: bytes):
    if not raw:
        raise ValueError("empty frame")
    version = raw[0]
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version}")
    body = json.loads(raw[1:].decode("utf-8"))
    t = body["t"]
    if t == "ack":
        return Ack(body["job_id"], body["pos"])
    if t == "progress":
        return Progress(body["job_id"], body["step"], body["total"], body["stage"])
    if t == "result":
        b = body["blob"]
        return Result(body["job_id"], body["seed"], SharedMemBlob(b["name"], b["size"]))
    if t == "error":
        return BackplaneError(BackplaneErrorCode(body["code"]), body["msg"])
    raise ValueError(f"unknown frame tag {t!r}")
