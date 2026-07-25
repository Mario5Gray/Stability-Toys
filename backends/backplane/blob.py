from __future__ import annotations

import json
from multiprocessing import shared_memory

from .frames import (
    Ack, Progress, Result, BlobRef, BackplaneError, BackplaneErrorCode,
)

SCHEMA_VERSION = 1


class InProcBlob(BlobRef):
    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data

    def read_sync(self) -> bytes:
        """In-proc-only synchronous read — the facade path has no event loop to await read()."""
        return self._data

    def close(self) -> None:
        self._data = b""


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
