from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class BlobRef(ABC):
    """Transport-resolved payload handle. read() once, then close()."""

    @abstractmethod
    async def read(self) -> bytes: ...

    @abstractmethod
    def close(self) -> None: ...


@dataclass(frozen=True)
class Ack:
    job_id: str
    queued_position: int = 0


@dataclass(frozen=True)
class Progress:
    job_id: str
    step: int
    total: int
    stage: str = "denoise"  # total == -1 => indeterminate


@dataclass(frozen=True)
class Result:
    job_id: str
    seed: int
    image: BlobRef


class BackplaneErrorCode(Enum):
    OOM = "oom"
    STALE_EPOCH = "stale_epoch"
    CANCELLED = "cancelled"
    GENERIC = "generic"
    TIMEOUT = "timeout"  # RESERVED — not emitted this issue (consumer-side today)


def classify_exception(exc: BaseException) -> BackplaneErrorCode:
    """Duck-type classification — no torch / worker_pool imports (avoids a cycle)."""
    name = type(exc).__name__
    if isinstance(exc, concurrent.futures.CancelledError) or name == "CancelledError":
        return BackplaneErrorCode.CANCELLED
    if name == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return BackplaneErrorCode.OOM
    if name == "StaleResolutionError":
        return BackplaneErrorCode.STALE_EPOCH
    return BackplaneErrorCode.GENERIC


def _reconstruct(code: BackplaneErrorCode, message: str) -> Exception:
    """IPC-only: rebuild an exception from a code when the live instance can't cross."""
    if code is BackplaneErrorCode.CANCELLED:
        return concurrent.futures.CancelledError(message)
    if code is BackplaneErrorCode.OOM:
        try:
            import torch  # lazy — parent side has torch
            return torch.cuda.OutOfMemoryError(message)
        except Exception:
            return RuntimeError(message)
    # STALE_EPOCH reconstruction across IPC is deferred (see plan reconciliation #3).
    return RuntimeError(message)


class BackplaneError(Exception):
    def __init__(self, code: BackplaneErrorCode, message: str = "", original: Exception | None = None):
        super().__init__(message or code.value)
        self.code = code
        self.message = message
        self.original = original

    @classmethod
    def from_exc(cls, exc: Exception) -> "BackplaneError":
        return cls(classify_exception(exc), str(exc), original=exc)

    def to_exception(self) -> Exception:
        if self.original is not None:
            return self.original  # in-proc invariant: the live instance
        return _reconstruct(self.code, self.message)
