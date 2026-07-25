from __future__ import annotations

from abc import ABC, abstractmethod

from .frames import BackplaneError, BlobRef


class JobSink(ABC):
    """Producer-side handle the worker drives. ack/progress are non-blocking;
    result/complete/error are synchronous must-deliver."""

    @abstractmethod
    def ack(self, queued_position: int = 0) -> None: ...

    @abstractmethod
    def progress(self, step: int, total: int, stage: str = "denoise") -> None: ...

    @abstractmethod
    def result(self, seed: int, blob: BlobRef) -> None: ...

    @abstractmethod
    def complete(self) -> None: ...

    @abstractmethod
    def error(self, err: BackplaneError) -> None: ...

    @property
    @abstractmethod
    def cancelled(self) -> bool: ...
