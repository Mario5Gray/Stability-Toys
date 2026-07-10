from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

from .artifacts import ConditioningArtifact

if TYPE_CHECKING:
    from .contracts import (
        ConditioningRequest,
        ConditioningService,
        ModelContext,
    )


logger = logging.getLogger(__name__)


class ConditioningInvocation(Protocol):
    def result(self, timeout: float | None = None) -> ConditioningArtifact: ...

    def done(self) -> bool: ...

    def cancel(self) -> bool: ...

    def exception(self, timeout: float | None = None) -> BaseException | None: ...


@dataclass(frozen=True)
class CompletedInvocation:
    _artifact: ConditioningArtifact | None = None
    _exception: BaseException | None = None

    @classmethod
    def success(cls, artifact: ConditioningArtifact) -> CompletedInvocation:
        return cls(_artifact=artifact)

    @classmethod
    def failure(cls, exception: BaseException) -> CompletedInvocation:
        return cls(_exception=exception)

    def result(self, timeout: float | None = None) -> ConditioningArtifact:
        del timeout
        if self._exception is not None:
            raise self._exception
        if self._artifact is None:
            raise RuntimeError("completed invocation has no result")
        return self._artifact

    def done(self) -> bool:
        return True

    def cancel(self) -> bool:
        return False

    def exception(self, timeout: float | None = None) -> BaseException | None:
        del timeout
        return self._exception


@dataclass(frozen=True)
class TransformingInvocation:
    downstream: ConditioningInvocation
    on_result: Callable[[ConditioningArtifact], ConditioningArtifact]

    def result(self, timeout: float | None = None) -> ConditioningArtifact:
        return self.on_result(self.downstream.result(timeout))

    def done(self) -> bool:
        return self.downstream.done()

    def cancel(self) -> bool:
        return self.downstream.cancel()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        try:
            self.result(timeout)
        except BaseException as error:
            return error
        return None


class NativeFallbackInvocation:
    def __init__(
        self,
        primary: ConditioningInvocation,
        native_service: ConditioningService,
        request: ConditioningRequest,
        context: ModelContext,
        service_name: str,
    ) -> None:
        self.primary = primary
        self.native_service = native_service
        self.request = request
        self.context = context
        self.service_name = service_name
        self._fallback: ConditioningInvocation | None = None
        self._cancelled = False

    def result(self, timeout: float | None = None) -> ConditioningArtifact:
        if self._fallback is not None:
            return self._fallback.result(timeout)

        try:
            return self.primary.result(timeout)
        except (Exception, asyncio.CancelledError) as error:
            if self._cancelled:
                raise
            logger.warning(
                "conditioning fallback for service %s after %r",
                self.service_name,
                error,
            )
            self._fallback = self.native_service.invoke(self.request, self.context)
            return self._fallback.result(timeout)

    def done(self) -> bool:
        active = self._fallback if self._fallback is not None else self.primary
        return active.done()

    def cancel(self) -> bool:
        active = self._fallback if self._fallback is not None else self.primary
        cancelled = active.cancel()
        if cancelled:
            self._cancelled = True
        return cancelled

    def exception(self, timeout: float | None = None) -> BaseException | None:
        try:
            self.result(timeout)
        except BaseException as error:
            return error
        return None
