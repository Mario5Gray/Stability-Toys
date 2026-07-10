import asyncio
from dataclasses import dataclass
from typing import Callable

from .contracts import (
    ConditioningConfig,
    ConditioningFilter,
    ConditioningRequest,
    ConditioningService,
    ModelContext,
)
from .invocation import (
    CompletedInvocation,
    ConditioningInvocation,
    NativeFallbackInvocation,
)
from .native import NativeConditioningService


ServiceFactory = Callable[[], ConditioningService]
FilterFactory = Callable[[], ConditioningFilter]


class ConditioningRegistry:
    def __init__(self) -> None:
        self._services: dict[str, ServiceFactory] = {}
        self._filters: dict[str, FilterFactory] = {}

    @classmethod
    def with_builtins(cls) -> "ConditioningRegistry":
        registry = cls()
        registry.register_service("native", NativeConditioningService)
        return registry

    def register_service(self, name: str, factory: ServiceFactory) -> None:
        if name in self._services:
            raise ValueError(f"conditioning service '{name}' is already registered")
        self._services[name] = factory

    def register_filter(self, name: str, factory: FilterFactory) -> None:
        if name in self._filters:
            raise ValueError(f"conditioning filter '{name}' is already registered")
        self._filters[name] = factory

    def create_service(self, name: str) -> ConditioningService:
        try:
            factory = self._services[name]
        except KeyError as error:
            raise ValueError(f"unknown conditioning service '{name}'") from error
        return factory()

    def create_filter(self, name: str) -> ConditioningFilter:
        try:
            factory = self._filters[name]
        except KeyError as error:
            raise ValueError(f"unknown conditioning filter '{name}'") from error
        return factory()


@dataclass(frozen=True)
class ConditioningChain:
    _service: ConditioningService

    def invoke(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> ConditioningInvocation:
        return self._service.invoke(request, context)


@dataclass(frozen=True)
class _FallbackConditioningService:
    primary: ConditioningService
    native_service: ConditioningService
    service_name: str

    def invoke(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> ConditioningInvocation:
        try:
            primary = self.primary.invoke(request, context)
        except (Exception, asyncio.CancelledError) as error:
            primary = CompletedInvocation.failure(error)
        return NativeFallbackInvocation(
            primary=primary,
            native_service=self.native_service,
            request=request,
            context=context,
            service_name=self.service_name,
        )


@dataclass(frozen=True)
class _FilteredConditioningService:
    conditioning_filter: ConditioningFilter
    next_service: ConditioningService

    def invoke(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> ConditioningInvocation:
        return self.conditioning_filter.apply(request, context, self.next_service)


def build_conditioning_chain(
    config: ConditioningConfig,
    context: ModelContext,
    registry: ConditioningRegistry | None = None,
) -> ConditioningChain:
    registry = registry or ConditioningRegistry.with_builtins()

    service_name = config.service
    if not service_name:
        if not config.fallback.native_when_unconfigured:
            raise ValueError("conditioning service is required")
        service_name = "native"

    service = registry.create_service(service_name)
    if service.requirements.local_encoder_bundle and context.local_encoder_bundle is None:
        raise ValueError(
            f"conditioning service '{service_name}' requires a local encoder bundle"
        )

    if config.fallback.native_on_failure and service_name != "native":
        service = _FallbackConditioningService(
            primary=service,
            native_service=registry.create_service("native"),
            service_name=service_name,
        )

    for filter_name in reversed(config.filters):
        service = _FilteredConditioningService(
            conditioning_filter=registry.create_filter(filter_name),
            next_service=service,
        )

    return ConditioningChain(service)
