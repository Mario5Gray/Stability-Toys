from dataclasses import dataclass, field
from typing import Literal, Protocol

from .invocation import ConditioningInvocation


@dataclass(frozen=True)
class ConditioningRequest:
    prompt: str
    negative_prompt: str | None


@dataclass(frozen=True)
class ModelContextDescriptor:
    model_family: Literal["sd15", "sdxl"]
    tokenizer_max_length: int
    encoder_identities: tuple[str, ...]
    hidden_dimensions: tuple[int, ...]
    pooled_required: bool
    encode_dtype_name: str
    device: str


class LocalEncoderBundle(Protocol):
    def tokenizers(self) -> tuple[object, ...]: ...

    def text_encoders(self) -> tuple[object, ...]: ...

    def live_dtype(self) -> object: ...


@dataclass(frozen=True)
class ModelContext:
    descriptor: ModelContextDescriptor
    local_encoder_bundle: LocalEncoderBundle | None = None


@dataclass(frozen=True)
class ConditioningServiceRequirements:
    local_encoder_bundle: bool = False


@dataclass(frozen=True)
class ConditioningFallbackConfig:
    native_when_unconfigured: bool = True
    native_on_failure: bool = False


@dataclass(frozen=True)
class ConditioningConfig:
    service: str | None = None
    filters: tuple[str, ...] = ()
    fallback: ConditioningFallbackConfig = field(
        default_factory=ConditioningFallbackConfig
    )

    def requires_configurable_worker(self) -> bool:
        return bool((self.service and self.service != "native") or self.filters)


class ConditioningService(Protocol):
    requirements: ConditioningServiceRequirements

    def invoke(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> ConditioningInvocation: ...


class ConditioningFilter(Protocol):
    def apply(
        self,
        request: ConditioningRequest,
        context: ModelContext,
        next_service: ConditioningService,
    ) -> ConditioningInvocation: ...
