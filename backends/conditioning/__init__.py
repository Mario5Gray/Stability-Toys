from .artifacts import (
    ConditioningArtifact,
    ConditioningCompatibility,
    DelegatedConditioning,
    MaterializedConditioning,
)
from .contracts import (
    ConditioningConfig,
    ConditioningFallbackConfig,
    ConditioningFilter,
    ConditioningRequest,
    ConditioningService,
    ConditioningServiceRequirements,
    LocalEncoderBundle,
    ModelContext,
    ModelContextDescriptor,
)
from .invocation import (
    CompletedInvocation,
    ConditioningInvocation,
    NativeFallbackInvocation,
    TransformingInvocation,
)
from .native import NativeConditioningService
from .registry import (
    ConditioningChain,
    ConditioningRegistry,
    build_conditioning_chain,
)


def __getattr__(name: str):
    if name == "CompelConditioningService":
        from .compel_service import CompelConditioningService

        return CompelConditioningService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompletedInvocation",
    "CompelConditioningService",
    "ConditioningArtifact",
    "ConditioningCompatibility",
    "ConditioningConfig",
    "ConditioningFallbackConfig",
    "ConditioningFilter",
    "ConditioningInvocation",
    "ConditioningChain",
    "ConditioningRegistry",
    "ConditioningRequest",
    "ConditioningService",
    "ConditioningServiceRequirements",
    "DelegatedConditioning",
    "LocalEncoderBundle",
    "MaterializedConditioning",
    "ModelContext",
    "ModelContextDescriptor",
    "NativeFallbackInvocation",
    "NativeConditioningService",
    "TransformingInvocation",
    "build_conditioning_chain",
]
