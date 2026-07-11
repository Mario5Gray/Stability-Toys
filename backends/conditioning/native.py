from .artifacts import DelegatedConditioning
from .contracts import (
    ConditioningRequest,
    ConditioningServiceRequirements,
    ModelContext,
)
from .invocation import CompletedInvocation


class NativeConditioningService:
    requirements = ConditioningServiceRequirements()

    def invoke(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> CompletedInvocation:
        del context
        return CompletedInvocation.success(
            DelegatedConditioning(request.prompt, request.negative_prompt)
        )
