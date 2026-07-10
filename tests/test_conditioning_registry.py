import logging

import pytest

from backends.conditioning.artifacts import DelegatedConditioning
from backends.conditioning.contracts import (
    ConditioningConfig,
    ConditioningFallbackConfig,
    ConditioningRequest,
    ConditioningServiceRequirements,
    ModelContext,
    ModelContextDescriptor,
)
from backends.conditioning.invocation import (
    CompletedInvocation,
    NativeFallbackInvocation,
    TransformingInvocation,
)
from backends.conditioning.registry import (
    ConditioningRegistry,
    build_conditioning_chain,
)


def context(local_bundle=None):
    return ModelContext(
        descriptor=ModelContextDescriptor(
            model_family="sd15",
            tokenizer_max_length=77,
            encoder_identities=("clip-l",),
            hidden_dimensions=(768,),
            pooled_required=False,
            encode_dtype_name="float16",
            device="cuda:0",
        ),
        local_encoder_bundle=local_bundle,
    )


class RecordingFilter:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def apply(self, request, model_context, next_service):
        self.events.append(f"{self.name}:before")
        downstream = next_service.invoke(request, model_context)

        def record_after(artifact):
            self.events.append(f"{self.name}:after")
            return artifact

        return TransformingInvocation(downstream, record_after)


class RaisingResultFilter:
    def apply(self, request, model_context, next_service):
        downstream = next_service.invoke(request, model_context)

        def fail_after(artifact):
            del artifact
            raise ValueError("filter transform failed")

        return TransformingInvocation(downstream, fail_after)


class NeedsLocalService:
    requirements = ConditioningServiceRequirements(local_encoder_bundle=True)

    def invoke(self, request, model_context):
        del request, model_context
        return CompletedInvocation.success(DelegatedConditioning("local", None))


class BrokenService:
    requirements = ConditioningServiceRequirements()

    def invoke(self, request, model_context):
        del request, model_context
        return CompletedInvocation.failure(RuntimeError("boom"))


class RaisingService:
    requirements = ConditioningServiceRequirements()

    def invoke(self, request, model_context):
        del request, model_context
        raise RuntimeError("invoke failed")


class SuccessfulService:
    requirements = ConditioningServiceRequirements()

    def invoke(self, request, model_context):
        del request, model_context
        return CompletedInvocation.success(DelegatedConditioning("conditioned", None))


class PendingInvocation:
    def __init__(self):
        self.cancel_calls = 0

    def result(self, timeout=None):
        del timeout
        raise RuntimeError("not complete")

    def done(self):
        return False

    def cancel(self):
        self.cancel_calls += 1
        return True

    def exception(self, timeout=None):
        del timeout
        return None


class RecordingNativeService:
    def __init__(self):
        self.calls = 0

    def invoke(self, request, model_context):
        del request, model_context
        self.calls += 1
        return CompletedInvocation.success(DelegatedConditioning("native", None))


def test_duplicate_service_registration_is_rejected():
    registry = ConditioningRegistry()
    registry.register_service("native", lambda: object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_service("native", lambda: object())


def test_duplicate_filter_registration_is_rejected():
    registry = ConditioningRegistry()
    registry.register_filter("trace", lambda: object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_filter("trace", lambda: object())


def test_empty_configuration_builds_native_chain():
    chain = build_conditioning_chain(ConditioningConfig(), context())
    artifact = chain.invoke(ConditioningRequest("cat", None), context()).result()
    assert artifact == DelegatedConditioning("cat", None)


def test_unconfigured_service_fails_when_native_default_is_disabled():
    config = ConditioningConfig(
        fallback=ConditioningFallbackConfig(native_when_unconfigured=False)
    )
    with pytest.raises(ValueError, match="conditioning service is required"):
        build_conditioning_chain(config, context())


def test_unknown_explicit_service_fails_composition():
    with pytest.raises(ValueError, match="unknown conditioning service 'missing'"):
        build_conditioning_chain(ConditioningConfig(service="missing"), context())


def test_unknown_filter_fails_composition():
    with pytest.raises(ValueError, match="unknown conditioning filter 'missing'"):
        build_conditioning_chain(
            ConditioningConfig(filters=("missing",)),
            context(),
        )


def test_first_configured_filter_is_outermost_and_result_transform_is_lazy():
    events = []
    registry = ConditioningRegistry.with_builtins()
    registry.register_filter("outer", lambda: RecordingFilter("outer", events))
    registry.register_filter("inner", lambda: RecordingFilter("inner", events))
    chain = build_conditioning_chain(
        ConditioningConfig(filters=("outer", "inner")),
        context(),
        registry,
    )

    invocation = chain.invoke(ConditioningRequest("cat", None), context())
    assert events == ["outer:before", "inner:before"]
    invocation.result()
    assert events == ["outer:before", "inner:before", "inner:after", "outer:after"]


def test_missing_required_local_bundle_fails_composition():
    registry = ConditioningRegistry.with_builtins()
    registry.register_service("needs-local", NeedsLocalService)
    with pytest.raises(ValueError, match="local encoder bundle"):
        build_conditioning_chain(
            ConditioningConfig(service="needs-local"),
            context(),
            registry,
        )


def test_required_local_bundle_allows_composition_when_present():
    registry = ConditioningRegistry.with_builtins()
    registry.register_service("needs-local", NeedsLocalService)
    chain = build_conditioning_chain(
        ConditioningConfig(service="needs-local"),
        context(local_bundle=object()),
        registry,
    )
    assert chain.invoke(ConditioningRequest("cat", None), context()).result() == (
        DelegatedConditioning("local", None)
    )


def test_native_fallback_handles_service_failure_and_logs(caplog):
    registry = ConditioningRegistry.with_builtins()
    registry.register_service("broken", BrokenService)
    config = ConditioningConfig(
        service="broken",
        fallback=ConditioningFallbackConfig(native_on_failure=True),
    )
    with caplog.at_level(logging.WARNING):
        artifact = build_conditioning_chain(config, context(), registry).invoke(
            ConditioningRequest("cat", None),
            context(),
        ).result()
    assert artifact == DelegatedConditioning("cat", None)
    assert "conditioning fallback" in caplog.text


def test_native_fallback_handles_synchronous_service_raise_lazily():
    registry = ConditioningRegistry.with_builtins()
    registry.register_service("raising", RaisingService)
    config = ConditioningConfig(
        service="raising",
        fallback=ConditioningFallbackConfig(native_on_failure=True),
    )

    invocation = build_conditioning_chain(config, context(), registry).invoke(
        ConditioningRequest("cat", None),
        context(),
    )

    assert invocation.result() == DelegatedConditioning("cat", None)


def test_filter_result_failure_does_not_enter_native_fallback():
    registry = ConditioningRegistry.with_builtins()
    registry.register_service("successful", SuccessfulService)
    registry.register_filter("raising", RaisingResultFilter)
    config = ConditioningConfig(
        service="successful",
        filters=("raising",),
        fallback=ConditioningFallbackConfig(native_on_failure=True),
    )
    invocation = build_conditioning_chain(config, context(), registry).invoke(
        ConditioningRequest("cat", None),
        context(),
    )

    with pytest.raises(ValueError, match="filter transform failed"):
        invocation.result()


def test_outer_cancel_never_starts_native_fallback():
    primary = PendingInvocation()
    invocation = NativeFallbackInvocation(
        primary=primary,
        native_service=RecordingNativeService(),
        request=ConditioningRequest("cat", None),
        context=context(),
        service_name="remote",
    )
    assert invocation.cancel() is True
    assert invocation.native_service.calls == 0
