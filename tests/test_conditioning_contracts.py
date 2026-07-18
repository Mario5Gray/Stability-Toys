import asyncio
from concurrent.futures import CancelledError

import pytest

from backends.conditioning.artifacts import (
    ConditioningCompatibility,
    DelegatedConditioning,
    MaterializedConditioning,
)
from backends.conditioning.contracts import (
    ConditioningRequest,
    ModelContext,
    ModelContextDescriptor,
)
from backends.conditioning.invocation import (
    CompletedInvocation,
    NativeFallbackInvocation,
    TransformingInvocation,
)


def _context() -> ModelContext:
    return ModelContext(
        descriptor=ModelContextDescriptor(
            model_family="sd15",
            tokenizer_max_length=77,
            encoder_identities=("clip-l",),
            hidden_dimensions=(768,),
            pooled_required=False,
            encode_dtype_name="float16",
            device="cuda:0",
        )
    )


class PendingInvocation:
    def __init__(self) -> None:
        self.result_calls = 0
        self.cancel_calls = 0
        self.cancelled = False

    def result(self, timeout: float | None = None):
        del timeout
        self.result_calls += 1
        if self.cancelled:
            raise CancelledError()
        raise RuntimeError("not complete")

    def done(self) -> bool:
        return False

    def cancel(self) -> bool:
        self.cancel_calls += 1
        self.cancelled = True
        return True

    def exception(self, timeout: float | None = None):
        del timeout
        return None


class RecordingNativeService:
    def __init__(self, artifact: DelegatedConditioning) -> None:
        self.artifact = artifact
        self.calls = 0

    def invoke(self, request: ConditioningRequest, context: ModelContext):
        del request, context
        self.calls += 1
        return CompletedInvocation.success(self.artifact)


def test_unknown_conditioning_family_fails_at_construction():
    from backends.family_profiles import UnknownFamilyError

    with pytest.raises(UnknownFamilyError):
        ModelContextDescriptor(
            model_family="hunyuandit",
            tokenizer_max_length=77,
            encoder_identities=("clip-l",),
            hidden_dimensions=(768,),
            pooled_required=False,
            encode_dtype_name="float16",
            device="cuda:0",
        )

    with pytest.raises(UnknownFamilyError):
        ConditioningCompatibility(
            model_family="nope",
            encoder_identities=("clip-l",),
            hidden_dimensions=(768,),
            pooled_required=False,
            dtype_name="float16",
        )


def test_open_conditioning_family_strings_still_accept_known_ids():
    descriptor = ModelContextDescriptor(
        model_family="sdxl",
        tokenizer_max_length=77,
        encoder_identities=("clip-l", "clip-g"),
        hidden_dimensions=(768, 1280),
        pooled_required=True,
        encode_dtype_name="float16",
        device="cuda:0",
    )
    assert descriptor.model_family == "sdxl"


def test_conditioning_request_preserves_optional_negative_prompt():
    request = ConditioningRequest(prompt="cat", negative_prompt=None)
    assert request.prompt == "cat"
    assert request.negative_prompt is None


def test_artifact_union_keeps_payload_types_out_of_descriptor():
    marker = object()
    compatibility = ConditioningCompatibility(
        model_family="sd15",
        encoder_identities=("clip-l",),
        hidden_dimensions=(768,),
        pooled_required=False,
        dtype_name="float16",
    )
    artifact = MaterializedConditioning(
        slots={"prompt_embeds": marker, "negative_prompt_embeds": marker},
        compatibility=compatibility,
    )
    assert artifact.kind == "materialized"
    assert artifact.slots["prompt_embeds"] is marker
    assert compatibility.dtype_name == "float16"


def test_completed_invocation_returns_artifact_and_cannot_cancel_completed_work():
    artifact = DelegatedConditioning(prompt="cat", negative_prompt=None)
    invocation = CompletedInvocation.success(artifact)
    assert invocation.done() is True
    assert invocation.cancel() is False
    assert invocation.exception() is None
    assert invocation.result() is artifact


def test_completed_invocation_reraises_stored_exception():
    failure = RuntimeError("encode failed")
    invocation = CompletedInvocation.failure(failure)
    assert invocation.done() is True
    assert invocation.exception() is failure
    with pytest.raises(RuntimeError, match="encode failed"):
        invocation.result()


def test_transforming_invocation_is_lazy_and_transforms_result():
    artifact = DelegatedConditioning(prompt="cat", negative_prompt=None)
    downstream = CompletedInvocation.success(artifact)
    calls = []
    invocation = TransformingInvocation(
        downstream,
        lambda result: calls.append(result) or result,
    )

    assert calls == []
    assert invocation.result() is artifact
    assert calls == [artifact]


def test_transforming_invocation_propagates_cancel_without_observing_result():
    downstream = PendingInvocation()
    invocation = TransformingInvocation(downstream, lambda result: result)

    assert invocation.cancel() is True
    assert downstream.cancel_calls == 1
    assert downstream.result_calls == 0


def test_transforming_invocation_exception_includes_transform_failure():
    artifact = DelegatedConditioning(prompt="cat", negative_prompt=None)

    def fail_transform(result):
        del result
        raise ValueError("transform failed")

    invocation = TransformingInvocation(
        CompletedInvocation.success(artifact),
        fail_transform,
    )

    error = invocation.exception()
    assert isinstance(error, ValueError)
    assert str(error) == "transform failed"


def test_native_fallback_is_lazy_and_handles_primary_failure():
    artifact = DelegatedConditioning(prompt="cat", negative_prompt=None)
    native = RecordingNativeService(artifact)
    invocation = NativeFallbackInvocation(
        primary=CompletedInvocation.failure(RuntimeError("encode failed")),
        native_service=native,
        request=ConditioningRequest("cat", None),
        context=_context(),
        service_name="compel",
    )

    assert native.calls == 0
    assert invocation.result() is artifact
    assert native.calls == 1


def test_native_fallback_handles_asyncio_cancellation_without_outer_cancel():
    artifact = DelegatedConditioning(prompt="cat", negative_prompt=None)
    native = RecordingNativeService(artifact)
    invocation = NativeFallbackInvocation(
        primary=CompletedInvocation.failure(asyncio.CancelledError()),
        native_service=native,
        request=ConditioningRequest("cat", None),
        context=_context(),
        service_name="async-adapter",
    )

    assert invocation.result() is artifact
    assert native.calls == 1


def test_native_fallback_cancel_never_starts_native_service():
    primary = PendingInvocation()
    native = RecordingNativeService(DelegatedConditioning("cat", None))
    invocation = NativeFallbackInvocation(
        primary=primary,
        native_service=native,
        request=ConditioningRequest("cat", None),
        context=_context(),
        service_name="remote",
    )

    assert invocation.cancel() is True
    assert native.calls == 0
    assert primary.result_calls == 0


def test_native_fallback_exception_observes_final_success():
    native = RecordingNativeService(DelegatedConditioning("cat", None))
    invocation = NativeFallbackInvocation(
        primary=CompletedInvocation.failure(RuntimeError("encode failed")),
        native_service=native,
        request=ConditioningRequest("cat", None),
        context=_context(),
        service_name="compel",
    )

    assert invocation.exception() is None
    assert native.calls == 1
