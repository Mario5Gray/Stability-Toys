from dataclasses import replace

import pytest
import torch

from backends.conditioning.artifacts import MaterializedConditioning
from backends.conditioning.compel_service import CompelConditioningService
from backends.conditioning.contracts import (
    ConditioningConfig,
    ConditioningRequest,
    ConditioningServiceRequirements,
    ModelContext,
    ModelContextDescriptor,
)
from backends.conditioning.registry import (
    ConditioningRegistry,
    build_conditioning_chain,
)


class FakeEncoderBundle:
    def __init__(
        self,
        *,
        tokenizers=("tokenizer-1",),
        encoders=("encoder-1",),
        dtype=torch.float32,
    ):
        self._tokenizers = tokenizers
        self._encoders = encoders
        self._dtype = dtype

    def tokenizers(self):
        return self._tokenizers

    def text_encoders(self):
        return self._encoders

    def live_dtype(self):
        return self._dtype

    def set_live_dtype(self, dtype):
        self._dtype = dtype


class CompelSpy:
    def __init__(self):
        self.instances = []
        self.prompts = []
        self.return_lengths = (77, 77)
        self.hidden_width = 768
        self.pooled_width = 1280

    def class_factory(self):
        spy = self

        class FakeCompel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                spy.instances.append(self)

            def __call__(self, prompt):
                index = len(spy.prompts)
                spy.prompts.append(prompt)
                length = spy.return_lengths[index % len(spy.return_lengths)]
                if isinstance(self.kwargs["tokenizer"], list):
                    embeds = torch.full((1, length, spy.hidden_width), index + 1.0)
                    pooled = torch.full((1, spy.pooled_width), index + 1.0)
                    return embeds, pooled
                return torch.full((1, length, spy.hidden_width), index + 1.0)

            def pad_conditioning_tensors_to_same_length(self, tensors):
                max_length = max(tensor.shape[1] for tensor in tensors)
                padded = []
                for tensor in tensors:
                    if tensor.shape[1] == max_length:
                        padded.append(tensor)
                        continue
                    pad_shape = (tensor.shape[0], max_length - tensor.shape[1], tensor.shape[2])
                    padded.append(torch.cat([tensor, torch.zeros(pad_shape)], dim=1))
                return padded

        return FakeCompel


class FakeReturnedEmbeddingsType:
    PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED = "penultimate"


@pytest.fixture
def compel_spy(monkeypatch):
    spy = CompelSpy()
    monkeypatch.setattr(
        "backends.conditioning.compel_service._load_compel",
        lambda: (spy.class_factory(), FakeReturnedEmbeddingsType),
    )
    return spy


@pytest.fixture
def sd15_context():
    return ModelContext(
        descriptor=ModelContextDescriptor(
            model_family="sd15",
            tokenizer_max_length=77,
            encoder_identities=("clip-l",),
            hidden_dimensions=(768,),
            pooled_required=False,
            encode_dtype_name="float32",
            device="cuda:0",
        ),
        local_encoder_bundle=FakeEncoderBundle(),
    )


@pytest.fixture
def sdxl_context():
    return ModelContext(
        descriptor=ModelContextDescriptor(
            model_family="sdxl",
            tokenizer_max_length=77,
            encoder_identities=("clip-l", "clip-g"),
            hidden_dimensions=(768, 1280),
            pooled_required=True,
            encode_dtype_name="float32",
            device="cuda:0",
        ),
        local_encoder_bundle=FakeEncoderBundle(
            tokenizers=("tokenizer-1", "tokenizer-2"),
            encoders=("encoder-1", "encoder-2"),
        ),
    )


@pytest.fixture
def context_for_family(sd15_context, sdxl_context):
    return {"sd15": sd15_context, "sdxl": sdxl_context}.__getitem__


def test_none_negative_prompt_is_encoded_as_empty_string(compel_spy, sd15_context):
    artifact = (
        CompelConditioningService().invoke(ConditioningRequest("cat", None), sd15_context).result()
    )

    assert compel_spy.prompts == ["cat", ""]
    assert set(artifact.slots) == {"prompt_embeds", "negative_prompt_embeds"}


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_prompt_and_negative_are_padded_to_same_sequence_length(
    family, context_for_family, compel_spy
):
    compel_spy.return_lengths = (154, 77)
    artifact = (
        CompelConditioningService()
        .invoke(ConditioningRequest("long prompt", "short"), context_for_family(family))
        .result()
    )

    assert artifact.slots["prompt_embeds"].shape[1] == 154
    assert artifact.slots["negative_prompt_embeds"].shape[1] == 154


def test_sdxl_materializes_pooled_pair(compel_spy, sdxl_context):
    artifact = (
        CompelConditioningService().invoke(ConditioningRequest("cat", "bad"), sdxl_context).result()
    )

    assert set(artifact.slots) == {
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    }
    assert compel_spy.instances[0].kwargs["requires_pooled"] == [False, True]
    assert (
        compel_spy.instances[0].kwargs["returned_embeddings_type"]
        == FakeReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
    )


def test_service_uses_live_bundle_dtype_not_snapshot_descriptor(sd15_context, compel_spy):
    del compel_spy
    stale_context = replace(
        sd15_context,
        descriptor=replace(sd15_context.descriptor, encode_dtype_name="float32"),
    )
    stale_context.local_encoder_bundle.set_live_dtype(torch.float16)

    artifact = (
        CompelConditioningService().invoke(ConditioningRequest("cat", None), stale_context).result()
    )

    assert artifact.compatibility.dtype_name == "float16"
    assert all(tensor.dtype == torch.float16 for tensor in artifact.slots.values())


def test_long_prompt_keeps_chunked_sequence_longer_than_clip_window(compel_spy, sd15_context):
    compel_spy.return_lengths = (154, 154)

    artifact = (
        CompelConditioningService()
        .invoke(ConditioningRequest(" ".join(["cat"] * 90), None), sd15_context)
        .result()
    )

    assert artifact.slots["prompt_embeds"].shape[1] > sd15_context.descriptor.tokenizer_max_length
    assert (
        artifact.slots["negative_prompt_embeds"].shape[1]
        > sd15_context.descriptor.tokenizer_max_length
    )


def test_short_unweighted_prompt_preserves_fake_encoder_output(compel_spy, sd15_context):
    del compel_spy
    artifact = (
        CompelConditioningService().invoke(ConditioningRequest("cat", None), sd15_context).result()
    )

    assert torch.allclose(artifact.slots["prompt_embeds"], torch.ones((1, 77, 768)))
    assert torch.allclose(artifact.slots["negative_prompt_embeds"], torch.full((1, 77, 768), 2.0))


def test_missing_compel_dependency_reports_clear_error(monkeypatch, sd15_context):
    def fail_import():
        raise RuntimeError(
            "Compel is not installed; install requirements-conditioning.txt "
            "or select the native conditioning service"
        )

    monkeypatch.setattr("backends.conditioning.compel_service._load_compel", fail_import)

    with pytest.raises(RuntimeError, match="Compel is not installed"):
        CompelConditioningService().invoke(ConditioningRequest("cat", None), sd15_context).result()


def test_builtin_registry_selects_compel_lazily(compel_spy, sd15_context):
    registry = ConditioningRegistry.with_builtins()
    chain = build_conditioning_chain(ConditioningConfig(service="compel"), sd15_context, registry)

    artifact = chain.invoke(ConditioningRequest("cat", None), sd15_context).result()

    assert isinstance(artifact, MaterializedConditioning)
    assert compel_spy.prompts == ["cat", ""]


def test_compel_requires_local_encoder_bundle():
    assert CompelConditioningService().requirements == ConditioningServiceRequirements(
        local_encoder_bundle=True
    )
