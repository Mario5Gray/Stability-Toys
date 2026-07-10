from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias


@dataclass(frozen=True)
class ConditioningCompatibility:
    model_family: Literal["sd15", "sdxl"]
    encoder_identities: tuple[str, ...]
    hidden_dimensions: tuple[int, ...]
    pooled_required: bool
    dtype_name: str


@dataclass(frozen=True)
class DelegatedConditioning:
    prompt: str
    negative_prompt: str | None
    kind: Literal["delegated"] = "delegated"


@dataclass(frozen=True)
class MaterializedConditioning:
    slots: Mapping[str, object]
    compatibility: ConditioningCompatibility
    kind: Literal["materialized"] = "materialized"


ConditioningArtifact: TypeAlias = DelegatedConditioning | MaterializedConditioning
