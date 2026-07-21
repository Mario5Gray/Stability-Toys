from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

from ..family_profiles import validate_family_id


@dataclass(frozen=True)
class ConditioningCompatibility:
    model_family: str
    encoder_identities: tuple[str, ...]
    hidden_dimensions: tuple[int, ...]
    pooled_required: bool
    dtype_name: str

    def __post_init__(self) -> None:
        validate_family_id(self.model_family)


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
