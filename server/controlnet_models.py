from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ControlNetPreprocessRequest(BaseModel):
    id: str = Field(..., min_length=1)
    options: Dict[str, Any] = Field(default_factory=dict)


class ControlNetAttachment(BaseModel):
    attachment_id: str = Field(..., min_length=1)
    control_type: str = Field(..., min_length=1)
    model_id: Optional[str] = Field(default=None)
    map_asset_ref: Optional[str] = Field(default=None)
    source_asset_ref: Optional[str] = Field(default=None)
    preprocess: Optional[ControlNetPreprocessRequest] = Field(default=None)
    strength: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    start_percent: float = Field(default=0.0, ge=0.0, le=1.0)
    end_percent: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_input_path(self) -> "ControlNetAttachment":
        has_map = self.map_asset_ref is not None
        has_source = self.source_asset_ref is not None
        if not has_map and not has_source:
            raise ValueError("attachment must supply map_asset_ref or source_asset_ref")
        if has_map and has_source:
            raise ValueError("attachment must supply exactly one of map_asset_ref or source_asset_ref")
        if has_source and self.preprocess is None:
            raise ValueError("source_asset_ref requires a preprocess block")
        if has_map and self.preprocess is not None:
            raise ValueError("map_asset_ref is incompatible with a preprocess block")
        if self.start_percent > self.end_percent:
            raise ValueError("start_percent must be <= end_percent")
        return self
