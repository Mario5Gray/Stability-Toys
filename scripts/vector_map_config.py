"""Configuration model for st-vector-map (spec 5, 7; STABL-ascsgqha).

Precedence: built-in defaults, preset, recipe, explicit CLI. A later layer wins.
A layer value of None means "not supplied" and keeps the lower value.
A layer that chooses width_mm or height_mm replaces the lower dimension choice.

Recipe decoding checks the schema: field names, value types, and VTracer control names.
resolve() checks the merged values, including VTracer value ranges.

The canvas is a value. Derive it once at the entry and copy it into each stage.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import vector_map_vtracer

SCHEMA_VERSION = 1
INPUT_KINDS = ("mask", "edges", "image")
DIMENSIONS = ("width_mm", "height_mm")

# Later tasks that own the features the S2.2 walking skeleton rejects.
S23 = "S2.3 (STABL-vjpnctjh)"
S27 = "S2.7 (STABL-kfrksmnp)"
S31 = "S3.1 (STABL-memwrtos)"

# Recipe field -> value kind. These are also the settings fields.
_FIELD_KINDS = {
    "input": "path",
    "input_kind": "str",
    "width_mm": "number",
    "height_mm": "number",
    "line_width_mm": "number",
    "max_res": "int",
    "mask": "path",
    "invert": "bool",
    "alpha": "bool",
    "vtracer": "vtracer",
}
FIELDS = tuple(_FIELD_KINDS)

# Built-in defaults. resolve_options() adds the VTracer defaults.
DEFAULTS = {"invert": False, "alpha": False, "vtracer": {}}


class ConfigError(ValueError):
    """Invalid arguments or configuration. The CLI exits with code 2."""


def deferred(feature, owner):
    """Return the error for a feature that a later task implements."""
    return ConfigError(f"{feature} lands in {owner}. The S2.2 walking skeleton does not support it.")


@dataclass(frozen=True)
class Settings:
    input: Path
    input_kind: str
    width_mm: float | None
    height_mm: float | None
    line_width_mm: float | None
    max_res: int | None
    mask: Path | None
    invert: bool
    alpha: bool
    vtracer: MappingProxyType


@dataclass(frozen=True)
class Canvas:
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    mm_per_px: float


def physical_canvas(settings, width_px, height_px):
    """Convert processing pixels to millimetres once. The chosen dimension stays exact."""
    if settings.width_mm is not None:
        mm_per_px = settings.width_mm / width_px
        return Canvas(width_px, height_px, settings.width_mm, height_px * mm_per_px, mm_per_px)
    mm_per_px = settings.height_mm / height_px
    return Canvas(width_px, height_px, width_px * mm_per_px, settings.height_mm, mm_per_px)


def merge(layers):
    """Merge layers in order. None values keep the lower value. VTracer controls merge by name."""
    merged = {}
    for layer in layers:
        for name, value in layer.items():
            if name not in _FIELD_KINDS:
                raise ConfigError(f"Unknown setting {name!r}.")
            if value is None:
                continue
            merged[name] = {**merged.get("vtracer", {}), **value} if name == "vtracer" else value
        if any(layer.get(name) is not None for name in DIMENSIONS):
            for name in DIMENSIONS:
                merged[name] = layer.get(name)
    return merged


def resolve(*layers):
    """Merge the layers and check the merged values. Return frozen Settings."""
    merged = merge(layers)
    kind = merged.get("input_kind")
    if kind is None:
        raise ConfigError("input_kind is required. Use --input-kind mask, or the recipe field input_kind.")
    if kind not in INPUT_KINDS:
        raise ConfigError(f"input_kind must be one of {', '.join(INPUT_KINDS)}. Got {kind!r}.")
    if kind == "edges":
        raise deferred("Edge mode (input_kind edges)", S23)
    if kind == "image":
        raise deferred("Image mode (input_kind image)", S31)
    if merged.get("input") is None:
        raise ConfigError("A source image is required. Give SOURCE DESTINATION, or the recipe field input.")
    chosen = [name for name in DIMENSIONS if merged.get(name) is not None]
    if len(chosen) != 1:
        raise ConfigError("Give exactly one of width_mm and height_mm (--width-mm or --height-mm).")
    size = merged[chosen[0]]
    if not (math.isfinite(size) and size > 0):
        raise ConfigError(f"{chosen[0]} must be a positive finite number of millimetres. Got {size!r}.")
    for name, flag in (("max_res", "--max-res"), ("line_width_mm", "--line-width-mm"), ("mask", "--mask")):
        if merged.get(name) is not None:
            raise deferred(f"{name} ({flag})", S23)
    if merged.get("alpha"):
        raise deferred("Alpha selection (--alpha)", S23)
    try:
        vtracer = vector_map_vtracer.resolve_options(merged.get("vtracer", {}))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"vtracer: {exc}") from exc
    return Settings(
        input=Path(merged["input"]),
        input_kind=kind,
        width_mm=merged.get("width_mm"),
        height_mm=merged.get("height_mm"),
        line_width_mm=merged.get("line_width_mm"),
        max_res=merged.get("max_res"),
        mask=merged.get("mask"),
        invert=bool(merged.get("invert")),
        alpha=bool(merged.get("alpha")),
        vtracer=MappingProxyType(dict(vtracer)),
    )


def load_recipe(path):
    """Decode a schema_version 1 recipe. Resolve its paths relative to the recipe directory."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"Cannot read recipe {path}: {exc.strerror or exc}.") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Recipe {path} is not valid JSON: {exc}.") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Recipe {path} must be a JSON object.")
    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ConfigError(f"Recipe schema_version must be {SCHEMA_VERSION}. Got {version!r}.")
    unknown = sorted(set(data) - set(_FIELD_KINDS) - {"schema_version"})
    if unknown:
        raise ConfigError(f"Recipe has unknown fields: {', '.join(unknown)}.")
    return {
        name: _decode(name, value, path.parent)
        for name, value in data.items()
        if name != "schema_version"
    }


def _decode(name, value, base):
    kind = _FIELD_KINDS[name]
    if kind == "path" and isinstance(value, str):
        return base / value if not Path(value).is_absolute() else Path(value)
    if kind == "str" and isinstance(value, str):
        return value
    if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if kind == "int" and type(value) is int:
        return value
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "vtracer" and isinstance(value, dict):
        unsupported = sorted(set(value) - set(vector_map_vtracer.DEFAULT_OPTIONS))
        if unsupported:
            raise ConfigError(
                f"Recipe vtracer controls are not supported: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(vector_map_vtracer.DEFAULT_OPTIONS))}."
            )
        return dict(value)
    expected = {"path": "a string path", "str": "a string", "number": "a number", "int": "an integer",
                "bool": "true or false", "vtracer": "an object"}[kind]
    raise ConfigError(f"Recipe field {name} must be {expected}. Got {value!r}.")
