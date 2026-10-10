"""Configuration model for st-vector-map (spec 5, 7; STABL-ascsgqha).

Precedence: built-in defaults, preset, recipe, explicit CLI. A later layer wins.
A layer value of None means "not supplied" and keeps the lower value.
A layer that chooses width_mm or height_mm replaces the lower dimension choice.

Recipe decoding checks the schema: field names, value types, and VTracer control names.
resolve() checks the merged values, including VTracer value ranges.

Image mode (S3.1, STABL-memwrtos) selects one silhouette method: mask, alpha, or threshold.
The three fields form one precedence group. A later layer that selects a method replaces it.

The canvas is a value. Derive it once at the entry and copy it into each stage.

Layered image mode (S3.3, STABL-qlagdbmh) selects roles with layers and configures optional roles with
the structure and detail objects. A role object merges by field. A later layer that sets source drops
the lower configuration of the other source. A role setting outside the selection is an error.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import vector_map_vtracer
from vector_map_svg import SvgLimits

SCHEMA_VERSION = 1
INPUT_KINDS = ("mask", "edges", "image")
DIMENSIONS = ("width_mm", "height_mm")
SILHOUETTE_METHODS = ("mask", "alpha", "threshold")

# Recipe field -> value kind. These are also the settings fields.
_FIELD_KINDS = {
    "input": "path",
    "input_kind": "str",
    "width_mm": "number",
    "height_mm": "number",
    "line_width_mm": "number",
    "max_res": "int",
    "mask": "path",
    "include_mask": "path",
    "exclude_mask": "path",
    "invert": "bool",
    "alpha": "bool",
    "threshold": "int",
    "vtracer": "vtracer",
    "max_svg_bytes": "int",
    "max_paths": "int",
    "max_path_commands": "int",
    "layers": "layers",
    "structure": "role",
    "detail": "role",
}
SVG_LIMITS = ("max_svg_bytes", "max_paths", "max_path_commands")
FIELDS = tuple(_FIELD_KINDS)
LAYER_ROLES = ("silhouette", "structure", "detail")
OPTIONAL_ROLES = LAYER_ROLES[1:]
# Role object field -> value kind. A role request takes these names.
_ROLE_FIELD_KINDS = {
    "source": "str",
    "path": "path",
    "canny": "canny",
    "width_mm": "number",
    "gap_close_mm": "number",
    "include_mask": "path",
    "exclude_mask": "path",
}
CANNY_FIELDS = ("low_threshold", "high_threshold", "blur")

# Built-in defaults. resolve_options() adds the VTracer defaults.
# SVG limits are wrapper policy (S2.5, STABL-npoznayt). They never go to VTracer.
DEFAULTS = {
    "invert": False, "alpha": False, "vtracer": {},
    **{name: getattr(SvgLimits(), name) for name in SVG_LIMITS},
}


class ConfigError(ValueError):
    """Invalid arguments or configuration. The CLI exits with code 2."""


@dataclass(frozen=True)
class Settings:
    input: Path
    input_kind: str
    width_mm: float | None
    height_mm: float | None
    line_width_mm: float | None
    max_res: int | None
    mask: Path | None
    include_mask: Path | None
    exclude_mask: Path | None
    invert: bool
    alpha: bool
    vtracer: MappingProxyType
    svg_limits: SvgLimits
    threshold: int | None = None
    layers: tuple[str, ...] | None = None
    role_requests: tuple = ()

    @property
    def silhouette_method(self):
        """Return the image-mode method, or None in standalone modes."""
        if self.input_kind != "image":
            return None
        return "mask" if self.mask is not None else "threshold" if self.threshold is not None else "alpha"


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
        canvas = Canvas(width_px, height_px, settings.width_mm, height_px * mm_per_px, mm_per_px)
    else:
        mm_per_px = settings.height_mm / height_px
        canvas = Canvas(width_px, height_px, width_px * mm_per_px, settings.height_mm, mm_per_px)
    if not all(math.isfinite(value) and value > 0 for value in (canvas.width_mm, canvas.height_mm, mm_per_px)):
        raise ConfigError("Physical scale must remain positive and finite at processing resolution.")
    return canvas


def merge(layers):
    """Merge layers in order. None values keep the lower value. VTracer controls merge by name."""
    merged = {}
    for layer in layers:
        for name, value in layer.items():
            if name not in _FIELD_KINDS:
                raise ConfigError(f"Unknown setting {name!r}.")
            if value is None:
                continue
            if name == "vtracer":
                merged[name] = {**merged.get("vtracer", {}), **value}
            elif name in OPTIONAL_ROLES:
                merged[name] = _merge_role(merged.get(name), value)
            else:
                merged[name] = value
        if any(layer.get(name) is not None for name in DIMENSIONS):
            for name in DIMENSIONS:
                merged[name] = layer.get(name)
    return merged


def _merge_role(lower, upper):
    """Merge role fields. A source choice drops the lower configuration of the other source."""
    merged = dict(lower or {})
    source = upper.get("source")
    if source == "map":
        merged.pop("canny", None)
    elif source == "canny":
        merged.pop("path", None)
    merged.update(upper)
    return merged


def _selection(value):
    """Check a layers selection. Return the roles in fixed output order."""
    if isinstance(value, str):
        names = value.split(",") if value else []
    else:
        names = list(value)
    if not names:
        raise ConfigError("layers must name at least one role. Include silhouette.")
    for name in names:
        if name == "":
            raise ConfigError("layers has an empty role name. Separate role names with one comma.")
        if name not in LAYER_ROLES:
            raise ConfigError(f"Unknown layer {name!r}. Use {', '.join(LAYER_ROLES)}.")
        if names.count(name) > 1:
            raise ConfigError(f"Layer {name} is selected more than once. Name each role once.")
    if "silhouette" not in names:
        raise ConfigError("layers must include silhouette. Structure and detail are clipped to it.")
    return tuple(role for role in LAYER_ROLES if role in names)


def _role_requests(merged, selection):
    """Admit role settings only for selected roles. Return checked requests in fixed role order."""
    for role in OPTIONAL_ROLES:
        if merged.get(role) is not None and (selection is None or role not in selection):
            raise ConfigError(
                f"{role} settings were given, but {role} is not selected. "
                f"Add {role} to --layers or the recipe field layers."
            )
    if selection is None:
        return ()
    # Local import: vector_map_layers imports this module.
    import vector_map_layers as layers

    requests = []
    for role in OPTIONAL_ROLES:
        if role not in selection:
            continue
        fields = dict(merged.get(role) or {})
        canny = fields.pop("canny", None)
        requests.append(layers.RoleRequest(
            role, canny=None if canny is None else layers.CannySpec(**canny), **fields,
        ))
    layers.check_requests(requests)
    return tuple(requests)


def _silhouette_method(layers):
    """Apply group precedence to mask, alpha, and threshold. Return the selected method or None.

    A layer that selects one method replaces the earlier method. alpha=False clears only alpha.
    """
    method = None
    for layer in layers:
        chosen = [name for name in SILHOUETTE_METHODS
                  if (layer.get(name) is True if name == "alpha" else layer.get(name) is not None)]
        if len(chosen) > 1:
            raise ConfigError(
                f"Give one silhouette method per source. One source selects {' and '.join(chosen)}."
            )
        if chosen:
            method = chosen[0]
        elif layer.get("alpha") is False and method == "alpha":
            method = None
    return method


def _check_threshold(value):
    if type(value) is not int or not 0 <= value <= 255:
        raise ConfigError(f"threshold must be an integer from 0 to 255. Got {value!r}.")


def resolve(*layers):
    """Merge the layers and check the merged values. Return frozen Settings."""
    merged = merge(layers)
    kind = merged.get("input_kind")
    if kind is None:
        raise ConfigError("input_kind is required. Use --input-kind mask, or the recipe field input_kind.")
    if kind not in INPUT_KINDS:
        raise ConfigError(f"input_kind must be one of {', '.join(INPUT_KINDS)}. Got {kind!r}.")
    if kind == "image":
        method = _silhouette_method(layers)
        if method is None:
            raise ConfigError(
                "Image mode needs one silhouette method: --mask PATH (white material), --alpha "
                "(source alpha >= 128), or --threshold N (source luminance >= N, 0 to 255). "
                "Add --invert to reverse the selection. Learned segmentation is not supported."
            )
        for name in SILHOUETTE_METHODS:
            if name != method:
                merged[name] = False if name == "alpha" else None
        if method == "threshold":
            _check_threshold(merged["threshold"])
        selection = None if merged.get("layers") is None else _selection(merged["layers"])
        role_requests = _role_requests(merged, selection)
    else:
        if merged.get("layers") is not None or any(merged.get(role) is not None for role in OPTIONAL_ROLES):
            raise ConfigError("--layers and role settings (structure, detail) require input_kind image.")
        selection, role_requests = None, ()
        for name in ("mask", "threshold"):
            if merged.get(name) is not None:
                raise ConfigError(f"{name} selects an image silhouette and requires input_kind image.")
    if merged.get("input") is None:
        raise ConfigError("A source image is required. Give SOURCE DESTINATION, or the recipe field input.")
    chosen = [name for name in DIMENSIONS if merged.get(name) is not None]
    if len(chosen) != 1:
        raise ConfigError("Give exactly one of width_mm and height_mm (--width-mm or --height-mm).")
    size = merged[chosen[0]]
    if not (math.isfinite(size) and size > 0):
        raise ConfigError(f"{chosen[0]} must be a positive finite number of millimetres. Got {size!r}.")
    max_res = merged.get("max_res")
    if max_res is not None and (type(max_res) is not int or max_res <= 0):
        raise ConfigError("max_res must be a positive integer pixel count.")
    line_width = merged.get("line_width_mm")
    if line_width is not None:
        if kind != "edges":
            raise ConfigError("line_width_mm requires input_kind edges.")
        if not (math.isfinite(line_width) and line_width > 0):
            raise ConfigError("line_width_mm must be positive and finite.")
    try:
        vtracer = vector_map_vtracer.resolve_options(merged.get("vtracer", {}))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"vtracer: {exc}") from exc
    defaults = SvgLimits()
    try:
        svg_limits = SvgLimits(**{name: merged.get(name, getattr(defaults, name)) for name in SVG_LIMITS})
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return Settings(
        input=Path(merged["input"]),
        input_kind=kind,
        width_mm=merged.get("width_mm"),
        height_mm=merged.get("height_mm"),
        line_width_mm=merged.get("line_width_mm"),
        max_res=merged.get("max_res"),
        mask=merged.get("mask"),
        include_mask=merged.get("include_mask"),
        exclude_mask=merged.get("exclude_mask"),
        invert=bool(merged.get("invert")),
        alpha=bool(merged.get("alpha")),
        vtracer=MappingProxyType(dict(vtracer)),
        svg_limits=svg_limits,
        threshold=merged.get("threshold"),
        layers=selection,
        role_requests=role_requests,
    )


def load_recipe(path, *, snapshots=None):
    """Decode a schema_version 1 recipe. Resolve its paths relative to the recipe directory."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read recipe {path}: {exc.strerror or exc}.") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Recipe {path} is not valid UTF-8: {exc}.") from exc
    if snapshots is not None:
        snapshots[path] = raw
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


def _decode(name, value, base, *, kind=None):
    kind = kind or _FIELD_KINDS[name]
    if kind == "layers" and isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    if kind == "role" and isinstance(value, dict):
        unknown = sorted(set(value) - set(_ROLE_FIELD_KINDS))
        if unknown:
            raise ConfigError(f"Recipe field {name} has unknown fields: {', '.join(unknown)}.")
        return {field: _decode(f"{name}.{field}", item, base, kind=_ROLE_FIELD_KINDS[field])
                for field, item in value.items()}
    if kind == "canny" and isinstance(value, dict):
        unknown = sorted(set(value) - set(CANNY_FIELDS))
        if unknown:
            raise ConfigError(f"Recipe field {name} has unknown fields: {', '.join(unknown)}. "
                              f"Supported: {', '.join(CANNY_FIELDS)}.")
        return {field: _decode(f"{name}.{field}", item, base, kind="int") for field, item in value.items()}
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
                "bool": "true or false", "vtracer": "an object", "layers": "a list of role names",
                "role": "an object", "canny": "an object"}[kind]
    raise ConfigError(f"Recipe field {name} must be {expected}. Got {value!r}.")
