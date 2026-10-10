"""Prepare structure and detail candidate masks on the silhouette canvas. STABL-uifsadne, spec 6.1.

Plan: docs/superpowers/plans/2026-10-09-vector-map-s3-2.md.
Role requests are internal. S3.3 exposes them through the CLI and recipes.
Every candidate uses the S3.1 silhouette canvas. No stage derives another canvas.
"""

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

import canny_map
import vector_map_raster as raster
from vector_map_config import Canvas, ConfigError

ROLES = ("structure", "detail")
SOURCES = ("canny", "map")


@dataclass(frozen=True)
class CannySpec:
    low_threshold: int | None = None
    high_threshold: int | None = None
    blur: int | None = None


@dataclass(frozen=True)
class RoleRequest:
    role: Literal["structure", "detail"]
    source: Literal["canny", "map"] = "canny"
    path: Path | None = None
    canny: CannySpec | None = None
    width_mm: float | None = None
    gap_close_mm: float | None = None
    include_mask: Path | None = None
    exclude_mask: Path | None = None


@dataclass(frozen=True)
class PreparedCandidate:
    role: str
    source: str
    material: np.ndarray
    canny: CannySpec | None
    expansion: raster.BandExpansion | None
    diagnostics: tuple[dict, ...]


@dataclass(frozen=True)
class PreparedLayers:
    silhouette: raster.PreparedRaster
    structure: PreparedCandidate | None = None
    detail: PreparedCandidate | None = None

    @property
    def canvas(self) -> Canvas:
        return self.silhouette.canvas

    @property
    def roles(self):
        """Selected roles in fixed output order. The silhouette is always selected."""
        return ("silhouette", *(role for role in ROLES if getattr(self, role) is not None))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_canny(role, spec):
    low, high, blur = spec.low_threshold, spec.high_threshold, spec.blur
    if not (_is_int(low) and _is_int(high) and 0 <= low < high):
        raise ConfigError(f"{role}: Canny thresholds must be integers with 0 <= low_threshold < high_threshold.")
    if not _is_int(blur) or blur < 0 or (blur and blur % 2 == 0):
        raise ConfigError(f"{role}: Canny blur must be 0 or a positive odd integer.")


def _check_length(role, name, value):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{role}: {name} must be a positive finite number.")


def _validate(requests):
    """Check each request. Return requests keyed by role."""
    by_role = {}
    for request in requests:
        role = request.role
        if role not in ROLES:
            raise ConfigError(f"Unknown optional role {role!r}. Expected one of {ROLES}.")
        if role in by_role:
            raise ConfigError(f"{role}: duplicate role request.")
        if request.source not in SOURCES:
            raise ConfigError(f"{role}: unknown source {request.source!r}. Expected one of {SOURCES}.")
        if request.source == "map":
            if request.path is None:
                raise ConfigError(f"{role}: a map source requires a path.")
            if request.canny is not None:
                raise ConfigError(f"{role}: a map source rejects canny values.")
        elif request.path is not None:
            raise ConfigError(f"{role}: a canny source rejects a map path.")
        _check_length(role, "width_mm", request.width_mm)
        _check_length(role, "gap_close_mm", request.gap_close_mm)
        by_role[role] = request
    return by_role


def _resolve_canny(request):
    """Fill omitted Canny values from the role defaults. Validate the result."""
    explicit = request.canny or CannySpec()
    default = DEFAULT_CANNY.get(request.role, CannySpec())
    spec = CannySpec(*(default_value if value is None else value for value, default_value in (
        (explicit.low_threshold, default.low_threshold),
        (explicit.high_threshold, default.high_threshold),
        (explicit.blur, default.blur),
    )))
    _check_canny(request.role, spec)
    return spec


def source_rgb(path, oriented_size, processed_size, *, data=None):
    """Orient, composite transparency onto opaque black, convert to RGB, then LANCZOS-resize.

    Do not use canny_map.load_image. It has different orientation and resize rules.
    """
    with Image.open(path if data is None else BytesIO(data)) as source:
        with ImageOps.exif_transpose(source) as oriented:
            if oriented.size != tuple(oriented_size):
                raise ValueError(f"{path}: oriented size {oriented.size} differs from prepared {tuple(oriented_size)}.")
            with oriented.convert("RGBA") as rgba:
                with Image.new("RGBA", rgba.size, (0, 0, 0, 255)) as black:
                    with Image.alpha_composite(black, rgba) as composite, composite.convert("RGB") as rgb:
                        with rgb.resize(tuple(processed_size), resample=Image.Resampling.LANCZOS) as resized:
                            return np.asarray(resized).copy()


def canny_material(rgb, spec):
    """Run canny_map.canny_edges once on prepared pixels. White edge pixels are material."""
    with Image.fromarray(rgb, "RGB") as image:
        edges = canny_map.canny_edges(image, low_threshold=spec.low_threshold, high_threshold=spec.high_threshold,
                                      blur=spec.blur, invert=False)
        with edges:
            return np.asarray(edges) >= raster.FOREGROUND_LUMINANCE


def prepare_layers(settings, requests, *, input_bytes=None):
    """Prepare the silhouette and selected optional roles on one canvas."""
    if settings.input_kind != "image":
        raise ConfigError("Optional roles require image mode.")
    by_role = _validate(requests)
    specs = {role: _resolve_canny(request) for role, request in by_role.items() if request.source == "canny"}
    input_bytes = input_bytes if input_bytes is not None else {}
    silhouette = raster.prepare(settings, input_bytes=input_bytes)
    rgb = None
    if specs:
        rgb = source_rgb(settings.input, silhouette.oriented_size, silhouette.processed_size,
                         data=input_bytes.get(settings.input))
    candidates = {}
    for role, request in by_role.items():
        material = canny_material(rgb, specs[role]) & silhouette.material
        material.setflags(write=False)
        candidates[role] = PreparedCandidate(role, request.source, material, specs.get(role), None, ())
    return PreparedLayers(silhouette, **candidates)


# Measured on the reviewed corpus: docs/superpowers/reports/2026-10-09-vector-map-s3-2-canny.md.
# Equal blur makes detail a superset of structure, so structure removal leaves only weaker edges.
DEFAULT_CANNY = {
    "structure": CannySpec(low_threshold=100, high_threshold=200, blur=3),
    "detail": CannySpec(low_threshold=50, high_threshold=100, blur=3),
}
