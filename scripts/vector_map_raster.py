"""Prepare binary rasters on one physical canvas. STABL-vjpnctjh, spec 5 and 6.1."""

import math
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from vector_map_config import Canvas, ConfigError, physical_canvas
from vector_map_diagnostics import feature_diagnostics

FOREGROUND_LUMINANCE = 128


@dataclass(frozen=True)
class BandExpansion:
    requested_width_mm: float
    radius_px: int
    kernel_size_px: int
    achieved_width_mm: float
    expansion_mm: float


@dataclass(frozen=True)
class PreparedRaster:
    material: np.ndarray
    canvas: Canvas
    original_size: tuple[int, int]
    oriented_size: tuple[int, int]
    processed_size: tuple[int, int]
    expansion: BandExpansion | None
    diagnostics: tuple[dict, ...]


def _decode(path, *, alpha=False, invert=False, constraint=False):
    """Decode and orient before thresholding. Return material, source dimensions, and warnings."""
    diagnostics = []
    try:
        with Image.open(path) as source:
            if source.format not in ("PNG", "JPEG"):
                raise ConfigError(f"{path}: expected PNG or JPEG, got {source.format}.")
            if source.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                raise ConfigError(
                    f"Unsupported pixel mode {source.mode!r} in {path}. "
                    "Convert to 8-bit L/RGB, or RGBA PNG to keep alpha."
                )
            original_size = source.size
            image = ImageOps.exif_transpose(source)
            try:
                has_alpha = "A" in image.getbands() or "transparency" in image.info
                if alpha and not has_alpha:
                    raise ConfigError(f"{path}: --alpha requires an alpha channel or palette transparency.")
                if has_alpha and not alpha:
                    message = (
                        "constraint has alpha, luminance used. Source --alpha does not select constraint alpha."
                        if constraint else "source has alpha, luminance used, pass --alpha to select it."
                    )
                    diagnostics.append({
                        "level": "warning", "code": "alpha_ignored",
                        "message": f"{path}: {message}",
                    })
                channel = image.convert("RGBA").getchannel("A") if alpha else image.convert("L")
                material = np.asarray(channel) >= FOREGROUND_LUMINANCE
                channel.close()
            finally:
                image.close()
    except Image.DecompressionBombError as exc:
        raise ValueError(f"{path} exceeds the Pillow pixel limit: {exc}") from exc
    return (~material if invert else material), original_size, diagnostics


def prepare_mask(path, *, invert):
    """Return oriented luminance material. Report ignored alpha through Python warnings."""
    material, _, diagnostics = _decode(path, invert=invert)
    for diagnostic in diagnostics:
        warnings.warn(diagnostic["message"], UserWarning, stacklevel=2)
    return material


def _processing_size(size, max_res):
    """Limit longest side. Round shorter side half upward without floating-point arithmetic."""
    longest = max(size)
    if max_res is None or longest <= max_res:
        return size
    return tuple(max(1, (2 * side * max_res + longest) // (2 * longest)) for side in size)


def _resize(material, size):
    if material.shape[::-1] == size:
        return material
    with Image.fromarray(material) as image:
        with image.resize(size, resample=Image.Resampling.NEAREST) as resized:
            return np.asarray(resized).copy()


def _constrain(material, include, exclude):
    if include is not None:
        material = material & include
    if exclude is not None:
        material = material & ~exclude
    return material


def _expand(material, requested, scale):
    """Expand symmetrically to odd nominal width. Wider input bands remain wider."""
    pixels = requested / scale
    if not math.isfinite(pixels) or pixels > np.iinfo(np.int32).max:
        raise ConfigError("Requested line width exceeds supported kernel dimensions.")
    # Suppress float noise at exact odd widths before rounding upward.
    radius = max(0, math.ceil((pixels - 1) / 2 - 1e-9))
    side = 2 * radius + 1
    achieved = side * scale
    if not math.isfinite(achieved):
        raise ConfigError("Achieved line width must remain finite.")
    expansion = BandExpansion(requested, radius, side, achieved, radius * scale)
    # Beyond canvas extent, extra kernel reach cannot add material.
    rx = min(radius, material.shape[1] - 1)
    ry = min(radius, material.shape[0] - 1)
    expanded = cv2.dilate(material.astype(np.uint8), np.ones((1, 2 * rx + 1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    expanded = cv2.dilate(expanded, np.ones((2 * ry + 1, 1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return expanded.astype(bool), expansion


def prepare(settings):
    """Prepare one layer and retain complete source canvas. Diagnostics never repair geometry."""
    material, original_size, diagnostics = _decode(settings.input, alpha=settings.alpha, invert=settings.invert)
    oriented_size = material.shape[::-1]
    processed_size = _processing_size(oriented_size, settings.max_res)
    constraints = []
    for path in (settings.include_mask, settings.exclude_mask):
        if path is None:
            constraints.append(None)
            continue
        constraint, _, notices = _decode(path, constraint=True)
        diagnostics.extend(notices)
        if constraint.shape[::-1] != oriented_size:
            raise ConfigError(
                f"{path}: oriented dimensions {constraint.shape[::-1]} differ from source dimensions {oriented_size}."
            )
        constraints.append(_resize(constraint, processed_size))
    material = _resize(material, processed_size)
    canvas = physical_canvas(settings, *processed_size)
    material = _constrain(material, *constraints)
    expansion = None
    if settings.line_width_mm is not None:
        material, expansion = _expand(material, settings.line_width_mm, canvas.mm_per_px)
        if settings.line_width_mm / canvas.mm_per_px < 4:
            diagnostics.append({
                "level": "warning", "code": "requested_width_below_four_pixels",
                "message": "Requested line width covers fewer than four processing pixels.",
            })
        material = _constrain(material, *constraints)
    diagnostics.extend(feature_diagnostics(material))
    material.setflags(write=False)
    return PreparedRaster(material, canvas, original_size, oriented_size, processed_size, expansion, tuple(diagnostics))
