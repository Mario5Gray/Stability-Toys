"""VTracer adapter for st-vector-map (spec 6.2, STABL-lohtpfiy).

One upstream interface: vtracer==0.6.15 convert_raw_image_to_svg (Q1, STABL-orcwoxml).

The input is one prepared binary layer. White (True or 255) is material.
VTracer binary mode traces DARK pixels as material, so the adapter always inverts.
Do not infer polarity from the output: a border-touching layer traced without
inversion has no full-canvas path (S1.2, STABL-snyaxjef).

VTracer does not check enums or most ranges. A bad mode silently becomes spline.
VTracer ignores alpha. This module checks the layer and the options first.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO

import numpy as np
from PIL import Image

# Fixed upstream values. Not user controls. Every value goes upstream explicitly,
# because the 0.6.15 stub defaults are not reliable.
_FIXED_ARGS = {
    "img_format": "png",
    "colormode": "binary",
    "hierarchical": "stacked",
    "color_precision": 6,
    "layer_difference": 16,
    "corner_threshold": 60,
    "length_threshold": 4.0,
    "max_iterations": 10,
    "splice_threshold": 45,
    "path_precision": 8,
}

# Spline-only fitting controls. They have no measured effect in polygon mode.
_SPLINE_CONTROLS = frozenset(
    {"corner_threshold", "length_threshold", "splice_threshold", "path_precision", "max_iterations"}
)

# The upstream web app exposes filter_speckle as 0..128.
FILTER_SPECKLE_RANGE = (0, 128)
DEFAULT_OPTIONS = {"mode": "polygon", "filter_speckle": 4}

_SVG_PATH = "{http://www.w3.org/2000/svg}path"


class UnintendedBackgroundError(ValueError):
    """The trace has a full-canvas path that the input material does not explain."""


@dataclass
class TraceResult:
    svg: str
    upstream_args: dict
    vtracer_version: str


def trace_layer(layer, options=None):
    """Trace one binary layer. Return the upstream SVG and the resolved upstream values."""
    material = _material(layer)
    resolved = _resolve(options)
    try:
        import vtracer
    except ImportError as exc:
        raise RuntimeError("st-vector-map needs vtracer==0.6.15. Install the vector extra.") from exc

    args = {**_FIXED_ARGS, **resolved}
    svg = vtracer.convert_raw_image_to_svg(_inverted_png(material), **args)
    check_full_canvas(svg, material)
    return TraceResult(svg=svg, upstream_args=dict(args), vtracer_version=version("vtracer"))


def check_full_canvas(svg, material):
    """Reject a full-canvas path when the material does not cover the whole border."""
    if _covers_border(material):
        return
    height, width = material.shape
    for element in ET.fromstring(svg).iter(_SVG_PATH):
        dx, dy = _translate(element.get("transform", ""))
        for points in _subpaths(element.get("d", "")):
            xs = [x + dx for x, _ in points]
            ys = [y + dy for _, y in points]
            if (min(xs), min(ys), max(xs), max(ys)) == (0, 0, width, height):
                raise UnintendedBackgroundError(
                    "VTracer returned a path that covers the complete canvas, "
                    "but the layer material does not cover the canvas border."
                )


def _material(layer):
    if not isinstance(layer, np.ndarray):
        raise TypeError(f"Layer must be a numpy array, not {type(layer).__name__}.")
    if layer.ndim != 2:
        raise ValueError(
            f"Layer must be a 2-D binary array. Got shape {layer.shape}. "
            "Flatten alpha and colour before tracing."
        )
    if layer.size == 0:
        raise ValueError("Layer is empty.")
    if layer.dtype == bool:
        return layer
    if layer.dtype == np.uint8 and np.isin(layer, (0, 255)).all():
        return layer == 255
    raise ValueError("Layer must be binary: bool, or uint8 with values 0 and 255 only.")


def _resolve(options):
    resolved = dict(DEFAULT_OPTIONS)
    for name, value in (options or {}).items():
        if name in _SPLINE_CONTROLS:
            raise ValueError(f"{name} has no effect in polygon mode. Polygon mode exposes filter_speckle only.")
        if name not in DEFAULT_OPTIONS:
            raise ValueError(f"Option {name!r} is unknown. Allowed: {sorted(DEFAULT_OPTIONS)}.")
        resolved[name] = value
    if resolved["mode"] == "spline":
        raise ValueError(
            "mode 'spline' is blocked until it passes the hole, alignment and OpenSCAD cases."
        )
    if resolved["mode"] != "polygon":
        raise ValueError(f"mode must be 'polygon'. Got {resolved['mode']!r}.")
    low, high = FILTER_SPECKLE_RANGE
    speckle = resolved["filter_speckle"]
    if type(speckle) is not int or not low <= speckle <= high:
        raise ValueError(f"filter_speckle must be an int from {low} to {high}. Got {speckle!r}.")
    return resolved


def _inverted_png(material):
    """Encode material as dark pixels in a grayscale PNG with no alpha channel."""
    out = BytesIO()
    Image.fromarray(np.where(material, 0, 255).astype(np.uint8), mode="L").save(out, "PNG")
    return out.getvalue()


def _covers_border(material):
    return bool(material[0].all() and material[-1].all() and material[:, 0].all() and material[:, -1].all())


def _translate(transform):
    """Read translate(x,y). VTracer 0.6.15 writes no other transform."""
    if not transform:
        return 0.0, 0.0
    name, _, rest = transform.partition("(")
    if name.strip() != "translate" or not rest.endswith(")"):
        raise ValueError(f"Unexpected VTracer transform: {transform!r}")
    x, _, y = rest[:-1].partition(",")
    return float(x), float(y or 0)


def _subpaths(d):
    """Split polygon-mode path data ('M0,0 L4,0 ... Z') into point lists."""
    subpaths, points = [], []
    for token in d.split():
        command, coords = token[0], token[1:]
        if command == "Z":
            subpaths.append(points)
            points = []
        elif command in "ML":
            x, _, y = coords.partition(",")
            points.append((float(x), float(y)))
        else:
            raise ValueError(f"Unexpected polygon path command: {token!r}")
    if points:
        subpaths.append(points)
    return subpaths
