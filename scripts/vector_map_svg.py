"""SVG inspection and sizing for st-vector-map (spec 6.3; STABL-ascsgqha, STABL-npoznayt).

The accepted dialect is the pinned vtracer==0.6.15 polygon output, and nothing more:
one svg root, g and path elements, black nonzero fill, translate transforms,
and closed M/L/Z path data. Unsupported content fails. It is never removed or rewritten.

OpenSCAD imports the mm root size plus pixel viewBox exactly and independent of dpi
(S1.3, STABL-nygbbrrn).
Parse with an XML parser only. Never use regular expressions.
This module must not import configuration or the adapter at runtime.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

SVG_NS = "http://www.w3.org/2000/svg"
_SVG = f"{{{SVG_NS}}}svg"
_G = f"{{{SVG_NS}}}g"
_PATH = f"{{{SVG_NS}}}path"

_ATTRIBUTES = {
    _SVG: frozenset({"version", "width", "height", "viewBox"}),
    _G: frozenset({"id", "transform", "fill", "fill-rule"}),
    _PATH: frozenset({"id", "d", "transform", "fill", "fill-rule"}),
}
_BLACK = frozenset({"black", "#000", "#000000"})
_NUMBER_CHARS = frozenset("0123456789+-.eE")

_SUGGESTION = (
    "Lower --max-res, or adjust recipe vtracer.filter_speckle within 0..128. "
    "Both can change the geometry. This tool does not change them for you."
)
_DEGENERATE_ONLY = (
    "VTracer returned {count} paths containing only points or lines. "
    "Each path needs one polygon with three non-collinear points. "
    "A higher --max-res can help only when processing resolution can increase. "
    "A higher filter_speckle removes small islands on purpose."
)

ET.register_namespace("", SVG_NS)


class SvgInspectionError(ValueError):
    """Traced SVG is malformed, unsupported, or over a configured limit."""


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {value!r}.")


@dataclass(frozen=True)
class SvgLimits:
    """Wrapper policy, not measured VTracer or OpenSCAD limits. Counts are per layer."""

    max_svg_bytes: int = 20_971_520
    max_paths: int = 10_000
    max_path_commands: int = 1_000_000

    def __post_init__(self):
        for name in ("max_svg_bytes", "max_paths", "max_path_commands"):
            _positive_int(name, getattr(self, name))


@dataclass(frozen=True)
class SvgMetrics:
    raw_bytes: int
    normalized_bytes: int | None
    paths: int
    commands: int
    subpaths: int
    degenerate_subpaths: int


@dataclass(frozen=True)
class SvgDocument:
    """Inspection result. subpaths hold canvas-pixel points after accumulated translations."""

    metrics: SvgMetrics
    subpaths: tuple
    _root: ET.Element = field(repr=False, compare=False)


def inspect_svg(svg, *, width_px, height_px, limits=None, allow_empty=False):
    """Check one traced layer against the polygon dialect and the limits. Change nothing."""
    limits = limits or SvgLimits()
    raw_bytes = _check_bytes(svg, limits, "SVG")
    root = _parse(svg)
    _check_root(root, width_px, height_px)
    state = _Walk(limits)
    for child in root:
        state.visit(child, (0.0, 0.0), None)
    if state.degenerate_only:
        raise SvgInspectionError(_DEGENERATE_ONLY.format(count=state.degenerate_only))
    if state.paths == 0 and not allow_empty:
        raise SvgInspectionError("The traced SVG contains no paths.")
    metrics = SvgMetrics(
        raw_bytes=raw_bytes,
        normalized_bytes=None,
        paths=state.paths,
        commands=state.commands,
        subpaths=len(state.subpaths),
        degenerate_subpaths=state.degenerate,
    )
    return SvgDocument(metrics=metrics, subpaths=tuple(state.subpaths), _root=root)


def _check_bytes(svg, limits, label):
    size = len(svg.encode("utf-8"))
    if size > limits.max_svg_bytes:
        raise SvgInspectionError(
            f"{label} size {size} bytes exceeds limit {limits.max_svg_bytes} (--max-svg-bytes). {_SUGGESTION}"
        )
    return size


class _Builder(ET.TreeBuilder):
    """Reject declarations through parser callbacks, before they can affect interpretation."""

    def doctype(self, *_declaration):
        raise SvgInspectionError("The traced SVG has a DTD (DOCTYPE) declaration. DTDs are not supported.")

    def pi(self, target, *_text):
        raise SvgInspectionError(f"The traced SVG has a processing instruction ({target}). It is not supported.")


def _parse(svg):
    parser = ET.XMLParser(target=_Builder())
    try:
        parser.feed(svg)
        return parser.close()
    except ET.ParseError as exc:
        raise SvgInspectionError(f"VTracer returned malformed SVG: {exc}.") from exc


def _check_root(root, width_px, height_px):
    if root.tag != _SVG:
        raise SvgInspectionError(f"The SVG root must be svg in namespace {SVG_NS}. Got {root.tag!r}.")
    _check_element(root)
    version = root.get("version")
    if version not in (None, "1.1"):
        raise SvgInspectionError(f"SVG version {version!r} is not supported. Use 1.1.")
    for name, expected in (("width", width_px), ("height", height_px)):
        value = root.get(name)
        size = _number(value[:-2] if value and value.endswith("px") else value, f"SVG {name}", SvgInspectionError)
        if size != expected:
            raise SvgInspectionError(f"The traced SVG {name} is {value!r}, but the canvas is {expected} px.")
    view_box = root.get("viewBox")
    if view_box is not None:
        fields = view_box.replace(",", " ").split()
        try:
            numbers = [_number(value, "viewBox", SvgInspectionError) for value in fields]
        except SvgInspectionError:
            numbers = None
        if numbers != [0, 0, width_px, height_px]:
            raise SvgInspectionError(f"viewBox {view_box!r} must be '0 0 {width_px} {height_px}'.")


def _check_element(element):
    allowed = _ATTRIBUTES[element.tag]
    for name in element.attrib:
        if name not in allowed:
            raise SvgInspectionError(f"Unsupported attribute {name!r} on {_local(element.tag)}.")
    for label, text in (("text", element.text), ("tail text", element.tail)):
        if text and text.strip():
            raise SvgInspectionError(f"Unexpected {label} {text.strip()[:20]!r} near {_local(element.tag)}.")


def _local(tag):
    return tag.rpartition("}")[2]


class _Walk:
    """Depth-first walk. Counts are checked against limits as they grow."""

    def __init__(self, limits):
        self.limits = limits
        self.paths = 0
        self.commands = 0
        self.degenerate = 0
        self.degenerate_only = 0
        self.subpaths = []

    def visit(self, element, offset, fill):
        if element.tag not in (_G, _PATH):
            raise SvgInspectionError(f"unsupported element {_local(element.tag)}. Only g and path are accepted.")
        _check_element(element)
        offset = _add(offset, _translate(element.get("transform")))
        fill = _fill(element, fill)
        if element.tag == _G:
            for child in element:
                self.visit(child, offset, fill)
            return
        if len(element):
            raise SvgInspectionError("A path must be a leaf element. It has child elements.")
        self.paths += 1
        if self.paths > self.limits.max_paths:
            raise SvgInspectionError(
                f"path count at least {self.paths} exceeds limit {self.limits.max_paths} (--max-paths). {_SUGGESTION}"
            )
        self.path(element.get("d"), offset)

    def path(self, d, offset):
        polygons = 0
        for points in self._subpaths(d):
            translated = tuple(_add(offset, point) for point in points)
            if not all(math.isfinite(value) for point in translated for value in point):
                raise SvgInspectionError("A translated path point is not finite.")
            distinct = set(points)
            if len(distinct) < 3:
                self.degenerate += 1
            elif _collinear(points):
                raise SvgInspectionError(f"A subpath with {len(distinct)} distinct points is collinear.")
            else:
                polygons += 1
            self.subpaths.append(translated)
        if polygons == 0:
            self.degenerate_only += 1

    def _subpaths(self, d):
        if not d or not d.strip():
            raise SvgInspectionError("A path has empty path data.")
        points = None
        # XML attribute normalization turns tab, CR and LF into spaces. Other whitespace stays in a token.
        for token in filter(None, d.split(" ")):
            self.commands += 1
            if self.commands > self.limits.max_path_commands:
                raise SvgInspectionError(
                    f"path command count at least {self.commands} exceeds limit "
                    f"{self.limits.max_path_commands} (--max-path-commands). {_SUGGESTION}"
                )
            command, coords = token[0], token[1:]
            if command == "Z":
                if coords:
                    raise SvgInspectionError(f"Z must stand alone. Got {token!r}.")
                if points is None:
                    raise SvgInspectionError("Z has no open subpath to close.")
                yield points
                points = None
            elif command == "M":
                if points is not None:
                    raise SvgInspectionError("A subpath is not closed with Z before the next M.")
                points = [_pair(coords)]
            elif command == "L":
                if points is None:
                    raise SvgInspectionError("L comes before M.")
                points.append(_pair(coords))
            else:
                raise SvgInspectionError(f"unsupported path command in {token!r}. Only M, L and Z are accepted.")
        if points is not None:
            raise SvgInspectionError("A subpath is not closed with Z.")


def _pair(coords):
    fields = coords.split(",")
    if len(fields) != 2:
        raise SvgInspectionError(f"A path coordinate must be one x,y pair. Got {coords!r}.")
    return tuple(_number(value, "path coordinate", SvgInspectionError) for value in fields)


def _number(text, label, error):
    """Finite ASCII decimal number. float() alone accepts NaN, inf and underscores."""
    if not text or not set(text) <= _NUMBER_CHARS:
        raise error(f"{label} {text!r} is not a decimal number.")
    try:
        value = float(text)
    except ValueError:
        raise error(f"{label} {text!r} is not a decimal number.") from None
    if not math.isfinite(value):
        raise error(f"{label} {text!r} is not a finite number.")
    return value


def _translate(transform):
    """Read one translate(x) or translate(x,y). VTracer 0.6.15 writes no other transform."""
    if transform is None:
        return 0.0, 0.0
    name, _, rest = transform.strip().partition("(")
    fields = rest[:-1].split(",") if rest.endswith(")") else []
    if name.strip() != "translate" or not 1 <= len(fields) <= 2:
        raise SvgInspectionError(f"Unsupported transform {transform!r}. Only one translate is accepted.")
    x, y = [_number(value.strip(), "transform", SvgInspectionError) for value in fields] + [0.0] * (2 - len(fields))
    return x, y


def _add(a, b):
    x, y = a[0] + b[0], a[1] + b[1]
    if not (math.isfinite(x) and math.isfinite(y)):
        raise SvgInspectionError("An accumulated translation is not finite.")
    return x, y


def _fill(element, inherited):
    fill = element.get("fill", inherited)
    if fill is not None and fill.lower() not in _BLACK:
        raise SvgInspectionError(f"fill {fill!r} is not supported. Use black.")
    rule = element.get("fill-rule")
    if rule not in (None, "nonzero"):
        raise SvgInspectionError(f"fill-rule {rule!r} is not supported. Use nonzero.")
    return fill


def _collinear(points):
    origin = points[0]
    second = next(point for point in points if point != origin)
    ax, ay = second[0] - origin[0], second[1] - origin[1]
    return all(ax * (y - origin[1]) - ay * (x - origin[0]) == 0 for x, y in points)


def size_svg(svg, canvas):
    """State the canvas in mm with a px viewBox. Keep every path and transform unchanged."""
    root = ET.fromstring(svg)
    traced = (root.get("width"), root.get("height"))
    if traced != (str(canvas.width_px), str(canvas.height_px)):
        raise ValueError(
            f"The traced SVG is {traced[0]}x{traced[1]} px, "
            f"but the canvas is {canvas.width_px}x{canvas.height_px} px."
        )
    root.set("width", _mm(canvas.width_mm))
    root.set("height", _mm(canvas.height_mm))
    root.set("viewBox", f"0 0 {canvas.width_px} {canvas.height_px}")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def count_paths(svg):
    """Count <path> elements at any depth."""
    return sum(1 for _ in ET.fromstring(svg).iter(_PATH))


def _mm(value):
    """Fixed-point mm with at most 6 decimals, as in the S1.3 OpenSCAD proof."""
    return f"{value:.6f}".rstrip("0").rstrip(".") + "mm"
