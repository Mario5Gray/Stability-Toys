"""CAD compatibility for published layer SVGs (STABL-ygtexxzr).

OpenSCAD exits 0 on a layer whose rings touch at one point, then reports a non-closed mesh. The
material between two rings that meet at a point has zero width, and the extrusion has no closed
wall there. normalize_svg() promises to change no geometry, so the repair lives in this step. The
caller runs it on each normalized layer before the layer is combined and published.

Repair rule, fixed and explicit. Only rings with three distinct points count. A pinch is a vertex of
one ring that lies on the boundary of another ring. A hole is a ring that winds against the first
ring of its path. One ring gives way. A hole gives way before an island. Between two holes or two
islands the smaller area gives way, then the earlier ring. Each contact point on that ring moves by
the pinch width. A contact point inside an edge becomes a new vertex first. A hole moves into
itself, so it shrinks and the holes stay apart. An island moves out of itself, so it grows and
overlaps its neighbour. Both add material. Neither removes any.

The pinch width is the smaller of 0.01 mm and a quarter pixel, so it follows the scale. A repair
that leaves a pinch, or that makes a new one, is rejected whole.

Structure and detail layers need a second rule. A detail ring that touches a structure ring at a
point makes a vertical edge with four faces where the two extrusions meet, because the tops
differ. structure_detail_contact() finds this contact. The SCAD source then grows the detail by
detail_growth_mm(). That growth is a SCAD operation. Python holds no polygon clipper.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

from vector_map_svg import NormalizedSvg, _add, _G, _PATH, _parse, _translate

PINCH_CAP_MM = 0.01
PINCH_CAP_PX = 0.25
CONTACT_TOLERANCE_PX = 1e-9


@dataclass(frozen=True)
class Pinch:
    """Two rings that touch. a and b are (path index, ring index) in document order."""
    a: tuple
    b: tuple
    point: tuple


@dataclass(frozen=True)
class CadResult:
    layer: NormalizedSvg
    closed: int
    unrepaired: int


def pinch_width_px(canvas):
    """The repair width in pixels: at most 0.01 mm and at most a quarter pixel."""
    return min(PINCH_CAP_MM / canvas.mm_per_px, PINCH_CAP_PX)


def detail_growth_mm(canvas):
    """The detail growth in millimetres. It equals the pinch width."""
    return pinch_width_px(canvas) * canvas.mm_per_px


def find_pinches(svg):
    """Return every pinch in svg text, in document order. Translate is applied before the test."""
    return _pinches(_rings(_parse(svg))[1])


def structure_detail_contact(structure_svg, detail_svg):
    """True when a vertex of one layer's ring lies on the boundary of the other layer's ring."""
    structure = [ring for ring in _rings(_parse(structure_svg))[1] if ring.solid]
    detail = [ring for ring in _rings(_parse(detail_svg))[1] if ring.solid]
    return any(_touch(a, b) for a in structure for b in detail)


def close_pinches(layer, canvas):
    """Return CadResult. A clean layer comes back as the same object with nothing counted."""
    root = _parse(layer.svg)
    paths, rings = _rings(root)
    pinches = _pinches(rings)
    if not pinches:
        return CadResult(layer, 0, 0)
    width = pinch_width_px(canvas)
    by_key = {(ring.path, ring.index): ring for ring in rings}
    edits = {}
    for pinch in pinches:
        giving = _giving_way(by_key[pinch.a], by_key[pinch.b])
        taking = by_key[pinch.b] if giving is by_key[pinch.a] else by_key[pinch.a]
        edits.setdefault((giving.path, giving.index), []).append(taking)
    for (path, index), others in edits.items():
        ring = by_key[(path, index)]
        ring.local = _moved(ring, others, width if ring.hole else -width)
    _write(paths, rings, edits)
    text = _serialize(root)
    after = _pinches(_rings(_parse(text))[1])
    if after:
        return CadResult(layer, 0, len(pinches))
    metrics = replace(layer.metrics, normalized_bytes=len(text.encode("utf-8")))
    return CadResult(NormalizedSvg(svg=text, metrics=metrics), len(pinches), 0)


class _Ring:
    def __init__(self, path, index, local, offset):
        self.path, self.index, self.local, self.offset = path, index, list(local), offset
        self.solid = len(set(self.local)) >= 3
        self.hole = False

    @property
    def absolute(self):
        return [(x + self.offset[0], y + self.offset[1]) for x, y in self.local]

    def area(self):
        points = self.local
        return sum(points[i][0] * points[(i + 1) % len(points)][1]
                   - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))) / 2


def _rings(root):
    """Return (path elements, rings). A hole winds against the first solid ring of its path."""
    paths, rings = [], []

    def visit(element, offset):
        offset = _add(offset, _translate(element.get("transform")))
        if element.tag == _G:
            for child in element:
                visit(child, offset)
        elif element.tag == _PATH:
            paths.append(element)
            first = None
            for index, points in enumerate(_subpaths(element.get("d"))):
                ring = _Ring(len(paths) - 1, index, points, offset)
                if ring.solid:
                    if first is None:
                        first = ring.area()
                    ring.hole = ring.area() * first < 0
                rings.append(ring)

    for child in root:
        visit(child, (0.0, 0.0))
    return paths, rings


def _subpaths(d):
    """Closed M/L/Z data in the accepted dialect. Inspection already ran, so no format checks here."""
    subpaths, points = [], None
    for token in filter(None, d.split(" ")):
        command, coords = token[0], token[1:]
        if command == "Z":
            subpaths.append(points)
            points = None
        else:
            x, y = coords.split(",")
            if command == "M":
                points = []
            points.append((float(x), float(y)))
    return subpaths


def _pinches(rings):
    solid = [ring for ring in rings if ring.solid]
    boxes = [_box(ring) for ring in solid]
    found = []
    for i, a in enumerate(solid):
        for j in range(i + 1, len(solid)):
            b = solid[j]
            if _apart(boxes[i], boxes[j]):
                continue
            point = _touch(a, b)
            if point:
                found.append(Pinch((a.path, a.index), (b.path, b.index), point))
    return found


def _box(ring):
    points = ring.absolute
    pad = CONTACT_TOLERANCE_PX
    return (min(p[0] for p in points) - pad, min(p[1] for p in points) - pad,
            max(p[0] for p in points) + pad, max(p[1] for p in points) + pad)


def _apart(a, b):
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _touch(a, b):
    """The first vertex of one ring that lies on the boundary of the other, or None."""
    for source, target in ((a, b), (b, a)):
        edges = _edges(target.absolute)
        for point in source.absolute:
            if any(_distance(point, start, end) <= CONTACT_TOLERANCE_PX for start, end in edges):
                return point
    return None


def _edges(points):
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length))
    return math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


def _giving_way(a, b):
    """The ring that moves. A hole before an island, then the smaller area, then the earlier ring."""
    holes = [ring for ring in (a, b) if ring.hole]
    return min(holes or (a, b), key=lambda ring: (abs(ring.area()), ring.path, ring.index))


def _moved(ring, others, width):
    """The ring with every contact point moved into the ring by width. Negative width moves out. Edge contacts gain a vertex."""
    points = list(ring.local)
    marked = set()
    for other in others:
        for point in other.absolute:
            local = (point[0] - ring.offset[0], point[1] - ring.offset[1])
            edges = _edges(points)
            nearest = min(range(len(edges)), key=lambda i: _distance(local, *edges[i]))
            if _distance(local, *edges[nearest]) > CONTACT_TOLERANCE_PX:
                continue
            at = next((i for i, p in enumerate(points) if math.dist(p, local) <= CONTACT_TOLERANCE_PX), None)
            if at is None:
                points.insert(nearest + 1, local)
                marked = {i + 1 if i > nearest else i for i in marked}
                at = nearest + 1
            marked.add(at)
    for other in others:  # the hole's own vertices on the other ring's boundary
        edges = _edges([(x - ring.offset[0], y - ring.offset[1]) for x, y in other.absolute])
        marked |= {i for i, p in enumerate(points)
                   if any(_distance(p, start, end) <= CONTACT_TOLERANCE_PX for start, end in edges)}
    ccw = ring.area() > 0
    out = list(points)
    for i in marked:
        out[i] = _inward(points, i, ccw, width)
    return out


def _inward(points, i, ccw, width):
    """Move vertex i by width along the sum of the two inward edge normals."""
    here = points[i]
    normals = []
    for start, end in ((points[i - 1], here), (here, points[(i + 1) % len(points)])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        normals.append((-dy / length, dx / length) if ccw else (dy / length, -dx / length))
    sx, sy = normals[0][0] + normals[1][0], normals[0][1] + normals[1][1]
    size = math.hypot(sx, sy)
    if size < 1e-12:
        sx, sy, size = normals[0][0], normals[0][1], 1.0
    return (here[0] + width * sx / size, here[1] + width * sy / size)


def _write(paths, rings, edits):
    for path_index in {path for path, _ in edits}:
        data = ""
        for ring in (r for r in rings if r.path == path_index):
            data += "M" + " L".join(f"{_number(x)},{_number(y)}" for x, y in ring.local) + " Z "
        paths[path_index].set("d", data)


def _number(value):
    text = repr(round(value, 6))
    if text.endswith(".0"):
        text = text[:-2]
    return "0" if text == "-0" else text


def _serialize(root):
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"
