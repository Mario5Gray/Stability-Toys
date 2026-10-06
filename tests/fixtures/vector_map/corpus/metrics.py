"""S1.4 locked acceptance metrics (STABL-cbwzjyky, spec 10).

Phase one locks every value and algorithm in this module before any fitting sweep.
Do not relax a tolerance only to make a candidate pass.

Conventions:
- A material array is 2-D bool. True is material. Row 0 is the top. Column 0 is the left.
- Connectivity: material and background components use 4-connectivity
  (scipy.ndimage.label default cross). A fixture must give the same counts with 8-connectivity.
- A hole is a background component that touches no canvas edge.
- Perimeter: the number of horizontally or vertically adjacent in-canvas pixel pairs
  with unequal values. The canvas edge is not a boundary.
- Boundary distance of a pixel: the Euclidean distance from its center to the nearest
  center of an opposite-class pixel (scipy distance_transform_edt, unit sampling),
  minus 0.5. Infinite when no opposite-class pixel exists.
- Bounds: (x0, y0, x1, y1) in pixel edges. x1 and y1 are exclusive.
- Centroid: the mean of pixel centers (column + 0.5, row + 0.5), in px from the top-left corner.
- SVG: parsed with xml.etree. Paths are <path> elements. Path data tokens split on
  whitespace. Each token starts with M, L or Z (polygon mode). Any other command is an error.
  A subpath starts at each M. VTracer 0.6.15 writes only translate(x,y) transforms.
- Polygon area: per path, the absolute sum of signed shoelace areas of its subpaths
  (VTracer writes holes with opposite winding). The total sums all paths. Unit: px^2.
"""

import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
from scipy import ndimage

TOLERANCES = {
    "topology": "exact",
    "bounds_shift_px": 1.0,
    "bbox_size_delta_px": 1.0,
    "centroid_shift_px": 0.5,
    "orientation": "strictly_lowest_xor",
    "xor_per_perimeter_px": 0.5,
    "boundary_max_distance_px": 1.5,
    "area_dev_abs_per_perimeter_px": 0.5,
    "mesh_volume_rel": 1e-4,
    "mesh_z_abs_mm": 1e-6,
}

# Full logarithmic grid over the wrapper policy range 0..128 (adapter FILTER_SPECKLE_RANGE).
SPECKLE_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128)
CURRENT_DEFAULT = 4

RENDER = {
    "renderer": "resvg-py",
    "version": "0.5.0",
    "shape_rendering": "crisp_edges",
    "background": "#ffffff",
    "material": "luma < 128",
    "resolution": "processing size, 1 SVG unit = 1 px",
}

TIMING = {"warmup_calls": 1, "timed_calls": 5, "statistic": "median"}

# S1.3 mapping (STABL-nygbbrrn): OpenSCAD 2021.01 imports a mm-header SVG exactly, so the
# extruded volume is polygon area x thickness. The SVG header states mm width/height plus a
# px viewBox (tests/test_vector_map_openscad.py physical_svg). Volume is translation-invariant.
EXTRUSION = {"mm_per_px": 25.4 / 96, "thickness_mm": 2.0, "scad": "tests/fixtures/vector_map/relief_proof.scad"}


def expected_volume(polygon_area_px):
    """Return polygon area x thickness in mm^3."""
    return polygon_area_px * EXTRUSION["mm_per_px"] ** 2 * EXTRUSION["thickness_mm"]

_SVG = "{http://www.w3.org/2000/svg}"
_EIGHT = np.ones((3, 3), bool)


# --- Raster algorithms --------------------------------------------------------


def topology(material, structure=None):
    """Return (components, holes). Holes are background components that touch no canvas edge."""
    _, components = ndimage.label(material, structure=structure)
    background, count = ndimage.label(~material, structure=structure)
    edge = np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]])
    holes = count - len(set(np.unique(edge)) - {0})
    return components, holes


def connectivity_agrees(material):
    """True when 4- and 8-connectivity give the same component and hole counts."""
    return topology(material) == topology(material, structure=_EIGHT)


def perimeter(material):
    """Count adjacent in-canvas pixel pairs with unequal values."""
    material = np.asarray(material, bool)
    return int(np.count_nonzero(material[1:, :] != material[:-1, :]) + np.count_nonzero(material[:, 1:] != material[:, :-1]))


def boundary_distance(material):
    """Distance of each pixel center to the nearest opposite-class center, minus 0.5."""
    material = np.asarray(material, bool)
    if material.all() or not material.any():
        return np.full(material.shape, np.inf)
    inside = ndimage.distance_transform_edt(material)
    outside = ndimage.distance_transform_edt(~material)
    return np.where(material, inside, outside) - 0.5


def bounds(material):
    """Return (x0, y0, x1, y1) with exclusive maxima, or None for an empty array."""
    ys, xs = np.nonzero(material)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def centroid(material):
    """Return the mean pixel center (x, y), or None for an empty array."""
    ys, xs = np.nonzero(material)
    if xs.size == 0:
        return None
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5)


def orientation_variants(material):
    """Yield (name, array) for each wrong orientation with the same shape that differs from the input.

    Flips and the 180-degree rotation apply to every case. Transpose variants apply to square cases only.
    """
    candidates = [("fliplr", np.fliplr), ("flipud", np.flipud), ("rot180", lambda a: np.rot90(a, 2))]
    if material.shape[0] == material.shape[1]:
        candidates += [
            ("transpose", np.transpose),
            ("antitranspose", lambda a: np.rot90(a, 2).T),
            ("rot90", lambda a: np.rot90(a, 1)),
            ("rot270", lambda a: np.rot90(a, 3)),
        ]
    out = []
    for name, op in candidates:
        variant = op(material)
        if variant.shape == material.shape and not np.array_equal(variant, material):
            out.append((name, variant))
    return out


# --- SVG algorithms -----------------------------------------------------------


def _paths(svg):
    root = ET.fromstring(svg)
    for element in root.iter(f"{_SVG}path"):
        yield element.get("d", ""), _translate(element.get("transform", ""))


def _translate(transform):
    if not transform:
        return 0.0, 0.0
    name, _, rest = transform.partition("(")
    if name.strip() != "translate" or not rest.endswith(")"):
        raise ValueError(f"Unexpected transform: {transform!r}")
    x, _, y = rest[:-1].partition(",")
    return float(x), float(y or 0)


def _subpaths(d):
    """Return (point lists, command count). Each M starts a subpath. Z closes it."""
    subpaths, commands = [], 0
    for token in d.split():
        command = token[0]
        if command not in "MLZ":
            raise ValueError(f"Unexpected polygon path command: {token!r}")
        commands += 1
        if command == "M":
            subpaths.append([])
        if command in "ML":
            if not subpaths:
                raise ValueError(f"Path data must start with M: {d[:40]!r}")
            x, _, y = token[1:].partition(",")
            subpaths[-1].append((float(x), float(y)))
    return subpaths, commands


def svg_canvas(svg):
    """Return the root (width, height) as numbers."""
    root = ET.fromstring(svg)
    width, height = root.get("width"), root.get("height")
    if width is None or height is None:
        raise ValueError("SVG root has no width or height.")
    return int(float(width)), int(float(height))


def svg_complexity(svg):
    """Return path, subpath, command and byte counts."""
    paths = subpaths = commands = 0
    for d, _ in _paths(svg):
        parts, count = _subpaths(d)
        paths += 1
        subpaths += len(parts)
        commands += count
    return {"paths": paths, "subpaths": subpaths, "commands": commands, "svg_bytes": len(svg.encode("utf-8"))}


def _shoelace(points):
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    return 0.5 * float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def polygon_area(svg):
    """Return the filled polygon area in px^2. Translation does not change area."""
    return float(sum(abs(sum(_shoelace(points) for points in _subpaths(d)[0])) for d, _ in _paths(svg)))


# --- Rendering ----------------------------------------------------------------


def render(svg, width, height):
    """Rasterize with the locked renderer. Return the bool material array."""
    import resvg_py
    from PIL import Image

    png = bytes(
        resvg_py.svg_to_bytes(
            svg_string=svg,
            width=width,
            height=height,
            background=RENDER["background"],
            shape_rendering=RENDER["shape_rendering"],
        )
    )
    return np.asarray(Image.open(BytesIO(png)).convert("L")) < 128


# --- Mesh volume ----------------------------------------------------------------


def mesh_volume(triangles):
    """Return the absolute enclosed volume of a closed triangle mesh, shape (n, 3, 3)."""
    t = np.asarray(triangles, float)
    return abs(float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0)


def mesh_volume_ok(volume, expected):
    return abs(volume - expected) <= TOLERANCES["mesh_volume_rel"] * abs(expected)


# --- Per-case evaluation --------------------------------------------------------


def evaluate(mask, rendered, polygon_area):
    """Measure one rendered trace against its mask. Return per-rule checks and values."""
    mask = np.asarray(mask, bool)
    rendered = np.asarray(rendered, bool)
    tol = TOLERANCES
    edge = perimeter(mask)
    mask_area = int(mask.sum())
    area_signed = (polygon_area - mask_area) / edge if edge else float("inf")
    result = {
        "mask_area_px": mask_area,
        "perimeter_px": edge,
        "polygon_area_px": polygon_area,
        "area_dev_signed_per_perimeter_px": area_signed,
        "area_dev_abs_per_perimeter_px": abs(area_signed),
        "topology_mask": list(topology(mask)),
    }
    checks = {"area": abs(area_signed) <= tol["area_dev_abs_per_perimeter_px"]}

    if rendered.shape != mask.shape:
        result.update(render_shape=list(rendered.shape))
        for name in ("scale", "topology", "bounds", "centroid", "orientation", "xor", "boundary"):
            checks[name] = False
        return _finish(result, checks)

    xor = rendered ^ mask
    xor_px = int(xor.sum())
    distance = boundary_distance(mask)
    max_distance = float(distance[xor].max()) if xor_px else 0.0
    result.update(
        topology_render=list(topology(rendered)),
        xor_px=xor_px,
        xor_pct_of_material=100.0 * xor_px / mask_area if mask_area else float("inf"),
        xor_pct_of_canvas=100.0 * xor_px / mask.size,
        xor_per_perimeter_px=xor_px / edge if edge else float("inf"),
        boundary_max_distance_px=max_distance,
    )
    checks["topology"] = result["topology_render"] == result["topology_mask"]
    checks["xor"] = result["xor_per_perimeter_px"] <= tol["xor_per_perimeter_px"]
    checks["boundary"] = max_distance <= tol["boundary_max_distance_px"]

    mb, rb = bounds(mask), bounds(rendered)
    if mb is None or rb is None:
        checks.update(bounds=False, scale=False, centroid=False)
        result.update(bounds_mask=mb, bounds_render=rb)
    else:
        shift = max(abs(a - b) for a, b in zip(mb, rb))
        size_delta = max(abs((mb[2] - mb[0]) - (rb[2] - rb[0])), abs((mb[3] - mb[1]) - (rb[3] - rb[1])))
        mc, rc = centroid(mask), centroid(rendered)
        assert mc is not None and rc is not None
        centroid_shift = float(np.hypot(mc[0] - rc[0], mc[1] - rc[1]))
        result.update(
            bounds_mask=list(mb),
            bounds_render=list(rb),
            bounds_shift_px=shift,
            bbox_size_delta_px=size_delta,
            centroid_shift_px=centroid_shift,
        )
        checks["bounds"] = shift <= tol["bounds_shift_px"]
        checks["scale"] = size_delta <= tol["bbox_size_delta_px"]
        checks["centroid"] = centroid_shift <= tol["centroid_shift_px"]

    variants = {name: int((rendered ^ variant).sum()) for name, variant in orientation_variants(mask)}
    result["orientation_xor_px"] = variants
    checks["orientation"] = all(xor_px < other for other in variants.values())
    return _finish(result, checks)


def _finish(result, checks):
    result["checks"] = checks
    result["failures"] = sorted(name for name, ok in checks.items() if not ok)
    result["passed"] = not result["failures"]
    return result


# --- Default selection ----------------------------------------------------------


def _dominates(a, b):
    """a dominates b: no case worsens on XOR-per-perimeter or command count, and one case improves."""
    strict = False
    for case, ra in a.items():
        rb = b[case]
        if ra["xor_per_perimeter_px"] > rb["xor_per_perimeter_px"] or ra["commands"] > rb["commands"]:
            return False
        if ra["xor_per_perimeter_px"] < rb["xor_per_perimeter_px"] or ra["commands"] < rb["commands"]:
            strict = True
    return strict


def select_default(results):
    """Select one filter_speckle value. Return (value or None, reason).

    results maps candidate -> case -> {passed, xor_per_perimeter_px, commands}.
    1. Take the highest pass count. A unique leader wins.
    2. On a tie, take the candidate that dominates every other tied candidate.
    3. Otherwise keep CURRENT_DEFAULT when it is tied.
    4. Otherwise return None: the human review decides.
    """
    counts = {value: sum(r["passed"] for r in cases.values()) for value, cases in results.items()}
    best = max(counts.values())
    tied = [value for value in results if counts[value] == best]
    if len(tied) == 1:
        return tied[0], f"unique highest pass count ({best}/{len(results[tied[0]])})"
    for value in tied:
        if all(_dominates(results[value], results[other]) for other in tied if other != value):
            return value, f"dominates every candidate tied at {best} passes"
    if CURRENT_DEFAULT in tied:
        return CURRENT_DEFAULT, f"no unique dominator among {tied}; keep {CURRENT_DEFAULT}"
    return None, f"no unique dominator among {tied}, and {CURRENT_DEFAULT} is not tied"
