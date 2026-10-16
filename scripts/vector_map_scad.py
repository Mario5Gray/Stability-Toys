"""OpenSCAD source assembly for the st-vector-map relief (S3.4, STABL-gorlibrt; spec 6.5, 7).

The builder is pure. It returns source text and never starts an OpenSCAD process. Every import is
relative to the source file and reads one published S2.5-normalized layer SVG.

Geometry. Backing B spans the canvas under every other solid. The silhouette occupies [B, B + S].
Raised structure and detail sit on the silhouette top. Engraved detail removes material from the
silhouette only, so structure ridges stay uncut. A bounded OVERLAP joins solids at support planes.
No top height changes and no geometry goes below Z = 0.

Nominal Z intervals: backing [0, B], silhouette [B, B + S], raised structure [B + S, B + S + H],
raised detail [B + S, B + S + D], engraving removal [B + S - D, B + S].
"""

from decimal import Decimal

from vector_map_config import DETAIL_MODES, ConfigError, check_relief_length

OVERLAP_MM = 0.01
ROLE_ORDER = ("silhouette", "structure", "detail")
HEIGHT_FIELDS = {
    "silhouette": "silhouette_thickness_mm",
    "structure": "structure_height_mm",
    "detail": "detail_height_mm",
}


def build_scad(canvas, layers, *, center=False, detail_mode="raised", backing_mm=0.0,
               silhouette_thickness_mm=None, structure_height_mm=None, detail_height_mm=None,
               detail_growth_mm=0.0, recess_bridge_mm=0.0, recess_bridges=()):
    """Return deterministic OpenSCAD source for the relief assembly.

    canvas gives width_mm and height_mm. layers maps every published role to its SVG path relative
    to the source file. An unpublished role gets no import and no solid. center applies one outer
    translate to the whole assembly. detail_growth_mm above zero grows raised detail by that much
    (CAD compatibility, STABL-ygtexxzr). It applies only with raised detail and a structure layer.
    Engraved detail is cut by engrave_2d(): raw detail minus raw structure, with no silhouette clip.
    recess_bridges lists pixel points where the recess meets the outline at an isolated point. A
    square of half side recess_bridge_mm stays uncut at each. They apply only with engraved detail.
    """
    unknown = sorted(set(layers) - set(ROLE_ORDER))
    if unknown:
        raise ConfigError(f"SCAD layers name unknown roles: {', '.join(unknown)}.")
    if "silhouette" not in layers:
        raise ConfigError("SCAD source needs the silhouette layer. Publish silhouette in the layered run.")
    if detail_mode not in DETAIL_MODES:
        raise ConfigError(f"detail_mode must be one of {', '.join(DETAIL_MODES)}. Got {detail_mode!r}.")
    heights = {
        "silhouette": _published_height(layers, "silhouette", silhouette_thickness_mm),
        "structure": _published_height(layers, "structure", structure_height_mm),
        "detail": _published_height(layers, "detail", detail_height_mm),
    }
    backing = check_relief_length("backing_mm", backing_mm, positive=False)
    if detail_mode == "engraved":
        thickness, detail = heights["silhouette"], heights["detail"]
        if detail is not None and thickness is not None and detail >= thickness:
            raise ConfigError(
                f"Engraved detail_height_mm {_number(detail)} must be below the silhouette "
                f"thickness {_number(thickness)} mm. Reduce detail_height_mm."
            )
    growth = detail_growth_mm if detail_mode == "raised" and {"structure", "detail"} <= set(layers) else 0.0
    engraved = detail_mode == "engraved" and "detail" in layers
    bridges = list(recess_bridges) if engraved and recess_bridge_mm > 0 else []
    return "\n".join(_source(canvas, layers, center, detail_mode, _number(backing), heights, growth,
                             recess_bridge_mm if bridges else 0.0, bridges)) + "\n"


def _published_height(layers, role, value):
    """Require a usable height for a published role. An unpublished role needs none."""
    if role not in layers:
        return None
    name = HEIGHT_FIELDS[role]
    if value is None:
        raise ConfigError(f"{role} is published, so {name} is required for SCAD export.")
    return check_relief_length(name, value)


def _source(canvas, layers, center, detail_mode, backing, heights, growth=0.0, bridge=0.0, bridges=()):
    width, height = _number(canvas.width_mm), _number(canvas.height_mm)
    lines = [
        "// st-vector-map S3.4 relief source (STABL-gorlibrt). Generated file.",
        "// Edit the parameters above the modules, then reopen the file in OpenSCAD.",
        "// Layer imports are relative to this file, uncentered, in millimetres.",
        "// Top heights: backing B, silhouette B + S, structure B + S + H, detail B + S + D.",
        "",
        f"B = {backing}; // backing thickness. 0 gives no backing.",
        f"S = {_number(heights['silhouette'])}; // silhouette thickness.",
    ]
    if "structure" in layers:
        lines.append(f"H = {_number(heights['structure'])}; // structure height above the silhouette.")
    if "detail" in layers:
        note = ("engraving depth below the silhouette top." if detail_mode == "engraved"
                else "detail height above the silhouette.")
        lines.append(f"D = {_number(heights['detail'])}; // {note}")
    lines += [
        f"OVERLAP = {_number(OVERLAP_MM)}; // bounded support overlap. Z never goes below 0.",
    ]
    if growth > 0:
        lines.append(f"DETAIL_GROW = {_number(growth)}; // detail grows by this much so it never meets structure at a point.")
    if bridges:
        points = ", ".join(f"[{_number(x * canvas.mm_per_px)}, {_number(canvas.height_mm - y * canvas.mm_per_px)}]"
                           for x, y in bridges)
        lines.append(f"RECESS_BRIDGE = {_number(bridge)}; // half side of the uncut square where the recess meets the outline at a point.")
        lines.append(f"RECESS_BRIDGES = [{points}]; // those points, in millimetres.")
    lines += [
        "",
        "module silhouette_2d() {",
        f"  import({_literal(layers['silhouette'])}, center=false);",
        "}",
        "",
    ]
    if "structure" in layers:
        lines += [
            "module structure_2d() {",
            "  intersection() {",
            f"    import({_literal(layers['structure'])}, center=false);",
            "    silhouette_2d();",
            "  }",
            "}",
            "",
        ]
    if "detail" in layers:
        if "structure" in layers:
            grow = "offset(delta = DETAIL_GROW) " if growth > 0 else ""
            shrink = "offset(delta = -DETAIL_GROW) " if growth > 0 else ""
            lines += [
                "module detail_2d() {",
                "  difference() {",
                "    intersection() {",
                f"      {grow}import({_literal(layers['detail'])}, center=false);",
                "      silhouette_2d();",
                "    }",
                f"    {shrink}structure_2d();",
                "  }",
                "}",
                "",
            ]
        else:
            lines += [
                "module detail_2d() {",
                "  intersection() {",
                f"    import({_literal(layers['detail'])}, center=false);",
                "    silhouette_2d();",
                "  }",
                "}",
                "",
            ]
    if detail_mode == "engraved" and "detail" in layers:
        lines += _engrave_module(layers, bridges)
    lines += ["module relief() {", "  if (B > 0) {",
              f"    linear_extrude(height = B) square([{width}, {height}]);", "  }"]
    engraving = detail_mode == "engraved" and "detail" in layers
    silhouette = [
        "  translate([0, 0, max(0, B - OVERLAP)])",
        "    linear_extrude(height = S + min(B, OVERLAP)) silhouette_2d();",
    ]
    if engraving:
        lines += [
            "  difference() {",
            "    translate([0, 0, max(0, B - OVERLAP)])",
            "      linear_extrude(height = S + min(B, OVERLAP)) silhouette_2d();",
            "    translate([0, 0, B + S - D])",
            "      linear_extrude(height = D + OVERLAP) engrave_2d();",
            "  }",
        ]
    else:
        lines += silhouette
    if "structure" in layers:
        lines += [
            "  translate([0, 0, max(0, B + S - OVERLAP)])",
            "    linear_extrude(height = H + min(B + S, OVERLAP)) structure_2d();",
        ]
    if "detail" in layers and not engraving:
        lines += [
            "  translate([0, 0, max(0, B + S - OVERLAP)])",
            "    linear_extrude(height = D + min(B + S, OVERLAP)) detail_2d();",
        ]
    lines.append("}")
    lines.append("")
    if center:
        lines.append(f"translate([{_number(-canvas.width_mm / 2)}, {_number(-canvas.height_mm / 2)}, 0]) relief();")
    else:
        lines.append("relief();")
    return lines


def _engrave_module(layers, bridges):
    """The engraving cut: raw detail minus raw structure and the bridges. No silhouette clip."""
    removed = [f"    import({_literal(layers['structure'])}, center=false);"] if "structure" in layers else []
    if bridges:
        removed.append("    for (p = RECESS_BRIDGES) translate(p) square(2 * RECESS_BRIDGE, center = true);")
    detail = f"import({_literal(layers['detail'])}, center=false);"
    if not removed:
        return ["module engrave_2d() {", f"  {detail}", "}", ""]
    return ["module engrave_2d() {", "  difference() {", f"    {detail}", *removed, "  }", "}", ""]


def _number(value):
    """Plain decimal from the shortest float repr. No exponent and no trailing zeros."""
    text = format(Decimal(repr(float(value))), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _literal(path):
    """Quote one OpenSCAD filename string. Escape the backslash and the quote."""
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'
