#!/usr/bin/env python3
"""st-vector-map: trace masks and edge bands into SVG with physical size (spec 5, 7, 9).

Thin CLI facade. Configuration, raster preparation, the VTracer adapter, and SVG sizing
live in sibling modules. S2.3 (STABL-vjpnctjh) adds oriented raster preparation and edge bands.
S3.1 (STABL-memwrtos) adds image mode with one silhouette method: --mask, --alpha, or --threshold.

Exit codes: 0 converted, 2 invalid arguments or configuration, 1 processing or I/O failure.
Progress and diagnostics go to stderr. --json prints one result object on stdout.
"""

import argparse
import json
import sys
from pathlib import Path

import vector_map_config as config
import vector_map_raster as raster
import vector_map_svg as svg_io
import vector_map_vtracer as adapter
import vector_map_artifacts as artifacts
import vector_map_preview as preview

RESULT_SCHEMA_VERSION = 1


class UsageError(Exception):
    """argparse rejected the command line."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message)


def build_parser():
    parser = _Parser(
        prog="st-vector-map",
        description="Trace a binary mask into an SVG with a physical size in millimetres.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="SOURCE DESTINATION. With --recipe, give DESTINATION only to use the recipe input.",
    )
    parser.add_argument("--input-kind", choices=config.INPUT_KINDS, help="Input meaning. Image mode needs --mask, --alpha, or --threshold.")
    parser.add_argument("--width-mm", type=float, help="Physical width of the complete canvas.")
    parser.add_argument("--height-mm", type=float, help="Physical height of the complete canvas.")
    parser.add_argument("--line-width-mm", type=float, help="Nominal one-pixel edge width. Round upward to odd pixels. Existing thick bands expand further.")
    parser.add_argument("--max-res", type=int, metavar="PX", help="Longest processing side. Nearest-neighbour binary resize. Never upscale.")
    parser.add_argument("--mask", type=Path,
                        help="Image mode: white pixels (luminance >= 128) of PATH form the silhouette. Source sets the canvas.")
    parser.add_argument("--threshold", type=int, metavar="N",
                        help="Image mode: source luminance >= N (0 to 255) forms the silhouette.")
    parser.add_argument("--include-mask", type=Path, help="Limit material to white mask pixels. Oriented dimensions must match source.")
    parser.add_argument("--exclude-mask", type=Path, help="Remove white mask pixels before and after expansion. Exclusion wins.")
    parser.add_argument(
        "--invert",
        action=argparse.BooleanOptionalAction,
        help="Reverse the material selection. Image mode: applies to every silhouette method. --no-invert overrides a recipe value.",
    )
    parser.add_argument(
        "--alpha",
        action=argparse.BooleanOptionalAction,
        help="Select source alpha >= 128 instead of luminance. Image mode: a silhouette method. "
             "Requires alpha channel or palette transparency.",
    )
    parser.add_argument("--layers", metavar="ROLES",
                        help="Image mode: comma-separated roles from silhouette, structure, detail. Must include "
                             "silhouette. Writes STEM.svg with named groups and STEM.layers/ROLE.svg per nonempty role.")
    for role in config.OPTIONAL_ROLES:
        parser.add_argument(f"--{role}-map", type=Path, metavar="PATH",
                            help=f"Layered image mode: white pixels of PATH form the {role} candidate instead of Canny. "
                                 f"Requires {role} in --layers.")
        parser.add_argument(f"--{role}-width-mm", type=float, metavar="MM",
                            help=f"Layered image mode: widen {role} lines to this nominal width. "
                                 f"Requires {role} in --layers.")
    parser.add_argument("--max-svg-bytes", type=int, metavar="N",
                        help="Largest raw or normalized SVG, in UTF-8 bytes. Default 20971520.")
    parser.add_argument("--max-paths", type=int, metavar="N", help="Most path elements per layer. Default 10000.")
    parser.add_argument("--max-path-commands", type=int, metavar="N",
                        help="Most M, L and Z path commands per layer. Default 1000000.")
    parser.add_argument("--preview", action="store_true",
                        help="Write STEM.preview.png: prepared mask, rendered vector, and source overlay. Needs resvg-py.")
    parser.add_argument("--debug-bundle", action="store_true", help="Save prepared mask and replay recipe in STEM.debug.")
    parser.add_argument("--overwrite", action="store_true", help="Replace owned bundle files. Preserve inputs and unrelated files.")
    parser.add_argument("--json", action="store_true", help="Print one JSON result object on stdout.")
    parser.add_argument("--recipe", type=Path, help="schema_version 1 recipe. Explicit options override it.")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    as_json = "--json" in argv
    parser = build_parser()
    diagnostics = []
    published = {}
    try:
        args = parser.parse_args(argv)
        destination, paths = _convert(args, diagnostics, published)
    except UsageError as exc:
        print(parser.format_usage(), end="", file=sys.stderr)
        return _fail(2, "invalid", exc, as_json, diagnostics, published)
    except config.ConfigError as exc:
        return _fail(2, "invalid", exc, as_json, diagnostics, published)
    except (OSError, RuntimeError, ValueError) as exc:
        return _fail(1, "failed", exc, as_json, diagnostics, published)
    if as_json:
        print(json.dumps(_result("converted", svg=str(destination), layers=1, paths=paths, diagnostics=diagnostics, published=published)))
    return 0


def _convert(args, diagnostics, published):
    snapshots = {}
    destination, settings = _settings(args, snapshots=snapshots)
    bundle = artifacts.Bundle(destination, debug=args.debug_bundle, preview=args.preview, overwrite=args.overwrite,
                              inputs=artifacts.input_paths(settings, args.recipe))
    bundle.check()
    if args.preview:
        preview.require_renderer()
    inputs = artifacts.snapshot_inputs(settings, args.recipe, snapshots)

    _progress(f"loading  {settings.input}")
    prepared = raster.prepare(settings, input_bytes=snapshots)
    material = prepared.material
    canvas = prepared.canvas
    diagnostics.extend(prepared.diagnostics)
    for diagnostic in prepared.diagnostics:
        _progress(f"warning: {diagnostic['message']}")
    _progress(
        f"raster   original {prepared.original_size[0]}x{prepared.original_size[1]}, "
        f"oriented {prepared.oriented_size[0]}x{prepared.oriented_size[1]}, "
        f"processed {prepared.processed_size[0]}x{prepared.processed_size[1]} px, "
        f"scale {canvas.mm_per_px:g} mm/px"
    )
    if prepared.expansion is not None:
        band = prepared.expansion
        _progress(
            f"band     requested {band.requested_width_mm:g} mm, nominal achieved {band.achieved_width_mm:g} mm, "
            f"kernel {band.kernel_size_px} px, radius {band.radius_px} px, "
            f"expansion {band.expansion_mm:g} mm per side"
        )
    height, width = material.shape
    _progress(
        f"tracing  {width}x{height} px as {canvas.width_mm:g}x{canvas.height_mm:g} mm"
        f" (vtracer {dict(settings.vtracer)})"
    )
    debug = artifacts.debug_files(bundle, settings, prepared) if args.debug_bundle else {}
    try:
        traced = adapter.trace_layer(material, settings.vtracer, svg_limits=settings.svg_limits)
        normalized = svg_io.normalize_svg(traced.svg, canvas, limits=settings.svg_limits)
    except (OSError, RuntimeError, ValueError) as exc:
        message = f"VTracer {_version('vtracer')} layer {artifacts.layer_id(settings)}: {exc}"
        raise RuntimeError(_retain_debug(bundle, debug, message, published)) from exc
    paths = normalized.metrics.paths
    degenerate = normalized.metrics.degenerate_subpaths
    if degenerate:
        diagnostic = {
            "level": "warning", "code": "degenerate_subpaths", "subpaths": degenerate,
            "message": (
                f"VTracer returned {degenerate} degenerate subpaths (points or lines) inside valid paths. "
                "They are kept unchanged and add no area."
            ),
        }
        diagnostics.append(diagnostic)
        _progress(f"warning: {diagnostic['message']}")
    files = {bundle.relative(destination): normalized.svg.encode("utf-8"), **debug}
    rendered = None
    if args.preview:
        _progress(f"preview  {bundle.preview}")
        try:
            rendered = preview.compose(prepared, normalized.svg, snapshots[settings.input])
        except (OSError, RuntimeError, ValueError) as exc:
            message = f"Preview renderer resvg-py {_version('resvg-py')}: {exc}"
            raise RuntimeError(_retain_debug(bundle, debug, message, published)) from exc
        files[bundle.relative(bundle.preview)] = rendered.png
    manifest = artifacts.manifest_bytes(settings, prepared, normalized, inputs, files, diagnostics,
                                        upstream_args=traced.upstream_args,
                                        preview=rendered.provenance if rendered else None)
    bundle.publish(files, manifest)
    published["manifest"] = str(bundle.manifest)
    if rendered:
        published["preview"] = str(bundle.preview)
    if debug:
        published["debug"] = str(bundle.debug_directory)
    _progress(f"saved    {destination} (paths: {paths})")
    return destination, paths


def _version(name):
    try:
        return artifacts.package_version(name)
    except RuntimeError:
        return "unavailable"


def _retain_debug(bundle, debug, message, published):
    """Publish debug files after an upstream failure. Return the failure message."""
    if not debug:
        return message
    try:
        bundle.publish(debug)
    except (OSError, RuntimeError, ValueError) as debug_exc:
        return message + f" Debug publication failed: {debug_exc}"
    published["debug"] = str(bundle.debug_directory)
    return message + (
        f" Debug files retained at {bundle.debug_directory}. "
        "Rerun with --overwrite to replace mask.png and recipe.json."
    )


def _settings(args, *, snapshots=None):
    """Apply the positional grammar, then defaults, preset, recipe, and explicit CLI values."""
    if len(args.paths) > 2:
        raise config.ConfigError("Give at most two paths: SOURCE DESTINATION.")
    recipe = config.load_recipe(args.recipe, snapshots=snapshots) if args.recipe else {}
    cli = {
        "input": args.paths[0] if len(args.paths) == 2 else None,
        "input_kind": args.input_kind,
        "width_mm": args.width_mm,
        "height_mm": args.height_mm,
        "line_width_mm": args.line_width_mm,
        "max_res": args.max_res,
        "mask": args.mask,
        "include_mask": args.include_mask,
        "exclude_mask": args.exclude_mask,
        "invert": args.invert,
        "alpha": args.alpha,
        "threshold": args.threshold,
        "vtracer": None,
        "max_svg_bytes": args.max_svg_bytes,
        "max_paths": args.max_paths,
        "max_path_commands": args.max_path_commands,
        "layers": args.layers,
        **{role: _role_layer(args, role) for role in config.OPTIONAL_ROLES},
    }
    # No preset layer yet. S3.5 (STABL-mknlfcui) adds --preset and relief-0.4.
    return args.paths[-1], config.resolve(config.DEFAULTS, {}, recipe, cli)


def _role_layer(args, role):
    """Explicit role flags as one role layer. A CLI map selects the map source. None when no flag is given."""
    fields = {}
    path = getattr(args, f"{role}_map")
    if path is not None:
        fields.update(source="map", path=path)
    width = getattr(args, f"{role}_width_mm")
    if width is not None:
        fields["width_mm"] = width
    return fields or None


def _result(status, *, svg=None, layers=0, paths=0, diagnostics=(), published=None):
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "artifacts": {"svg": svg, **(published or {})},
        "counts": {"layers": layers, "paths": paths},
        "diagnostics": list(diagnostics),
    }


def _fail(code, status, exc, as_json, diagnostics=(), published=None):
    print(f"error: {exc}", file=sys.stderr)
    if as_json:
        print(json.dumps(_result(status, diagnostics=[{"level": "error", "message": str(exc)}, *diagnostics], published=published)))
    return code


def _progress(message):
    print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
