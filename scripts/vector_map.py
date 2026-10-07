#!/usr/bin/env python3
"""st-vector-map: trace masks and edge bands into SVG with physical size (spec 5, 7, 9).

Thin CLI facade. Configuration, raster preparation, the VTracer adapter, and SVG sizing
live in sibling modules. S2.3 (STABL-vjpnctjh) adds oriented raster preparation and edge bands.
Deferred features fail with exit 2 and name the task that implements them.

Exit codes: 0 converted, 2 invalid arguments or configuration, 1 processing or I/O failure.
Progress and diagnostics go to stderr. --json prints one result object on stdout.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import vector_map_config as config
import vector_map_raster as raster
import vector_map_svg as svg_io
import vector_map_vtracer as adapter

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
    parser.add_argument("--input-kind", choices=config.INPUT_KINDS, help="Input meaning. Supports mask and edges. Image mode lands in S3.1.")
    parser.add_argument("--width-mm", type=float, help="Physical width of the complete canvas.")
    parser.add_argument("--height-mm", type=float, help="Physical height of the complete canvas.")
    parser.add_argument("--line-width-mm", type=float, help="Nominal one-pixel edge width. Round upward to odd pixels. Existing thick bands expand further.")
    parser.add_argument("--max-res", type=int, metavar="PX", help="Longest processing side. Nearest-neighbour binary resize. Never upscale.")
    parser.add_argument("--mask", type=Path, help="Image silhouette source. Lands in S3.1.")
    parser.add_argument("--include-mask", type=Path, help="Limit material to white mask pixels. Oriented dimensions must match source.")
    parser.add_argument("--exclude-mask", type=Path, help="Remove white mask pixels before and after expansion. Exclusion wins.")
    parser.add_argument(
        "--invert",
        action=argparse.BooleanOptionalAction,
        help="Treat dark pixels as material. --no-invert overrides a recipe value.",
    )
    parser.add_argument(
        "--alpha",
        action=argparse.BooleanOptionalAction,
        help="Select alpha >= 128 instead of luminance. Requires alpha channel or palette transparency.",
    )
    parser.add_argument("--max-svg-bytes", type=int, metavar="N",
                        help="Largest raw or normalized SVG, in UTF-8 bytes. Default 20971520.")
    parser.add_argument("--max-paths", type=int, metavar="N", help="Most path elements per layer. Default 10000.")
    parser.add_argument("--max-path-commands", type=int, metavar="N",
                        help="Most M, L and Z path commands per layer. Default 1000000.")
    parser.add_argument("--preview", action="store_true", help="Write a preview PNG. Lands in S2.7.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination.")
    parser.add_argument("--json", action="store_true", help="Print one JSON result object on stdout.")
    parser.add_argument("--recipe", type=Path, help="schema_version 1 recipe. Explicit options override it.")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    as_json = "--json" in argv
    parser = build_parser()
    diagnostics = []
    try:
        args = parser.parse_args(argv)
        destination, paths = _convert(args, diagnostics)
    except UsageError as exc:
        print(parser.format_usage(), end="", file=sys.stderr)
        return _fail(2, "invalid", exc, as_json, diagnostics)
    except config.ConfigError as exc:
        return _fail(2, "invalid", exc, as_json, diagnostics)
    except (OSError, RuntimeError, ValueError) as exc:
        return _fail(1, "failed", exc, as_json, diagnostics)
    if as_json:
        print(json.dumps(_result("converted", svg=str(destination), layers=1, paths=paths, diagnostics=diagnostics)))
    return 0


def _convert(args, diagnostics):
    destination, settings = _settings(args)
    if args.preview:
        raise config.deferred("--preview", config.S27)
    _refuse_aliases(destination, (
        ("source", settings.input), ("recipe", args.recipe),
        ("include-mask", settings.include_mask), ("exclude-mask", settings.exclude_mask),
    ))
    if destination.exists() and not args.overwrite:
        raise config.ConfigError(f"{destination} exists. Use --overwrite to replace it.")

    _progress(f"loading  {settings.input}")
    prepared = raster.prepare(settings)
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
    traced = adapter.trace_layer(material, settings.vtracer, svg_limits=settings.svg_limits)
    normalized = svg_io.normalize_svg(traced.svg, canvas, limits=settings.svg_limits)
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
    _publish(destination, normalized.svg)
    _progress(f"saved    {destination} (paths: {paths})")
    return destination, paths


def _refuse_aliases(destination, inputs):
    """--overwrite must never replace an input. samefile follows symlinks and matches hard links."""
    for label, path in inputs:
        if path is None:
            continue
        try:
            same = os.path.samefile(destination, path)
        except OSError:
            continue  # One of the two files does not exist, so no input can be replaced.
        if same:
            raise config.ConfigError(
                f"Destination {destination} is the same file as the {label} {path}. Choose another destination."
            )


def _settings(args):
    """Apply the positional grammar, then defaults, preset, recipe, and explicit CLI values."""
    if len(args.paths) > 2:
        raise config.ConfigError("Give at most two paths: SOURCE DESTINATION.")
    recipe = config.load_recipe(args.recipe) if args.recipe else {}
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
        "vtracer": None,
        "max_svg_bytes": args.max_svg_bytes,
        "max_paths": args.max_paths,
        "max_path_commands": args.max_path_commands,
    }
    # No preset layer yet. S3.5 (STABL-mknlfcui) adds --preset and relief-0.4.
    return args.paths[-1], config.resolve(config.DEFAULTS, {}, recipe, cli)


def _publish(destination, text):
    """Write only after every check passed. S2.6 (STABL-fmjwbrzw) owns staged bundle publication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _result(status, *, svg=None, layers=0, paths=0, diagnostics=()):
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "artifacts": {"svg": svg},
        "counts": {"layers": layers, "paths": paths},
        "diagnostics": list(diagnostics),
    }


def _fail(code, status, exc, as_json, diagnostics=()):
    print(f"error: {exc}", file=sys.stderr)
    if as_json:
        print(json.dumps(_result(status, diagnostics=[{"level": "error", "message": str(exc)}, *diagnostics])))
    return code


def _progress(message):
    print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
