#!/usr/bin/env python3
"""st-vector-map: trace a binary mask into an SVG with a physical size (spec 5, 7, 9).

Thin CLI facade. Configuration, raster preparation, the VTracer adapter, and SVG sizing
live in sibling modules. S2.2 (STABL-ascsgqha) is a walking skeleton: mask mode only.
Deferred features fail with exit 2 and name the task that implements them.

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
    parser.add_argument("--input-kind", choices=config.INPUT_KINDS, help="Input meaning. S2.2 supports mask only.")
    parser.add_argument("--width-mm", type=float, help="Physical width of the complete canvas.")
    parser.add_argument("--height-mm", type=float, help="Physical height of the complete canvas.")
    parser.add_argument("--line-width-mm", type=float, help="Edge band width. Lands in S2.3.")
    parser.add_argument("--max-res", type=int, metavar="PX", help="Processing resolution cap. Lands in S2.3.")
    parser.add_argument("--mask", type=Path, help="Mask input. Lands in S2.3.")
    parser.add_argument(
        "--invert",
        action=argparse.BooleanOptionalAction,
        help="Treat dark pixels as material. --no-invert overrides a recipe value.",
    )
    parser.add_argument(
        "--alpha",
        action=argparse.BooleanOptionalAction,
        help="Select material from alpha. Lands in S2.3.",
    )
    parser.add_argument("--preview", action="store_true", help="Write a preview PNG. Lands in S2.7.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination.")
    parser.add_argument("--json", action="store_true", help="Print one JSON result object on stdout.")
    parser.add_argument("--recipe", type=Path, help="schema_version 1 recipe. Explicit options override it.")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    as_json = "--json" in argv
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        destination, paths = _convert(args)
    except UsageError as exc:
        print(parser.format_usage(), end="", file=sys.stderr)
        return _fail(2, "invalid", exc, as_json)
    except config.ConfigError as exc:
        return _fail(2, "invalid", exc, as_json)
    except (OSError, RuntimeError, ValueError) as exc:
        return _fail(1, "failed", exc, as_json)
    if as_json:
        print(json.dumps(_result("converted", svg=str(destination), layers=1, paths=paths)))
    return 0


def _convert(args):
    destination, settings = _settings(args)
    if args.preview:
        raise config.deferred("--preview", config.S27)
    if destination.exists() and not args.overwrite:
        raise config.ConfigError(f"{destination} exists. Use --overwrite to replace it.")

    _progress(f"loading  {settings.input}")
    material = raster.prepare_mask(settings.input, invert=settings.invert)
    height, width = material.shape
    canvas = config.physical_canvas(settings, width, height)
    _progress(
        f"tracing  {width}x{height} px as {canvas.width_mm:g}x{canvas.height_mm:g} mm"
        f" (vtracer {dict(settings.vtracer)})"
    )
    traced = adapter.trace_layer(material, settings.vtracer)
    paths = svg_io.count_paths(traced.svg)
    if paths == 0:
        raise ValueError("The traced mask is empty: VTracer found no material paths.")
    _publish(destination, svg_io.size_svg(traced.svg, canvas))
    _progress(f"saved    {destination} (paths: {paths})")
    return destination, paths


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
        "invert": args.invert,
        "alpha": args.alpha,
        "vtracer": None,
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


def _fail(code, status, exc, as_json):
    print(f"error: {exc}", file=sys.stderr)
    if as_json:
        print(json.dumps(_result(status, diagnostics=[{"level": "error", "message": str(exc)}])))
    return code


def _progress(message):
    print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
