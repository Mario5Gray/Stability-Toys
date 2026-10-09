"""Vector bundle provenance and publication. STABL-fmjwbrzw."""

import hashlib
import errno
import json
import os
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path

from vector_map_config import ConfigError


def json_bytes(value):
    """Encode deterministic UTF-8 JSON. Reject nonfinite numbers."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def input_paths(settings, recipe):
    return [(role, Path(path)) for role, path in (
        ("source", settings.input), ("recipe", recipe), ("mask", settings.mask),
        ("include-mask", settings.include_mask), ("exclude-mask", settings.exclude_mask),
    ) if path is not None]


def snapshot_inputs(settings, recipe, snapshots):
    """Hash captured recipe and raster bytes. Raster decoder consumes same snapshots."""
    inputs = []
    for role, path in input_paths(settings, recipe):
        if path not in snapshots:
            snapshots[path] = path.read_bytes()
        data = snapshots[path]
        inputs.append({"role": role, "path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return inputs


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Missing {name} package metadata. Install st-controlnet-helpers[vector].") from exc


def manifest_bytes(settings, prepared, normalized, inputs, files, diagnostics, *, upstream_args=None, preview=None):
    """Describe consumed inputs and exact output bytes. Manifest excludes its own hash.

    preview holds renderer and font provenance when --preview ran, else null.
    """
    preparation = {
        name: getattr(settings, name)
        for name in ("input_kind", "width_mm", "height_mm", "line_width_mm", "max_res", "invert", "alpha")
    }
    preparation.update({
        "include_mask": str(settings.include_mask) if settings.include_mask is not None else None,
        "exclude_mask": str(settings.exclude_mask) if settings.exclude_mask is not None else None,
        "original_size": prepared.original_size,
        "oriented_size": prepared.oriented_size,
        "processed_size": prepared.processed_size,
        "expansion": asdict(prepared.expansion) if prepared.expansion else None,
    })
    if settings.input_kind == "image":
        preparation["silhouette"] = {
            "method": settings.silhouette_method,
            "mask": str(settings.mask) if settings.mask is not None else None,
            "threshold": settings.threshold,
        }
    return json_bytes({
        "schema_version": 1,
        "versions": {"wrapper": package_version("st-controlnet-helpers"), "vtracer": package_version("vtracer")},
        "inputs": inputs,
        "canvas": asdict(prepared.canvas),
        "preparation": preparation,
        "vtracer": dict(settings.vtracer),
        "upstream": upstream_args,
        "svg_limits": asdict(settings.svg_limits),
        "layers": [{"id": layer_id(settings), "height_mm": None}],
        "artifacts": [
            {"path": path.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for path, data in sorted(files.items())
        ],
        "counts": asdict(normalized.metrics),
        "warnings": diagnostics,
        "preview": preview,
    })


def layer_id(settings):
    """Image mode traces the silhouette role. Mask and edge modes trace one standalone layer."""
    return "silhouette" if settings.input_kind == "image" else "standalone"


def debug_files(bundle, settings, prepared):
    """Save final mask and replay recipe. Original preparation must not run twice."""
    from PIL import Image

    buffer = BytesIO()
    with Image.fromarray(prepared.material.astype("uint8") * 255) as image:
        image.save(buffer, format="PNG")
    recipe = {
        "schema_version": 1, "input": "mask.png", "input_kind": "mask",
        "width_mm": prepared.canvas.width_mm, "invert": False, "alpha": False,
        "vtracer": dict(settings.vtracer), **asdict(settings.svg_limits),
    }
    return {
        bundle.relative(bundle.debug_directory / "mask.png"): buffer.getvalue(),
        bundle.relative(bundle.debug_directory / "recipe.json"): json_bytes(recipe),
    }


class Bundle:
    """Own fixed output names. Completion marker publishes last."""

    def __init__(self, destination, *, debug=False, preview=False, overwrite=False, inputs=()):
        self.svg = Path(destination)
        self.manifest = self.svg.with_suffix(".vector.json")
        self.preview = self.svg.with_suffix(".preview.png")
        self.debug_directory = self.svg.with_suffix(".debug")
        self.debug = debug
        self.overwrite = overwrite
        self.inputs = tuple(inputs)
        self.members = [self.svg, self.manifest]
        if preview:
            self.members.append(self.preview)
        if debug:
            self.members.extend(self.debug_directory / name for name in ("mask.png", "recipe.json"))

    def relative(self, path):
        return path.relative_to(self.svg.parent)

    def check(self):
        """Refuse collisions and input aliases before processing or staging."""
        if self.debug and self.debug_directory.is_symlink():
            raise ConfigError(f"Debug directory {self.debug_directory} must not be a symlink.")
        if self.debug and self.debug_directory.exists() and not self.debug_directory.is_dir():
            raise ConfigError(f"Debug path {self.debug_directory} must be a directory.")
        for target in self.members:
            for role, source in self.inputs:
                if _same_path(target, Path(source)):
                    raise ConfigError(f"Destination {target} is the same file as the {role} {source}. Choose another destination.")
            if target.is_dir():
                raise ConfigError(f"Output {target} names a directory.")
            if os.path.lexists(target) and not self.overwrite:
                raise ConfigError(f"{target} exists. Use --overwrite to replace this bundle.")
        for index, target in enumerate(self.members):
            if any(_same_path(target, other) for other in self.members[:index]):
                raise ConfigError(f"Bundle output {target} aliases another output.")

    def publish(self, files, manifest=None):
        """Stage bytes before invalidation. Failed publication cannot create completion marker."""
        owned = {self.relative(path) for path in self.members if path != self.manifest}
        if not set(files) <= owned:
            raise ValueError("Publication contains a path not owned by this bundle.")
        self.check()
        self.svg.parent.mkdir(parents=True, exist_ok=True)
        completed = False
        try:
            with tempfile.TemporaryDirectory(prefix=f".{self.svg.name}.stage-", dir=self.svg.parent) as directory:
                payloads = list(files.items())
                if manifest is not None:
                    payloads.append((self.relative(self.manifest), manifest))
                staged = []
                for index, (relative, data) in enumerate(payloads):
                    path = Path(directory) / str(index)
                    path.write_bytes(data)
                    staged.append((path, self.svg.parent / relative))
                if self.debug:
                    self.debug_directory.mkdir(exist_ok=True)
                self.check()
                if self.overwrite:
                    self.manifest.unlink(missing_ok=True)
                for source, target in staged:
                    if self.overwrite:
                        os.replace(source, target)
                    else:
                        _link(source, target)
                    if target == self.manifest:
                        completed = True
                    if not self.overwrite:
                        source.unlink()
        except OSError:
            if completed:
                self.manifest.unlink(missing_ok=True)
            raise


def _same_path(left, right):
    if left.resolve() == right.resolve():
        return True
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def _link(source, target):
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno in (errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS):
            raise OSError(
                exc.errno,
                f"Filesystem cannot publish {target} through hard links. "
                "Rerun with --overwrite to permit os.replace for owned bundle files. "
                "Input-alias checks still apply.",
            ) from exc
        raise
