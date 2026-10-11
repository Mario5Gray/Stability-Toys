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

from vector_map_config import LAYER_ROLES, ConfigError


def json_bytes(value):
    """Encode deterministic UTF-8 JSON. Reject nonfinite numbers."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def input_paths(settings, recipe):
    """Every consumed file with its input role. A shared path appears once per role."""
    paths = [
        ("source", settings.input), ("recipe", recipe), ("mask", settings.mask),
        ("include-mask", settings.include_mask), ("exclude-mask", settings.exclude_mask),
    ]
    for request in settings.role_requests:
        paths.extend((f"{request.role}-{name}", path) for name, path in (
            ("map", request.path), ("include-mask", request.include_mask), ("exclude-mask", request.exclude_mask),
        ))
    return [(role, Path(path)) for role, path in paths if path is not None]


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
    return json_bytes({
        **_common(settings, prepared, inputs, upstream_args),
        "layers": [{"id": layer_id(settings), "height_mm": None}],
        "artifacts": _artifacts(files),
        "counts": asdict(normalized.metrics),
        "warnings": diagnostics,
        "preview": preview,
    })


def layered_manifest_bytes(settings, prepared, normalized, inputs, files, diagnostics, *, combined, layer_files,
                           upstream_args=None, preview=None):
    """Describe a layered bundle. STABL-qlagdbmh, spec 8.

    combined and layer_files are bundle-relative paths of the combined SVG and of each published layer.
    Every selected role has a layer entry. An empty optional role has svg null and no artifact.
    Layer heights and modes are S3.4 values, so they are null here.
    """
    layers = []
    for role in prepared.roles:
        layer = normalized.get(role)
        layers.append({
            "id": role, "height_mm": None, "mode": None,
            "svg": layer_files[role].as_posix() if layer else None,
            "counts": asdict(layer.metrics) if layer else None,
        })
    requests = {request.role: request for request in settings.role_requests}
    roles = {}
    for role in prepared.roles[1:]:
        candidate = getattr(prepared, role)
        request = requests[role]
        roles[role] = {
            "source": candidate.source,
            "path": _path(request.path),
            "canny": asdict(candidate.canny) if candidate.canny else None,
            "width_mm": request.width_mm,
            "gap_close_mm": request.gap_close_mm,
            "include_mask": _path(request.include_mask),
            "exclude_mask": _path(request.exclude_mask),
            "expansion": asdict(candidate.expansion) if candidate.expansion else None,
            "gap_closing": asdict(candidate.gap_closing) if candidate.gap_closing else None,
        }
    metrics = [layer.metrics for layer in normalized.values()]
    counts = {name: sum(getattr(item, name) for item in metrics)
              for name in ("raw_bytes", "normalized_bytes", "paths", "commands", "subpaths", "degenerate_subpaths")}
    counts.update(layers=len(normalized), combined_bytes=len(files[combined]))
    common = _common(settings, prepared.silhouette, inputs, upstream_args)
    common["preparation"]["layers"] = list(settings.layers)
    return json_bytes({
        **common,
        "roles": roles,
        "layers": layers,
        "artifacts": _artifacts(files),
        "counts": counts,
        "warnings": diagnostics,
        "preview": preview,
    })


def _path(path):
    return str(path) if path is not None else None


def _artifacts(files):
    return [
        {"path": path.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for path, data in sorted(files.items())
    ]


def _common(settings, prepared, inputs, upstream_args):
    """Manifest fields shared by standalone and layered bundles."""
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
    return {
        "schema_version": 1,
        "versions": {"wrapper": package_version("st-controlnet-helpers"), "vtracer": package_version("vtracer")},
        "inputs": inputs,
        "canvas": asdict(prepared.canvas),
        "preparation": preparation,
        "vtracer": dict(settings.vtracer),
        "upstream": upstream_args,
        "svg_limits": asdict(settings.svg_limits),
    }


def layer_id(settings):
    """Image mode traces the silhouette role. Mask and edge modes trace one standalone layer."""
    return "silhouette" if settings.input_kind == "image" else "standalone"


def _png(material):
    from PIL import Image

    buffer = BytesIO()
    with Image.fromarray(material.astype("uint8") * 255) as image:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def debug_files(bundle, settings, prepared):
    """Save final mask and replay recipe. Original preparation must not run twice."""
    recipe = {
        "schema_version": 1, "input": "mask.png", "input_kind": "mask",
        "width_mm": prepared.canvas.width_mm, "invert": False, "alpha": False,
        "vtracer": dict(settings.vtracer), **asdict(settings.svg_limits),
    }
    return {
        bundle.relative(bundle.debug_directory / "mask.png"): _png(prepared.material),
        bundle.relative(bundle.debug_directory / "recipe.json"): json_bytes(recipe),
    }


def layered_debug_files(bundle, settings, prepared):
    """Save each selected prepared mask and a layered replay recipe. STABL-qlagdbmh.

    The recipe replays the saved masks, not the original preparation: the silhouette PNG is the source
    and the silhouette mask, and each optional role reads its PNG as a map. No constraint, gap closing or
    widening runs again, so the replay traces the same masks. An empty role stays selected and empty.
    """
    materials = {"silhouette": prepared.silhouette.material}
    materials.update({role: getattr(prepared, role).material for role in prepared.roles[1:]})
    recipe = {
        "schema_version": 1, "input": "silhouette.png", "input_kind": "image", "mask": "silhouette.png",
        "width_mm": prepared.canvas.width_mm, "invert": False, "alpha": False, "layers": list(prepared.roles),
        **{role: {"source": "map", "path": f"{role}.png"} for role in prepared.roles[1:]},
        "vtracer": dict(settings.vtracer), **asdict(settings.svg_limits),
    }
    files = {bundle.relative(bundle.debug_directory / f"{role}.png"): _png(material)
             for role, material in materials.items()}
    files[bundle.relative(bundle.debug_directory / "recipe.json")] = json_bytes(recipe)
    return files


class Bundle:
    """Own fixed output names. Completion marker publishes last."""

    def __init__(self, destination, *, debug=False, preview=False, overwrite=False, inputs=(), layered=False):
        self.svg = Path(destination)
        self.manifest = self.svg.with_suffix(".vector.json")
        self.preview = self.svg.with_suffix(".preview.png")
        self.debug_directory = self.svg.with_suffix(".debug")
        self.layer_directory = self.svg.with_suffix(".layers")
        self.debug = debug
        self.layered = layered
        self.overwrite = overwrite
        self.inputs = tuple(inputs)
        self.members = [self.svg, self.manifest]
        if layered:
            # A layered run owns every fixed role name, selected or not.
            self.members.extend(self.layer_path(role) for role in LAYER_ROLES)
        if preview:
            self.members.append(self.preview)
        # A layered debug bundle owns every fixed role PNG name, selected or not.
        names = [f"{role}.png" for role in LAYER_ROLES] if layered else ["mask.png"]
        self.debug_members = [self.debug_directory / name for name in (*names, "recipe.json")] if debug else []
        self.members.extend(self.debug_members)

    def relative(self, path):
        return path.relative_to(self.svg.parent)

    def layer_path(self, role):
        return self.layer_directory / f"{role}.svg"

    def _directories(self):
        return [(label, path) for label, path, used in (
            ("Debug", self.debug_directory, self.debug), ("Layer", self.layer_directory, self.layered),
        ) if used]

    def check(self):
        """Refuse collisions and input aliases before processing or staging."""
        for label, directory in self._directories():
            if directory.is_symlink():
                raise ConfigError(f"{label} directory {directory} must not be a symlink.")
            if directory.exists() and not directory.is_dir():
                raise ConfigError(f"{label} path {directory} must be a directory.")
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

    def publish(self, files, manifest=None, *, prune=None):
        """Stage bytes before invalidation. Failed publication cannot create completion marker.

        With overwrite, an owned member absent from files is stale. Remove it after the old manifest is
        invalidated and before the new manifest. prune limits removal to these members. The default is
        every member for a complete publication and no member otherwise.
        """
        owned = {self.relative(path) for path in self.members if path != self.manifest}
        if not set(files) <= owned:
            raise ValueError("Publication contains a path not owned by this bundle.")
        if prune is None:
            prune = self.members if manifest is not None else ()
        stale = [path for path in prune if path != self.manifest and self.relative(path) not in files]
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
                for _, owned_directory in self._directories():
                    owned_directory.mkdir(exist_ok=True)
                self.check()
                if self.overwrite:
                    self.manifest.unlink(missing_ok=True)
                for source, target in staged:
                    if target == self.manifest:
                        self._remove(stale)
                    if self.overwrite:
                        os.replace(source, target)
                    else:
                        _link(source, target)
                    if target == self.manifest:
                        completed = True
                    if not self.overwrite:
                        source.unlink()
                if manifest is None:
                    self._remove(stale)
        except OSError:
            if completed:
                self.manifest.unlink(missing_ok=True)
            raise


    def _remove(self, stale):
        """Remove stale owned files. Without overwrite, preflight refused every existing member."""
        if self.overwrite:
            for path in stale:
                path.unlink(missing_ok=True)


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
