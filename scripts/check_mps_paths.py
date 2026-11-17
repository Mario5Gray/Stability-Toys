"""Check paths that the native MPS server needs before startup.

A missing model file is a warning. The server starts. Processing is refused
for a mode with a missing model until a switch lands on a present model.
Config errors stay fatal: missing modes.yml, invalid YAML, missing model_root,
and a missing storage directory.
"""

import os
from pathlib import Path

import yaml


def main() -> int:
    missing = 0
    warnings = 0

    def check(label: str, value: str, *, kind: str = "path", warn: bool = False) -> bool:
        nonlocal missing, warnings
        path = Path(value).expanduser()
        exists = path.is_dir() if kind == "directory" else path.is_file() if kind == "file" else path.exists()
        if exists:
            print(f"OK {label}: {path}")
            return True
        if warn:
            warnings += 1
            print(f"WARN {label}: {path}")
            return False
        missing += 1
        print(f"MISSING {label}: {path}")
        return False

    config_dir = Path(os.environ.get("MODE_CONFIG_PATH", "conf")).expanduser()
    modes_path = config_dir / "modes.yml"
    modes = {}
    if check("mode configuration", str(modes_path), kind="file"):
        try:
            modes = yaml.safe_load(modes_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            print(f"INVALID mode configuration: {exc}")
            return 1
        model_root = modes.get("model_root")
        if model_root:
            check("model root", str(model_root), kind="directory")
            for mode_name, mode in (modes.get("modes") or {}).items():
                model = mode.get("model")
                if not model:
                    continue
                model_path = Path(model).expanduser()
                if not model_path.is_absolute():
                    model_path = Path(model_root).expanduser() / model_path
                label = "default model" if mode_name == modes.get("default_mode") else f"mode {mode_name} model"
                check(label, str(model_path), warn=True)

    legacy_model_root = os.environ.get("MODEL_ROOT")
    if legacy_model_root and legacy_model_root != str(modes.get("model_root", "")):
        check("MODEL_ROOT", legacy_model_root, kind="directory")

    if os.environ.get("CONTROLNET_REGISTRY_VALIDATION", "strict").strip().lower() == "strict":
        registry_path = os.environ.get("CONTROLNET_REGISTRY_PATH") or str(config_dir / "controlnets.yaml")
        if check("ControlNet registry", registry_path, kind="file"):
            try:
                registry = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                print(f"INVALID ControlNet registry: {exc}")
                return 1
            for model_id, spec in (registry.get("models") or {}).items():
                check(f"ControlNet model {model_id}", str(spec["path"]), warn=True)

    providers = (os.environ.get("STORAGE_PROVIDER", "DISABLED"), os.environ.get("ASSET_STORE_PROVIDER", "DISABLED"))
    if any(provider.upper() in {"FS", "FILESYSTEM"} for provider in providers):
        storage_dir = os.environ.get("FS_STORAGE_DIR") or "/data/image-cache"
        check("storage directory", storage_dir, kind="directory")

    if missing:
        print("Create or configure missing paths before starting the MPS server.")
        return 1
    if warnings:
        print(
            "Server will start. Processing is refused for a mode with a missing model "
            "until a switch lands on a present model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
