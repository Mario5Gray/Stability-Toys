import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_mps_paths.py"
ROOT = SCRIPT.parent.parent


def _run_preflight(tmp_path, *, create_root_and_store, create_models):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    model_root = tmp_path / "models"
    storage_dir = tmp_path / "store"
    if create_root_and_store:
        model_root.mkdir()
        storage_dir.mkdir()
    if create_models:
        (model_root / "diffusers" / "default").mkdir(parents=True)
        (model_root / "diffusers" / "other").mkdir(parents=True)
        (model_root / "controlnets" / "canny").mkdir(parents=True)
    (config_dir / "modes.yml").write_text(
        f"model_root: {model_root}\n"
        "default_mode: default\n"
        "modes:\n"
        "  default:\n"
        "    model: diffusers/default\n"
        "  other:\n"
        "    model: diffusers/other\n",
        encoding="utf-8",
    )
    (config_dir / "controlnets.yaml").write_text(
        "models:\n"
        "  canny:\n"
        f"    path: {model_root / 'controlnets' / 'canny'}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    for name in ("CONTROLNET_REGISTRY_PATH", "CONTROLNET_REGISTRY_VALIDATION", "ASSET_STORE_PROVIDER"):
        env.pop(name, None)
    env.update(
        MODE_CONFIG_PATH=str(config_dir),
        MODEL_ROOT=str(model_root),
        STORAGE_PROVIDER="FILESYSTEM",
        FS_STORAGE_DIR=str(storage_dir),
    )
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True), model_root, storage_dir


def test_preflight_warns_about_missing_model_files_and_still_passes(tmp_path):
    result, model_root, _ = _run_preflight(
        tmp_path, create_root_and_store=True, create_models=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"WARN default model: {model_root / 'diffusers' / 'default'}" in result.stdout
    assert f"WARN mode other model: {model_root / 'diffusers' / 'other'}" in result.stdout
    assert f"WARN ControlNet model canny: {model_root / 'controlnets' / 'canny'}" in result.stdout
    assert "MISSING" not in result.stdout
    assert "Server will start." in result.stdout


def test_preflight_fails_on_missing_model_root_and_storage(tmp_path):
    result, model_root, storage_dir = _run_preflight(
        tmp_path, create_root_and_store=False, create_models=False
    )

    assert result.returncode == 1
    assert f"MISSING model root: {model_root}" in result.stdout
    assert f"MISSING storage directory: {storage_dir}" in result.stdout
    assert "WARN default model:" in result.stdout
    assert "Create or configure missing paths before starting the MPS server." in result.stdout


def test_preflight_fails_on_missing_mode_configuration(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()

    env = os.environ.copy()
    env.update(MODE_CONFIG_PATH=str(config_dir), STORAGE_PROVIDER="DISABLED")
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "MISSING mode configuration:" in result.stdout


def test_preflight_accepts_existing_paths(tmp_path):
    result, model_root, storage_dir = _run_preflight(
        tmp_path, create_root_and_store=True, create_models=True
    )

    assert result.returncode == 0, result.stderr
    assert f"OK model root: {model_root}" in result.stdout
    assert f"OK storage directory: {storage_dir}" in result.stdout
    assert "MISSING" not in result.stdout
    assert "WARN" not in result.stdout


def test_env_mps_uses_host_paths_and_enables_storage(tmp_path):
    env = os.environ.copy()
    env.pop("STORAGE_PROVIDER", None)
    env["MODE_CONFIG_PATH"] = str(tmp_path / "conf")
    env["FS_STORAGE_DIR"] = str(tmp_path / "store")

    result = subprocess.run(
        ["sh", "-c", '. ./env.mps; printf "%s\\n%s\\n%s\\n" "$MODE_CONFIG_PATH" "$FS_STORAGE_DIR" "$STORAGE_PROVIDER"'],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [env["MODE_CONFIG_PATH"], env["FS_STORAGE_DIR"], "FILESYSTEM"]
