import pytest
from types import SimpleNamespace

from server.mode_config import (
    ControlNetControlTypePolicy,
    ControlNetPolicy,
)


def test_registry_loads_local_controlnet_specs(tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    model_dir = tmp_path / "models" / "sdxl-canny"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  sdxl-canny:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )

    from server.controlnet_registry import load_controlnet_registry

    registry = load_controlnet_registry(config_path=str(config_path), validation_mode="strict")
    spec = registry.get_required("sdxl-canny")
    assert spec.model_id == "sdxl-canny"
    assert spec.control_types == ("canny",)
    assert spec.compatible_with == ("sdxl",)


def test_registry_rejects_unknown_compatible_with_family(tmp_path):
    """A typo'd compatible_with family must fail at load, not silently later."""
    config_path = tmp_path / "controlnets.yaml"
    model_dir = tmp_path / "models" / "x"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  bad-family:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl, sdxxl]\n",  # 'sdxxl' is not a known family
        encoding="utf-8",
    )

    from server.controlnet_registry import load_controlnet_registry

    with pytest.raises(Exception) as exc:
        load_controlnet_registry(config_path=str(config_path), validation_mode="strict")
    assert "sdxxl" in str(exc.value)


def test_registry_rejects_missing_local_path_in_strict_mode(tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    config_path.write_text(
        "models:\n"
        "  sdxl-depth:\n"
        "    path: /does/not/exist\n"
        "    control_types: [depth]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )

    from server.controlnet_registry import load_controlnet_registry

    with pytest.raises(ValueError, match="does not exist"):
        load_controlnet_registry(config_path=str(config_path), validation_mode="strict")


def test_strict_registry_validation_runs_at_startup(monkeypatch, tmp_path):
    bad_config = tmp_path / "controlnets.yaml"
    bad_config.write_text(
        "models:\n"
        "  broken:\n"
        "    path: /does/not/exist\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTROLNET_REGISTRY_PATH", str(bad_config))
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "strict")

    from server.controlnet_registry import reset_controlnet_registry
    from server.lcm_sr_server import _validate_controlnet_registry_for_startup

    reset_controlnet_registry()
    with pytest.raises(ValueError, match="does not exist"):
        _validate_controlnet_registry_for_startup()


def _sdxl_detect(path: str):
    """A detector stub whose facts resolve to the sdxl family (base_arch unet, CAD 2048)."""
    from utils.model_detector import ModelInfo, ModelVariant

    return ModelInfo(
        path=path,
        variant=ModelVariant.SDXL_BASE,
        cross_attention_dim=2048,
        base_arch="unet",
    )


def _mode_config_stub(*modes):
    by_name = {mode.name: mode for mode in modes}

    class _StubModeConfig:
        def list_modes(self):
            return list(by_name)

        def get_mode(self, name):
            return by_name[name]

    return _StubModeConfig()


def _controlnet_mode(*, name: str, checkpoint_variant: str, control_type: str, model_id: str):
    return SimpleNamespace(
        name=name,
        model_path=f"/tmp/{name}.safetensors",
        checkpoint_variant=checkpoint_variant,
        controlnet_policy=ControlNetPolicy(
            enabled=True,
            max_attachments=1,
            allow_reuse_emitted_maps=True,
            allowed_control_types={
                control_type: ControlNetControlTypePolicy(
                    default_model_id=model_id,
                    allowed_model_ids=[model_id],
                )
            },
        ),
    )


def test_strict_registry_validation_rejects_unknown_mode_model_id(monkeypatch, tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    model_dir = tmp_path / "models" / "sdxl-canny"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  sdxl-canny:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTROLNET_REGISTRY_PATH", str(config_path))
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "strict")

    from server.controlnet_registry import reset_controlnet_registry
    from server import lcm_sr_server

    reset_controlnet_registry()
    mode_config = _mode_config_stub(
        _controlnet_mode(
            name="sdxl-mode",
            checkpoint_variant="sdxl-base",
            control_type="canny",
            model_id="missing-model",
        )
    )

    monkeypatch.setattr(lcm_sr_server, "get_mode_config", lambda: mode_config)
    monkeypatch.setattr("utils.model_detector.detect_model", _sdxl_detect)
    with pytest.raises(ValueError, match="unknown ControlNet model_id 'missing-model'"):
        lcm_sr_server._validate_controlnet_registry_for_startup()


def test_strict_registry_validation_rejects_incompatible_mode_model_id(monkeypatch, tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    model_dir = tmp_path / "models" / "sd15-canny"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  sd15-canny:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sd15]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTROLNET_REGISTRY_PATH", str(config_path))
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "strict")

    from server.controlnet_registry import reset_controlnet_registry
    from server import lcm_sr_server

    reset_controlnet_registry()
    mode_config = _mode_config_stub(
        _controlnet_mode(
            name="sdxl-mode",
            checkpoint_variant="sdxl-base",
            control_type="canny",
            model_id="sd15-canny",
        )
    )

    monkeypatch.setattr(lcm_sr_server, "get_mode_config", lambda: mode_config)
    monkeypatch.setattr("utils.model_detector.detect_model", _sdxl_detect)
    with pytest.raises(ValueError, match="incompatible with mode family 'sdxl'"):
        lcm_sr_server._validate_controlnet_registry_for_startup()


def test_conflicting_checkpoint_variant_cannot_alter_detected_family(monkeypatch, tmp_path):
    """checkpoint_variant is not authoritative: the detected family (sdxl) governs
    ControlNet compatibility even when the mode declares checkpoint_variant=sd15."""
    config_path = tmp_path / "controlnets.yaml"
    model_dir = tmp_path / "models" / "sd15-canny"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  sd15-canny:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sd15]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTROLNET_REGISTRY_PATH", str(config_path))
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "strict")

    from server.controlnet_registry import reset_controlnet_registry
    from server import lcm_sr_server

    reset_controlnet_registry()
    mode_config = _mode_config_stub(
        _controlnet_mode(
            name="lying-mode",
            checkpoint_variant="sd15",  # conflicts with the detected sdxl family
            control_type="canny",
            model_id="sd15-canny",
        )
    )

    monkeypatch.setattr(lcm_sr_server, "get_mode_config", lambda: mode_config)
    monkeypatch.setattr("utils.model_detector.detect_model", _sdxl_detect)
    with pytest.raises(ValueError, match="incompatible with mode family 'sdxl'"):
        lcm_sr_server._validate_controlnet_registry_for_startup()


@pytest.mark.asyncio
async def test_lazy_registry_validation_skips_startup_hook(monkeypatch):
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "lazy")

    from server import lcm_sr_server

    fake_runtime = SimpleNamespace(_pool=None, _service=None, shutdown=lambda: None)
    fake_provider = SimpleNamespace(
        backend_id="cpu",
        create_generation_runtime=lambda **_: fake_runtime,
        create_superres_runtime=lambda settings: None,
    )
    fake_sr_settings = SimpleNamespace(enabled=False)
    fake_storage_provider = SimpleNamespace(close=lambda: None)
    fake_task = SimpleNamespace(cancel=lambda: None)

    monkeypatch.setattr(
        lcm_sr_server,
        "_validate_controlnet_registry_for_startup",
        lambda: (_ for _ in ()).throw(AssertionError("startup validation should be skipped")),
    )
    monkeypatch.setattr(lcm_sr_server, "get_backend_provider", lambda: fake_provider)
    monkeypatch.setattr(lcm_sr_server, "load_superres_runtime_settings", lambda env: fake_sr_settings)
    monkeypatch.setattr(
        lcm_sr_server.StorageProvider,
        "make_storage_provider_from_env",
        staticmethod(lambda: fake_storage_provider),
    )
    monkeypatch.setattr(lcm_sr_server, "register_job_hook", lambda: None)
    monkeypatch.setattr(
        lcm_sr_server.asyncio,
        "create_task",
        lambda coro: (coro.close(), fake_task)[1],
    )

    app = SimpleNamespace(state=SimpleNamespace())
    async with lcm_sr_server.lifespan(app):
        assert app.state.generation_runtime is fake_runtime


def test_registry_defaults_to_mode_config_path(monkeypatch, tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_path = config_dir / "controlnets.yaml"
    model_dir = tmp_path / "models" / "sdxl-canny"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  sdxl-canny:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODE_CONFIG_PATH", str(config_dir))
    monkeypatch.delenv("CONTROLNET_REGISTRY_PATH", raising=False)

    from server.controlnet_registry import get_controlnet_registry, reset_controlnet_registry

    reset_controlnet_registry()
    registry = get_controlnet_registry()

    assert registry.get_required("sdxl-canny").path == str(model_dir)


def test_production_registry_has_hunyuandit_canny_compatible_only_with_hunyuandit():
    """Task 10: the production controlnets.yaml carries one Hunyuan Canny entry,
    resolvable as compatible only with the hunyuandit family (never the SD
    families). Loaded lazily so local /models paths are not required here."""
    from pathlib import Path

    from server.controlnet_registry import load_controlnet_registry

    conf_path = Path(__file__).resolve().parents[1] / "conf" / "controlnets.yaml"
    registry = load_controlnet_registry(config_path=str(conf_path), validation_mode="lazy")

    spec = registry.get("hunyuandit-canny")
    assert spec is not None
    assert spec.control_types == ("canny",)
    assert spec.compatible_with == ("hunyuandit",)
    # Compatible only with hunyuandit — never advertised for the SD families.
    assert "sd15" not in spec.compatible_with
    assert "sdxl" not in spec.compatible_with
