"""Functional tests for worker_factory.

The factory dispatches by neutral family: it looks up the canonical CUDA cell
from ``resolved.profile.family_id``, imports ``worker_ref`` lazily, and builds the
worker from ``binding.model_path`` + thawed ``resolved.info``. It never re-detects.
"""

import sys

import pytest

from backends.family_profiles import SD15_PROFILE, SDXL_PROFILE, FamilyProfile
from backends.model_resolution import (
    LocalModelBinding,
    build_resolved,
    hub_ref,
)
from backends.platforms.base import UnsupportedFamilyError
from utils.model_detector import ModelInfo, ModelVariant


def _resolved(profile, *, variant=ModelVariant.SD15, cad=768, checkpoint_variant="sd15"):
    info = ModelInfo(
        path="/host/only",
        variant=variant,
        cross_attention_dim=cad,
        base_arch="unet",
        loader_format="single_file",
        checkpoint_variant=checkpoint_variant,
        scheduler_profile="native",
    )
    return build_resolved(
        model_ref=hub_ref("org/repo", None),
        raw_info=info,
        profile=profile,
        info=info,
    )


class _RecordingWorker:
    instances: list["_RecordingWorker"] = []

    def __init__(self, worker_id, model_path, model_info=None, family_profile=None):
        self.worker_id = worker_id
        self.model_path = model_path
        self.model_info = model_info
        self.family_profile = family_profile
        self.cls_name = type(self).__name__
        _RecordingWorker.instances.append(self)


class _FakeCudaWorkerModule:
    def __init__(self):
        self.DiffusersCudaWorker = self._factory("DiffusersCudaWorker")
        self.DiffusersSDXLCudaWorker = self._factory("DiffusersSDXLCudaWorker")
        self.DiffusersHunyuanDiTCudaWorker = self._factory("DiffusersHunyuanDiTCudaWorker")

    @staticmethod
    def _factory(name):
        def make(worker_id, model_path, model_info=None, family_profile=None):
            worker = _RecordingWorker(worker_id, model_path, model_info, family_profile)
            worker.cls_name = name
            return worker

        return make


@pytest.fixture
def fake_cuda_worker(monkeypatch):
    _RecordingWorker.instances = []
    module = _FakeCudaWorkerModule()
    monkeypatch.setitem(sys.modules, "backends.cuda_worker", module)
    return module


def test_sd15_family_builds_the_sd15_worker(fake_cuda_worker):
    from backends.worker_factory import create_cuda_worker

    resolved = _resolved(SD15_PROFILE)
    binding = LocalModelBinding("/node/local/sd15")
    worker = create_cuda_worker(0, resolved, binding)

    assert worker.cls_name == "DiffusersCudaWorker"
    assert worker.model_path == "/node/local/sd15"
    # model_info is thawed from the snapshot, rebinding the node-local path.
    assert worker.model_info.path == "/node/local/sd15"
    assert worker.model_info.base_arch == "unet"


def test_sdxl_family_builds_the_sdxl_worker(fake_cuda_worker):
    from backends.worker_factory import create_cuda_worker

    resolved = _resolved(SDXL_PROFILE, variant=ModelVariant.SDXL_BASE, cad=2048,
                         checkpoint_variant="sdxl-base")
    worker = create_cuda_worker(3, resolved, LocalModelBinding("/node/local/sdxl"))

    assert worker.cls_name == "DiffusersSDXLCudaWorker"
    assert worker.worker_id == 3


def test_factory_threads_resolved_profile_not_subclass_default(fake_cuda_worker):
    from backends.worker_factory import create_cuda_worker

    # A distinct profile object (still the sd15 family so it maps to the sd15 cell).
    # If the factory relied on the subclass default it would NOT be this object.
    custom = FamilyProfile(
        family_id="sd15",
        encoder_roles=("text_encoder",),
        pooled_required=False,
        pooled_projection_role=None,
        control_image_kwarg="image",
    )
    resolved = _resolved(custom)
    worker = create_cuda_worker(0, resolved, LocalModelBinding("/node/x"))

    assert worker.family_profile is custom
    assert worker.family_profile is resolved.profile


def test_hunyuandit_family_builds_the_hunyuandit_worker(fake_cuda_worker):
    from backends.worker_factory import create_cuda_worker
    from backends.family_profiles import HUNYUANDIT_PROFILE

    resolved = _resolved(HUNYUANDIT_PROFILE, variant=ModelVariant.UNKNOWN, cad=None,
                         checkpoint_variant="hunyuandit")
    worker = create_cuda_worker(1, resolved, LocalModelBinding("/node/local/hunyuan"))

    assert worker.cls_name == "DiffusersHunyuanDiTCudaWorker"
    assert worker.model_path == "/node/local/hunyuan"
    # The Hunyuan family flows from the resolved model, not a subclass default.
    assert worker.family_profile is HUNYUANDIT_PROFILE


def test_factory_never_calls_detect_model(fake_cuda_worker, monkeypatch):
    import utils.model_detector as detector

    monkeypatch.setattr(
        detector, "detect_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("factory re-detected")),
    )
    from backends.worker_factory import create_cuda_worker

    create_cuda_worker(0, _resolved(SD15_PROFILE), LocalModelBinding("/node/x"))


def test_worker_ref_resolves_only_inside_create_cuda_worker(monkeypatch):
    # Importing the bindings table must not import the CUDA worker module.
    monkeypatch.delitem(sys.modules, "backends.cuda_worker", raising=False)
    import importlib

    import backends.platforms.cuda_bindings as cb
    importlib.reload(cb)
    assert "backends.cuda_worker" not in sys.modules


def test_known_family_without_platform_binding_is_unsupported(fake_cuda_worker):
    from backends.worker_factory import create_cuda_worker

    orphan = FamilyProfile(
        family_id="pixart",  # valid-looking family, but no CUDA cell
        encoder_roles=("text_encoder",),
        pooled_required=False,
        pooled_projection_role=None,
        control_image_kwarg="control_image",
    )
    resolved = _resolved(orphan, variant=ModelVariant.UNKNOWN, cad=None,
                         checkpoint_variant="unknown")
    with pytest.raises(UnsupportedFamilyError):
        create_cuda_worker(0, resolved, LocalModelBinding("/node/x"))


def test_model_info_to_dict_includes_recommended_size():
    model_info = ModelInfo(
        path="/models/checkpoints/sdxl-base.safetensors",
        variant=ModelVariant.SDXL_BASE,
        recommended_size="896x1152",
    )
    assert model_info.to_dict()["recommended_size"] == "896x1152"
