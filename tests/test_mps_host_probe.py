"""MPS host probe contract tests (STABL-npwtkfpf).

Patch torch.backends.mps and torch.mps. Never remove real torch from
sys.modules. These tests prove the probe contract on any host; they are not
live MPS evidence.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "mps_host_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("mps_host_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

VALUES = [0.673828125, 0.25, 0.125, 0.0625]
MIB_4 = 4 * 1024 * 1024


class _FakeTensor:
    def __init__(self, values, dtype):
        self._values = list(values)
        self.dtype = dtype
        self.device = torch.device("mps")

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeGenerator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        return self


def _fake_rand(*shape, generator, device, dtype):
    return _FakeTensor(VALUES, dtype)


def _fake_zeros(*shape, dtype, device):
    return _FakeTensor([], dtype)


@pytest.fixture
def clean_env():
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("CUDA_") and name != "BACKEND"
    }
    with patch.dict(os.environ, env, clear=True):
        yield


@pytest.fixture
def mps_host(clean_env):
    from backends.platforms.mps_policy import MPS_TORCH_POLICY

    stats = Mock(side_effect=[(1000, 2000), (1000 + MIB_4, 2000 + MIB_4), (1000, 2000)])
    with (
        patch("torch.backends.mps.is_built", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.Generator", _FakeGenerator),
        patch("torch.rand", _fake_rand),
        patch("torch.zeros", _fake_zeros),
        patch.object(MPS_TORCH_POLICY, "memory_stats", stats),
        patch.object(MPS_TORCH_POLICY, "release_cache", Mock()),
    ):
        yield stats


def test_probe_passes_on_ready_host_with_stable_schema(mps_host):
    code, report = probe.run_probe()

    assert code == probe.EXIT_PASS == 0
    assert report["status"] == "pass"
    assert report["exit_code"] == 0
    assert set(report) == {
        "schema",
        "status",
        "exit_code",
        "commit",
        "dirty",
        "host",
        "torch",
        "checks",
    }
    assert report["schema"] == "mps_host_probe/v1"
    assert set(report["host"]) == {"system", "release", "mac_ver", "machine", "python"}
    assert report["torch"] == {
        "version": torch.__version__,
        "mps_built": True,
        "mps_available": True,
    }
    checks = report["checks"]
    assert set(checks) == {
        "policy",
        "seeded_tensor",
        "memory",
        "provider",
        "unsupported_family",
    }
    assert set(checks["policy"]) == {
        "status",
        "error",
        "device",
        "dtype",
        "attention_slicing",
    }
    assert set(checks["seeded_tensor"]) == {
        "status",
        "error",
        "seed",
        "shape",
        "dtype",
        "device",
        "values",
        "repeatable_in_process",
        "matches_expected",
    }
    assert set(checks["memory"]) == {
        "status",
        "error",
        "alloc_bytes",
        "before",
        "after_alloc",
        "after_release",
    }
    assert set(checks["provider"]) == {"status", "error", "backend", "provider"}
    assert set(checks["unsupported_family"]) == {"status", "error", "family", "message"}
    assert all(check["status"] == "pass" for check in checks.values())
    assert all(check["error"] is None for check in checks.values())

    assert checks["policy"]["device"] == "mps"
    assert checks["policy"]["dtype"] == "torch.float16"
    assert checks["seeded_tensor"]["values"] == VALUES
    assert checks["seeded_tensor"]["device"] == "mps"
    assert checks["seeded_tensor"]["dtype"] == "torch.float16"
    assert checks["seeded_tensor"]["repeatable_in_process"] is True
    assert checks["seeded_tensor"]["matches_expected"] is None
    assert checks["memory"]["alloc_bytes"] == MIB_4
    assert checks["memory"]["after_alloc"] == {
        "current_allocated": 1000 + MIB_4,
        "driver_allocated": 2000 + MIB_4,
    }
    assert checks["provider"] == {
        "status": "pass",
        "error": None,
        "backend": "mps",
        "provider": "MPSProvider",
    }
    assert (
        checks["unsupported_family"]["message"]
        == "family 'sdxl' has no MPS platform binding"
    )
    json.dumps(report)


@pytest.mark.parametrize("missing", ["is_built", "is_available"])
def test_unavailable_runtime_is_never_a_pass(mps_host, missing):
    with patch(f"torch.backends.mps.{missing}", return_value=False):
        code, report = probe.run_probe()

    assert code == probe.EXIT_UNAVAILABLE
    assert code not in (probe.EXIT_PASS, probe.EXIT_FAILED)
    assert report["status"] == "unavailable"
    checks = report["checks"]
    for name in ("policy", "seeded_tensor", "memory"):
        assert checks[name]["status"] == "skipped"
    assert checks["provider"]["status"] == "pass"
    assert checks["unsupported_family"]["status"] == "pass"
    mps_host.assert_not_called()


def test_failed_check_gives_failure_code(mps_host):
    mps_host.side_effect = [(1000, 2000), (1000, 2000), (1000, 2000)]

    code, report = probe.run_probe()

    assert code == probe.EXIT_FAILED
    assert code not in (probe.EXIT_PASS, probe.EXIT_UNAVAILABLE)
    assert report["status"] == "failed"
    assert report["checks"]["memory"]["status"] == "fail"
    assert "allocation" in report["checks"]["memory"]["error"]


def test_failed_check_outranks_unavailable(mps_host):
    import backends.platforms.mps_bindings as mps_bindings

    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch.dict(mps_bindings.MPS_FAMILY_BINDINGS, {"sdxl": object()}),
        patch("backends.worker_factory.create_bound_worker", return_value=object()),
    ):
        code, report = probe.run_probe()

    assert code == probe.EXIT_FAILED
    assert report["status"] == "failed"
    assert report["checks"]["unsupported_family"]["status"] == "fail"


def test_policy_rejection_is_a_failed_check(mps_host):
    with patch.dict(os.environ, {"CUDA_DTYPE": "bf16"}):
        code, report = probe.run_probe()

    assert code == probe.EXIT_FAILED
    assert report["checks"]["policy"]["status"] == "fail"
    assert "CUDA_DTYPE" in report["checks"]["policy"]["error"]


def test_expected_values_mismatch_fails(mps_host, tmp_path):
    expected = tmp_path / "previous.json"
    expected.write_text(json.dumps({"checks": {"seeded_tensor": {"values": [0.5]}}}))

    code, report = probe.run_probe(expected_values=probe.load_expected_values(expected))

    assert code == probe.EXIT_FAILED
    assert report["checks"]["seeded_tensor"]["status"] == "fail"
    assert report["checks"]["seeded_tensor"]["matches_expected"] is False


def test_expected_values_match_passes(mps_host, tmp_path):
    expected = tmp_path / "previous.json"
    expected.write_text(json.dumps({"checks": {"seeded_tensor": {"values": VALUES}}}))

    code, report = probe.run_probe(expected_values=probe.load_expected_values(expected))

    assert code == probe.EXIT_PASS
    assert report["checks"]["seeded_tensor"]["matches_expected"] is True


def test_probe_restores_backend_env_and_provider(mps_host):
    import backends.platform_registry as platform_registry

    platform_registry.reset_backend_provider()
    with patch.dict(os.environ, {"BACKEND": "cpu"}):
        probe.run_probe()
        assert os.environ["BACKEND"] == "cpu"
    assert platform_registry._provider is None


def test_main_prints_one_json_report_and_returns_code(mps_host, capsys):
    code = probe.main([])

    report = json.loads(capsys.readouterr().out)
    assert code == report["exit_code"] == probe.EXIT_PASS


def test_main_returns_unavailable_code(mps_host, capsys):
    with patch("torch.backends.mps.is_available", return_value=False):
        code = probe.main([])

    report = json.loads(capsys.readouterr().out)
    assert code == report["exit_code"] == probe.EXIT_UNAVAILABLE


def test_probe_import_is_torch_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('p', {str(PROBE_PATH)!r}); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert not {'torch', 'backends'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
