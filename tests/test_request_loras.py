"""Request LoRA validation and the derived mode (STABL-fvwsvjvn).

tmp_path/root is lora_root. Files outside the root live in tmp_path/outside.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from backends.mode_loras import LoraSpec, RequestLoraError, derive_request_mode
from server.mode_config import LoRAConfig
from server.request_loras import resolve_request_loras


@pytest.fixture
def root(tmp_path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    return r


@pytest.fixture
def outside(tmp_path) -> Path:
    o = tmp_path / "outside"
    o.mkdir()
    return o


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def _e(path, strength=1.0):
    return SimpleNamespace(path=str(path), strength=strength)


def test_none_and_empty_give_empty_tuple(root):
    assert resolve_request_loras(None, str(root)) == ()
    assert resolve_request_loras([], str(root)) == ()


def test_relative_path_resolves_under_root(root):
    _touch(root / "a.safetensors")

    out = resolve_request_loras([_e("a.safetensors", 0.8)], str(root))

    assert out == (LoraSpec(str((root / "a.safetensors").resolve()), 0.8),)


def test_absolute_path_inside_root_admitted(root):
    f = _touch(root / "sub" / "a.safetensors")

    out = resolve_request_loras([_e(f.resolve())], str(root))

    assert out == (LoraSpec(str(f.resolve()), 1.0),)


def test_absolute_path_outside_root_refused(root, outside):
    f = _touch(outside / "x.safetensors")

    with pytest.raises(RequestLoraError, match="lora_root"):
        resolve_request_loras([_e(f)], str(root))


def test_dotdot_escape_refused(root, tmp_path):
    _touch(tmp_path / "x.safetensors")

    with pytest.raises(RequestLoraError, match="lora_root"):
        resolve_request_loras([_e("../x.safetensors")], str(root))


def test_symlink_escape_refused(root, outside):
    target = _touch(outside / "x.safetensors")
    (root / "link.safetensors").symlink_to(target)

    with pytest.raises(RequestLoraError, match="lora_root"):
        resolve_request_loras([_e("link.safetensors")], str(root))


def test_missing_path_refused(root):
    with pytest.raises(RequestLoraError, match="not found"):
        resolve_request_loras([_e("missing.safetensors")], str(root))


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_non_finite_strength_refused(root, bad):
    _touch(root / "a.safetensors")

    with pytest.raises(RequestLoraError, match="strength"):
        resolve_request_loras([_e("a.safetensors", bad)], str(root))


def test_duplicate_file_refused(root):
    _touch(root / "a.safetensors")

    with pytest.raises(RequestLoraError, match="same file"):
        resolve_request_loras([_e("a.safetensors"), _e("./a.safetensors")], str(root))


def test_directory_symlink_escape_refused(root, outside):
    target = _touch(outside / "w.safetensors")
    (root / "d").mkdir()
    (root / "d" / "w.safetensors").symlink_to(target)

    with pytest.raises(RequestLoraError, match="lora_root"):
        resolve_request_loras([_e("d")], str(root))


def test_directory_and_its_file_duplicate_refused(root):
    _touch(root / "d" / "w.safetensors")

    with pytest.raises(RequestLoraError, match="same file"):
        resolve_request_loras([_e("d"), _e("d/w.safetensors")], str(root))


def test_directory_selects_adapter_model(root):
    _touch(root / "d" / "adapter_model.safetensors")
    _touch(root / "d" / "other.safetensors")

    out = resolve_request_loras([_e("d")], str(root))

    assert out[0].path == str((root / "d" / "adapter_model.safetensors").resolve())


def test_directory_with_two_candidates_refused(root):
    _touch(root / "d" / "a.safetensors")
    _touch(root / "d" / "b.safetensors")

    with pytest.raises(RequestLoraError, match="d"):
        resolve_request_loras([_e("d")], str(root))


def test_directory_with_no_candidate_refused(root):
    (root / "d").mkdir()

    with pytest.raises(RequestLoraError):
        resolve_request_loras([_e("d")], str(root))


def test_spec_path_is_a_file(root):
    _touch(root / "d" / "only.safetensors")
    _touch(root / "b.safetensors")

    out = resolve_request_loras([_e("d"), _e("b.safetensors")], str(root))

    assert all(Path(s.path).is_file() for s in out)


def test_sidecar_symlink_outside_root_refused(root, outside):
    _touch(root / "d" / "adapter_model.safetensors")
    cfg = _touch(outside / "adapter_config.json")
    (root / "d" / "adapter_config.json").symlink_to(cfg)

    with pytest.raises(RequestLoraError, match="adapter_config.json"):
        resolve_request_loras([_e("d")], str(root))


def test_missing_sidecar_allowed(root):
    _touch(root / "a.safetensors")

    assert len(resolve_request_loras([_e("a.safetensors")], str(root))) == 1


def test_in_root_sidecar_allowed(root):
    _touch(root / "d" / "adapter_model.safetensors")
    _touch(root / "d" / "adapter_config.json")

    assert len(resolve_request_loras([_e("d")], str(root))) == 1


def test_default_strength_is_one(root):
    _touch(root / "a.safetensors")

    out = resolve_request_loras([{"path": "a.safetensors"}], str(root))

    assert out[0].strength == 1.0


def _mode(name="plain", loras=None, lora_paths=None):
    return SimpleNamespace(name=name, loras=list(loras or []), lora_paths=list(lora_paths or []))


def test_derive_sets_loras_and_paths_in_order():
    specs = (LoraSpec("/l/b.safetensors", 0.5), LoraSpec("/l/a.safetensors", 0.8))

    m = derive_request_mode(_mode(), specs)

    assert m.lora_paths == ["/l/b.safetensors", "/l/a.safetensors"]
    assert [(l.path, l.strength) for l in m.loras] == [("/l/b.safetensors", 0.5), ("/l/a.safetensors", 0.8)]
    assert all(isinstance(l, LoRAConfig) for l in m.loras)


def test_derive_refuses_mode_with_loras():
    m = _mode("styled", loras=[LoRAConfig(path="x.safetensors")], lora_paths=["/l/x.safetensors"])

    with pytest.raises(RequestLoraError, match="styled.*without loras"):
        derive_request_mode(m, (LoraSpec("/l/a.safetensors", 1.0),))


def test_derive_with_no_specs_keeps_mode_loras():
    m = _mode("styled", loras=[LoRAConfig(path="x.safetensors")], lora_paths=["/l/x.safetensors"])
    before = copy.deepcopy(m)

    out = derive_request_mode(m, ())

    assert out is m
    assert out.lora_paths == before.lora_paths
    assert [l.path for l in out.loras] == [l.path for l in before.loras]
