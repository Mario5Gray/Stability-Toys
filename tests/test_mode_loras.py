"""Mode LoRA specs and the configure_loras hook (STABL-ykowysrg)."""
from __future__ import annotations

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backends.mode_loras import LoraSpec, apply_mode_loras, lora_specs_for_mode


def _mode(loras, paths):
    return SimpleNamespace(
        loras=[SimpleNamespace(path=p, strength=s) for p, s in loras],
        lora_paths=list(paths),
    )


class _HookWorker:
    def __init__(self):
        self.got = None

    def configure_loras(self, specs):
        self.got = specs


def test_specs_pair_paths_and_strength():
    mode = _mode([("a.safetensors", 0.8)], ["/models/loras/a.safetensors"])
    assert lora_specs_for_mode(mode) == [LoraSpec("/models/loras/a.safetensors", 0.8)]


def test_mock_mode_yields_no_specs():
    assert lora_specs_for_mode(Mock()) == []


def test_mode_without_loras_attr_yields_no_specs():
    assert lora_specs_for_mode(SimpleNamespace()) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        lora_specs_for_mode(_mode([("a", 1.0)], []))


def test_apply_refuses_worker_without_hook():
    with pytest.raises(RuntimeError, match="does not support LoRAs"):
        apply_mode_loras(object(), _mode([("a", 1.0)], ["/m/a"]))


def test_apply_calls_hook_with_specs():
    w = _HookWorker()
    apply_mode_loras(w, _mode([("a", 0.5)], ["/m/a"]))
    assert w.got == [LoraSpec("/m/a", 0.5)]


def test_apply_skips_worker_without_hook_when_no_loras():
    apply_mode_loras(object(), _mode([], []))
    apply_mode_loras(object(), Mock())


def test_lora_spec_pickles():
    spec = LoraSpec("/m/a", 0.8)
    assert pickle.loads(pickle.dumps(spec)) == spec
