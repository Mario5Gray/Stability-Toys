"""Governor authority key = (base mode name, request LoRAs) (STABL-fvwsvjvn)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backends.governor import Governor, ModeSwitchJob
from backends.mode_loras import LoraSpec, RequestLoraError
from server.mode_config import LoRAConfig
from tests.test_governor import (
    StubHandle,
    _drain_queue,
    _freeze_dispatch,
    _make_mock_registry,
    _make_multi_mode_config,
    _resolve_by_path,
)

S1 = (LoraSpec("/l/a.safetensors", 0.8),)
S2 = (LoraSpec("/l/a.safetensors", 1.0),)


@pytest.fixture
def resolve():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        yield


def _gov(*names, default=None, lora_mode=None):
    config = _make_multi_mode_config(*names, default=default)
    if lora_mode is not None:
        config._modes[lora_mode].loras = [LoRAConfig(path="x.safetensors")]
        config._modes[lora_mode].lora_paths = ["/l/x.safetensors"]
    handle = StubHandle()
    gov = Governor(handle=handle, mode_config=config, registry=_make_mock_registry())
    return gov, handle, config


def _wait_loaded(gov, mode_name, request_loras=(), timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = gov.get_active_model_snapshot()
        if (
            snap is not None
            and snap.mode_name == mode_name
            and snap.request_loras == request_loras
            and not gov._pending_authorities
        ):
            return snap
        time.sleep(0.01)
    raise AssertionError(f"({mode_name}, {request_loras}) never loaded")


def _shutdown(gov):
    _freeze_dispatch(gov)
    _drain_queue(gov)
    gov.shutdown()


def test_same_request_loras_do_not_reserve_again(resolve):
    gov, _, _ = _gov("mode-a", "mode-b", default="mode-b")
    try:
        _freeze_dispatch(gov)
        before = gov._resolution_epoch
        first = gov.admit_generation("mode-a", S1)
        second = gov.admit_generation("mode-a", S1)
        assert gov._resolution_epoch == before + 1
        assert second is first
    finally:
        _shutdown(gov)


def test_different_strength_reserves_new_authority(resolve):
    gov, _, _ = _gov("mode-a", "mode-b", default="mode-b")
    try:
        _freeze_dispatch(gov)
        before = gov._resolution_epoch
        gov.admit_generation("mode-a", S1)
        last = gov.admit_generation("mode-a", S2)
        assert gov._resolution_epoch == before + 2
        assert last.request_loras == S2
        assert gov._pending_authorities[-1] is last
    finally:
        _shutdown(gov)


def test_no_mode_request_keeps_request_loras(resolve):
    gov, _, _ = _gov("mode-a", default="mode-a")
    try:
        gov.admit_generation("mode-a", S1)
        _wait_loaded(gov, "mode-a", S1)
        snap = gov.admit_generation(None)
        assert snap.mode_name == "mode-a"
        assert snap.request_loras == S1
    finally:
        _shutdown(gov)


def test_no_mode_with_loras_targets_terminal_base(resolve):
    gov, _, _ = _gov("mode-a", "mode-b", default="mode-a")
    try:
        _freeze_dispatch(gov)
        gov.admit_generation("mode-b")
        snap = gov.admit_generation(None, S1)
        assert snap.mode_name == "mode-b"
        assert snap.request_loras == S1
    finally:
        _shutdown(gov)


def test_no_mode_no_terminal_uses_default_mode(resolve):
    gov, _, _ = _gov("mode-a", "mode-b", default="mode-b")
    try:
        gov.unload_current_model()
        _freeze_dispatch(gov)
        snap = gov.admit_generation(None, S1)
        assert snap.mode_name == "mode-b"
        assert snap.request_loras == S1
    finally:
        _shutdown(gov)


def test_mode_without_loras_reloads_configured(resolve):
    gov, _, _ = _gov("mode-a", default="mode-a")
    try:
        gov.admit_generation("mode-a", S1)
        _wait_loaded(gov, "mode-a", S1)
        _freeze_dispatch(gov)
        snap = gov.admit_generation("mode-a")
        assert snap.mode_name == "mode-a"
        assert snap.request_loras == ()
        assert gov._pending_authorities[-1] is snap
    finally:
        _shutdown(gov)


def test_request_loras_on_lora_mode_refused(resolve):
    gov, _, _ = _gov("mode-a", "mode-l", default="mode-a", lora_mode="mode-l")
    try:
        _freeze_dispatch(gov)
        epoch = gov._resolution_epoch
        pending = list(gov._pending_authorities)
        with pytest.raises(RequestLoraError, match="mode-l.*without loras"):
            gov.admit_generation("mode-l", S1)
        assert gov._resolution_epoch == epoch
        assert gov._pending_authorities == pending
    finally:
        _shutdown(gov)


def test_no_mode_request_loras_on_lora_mode_refused(resolve):
    gov, handle, _ = _gov("mode-l", default="mode-l", lora_mode="mode-l")
    try:
        _wait_loaded(gov, "mode-l")
        _freeze_dispatch(gov)
        epoch = gov._resolution_epoch
        active = gov.get_active_model_snapshot()
        starts = len(handle.start_calls)
        with pytest.raises(RequestLoraError, match="mode-l"):
            gov.admit_generation(None, S1)
        assert gov._resolution_epoch == epoch
        assert gov._pending_authorities == []
        assert gov.get_active_model_snapshot() is active
        assert len(handle.start_calls) == starts
    finally:
        _shutdown(gov)


def test_derived_mode_holds_request_loras(resolve):
    gov, _, config = _gov("mode-a", "mode-b", default="mode-b")
    try:
        _freeze_dispatch(gov)
        snap = gov.admit_generation("mode-a", S1)
        assert snap.mode.lora_paths == ["/l/a.safetensors"]
        assert snap.mode.loras[0].strength == 0.8
        assert config._modes["mode-a"].loras == []
    finally:
        _shutdown(gov)


def test_switch_mode_targets_plain_key(resolve):
    gov, _, _ = _gov("mode-a", default="mode-a")
    try:
        gov.admit_generation("mode-a", S1)
        _wait_loaded(gov, "mode-a", S1)
        _freeze_dispatch(gov)
        gov.switch_mode("mode-a")
        last = gov._pending_authorities[-1]
        assert last.mode_name == "mode-a"
        assert last.request_loras == ()
    finally:
        _shutdown(gov)


def test_dispatch_fast_path_compares_key(resolve):
    gov, handle, _ = _gov("mode-a", default="mode-a")
    try:
        _wait_loaded(gov, "mode-a")
        starts = len(handle.start_calls)
        job = ModeSwitchJob(target_mode="mode-a", request_loras=S1)
        gov.q.put(job)
        result = job.fut.result(timeout=5.0)
        assert result["status"] != "already_loaded"
        assert len(handle.start_calls) == starts + 1
        assert handle.start_calls[-1][2].lora_paths == ["/l/a.safetensors"]
    finally:
        _shutdown(gov)


def test_reload_current_mode_keeps_request_loras(resolve):
    gov, handle, _ = _gov("mode-a", default="mode-a")
    try:
        gov.admit_generation("mode-a", S1)
        _wait_loaded(gov, "mode-a", S1)
        starts = len(handle.start_calls)
        gov.reload_current_mode()
        assert len(handle.start_calls) == starts + 1
        assert handle.start_calls[-1][2].lora_paths == ["/l/a.safetensors"]
        assert gov.get_active_model_snapshot().request_loras == S1
    finally:
        _shutdown(gov)


def test_reload_if_current_keeps_request_loras(resolve):
    gov, handle, _ = _gov("mode-a", default="mode-a")
    try:
        gov.admit_generation("mode-a", S1)
        _wait_loaded(gov, "mode-a", S1)
        _freeze_dispatch(gov)
        assert gov.reload_if_current("mode-a") is True
        assert gov._pending_authorities[-1].request_loras == S1
    finally:
        _shutdown(gov)


def test_demand_reload_keeps_request_loras(resolve):
    gov, handle, _ = _gov("mode-a", default="mode-a")
    try:
        gov.admit_generation("mode-a", S1)
        _wait_loaded(gov, "mode-a", S1)
        _freeze_dispatch(gov)
        gov._unload_current_worker(reason="idle")
        gov._reload_from_snapshot()
        assert handle.start_calls[-1][2].lora_paths == ["/l/a.safetensors"]
        assert gov.get_active_model_snapshot().request_loras == S1
    finally:
        _shutdown(gov)


def test_runtime_status_reports_current_loras(resolve):
    gov, _, _ = _gov("mode-a", default="mode-a")
    try:
        gov.admit_generation("mode-a", S1)
        _wait_loaded(gov, "mode-a", S1)
        status = gov._build_runtime_status()
        assert status["current_loras"] == [{"path": "/l/a.safetensors", "strength": 0.8}]
        assert status["current_mode"] == "mode-a"
    finally:
        _shutdown(gov)


def test_runtime_status_current_loras_empty_when_unloaded(resolve):
    gov, _, _ = _gov("mode-a", default="mode-a")
    try:
        gov.unload_current_model()
        assert gov.get_current_loras() == []
    finally:
        _shutdown(gov)


def test_request_lora_switch_logs_count(resolve, caplog):
    import logging

    gov, _, _ = _gov("mode-a", "mode-b", default="mode-b")
    try:
        _freeze_dispatch(gov)
        with caplog.at_level(logging.INFO, logger="backends.governor"):
            gov.admit_generation("mode-a", S1)
        assert "Mode mode-a with 1 request LoRA(s)" in caplog.text
    finally:
        _shutdown(gov)
