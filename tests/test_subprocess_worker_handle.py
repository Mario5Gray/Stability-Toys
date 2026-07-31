"""SubprocessWorkerHandle isolation tests — M1 across a REAL spawn boundary.

Uses the module-level FaultWorker (tests/_fault_worker.py) so the spawn child can
import it by dotted ref; no real GPU. Proves start()->READY, submit()->Publisher
driven by the child, opaque-return pickle round-trip, and stop()->dead.
"""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from backends.worker_handle_subprocess import (
    SubprocessWorkerHandle,
    _SubprocessFutureBridge,
)
from backends.governor import GenerationJob
from backends.backplane.frames import BackplaneErrorCode
from server.lcm_sr_server import GenerateRequest


def _req(prompt="hello"):
    # Real GenerateRequest fields (server/lcm_sr_server.py:136): num_inference_steps
    # + size, NOT the plan's stale steps/width/height (same T0 correction).
    return GenerateRequest(prompt=prompt, num_inference_steps=4, size="512x512")


def test_subprocess_handle_runs_a_job_end_to_end():
    h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    # start() args are pickled into the spawn child, so they must be picklable —
    # Mock() is not. FaultWorker ignores resolved/binding/mode, so None stands in.
    # Real resolved/binding/mode picklability is validated by Task 8 live acceptance.
    h.start(None, None, None)                   # spawns child, loads FaultWorker
    assert h.health().state == "ready"
    assert h.worker is None                     # no in-proc worker
    job = GenerationJob(req=_req("hello"), resolution_epoch=0)
    pub = h.submit(job)
    fut = Future()
    pub.subscribe(_SubprocessFutureBridge(fut))   # unpickles the opaque return
    assert fut.result(timeout=15) == b"PNG:hello"  # pickle(bytes) round-trips to bytes
    h.stop()
    assert h.health().state == "dead"


def test_subprocess_handle_propagates_oom_as_oom_code():
    """An in-band OOM is captured by the bridge as BackplaneErrorCode.OOM.

    This is the handle-level half of Task 7: the child catches the error, emits
    a terminal frame, and stays alive. The bridge must classify it as OOM so the
    Governor knows to kill the poisoned child.
    """
    h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    h.start(None, None, None)
    job = GenerationJob(req=_req("__OOM__"), resolution_epoch=0)
    pub = h.submit(job)
    fut = Future()
    bridge = _SubprocessFutureBridge(fut)
    pub.subscribe(bridge)
    with pytest.raises(Exception) as exc_info:
        fut.result(timeout=15)
    assert bridge.terminal_error_code is BackplaneErrorCode.OOM
    assert isinstance(exc_info.value, torch.cuda.OutOfMemoryError)
    h.stop()


def test_subprocess_handle_configures_conditioning_in_child():
    """The spawn child calls configure_conditioning(mode.conditioning) just like
    InProcessWorkerHandle.start() does (M-A conditioning gap)."""
    from backends.conditioning.contracts import ConditioningConfig

    h = SubprocessWorkerHandle("tests._fault_worker.make_recording_worker")
    mode = SimpleNamespace(
        model_path=None,  # test bypass: child skips real resolve_model
        conditioning=ConditioningConfig(),
    )
    h.start(None, None, mode)

    job = GenerationJob(req=_req("conditioned"), resolution_epoch=0)
    pub = h.submit(job)
    fut = Future()
    pub.subscribe(_SubprocessFutureBridge(fut))
    assert fut.result(timeout=15) == b"PNG:conditioned"
    h.stop()


def test_subprocess_handle_pickle_real_mode_config():
    """A real ModeConfig pickles across spawn so start() can send mode to the child."""
    import pickle
    from server.mode_config import ModeConfig, ConditioningConfig

    mode = ModeConfig(
        name="test",
        model="test.safetensors",
        conditioning=ConditioningConfig(),
    )
    # Confirm the exact object Governor._load_mode deepcopies and passes to start()
    # round-trips through pickle (the spawn boundary).
    round_tripped = pickle.loads(pickle.dumps(mode))
    assert round_tripped.name == mode.name
    assert round_tripped.model == mode.model


# ---------------------------------------------------------------------------
# Gate: the ResolvedModel crosses the spawn boundary via its CODEC, and the
# child rebuilds rather than re-resolving (STABL-rgvxuedo, M-A correction).
# ---------------------------------------------------------------------------

def _real_resolved(tmp_path):
    """A genuine ResolvedModel, built the same way test_model_resolution does."""
    from typing import Any
    from backends.family_profiles import SD15_PROFILE
    from backends.model_resolution import build_resolved, local_artifact_ref
    from utils.model_detector import ModelInfo, ModelVariant

    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_bytes(b"{}")

    def _info(**overrides) -> ModelInfo:
        base: dict[str, Any] = dict(
            variant=ModelVariant.SD15,
            cross_attention_dim=768,
            text_encoder_hidden_size=768,
            base_arch="unet",
            format="diffusers",
            confidence=0.9,
            detected_by=["DiffusersDetector"],
            metadata={"source": "unit-test"},
        )
        base.update(overrides)
        return ModelInfo(path=str(model_dir), **base)

    return build_resolved(
        model_ref=local_artifact_ref(str(model_dir)),
        raw_info=_info(),
        profile=SD15_PROFILE,
        info=_info(checkpoint_variant="fp16"),
    )


def test_resolved_model_is_not_picklable_which_is_why_the_codec_exists(tmp_path):
    """Regression guard for the ROOT cause. ResolvedModel holds MappingProxyType and
    cannot be pickled, so it can never be passed to Process(args=...) directly. If
    this ever starts passing, the codec indirection below is no longer load-bearing
    and someone should say so deliberately rather than discover it by accident."""
    import pickle

    resolved = _real_resolved(tmp_path)
    with pytest.raises(Exception):
        pickle.loads(pickle.dumps(resolved))


def test_start_sends_the_codec_wire_form_and_it_round_trips(tmp_path):
    """The handle boundary carries the ResolvedModel's JSON dict, and decoding it
    yields an equal ResolvedModel — so the child rebuilds and never re-resolves.

    Asserts on what start() actually hands to Process, without spawning."""
    from unittest.mock import MagicMock, patch
    from backends.model_resolution import (
        LocalModelBinding,
        resolved_model_from_json_dict,
    )
    from backends.worker_handle_subprocess import SubprocessWorkerHandle

    resolved = _real_resolved(tmp_path)
    binding = LocalModelBinding(model_path=str(tmp_path / "model"))
    mode = SimpleNamespace(model_path=str(tmp_path / "model"), loras=[], conditioning=None)

    from backends.worker_handle_subprocess import _READY

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    ctx = MagicMock()
    parent_conn = MagicMock()
    parent_conn.recv_bytes.return_value = _READY      # skip the real handshake
    ctx.Pipe.return_value = (parent_conn, MagicMock())
    with patch.object(handle, "_ctx", ctx):
        handle.start(resolved, binding, mode)
        wire = ctx.Process.call_args.kwargs["args"][2]

    # It is the codec's JSON dict, not the object and not a bare path.
    assert isinstance(wire, dict), f"wire form is {type(wire).__name__}, expected dict"
    assert not isinstance(wire, str), "sending model_path makes the child re-resolve"

    # It survives pickling (the object itself cannot) and round-trips exactly.
    import pickle
    rebuilt = resolved_model_from_json_dict(pickle.loads(pickle.dumps(wire)))
    assert rebuilt == resolved
    assert rebuilt.resolution_id == resolved.resolution_id
    assert rebuilt.profile.family_id == resolved.profile.family_id


def test_child_starts_when_the_model_path_is_not_resolvable_on_disk(tmp_path):
    """The regression that hung. With a child-side re-resolve, a model_path the child
    cannot resolve raises before _READY is ever sent, and the parent blocks forever in
    recv_bytes (start() has no timeout — see the note on this file's gate section).
    Rebuilding from the codec needs no filesystem, so the child reaches ready
    regardless of what the path points at.

    start() is driven on a thread with a bounded join so that a broken contract FAILS
    here instead of hanging the suite for the 300s pytest timeout."""
    import threading
    from backends.conditioning.contracts import ConditioningConfig
    from backends.model_resolution import LocalModelBinding
    from backends.worker_handle_subprocess import SubprocessWorkerHandle

    resolved = _real_resolved(tmp_path)
    missing = str(tmp_path / "does-not-exist" / "model.safetensors")
    binding = LocalModelBinding(model_path=missing)
    mode = SimpleNamespace(
        model_path=missing, loras=[], conditioning=ConditioningConfig()
    )

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    error: list[BaseException] = []

    def _start():
        try:
            handle.start(resolved, binding, mode)
        except BaseException as exc:            # noqa: BLE001 — surfaced below
            error.append(exc)

    t = threading.Thread(target=_start, daemon=True)
    t.start()
    t.join(timeout=30.0)

    try:
        assert not t.is_alive(), (
            "start() never returned: the child failed before signalling READY and "
            "the parent is blocked in recv_bytes. This is the child-side re-resolve "
            "regression."
        )
        assert not error, f"start() raised: {error[0]!r}"
        assert handle.health().state in ("ready", "busy")
    finally:
        handle.stop()
