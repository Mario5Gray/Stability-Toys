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


# ---------------------------------------------------------------------------
# The startup handshake is BOUNDED (STABL-wotsqcjb).
#
# Every test here drives start() on a thread with a bounded join: a regression
# reinstates an unbounded recv_bytes(), so asserting on the raised error alone
# would hang the suite for the 300s pytest timeout instead of failing.
# ---------------------------------------------------------------------------


def _start_bounded(handle, *, join_timeout: float):
    """Run handle.start(None, None, None) on a thread; return (thread, errors)."""
    import threading

    errors: list[BaseException] = []

    def _run():
        try:
            handle.start(None, None, None)
        except BaseException as exc:        # noqa: BLE001 — asserted by the caller
            errors.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=join_timeout)
    return t, errors


def test_start_raises_with_the_child_traceback_when_the_worker_fails_to_build():
    """An ordinary startup exception reaches the parent as a WorkerStartError
    carrying the CHILD's traceback — not an exit code, and not a hang.

    This is guard B. Without it the best a parent can say is "exited with code 1",
    which is the diagnosis cost this issue exists to remove."""
    from backends.worker_handle_subprocess import WorkerStartError

    handle = SubprocessWorkerHandle(
        "tests._fault_worker.make_exploding_worker", start_timeout_s=60.0
    )
    t, errors = _start_bounded(handle, join_timeout=60.0)

    try:
        assert not t.is_alive(), "start() never returned — the handshake is unbounded"
        assert errors, "start() returned normally although the child never signalled READY"
        assert isinstance(errors[0], WorkerStartError), f"raised {errors[0]!r}"
        assert "worker construction failed (injected)" in str(errors[0]), (
            f"the child's own traceback did not reach the parent: {errors[0]}"
        )
        assert handle.health().state == "dead"
    finally:
        handle.stop()


def test_start_raises_when_the_child_is_hard_killed_before_ready():
    """A SIGKILLed child cannot send a failure frame, so only the parent's
    is_alive() check can end the wait. This is guard A's liveness half — it is
    also what covers the kernel OOM-killer."""
    from backends.worker_handle_subprocess import WorkerStartError

    handle = SubprocessWorkerHandle(
        "tests._fault_worker.make_suiciding_worker", start_timeout_s=60.0
    )
    t, errors = _start_bounded(handle, join_timeout=60.0)

    try:
        assert not t.is_alive(), (
            "start() never returned: the child died without a frame and nobody "
            "closed the pipe, so recv_bytes blocks forever"
        )
        assert errors, "start() returned normally although the child was killed"
        assert isinstance(errors[0], WorkerStartError), f"raised {errors[0]!r}"
        assert str(handle._proc.exitcode) in str(errors[0]), (
            f"the error should name the child's exit code: {errors[0]}"
        )
        assert handle.health().state == "dead"
    finally:
        handle.stop()


def test_start_times_out_on_a_child_that_hangs_while_still_alive():
    """A child that blocks forever but stays ALIVE defeats the liveness check, so
    only the deadline ends the wait. The child must also be killed: an orphan
    holding a CUDA context is exactly what facet-3 exists to prevent."""
    import time

    from backends.worker_handle_subprocess import WorkerStartError

    handle = SubprocessWorkerHandle(
        "tests._fault_worker.make_hanging_worker", start_timeout_s=2.0
    )
    t0 = time.monotonic()
    t, errors = _start_bounded(handle, join_timeout=45.0)
    elapsed = time.monotonic() - t0

    try:
        assert not t.is_alive(), "start() never returned — the deadline is not enforced"
        assert errors, "start() returned normally although the child never signalled READY"
        assert isinstance(errors[0], WorkerStartError), f"raised {errors[0]!r}"
        assert elapsed < 30.0, (
            f"start() took {elapsed:.1f}s for a 2.0s timeout — the deadline is not honoured"
        )
        assert handle.health().state == "dead"
        assert not handle._proc.is_alive(), "the hung child was left orphaned"
    finally:
        handle.stop()


def test_start_timeout_defaults_to_the_module_constant():
    """The env-driven constant is the production lever; the constructor arg exists
    so tests can inject a sub-second deadline without touching process-global env."""
    from backends.worker_handle_subprocess import (
        DEFAULT_START_TIMEOUT_S,
        SubprocessWorkerHandle,
    )

    assert SubprocessWorkerHandle("x.y")._start_timeout_s == DEFAULT_START_TIMEOUT_S
    assert SubprocessWorkerHandle("x.y", start_timeout_s=1.5)._start_timeout_s == 1.5


# ---------------------------------------------------------------------------
# The envelope carries the WHOLE job (STABL-spxwqlan).
#
# encode_job carried only (req, job_id, resolution_epoch), so init_image and
# controlnet_bindings took their dataclass defaults in the child — None and [],
# which is the legitimate txt2img shape. An img2img request silently became
# txt2img; a ControlNet request silently generated uncontrolled. No error.
#
# These run across a REAL spawn boundary on purpose: a mocked transport is what
# missed this in the first place.
# ---------------------------------------------------------------------------


def _binding(attachment_id: str, image: bytes, strength: float = 0.8):
    from server.controlnet_execution import ControlNetBinding

    return ControlNetBinding(
        attachment_id=attachment_id,
        control_type="canny",
        model_id="cn-model",
        model_path="/models/cn",
        control_image_bytes=image,
        strength=strength,
        start_percent=0.0,
        end_percent=1.0,
    )


def test_controlnet_bindings_reach_the_child():
    """A dropped binding produces an UNCONTROLLED image with no error, so this is
    asserted on the child's own report of what arrived — not on the parent's job."""
    h = SubprocessWorkerHandle("tests._fault_worker.make_payload_echo_worker")
    h.start(None, None, None)
    try:
        job = GenerationJob(
            req=_req("controlled"),
            resolution_epoch=0,
            controlnet_bindings=[
                _binding("cn_1", b"CANNY-EDGES-1", strength=0.8),
                _binding("cn_2", b"DEPTH-MAP-2", strength=0.4),
            ],
        )
        fut = Future()
        h.submit(job).subscribe(_SubprocessFutureBridge(fut))
        seen = fut.result(timeout=20)

        assert seen["binding_ids"] == ["cn_1", "cn_2"], (
            f"child received bindings {seen['binding_ids']} — a ControlNet request "
            f"silently generated without control"
        )
        assert seen["control_image_bytes"] == [b"CANNY-EDGES-1", b"DEPTH-MAP-2"], (
            "control image bytes did not survive the boundary"
        )
        assert seen["strengths"] == [0.8, 0.4]
    finally:
        h.stop()


def test_init_image_reaches_the_child():
    """A dropped init_image turns img2img into txt2img — a valid-looking result that
    ignored the user's image."""
    h = SubprocessWorkerHandle("tests._fault_worker.make_payload_echo_worker")
    h.start(None, None, None)
    try:
        job = GenerationJob(
            req=_req("from-this-image"),
            resolution_epoch=0,
            init_image=b"INIT-IMAGE-PNG-BYTES",
        )
        fut = Future()
        h.submit(job).subscribe(_SubprocessFutureBridge(fut))
        seen = fut.result(timeout=20)

        assert seen["init_image"] == b"INIT-IMAGE-PNG-BYTES", (
            f"child received init_image={seen['init_image']!r} — an img2img request "
            f"silently became txt2img"
        )
    finally:
        h.stop()


def test_every_generation_job_field_is_carried_or_explicitly_excused():
    """THE STRUCTURAL GUARD. Fails when a field is added to GenerationJob without
    wire support — which is exactly how STABL-spxwqlan was born.

    A runtime rejection of unsupported fields would be dead code once the envelope
    carries them; this keeps working for the next field someone adds.
    """
    import dataclasses
    from backends.governor import GenerationJob
    from backends.job_envelope import CARRIED_JOB_FIELDS, NOT_CARRIED_JOB_FIELDS

    declared = {f.name for f in dataclasses.fields(GenerationJob)}
    accounted = set(CARRIED_JOB_FIELDS) | set(NOT_CARRIED_JOB_FIELDS)
    missing = declared - accounted

    assert not missing, (
        f"GenerationJob fields {sorted(missing)} are neither carried across the spawn "
        f"boundary nor listed as deliberately excluded. A field the envelope drops "
        f"takes its DEFAULT in the child, and for init_image/controlnet_bindings that "
        f"default is the legitimate txt2img shape — so the job silently produces the "
        f"wrong image instead of failing. Add it to CARRIED_JOB_FIELDS (and to "
        f"encode_job/decode_job) or to NOT_CARRIED_JOB_FIELDS with a reason."
    )


def test_a_v1_envelope_is_rejected_rather_than_default_filled():
    """A stale v1 body would decode into a job with init_image=None and no bindings —
    the exact silent degradation this issue is about. It must fail loudly."""
    import pickle
    from backends.job_envelope import decode_job

    v1_body = pickle.dumps((_req("old"), "job-1", 0))
    stale = bytes([1]) + v1_body

    with pytest.raises(ValueError) as exc_info:
        decode_job(stale)
    assert "schema_version" in str(exc_info.value)


# ---------------------------------------------------------------------------
# The CHILD's VRAM is attributed to the child (STABL-xtkhoidu).
#
# SubprocessWorkerHandle registered NO DeviceMemory consumer, so on the
# production isolation path 100% of worker VRAM landed in unattributed_bytes.
# DeviceMemory (STABL-hjldxurg) predates the facet-3 wiring.
# ---------------------------------------------------------------------------


def test_subprocess_consumer_reports_the_CHILD_process_not_the_parent():
    """Attribution must name the process that actually holds the memory.

    Reported across a REAL spawn boundary: a mocked transport would happily
    return the parent's pid and look correct, which is how STABL-spxwqlan
    survived review.
    """
    import os

    h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    h.start(None, None, None)
    try:
        consumer = h.memory_consumer()
        assert consumer is not None, "subprocess handle exposes no memory consumer"
        assert consumer.label == "worker", (
            "ModelRegistry._worker_entry() selects on label == 'worker'; "
            "get_reserved_vram/get_used_vram/the /status stale flag all depend on it"
        )

        cm = consumer.pool_stats()
        assert cm.pid == h._proc.pid, (
            f"consumer reported pid {cm.pid}; the memory lives in the CHILD "
            f"({h._proc.pid}), not the parent ({os.getpid()})"
        )
        assert cm.pid != os.getpid()
        assert cm.stale is False, "consumers never self-declare staleness"
    finally:
        h.stop()


def test_a_stats_request_does_not_corrupt_an_in_flight_job():
    """Why the control channel is a SEPARATE pipe, not a preference.

    The data pipe is being read concurrently by drain_to_subscriber while a job
    runs; a stats request/reply interleaved there would be consumed as a job
    frame and corrupt the stream.
    """
    h = SubprocessWorkerHandle("tests._fault_worker.make_payload_echo_worker")
    h.start(None, None, None)
    try:
        consumer = h.memory_consumer()
        job = GenerationJob(
            req=_req("concurrent"),
            resolution_epoch=0,
            init_image=b"INIT-BYTES",
        )
        fut = Future()
        h.submit(job).subscribe(_SubprocessFutureBridge(fut))

        # Hammer the control channel while the job is in flight.
        for _ in range(5):
            consumer.pool_stats()

        seen = fut.result(timeout=20)
        assert seen["prompt"] == "concurrent"
        assert seen["init_image"] == b"INIT-BYTES", (
            "the job's payload did not survive concurrent stats traffic — the "
            "control channel is interleaving with the data pipe"
        )
    finally:
        h.stop()


def test_an_unresponsive_child_degrades_to_stale_not_an_exception():
    """A dead child must not raise out of pool_stats into DeviceMemory's fan-out.

    The registry bounds every consumer at POOL_STATS_TIMEOUT_S and substitutes
    last-known with stale=True — the path already built for a wedged worker
    (STABL-hjldxurg). This asserts the consumer participates in it rather than
    inventing its own failure mode.
    """
    from backends.device_memory import get_device_memory, reset_device_memory

    h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    h.start(None, None, None)
    consumer = h.memory_consumer()
    h.stop()                      # child is now dead; the control pipe answers nothing

    reset_device_memory()
    dm = get_device_memory()
    reg = dm.register(consumer)
    try:
        snap = dm.snapshot()
        entry = next((c for c in snap.consumers if c.label == "worker"), None)
        assert entry is not None, "the dead child's consumer vanished from the snapshot"
        assert entry.stale is True, (
            "a dead child must surface as stale, not as a zero that reads as truth"
        )
    finally:
        reg.close()
        reset_device_memory()
