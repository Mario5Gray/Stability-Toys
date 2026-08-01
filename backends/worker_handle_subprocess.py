"""SubprocessWorkerHandle — hosts the worker in a spawn child (facet-3, M1).

The durable-OOM-recovery locality: the CUDA context lives in a child process, so a
poisoned context is dropped by killing the process (Task 7), which in-process
empty_cache/del cannot do. This handle implements the merged WorkerHandle ABC, so
the Governor programs to it identically to InProcessWorkerHandle — except its
`.worker` is always None (no in-proc worker), and the Governor reads liveness via
`health().state` (the `_worker_available()` flip, Task 6).

Child protocol: parent sends the versioned job envelope (Task 3); the child decodes,
runs `worker.run_job`, and drives an IpcJobSink (Task 4). The opaque run_job return
(bytes OR (png, seed) tuple — base.py:37) is PICKLED into the shared-memory blob
(recon #5C) and unpickled by `_SubprocessFutureBridge`, preserving the in-proc
opaque-carry contract across the boundary. Big payload rides shared memory, not the
pipe.
"""
from __future__ import annotations

import importlib
import os
import pickle
import threading
import time
import multiprocessing as mp
from typing import Optional

from backends.worker_handle import WorkerHandle, WorkerHealth
from backends.liveness import SubprocessLiveness
from backends.model_resolution import resolved_model_to_json_dict
from backends.job_envelope import encode_job, decode_job
from backends.backplane.ipc import IpcJobSink, drain_to_subscriber
from backends.backplane.reactivestreams import Publisher, Subscriber
from backends.backplane.frames import Progress, Result

_READY = b"\x00READY"
_FAILED = b"\x00FAILED"
_STATS = b"\x00STATS"

# STABL-xtkhoidu. DeviceMemory's fan-out bounds every consumer at
# POOL_STATS_TIMEOUT_S (0.5s) and substitutes last-known with stale=True, so the
# reply wait must expire INSIDE that budget — otherwise the fan-out times out
# first and we lose the distinction between "child is busy" and "child is gone".
_STATS_REPLY_TIMEOUT_S = 0.25

# STABL-wotsqcjb. Generous because the spawn child imports torch + diffusers cold
# before it can signal READY. It costs nothing in the common failure: the liveness
# check below raises the moment the child dies, so the full window is only ever
# spent on a child that is hung but ALIVE.
DEFAULT_START_TIMEOUT_S: float = float(os.environ.get("WORKER_START_TIMEOUT_S", "300"))
_READY_POLL_INTERVAL_S: float = 0.1
_MAX_FAILURE_BYTES = 8192


class WorkerStartError(RuntimeError):
    """The child never reached READY. A RuntimeError subclass on purpose: the
    Governor's `except Exception` around handle.start() (governor.py:447) catches
    it unchanged and still runs _drop_reservation(dead=True)."""


def _resolve_ref(dotted: str):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _serve_stats(control_conn):
    """Child-side control server: answer VRAM stats requests (STABL-xtkhoidu).

    Runs on its own daemon thread reading its OWN pipe. It cannot share the data
    pipe: `drain_to_subscriber` reads that concurrently while a job runs, so an
    interleaved request/reply would be consumed as a job frame and corrupt the
    stream.

    A separate thread is also what lets stats answer DURING a generation — torch
    releases the GIL across CUDA calls, so the reply lands inside the fan-out
    budget instead of queueing behind the denoise.
    """
    import torch

    while True:
        try:
            if control_conn.recv_bytes() != _STATS:
                continue
            control_conn.send_bytes(pickle.dumps({
                "pid": os.getpid(),
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
            }))
        except (EOFError, OSError):
            break                      # parent closed the pipe: the child is going away
        except Exception:              # noqa: BLE001 — never kill the child over stats
            try:
                control_conn.send_bytes(pickle.dumps(None))
            except Exception:          # noqa: BLE001
                break


class _SubprocessMemoryConsumer:
    """The CHILD process as a DeviceMemory consumer.

    Label is "worker" because that spelling is load-bearing:
    `ModelRegistry._worker_entry()` selects on it, and `get_reserved_vram()`,
    `get_used_vram()` and the `/status` stale flag all hang off that lookup.

    Raising on an unreachable child is the CORRECT behaviour, not a gap:
    `_ConsumerRegistry._read_consumer` catches it and substitutes last-known with
    `stale=True` — the path built for a wedged worker. Returning zeros instead
    would report "the worker holds no VRAM", which reads as truth.
    """

    label = "worker"

    def __init__(self, handle):
        self._handle = handle

    def pool_stats(self):
        from backends.device_memory import ConsumerMemory

        stats = self._handle.request_stats(timeout_s=_STATS_REPLY_TIMEOUT_S)
        if stats is None:
            raise RuntimeError("subprocess worker did not report memory stats")
        return ConsumerMemory(
            label=self.label,
            pid=stats["pid"],
            allocated_bytes=stats["allocated_bytes"],
            reserved_bytes=stats["reserved_bytes"],
            stale=False,   # consumers never self-declare staleness (spec §2.3)
        )

    def reclaim(self) -> None:
        # No-op: trimming the child's allocator needs a control verb of its own, and
        # the Governor's OOM recovery already reclaims by KILLING the process, which
        # is strictly more effective than empty_cache (the facet-3 thesis).
        return None


def _worker_main(conn, factory_ref, wire_resolved, binding, mode, control_conn=None):
    """Spawn-child entrypoint: rebuild the resolution, build + condition the worker,
    signal READY.

    ``ResolvedModel`` cannot be pickled (it holds ``MappingProxyType``), so the wire
    form is its JSON dict — the codec `model_resolution` was designed to provide
    (module docstring: "wire-safe ResolvedModel that may appear verbatim in a request
    trace and be consumed by a future remote processor **without re-detecting or
    re-resolving family**"). The child rebuilds, it does NOT re-resolve: resolution
    authority stays parent-side, the child touches no filesystem, and a parent-side
    `resolve_model` patch still governs the result even though patches do not cross a
    spawn boundary.

    Serial by construction — one job at a time. The opaque run_job return is pickled
    into the blob (recon #5C); errors ride the sink terminal so a waiting Future
    never hangs. A frameless death (SIGKILL) is caught parent-side by the Task 4 EOF
    guard, not here.

    STABL-wotsqcjb guard B: everything up to READY runs under a try/except that sends
    the traceback back as a _FAILED frame before the child dies, so the parent raises
    with the real cause rather than an exit code. It cannot cover a hard kill — that
    is the parent's liveness check (guard A)."""
    try:
        from backends.model_resolution import resolved_model_from_json_dict

        factory = _resolve_ref(factory_ref)
        resolved = (
            resolved_model_from_json_dict(wire_resolved) if wire_resolved is not None else None
        )
        worker = factory(0, resolved, binding)

        # M-A: replicate InProcessWorkerHandle.start() conditioning configuration.
        if mode is not None:
            configure_conditioning = getattr(worker, "configure_conditioning", None)
            if callable(configure_conditioning):
                configure_conditioning(mode.conditioning)
            elif mode.conditioning.requires_configurable_worker():
                raise RuntimeError(
                    f"mode configures conditioning but worker {type(worker).__name__} "
                    "does not support conditioning"
                )
    except BaseException:       # noqa: BLE001 — reported to the parent, then re-raised
        import traceback
        try:
            conn.send_bytes(
                _FAILED + traceback.format_exc().encode()[:_MAX_FAILURE_BYTES]
            )
        except Exception:       # noqa: BLE001 — a broken pipe leaves guard A to notice
            pass
        raise

    # Start the control server only after the worker is built: answering stats for a
    # half-constructed child would report numbers nobody can act on.
    if control_conn is not None:
        threading.Thread(target=_serve_stats, args=(control_conn,), daemon=True).start()

    conn.send_bytes(_READY)
    while True:
        try:
            raw = conn.recv_bytes()
        except EOFError:
            break
        d = decode_job(raw)
        from backends.governor import GenerationJob
        # Every transported field must be reconstructed here. Omitting one does not
        # raise: it takes the dataclass DEFAULT, and for init_image/controlnet_bindings
        # None/[] is the legitimate txt2img shape, so the job silently produces the
        # wrong image (STABL-spxwqlan).
        job = GenerationJob(
            req=d.req,
            resolution_epoch=d.resolution_epoch,
            job_id=d.job_id,
            init_image=d.init_image,
            controlnet_bindings=list(d.controlnet_bindings or []),
        )
        sink = IpcJobSink(conn, job_id=d.job_id)
        try:
            # STABL-zueslhah Task 3: thread the IpcJobSink's progress emitter so the
            # diffusion step callback streams Progress over the pipe that
            # drain_to_subscriber already reads.
            result = worker.run_job(job, progress=sink.progress)  # opaque: bytes (FaultWorker) or (png, seed) tuple (real)
            sink.result(0, pickle.dumps(result))      # recon #5C: pickle the opaque return into the blob
            sink.complete()
        except Exception as e:   # noqa: BLE001 — rides the sink terminal
            from backends.backplane.frames import BackplaneError
            sink.error(BackplaneError.from_exc(e))


class _SubprocessFutureBridge(Subscriber):
    """Fulfils a Future from the subprocess stream: unpickles the opaque run_job
    return (recon #5C), and records terminal_error_code (recon #5E) so the Governor
    distinguishes in-band OOM (child alive -> must kill) from success/other errors."""

    def __init__(self, fut, on_progress=None):
        self._fut = fut
        self.terminal_error_code = None
        # STABL-zueslhah: forward non-terminal Progress to a consumer (WS in Task 4)
        # instead of dropping it; None preserves today's Future-only behaviour.
        self._on_progress = on_progress

    def on_subscribe(self, subscription):
        subscription.request(1 << 62)

    def on_next(self, value):
        if isinstance(value, Progress):
            if self._on_progress is not None:
                self._on_progress(value.step, value.total, value.stage)
            return
        if isinstance(value, Result) and not self._fut.done():
            self._fut.set_result(pickle.loads(value.image.read_sync()))

    def on_error(self, error):
        self.terminal_error_code = error.code
        if not self._fut.done():
            self._fut.set_exception(error.to_exception())

    def on_complete(self):
        pass


class _SubprocPublisher(Publisher):
    def __init__(self, conn):
        self._conn = conn

    def subscribe(self, subscriber):
        t = threading.Thread(
            target=drain_to_subscriber, args=(self._conn, subscriber), daemon=True
        )
        t.start()


class SubprocessWorkerHandle(WorkerHandle):
    def __init__(self, worker_factory_ref: str, start_timeout_s: Optional[float] = None,
                 device_memory=None):
        self._factory_ref = worker_factory_ref
        self._dm = device_memory
        self._registration = None
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._parent_conn = None
        self._control_conn = None
        # Serialises control round-trips: DeviceMemory fans out on a thread pool, so
        # two concurrent snapshots could otherwise interleave request and reply.
        self._control_lock = threading.Lock()
        self._liveness: Optional[SubprocessLiveness] = None
        self._state = "starting"
        # None => the env-driven module constant. The argument exists so tests can
        # inject a sub-second deadline without mutating process-global env.
        self._start_timeout_s = (
            DEFAULT_START_TIMEOUT_S if start_timeout_s is None else start_timeout_s
        )

    @property
    def worker(self):
        return None

    def start(self, resolved_mode, binding, mode) -> None:
        # M-A: ResolvedModel is unpicklable (MappingProxyType), so it crosses the spawn
        # boundary as its JSON dict via the codec model_resolution already provides.
        # The parent stays the single resolution authority — the child rebuilds rather
        # than re-resolving, so it needs no filesystem access and a parent-side
        # resolve_model patch still governs what the child sees.
        wire_resolved = (
            resolved_model_to_json_dict(resolved_mode) if resolved_mode is not None else None
        )
        self._parent_conn, child_conn = self._ctx.Pipe()
        # A SECOND pipe for control/stats (STABL-xtkhoidu). Not a preference: the data
        # pipe above is read concurrently by drain_to_subscriber while a job runs, so a
        # stats request/reply sharing it would be consumed as a job frame.
        self._control_conn, child_control = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self._factory_ref, wire_resolved, binding, mode,
                  child_control),
            daemon=True,
        )
        self._proc.start()
        # recon #5B: is_alive()-only liveness in M1 — staleness disabled until the
        # periodic-heartbeat follow-on, so an idle-but-alive child is never dead.
        self._liveness = SubprocessLiveness(self._proc, stale_after_s=float("inf"))
        self._await_ready()
        self._liveness.note_heartbeat()
        self._state = "ready"
        # Register AFTER ready, mirroring InProcessWorkerHandle: a snapshot must never
        # sample a half-built consumer (spec §5, event 1).
        if self._dm is None:
            from backends.device_memory import get_device_memory
            self._dm = get_device_memory()
        self._registration = self._dm.register(self.memory_consumer())

    def _await_ready(self) -> None:
        """Wait for the child's READY, bounded by liveness AND a deadline (guard A).

        `recv_bytes()` alone blocks forever: the child is daemon=True, so a child that
        dies before READY closes nothing the parent would raise EOFError on. Polling
        with an is_alive() check covers every death mode including SIGKILL and the
        kernel OOM-killer; the deadline covers a child that hangs while still alive.

        poll() is checked BEFORE is_alive() so a child that writes _FAILED and exits
        immediately still has its frame read — buffered data stays readable after the
        writer dies."""
        deadline = time.monotonic() + self._start_timeout_s
        while not self._parent_conn.poll(_READY_POLL_INTERVAL_S):
            if not self._proc.is_alive():
                # A child that wrote _FAILED and exited inside the poll gap still has
                # its frame buffered. Prefer that traceback over a bare exit code —
                # discarding it here would defeat guard B in the very case it targets.
                if self._parent_conn.poll(0):
                    break
                exitcode = self._proc.exitcode
                self.stop()
                raise WorkerStartError(
                    f"subprocess worker '{self._factory_ref}' exited with code "
                    f"{exitcode} before signalling READY, and sent no failure frame "
                    f"(hard kill, or a crash before the child could report)"
                )
            if time.monotonic() >= deadline:
                self.stop()
                raise WorkerStartError(
                    f"subprocess worker '{self._factory_ref}' did not signal READY "
                    f"within {self._start_timeout_s}s (still alive — hung during "
                    f"startup); child killed. Raise WORKER_START_TIMEOUT_S if a "
                    f"legitimate load needs longer."
                )

        frame = self._parent_conn.recv_bytes()
        if frame == _READY:
            return

        self.stop()
        if frame.startswith(_FAILED):
            detail = frame[len(_FAILED):].decode(errors="replace")
            raise WorkerStartError(
                f"subprocess worker '{self._factory_ref}' failed during startup:\n{detail}"
            )
        raise WorkerStartError(
            f"subprocess worker '{self._factory_ref}' sent {frame[:64]!r} instead of READY"
        )

    def memory_consumer(self):
        """This child as a DeviceMemory consumer (STABL-xtkhoidu).

        Without it the subprocess path registers nothing, so 100% of worker VRAM
        lands in unattributed_bytes — DeviceMemory predates the facet-3 wiring.
        """
        return _SubprocessMemoryConsumer(self)

    def request_stats(self, *, timeout_s: float) -> Optional[dict]:
        """Bounded control round-trip. Returns None if the child does not answer in
        time; the caller turns that into the registry's stale substitution."""
        conn = self._control_conn
        if conn is None or self._proc is None or not self._proc.is_alive():
            return None
        with self._control_lock:
            try:
                conn.send_bytes(_STATS)
                if not conn.poll(timeout_s):
                    return None
                return pickle.loads(conn.recv_bytes())
            except (EOFError, OSError):
                return None

    def submit(self, job) -> Publisher:
        self._state = "busy"
        self._parent_conn.send_bytes(encode_job(job))
        self._liveness.note_heartbeat()
        return _SubprocPublisher(self._parent_conn)

    def health(self) -> WorkerHealth:
        state = "dead" if (self._liveness is None or self._liveness.state() == "dead") else self._state
        return WorkerHealth(state=state, vram_free_bytes=0, vram_total_bytes=0, mode=None)

    def unload(self) -> None:
        self.stop()

    def stop(self) -> None:
        # Deregister BEFORE the kill so no snapshot fan-out samples a dying child
        # (spec §5, event 2 — the same ordering InProcessWorkerHandle.unload uses).
        if self._registration is not None:
            self._registration.close()
            self._registration = None
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=5.0)
        self._state = "dead"
