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
import pickle
import threading
import multiprocessing as mp
from typing import Optional

from backends.worker_handle import WorkerHandle, WorkerHealth
from backends.liveness import SubprocessLiveness
from backends.model_resolution import resolved_model_to_json_dict
from backends.job_envelope import encode_job, decode_job
from backends.backplane.ipc import IpcJobSink, drain_to_subscriber
from backends.backplane.reactivestreams import Publisher, Subscriber
from backends.backplane.frames import Result

_READY = b"\x00READY"


def _resolve_ref(dotted: str):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _worker_main(conn, factory_ref, wire_resolved, binding, mode):
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
    guard, not here."""
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

    conn.send_bytes(_READY)
    while True:
        try:
            raw = conn.recv_bytes()
        except EOFError:
            break
        d = decode_job(raw)
        from backends.governor import GenerationJob
        job = GenerationJob(req=d.req, resolution_epoch=d.resolution_epoch, job_id=d.job_id)
        sink = IpcJobSink(conn, job_id=d.job_id)
        try:
            result = worker.run_job(job)              # opaque: bytes (FaultWorker) or (png, seed) tuple (real)
            sink.result(0, pickle.dumps(result))      # recon #5C: pickle the opaque return into the blob
            sink.complete()
        except Exception as e:   # noqa: BLE001 — rides the sink terminal
            from backends.backplane.frames import BackplaneError
            sink.error(BackplaneError.from_exc(e))


class _SubprocessFutureBridge(Subscriber):
    """Fulfils a Future from the subprocess stream: unpickles the opaque run_job
    return (recon #5C), and records terminal_error_code (recon #5E) so the Governor
    distinguishes in-band OOM (child alive -> must kill) from success/other errors."""

    def __init__(self, fut):
        self._fut = fut
        self.terminal_error_code = None

    def on_subscribe(self, subscription):
        subscription.request(1 << 62)

    def on_next(self, value):
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
    def __init__(self, worker_factory_ref: str):
        self._factory_ref = worker_factory_ref
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._parent_conn = None
        self._liveness: Optional[SubprocessLiveness] = None
        self._state = "starting"

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
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self._factory_ref, wire_resolved, binding, mode),
            daemon=True,
        )
        self._proc.start()
        # recon #5B: is_alive()-only liveness in M1 — staleness disabled until the
        # periodic-heartbeat follow-on, so an idle-but-alive child is never dead.
        self._liveness = SubprocessLiveness(self._proc, stale_after_s=float("inf"))
        if self._parent_conn.recv_bytes() != _READY:   # blocks until READY
            raise RuntimeError("subprocess worker failed to signal READY")
        self._liveness.note_heartbeat()
        self._state = "ready"

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
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=5.0)
        self._state = "dead"
