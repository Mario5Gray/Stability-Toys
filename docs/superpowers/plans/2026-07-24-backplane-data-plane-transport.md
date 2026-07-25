# Backplane Data-Plane Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Project policy forbids sub-agent-driven development (AGENTS.md) — do NOT use superpowers:subagent-driven-development.**

**Goal:** Abstract the worker's result/progress stream behind a pluggable "backplane" data-plane seam so the worker boundary can carry streaming progress + final result across same-process → subprocess → microservice without changing callers, landing first as a provable no-op behind today's `Future` API.

**Architecture:** A `backends/backplane/` package exposes a reactive-streams-shaped two-sided interface (producer `JobSink` / consumer `Publisher[Frame]`) over vendored `Publisher`/`Subscriber`/`Subscription` ABCs. `WorkerPool.submit_job` keeps returning a `Future` (a compat Subscriber fulfills it), so `ws_routes.py` gets zero diff. A stdlib IPC transport (`multiprocessing.Connection` frames + `shared_memory` payload) proves the same interface across a real process boundary.

**Tech Stack:** Python 3.11, stdlib `multiprocessing` / `multiprocessing.shared_memory`, `dataclasses`, `pytest`. **No new runtime dependency** (reactive-streams ABCs are vendored; the worker-loop path uses no event loop / no `anyio`).

**Spec:** `docs/superpowers/specs/2026-07-24-backplane-data-plane-transport-design.md` (accepted, commit `d829a2c`).

## Spec reconciliations discovered during planning

1. **§3.3 `anyio` bridge is deferred, not used here.** The `worker_pool` worker loop runs on a plain thread with no asyncio loop; the facade's compat Subscriber therefore delivers **synchronously in-thread** (`sink.result → subscriber.on_next → fut.set_result`, the exact call chain of today's `fut.set_result`). The `anyio.from_thread` bridge in §3.3 applies only to the async-consumer path (progress→WS), which is a deferred follow-up. This issue needs no `anyio` and no event loop anywhere (the IPC boundary test's parent reads the pipe synchronously too). This is a *truer* no-op than the spec wording.
2. **Cancel-discard rule (carried review note).** `sink.cancelled` is checked at **both** existing worker-loop boundaries — the pre-`execute` skip (`worker_pool.py:718-723`) and the post-`execute` discard (`worker_pool.py:758-763`). Whichever fires first, **no `Result` frame is emitted**; the terminal is `on_error(CANCELLED)`. This is an explicit acceptance criterion of Task 4.
3. **In-proc never reconstructs exceptions.** The compat Subscriber calls `fut.set_exception(err.original)` — the live instance. The `code → exception_factory` reconstruction table is exercised only for `OOM` / `CANCELLED` / `GENERIC`. **`STALE_EPOCH` reconstruction is a facet-3 debt, not a Task 5 one** (see "Deferred debt" below): the in-proc stale path (Task 4) uses `.original`; Task 5's IPC boundary test uses *synthetic* frames, not a real stale-producing worker. A real worker producing `StaleResolutionError` across IPC exists only under facet-3 (subprocess). Resolving it by lazy-importing `StaleResolutionError` (worker_pool.py:43) into `frames.py` would pull a **control-plane** type into the data-plane contract, denting the spec §7 "STALE_EPOCH is an opaque label" boundary — so the intended resolution is a **consumer-injected reconstruction registry** (the parent/Governor supplies `code → factory`), keeping `frames.py` free of control-plane types. Until facet-3, `_reconstruct(STALE_EPOCH)` degrades to `RuntimeError(message)` — safe because nothing in this issue's scope invokes it.
4. **§5 Result is carried OPAQUELY, not decomposed (discovered executing Task 4).** Spec §5 said the compat Subscriber does `Result → fut.set_result((png_bytes, seed))` — i.e. it *decomposes*. But existing tests assert `fut.result() == "test_result"` (`test_worker_pool.py:130` mocks `run_job` returning an opaque string), and production `run_job` returns a `(png, seed)` tuple; decomposing a non-tuple throws and breaks the no-op. So `_FutureBridge` carries the worker's return value through the frame's `image` (`InProcBlob`) **untouched** and calls `fut.set_result(value.image.read_sync())` — reproducing today's `set_result(result)` verbatim for any shape. `sink.result(seed=0, InProcBlob(result))` uses a placeholder seed (the facade bridge ignores it). Typed decomposition into real `(seed, PNG-BlobRef)` is deferred to the streaming/IPC consumers (progress→WS child) that actually need the fields separated; `InProcBlob` was relaxed to hold an opaque payload for this path.

## Global Constraints

- **No new runtime dependency.** Reactive-streams ABCs are vendored under `backends/backplane/reactivestreams/`. `anyio` (already present via Starlette) is **not** used by any code in this issue.
- **Zero `server/ws_routes.py` diff.** The empty diff is the no-op proof. If a change to `ws_routes.py` seems necessary, stop — the design is being violated.
- **Existing suite stays green unmodified:** `tests/test_worker_pool.py`, `tests/test_ws_routes.py`, `tests/test_model_routes.py`. No edits to these files.
- **Python env:** `conda activate stability-toys` before running pytest (per AGENTS.md); use `python`, not `python3`.
- **Method surface is snake_case:** `on_subscribe` / `on_next` / `on_error` / `on_complete`, `request(n)`, `cancel()`.
- **Frame wire records carry `schema_version: int` as the first field.**
- **Commit discipline (AGENTS.md / stopping-point policy):** every commit message includes the FP id `STABL-yoauoqao`, what changed, and the exact next step. End messages with the `Co-Authored-By` trailer.
- **No waveplan for this track** (human-driven, kept close). Do not create waveplan rows or FP subissues unless the human asks.

---

## File Structure

**New package `backends/backplane/`:**

| File | Responsibility |
|---|---|
| `reactivestreams/__init__.py` | Re-export `Publisher`, `Subscriber`, `Subscription` |
| `reactivestreams/subscription.py` | `Subscription` ABC (`request`/`cancel`) |
| `reactivestreams/subscriber.py` | `Subscriber` ABC (`on_subscribe`/`on_next`/`on_error`/`on_complete`) |
| `reactivestreams/publisher.py` | `Publisher` ABC (`subscribe`) |
| `frames.py` | `Ack`/`Progress`/`Result` dataclasses; `BackplaneErrorCode`; `BackplaneError` (+ `from_exc`/`to_exception`); `classify_exception`; `BlobRef` ABC |
| `blob.py` | `InProcBlob`, `SharedMemBlob`; `encode_frame`/`decode_frame` codec (`schema_version`-tagged) |
| `interface.py` | `JobSink` producer ABC; `Backplane` protocol (`open() -> (JobSink, Publisher)`) |
| `inproc.py` | `InProcBackplane`: synchronous `Publisher`/`Subscription`, conflating progress operator, `InProcJobSink` |
| `ipc.py` | `IpcBackplane`: `Connection` frames + `shared_memory` payload, producer-side orphan reaper |

**Modified consumer:**

| File | Change |
|---|---|
| `backends/worker_pool.py` | `submit_job` opens a backplane channel + attaches a compat Subscriber to `job.fut`; `_worker_loop` drives the `JobSink`; cancel paths keep every `fut.cancel()` and additionally arm `sink.cancelled`. |

**New tests:**

| File | Covers |
|---|---|
| `tests/test_backplane_frames.py` | Frame taxonomy, error classification, `to_exception`/`from_exc` |
| `tests/test_backplane_blob.py` | `InProcBlob`/`SharedMemBlob` read/close; codec round-trip incl. `schema_version` |
| `tests/test_backplane_inproc.py` | Ordering, exception-instance preservation, progress conflation under bounded demand, `cancel()` → `sink.cancelled` |
| `tests/test_backplane_ipc.py` | Process-boundary: `ack→progress→result(bytes)→complete`, cancel reaches child, segment unlinked |

---

## Task 1: Reactive-streams ABCs + frame taxonomy + error model

**Files:**
- Create: `backends/backplane/__init__.py` (empty)
- Create: `backends/backplane/reactivestreams/__init__.py`
- Create: `backends/backplane/reactivestreams/subscription.py`
- Create: `backends/backplane/reactivestreams/subscriber.py`
- Create: `backends/backplane/reactivestreams/publisher.py`
- Create: `backends/backplane/frames.py`
- Test: `tests/test_backplane_frames.py`

**Interfaces:**
- Produces: `Publisher.subscribe(subscriber)`, `Subscriber.on_subscribe/on_next/on_error/on_complete`, `Subscription.request(n)/cancel()`. `Ack(job_id, queued_position=0)`, `Progress(job_id, step, total, stage="denoise")`, `Result(job_id, seed, image: BlobRef)`. `BackplaneErrorCode` (`OOM/STALE_EPOCH/CANCELLED/GENERIC/TIMEOUT`). `BackplaneError(code, message="", original=None)` with `.from_exc(exc)` classmethod and `.to_exception()`. `classify_exception(exc) -> BackplaneErrorCode`. `BlobRef` ABC with `async read() -> bytes` and `close() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backplane_frames.py
import concurrent.futures
import pytest
from backends.backplane.frames import (
    Ack, Progress, Result, BlobRef,
    BackplaneError, BackplaneErrorCode, classify_exception,
)
from backends.backplane.reactivestreams import Publisher, Subscriber, Subscription


def test_frames_carry_job_id_and_defaults():
    assert Ack("j1").queued_position == 0
    assert Progress("j1", step=3, total=20).stage == "denoise"
    assert Progress("j1", step=0, total=-1).total == -1  # indeterminate allowed


def test_classify_exception_by_ducktype():
    assert classify_exception(concurrent.futures.CancelledError()) is BackplaneErrorCode.CANCELLED
    assert classify_exception(RuntimeError("CUDA out of memory")) is BackplaneErrorCode.OOM

    class StaleResolutionError(RuntimeError):
        pass
    assert classify_exception(StaleResolutionError("x")) is BackplaneErrorCode.STALE_EPOCH
    assert classify_exception(ValueError("boom")) is BackplaneErrorCode.GENERIC


def test_backplane_error_preserves_live_instance():
    orig = RuntimeError("CUDA out of memory")
    err = BackplaneError.from_exc(orig)
    assert err.code is BackplaneErrorCode.OOM
    assert err.to_exception() is orig  # in-proc: the live instance, not a rebuild


def test_backplane_error_reconstructs_from_code_when_no_original():
    err = BackplaneError(BackplaneErrorCode.CANCELLED, "gone")
    rebuilt = err.to_exception()
    assert isinstance(rebuilt, concurrent.futures.CancelledError)


def test_abcs_are_abstract():
    for abc in (Publisher, Subscriber, Subscription):
        with pytest.raises(TypeError):
            abc()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.backplane'`

- [ ] **Step 3: Write the reactive-streams ABCs**

```python
# backends/backplane/reactivestreams/subscription.py
from abc import ABC, abstractmethod


class Subscription(ABC):
    @abstractmethod
    def request(self, n: int) -> None:
        """Signal demand for up to n more items (n may be a large sentinel for unbounded)."""

    @abstractmethod
    def cancel(self) -> None:
        """Stop the stream; no further on_next/on_complete/on_error after this returns."""
```

```python
# backends/backplane/reactivestreams/subscriber.py
from abc import ABC, abstractmethod

from .subscription import Subscription


class Subscriber(ABC):
    @abstractmethod
    def on_subscribe(self, subscription: Subscription) -> None: ...

    @abstractmethod
    def on_next(self, value) -> None: ...

    @abstractmethod
    def on_error(self, error: Exception) -> None: ...

    @abstractmethod
    def on_complete(self) -> None: ...
```

```python
# backends/backplane/reactivestreams/publisher.py
from abc import ABC, abstractmethod

from .subscriber import Subscriber


class Publisher(ABC):
    @abstractmethod
    def subscribe(self, subscriber: Subscriber) -> None: ...
```

```python
# backends/backplane/reactivestreams/__init__.py
from .publisher import Publisher
from .subscriber import Subscriber
from .subscription import Subscription

__all__ = ["Publisher", "Subscriber", "Subscription"]
```

Create empty `backends/backplane/__init__.py`.

- [ ] **Step 4: Write the frame taxonomy + error model**

```python
# backends/backplane/frames.py
from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class BlobRef(ABC):
    """Transport-resolved payload handle. read() once, then close()."""

    @abstractmethod
    async def read(self) -> bytes: ...

    @abstractmethod
    def close(self) -> None: ...


@dataclass(frozen=True)
class Ack:
    job_id: str
    queued_position: int = 0


@dataclass(frozen=True)
class Progress:
    job_id: str
    step: int
    total: int
    stage: str = "denoise"  # total == -1 => indeterminate


@dataclass(frozen=True)
class Result:
    job_id: str
    seed: int
    image: BlobRef


class BackplaneErrorCode(Enum):
    OOM = "oom"
    STALE_EPOCH = "stale_epoch"
    CANCELLED = "cancelled"
    GENERIC = "generic"
    TIMEOUT = "timeout"  # RESERVED — not emitted this issue (consumer-side today)


def classify_exception(exc: BaseException) -> BackplaneErrorCode:
    """Duck-type classification — no torch / worker_pool imports (avoids a cycle)."""
    name = type(exc).__name__
    if isinstance(exc, concurrent.futures.CancelledError) or name == "CancelledError":
        return BackplaneErrorCode.CANCELLED
    if name == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return BackplaneErrorCode.OOM
    if name == "StaleResolutionError":
        return BackplaneErrorCode.STALE_EPOCH
    return BackplaneErrorCode.GENERIC


def _reconstruct(code: BackplaneErrorCode, message: str) -> Exception:
    """IPC-only: rebuild an exception from a code when the live instance can't cross."""
    if code is BackplaneErrorCode.CANCELLED:
        return concurrent.futures.CancelledError(message)
    if code is BackplaneErrorCode.OOM:
        try:
            import torch  # lazy — parent side has torch
            return torch.cuda.OutOfMemoryError(message)
        except Exception:
            return RuntimeError(message)
    # STALE_EPOCH reconstruction across IPC is deferred (see plan reconciliation #3).
    return RuntimeError(message)


class BackplaneError(Exception):
    def __init__(self, code: BackplaneErrorCode, message: str = "", original: Exception | None = None):
        super().__init__(message or code.value)
        self.code = code
        self.message = message
        self.original = original

    @classmethod
    def from_exc(cls, exc: Exception) -> "BackplaneError":
        return cls(classify_exception(exc), str(exc), original=exc)

    def to_exception(self) -> Exception:
        if self.original is not None:
            return self.original  # in-proc invariant: the live instance
        return _reconstruct(self.code, self.message)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_frames.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add backends/backplane/__init__.py backends/backplane/reactivestreams backends/backplane/frames.py tests/test_backplane_frames.py
git commit -m "feat(backplane): reactive-streams ABCs + frame taxonomy + error model (STABL-yoauoqao) - next: Task 2 BlobRef impls + wire codec

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: BlobRef implementations + wire codec

**Files:**
- Create: `backends/backplane/blob.py`
- Test: `tests/test_backplane_blob.py`

**Interfaces:**
- Consumes: `BlobRef`, `Ack`, `Progress`, `Result`, `BackplaneError`, `BackplaneErrorCode` from `frames.py`.
- Produces: `InProcBlob(data: bytes)`; `SharedMemBlob(name: str, size: int)` (+ classmethod `create(data) -> SharedMemBlob`); `encode_frame(frame) -> bytes` / `decode_frame(raw: bytes) -> frame` with a leading `schema_version` (`SCHEMA_VERSION = 1`). Codec handles `Ack`/`Progress`/`Result`(with `SharedMemBlob`)/`BackplaneError` terminal.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backplane_blob.py
import asyncio
import pytest
from backends.backplane.blob import InProcBlob, SharedMemBlob, encode_frame, decode_frame, SCHEMA_VERSION
from backends.backplane.frames import Ack, Progress, Result, BackplaneError, BackplaneErrorCode


def _read(blob):
    return asyncio.get_event_loop().run_until_complete(blob.read())


def test_inproc_blob_read_returns_bytes_close_noop():
    b = InProcBlob(b"hello")
    assert _read(b) == b"hello"
    assert b.read_sync() == b"hello"  # sync path used by the loop-less facade
    b.close()  # no raise


def test_sharedmem_blob_roundtrip_then_unlink():
    src = SharedMemBlob.create(b"PNGDATA")
    assert _read(src) == b"PNGDATA"
    src.close()  # close + unlink
    # After unlink, re-attaching by name must fail.
    from multiprocessing import shared_memory
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=src.name)


def test_codec_roundtrips_ack_and_progress_with_schema_version():
    for frame in (Ack("j1", 2), Progress("j1", 5, 20, "decode")):
        raw = encode_frame(frame)
        assert raw[0] == SCHEMA_VERSION  # version is the first byte
        assert decode_frame(raw) == frame


def test_codec_roundtrips_result_with_sharedmem_blob():
    blob = SharedMemBlob.create(b"xy")
    out = decode_frame(encode_frame(Result("j1", 42, blob)))
    assert out.seed == 42
    assert isinstance(out.image, SharedMemBlob)
    assert out.image.name == blob.name
    blob.close()


def test_codec_roundtrips_error_terminal_code_only():
    err = BackplaneError(BackplaneErrorCode.OOM, "CUDA out of memory")
    out = decode_frame(encode_frame(err))
    assert out.code is BackplaneErrorCode.OOM
    assert out.message == "CUDA out of memory"
    assert out.original is None  # live instance does not cross the wire
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_blob.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.backplane.blob'`

- [ ] **Step 3: Write the BlobRef impls + codec**

```python
# backends/backplane/blob.py
from __future__ import annotations

import json
from multiprocessing import shared_memory

from .frames import (
    Ack, Progress, Result, BlobRef, BackplaneError, BackplaneErrorCode,
)

SCHEMA_VERSION = 1


class InProcBlob(BlobRef):
    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data

    def read_sync(self) -> bytes:
        """In-proc-only synchronous read — the facade path has no event loop to await."""
        return self._data

    def close(self) -> None:
        self._data = b""


class SharedMemBlob(BlobRef):
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size

    @classmethod
    def create(cls, data: bytes) -> "SharedMemBlob":
        shm = shared_memory.SharedMemory(create=True, size=max(len(data), 1))
        shm.buf[: len(data)] = data
        blob = cls(shm.name, len(data))
        shm.close()  # keep the segment alive (not unlinked); consumer re-attaches by name
        return blob

    async def read(self) -> bytes:
        shm = shared_memory.SharedMemory(name=self.name)
        try:
            return bytes(shm.buf[: self.size])
        finally:
            shm.close()

    def close(self) -> None:
        try:
            shm = shared_memory.SharedMemory(name=self.name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def encode_frame(frame) -> bytes:
    if isinstance(frame, Ack):
        body = {"t": "ack", "job_id": frame.job_id, "pos": frame.queued_position}
    elif isinstance(frame, Progress):
        body = {"t": "progress", "job_id": frame.job_id, "step": frame.step,
                "total": frame.total, "stage": frame.stage}
    elif isinstance(frame, Result):
        assert isinstance(frame.image, SharedMemBlob), "IPC Result carries a SharedMemBlob"
        body = {"t": "result", "job_id": frame.job_id, "seed": frame.seed,
                "blob": {"name": frame.image.name, "size": frame.image.size}}
    elif isinstance(frame, BackplaneError):
        body = {"t": "error", "code": frame.code.value, "msg": frame.message}
    else:
        raise TypeError(f"un-encodable frame: {type(frame)!r}")
    return bytes([SCHEMA_VERSION]) + json.dumps(body).encode("utf-8")


def decode_frame(raw: bytes):
    version = raw[0]
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version}")
    body = json.loads(raw[1:].decode("utf-8"))
    t = body["t"]
    if t == "ack":
        return Ack(body["job_id"], body["pos"])
    if t == "progress":
        return Progress(body["job_id"], body["step"], body["total"], body["stage"])
    if t == "result":
        b = body["blob"]
        return Result(body["job_id"], body["seed"], SharedMemBlob(b["name"], b["size"]))
    if t == "error":
        return BackplaneError(BackplaneErrorCode(body["code"]), body["msg"])
    raise ValueError(f"unknown frame tag {t!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_blob.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backends/backplane/blob.py tests/test_backplane_blob.py
git commit -m "feat(backplane): InProc/SharedMem BlobRef + schema_version wire codec (STABL-yoauoqao) - next: Task 3 in-proc transport + JobSink

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: JobSink + synchronous in-proc transport

**Files:**
- Create: `backends/backplane/interface.py`
- Create: `backends/backplane/inproc.py`
- Test: `tests/test_backplane_inproc.py`

**Interfaces:**
- Consumes: frames + ABCs from Tasks 1-2.
- Produces:
  - `JobSink` ABC: `ack(queued_position=0)`, `progress(step, total, stage="denoise")`, `result(seed, blob)`, `complete()`, `error(err: BackplaneError)`, property `cancelled: bool`.
  - `InProcBackplane(job_id)` with `.open() -> (JobSink, Publisher)`. The `Publisher` is synchronous: `subscribe(sub)` calls `sub.on_subscribe(subscription)`; the `Subscription.request(n)` grants demand; producer `sink.*` calls deliver **synchronously** to the subscriber when demand allows. Progress conflates (latest-wins) while demand is 0; `Result`/terminal are buffered until demand and never dropped. `subscription.cancel()` sets `sink.cancelled = True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backplane_inproc.py
import concurrent.futures
from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.backplane.frames import (
    Ack, Progress, Result, BackplaneError, BackplaneErrorCode,
)
from backends.backplane.reactivestreams import Subscriber


class Recorder(Subscriber):
    def __init__(self, demand=1 << 62):
        self.frames, self.error, self.completed = [], None, False
        self._demand = demand
        self.sub = None

    def on_subscribe(self, subscription):
        self.sub = subscription
        subscription.request(self._demand)

    def on_next(self, value):
        self.frames.append(value)

    def on_error(self, error):
        self.error = error

    def on_complete(self):
        self.completed = True


def test_unbounded_demand_delivers_ack_result_complete_in_order():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder()
    pub.subscribe(rec)
    sink.ack()
    sink.result(seed=7, blob=InProcBlob(b"png"))
    sink.complete()
    assert [type(f).__name__ for f in rec.frames] == ["Ack", "Result"]
    assert rec.frames[1].seed == 7 and rec.completed is True


def test_error_carries_live_exception_instance():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder()
    pub.subscribe(rec)
    orig = RuntimeError("CUDA out of memory")
    sink.error(BackplaneError.from_exc(orig))
    assert rec.error.to_exception() is orig  # live instance preserved


def test_progress_conflates_under_zero_demand_result_never_dropped():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder(demand=0)  # no demand yet
    pub.subscribe(rec)
    sink.progress(1, 20)
    sink.progress(2, 20)
    sink.progress(3, 20)      # only the newest should survive
    sink.result(seed=9, blob=InProcBlob(b"x"))  # must-deliver, buffered
    rec.sub.request(1 << 62)  # open the floodgates
    progresses = [f for f in rec.frames if isinstance(f, Progress)]
    results = [f for f in rec.frames if isinstance(f, Result)]
    assert progresses == [Progress("j1", 3, 20)]  # conflated to latest
    assert len(results) == 1 and results[0].seed == 9  # never dropped


def test_cancel_sets_sink_cancelled():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder()
    pub.subscribe(rec)
    assert sink.cancelled is False
    rec.sub.cancel()
    assert sink.cancelled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_inproc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.backplane.inproc'`

- [ ] **Step 3: Write the JobSink interface**

```python
# backends/backplane/interface.py
from __future__ import annotations

from abc import ABC, abstractmethod

from .frames import BackplaneError, BlobRef


class JobSink(ABC):
    """Producer-side handle the worker drives. ack/progress are non-blocking;
    result/complete/error are synchronous must-deliver."""

    @abstractmethod
    def ack(self, queued_position: int = 0) -> None: ...

    @abstractmethod
    def progress(self, step: int, total: int, stage: str = "denoise") -> None: ...

    @abstractmethod
    def result(self, seed: int, blob: BlobRef) -> None: ...

    @abstractmethod
    def complete(self) -> None: ...

    @abstractmethod
    def error(self, err: BackplaneError) -> None: ...

    @property
    @abstractmethod
    def cancelled(self) -> bool: ...
```

- [ ] **Step 4: Write the synchronous in-proc transport**

```python
# backends/backplane/inproc.py
from __future__ import annotations

from typing import Optional

from .frames import Ack, Progress, Result, BackplaneError
from .interface import JobSink
from .reactivestreams import Publisher, Subscriber, Subscription

_UNBOUNDED = 1 << 62


class _InProcSubscription(Subscription):
    def __init__(self, channel: "_Channel"):
        self._channel = channel

    def request(self, n: int) -> None:
        self._channel.add_demand(n)

    def cancel(self) -> None:
        self._channel.cancel()


class _Channel:
    """Synchronous single-subscriber channel. Progress conflates while demand==0;
    Ack/Result/terminal are buffered must-deliver."""

    def __init__(self):
        self._subscriber: Optional[Subscriber] = None
        self._demand = 0
        self._cancelled = False
        self._pending_progress: Optional[Progress] = None
        self._buffer: list = []          # Ack / Result (must-deliver)
        self._terminal = None            # ("complete", None) | ("error", BackplaneError)

    # producer side -------------------------------------------------------
    def emit_must_deliver(self, frame) -> None:
        self._buffer.append(frame)
        self._drain()

    def emit_progress(self, frame: Progress) -> None:
        self._pending_progress = frame   # latest-wins conflation
        self._drain()

    def emit_terminal(self, terminal) -> None:
        self._terminal = terminal
        self._drain()

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # consumer side -------------------------------------------------------
    def attach(self, subscriber: Subscriber) -> None:
        self._subscriber = subscriber
        subscriber.on_subscribe(_InProcSubscription(self))

    def add_demand(self, n: int) -> None:
        self._demand += n
        self._drain()

    def _drain(self) -> None:
        if self._subscriber is None or self._cancelled:
            return
        # Ack/Result first (ordered, must-deliver), then conflated progress.
        while self._buffer and self._demand > 0:
            self._demand -= 1
            self._subscriber.on_next(self._buffer.pop(0))
        if self._pending_progress is not None and self._demand > 0:
            self._demand -= 1
            frame, self._pending_progress = self._pending_progress, None
            self._subscriber.on_next(frame)
        if not self._buffer and self._terminal is not None:
            kind, payload = self._terminal
            self._terminal = None
            if kind == "complete":
                self._subscriber.on_complete()
            else:
                self._subscriber.on_error(payload)


class _InProcJobSink(JobSink):
    def __init__(self, job_id: str, channel: _Channel):
        self._job_id = job_id
        self._channel = channel

    def ack(self, queued_position: int = 0) -> None:
        self._channel.emit_must_deliver(Ack(self._job_id, queued_position))

    def progress(self, step: int, total: int, stage: str = "denoise") -> None:
        self._channel.emit_progress(Progress(self._job_id, step, total, stage))

    def result(self, seed: int, blob) -> None:
        self._channel.emit_must_deliver(Result(self._job_id, seed, blob))

    def complete(self) -> None:
        self._channel.emit_terminal(("complete", None))

    def error(self, err: BackplaneError) -> None:
        self._channel.emit_terminal(("error", err))

    @property
    def cancelled(self) -> bool:
        return self._channel.cancelled


class _InProcPublisher(Publisher):
    def __init__(self, channel: _Channel):
        self._channel = channel

    def subscribe(self, subscriber: Subscriber) -> None:
        self._channel.attach(subscriber)


class InProcBackplane:
    def __init__(self, job_id: str):
        self._job_id = job_id

    def open(self):
        channel = _Channel()
        return _InProcJobSink(self._job_id, channel), _InProcPublisher(channel)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_inproc.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add backends/backplane/interface.py backends/backplane/inproc.py tests/test_backplane_inproc.py
git commit -m "feat(backplane): JobSink + synchronous in-proc transport with progress conflation (STABL-yoauoqao) - next: Task 4 WorkerPool Future facade (no-op landing)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: WorkerPool Future facade (the no-op landing)

**Files:**
- Modify: `backends/worker_pool.py` (`submit_job`, `_worker_loop` generation branches, cancel paths)
- Test: `tests/test_backplane_facade.py`

**Interfaces:**
- Consumes: `InProcBackplane`, `InProcBlob`, `JobSink`, `BackplaneError`, `Subscriber` from the backplane package.
- Produces: no new public signature — `submit_job(job) -> Future` unchanged. Internally attaches a `_FutureBridge(Subscriber)` per generation job and stores its `JobSink` on the job record for the worker loop.

**Acceptance criteria (carried from review):**
1. `tests/test_worker_pool.py`, `tests/test_ws_routes.py`, `tests/test_model_routes.py` pass **unmodified**. `git diff --stat server/ws_routes.py` is empty.
2. The compat Subscriber touches **only `job.fut`** — never `_job_lock`, never pool methods.
3. Every existing `fut.cancel()` call site is retained (`cancel_job` queued branch `:661`, `cancel_pending_generation_jobs` `:598`, `_mark_running_..._cancel_requested`); `subscription.cancel()` / `sink.cancelled` is *additive*.
4. `sink.cancelled` is checked at **both** worker-loop boundaries (pre-`execute` skip `:718-723`, post-`execute` discard `:758-763`); whichever fires first, **no `Result` is emitted** — the terminal is `on_error(CANCELLED)` → `fut.set_exception(CancelledError())`.

- [ ] **Step 1: Write the failing test**

**Fixture note:** this test reuses the `worker_pool` / `mock_worker_factory` / `mock_mode_config` / `mock_registry` fixtures and the `_gen_job` helper defined in `tests/test_worker_pool.py`. pytest resolves fixtures from names present in a test module's namespace, so importing them (below) works. **If pytest fails to resolve them via import**, relocate those fixture definitions to `tests/conftest.py` — a mechanical move that leaves `test_worker_pool.py`'s own test functions unmodified (the no-op constraint is about test bodies, not fixture location).

```python
# tests/test_backplane_facade.py
"""The facade behaves as a Future exactly as today — proven independently of ws_routes."""
import concurrent.futures
import threading
import pytest
from unittest.mock import Mock
from tests.test_worker_pool import (  # fixtures + helper (see fixture note above)
    worker_pool, mock_worker_factory, mock_mode_config, mock_registry, _gen_job,
)


def test_result_fulfils_future_with_png_seed_tuple(worker_pool, mock_worker_factory):
    worker = mock_worker_factory.return_value
    worker.run_job.return_value = (b"PNG", 321)
    fut = worker_pool.submit_job(_gen_job(worker_pool, req=Mock(), job_id="ok-1"))
    assert fut.result(timeout=2.0) == (b"PNG", 321)


def test_backend_exception_instance_is_preserved(worker_pool, mock_worker_factory):
    worker = mock_worker_factory.return_value
    boom = RuntimeError("backend exploded")
    worker.run_job.side_effect = boom
    fut = worker_pool.submit_job(_gen_job(worker_pool, req=Mock(), job_id="err-1"))
    with pytest.raises(RuntimeError) as ei:
        fut.result(timeout=2.0)
    assert ei.value is boom  # the live instance, unwrapped — not a BackplaneError


def test_running_cancel_discards_result_producer_side(worker_pool, mock_worker_factory):
    started, release = threading.Event(), threading.Event()

    def run_job(job):
        started.set()
        release.wait(timeout=5.0)
        return (b"late", 1)  # produced AFTER cancel — must be discarded

    mock_worker_factory.return_value.run_job.side_effect = run_job
    fut = worker_pool.submit_job(_gen_job(worker_pool, req=Mock(), job_id="cx-1"))
    assert started.wait(timeout=1.0)
    assert worker_pool.cancel_job("cx-1") is True
    release.set()
    with pytest.raises(concurrent.futures.CancelledError):
        fut.result(timeout=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_facade.py -v`
Expected: FAIL — `test_backend_exception_instance_is_preserved` may pass today by luck, but `test_running_cancel_discards_result_producer_side` and the wiring will fail/behave via the old path until the facade is wired. (If all three pass pre-change, the facade is a no-op — proceed; the goal is they stay green after.)

- [ ] **Step 3: Add the FutureBridge Subscriber + open a channel in `submit_job`**

At the top of `backends/worker_pool.py`, add imports:

```python
from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.backplane.frames import Ack, Progress, Result, BackplaneError, BackplaneErrorCode
from backends.backplane.reactivestreams import Subscriber
```

Add the bridge class near the job records:

```python
class _FutureBridge(Subscriber):
    """Fulfils a concurrent.futures.Future from the backplane stream.
    Touches ONLY fut — never pool state / _job_lock (lock invariant)."""

    def __init__(self, fut):
        self._fut = fut

    def on_subscribe(self, subscription):
        subscription.request(1 << 62)  # unbounded — the Future is single-valued

    def on_next(self, value):
        if isinstance(value, Result):
            # facade path has no event loop; InProcBlob exposes a sync read
            png = value.image.read_sync() if isinstance(value.image, InProcBlob) else None
            if not self._fut.done():
                self._fut.set_result((png, value.seed))
        # Ack / Progress: nothing to do for the Future facade

    def on_error(self, error):
        if not self._fut.done():
            self._fut.set_exception(error.to_exception())  # live instance in-proc

    def on_complete(self):
        pass
```

In `submit_job`, for generation jobs, open a channel and attach the bridge, stashing the sink on the record. Modify the registration path (near `self._register_job(job)`):

```python
        self._register_job(job)
        if isinstance(job, GenerationJob):
            sink, publisher = InProcBackplane(job.job_id).open()
            record = self._get_job_record(job.job_id)
            record.sink = sink                      # add `sink=None` field to the record dataclass
            publisher.subscribe(_FutureBridge(job.fut))  # attach + request(unbounded) NOW
        # ... existing self.q.put(job) / put_nowait happens AFTER this block ...
```

**ORDERING INVARIANT (load-bearing — from Task 3 review of `fe960a8`).** The
`subscribe(...)` call above MUST run **before** the job is enqueued (`self.q.put`).
`_FutureBridge.on_subscribe` requests unbounded demand synchronously, so by the time
the worker thread dequeues and emits, the channel is attached with demand and
`sink.result()`/`sink.error()` deliver **synchronously** (matching today's
`fut.set_result`). If the subscribe were placed after the enqueue, the worker could
emit into an unattached channel — the frame buffers, the Future never resolves, and
the no-op tests **hang** rather than fail loudly. Keep subscribe strictly before the
existing `put`/`put_nowait`. (`_Channel`'s docstring states the same invariant.)

Add a `sink: Optional[JobSink] = None` field to the job-record class — the mutable record type returned by `_get_job_record()` and constructed in `_register_job()` (it already holds `.job`, `.state`, `.cancel_requested`, so it is mutable; adding an attribute is safe). Import `JobSink` from `backends.backplane.interface` for the annotation, or type it loosely as `Optional[object]` to avoid the import if the record type lives in a module the backplane would import back (check for a cycle first with `mcp__lsp__references` on the record class).

- [ ] **Step 4: Drive the sink from `_worker_loop`**

Replace the generation branch's terminal calls (around `worker_pool.py:756-768`) so the sink is the delivery path. The pre-`execute` cancelled skip (`:718-723`) already `continue`s without emitting — leave it, but route its terminal through the sink:

```python
            # pre-execute skip (existing :718): cancelled before run
            if job_record is not None and (job_record.cancel_requested or job.fut.cancelled()):
                if job_record.sink is not None:
                    job_record.sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
                job_record.state = "cancelled"
                self._finalize_job_record(generation_job.job_id)
                continue
            ...
            sink = job_record.sink if job_record is not None else None
            if sink is not None:
                sink.ack()
            result = job.execute(self._worker)   # (png_bytes, seed)
            if job_record is not None and job_record.cancel_requested:
                # post-execute discard (existing :758): DO NOT emit Result
                job_record.state = "cancelled"
                if sink is not None:
                    sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
                self._finalize_job_record(generation_job.job_id)
            elif sink is not None:
                png_bytes, seed = result
                sink.result(seed, InProcBlob(png_bytes))
                sink.complete()
                self._finalize_job_record(generation_job.job_id)
```

In the `except Exception as e:` handler (`:770-804`), replace the generation `job.fut.set_exception(e)` calls with `sink.error(BackplaneError.from_exc(e))` (keeping the OOM `_cleanup_vram` pre-step and the `cancel_requested`/`_oom` branch structure unchanged). Non-generation (`ModeSwitchJob`/`CustomJob`) branches keep using `job.fut.set_result/set_exception` directly — they are not backplane jobs.

- [ ] **Step 5: Run the new facade tests + the full existing regression suite**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_facade.py tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py -q`
Expected: PASS (all green). Then verify the no-op proof:

Run: `git diff --stat server/ws_routes.py`
Expected: **empty output** (zero diff).

- [ ] **Step 6: Commit**

```bash
git add backends/worker_pool.py tests/test_backplane_facade.py
git commit -m "feat(backplane): drive WorkerPool via backplane behind preserved Future facade (STABL-yoauoqao) - next: Task 5 stdlib IPC transport + process-boundary test

No-op landing: submit_job still returns a Future (a _FutureBridge subscriber
fulfils it); worker loop drives a JobSink. Live exception instances preserved;
producer-side cancel discard at both worker-loop boundaries. Zero ws_routes.py
diff; existing worker_pool/ws_routes/model_routes suites green unmodified.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Stdlib IPC transport + process-boundary test

**Files:**
- Create: `backends/backplane/ipc.py`
- Test: `tests/test_backplane_ipc.py`

**Interfaces:**
- Consumes: `encode_frame`/`decode_frame`, `SharedMemBlob`, frames, `JobSink`, `Publisher`/`Subscriber`.
- Produces:
  - `IpcJobSink(conn)` — a `JobSink` whose `result(seed, bytes_or_blob)` allocates a `SharedMemBlob`, sends the `Result` frame over `conn`; `error`/`complete` send terminal markers; a producer-side reaper unlinks an unsent/unread segment if the stream ends on error/cancel.
  - `IpcPublisher(conn)` — drives a `Subscriber` by reading frames from `conn` until terminal (`request(n)` bounded demand carried as a reverse control frame; the boundary test uses unbounded demand).
  - `spawn_worker(target, conn)` helper using `multiprocessing.get_context("spawn")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backplane_ipc.py
import multiprocessing as mp
from multiprocessing import shared_memory
import pytest
from backends.backplane.ipc import IpcJobSink, drain_to_subscriber
from backends.backplane.frames import Ack, Progress, Result
from backends.backplane.reactivestreams import Subscriber


def _worker(conn):
    sink = IpcJobSink(conn)
    sink.ack()
    sink.progress(1, 2)
    sink.progress(2, 2)
    sink.result(seed=99, blob=b"PNGBYTES")
    sink.complete()
    conn.close()


class Collector(Subscriber):
    def __init__(self):
        self.frames, self.done = [], False
    def on_subscribe(self, s): s.request(1 << 62)
    def on_next(self, v): self.frames.append(v)
    def on_error(self, e): self.frames.append(e)
    def on_complete(self): self.done = True


@pytest.mark.timeout(30)
def test_frames_and_bytes_cross_a_real_process_boundary():
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_worker, args=(child_conn,))
    proc.start()
    child_conn.close()

    col = Collector()
    result_blob_name = drain_to_subscriber(parent_conn, col)  # returns the Result blob name
    proc.join(timeout=10)

    kinds = [type(f).__name__ for f in col.frames]
    assert "Ack" in kinds and "Result" in kinds and col.done is True
    result = next(f for f in col.frames if isinstance(f, Result))
    # bytes round-tripped through shared memory:
    import asyncio
    assert asyncio.get_event_loop().run_until_complete(result.image.read()) == b"PNGBYTES"
    result.image.close()
    # segment unlinked after close:
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=result_blob_name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_ipc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.backplane.ipc'`

- [ ] **Step 3: Write the IPC transport**

```python
# backends/backplane/ipc.py
from __future__ import annotations

from .blob import encode_frame, decode_frame, SharedMemBlob
from .frames import Ack, Progress, Result, BackplaneError
from .interface import JobSink


class IpcJobSink(JobSink):
    def __init__(self, conn):
        self._conn = conn
        self._cancelled = False
        self._live_blob = None  # for the reaper

    def _send(self, frame) -> None:
        self._conn.send_bytes(encode_frame(frame))

    def ack(self, queued_position: int = 0) -> None:
        self._send(Ack("job", queued_position))

    def progress(self, step: int, total: int, stage: str = "denoise") -> None:
        # metadata only — never allocate shared-mem for progress
        self._send(Progress("job", step, total, stage))

    def result(self, seed: int, blob) -> None:
        data = blob if isinstance(blob, (bytes, bytearray)) else None
        shm_blob = SharedMemBlob.create(bytes(data)) if data is not None else blob
        self._live_blob = shm_blob
        self._send(Result("job", seed, shm_blob))
        self._live_blob = None  # ownership handed to the consumer

    def complete(self) -> None:
        self._conn.send_bytes(b"\x00COMPLETE")

    def error(self, err: BackplaneError) -> None:
        self._reap()
        self._conn.send_bytes(encode_frame(err))

    def _reap(self) -> None:
        if self._live_blob is not None:
            self._live_blob.close()  # unlink the orphan
            self._live_blob = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled


def drain_to_subscriber(conn, subscriber):
    """Synchronously pump frames from conn into subscriber until terminal.
    Returns the Result blob name (for lifecycle assertions), or None."""
    class _Sub:
        def request(self, n): pass
        def cancel(self): pass
    subscriber.on_subscribe(_Sub())
    result_name = None
    while True:
        try:
            raw = conn.recv_bytes()
        except EOFError:
            break
        if raw == b"\x00COMPLETE":
            subscriber.on_complete()
            break
        frame = decode_frame(raw)
        if isinstance(frame, Result):
            result_name = frame.image.name
        if isinstance(frame, BackplaneError):
            subscriber.on_error(frame)
            break
        subscriber.on_next(frame)
    return result_name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_ipc.py -v`
Expected: PASS (1 passed). If `pytest-timeout` is unavailable, drop the `@pytest.mark.timeout(30)` decorator.

- [ ] **Step 5: Run the full backplane + regression suite**

Run: `conda activate stability-toys && python -m pytest tests/test_backplane_frames.py tests/test_backplane_blob.py tests/test_backplane_inproc.py tests/test_backplane_facade.py tests/test_backplane_ipc.py tests/test_worker_pool.py tests/test_ws_routes.py tests/test_model_routes.py -q`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add backends/backplane/ipc.py tests/test_backplane_ipc.py
git commit -m "feat(backplane): stdlib IPC transport (Connection frames + shared_memory payload) proven across a spawn process boundary (STABL-yoauoqao) - next: FP close-out + facet-3 subprocess follow-up

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Acceptance criteria (spec §10)

1. **In-proc transport, no observable client change** → Tasks 3-4. Proof: `git diff --stat server/ws_routes.py` empty; `test_worker_pool.py` / `test_ws_routes.py` / `test_model_routes.py` green unmodified.
2. **Shared-mem/IPC transport across a real process boundary in a test** → Task 5 (`test_backplane_ipc.py`, `spawn` context).
3. **Worker + WS-route code reference only the backplane interface** → worker loop drives `JobSink`; `submit_job` attaches a `Subscriber`; no transport imports in `ws_routes.py` (untouched).

## Deferred (named siblings, not this plan)

- `anyio`-bridged async consumer + Progress→WS wiring + real per-step `callback_on_step_end` emission (progress→WS child).
- rsocket-over-UDS transport (pins rsocket-py, adapts to its ABC surface).
- Facet-3: move CudaWorker to a spawn subprocess (consumes both transports).
  - **Deferred debt — `STALE_EPOCH` IPC reconstruction (from Task 1 review of `de1d772`).** `_reconstruct(STALE_EPOCH)` currently degrades to `RuntimeError(message)`. This only matters when a real worker produces `StaleResolutionError` *across* the IPC boundary, which exists only under facet-3 (in-proc uses `.original`; the Task 5 boundary test uses synthetic frames). Resolve via a **consumer-injected reconstruction registry** (parent/Governor supplies `code → factory`) rather than importing `StaleResolutionError` (worker_pool.py:43) into `frames.py`, preserving spec §7's "STALE_EPOCH is an opaque label" data/control boundary. Must land before facet-3 makes `test_stale_generation_job_raises_before_run_job` cross a process boundary.
- Governor extraction (STABL-vdkdruox): epoch/snapshot authority, admission barrier, lifecycle/respawn.
- `CustomJob` callable → typed control messages; `ControlNetBinding` wire form; `superres` second GPU consumer.
