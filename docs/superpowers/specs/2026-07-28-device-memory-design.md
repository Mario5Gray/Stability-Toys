# DeviceMemory — backend-neutral device memory accounting

**FP:** STABL-hjldxurg (child of umbrella STABL-nvmieaxh)
**Status:** design approved; awaiting implementation plan
**Depended on by:** STABL-ptoicrho (subprocess wiring consumes this)
**Related:** STABL-cchxvuhs (UUID identity), STABL-qfjfflrx (superres as 2nd consumer), STABL-sqqlkmdl (driver-truth predecessor)

---

## 1. Motivation

The VRAM umbrella's stated goal is *honest accounting*, but the accounting itself leaks its hardware abstraction. VRAM is read three ways in three places, each a direct `torch.cuda` call that assumes an **in-process CUDA context**:

- `ModelRegistry.get_available_vram()` — `torch.cuda.mem_get_info()[0]` (`model_registry.py:240`)
- `InProcessWorkerHandle.health()` — `torch.cuda.mem_get_info()` (`worker_handle.py:168`)
- `Governor._build_runtime_status()` — `torch.cuda.memory_allocated()/memory_reserved()` (`governor.py:513-514`); only `total` already routes through the registry (`:515`)

Two concrete failures follow:

1. **The reads break out-of-process.** Facet-3 (`SubprocessWorkerHandle`, merged PR #23, currently dormant) moves the worker — and its CUDA context — into a spawn child. The parent then has no context, so every reader above returns 0/stale. The leak is literal: `SubprocessWorkerHandle.health()` returns `vram_free_bytes=0, vram_total_bytes=0` (`worker_handle_subprocess.py:141`).
2. **The reads burn a context.** `torch.cuda.mem_get_info()` initializes the primary CUDA context on first call. A health/status read on an idle, pre-first-load server permanently burns the ~0.5–1.5 GB context (umbrella finding #2: contexts release only at process exit). Narrow window, but real.

`DeviceMemory` replaces the three ad-hoc readers with one backend-neutral authority: **driver truth (NVML by UUID, no CUDA context in the caller) merged with a registry of per-consumer torch pools.** The registry shape is the payoff — superres (a second GPU consumer, `STABL-qfjfflrx`), future multi-GPU (`STABL-cchxvuhs`), and non-CUDA backends (MLX/RKNN unified memory) all fit by calling `register()`, with no Governor change.

---

## 2. Contract

New module `backends/device_memory.py`, **torch-free** (imports no `torch`; providers import `pynvml`/`psutil` lazily, inside themselves).

```python
class MemoryTopology(Enum):
    DISCRETE = "discrete"   # separate device pool (CUDA)
    UNIFIED  = "unified"    # shares host RAM (MLX, RKNN, CPU)

@dataclass(frozen=True)
class ConsumerMemory:            # one GPU consumer's slice
    label: str                  # "worker", "superres", "video", ...
    pid: int | None
    allocated_bytes: int        # torch/framework pool: live allocations
    reserved_bytes: int         # torch/framework pool: full held pool
    stale: bool = False         # snapshot-substituted last-known (see §2.3)

@dataclass(frozen=True)
class DeviceMemorySnapshot:
    device_uuid: str            # ties to STABL-cchxvuhs
    topology: MemoryTopology
    total_bytes: int            # device total (DISCRETE) | host total (UNIFIED)
    free_bytes: int             # driver-truth free (NVML | psutil)
    consumers: tuple[ConsumerMemory, ...]

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    @property
    def unattributed_bytes(self) -> int:
        # driver-observed device usage not attributed to any registered
        # consumer's torch pool: CUDA contexts, non-torch workspaces
        # (cuDNN/cuBLAS/xformers), and unregistered/other-process usage.
        # RESERVED, not allocated: a consumer's cached-but-free pool blocks
        # belong to that consumer; subtracting allocated would mislabel
        # torch's cache as unexplained device usage.
        # On DISCRETE single-consumer (enigma) this ~= the CUDA context.
        # On UNIFIED this is host RAM incl. OS + unrelated processes:
        # INFORMATIONAL ONLY — never alert on it.
        return max(0, self.used_bytes - sum(c.reserved_bytes for c in self.consumers))

class MemoryConsumer(Protocol):
    def pool_stats(self) -> ConsumerMemory: ...   # MUST return stale=False
    def reclaim(self) -> None: ...                # soft pool-trim; no-op where N/A

class Registration(Protocol):
    def close(self) -> None: ...   # idempotent + crash-safe; deregisters

class DeviceMemory(Protocol):
    def register(self, c: MemoryConsumer) -> Registration: ...
    def snapshot(self) -> DeviceMemorySnapshot: ...   # driver truth + consumer pools; refreshes cache
    def cached_snapshot(self) -> DeviceMemorySnapshot: ...  # last computed; NO fan-out
    def available_for_load(self) -> int: ...          # cheap, topology-aware, NO fan-out
    def reclaim(self) -> None: ...                     # fan out to live consumers
```

`snapshot()` computes fresh with a bounded best-effort fan-out (§2.3) and stores the result. `cached_snapshot()` returns that stored result with **no fan-out** — it is how views (the registry, §4.1) read `consumers[]` without inheriting the hang that `available_for_load()` and the admission path are kept clear of.

**Pre-seed:** before the first `snapshot()`, the cache holds a **driver-truth-populated snapshot with `consumers=()`** — `total_bytes`/`free_bytes` from NVML (cheap, no context, no fan-out), zero consumers. So `total`/`free` reads are correct from startup, and `allocated`/`reserved` read 0 while nothing is registered (correct: nothing *is*). It is never `None` and never a zero-total snapshot. `available_for_load()` never depends on the cache regardless.

**Atomicity:** `snapshot()` stores by rebinding a single reference to a new frozen `DeviceMemorySnapshot`; `cached_snapshot()` reads that reference. The rebind is atomic under the GIL, so a concurrent `/status` read mid-refresh sees either the whole old snapshot or the whole new one — never a torn read. No lock needed.

### 2.1 `unattributed_bytes` is derived, not stored

A `@property`, computed from `used_bytes` and `consumers[]`, never a stored field that can drift from its inputs. It clamps at 0; a **negative pre-clamp value is a signal** (a consumer over-reporting its pool) and `snapshot()` implementations SHOULD debug-log it. `context_overhead_bytes` was rejected: it lies on shared GPUs and under-describes even enigma (context + non-torch workspaces).

### 2.2 Invariants (hard contract lines)

- **`available_for_load()` MUST NOT fan out.** Driver truth only (NVML/psutil), zero consumer round-trips. This is a **liveness property, not an optimization**: the admission path cannot block on the wedged worker the system exists to survive.
- **`reclaim()` ≠ recovery AND ≠ teardown.** `reclaim()` is the *soft, opportunistic* pool-trim (today's `empty_cache` sites), fanned out to **live consumers only**. It is NOT durable OOM recovery — that is `WorkerHandle` kill+respawn (umbrella finding #3), and lives nowhere near `DeviceMemory`. It is also NOT teardown — teardown flushes the departing consumer's own pool inline (controlnet-cache clear → `gc` → `empty_cache`) as part of `unload()`/`stop()`. No future reader may "fix" OOM by calling `reclaim()`.
- **`Registration.close()` is idempotent + crash-safe, and the parent is the sole closer.** A SIGKILLed subprocess runs no clean `close()`; the parent-side `WorkerHandle` closes on reap. Consumers never close their own Registration.

### 2.3 `stale` is snapshot-authoritative

`pool_stats()` MUST return `stale=False` — a consumer cannot self-declare staleness (it would let a hung consumer lie about its own liveness). Only the `snapshot()` fan-out sets `stale=True`, and only when a bounded `pool_stats()` read times out: it substitutes the consumer's **last-known** `ConsumerMemory` with `stale=True` (never omits it — omission would inflate `unattributed_bytes` by that consumer's footprint). The substitution event is logged at the moment it happens (same implementation-side duty as the negative-residual log). `sampled_at`/age is rejected as YAGNI until a real consumer of staleness-age appears.

---

## 3. Providers and the driver-truth source

Driver truth comes from **NVML keyed by UUID**, not `torch.cuda.mem_get_info()`.

- `nvmlInit()` → `nvmlDeviceGetHandleByUUID(uuid)` → `nvmlDeviceGetMemoryInfo()` → `(total, free, used)`.
- NVML is a **driver query — it initializes no CUDA context in the caller**. That is precisely why it replaces `mem_get_info()` (which does): the parent reads device truth without paying for a context.
- **UUID seeds `STABL-cchxvuhs`.** v1 (single GPU) resolves the device UUID once at startup, stores it on the provider, populates `snapshot().device_uuid`. Handle-by-UUID also means `CUDA_VISIBLE_DEVICES` reordering cannot misroute the query — the cchxvuhs motivation, obtained for free. v1 does not implement multi-GPU allocation; it reads one UUID.

Providers, selected by topology once at startup:

| Provider | When | free/total source |
|---|---|---|
| `CudaDeviceMemory(uuid)` | DISCRETE + NVIDIA, NVML init OK | NVML by UUID |
| `UnifiedDeviceMemory` | UNIFIED (MLX/RKNN/CPU) | psutil (host RAM) |
| `NullDeviceMemory` | NVML init fails on a CUDA host | 0 / UNKNOWN |

**`NullDeviceMemory` degrades; it does not borrow.** Falling back to `mem_get_info()` through a consumer's context would reintroduce the exact leak this design removes. Null preserves today's CUDA-unavailable → 0 behavior (the `:513-514` guard), so behavior stays consistent on non-NVIDIA dev hosts.

New CUDA-image-only dependency: `nvidia-ml-py` (small pure-Python bindings over the already-present `libnvidia-ml`), pinned like the conditioning extras. `psutil` for unified hosts.

---

## 4. ModelRegistry migration, WorkerHealth, Governor status

The three direct-torch readers collapse into views over one `DeviceMemory`.

### 4.1 ModelRegistry becomes a pure view

The registry stops calling `torch.cuda` and reads **only** through the snapshot surface — never `pool_stats()` directly (calling `pool_stats()` would drag the wedged-worker hang into `/status` through the back door, defeating the §2.2 no-fan-out invariant):

- `get_total_vram()` → `cached_snapshot().total_bytes`
- `get_available_vram()` → `available_for_load()` (was `mem_get_info()[0]` at `:240`)
- `get_allocated_vram()` / `get_reserved_vram()` → the worker's entry in **`cached_snapshot().consumers[]`** (was `memory_allocated/reserved` at `:220`/`:198`) — a no-fan-out read of the last computed snapshot, inheriting last-known+`stale` semantics for free. The registry never calls `snapshot()` (fresh, fan-out) itself; `/status` and diagnostics refresh the cache.

**Load-time measurement is the one exception, and reads fresh.** The Governor's `_load_mode` computes a model's footprint before/after the load (`governor.py:330-331`, `:348-352`) to register `vram_bytes` per model. At those two sites it reads the worker consumer's `allocated` from a **fresh `snapshot()`** (bounded fan-out), NOT the cache-backed `get_allocated_vram()` — a cold/stale cache would register garbage `vram_bytes`. This is safe: the load path is not the admission hot path (it already blocks on model I/O), so fan-out is permitted there. The cache serves `/status`-shaped readers only.

### 4.2 WorkerHealth

`InProcessWorkerHandle.health()` (`worker_handle.py:166-176`) stops calling `torch.cuda.mem_get_info()` directly (`:168`); it reads `vram_free/total` from the injected `DeviceMemory`. The handle gains a `DeviceMemory` reference (injected like the factory). This removes the last direct-torch VRAM read from the handle, and closes the `worker_handle_subprocess.py:141` `0,0` leak (in-proc in v1; the subprocess consumer arrives with ptoicrho).

### 4.3 Governor status

`_build_runtime_status` swaps its direct reads at `governor.py:513-514` for the worker's entry in the latest snapshot's `consumers[]` — the same read path as the registry view, one consistent source. `total` already routes through the registry (`:515`).

---

## 5. Registration lifecycle — bind/unbind across kill+respawn

**Sole owner = the parent-side `WorkerHandle`. The consumer never closes its own Registration.** This is forced by SIGKILL: a killed subprocess runs no `finally`, no `close()`, so the parent must be the closer. `Registration` lifetime is strictly nested inside worker liveness.

The four lifecycle events, against real code:

1. **`start()` (load):** build/spawn → live (`READY`) → `self._registration = device_memory.register(worker_consumer)`. Register *after* liveness so `pool_stats()` is immediately valid. `health()` now sources VRAM from `device_memory`, so the leak closes the instant `start()` returns.
2. **clean `unload()` (mode switch / evict / shutdown):** `close()` the Registration **first** (so no fan-out samples a tearing-down consumer), *then* drop the worker and flush its own pool inline (teardown, per §2.2).
3. **OOM-alive kill+respawn (T7, child alive-but-poisoned):** Governor recovery = unregister_model + `handle.stop()` + `_reload_from_snapshot()`. At the memory layer: `stop()` (`worker_handle_subprocess.py:146`) closes the old Registration (crash-safe) + `proc.kill()/join`; the reload's `start()` opens a **new** Registration. Close-old + open-new, never mutate-in-place — a fresh process is a fresh consumer identity (pid changes).
4. **frameless death (SIGKILL, child already dead):** `stop():146` is a no-op join *but still closes the Registration* — the parent closes what the corpse never could. Respawn opens a new one. This is the only place the parent-sole-closer invariant is *tested* rather than merely stated.

**Registration-gap honesty and debounce.** Between close-old and open-new, `snapshot().consumers` lacks the worker, so `unattributed_bytes` transiently spikes — which is **correct** (mid-reclaim, the killed process's memory is genuinely unattributed until the OS reaps it). Therefore **any consumer of `unattributed_bytes` MUST debounce across every registration gap — initial load, mode-switch, AND respawn — not respawn only.** A first mode load is itself a gap (no consumer registered until `start()` completes); alerting without debounce would trip on the first load. Spec line, not a code change.

**superres generalizes the shape (`STABL-qfjfflrx`).** The second consumer just `register()`s its own Registration — owned by the superres service, in-parent, long-lived — and `snapshot().consumers = [worker, superres]`, `unattributed = used − (worker.reserved + superres.reserved)`. Zero Governor change.

---

## 6. Wiring

- **Singleton + selection:** `get_device_memory(uuid: str | None = None)` (like `get_worker_pool()`). Provider chosen by topology once at startup; no per-call selection. The `uuid` parameter is present from v1 as a **forward-proof**: `None` resolves to the single/default device. v1 has exactly one device, so callers pass nothing; multi-device (§6.1) resolves a specific device behind the same accessor — no v1 call site changes.
- **Injection:** startup lifespan creates `DeviceMemory` → `ModelRegistry(device_memory)` → `Governor(registry, ...)`; the handle gets the ref too. DI constructor param, default `get_device_memory()` — same pattern as `mode_config`/`registry`.
- **Worker consumer adapter:** `WorkerMemoryConsumer(MemoryConsumer)` wraps the worker. `pool_stats()` → `ConsumerMemory(label="worker", pid, allocated=memory_allocated(), reserved=memory_reserved(), stale=False)`; `reclaim()` → `empty_cache()` (in-proc) / reclaim frame (subprocess, later). This is what `register()` takes.

### 6.1 Multi-device composition (cchxvuhs-facing)

The contract is **per-device by construction**, and multi-GPU / hybrid discrete+unified compose above it with **no `DeviceMemory` contract change** — only a new aggregation layer, which is `STABL-cchxvuhs`. This subsection documents that extension point so composability is proven on paper, not merely asserted; **none of it is built in v1** (v1 is exactly one device).

- **Per-device is already true.** `DeviceMemorySnapshot.device_uuid`, `CudaDeviceMemory(uuid)`, and per-device `register`/`snapshot`/`available_for_load` all key on a single device. `MemoryTopology` is per-provider, so each device carries its own DISCRETE/UNIFIED classification and its own `unattributed_bytes` semantics (the UNIFIED caveat is per-device, not global).
- **The aggregator (cchxvuhs).** A `Devices` map, `UUID → DeviceMemory`, holding one provider per physical device. On a hybrid box that is heterogeneous — e.g. `{uuid_a: CudaDeviceMemory(DISCRETE), uuid_b: UnifiedDeviceMemory(UNIFIED)}` — with no special-casing, because each entry reports its own topology.
- **Consumer→device binding.** A worker (or superres) uses exactly one device, so it `register()`s with that device's `DeviceMemory`. The consumer→device assignment is the Governor's allocation decision (cchxvuhs: allocate by UUID, `CUDA_VISIBLE_DEVICES` per worker); `DeviceMemory` itself never needs to know about siblings.
- **Admission stays per-device.** A load targets a device, then asks *that* device's `available_for_load()`. There is no global "available VRAM" — headroom is always a per-device question, which the contract already models.
- **The only accessor change is additive.** `get_device_memory(uuid=None)` (§6, forward-proofed in v1) becomes UUID-resolving in the aggregator; `None` keeps meaning "the default/single device" for legacy single-GPU callers. `ModelRegistry` becomes device-aware the same way (it already delegates through `DeviceMemory`, so it inherits whichever device it is handed).

Net: multi-GPU and hybrid discrete/unified require a `Devices` map + a consumer→device allocation policy (cchxvuhs), sitting **above** an unchanged per-device `DeviceMemory`. This spec builds the single-device case; the seam is UUID-keyed from day one so the aggregator is pure addition.

---

## 7. Acceptance criteria + behavioral no-op protocol

The no-op proof is **behavioral-equivalent-within-delta, NOT byte-identical** — the free/total source genuinely changes (`cuMemGetInfo` → NVML), so byte-identical is impossible and claiming it would be dishonest. This is explicitly a weaker proof than the backplane/Governor byte-proofs, justified because those were pure moves with identical semantics while this changes the source.

1. Contract types present in `backends/device_memory.py`, torch-free.
2. Providers `CudaDeviceMemory` (NVML-by-UUID) / `UnifiedDeviceMemory` (psutil) / `NullDeviceMemory` (degrade, no context-borrow); topology selection at startup.
3. `unattributed_bytes` derived `@property`, `max(0, used − Σ reserved)`, reserved-not-allocated (docstring reason), negative→debug-log, UNIFIED informational-only caveat.
4. Invariants enforced: `available_for_load()` no fan-out; `reclaim()` ≠ recovery ≠ teardown (soft, live-consumers-only); `Registration.close()` idempotent + crash-safe + parent-sole-closer.
5. `stale` snapshot-authoritative: `pool_stats()` always `False`; only `snapshot()` fan-out substitutes last-known with `stale=True` on timeout + logs.
6. Registry is a pure view (`cached_snapshot`/`available_for_load`/`consumers[]`); no direct torch; no fan-out from the registry (load-time footprint measurement is the sole exception and reads a fresh `snapshot()`, §4.1).
7. `WorkerHealth.health()` sources VRAM from `device_memory`; `worker_handle.py:168` `mem_get_info` removed.
8. Governor `_build_runtime_status:513-514` → worker `consumers[]` entry; total via registry view.
9. **Behavioral no-op:** (a) first live enigma read measures + documents the NVML-`used` vs `mem_get_info`-free delta and *why it's safe* (admission = observability + `can_fit`, not a hard gate); (b) recorded load/admission sequence replays to **identical admit/deny outcomes**; (c) `/models/status` VRAM within the stated ±tolerance; (d) frozen VRAM tests (`test_model_registry`, `test_model_lifecycle`) repointed to **stub the `DeviceMemory` provider interface**, never pynvml call sites — the `STABL-mrrrbmjp` mock-ordering vaccine.
10. Registration lifecycle: start→register, unload/stop→close, double-close idempotent (v1 in-proc; the SIGKILL-reap path is exercised at ptoicrho).
11. `/models/status` VRAM block gains a `stale: bool` field, default `False` in v1 (external-schema stability across the facet boundary; bool only, no age).
12. Spec honesty line carried into code comments where relevant: v1 = behavioral-within-delta, not 0-byte.

---

## 8. Scope

**v1 builds:** the contract; three providers + topology selection; the singleton + injection into registry/handle/Governor-status; the in-proc `WorkerMemoryConsumer`; the §4 rewire (registry view, WorkerHealth, Governor status); the behavioral acceptance.

**Deferred (shape present, built by the named consumers):**
- Out-of-process consumer proxy + IPC-backed `pool_stats()` → **STABL-ptoicrho** (which also exercises the real SIGKILL-reap-close at `worker_handle_subprocess.py:146`).
- superres as a second registered consumer → **STABL-qfjfflrx**.
- Multi-GPU allocation by UUID → **STABL-cchxvuhs** (v1 reads exactly one UUID).

**Dependency direction:** ptoicrho depends on DeviceMemory, not vice versa — v1 needs nothing subprocess-shaped. The ptoicrho M-A picklability spike (resolve-in-child) is unaffected and still valid.

---

## 9. Deferred and cross-references

- Umbrella finding #2 (context is unreclaimable pre-exit): the motivation for NVML-no-context reads.
- Umbrella finding #3 (OOM poisons the context; in-proc recovery can't fix it): the reason `reclaim()` ≠ recovery is a hard boundary.
- `STABL-sqqlkmdl` (driver-truth via `mem_get_info`): the predecessor this generalizes and moves off the CUDA-context requirement.
- `STABL-mrrrbmjp` (torch-mock ordering pollution): the anti-pattern the provider-interface test seam must not recreate in pynvml form.
