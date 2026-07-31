# Parent↔worker boundary contract — re-audit

**Issue:** STABL-qfjfflrx (child of STABL-nvmieaxh)
**Date:** 2026-07-31 | **Audited against:** `main @ 5b353e2`

The original inventory was written *before* the backplane, the Governor, authority
reservation, DeviceMemory, and facet-3 landed. This re-audits every claim against the code
as it stands, corrects four that are now wrong, and states what is genuinely left.

## Verdict first

The boundary is **mostly built**. What the inventory framed as a large refactor is now two
concrete gaps and one design decision. The recommended authority placement has already
landed; the biggest remaining blocker to a CUDA-free parent is superres, not the worker.

---

## Corrections to the filed inventory

**1. `ControlNetBinding` needs no bespoke wire form.** The inventory records it as
"resolved objects — may hold PIL/bytes, need a wire form". It is eight plain
`str`/`bytes`/`float` fields (`server/controlnet_execution.py:59-67`) and round-trips
pickle cleanly. That assumption is what justified deferring it out of the job envelope, and
the deferral silently broke ControlNet under `WORKER_ISOLATION=subprocess`
(**`STABL-spxwqlan`**, fixed in PR #34, confirmed live across multiple models).

**2. Authority placement is settled, not pending.** "`ActiveModelSnapshot` … currently
lives with the worker in-process" is stale. `STABL-ltefhpkk`/`iuiwzthc` moved it firmly
parent-side: the Governor reserves epoch + resolved model at *enqueue* time and admission
binds a generate to that reservation. The inventory's own recommendation — keep queue,
epoch, snapshot and admission in the parent — is what shipped.

**3. `CustomJob` does not cross the boundary, and is not a blocker.** The inventory says
its callable "cannot cross a process boundary. Must be redesigned into typed control
messages." In the dispatch loop only `GenerationJob` takes the `handle.submit()` path;
`CustomJob` runs in-parent via `job.execute(...)` (`governor.py:927`). Its one use is idle
eviction, whose handler operates on *Governor* state. It also already reads
`_worker_available()` rather than `handle.worker`, so eviction is correct under subprocess
isolation. Redesign is optional cleanup, not a prerequisite.

**4. The `is_available` capability gate is overstated.** `lcm_sr_server.py:219` sits in an
`else` branch: `BACKEND == "cuda"` and `BACKEND == "rknn"` are both handled explicitly
above it, so the torch import never runs for either supported backend. It is a fallback for
other backend ids, not a CUDA-in-parent blocker.

---

## CUDA in the parent — what is actually left

| site | status |
| --- | --- |
| `ws_routes` inline `mem_get_info` | **FIXED here** — now reads `pool.get_vram_stats()` |
| `lcm_sr_server:219` `is_available` | unreachable for `cuda`/`rknn`; see correction 4 |
| `superres_service.py` | **STILL OPEN** — the real blocker, see below |
| `backends/*` (worker, registry, cache) | correct home; unchanged |

### superres is the remaining blocker — and an accounting hole today

`server/superres_service.py` is a second, independent in-parent GPU consumer: its own
device (`cuda:0`, `:90`, `:501`), its own `is_available` (`:407`), its own `empty_cache`
(`:457-458`), its own OOM classification (`:613`), its own queue and workers.

Beyond blocking a CUDA-free parent, this is a **live accounting gap**: superres is not
registered with `DeviceMemory`, so every byte it holds lands in `unattributed_bytes`. The
umbrella's whole premise is honest VRAM accounting, and the second-largest consumer on the
box is invisible to it. Registering it is worth doing *independently* of where it ends up
running.

Filed as **`STABL-xtkhoidu`**.

---

## The service interface, as it actually is

The parent talks to the worker through exactly this set (grepped from `server/`):

- **Authority / admission** — `admit_generation`, `get_active_model_snapshot`,
  `current_resolution_epoch`, `get_pending_mode`
- **Job** — `submit_job`, `wait_for_result`, `cancel_job`
- **Lifecycle** — `switch_mode`, `reload_current_mode`, `reload_if_current`,
  `unload_current_model`, `free_vram`, `shutdown`
- **Status** — `get_current_mode`, `is_model_loaded`, `get_queue_size`, `get_vram_stats`

`wait_for_result` joined this set in `STABL-atzqpcte` and `get_vram_stats` here. Both were
added by *removing* a transport-side implementation, which is the shape to keep: when a
transport needs something about the worker, the answer is a seam method, not a local read.

## Per-job payload crossing the boundary

**IN** — `req`, `job_id`, `resolution_epoch`, `init_image`, `controlnet_bindings`
(envelope v2). Completeness is now enforced structurally by `CARRIED_JOB_FIELDS` /
`NOT_CARRIED_JOB_FIELDS` asserted against `GenerationJob`'s dataclass fields, so adding a
field without wire support fails a test.

**OUT** — result rides `SharedMemBlob`; errors ride the sink terminal with a
`BackplaneErrorCode`.

**PROGRESS** — still nothing for generation. `Progress` frames have a producer API
(`IpcJobSink.progress`) but nothing emits them for a generate and the only subscribers are
`_FutureBridge` / `_SubprocessFutureBridge`, which wait for a terminal. This remains the
inventory's most accurate open observation, and it is the other half of the
`STABL-atzqpcte` complaint: a long model load still looks like a hang to the client.

## Trap worth carrying

A source-text assertion (`"X" not in inspect.getsource(mod)`) is tripped by the **comment
explaining why X was removed**. This bit twice today. Strip comment lines before matching,
or assert on the call signature rather than the token.
