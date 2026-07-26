# Project Forward Notes

Live register of current structural shifts and active boundary guidance.
Stable policy lives in `AGENTS.md`. This file is operational and will drift.

---

## Current objectives

The VRAM umbrella surfaced running HunyuanDiT + ControlNet on enigma (RTX 3090,
24 GB) and has since become a deliberate **worker-as-a-service** refactor, not just
bug fixes. Human-driven, no waveplan, kept close. Two children merged; the rest
below are `todo`.

### Worker-as-a-service — umbrella `STABL-nvmieaxh`

Four-part model (control plane vs data plane): **Governor** (parent-side control) +
**Worker Handle** (locality-agnostic) + **Backplane** (data plane) + **Worker**
(executor). Boundary = the job queue; scale path in-proc thread → subprocess (spawn,
NOT fork — CUDA contexts don't survive fork) → microservice, one contract throughout.
Authority (resolution epoch, active snapshot, admission barrier) stays PARENT-side in
the Governor.

The enigma logs separated one apparent "leak" into three distinct failures — status:

1. **Accounting was fiction → FIXED.** `get_available_vram()` used
   `total - memory_reserved()` (torch's pool vs nameplate, ignoring the CUDA context /
   library workspaces / other processes), over-committing → OOM. **`STABL-sqqlkmdl`
   (done, merged `243455e`)** flipped it to driver truth via
   `torch.cuda.mem_get_info()`.
2. **Post-free residual is the CUDA context, not a torch leak.** After free-vram torch
   reports fully freed; the ~0.5–1.5 GB left in `nvidia-smi` is the per-process context,
   unreclaimable by `empty_cache()` — only process exit frees it. Not fragmentation.
   → drives the subprocess direction (facet-3).
3. **OOM poisons the context; in-process recovery can't fix it.** `_cleanup_vram` runs
   on the worker thread but `empty_cache`/`del` cannot drop a poisoned context. Durable
   fix = subprocess isolation (kill + respawn), which the backplane now makes possible.

**Merged children:**

- **`STABL-sqqlkmdl` (done)** — driver-truth VRAM accounting (`mem_get_info`).
- **`STABL-yoauoqao` (done, PR #19, merge `919a1d6`)** — the **Backplane** data-plane
  transport. `backends/backplane/`: vendored reactive-streams ABCs (no rsocket-py dep),
  frames + `BlobRef` + `schema_version` codec, synchronous in-proc transport behind a
  preserved `submit_job()→Future` facade (0-byte `ws_routes.py` diff = the no-op proof),
  and a stdlib IPC transport proven across a real spawn boundary incl. the cross-process
  cancel channel. See "Recently landed" for the carry-forwards.

**Remaining children (`todo`):**

- **`STABL-vdkdruox` — Worker Governor** (control plane): worker lifecycle
  (spawn/ready/health/kill/respawn-with-backoff), authoritative state, dispatch +
  admission. **This is where the epoch/snapshot authority lives**, so it owns the fix
  for the mode-switch races below. Consumes the backplane + a Worker Handle interface;
  extract from `WorkerPool` with an InProcHandle first (prove zero behavior change).
  Natural next step now the backplane has landed. **Scoping locked (2026-07-25):**
  pure no-op extraction v1, `WorkerPool` = thin facade, seam inventory split (WorkerHandle
  contract + CUDA audit in v1; ControlNetBinding wire form + `CustomJob`→typed-message map
  deferred to facet-3). Sub-question (b) demand-reload/eviction leans Governor; (a) dispatch
  replace-vs-wrap left for brainstorming. See `fp context STABL-vdkdruox`.
- **`STABL-qfjfflrx` — parent↔worker seam inventory**: the CUDA-in-parent audit + the
  map of every touchpoint the service split must cover (per-job payload wire form,
  `CustomJob` callable that can't cross a boundary, `superres` as a 2nd in-parent GPU
  consumer, authority placement). Feeds the Governor + facet-3.
- **`STABL-cchxvuhs` — global GPU identity** (UUID-keyed, not local index): governor
  allocates by UUID; `CUDA_VISIBLE_DEVICES` per worker. Not blocking the Governor's
  single-GPU path.
- **Facet-3 (no issue yet)** — move CudaWorker to a spawn subprocess; the durable OOM
  recovery + timed-out-job reap (`STABL-qvmdayhb`). Depends on the backplane (done) +
  Governor. The backplane's facet-3 carry-forwards (cancel_job→subscription wiring,
  `STALE_EPOCH` reconstruction registry, IPC `request(n)`/`job_id`/`result()` hardening)
  are tracked in the backplane plan's Deferred section.

`STABL-xdsdhmov` (ControlNet cache freed on unload/free-vram) is the merged
predecessor (`a3c1c64`): fixed retained ControlNet weights but not the accounting or
recovery facets.

### Mode-switch concurrency — folded into the Governor

A generate admitted concurrently with a mode switch resolves against transient
authority. Two windows of the one switch, **same root, one fix — now a Governor
concern** (authority placement, `STABL-vdkdruox`), not a standalone track:

| Issue | Window | Failure |
|---|---|---|
| `STABL-ltefhpkk` | old snapshot still live (old epoch) | `StaleResolutionError` at execution; retry works |
| `STABL-iuiwzthc` | new model still loading (`_active_snapshot` transiently `None`, `worker_pool.py:305-385`) | spurious "ControlNet provider not yet implemented"; retry works |

Fix both by resolving/admitting/stamping the generate against the mode it
**targets**, established atomically with the switch — implemented once in the Governor
where epoch/snapshot authority lives, not spread across the boundary.

Open, unowned (pre-existing):

| Issue | What |
|---|---|
| `STABL-vwcwmiku` | `.github/workflows/ci.yml` has never run — `.gitignore:21`'s bare `workflows` pattern means it was never committed. The Concourse pipeline in `../continuous` is already fully configured for this repo and is one `fly login` away. |

---

## Recently landed

### Backplane — data-plane transport — merged (PR #19)

**FP:** STABL-yoauoqao (done) | **Merge:** `919a1d6`
**Spec:** `docs/superpowers/specs/2026-07-24-backplane-data-plane-transport-design.md`
**Plan:** `docs/superpowers/plans/2026-07-24-backplane-data-plane-transport.md`

The first concrete piece of the worker-as-a-service split. New package
`backends/backplane/`. Five TDD tasks, each reviewed green; 156 passed; **0-byte
`server/ws_routes.py` diff** is the no-op proof. Design decisions worth carrying:

- **Vendored** reactive-streams `Publisher`/`Subscriber`/`Subscription` ABCs under
  `backends/backplane/reactivestreams/` — no runtime rsocket-py dep. The interface is
  rsocket-*shaped*; the rsocket transport is a follow-up child.
- `WorkerPool.submit_job` still returns a `Future`; a `_FutureBridge` Subscriber
  (attached with unbounded demand **before** enqueue) fulfils it, and the worker loop
  drives a `JobSink`. **No `anyio` / no event loop** — the worker loop is a plain
  thread, so the facade delivers synchronously in-thread. The `anyio.from_thread`
  bridge is only for the deferred async progress→WS consumer.
- **Result is carried opaquely**, not decomposed into `(png, seed)` — existing tests
  mock `run_job`→`"test_result"`; the bridge does `fut.set_result(image.read_sync())`
  verbatim. Typed seed/PNG split is deferred to the streaming consumers.
- **`BlobRef`** is transport-resolved (in-proc bytes / IPC `shared_memory`, read-once
  `close()`+unlink; `SharedMemBlob.create` unregisters the producer's `resource_tracker`
  to dodge the spawn handoff race). Frame codec carries a leading `schema_version` byte.
- **IPC** = duplex `multiprocessing.Connection` frames + `shared_memory` payload;
  inbound cancel = `subscription.cancel()` → reverse control frame → `sink.cancelled`
  (a subprocess worker can't read `cancel_requested`), proven across a real spawn
  boundary.

Deferred to facet-3 / Governor (tracked in the plan's Deferred, NOT done): wire
production `cancel_job → record.sink` subscription; `STALE_EPOCH` IPC reconstruction
via a consumer-injected `code→factory` registry (keep control-plane types out of
`frames.py`); IPC `request(n)` backpressure + `IpcJobSink(conn, job_id)` + `result()`
signature. Next umbrella step: the Governor (`STABL-vdkdruox`).

### HunyuanDiT family profile — merged (PR #17)

**FP:** STABL-ichgkgno | **Spec:** `docs/superpowers/specs/2026-07-16-hunyuandit-family-profile-design.md`
**Plan:** `docs/superpowers/plans/2026-07-17-hunyuandit-family-profile.md`
**Merge:** `a62bfb1` — also carried `STABL-fdurqnnn` (`drift check` now exits 0,
down from 12 stale anchors) and `STABL-svpnjbjh` (`make drift` targets).

Family dispatch is now a neutral registry (`FamilyProfile` + exact-one
`resolve_family`) resolved before mode policy, with CUDA workers selected from
one family-by-platform binding table by lazy dotted reference. HunyuanDiT runs
txt2img with zero or one Canny ControlNet through the production `WorkerPool`:
`(supports_img2img=False, supports_controlnet=True, combined=False)`, native
BERT+mT5 conditioning, `control_image`, `use_resolution_binning=True`, native
DDPMScheduler. Live acceptance at 1024x1024 peaks at 18.80 GiB — 2.57 GiB under
the spike observation, 5.2 GiB under the 24 GiB operator floor.

Three family-specific traps worth carrying forward to the next family:

- **Attention processor swaps are not universally safe.** `HunyuanDiT2DModel`
  passes rotary positional embeddings through `cross_attention_kwargs`, which
  `XFormersAttnProcessor` and `SlicedAttnProcessor` warn about and drop, so the
  transformer denoises without positional information and returns noise.
  `CudaWorkerBase.supports_attention_processor_swap` gates both; it costs ~10%
  per iteration and, measurably, no VRAM at all.
- **Shared ControlNet kwargs are not universally accepted.**
  `HunyuanDiTControlNetPipeline.__call__` takes `controlnet_conditioning_scale`
  but has no `control_guidance_start`/`end`, so the SD/SDXL-shaped kwargs are
  filtered per family.
- **Control-map fixtures are family-sensitive.** A border-to-border edge map
  drives this Canny checkpoint into noise while an inset one is fine.
  `tests/hunyuan_control_map.py` is the single fixture shared by the acceptance
  and `scripts/hunyuan_cn_probe.py` — they previously held separate maps, and
  the probe validating its own map while the acceptance ran a different one cost
  a long investigation into worker code that was correct throughout.

Diagnostics: `HUNYUAN_DEBUG_DUMP=1` dumps the exact control image, call kwargs,
conditioning keys, and pipe state per job under `HUNYUAN_DEBUG_ROOT`, read-only
and inert when unset. `scripts/hunyuan_cn_probe.py` runs the family with no app
plumbing and replays a dumped control image via `CONTROL_IMAGE`. Together they
split an output-quality failure into image-bytes versus pipe-state causes in one
run — worth reaching for before reading worker code.

Depth and Pose are registered and user-reachable but only Canny is live-verified.
Hunyuan img2img, combined img2img+ControlNet, materialized Hunyuan conditioning,
and `/models/status` family exposure remain deferred.

### Pluggable prompt conditioning + Compel long prompts — merged

**FP:** STABL-hvalobvn (done, incl. docs/container/live closeout `STABL-dxxgoevd`)
**Spec:** `docs/superpowers/specs/2026-07-09-long-prompt-compel-design.md`
**Plan:** `docs/superpowers/plans/2026-07-10-pluggable-prompt-conditioning.md`

CUDA workers now use a Stability-Toys-owned prompt-conditioning seam. Native
prompt delegation remains the empty-configuration default; per-mode
`conditioning.service: compel` opts CUDA modes into local Compel materialization
for SD1.5 and SDXL. Compel is pinned in `requirements-conditioning.txt` and
installed with `--no-deps` in CUDA-capable images to avoid Notebook/Jupyter
dependency creep.

The consumer boundary is intentionally CUDA-local and live: every SD1.5/SDXL
generation branch, including txt2img, img2img, ControlNet, combined
img2img+ControlNet, and both latent entry points, invokes one chain per job and
then validates the artifact against the exact target pipeline immediately before
calling Diffusers. Compatibility failures are structural consumer failures and
never enter native fallback; `native_on_failure` only covers configured-service
invocation failure and can restore native truncation.

Direct/proxy conditioning, Redis/Qdrant artifact storage, non-CUDA materialized
consumers, frontend changes, and new CLI flags remain deferred. Operators should
enable Compel only in CUDA deployment config, not in shared repo defaults.

### Combined img2img + ControlNet — merged (PR #6)

**FP:** STABL-ztaxgbhv (parent, 10 children) | **Spec:** `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`
**Plans:** `docs/superpowers/plans/2026-07-08-img2img-controlnet-{groundwork,pipeline-wiring,followups}.md`

img2img + ControlNet in one request now executes end-to-end on CUDA (SD1.5 and
SDXL), WS/CLI only. Both workers run
`StableDiffusion(XL)ControlNetImg2ImgPipeline.from_pipe(self.pipe, ...)` — zero
extra base-model VRAM — with `image=` (init) and `control_image=` (map) kept
distinct via an `image_kwarg` override on `_build_controlnet_kwargs`, shared-VAE
dtype normalization, and a 2%-tolerance aspect-ratio gate that rejects naming the
offending `attachment_id`. Requests are capability-gated: the WS guard
(`reject_combined_img2img_controlnet`) reads
`BackendCapabilities.supports_img2img_and_controlnet` (also surfaced in
`GET /models/status`) and rejects fail-fast **before preprocessing** on non-capable
backends. Design decisions: `denoise_strength` × `start/end_percent` pass through
without renormalization (low strength + narrow window can yield no visible
conditioning — documented caveat, not a bug); combined results stay uncached.
HTTP `/generate` intentionally cannot express img2img (no `init_image_ref`) —
adding it would be a separate API decision. This track also recorded a
cross-file `sys.modules`/`lru_cache` diffusers-stub pollution failure between
`test_cuda_worker_controlnet.py` and `test_worker_controlnet_metadata.py`; that
no longer reproduces (2026-07-20: 34 passed in one session, and both files are
clean in the full suite), most likely resolved when `STABL-ichgkgno` removed the
family-string branching those stubs interacted with. No FP issue was filed.

The combined-track test-hygiene follow-ups (`STABL-bclnlnzd` torch stubbing,
`STABL-zisphapv` Miniforge pin) are both now **done**.

### Earlier landed (settled; forward-relevant detail folded into boundary decisions)

- **AssetStore bucketed interface** — `STABL-hvkybzlg` (PR #3). Protocol +
  `InMemoryAssetStore`; flat `upload`/`control_map`/`ref_image` buckets, per-bucket
  fail-closed byte budgets, `promote(ref, target_bucket)`.
- **Tiered AssetStore persistence** — `STABL-slsbyhga` (PR #4). `TieredAssetStore` =
  bucketed hot cache + optional `StorageProvider` via `server/asset_codec.py`; strict
  write-through; `ASSET_STORE_PROVIDER` env (`DISABLED`/`MEMORY`/`FILESYSTEM`, Redis
  out of scope).
- **st read: ControlNet metadata** — `STABL-teiotvmc` (PR #5). Detects `lcm`,
  `controlnet`, `controlnet_map` PNG tEXt chunks; output wrapped by chunk keyword.
- **st CLI v1.x point release** — `STABL-csqqcjmo`. `st modes switch/show/reload`,
  `Generate()` `--stream`/`--quiet`, `--controlnet-file`, upload bucket arg,
  ControlNet presets.

---

## Active boundary decisions

### CLI-first, always
Frontend has no scope until CLI surface is complete and stable. This is not
a temporary freeze — it reflects the project's delivery philosophy. Any agent
suggesting a "quick UI" for a new capability is out of bounds.

### `st gen --reset` was removed — use `conflate off` / `on`

`gen --reset` (STABL-ykdsormc) added a per-run clean slate, then was reverted
(`0779f06`). With no explicit prompt it resolved to an **empty** prompt — the
conflation baseline was the only prompt source, and the WS handler defaults a
missing prompt to `""` (`ws_routes.py:370`), rendering noise. The clean-slate
path is `st conflate off; st gen ...; st conflate on`. Do not re-add `--reset`
without first solving that empty-prompt resolution.

### `--json` output contract is frozen
`st gen --json` emits exactly `{"output","seed","storage_key","storage_url"}` —
indented, terminal (single object, not stream). Do not add fields, do not
change to NDJSON. Scripts depend on this shape. The new NDJSON surface is
`--stream`.

### `pkg/stclient` is a shared surface — design accordingly
It was always intended as the shared layer between CLI and a future MCP server.
Changes to `stclient` must be clean enough to serve both. Do not add CLI-specific
concerns (flag state, stderr, cobra) into `stclient`.

### Backend WS re-attach is deferred
The backend does not support re-attaching to an in-flight job by `jobId` from
a second connection. `st watch --job <id>` (the other half of the canonical
pipeline) is blocked on this. Do not attempt an IPC workaround. Leave the
`--stream` output contract stable so `st watch` can be added non-breakingly
when backend support lands.

### Upload `type` routes to a store bucket (STABL-kcjkrpry)
`POST /v1/upload` reads the `type` form field and routes the file to a store
bucket: `canny`/`depth`/`pose` → the durable `control_map` bucket,
`image`/`ref` → `ref_image`, and any other or missing type → the ephemeral
`upload` bucket (5-minute TTL). Routed buckets are validated as decodable
images (400 otherwise); the `upload` bucket stays lenient. The response is
`{fileRef, bucket, width?, height?}` and `st upload --json` surfaces the
server-resolved bucket. The mapping is a local constant in
`server/upload_routes.py`, intentionally decoupled from the ControlNet
registry. (Supersedes the earlier "intent-only" note — the server now routes.)

---

## Deferred tracks (explicit, with rationale)

| Track | Why deferred |
|-------|-------------|
| `st watch --job` | Backend has no WS re-attach; IPC ruled out |
| `st watch --all` | Needs a backend queue-state endpoint that doesn't exist yet |
| MCP server (`st serve mcp`) | Second consumer of `pkg/stclient`; right after CLI surface stabilizes |
| Batch (`--batch`, `--variations`, `--concurrency`) | Requires goroutine pool + N WS connections; non-trivial concurrency model |
| Config management (`st config get/set/edit`) | Nice-to-have; unblocked but not urgent |
| `st modes set-default` | No `POST /api/modes/default` endpoint in backend |
| `--dry-run` | Deferred — scope (params only vs WS mock) not decided |
| `st doctor` | Deferred post-point-release |
| Non-CUDA img2img+ControlNet execution | Compounds onto the existing non-CUDA ControlNet deferral; explicit non-goal even after CUDA combined path (`STABL-ztaxgbhv`) ships |

---

## v2 brainstorm
**FP brainstorm:** `fp://brainstorm?id=ifnwzfkdyysvlweulcigrnubknzswavj`

Eight clusters (A–H). MCP server (F) is the highest-value deferred item —
it reuses `stclient` with minimal new logic. Batch (B) is highest effort.
No v2 plan exists yet. v1.x must ship first.

---

## Structural notes

- `modesCmd` in `modes.go` retains `RunE: runModes` even after subcommands are
  added. Cobra routes child invocations to child commands; the parent `RunE`
  fires for `st modes` with no args (list behavior preserved).
- `multipartFile(filename, data, fields map[string]string)` in `http.go` is
  the existing extension point for extra form fields. SuperRes uses it for
  `magnitude`. Upload bucket uses it for `type`. No new mechanism needed.
- `buildGenParams` in `gen.go` is the layering point: config → baked PNG →
  flags. Preset expansion (`@name`) and `--controlnet-file` both hook here,
  after the existing `--controlnet` JSON block.
