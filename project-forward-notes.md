# Project Forward Notes

Live register of current structural shifts and active boundary guidance.
Stable policy lives in `AGENTS.md`. This file is operational and will drift.

---

## Current objectives

The VRAM umbrella surfaced running HunyuanDiT + ControlNet on enigma (RTX 3090,
24 GB) and has since become a deliberate **worker-as-a-service** refactor, not just
bug fixes. Human-driven, no waveplan, kept close.

**Status as of 2026-07-31:** umbrella `STABL-nvmieaxh` is `in-progress`. **Four children
done** — `STABL-sqqlkmdl` (accounting), `STABL-yoauoqao` (backplane, PR #19),
`STABL-vdkdruox` (Governor, PR #20), `STABL-hjldxurg` (DeviceMemory, PR #24). Two
in-progress with code landed (`STABL-kfekehhc`, `STABL-rgvxuedo`). Three `todo`, one of
which is the load-bearing gap below.

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

**Remaining children:**

- **`STABL-vdkdruox` — Worker Governor — DONE** (PR #20, `2768802`). Authority
  (resolution epoch, active snapshot, admission barrier) now lives in
  `backends/governor.py`. The mode-switch races it was expected to own were fixed as a
  follow-on in PR #26 — see "Authority reservation" under Recently landed.
- **`STABL-qfjfflrx` — parent↔worker seam inventory**: the CUDA-in-parent audit + the
  map of every touchpoint the service split must cover (per-job payload wire form,
  `CustomJob` callable that can't cross a boundary, `superres` as a 2nd in-parent GPU
  consumer, authority placement). Feeds the Governor + facet-3.
- **`STABL-cchxvuhs` — global GPU identity** (UUID-keyed, not local index): governor
  allocates by UUID; `CUDA_VISIBLE_DEVICES` per worker. Not blocking the Governor's
  single-GPU path.
- **Facet-3 — LANDED AND PROVEN ON HARDWARE (2026-07-31).** See "Facet-3 subprocess
  worker" under Recently landed. `STABL-rgvxuedo` M1/M2 merged as PR #23; the
  `WORKER_ISOLATION=subprocess` wiring (`STABL-ptoicrho`) merged as PR #28, which is what
  finally made the code reachable; Task 8's live acceptance passed on enigma the same
  day. The backplane's facet-3 carry-forwards (`STALE_EPOCH` reconstruction registry,
  IPC `request(n)` backpressure) remain in the backplane plan's Deferred section.
- **`STABL-xtkhoidu` — superres, the 2nd in-parent GPU consumer** (split out of
  `STABL-qfjfflrx`). **Accounting half PROVEN ON HARDWARE (2026-08-01)** — see
  "Per-process VRAM attribution" under Recently landed; the parent and the child are now
  separately attributed consumers. **The CUDA-free-parent half is DECIDED but UNBUILT
  (2026-08-02)** — superres gets its own long-lived subprocess child (option 2, sticky);
  sharing the generation child is rejected on the record. Spec:
  `docs/superpowers/specs/2026-08-02-superres-worker-isolation-decision.md`; trigger-gated
  build filed as **`STABL-jylvadvb`**. **The measurement inverted the filed framing:** the
  box already runs two CUDA contexts, so option 2 *moves* the parent's rather than adding
  one, and option 3's entire material win is ~300 MiB of 24 GB. Do not start the build
  without one of the spec's three triggers.

**Timeout ↔ VRAM interaction — BOTH HALVES FIXED (clock 2026-07-31, reap 2026-08-03).** The
umbrella's 2026-07-22 comment notes the flat WS result timeout abandons a long job
*without stopping the backend*, so the worker keeps denoising and holds VRAM with the
result discarded. The `STABL-ltefhpkk` acceptance required `DEFAULT_TIMEOUT=600`, making
an abandoned job hold VRAM for **ten** minutes rather than two — correct for the admission
race, actively worse for this umbrella's goal.

The clock is fixed (`STABL-atzqpcte`, `4646005`, PR #32): two budgets split at
`JobRecord.executing_since`, so waiting is no longer charged to a generation budget.
`DEFAULT_TIMEOUT=600` was obsolete from that moment; **verified 2026-08-03 that it is set
nowhere** — no env file, compose file or config carries it. Nothing to remove.

The reap is **FIXED AND PROVEN ON HARDWARE (2026-08-03)** — `STABL-jredufxb`, branch
`fix/jredufxb-reap`. It was filed because a timed-out generation ran to completion
regardless: `cancel_requested` is read only at job boundaries and `run_job` never checked
it. (The 2026-07-22 comment cites `STABL-qvmdayhb`, which does not resolve; that is why
the concern needed re-filing.)

**Live acceptance, enigma RTX 3090** (`spikes/reap_acceptance.py`), both isolation modes:

```text
                              subprocess          inproc
un-reaped baseline            7.4s                7.3s
wall time to timeout          2.0s (budget 2s)    2.1s (budget 2s)
terminal                      CancelledError      CancelledError
child pid                     153 -> 153          n/a (no pid)
next job on the SAME worker   OK, 0.4s            OK, 0.4s
```

**`pid 153 -> 153` is the load-bearing line.** The job stopped without the process being
killed, so the model stayed resident — the next job returned in 0.4s with no reload. A
kill+respawn would also have "stopped" the job, at the cost of a full reload.

Spec: `docs/superpowers/specs/2026-08-03-timed-out-job-reap-design.md`.
Plan: `docs/superpowers/plans/2026-08-03-timed-out-job-reap.md`.

**The issue's premise — that kill+respawn is the only real mechanism — was already out
of date when it was filed, in the cheap direction.** It was written 2026-07-31, before
`STABL-zueslhah` landed. Progress work installed
`inject_step_progress` (`backends/step_progress.py`) into **every family's denoise loop** —
`callback_on_step_end` where the pipeline supports it, the legacy `callback`/
`callback_steps=1` pair otherwise. That is a per-step re-entry point into a running
generation: cooperative cancellation at **step granularity**, without killing the process
and therefore **without a model reload**. Two of the reap issue's three open design
questions were framed around a cost that has largely evaporated, and "inproc cannot reap"
is no longer true. That is what the fix is built on.

**Traps carried out of the implementation:**

- **A bare cancel flag cleared at job start is WRONG, and it looks right.** The job and
  the cancel travel down two DIFFERENT pipes, so a cancel can reach the child before it
  dequeues the job the cancel names — reliably so, because the child's first job start
  waits on the import lock held by the control thread's `import torch`. Clearing at job
  start discards exactly that cancel. Cancel is therefore **job-scoped**: the control
  message carries the job id and the child keeps a bounded ring of cancelled ids.
- **The exception must be `concurrent.futures.CancelledError`.** `classify_exception()`
  maps only that (or a class literally named `CancelledError`) to `CANCELLED`, and the
  subprocess parent path does no `cancel_requested` remap — a bespoke type arrives as a
  GENERIC failure. `asyncio.CancelledError` is `BaseException`-derived and would escape
  both `except Exception` handlers; `concurrent.futures.CancelledError` is not an alias of
  it and does not.
- **`num_inference_steps` is capped at 50** (pydantic `le=50`), so a "make the job long"
  acceptance cannot work by raising steps. Worse, a fast LCM run then finishes inside any
  fixed multiple of the budget, making "stopped early" true of a job that was never
  reaped. The spike measures an un-reaped baseline first and refuses to run when it is
  under 2x the budget.

**Original trap, still true:** `_emit` swallows every exception on purpose — *"a bad consumer
must never break generation"*. A cancel raised through the progress emitter is therefore
silently eaten. The check belongs in the `_modern` / `_legacy` wrapper, outside that
swallow.

**Umbrella triage, 2026-08-03 — the reap is the last VRAM-pressure gap.** Of the three
children still `todo`, none is about VRAM pressure and none should be started to close the
umbrella: `STABL-govweiat` (CustomJob callable) says in its own text *do not do this
speculatively* — re-audited, nothing broken, nothing blocked, a shape problem;
`STABL-cchxvuhs` (UUID GPU identity) is the multi-GPU future and cannot collide on a
one-card box; `STABL-jylvadvb` (superres child) is deliberately trigger-gated. All three
are long tail. `STABL-jredufxb` is the one live gap and it maps directly onto the
umbrella's own title.

**`STABL-jredufxb` is deliberately NOT parented under the umbrella**, so it will not appear
in `fp tree STABL-nvmieaxh`. It is tracked here instead, precisely because the concern has
already been lost once — the 2026-07-22 comment pointed at an id that does not resolve.

`STABL-xdsdhmov` (ControlNet cache freed on unload/free-vram) is the merged
predecessor (`a3c1c64`): fixed retained ControlNet weights but not the accounting or
recovery facets.

### Mode-switch concurrency — RESOLVED, merged (PR #26)

Both windows are fixed. See "Authority reservation" under Recently landed.

Open, unowned (pre-existing):

| Issue | What |
|---|---|
| `STABL-vwcwmiku` | `.github/workflows/ci.yml` has never run — `.gitignore:21`'s bare `workflows` pattern means it was never committed. The Concourse pipeline in `../continuous` is already fully configured for this repo and is one `fly login` away. |

---

## Recently landed

### Tempo tracing — the last observability pillar — PROVEN ON HARDWARE (PRs #69–#72)

**FP:** STABL-qnlaclof (done) — the fifth and last child of **STABL-oxbwjwvu**
**Merges:** `7a31542` (#69 Governor spans), `eba498e` (#70 propagation), `096a4e5` (#71 SDK + enable)
**Spec:** `docs/superpowers/specs/2026-08-12-tracing-span-map-and-boundary-fixes.md`
**Recipes:** `docs/observability.md` — TraceQL section

A generation is now one trace from the socket to the child process. The
acceptance is a **real generation on enigma**, trace
`c9f1a579d673a671ddbc7b7dfc881cc2`:

```text
span                kind          ms   job.id        mode         outcome
governor.dispatch   INTERNAL   5635.9  090f9c65c115  lcm-general  ok
governor.reload     INTERNAL   3861.3
worker.submit       PRODUCER      6.4  090f9c65c115
worker.execute      CONSUMER   1528.8  090f9c65c115
```

**The second row is the argument for the whole pillar.** 3861 ms of a 5636 ms job
is the demand reload after idle eviction; only 1529 ms is generation. Metrics
could aggregate that and logs could imply it; nothing before this could *show*
it without someone adding a timer first.

`worker.execute` ran in the spawn child — a different process — under a
PRODUCER → CONSUMER pair carrying the same `job.id`. Trace context crosses the
boundary a ContextVar cannot, by riding the job envelope (schema **v3**), exactly
as `job_id` does since `STABL-zuhuxwvf`.

**THE DEFECT PATTERN, three for three, and it is the point.** Every defect review
caught on this pillar was a config or topology error **that passes its own obvious
test**:

1. `Governor._span` replaced the caller's exception with `RuntimeError: generator
   didn't stop after throw()` — a `@contextmanager` yielding twice in its `except`.
   `classify_exception()` maps only `CancelledError` to CANCELLED and the
   subprocess branch tests `terminal_error_code == OOM`, so it silently disabled
   **both** the `STABL-jredufxb` reap and the facet-3 kill+respawn.
2. The `worker.submit` boundary span was never opened and the child's span kept the
   default INTERNAL kind. **A trace-id equality check passes in that state** — the
   acceptance meant to catch it would have signed it off.
3. The gate read `OTEL_EXPORTER_OTLP_ENDPOINT` while the resolver honoured
   `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, so the standard config resolved correctly,
   was discarded, and **logged that no endpoint was configured**.

In all three the system reports success while the telemetry is absent — the exact
failure class this umbrella exists to remove, reproduced by the tooling built to
remove it. Assume it of the next pillar.

**The endpoint trap, measured rather than reasoned:**

```text
OTLPSpanExporter(endpoint="http://c:4318")   -> http://c:4318
OTLPSpanExporter() with the env var set      -> http://c:4318/v1/traces
```

An explicit `endpoint=` is used **verbatim**; only the env-var path appends. The
design note said SDK exporters "append the signal path themselves" — true of the
variable, false of the argument, and the facade passes the argument. Every span
would have POSTed to the collector root and 404'd with the gate reporting healthy.

**Pinned at opentelemetry 1.27.0, for a reason unrelated to tracing.**
`requirements.txt:2` caps protobuf at 4.25.4 because `Dockerfile:175` needs
`mediapipe==0.10.14` for ControlNet pose. Unconstrained pip takes otel 1.44 and
protobuf 7.35.1; constrained, pip's own pick backslides to **1.15.0** (a 2023
release) when 1.27.0 satisfies everything — so the version must be explicit in
**both** directions. `opentelemetry-sdk` alone has no protobuf dependency; the
constraint is entirely the OTLP/HTTP exporter. Filed as `STABL-ksjdjawc`.

**Query traps worth carrying:**

- **A `limit` plus a chatty endpoint hides your data.**
  `{ resource.service.name = "stability-toys" }` returned 25 traces, **24 of them
  `GET /health`**, and the generation was crowded out — which reads as "tracing is
  broken". Query by `name` or `.job.id`, never by service alone. Same
  drowning-in-health-checks failure `docs/observability.md` documents for LogQL,
  one signal over, and it caught the author of that warning.
- **`job_id` in a log line is `.job.id` on a span.** That is the whole log↔trace
  join; no trace id in the log line is needed.
- **A 200 on `POST /v1/traces` is the collector accepting a batch**, not delivery
  to Tempo. Read a trace back by id.

**Build wiring, all found the hard way in a live container (PR #72):**

- `requirements.txt` now **includes** `requirements-tracing.txt`. The pins lived
  only in the separate file, so `pip install -r requirements.txt` in a container
  installed nothing and gave no hint. The split was copied from the conditioning
  pattern — but that one is split *because* it needs `--no-deps` to keep Jupyter
  out of the image, and tracing has no such quirk.
- **`make dev-build` cannot add a dependency.** `live-test.Dockerfile` has **zero
  `pip install` lines** — it is `FROM ${BASE_IMAGE}` plus `COPY`. A
  `requirements*.txt` change needs `docker compose -f docker-cuda.yml build`.
  Documented at `docker-compose.dev.yml:13`; still cost a session.
- **`make dev` now passes `--force-recreate`** — compose reuses a container when
  only the `env_file` changed, so an `env.dev` edit reads as a broken feature.
- **`python /app/spikes/x.py` puts `/app/spikes` on `sys.path`, not `/app`.**
  Python uses the *script's* directory, not the CWD, so `import server` fails and
  reads as a broken container. The tracing acceptance bootstraps its own path; the
  other six spikes still need `PYTHONPATH=/app`.

**Not proven, and stated rather than glossed:** the log↔trace join on the dev
container. `env.dev` sets no `LOG_FORMAT` (text by design for humans) and
`stability-toys-dev` is not currently discovered by promtail, so its lines are not
in Loki at all. The trace half is proven; the join is available on any deployment
where both are on.

**Deferred, filed, not done:** `STABL-bhkhuuir` (the `worker.recovery` event —
works in isolation, fails after `tests/test_controlnet_constraints.py` runs, with
the branch confirmed entered while `_dispatch_span` is None), `STABL-ksjdjawc`
(OTel version ceiling), `STABL-vhqfpsjp` (Loki/promtail/Tempo self-metrics are not
scraped). No `governor.admit` span — the barrier is inline and sub-millisecond,
and its output is a decision rather than a duration.

Older "Recently landed" notes are archived in `docs/notes-ledge.md`.

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
