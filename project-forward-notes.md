# Project Forward Notes

Live register of current structural shifts and active boundary guidance.
Stable policy lives in `AGENTS.md`. This file is operational and will drift.

---

## Active work

No track is currently in flight. The next likely pickup point is the **st CLI v2
brainstorm** (`STABL-kczspmud`, in-progress, one subtask done) — see "v2 brainstorm"
below.

---

## Recently landed

### AssetStore bucketed interface — merged (PR #3)

**FP:** STABL-hvkybzlg | **Spec:** `docs/superpowers/specs/2026-07-04-asset-store-bucketed-interface-design.md`
**Plan:** `docs/superpowers/plans/2026-07-05-asset-store-bucketed-interface.md`

`AssetStore` reshaped into a Protocol + `InMemoryAssetStore` implementation: flat
named buckets (`upload`, `control_map`, `ref_image`), per-bucket byte budgets with
fail-closed admission (rolls back cleanly under pin pressure — no silent deletion of
unrelated assets), and `promote(ref, target_bucket)` (copy semantics, new ref,
image-decode validated, metadata merged forward). `kind` → `bucket`, `insert` →
`write`, no compatibility aliases.

### Tiered AssetStore persistence — merged (PR #4)

**FP:** STABL-slsbyhga | **Spec:** `docs/superpowers/specs/2026-07-05-tiered-asset-store-persistence-design.md`
**Plan:** `docs/superpowers/plans/2026-07-05-tiered-asset-store-persistence.md`

`TieredAssetStore` composes the bucketed store (hot cache) with an optional
`StorageProvider` (durable tier) via a pure `server/asset_codec.py` seam. Strict
write-through (provider failure discards the just-admitted ref and raises — no
half-persisted state); `resolve` rehydrates from the provider on a memory miss.
Provider selection is a **dedicated** `ASSET_STORE_PROVIDER` env var, decoupled from
the existing `STORAGE_PROVIDER` (which drives the separate `/storage/*` endpoint) —
**Redis is intentionally out of scope** for this tier; only `DISABLED` (default),
`MEMORY`, and `FILESYSTEM` are supported. Lifecycle (the filesystem provider's cleanup
thread) is closed at the server's lifespan shutdown via `close_store()`.

### st read: ControlNet metadata — merged (PR #5)

**FP:** STABL-teiotvmc | **Spec:** `docs/superpowers/specs/2026-07-06-st-read-controlnet-metadata-design.md`
**Plan:** `docs/superpowers/plans/2026-07-06-st-read-controlnet-metadata.md`

`st read <image.png>` now detects all three PNG tEXt chunks the backend writes —
`lcm` (generation params), `controlnet` (per-attachment ControlNet provenance, stamped
alongside `lcm` whenever a generation used ControlNet), and `controlnet_map`
(provenance on standalone control-map files) — via a single `pngmeta.Parse` walk, and
prints one JSON key per chunk found. Output is now always wrapped by chunk keyword
(breaking change from the old flat `lcm`-only shape, by design — no back-compat shim).

### st CLI v1.x point release — done
**FP:** STABL-csqqcjmo (previously mis-cited here as STABL-kczspmud, which is actually
the unrelated, still-open v2 brainstorm below)

All six planned tasks landed: `st modes switch/show/reload` + `stclient.ReloadModes()`,
the `Generate()` callback refactor (`--stream`/`--quiet`), `--controlnet-file`, the
upload bucket argument, and ControlNet config presets.

---

## Active boundary decisions

### CLI-first, always
Frontend has no scope until CLI surface is complete and stable. This is not
a temporary freeze — it reflects the project's delivery philosophy. Any agent
suggesting a "quick UI" for a new capability is out of bounds.

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

### Upload bucket is intent-only
`POST /v1/upload` currently ignores the `type` form field. The bucket argument
(`st upload canny:./map.png`) is client-side semantic labeling only. Do not
assume the server routes on it. Do not add backend changes to make it do so
within this work.

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
