# stcn — ControlNet Attachment Former — Design

**FP issue:** STABL-nzrqaxla
**Status:** Authority artifact for the stcn v1 track.

## Goal

`stcn` forms a single ControlNet attachment from command-line flags and emits
it as compact JSON that `st gen --controlnet` accepts verbatim. It replaces
hand-writing `--controlnet '<json>'` strings — which today are parsed as an
untyped `map[string]any` with no schema behind them — with output that is
**forced to match the server's contract** by construction.

The one-line goal test:

```bash
st gen --prompt "a bridge" --cfg 7.2 \
  --controlnet $(stcn canny:Rmap1 --strength 0.8) \
  --controlnet $(stcn depth:Rmap2 --strength 0.4)
```

## Why this is schema-forced (the core idea)

The server's `ControlNetAttachment` (pydantic, `server/controlnet_models.py`)
is part of the FastAPI OpenAPI document. That document is snapshotted to
`cli/go/internal/openapi/openapi.snapshot.json` and code-generated into
`cli/go/internal/openapi/openapi.gen.go` via `make gen` (downspec 3.1→3.0 +
oapi-codegen), with `internal/openapi/drift_test.go` failing if the snapshot
diverges from the live `/openapi.json`.

`stcn` builds and marshals the **generated `openapi.ControlNetAttachment`
struct** — never a hand-authored map. The field names, JSON tags, and types
therefore come from the server's own schema. If the server contract changes,
the drift guard flags it and regeneration updates stcn's output shape. There
is no second, hand-maintained mirror of the attachment shape.

## Scope

`stcn` is a **pure, offline, single-purpose** binary:

- **In:** flags describing one attachment.
- **Out:** one compact JSON object on stdout.
- **Never:** contacts the server, uploads files, sends a generation, or reads
  any config. It has no `pkg/stclient` or config dependency — there is nothing
  to configure when the network is never touched.

One invocation forms exactly one attachment. Multiple attachments are composed
externally by repeating `--controlnet $(stcn ...)` on the `st gen` line (the
flag is already repeatable), so `stcn` needs no array or multi-attachment mode.

**v1 covers `map_asset_ref` only** (bring-your-own control map). The schema's
other input path — `source_asset_ref` + a required `preprocess` block — is a
richer nested object and is deferred to a follow-on. The generated type already
carries those fields, so adding them later is additive.

## Architecture

New binary at `cli/go/cmd/stcn`, module
`github.com/darkbit/stability-toys/cli/st`.

| Unit | Responsibility |
| --- | --- |
| `cmd/stcn/main.go` | flag parsing, wiring, exit codes |
| attachment builder (in `cmd/stcn`) | flags → `openapi.ControlNetAttachment`, with client-side validation |
| `internal/openapi` (reused) | the generated `ControlNetAttachment` type — the only external dependency |

The builder is a pure function (`buildAttachment(opts) (openapi.ControlNetAttachment, error)`)
so it is testable without a process or I/O.

## Flag surface

Head as a single positional argument:

```
stcn <control_type>:<map_asset_ref>
```

- `<control_type>` → `control_type` (required, non-empty).
- `<map_asset_ref>` → `map_asset_ref` (required, non-empty).
- Split on the first `:` only, so refs containing `:` survive.

Optional field flags, each mapping to exactly one schema field:

| Flag | Field | Rule |
| --- | --- | --- |
| `--strength <f>` | `strength` | `0.0 ≤ f ≤ 2.0` |
| `--start <f>` | `start_percent` | `0.0 ≤ f ≤ 1.0` |
| `--end <f>` | `end_percent` | `0.0 ≤ f ≤ 1.0` |
| `--model <s>` | `model_id` | non-empty when set |
| `--id <s>` | `attachment_id` | non-empty; **defaults to `control_type`** |

Cross-field rule: when both are set, `start ≤ end`.

Any optional flag left unset is omitted from the JSON (the field is a pointer /
`omitempty`), so the server applies its mode-policy defaults — identical to the
existing attachment path. `attachment_id` is the sole "optional-looking"
field that is always emitted, because the server requires it; it defaults to
the `control_type` string.

**Attachment-id uniqueness is the composer's responsibility.** Two attachments
of the same `control_type` composed onto one `st gen` line will both default to
that type as `attachment_id`; the user sets distinct `--id` values in that case.
This matches the "compose externally" model — `stcn` forms one object and does
not know about its siblings.

## Output

- A single JSON object, **compact** (Go `json.Marshal`, no indentation, no
  spaces), on one line, with a trailing newline.
- Compactness is a hard requirement, not a preference: the goal usage
  `--controlnet $(stcn ...)` is an **unquoted** command substitution, so any
  interior whitespace would be word-split by the shell into multiple arguments.
  `json.Marshal` output (`{"attachment_id":"canny",...}`) is exactly one shell
  token.
- The shape is the schema's shape, so "frozen output contract" is automatic —
  there is no independent field list to freeze.

## Validation and errors

- All range and presence rules above are checked in `buildAttachment` before
  anything is emitted.
- On any violation: nothing is written to stdout, a one-line diagnostic is
  written to stderr, and the process exits non-zero.
- The server re-validates on `/generate`; `stcn`'s checks are a fast, offline
  first line that keeps invalid JSON from ever being formed. `stcn` does not
  attempt to enforce policy-level bounds (a control type's
  `[min_strength, max_strength]`) — those are mode-config-dependent and live on
  the server; `stcn` enforces only the schema's own absolute ranges.

## Testing

- **Builder table tests:** each flag maps to its field; unset optionals are
  omitted; `--id` defaults to `control_type`; `--id` overrides it.
- **Validation rejection:** strength out of `[0,2]`, percents out of `[0,1]`,
  `start > end`, missing/empty `control_type` or `map_asset_ref`, malformed
  positional (no `:`), empty `--model`/`--id`. Each returns an error and emits
  nothing.
- **Compactness pin:** marshaled output contains no space, tab, or newline
  except the single trailing newline — i.e. it is exactly one shell word under
  `$(...)` splitting.
- **Schema round-trip:** the emitted bytes unmarshal cleanly back into
  `openapi.ControlNetAttachment` with all fields preserved — the
  schema-conformance guarantee.

## Non-goals (v1)

- No `source_asset_ref` / `preprocess` (deferred, additive later).
- No sending, uploading, or any network/server contact.
- No config or server-URL handling.
- No multi-attachment / array output mode (composition is repeated
  `--controlnet $(stcn ...)`).
- No changes to `st` or `pkg/stclient`.
- No policy-range (min/max strength per control type) validation — server-owned.

## Implementation order (input to the plan)

1. `buildAttachment(opts) (openapi.ControlNetAttachment, error)` with the full
   validation table (pure, fully unit-tested).
2. Compact marshal + stdout emit; compactness and round-trip pins.
3. `cmd/stcn/main.go`: positional + flag wiring, exit codes, `--help`.
4. `st gen --controlnet $(stcn ...)` end-to-end smoke (documented; the
   pass-through already works, so this is a wiring/usage check).
