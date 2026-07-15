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
`cli/go/openapi.snapshot.json` (the module root) and code-generated into
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

## Output and the unquoted-`$(...)` contract

- A single JSON object, **compact** (Go `json.Marshal`, no indentation, no
  structural spaces), on one line, with a trailing newline.
- The goal usage `--controlnet $(stcn ...)` is an **unquoted** command
  substitution, so the emitted bytes must be exactly one shell word. Compact
  marshaling removes *structural* whitespace, but that is not sufficient on its
  own: a user-controlled string **value** containing whitespace (e.g.
  `--id "my id"` → `{"attachment_id":"my id",...}`) still word-splits under an
  unquoted `$(...)`. Verified in zsh.
- Therefore `stcn` guarantees single-token output by **validating every emitted
  string field against a shell-token-safe character set** and rejecting anything
  outside it (see Validation). Combined with compact marshaling, the output is
  always exactly one argv token, so the unquoted form is safe by construction —
  not by convention.
- Quoting — `--controlnet "$(stcn ...)"` — is always safe too and is the
  universally robust form; the field validation is what additionally makes the
  **unquoted** form reliable, which is the ergonomic the tool exists to enable.
- The shape is the schema's shape, so "frozen output contract" is automatic —
  there is no independent field list to freeze.

## Shell-token-safe string fields

Every string that reaches the emitted JSON — `control_type`, `map_asset_ref`,
`model_id`, `attachment_id` — must match `^[A-Za-z0-9._:/-]+$`. This set covers
every realistic input (asset refs like `Rabc123` or `fileref:MAP1`, control
types like `canny`, model ids like `sdxl-canny`, colon- and slash-bearing
refs) while excluding whitespace, glob metacharacters (`*?[]`), quotes, and
other shell-active characters. A value outside the set is a validation error;
`stcn` never emits it. This is a deliberate client-side narrowing of the
server's "any non-empty string" rule — the server still accepts spaces, but
`stcn`'s contract is single-token composability, so it does not produce them.
`:` is intentionally allowed (refs use it) and is not shell-splitting.

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
- Every emitted string field is additionally checked against the
  shell-token-safe set `^[A-Za-z0-9._:/-]+$` (see above); a field with
  whitespace or a shell metacharacter is rejected, guaranteeing the compact
  output is one argv token under unquoted `$(...)`.

## Testing

- **Builder table tests:** each flag maps to its field; unset optionals are
  omitted; `--id` defaults to `control_type`; `--id` overrides it.
- **Validation rejection:** strength out of `[0,2]`, percents out of `[0,1]`,
  `start > end`, missing/empty `control_type` or `map_asset_ref`, malformed
  positional (no `:`), empty `--model`/`--id`. Each returns an error and emits
  nothing.
- **Shell-safety rejection:** each string field (`control_type`,
  `map_asset_ref`, `model_id`, `attachment_id`) with a space, tab, or shell
  metacharacter (`*`, `"`, `$`, …) is rejected; a colon-bearing ref
  (`fileref:MAP1`) and slash/dot/dash values are accepted.
- **Single-token pin:** the full emitted line contains no whitespace except the
  trailing newline — i.e. it is exactly one shell word under unquoted `$(...)`
  splitting, for both structural JSON and every field value.
- **Schema round-trip:** the emitted bytes unmarshal cleanly back into
  `openapi.ControlNetAttachment` with all fields preserved — the
  schema-conformance guarantee.

## Distribution

`stcn` is a **separately installed binary**, alongside `st` — not a dev-only
`go run` utility and not a hidden target. It is a distinct user-facing tool
with its own name, exactly as scoped.

- `cli/go/Makefile` `install` target installs both into `~/.local/bin`,
  matching the repo's other operator tools.
- `cli/go/README.md` command inventory gains an `stcn` row so the tool is
  discoverable next to `st`.
- No alias binaries (`st-cn`, `st-controlnet`) are produced in v1; the name is
  `stcn`. Operators who want aliases can symlink. (Additive later if wanted.)

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
4. Distribution: `Makefile` `install` installs `./cmd/stcn` too; README
   command inventory gains the `stcn` row.
5. `st gen --controlnet $(stcn ...)` end-to-end smoke (documented; the
   pass-through already works, so this is a wiring/usage check), including the
   unquoted-form single-token check.
