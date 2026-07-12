# `st gen` Conflation and Command History - v1 Design

**Status:** Approved design, awaiting written-spec review

## Goal

Add persistent command history to `st` and an opt-in conflation mode for
iterating on prior `st gen` requests. A user can submit only changed generation
parameters and let `st` inherit the rest from either a recent eligible run or a
pinned history entry.

History is global and always enabled. Conflation is persistent policy that can
be enabled, disabled, or changed independently of history collection.

Example:

```console
$ st gen "a painting of a horse" --cfg 3.2 --size 1024x1024
$ st conflate
Conflating recent successful gen runs.
$ st --cfg 4.3
initial command [id=1]: st gen --prompt 'a painting of a horse' --cfg 3.2 --size 1024x1024
next command [id=2]: st gen --prompt 'a painting of a horse' --cfg 4.3 --size 1024x1024
```

## Scope

V1 includes:

- an append-only global history of every `st` invocation;
- local monotonic integer history IDs beginning at `1`;
- a persistent conflation policy;
- conflation eligibility fixed to the `gen` command family;
- recent-history selection by exit status;
- an explicitly pinned history baseline;
- raw and effective representations for generation history;
- an abstract storage boundary with a filesystem implementation; and
- XDG state storage in one `history.jsonl` plus small sidecar files.

V1 does not include:

- history partitioning by project, directory, server, user, or mode;
- conflation for non-`gen` commands;
- remote or MongoDB storage implementations;
- history pruning, compaction, deletion, or editing;
- arbitrary history search or an interactive history picker;
- shell integration or shell-history parsing; or
- implicit conflation while policy is disabled.

## Command Surface

### Toggle and inspect policy

```console
$ st conflate
Conflating recent successful gen runs.
$ st conflate
Conflation off.
```

Bare `st conflate` toggles `enabled` and retains the configured selector. A new
installation starts disabled with the selector `recent gen exit 0`, so the first
toggle produces the first message above.

Deterministic forms are available for scripts:

```console
$ st conflate on
$ st conflate off
$ st conflate status
```

`on` and `off` are idempotent. `status` does not mutate policy. Each invocation
is still written to global history with family `conflate`, but such entries can
never become generation baselines.

### Configure a recent selector

```console
$ st conflate --inclusive gen
Conflating recent successful gen runs.

$ st conflate --with-exit 1
Conflating recent gen runs with exit code 1.
```

Both forms enable conflation, select recent-query mode, and clear any pinned
history ID. `--inclusive` accepts only `gen` in v1; any other value is an error.
Used without `--with-exit`, it resets the selector to successful exits (`[0]`).
It is retained in the CLI and policy model so eligibility can expand later
without changing the command shape.

The default exit-code selector is exactly `[0]`. Each repeated `--with-exit N`
replaces that default and builds the exact eligible set. Therefore
`--with-exit 1` means exit code `1`, not `0` or `1`. To select both, use:

```console
$ st conflate --with-exit 0 --with-exit 1
```

Exit codes must be integers in the process exit-code range `0..255`. Duplicate
values are removed. `history:<id>` is mutually exclusive with `--inclusive` and
`--with-exit`. Supplying policy selectors with `off` is an error.

### Pin a baseline

```console
$ st conflate history:12345
Conflating only the selected history reference:
st gen --prompt 'a horse talking with a bartender at night' --cfg 4.5 --size 1024x1024
```

This enables conflation and pins history ID `12345`. The selected entry may have
any exit code, including failure. It must have `family == "gen"` and a recorded
effective generation parameter object; otherwise configuration fails without
changing the existing policy.

Pinned mode never advances automatically. Derived successes and failures are
written to history, but every later conflated invocation continues to derive
from ID `12345` until another `st conflate ...` command changes the selector.

## Eligible Invocation Forms

When conflation is enabled, both forms below represent a `gen` patch:

```console
$ st gen --cfg 4.3
$ st --cfg 4.3
```

The root-level shorthand is accepted only when conflation is enabled. Only
flags belonging to `gen`, plus existing root persistent flags, are valid in the
shorthand. Positional text remains valid and replaces the prompt exactly as it
does for `st gen`.

When conflation is disabled, `st --cfg 4.3` is an ordinary Cobra parse error.
An explicit `st gen ...` remains a normal generation command and creates a
history entry regardless of policy state.

If recent mode has no eligible baseline, an explicit `st gen ...` is resolved
normally from config, baked PNG parameters, and its supplied flags. Root-level
shorthand fails before generation with an error explaining that no eligible
`gen` baseline exists and recommending a full `st gen ...` invocation or a
pinned history reference.

## Baseline Semantics

The policy has one of two selectors.

### Recent query

Before every conflated generation, resolve the history entry with the greatest
history ID that:

- has `family == "gen"`;
- has an exit code in the policy's exact `exit_codes` set; and
- has a complete effective generation parameter object.

The query runs again for each invocation. A newly appended run becomes the next
baseline only if its exit code matches the selector. Non-matching runs remain in
history but are skipped by the next query.

For `--with-exit 1`, this produces:

```text
A  exits 1  -> next patch uses A
A1 exits 1  -> next patch uses A1
A2 exits 0  -> next patch still uses A1
A3 exits 1  -> next patch uses A3
A4 exits 0  -> next patch still uses A3
```

### Pinned history

Every conflated invocation uses the exact pinned entry, independent of that
entry's exit code and independent of all later history:

```text
pin C
C1 exits 1  -> next patch still uses C
C2 exits 0  -> next patch still uses C
C3 exits 1  -> next patch still uses C
```

## Parameter Resolution

History stores the fully resolved backend generation parameter object. This
object, not a rendered command string, is the authoritative conflation source.

Without conflation, current precedence remains:

```text
config defaults < baked PNG params < explicit CLI generation flags
```

With a baseline, precedence is:

```text
config defaults < baked PNG params < baseline effective_params
                < explicit current CLI generation flags
```

Because the baseline is already resolved, inherited fields are frozen to the
recorded values and do not silently change when config defaults later change.
Current explicit flags always win, including explicit zero values.

Only fields in the backend `stclient.GenParams` request are inherited. The
following are execution controls and are never inherited:

- `--server`, `--config`, `--timeout`, and `--output-dir`;
- `--outfile`, `--json`, `--stream`, and `--quiet`; and
- local recipe or upload source paths such as `--recreate`, `--init-image`,
  `--control-image`, and `--controlnet-file`.

Resolved backend references produced from those local inputs, such as
`init_image_ref` and normalized `controlnets`, are part of `effective_params`
and can be inherited. This avoids depending on a local file still existing.

The stored effective seed is the concrete seed returned by a completed
generation when available. If a failed run never received a concrete seed, its
resolved request value, including `"random"`, remains in the effective object.

## History Model

The history store appends one JSON object per invocation. `argv` arrays are the
lossless command representation. Display strings are derived, shell-escaped
diagnostics and are never parsed back into commands.

```json
{
  "schema_version": 1,
  "id": 20001,
  "started_at": "2026-07-12T19:24:01.123Z",
  "finished_at": "2026-07-12T19:24:04.456Z",
  "family": "gen",
  "raw": {
    "argv": ["st", "--prompt", "two horses drinking", "--cfg", "4.5"],
    "display": "st --prompt 'two horses drinking' --cfg 4.5"
  },
  "effective": {
    "argv": ["st", "gen", "--prompt", "two horses drinking", "--cfg", "4.5", "--size", "1024x1024"],
    "display": "st gen --prompt 'two horses drinking' --cfg 4.5 --size 1024x1024",
    "params": {
      "prompt": "two horses drinking",
      "guidance_scale": 4.5,
      "genres": "1024x1024",
      "seed": 421337
    }
  },
  "exit_code": 0,
  "derived_from_history_id": 12345,
  "conflate_policy": {
    "selector": "history",
    "history_id": 12345
  },
  "error": null
}
```

Rules:

- `id`, timestamps, `family`, `raw`, and `exit_code` are required.
- `family` is the parsed top-level command name, `gen` for root shorthand, or
  `unknown` when parsing cannot identify a command.
- `effective` is required for a `gen` invocation once parameter resolution
  succeeds. It is absent for other families and for pre-resolution failures.
- `derived_from_history_id` is present only when a baseline was applied.
- `conflate_policy` snapshots the selector used for that invocation and is
  absent when no baseline was applied.
- `error` is a stable summary string for non-zero exits and `null` otherwise.
- Parse and validation failures are recorded. Help/version invocations and the
  `conflate` command are also recorded.
- History entries are immutable after append.

The canonical effective `argv` normalizes a positional prompt to `--prompt` and
renders known request fields in stable flag order. It is diagnostic and may omit
backend fields that have no CLI spelling; `effective.params` remains complete.

## Policy Model

`conflate-policy.json` is a versioned object:

```json
{
  "schema_version": 1,
  "enabled": true,
  "selector": {
    "kind": "recent",
    "family": "gen",
    "exit_codes": [0]
  },
  "updated_at": "2026-07-12T19:20:00Z"
}
```

Pinned form:

```json
{
  "schema_version": 1,
  "enabled": true,
  "selector": {
    "kind": "history",
    "history_id": 12345
  },
  "updated_at": "2026-07-12T19:21:00Z"
}
```

The file is validated before use. Unknown schema versions, selector kinds,
families, or invalid exit codes fail with a state-path-specific error rather
than silently resetting policy.

## Storage Architecture

Core conflation logic depends on interfaces rather than filesystem operations:

```go
type HistoryStore interface {
    ReserveID(context.Context) (int64, error)
    Append(context.Context, HistoryEntry) error
    Get(context.Context, int64) (HistoryEntry, error)
    Latest(context.Context, HistoryFilter) (HistoryEntry, error)
}

type PolicyStore interface {
    Load(context.Context) (ConflatePolicy, error)
    Save(context.Context, ConflatePolicy) error
}
```

The filesystem implementation is the only v1 backend. A future MongoDB backend
can implement these contracts without changing baseline selection or merging.
Selection criteria remain domain types; they are not filesystem query syntax.

### Filesystem layout

Use `$XDG_STATE_HOME/st/` when `XDG_STATE_HOME` is set, otherwise
`$HOME/.local/state/st/`:

```text
st/
  history.jsonl
  conflate-policy.json
  next-id
  state.lock
```

- `history.jsonl` is the single append-only global log.
- `next-id` contains the next integer to reserve and starts at `1`.
- `conflate-policy.json` stores the current policy.
- `state.lock` coordinates ID reservation, append, and policy replacement
  across processes.

State directories and files are user-private (`0700` directory, `0600` files).
V1 performs no generic argument redaction; users should treat the state
directory as sensitive because prompts and command arguments are persisted.

ID reservation and append are separate because the ID must be shown before the
generation begins. A crash after reservation may leave a gap. IDs are strictly
increasing but are not guaranteed contiguous. Recent selection uses greatest
eligible ID, not JSONL line order or completion timestamp.

`next-id` and policy updates use write-fsync-rename while holding `state.lock`.
History append writes one complete JSON line and fsyncs before releasing the
lock. Readers tolerate one incomplete final line from an interrupted append;
malformed interior lines are corruption errors.

## Execution Flow

For every invocation:

1. Resolve the XDG state directory and acquire enough state access to reserve a
   history ID. If state cannot be initialized, fail before remote side effects.
2. Capture raw argv and start time.
3. Parse the command. When conflation is enabled, recognize root-level `gen`
   shorthand before normal Cobra dispatch.
4. For a generation patch, load policy and resolve either the latest eligible
   entry or the pinned entry.
5. Resolve config and baked PNG inputs, overlay baseline `effective.params`,
   then overlay current explicit generation fields.
6. Resolve local uploads into backend references. Record the final request as
   `effective.params` before submitting it.
7. When a baseline was applied, print its canonical command and the reserved
   next command ID to stderr before normal generation output:

   ```text
   initial command [id=12345]: st gen --prompt 'a horse talking with a bartender at night' --cfg 4.5 --size 1024x1024
   next command [id=20001]: st gen --prompt 'two horses drinking...' --cfg 4.5 --size 1024x1024
   ```

8. Execute the command and capture its process exit code. On completed
   generation, replace a symbolic/random seed with the concrete result seed.
9. Append exactly one final history entry.

Conflation diagnostics use stderr so the frozen `--json` object and `--stream`
NDJSON stdout contracts do not change. `--quiet` suppresses these diagnostics.

The command lifecycle must return an exit code to one top-level history wrapper;
subcommands must not terminate the process directly. If final append fails after
a remote side effect, report the history failure on stderr and return a non-zero
CLI exit even if generation succeeded. The generated artifact is not deleted.

## Errors

Errors must identify the failed operation and state path where relevant.
Required cases include:

- no eligible recent baseline for root shorthand;
- pinned history ID not found;
- pinned entry is not `gen` or has no effective params;
- corrupt or unsupported policy/history schema;
- invalid `--inclusive` family or `--with-exit` value;
- conflicting toggle/selector arguments;
- state directory, lock, ID reservation, or append failure; and
- a baseline parameter object that cannot be validated as a generation request.

Changing policy is transactional: validate the complete proposed selector,
including pinned-entry eligibility, before replacing the policy file. A failed
`st conflate ...` leaves the previous policy intact and is itself recorded as a
failed history entry.

## Testing

### Unit tests

- parameter merge precedence, including explicit zeros and prompt replacement;
- separation of inheritable generation params from execution controls;
- recent selection by greatest ID, family, exact exit-code set, and effective
  params presence;
- pinned selection across later successes and failures;
- policy validation and transactional updates;
- JSONL round-trip, incomplete trailing line handling, interior corruption, and
  monotonic ID reservation; and
- canonical argv/display rendering and shell escaping.

### CLI integration tests

- history is written with conflation off for successful and failed commands;
- bare toggle, explicit `on|off|status`, `--inclusive gen`, repeated
  `--with-exit`, and `history:<id>` behavior;
- explicit `st gen` and root-level shorthand produce the same effective patch;
- recent exit-1 history advances only after exit-1 runs;
- a pinned failed baseline remains pinned through derived outcomes;
- missing/corrupt/unwritable state fails as specified;
- concurrent processes reserve unique increasing IDs and append valid JSONL;
- `--quiet` suppresses conflation diagnostics; and
- `--json` and `--stream` stdout remain byte-shape compatible with their frozen
  contracts.

## Acceptance Criteria

V1 is complete when:

- every `st` invocation receives an immutable local integer history entry;
- every resolved `gen` entry stores raw argv, effective argv/display, and the
  fully resolved backend parameter object;
- conflation is persistently toggleable and fixed to `gen` eligibility;
- recent selectors are re-evaluated per run using exact configured exit codes;
- pinned selectors remain fixed regardless of derived results;
- current explicit generation parameters override inherited values without
  inheriting execution/output controls;
- state lives under the XDG state directory behind storage interfaces; and
- existing generation stdout contracts and non-conflated behavior remain
  compatible.
