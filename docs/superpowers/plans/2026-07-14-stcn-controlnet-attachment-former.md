# stcn — ControlNet Attachment Former Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent-driven development is **forbidden** in this repo (AGENTS.md). Steps use checkbox (`- [ ]`) syntax for tracking.

**FP issue:** STABL-nzrqaxla
**Spec (authority):** `docs/superpowers/specs/2026-07-14-stcn-controlnet-attachment-former-design.md`

**Goal:** A `stcn` binary that forms one ControlNet attachment from flags and emits a compact, single-shell-token JSON object built on the generated `openapi.ControlNetAttachment`, consumable as `st gen --controlnet $(stcn ...)`.

**Architecture:** New `cli/go/cmd/stcn` binary. A pure `buildAttachment` function maps flags → `openapi.ControlNetAttachment` with full validation (schema ranges + shell-token-safe string fields); `main.go` parses the positional + flags via cobra and emits compact JSON. Reuses only `internal/openapi`; no `pkg/stclient`, config, or network.

**Tech Stack:** Go, cobra/pflag (matching `cmd/st`), `encoding/json`, generated `internal/openapi` types.

## Global Constraints

- **Module:** `github.com/darkbit/stability-toys/cli/st`; the openapi import is `github.com/darkbit/stability-toys/cli/st/internal/openapi` (same module root, so `cmd/stcn` may import the `internal/` package).
- **Schema-forced:** build and marshal `openapi.ControlNetAttachment` (fields `AttachmentId string`, `ControlType string`, `MapAssetRef *string`, `ModelId *string`, `Strength *float32`, `StartPercent *float32`, `EndPercent *float32`). Never a hand-authored map.
- **Compact output:** `json.Marshal` (no indent), single line, trailing newline.
- **Shell-token-safe fields:** every emitted string (`control_type`, `map_asset_ref`, `model_id`, `attachment_id`) must match `^[A-Za-z0-9._:/-]+$`; otherwise reject.
- **Ranges (schema absolute):** `strength ∈ [0,2]`, `start_percent ∈ [0,1]`, `end_percent ∈ [0,1]`, `start ≤ end`. Unset optionals stay `nil` (omitted). `attachment_id` always emitted, defaults to `control_type`.
- **v1 map-only:** no `source_asset_ref`/`preprocess`; no send/upload/config/network; no `st`/`stclient` changes.
- Run Go from `cli/go` (`go test ./...`, `go vet ./...`). Commits reference STABL-nzrqaxla and state the next step.

---

### Task 1: `buildAttachment` — flags → validated attachment

**Files:**
- Create: `cli/go/cmd/stcn/attach.go`
- Create: `cli/go/cmd/stcn/attach_test.go`

**Interfaces:**
- Consumes: `openapi.ControlNetAttachment` (generated).
- Produces (Task 2/3 rely on these):
  - `type attachOpts struct { controlType, mapAssetRef, model, id string; strength, start, end float64; strengthSet, startSet, endSet bool }`
  - `func buildAttachment(o attachOpts) (openapi.ControlNetAttachment, error)`
  - `func parseHead(arg string) (controlType, mapAssetRef string, err error)` — splits `type:ref` on the first colon.

- [ ] **Step 1: Write the failing tests**

`cli/go/cmd/stcn/attach_test.go`:

```go
package main

import (
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/openapi"
)

func TestParseHeadSplitsOnFirstColon(t *testing.T) {
	ct, ref, err := parseHead("canny:fileref:MAP1")
	if err != nil {
		t.Fatal(err)
	}
	if ct != "canny" || ref != "fileref:MAP1" {
		t.Fatalf("got (%q, %q)", ct, ref)
	}
}

func TestParseHeadRejectsMissingColon(t *testing.T) {
	if _, _, err := parseHead("canny"); err == nil {
		t.Fatal("want error for missing colon")
	}
}

func TestBuildAttachmentMinimalDefaultsIdToControlType(t *testing.T) {
	a, err := buildAttachment(attachOpts{controlType: "canny", mapAssetRef: "Rmap1"})
	if err != nil {
		t.Fatal(err)
	}
	if a.AttachmentId != "canny" || a.ControlType != "canny" {
		t.Fatalf("ids: %+v", a)
	}
	if a.MapAssetRef == nil || *a.MapAssetRef != "Rmap1" {
		t.Fatalf("map: %+v", a.MapAssetRef)
	}
	// Unset optionals stay nil (omitted from JSON).
	if a.Strength != nil || a.StartPercent != nil || a.EndPercent != nil || a.ModelId != nil {
		t.Fatalf("optionals should be nil: %+v", a)
	}
}

func TestBuildAttachmentCarriesAllFields(t *testing.T) {
	a, err := buildAttachment(attachOpts{
		controlType: "depth", mapAssetRef: "Rmap2", model: "sdxl-depth", id: "d0",
		strength: 0.8, strengthSet: true,
		start: 0.1, startSet: true, end: 0.9, endSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if a.AttachmentId != "d0" || *a.ModelId != "sdxl-depth" {
		t.Fatalf("ids/model: %+v", a)
	}
	if a.Strength == nil || a.StartPercent == nil || a.EndPercent == nil {
		t.Fatalf("floats nil: %+v", a)
	}
	if *a.Strength != float32(0.8) || *a.StartPercent != float32(0.1) || *a.EndPercent != float32(0.9) {
		t.Fatalf("floats: %+v", a)
	}
}

func TestBuildAttachmentRejectsOutOfRange(t *testing.T) {
	cases := []attachOpts{
		{controlType: "canny", mapAssetRef: "R", strength: 2.5, strengthSet: true},
		{controlType: "canny", mapAssetRef: "R", strength: -0.1, strengthSet: true},
		{controlType: "canny", mapAssetRef: "R", start: 1.5, startSet: true},
		{controlType: "canny", mapAssetRef: "R", end: -0.1, endSet: true},
		{controlType: "canny", mapAssetRef: "R", start: 0.9, startSet: true, end: 0.1, endSet: true},
		{controlType: "", mapAssetRef: "R"},
		{controlType: "canny", mapAssetRef: ""},
	}
	for i, o := range cases {
		if _, err := buildAttachment(o); err == nil {
			t.Fatalf("case %d: want error", i)
		}
	}
}

func TestBuildAttachmentRejectsShellUnsafeFields(t *testing.T) {
	cases := []attachOpts{
		{controlType: "canny", mapAssetRef: "R map"},          // space in ref
		{controlType: "ca nny", mapAssetRef: "R"},             // space in type
		{controlType: "canny", mapAssetRef: "R", model: "m*"}, // glob in model
		{controlType: "canny", mapAssetRef: "R", id: "my id"}, // space in id
		{controlType: "canny", mapAssetRef: `R"x`},            // quote
		{controlType: "canny", mapAssetRef: "R$x"},            // dollar
	}
	for i, o := range cases {
		if _, err := buildAttachment(o); err == nil {
			t.Fatalf("case %d: want shell-safety error", i)
		}
	}
	// Colon, slash, dot, dash are allowed.
	if _, err := buildAttachment(attachOpts{controlType: "canny", mapAssetRef: "fileref:MAP-1/a.png"}); err != nil {
		t.Fatalf("safe ref rejected: %v", err)
	}
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd cli/go && go test ./cmd/stcn/ -run 'TestParseHead|TestBuildAttachment' -v`
Expected: compile FAIL — `undefined: parseHead` / `buildAttachment` / `attachOpts`

- [ ] **Step 3: Implement `attach.go`**

```go
package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/darkbit/stability-toys/cli/st/internal/openapi"
)

type attachOpts struct {
	controlType string
	mapAssetRef string
	model       string
	id          string
	strength    float64
	start       float64
	end         float64
	strengthSet bool
	startSet    bool
	endSet      bool
}

// shellSafe matches the fields stcn will emit. It excludes whitespace and
// shell-active characters so the compact JSON is exactly one argv token under
// an unquoted $(stcn ...), while allowing ref-bearing punctuation (: / . -).
var shellSafe = regexp.MustCompile(`^[A-Za-z0-9._:/-]+$`)

// parseHead splits "control_type:map_asset_ref" on the FIRST colon, so refs
// that themselves contain a colon (e.g. fileref:MAP1) are preserved.
func parseHead(arg string) (string, string, error) {
	ct, ref, ok := strings.Cut(arg, ":")
	if !ok {
		return "", "", fmt.Errorf("positional must be <control_type>:<map_asset_ref>, got %q", arg)
	}
	return ct, ref, nil
}

func requireSafe(field, value string) error {
	if value == "" {
		return fmt.Errorf("%s must not be empty", field)
	}
	if !shellSafe.MatchString(value) {
		return fmt.Errorf("%s %q contains whitespace or a shell metacharacter (allowed: A-Z a-z 0-9 . _ : / -)", field, value)
	}
	return nil
}

func buildAttachment(o attachOpts) (openapi.ControlNetAttachment, error) {
	var a openapi.ControlNetAttachment

	if err := requireSafe("control_type", o.controlType); err != nil {
		return a, err
	}
	if err := requireSafe("map_asset_ref", o.mapAssetRef); err != nil {
		return a, err
	}

	id := o.id
	if id == "" {
		id = o.controlType // default attachment_id to the control type
	}
	if err := requireSafe("id", id); err != nil {
		return a, err
	}

	a.ControlType = o.controlType
	a.AttachmentId = id
	ref := o.mapAssetRef
	a.MapAssetRef = &ref

	if o.model != "" {
		if err := requireSafe("model", o.model); err != nil {
			return a, err
		}
		m := o.model
		a.ModelId = &m
	}

	if o.strengthSet {
		if o.strength < 0.0 || o.strength > 2.0 {
			return a, fmt.Errorf("strength %g out of range [0.0, 2.0]", o.strength)
		}
		s := float32(o.strength)
		a.Strength = &s
	}
	if o.startSet {
		if o.start < 0.0 || o.start > 1.0 {
			return a, fmt.Errorf("start %g out of range [0.0, 1.0]", o.start)
		}
		s := float32(o.start)
		a.StartPercent = &s
	}
	if o.endSet {
		if o.end < 0.0 || o.end > 1.0 {
			return a, fmt.Errorf("end %g out of range [0.0, 1.0]", o.end)
		}
		e := float32(o.end)
		a.EndPercent = &e
	}
	if o.startSet && o.endSet && o.start > o.end {
		return a, fmt.Errorf("start %g must be <= end %g", o.start, o.end)
	}

	return a, nil
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd cli/go && go test ./cmd/stcn/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/stcn/attach.go cli/go/cmd/stcn/attach_test.go
git commit -m "feat(stcn): buildAttachment with schema-range + shell-safe validation (STABL-nzrqaxla) — next: compact marshal"
```

---

### Task 2: Compact marshal + single-token guarantee

**Files:**
- Create: `cli/go/cmd/stcn/emit.go`
- Create: `cli/go/cmd/stcn/emit_test.go`

**Interfaces:**
- Consumes: `openapi.ControlNetAttachment`, Task 1's `buildAttachment`.
- Produces (Task 3 relies on this): `func marshalCompact(a openapi.ControlNetAttachment) ([]byte, error)` — compact JSON, no trailing newline (the caller adds it).

- [ ] **Step 1: Write the failing tests**

`cli/go/cmd/stcn/emit_test.go`:

```go
package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/openapi"
)

func TestMarshalCompactHasNoWhitespace(t *testing.T) {
	a, err := buildAttachment(attachOpts{
		controlType: "canny", mapAssetRef: "fileref:MAP1",
		strength: 0.8, strengthSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := marshalCompact(a)
	if err != nil {
		t.Fatal(err)
	}
	// Single-token pin: no space/tab/newline anywhere in the emitted bytes.
	if bytes.ContainsAny(out, " \t\n\r") {
		t.Fatalf("output has whitespace, would word-split: %q", out)
	}
}

func TestMarshalCompactRoundTripsIntoSchemaType(t *testing.T) {
	a, err := buildAttachment(attachOpts{
		controlType: "depth", mapAssetRef: "Rmap2", model: "sdxl-depth", id: "d0",
		strength: 0.5, strengthSet: true, start: 0.2, startSet: true, end: 0.8, endSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := marshalCompact(a)
	if err != nil {
		t.Fatal(err)
	}
	var back openapi.ControlNetAttachment
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatalf("does not round-trip into schema type: %v", err)
	}
	if back.AttachmentId != "d0" || back.ControlType != "depth" ||
		back.MapAssetRef == nil || *back.MapAssetRef != "Rmap2" ||
		back.ModelId == nil || *back.ModelId != "sdxl-depth" ||
		back.Strength == nil || back.StartPercent == nil || back.EndPercent == nil {
		t.Fatalf("fields lost in round-trip: %+v", back)
	}
	// Unset optionals must not appear as keys (omitempty).
	if strings.Contains(string(out), "source_asset_ref") || strings.Contains(string(out), "preprocess") {
		t.Fatalf("nil optionals leaked into JSON: %s", out)
	}
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd cli/go && go test ./cmd/stcn/ -run TestMarshalCompact -v`
Expected: compile FAIL — `undefined: marshalCompact`

- [ ] **Step 3: Implement `emit.go`**

```go
package main

import (
	"encoding/json"

	"github.com/darkbit/stability-toys/cli/st/internal/openapi"
)

// marshalCompact renders the attachment as compact JSON (no indentation, no
// spaces). Combined with shell-safe field validation, the result is exactly
// one argv token under an unquoted $(stcn ...).
func marshalCompact(a openapi.ControlNetAttachment) ([]byte, error) {
	return json.Marshal(a)
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd cli/go && go test ./cmd/stcn/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/stcn/emit.go cli/go/cmd/stcn/emit_test.go
git commit -m "feat(stcn): compact marshal with single-token + schema round-trip pins (STABL-nzrqaxla) — next: cli wiring"
```

---

### Task 3: `main.go` — CLI wiring, output, exit codes

**Files:**
- Create: `cli/go/cmd/stcn/main.go`
- Create: `cli/go/cmd/stcn/main_test.go`

**Interfaces:**
- Consumes: Task 1's `attachOpts`/`buildAttachment`/`parseHead`, Task 2's `marshalCompact`.
- Produces: the `stcn` binary. `run(args []string, stdout io.Writer) error` is the testable entrypoint (main() wraps it and maps error→exit 1).

- [ ] **Step 1: Write the failing tests**

`cli/go/cmd/stcn/main_test.go`:

```go
package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunEmitsCompactAttachment(t *testing.T) {
	var out bytes.Buffer
	err := run([]string{"canny:Rmap1", "--strength", "0.8"}, &out)
	if err != nil {
		t.Fatal(err)
	}
	line := out.String()
	if !strings.HasSuffix(line, "\n") {
		t.Fatalf("want trailing newline: %q", line)
	}
	trimmed := strings.TrimRight(line, "\n")
	if strings.ContainsAny(trimmed, " \t") {
		t.Fatalf("emitted line is not a single shell token: %q", trimmed)
	}
	var a map[string]any
	if err := json.Unmarshal([]byte(trimmed), &a); err != nil {
		t.Fatal(err)
	}
	if a["control_type"] != "canny" || a["map_asset_ref"] != "Rmap1" ||
		a["attachment_id"] != "canny" || a["strength"].(float64) != 0.8 {
		t.Fatalf("bad attachment: %v", a)
	}
}

func TestRunRequiresPositional(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"--strength", "0.8"}, &out); err == nil {
		t.Fatal("want error when positional missing")
	}
	if out.Len() != 0 {
		t.Fatalf("nothing must be emitted on error, got %q", out.String())
	}
}

func TestRunPropagatesValidationError(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"canny:Rmap1", "--strength", "9"}, &out); err == nil {
		t.Fatal("want validation error for out-of-range strength")
	}
	if out.Len() != 0 {
		t.Fatalf("nothing must be emitted on error, got %q", out.String())
	}
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd cli/go && go test ./cmd/stcn/ -run TestRun -v`
Expected: compile FAIL — `undefined: run`

- [ ] **Step 3: Implement `main.go`**

```go
// Command stcn forms a single ControlNet attachment from flags and prints it
// as compact JSON suitable for `st gen --controlnet $(stcn ...)`. It is a
// pure, offline tool: it never contacts the server, uploads, or reads config.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
)

func newRootCmd(out io.Writer, opts *attachOpts) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "stcn <control_type>:<map_asset_ref>",
		Short: "Form one ControlNet attachment as compact JSON",
		Long: `stcn forms a single ControlNet attachment from flags and prints it as
compact JSON. Compose into a generation by repeating the flag:

  st gen --prompt "..." \
    --controlnet $(stcn canny:Rmap1 --strength 0.8) \
    --controlnet $(stcn depth:Rmap2 --strength 0.4)

Only map_asset_ref (a pre-made control map) is supported in v1. Emitted
string fields must be shell-token-safe (A-Z a-z 0-9 . _ : / -) so the
unquoted $(stcn ...) form is a single argv token.`,
		Args:          cobra.ExactArgs(1),
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			ct, ref, err := parseHead(args[0])
			if err != nil {
				return err
			}
			opts.controlType = ct
			opts.mapAssetRef = ref
			opts.strengthSet = cmd.Flags().Changed("strength")
			opts.startSet = cmd.Flags().Changed("start")
			opts.endSet = cmd.Flags().Changed("end")

			a, err := buildAttachment(*opts)
			if err != nil {
				return err
			}
			b, err := marshalCompact(a)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "%s\n", b)
			return nil
		},
	}
	f := cmd.Flags()
	f.Float64Var(&opts.strength, "strength", 0, "conditioning strength (0.0-2.0; unset = mode default)")
	f.Float64Var(&opts.start, "start", 0, "start_percent (0.0-1.0)")
	f.Float64Var(&opts.end, "end", 0, "end_percent (0.0-1.0)")
	f.StringVar(&opts.model, "model", "", "model_id override (default = mode policy)")
	f.StringVar(&opts.id, "id", "", "attachment_id (default = control_type)")
	return cmd
}

// run is the testable entrypoint: parse args, emit to out, return any error.
func run(args []string, out io.Writer) error {
	var opts attachOpts
	cmd := newRootCmd(out, &opts)
	cmd.SetArgs(args)
	return cmd.Execute()
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd cli/go && go test ./cmd/stcn/ -v && go vet ./cmd/stcn/`
Expected: all PASS, vet clean

- [ ] **Step 5: Build + real-run smoke**

Run:
```bash
cd cli/go && go build -o /tmp/stcn ./cmd/stcn
/tmp/stcn canny:Rmap1 --strength 0.8
echo "exit=$?"
/tmp/stcn canny:Rmap1 --strength 9; echo "exit=$?"
```
Expected: first prints `{"attachment_id":"canny","control_type":"canny","map_asset_ref":"Rmap1","strength":0.8}` (field order may differ) with `exit=0`; second prints `error: strength 9 out of range [0.0, 2.0]` to stderr with `exit=1`.

- [ ] **Step 6: Commit**

```bash
git add cli/go/cmd/stcn/main.go cli/go/cmd/stcn/main_test.go
git commit -m "feat(stcn): cobra CLI wiring, compact stdout, exit codes (STABL-nzrqaxla) — next: distribution"
```

---

### Task 4: Distribution — install target + README

**Files:**
- Modify: `cli/go/Makefile` (`install` target)
- Modify: `cli/go/README.md` (command inventory + a short stcn note)

**Interfaces:**
- Consumes: the `cmd/stcn` binary from Tasks 1–3.
- Produces: `make install` installs `stcn`; README documents it.

- [ ] **Step 1: Update the Makefile install target**

Change `install:` in `cli/go/Makefile` from:

```make
install:
	go install ./cmd/st
```

to:

```make
install:
	go install ./cmd/st ./cmd/stcn
```

- [ ] **Step 2: Verify install builds both**

Run: `cd cli/go && go install ./cmd/st ./cmd/stcn && ls "$(go env GOPATH)/bin/stcn" "$(go env GOPATH)/bin/st"`
Expected: both binaries listed, no error.

- [ ] **Step 3: Document stcn in the README command inventory**

In `cli/go/README.md`, add a row to the Commands table after the `st validate-track3` row:

```markdown
| `stcn <type>:<ref>` | Form one ControlNet attachment as compact JSON for `st gen --controlnet $(stcn ...)` |
```

And add a short subsection after the Commands table (before the "Global flags" line):

```markdown
### stcn — ControlNet attachment former

`stcn` is a separate, offline binary that turns flags into one schema-valid
ControlNet attachment object, built on the generated OpenAPI type (so it
cannot drift from what the server accepts). It never contacts the server.
Compose attachments by repeating the flag:

    st gen --prompt "a bridge" --cfg 7.2 \
      --controlnet $(stcn canny:Rmap1 --strength 0.8) \
      --controlnet $(stcn depth:Rmap2 --strength 0.4)

Flags: `--strength` (0.0–2.0), `--start`/`--end` (0.0–1.0), `--model`, `--id`
(defaults to the control type). Emitted string fields are restricted to
`A-Z a-z 0-9 . _ : / -` so the unquoted `$(stcn ...)` form is a single shell
token. v1 supports `map_asset_ref` (a pre-made control map) only.
```

- [ ] **Step 4: Full workspace verification**

Run: `cd cli/go && go build ./... && go vet ./... && go test ./...`
Expected: all PASS (new `cmd/stcn` tests plus the existing suite; nothing else changed).

- [ ] **Step 5: Commit**

```bash
git add cli/go/Makefile cli/go/README.md
git commit -m "feat(stcn): install target + README command inventory (STABL-nzrqaxla) — next: review"
```

---

## Final Verification

- [ ] `cd cli/go && go test ./... && go vet ./...` — all green.
- [ ] `gofmt -l cmd/stcn/` — empty (formatted).
- [ ] Real-run: `go run ./cmd/stcn canny:Rmap1 --strength 0.8` emits a single-line compact object; `--id "my id"` and `--strength 9` both exit 1 with nothing on stdout.
- [ ] Single-token check: `printf '%s' "$(go run ./cmd/stcn canny:Rmap1 --strength 0.8)" | wc -w` prints `1`.
- [ ] FP comment on STABL-nzrqaxla per stopping-point policy; report ready for review.
