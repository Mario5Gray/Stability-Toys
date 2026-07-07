# `st read` ControlNet Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (Project policy forbids subagent-driven development — execute inline.)

**Goal:** Make `st read <image.png>` detect and print all three PNG tEXt metadata chunks the backend writes (`lcm`, `controlnet`, `controlnet_map`), not just `lcm`.

**Architecture:** `internal/pngmeta` gains a `Parse`/`Chunks` type that walks a PNG once and exposes `FindLCM`/`FindControlNet`/`FindControlNetMap` lookup methods against that single parse. `ReadLCM` (used by `st gen --recreate`) is reimplemented on top of `Parse`+`FindLCM` with its exact current contract preserved. `cmd/st/read.go`'s `runRead` calls all three `Find*` methods and emits one top-level JSON key per chunk found.

**Tech Stack:** Go 1.x, standard library only (`encoding/json`, `image/png` in tests). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-06-st-read-controlnet-metadata-design.md`

## Global Constraints

- `ReadLCM(pngBytes []byte) (map[string]any, error)` keeps its exact current signature and error contract (`fmt.Errorf("no lcm tEXt chunk")` when absent) — `BakedParams`/`st gen --recreate` must be unaffected.
- `runRead` must parse the PNG **once** (via `pngmeta.Parse`) and query all three chunks against that single result — not three independent `parseChunks` calls.
- Every chunk found gets its own top-level output key, named after its own keyword (`lcm`, `controlnet`, `controlnet_map`). No merging/normalizing `controlnet` (list) with `controlnet_map` (dict).
- If none of the three chunks are present, or a present chunk's JSON is malformed, `read` errors (fails loud).
- Existing tests `TestReadPrintsLCM`, `TestRecreateSeedsParams` (`cli/go/cmd/st/read_test.go`) and `TestWriteThenReadLCM`, `TestBakedParamsMapsToRequestFields` (`cli/go/internal/pngmeta/pngmeta_test.go`) must stay green.
- Run Go tests from `cli/go`: `go test ./... -run <Pattern> -v`.

## File Structure

- **Modify `cli/go/internal/pngmeta/pngmeta.go`:** add `Chunks` type, `Parse`, `Chunks.text`, `Chunks.FindLCM`, `Chunks.FindControlNet`, `Chunks.FindControlNetMap`; reimplement `ReadLCM` on top of `Parse`+`FindLCM`.
- **Modify `cli/go/internal/pngmeta/pngmeta_test.go`:** add tests for `Parse`/`FindControlNet`/`FindControlNetMap`/co-occurring lookups.
- **Modify `cli/go/cmd/st/read.go`:** `runRead` parses once, checks all three chunks, wraps output.
- **Modify `cli/go/cmd/st/read_test.go`:** tighten `TestReadPrintsLCM`; add new chunk-detection tests.
- **Modify `cli/go/README.md`:** `st read` one-liner.
- **Modify `cli/go/USAGE.md`:** "Reading PNG metadata" example + prose.

---

### Task 1: `pngmeta.Parse`/`Chunks` + `FindLCM`, `ReadLCM` reimplemented

**Files:**
- Modify: `cli/go/internal/pngmeta/pngmeta.go`
- Test: `cli/go/internal/pngmeta/pngmeta_test.go`

**Interfaces:**
- Produces:
  - `type Chunks struct{ chunks []chunk }`
  - `func Parse(pngBytes []byte) (Chunks, error)`
  - `func (c Chunks) text(keyword string) ([]byte, bool)` (unexported helper)
  - `func (c Chunks) FindLCM() (map[string]any, bool, error)`
  - `func ReadLCM(pngBytes []byte) (map[string]any, error)` — same signature as today, reimplemented internally

- [ ] **Step 1: Write the failing tests**

Append to `cli/go/internal/pngmeta/pngmeta_test.go`:

```go
func TestParseThenFindLCM(t *testing.T) {
	pngBytes := makePNGWithText(t, "lcm", `{"prompt":"owl","seed":42}`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	m, ok, err := chunks.FindLCM()
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected FindLCM ok=true")
	}
	if m["prompt"] != "owl" {
		t.Fatalf("got %+v", m)
	}
}

func TestFindLCMAbsentIsNotError(t *testing.T) {
	pngBytes := makePNGWithText(t, "other", `{"x":1}`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	_, ok, err := chunks.FindLCM()
	if err != nil {
		t.Fatalf("absence must not be an error: %v", err)
	}
	if ok {
		t.Fatal("expected FindLCM ok=false when lcm chunk absent")
	}
}

func TestReadLCMErrorsWhenAbsent(t *testing.T) {
	pngBytes := makePNGWithText(t, "other", `{"x":1}`)
	if _, err := ReadLCM(pngBytes); err == nil {
		t.Fatal("expected error when lcm chunk absent")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd cli/go && go test ./internal/pngmeta/... -run 'TestParseThenFindLCM|TestFindLCMAbsentIsNotError|TestReadLCMErrorsWhenAbsent' -v`
Expected: FAIL — `undefined: Parse` (compile error).

- [ ] **Step 3: Add `Chunks`/`Parse`/`FindLCM`, reimplement `ReadLCM`**

In `cli/go/internal/pngmeta/pngmeta.go`, add after `encodeChunk` and before `WriteText`:

```go
// Chunks is a PNG parsed once so its tEXt chunks can be queried by keyword
// without re-walking the file for each lookup.
type Chunks struct {
	chunks []chunk
}

// Parse walks pngBytes once into a queryable Chunks value.
func Parse(pngBytes []byte) (Chunks, error) {
	cs, err := parseChunks(pngBytes)
	if err != nil {
		return Chunks{}, err
	}
	return Chunks{chunks: cs}, nil
}

// text returns the raw tEXt payload for keyword, or ok=false if absent. Absence
// is not an error; a malformed-JSON chunk's error surfaces from the decoding
// Find* method instead.
func (c Chunks) text(keyword string) ([]byte, bool) {
	for _, ch := range c.chunks {
		if ch.typ != "tEXt" {
			continue
		}
		i := bytes.IndexByte(ch.data, 0x00)
		if i < 0 {
			continue
		}
		if string(ch.data[:i]) != keyword {
			continue
		}
		return ch.data[i+1:], true
	}
	return nil, false
}

// FindLCM returns the lcm chunk's decoded payload, if present.
func (c Chunks) FindLCM() (map[string]any, bool, error) {
	text, ok := c.text("lcm")
	if !ok {
		return nil, false, nil
	}
	var m map[string]any
	if err := json.Unmarshal(text, &m); err != nil {
		return nil, true, fmt.Errorf("lcm chunk not JSON: %w", err)
	}
	return m, true, nil
}
```

Replace the existing `ReadLCM` function body:

```go
// ReadLCM finds the `lcm` tEXt chunk and unmarshals its JSON text into a map.
func ReadLCM(pngBytes []byte) (map[string]any, error) {
	chunks, err := Parse(pngBytes)
	if err != nil {
		return nil, err
	}
	m, ok, err := chunks.FindLCM()
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("no lcm tEXt chunk")
	}
	return m, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd cli/go && go test ./internal/pngmeta/... -v`
Expected: PASS — all existing tests (`TestWriteThenReadLCM`, `TestWriteTextKeepsValidPNG`, `TestBakedParamsMapsToRequestFields`) plus the three new ones.

- [ ] **Step 5: Commit**

```bash
git add cli/go/internal/pngmeta/pngmeta.go cli/go/internal/pngmeta/pngmeta_test.go
git commit -m "feat(pngmeta): add Parse/Chunks single-walk PNG chunk lookup, reimplement ReadLCM on top — next: FindControlNet/FindControlNetMap"
```

---

### Task 2: `FindControlNet` (list) + `FindControlNetMap` (dict)

**Files:**
- Modify: `cli/go/internal/pngmeta/pngmeta.go`
- Test: `cli/go/internal/pngmeta/pngmeta_test.go`

**Interfaces:**
- Consumes: `Chunks`, `Chunks.text` (Task 1).
- Produces:
  - `func (c Chunks) FindControlNet() ([]any, bool, error)`
  - `func (c Chunks) FindControlNetMap() (map[string]any, bool, error)`

- [ ] **Step 1: Write the failing tests**

Append to `cli/go/internal/pngmeta/pngmeta_test.go`:

```go
func TestFindControlNetMapDict(t *testing.T) {
	pngBytes := makePNGWithText(t, "controlnet_map",
		`{"tool":"canny_map","control_type":"canny","source_width":8,"source_height":8}`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	m, ok, err := chunks.FindControlNetMap()
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected FindControlNetMap ok=true")
	}
	if m["tool"] != "canny_map" || m["control_type"] != "canny" {
		t.Fatalf("got %+v", m)
	}
}

func TestFindControlNetList(t *testing.T) {
	pngBytes := makePNGWithText(t, "controlnet",
		`[{"attachment_id":"cn_1","control_type":"canny"}]`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	list, ok, err := chunks.FindControlNet()
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected FindControlNet ok=true")
	}
	if len(list) != 1 {
		t.Fatalf("got %+v", list)
	}
	entry, ok := list[0].(map[string]any)
	if !ok || entry["attachment_id"] != "cn_1" {
		t.Fatalf("got %+v", list[0])
	}
}

func TestFindControlNetAbsentIsNotError(t *testing.T) {
	pngBytes := makePNGWithText(t, "lcm", `{"prompt":"owl"}`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	_, ok, err := chunks.FindControlNet()
	if err != nil {
		t.Fatalf("absence must not be an error: %v", err)
	}
	if ok {
		t.Fatal("expected FindControlNet ok=false when controlnet chunk absent")
	}
}

func TestFindControlNetMapMalformedJSONErrors(t *testing.T) {
	pngBytes := makePNGWithText(t, "controlnet_map", `not json`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	_, ok, err := chunks.FindControlNetMap()
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
	if !ok {
		t.Fatal("malformed-but-present chunk should report ok=true alongside the error")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd cli/go && go test ./internal/pngmeta/... -run 'TestFindControlNet' -v`
Expected: FAIL — `chunks.FindControlNetMap undefined` (compile error).

- [ ] **Step 3: Add the two methods**

In `cli/go/internal/pngmeta/pngmeta.go`, add after `FindLCM`:

```go
// FindControlNetMap returns the controlnet_map chunk's decoded payload (a flat
// dict), if present. Written onto standalone control-map PNGs by scripts/cn_metadata.py.
func (c Chunks) FindControlNetMap() (map[string]any, bool, error) {
	text, ok := c.text("controlnet_map")
	if !ok {
		return nil, false, nil
	}
	var m map[string]any
	if err := json.Unmarshal(text, &m); err != nil {
		return nil, true, fmt.Errorf("controlnet_map chunk not JSON: %w", err)
	}
	return m, true, nil
}

// FindControlNet returns the controlnet chunk's decoded payload (a list of
// per-attachment provenance entries), if present. Written onto generation-output
// PNGs alongside lcm whenever the generation used a ControlNet binding.
func (c Chunks) FindControlNet() ([]any, bool, error) {
	text, ok := c.text("controlnet")
	if !ok {
		return nil, false, nil
	}
	var list []any
	if err := json.Unmarshal(text, &list); err != nil {
		return nil, true, fmt.Errorf("controlnet chunk not JSON: %w", err)
	}
	return list, true, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd cli/go && go test ./internal/pngmeta/... -v`
Expected: PASS — all tests in the package.

- [ ] **Step 5: Commit**

```bash
git add cli/go/internal/pngmeta/pngmeta.go cli/go/internal/pngmeta/pngmeta_test.go
git commit -m "feat(pngmeta): FindControlNet (list) and FindControlNetMap (dict) chunk lookups — next: wire runRead"
```

---

### Task 3: `runRead` detects all three chunks, wraps output

**Files:**
- Modify: `cli/go/cmd/st/read.go`
- Test: `cli/go/cmd/st/read_test.go`

**Interfaces:**
- Consumes: `pngmeta.Parse`, `Chunks.FindLCM`, `Chunks.FindControlNet`, `Chunks.FindControlNetMap` (Tasks 1–2).

- [ ] **Step 1: Write the failing tests**

Replace `TestReadPrintsLCM` and add new tests in `cli/go/cmd/st/read_test.go`. Replace the existing `pngWithLCM` helper block and `TestReadPrintsLCM` with:

```go
func pngWithText(t *testing.T, keyword, jsonText string) string {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	out, err := pngmeta.WriteText(buf.Bytes(), keyword, jsonText)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "in.png")
	if err := os.WriteFile(p, out, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func pngWithLCM(t *testing.T, lcmJSON string) string {
	return pngWithText(t, "lcm", lcmJSON)
}

func TestReadPrintsLCMWrapped(t *testing.T) {
	path := pngWithLCM(t, `{"prompt":"owl","seed":42}`)
	out := runCmd(t, "read", path)
	if !strings.Contains(out, `"lcm"`) {
		t.Fatalf("read output missing lcm wrapper key: %q", out)
	}
	if !strings.Contains(out, `"prompt"`) || !strings.Contains(out, "owl") {
		t.Fatalf("read output missing lcm fields: %q", out)
	}
}

func TestReadPrintsControlNetMap(t *testing.T) {
	path := pngWithText(t, "controlnet_map", `{"tool":"canny_map","control_type":"canny"}`)
	out := runCmd(t, "read", path)
	if !strings.Contains(out, `"controlnet_map"`) || !strings.Contains(out, "canny_map") {
		t.Fatalf("read output missing controlnet_map fields: %q", out)
	}
}

func TestReadPrintsLCMAndControlNetTogether(t *testing.T) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	out1, err := pngmeta.WriteText(buf.Bytes(), "lcm", `{"prompt":"owl"}`)
	if err != nil {
		t.Fatal(err)
	}
	out2, err := pngmeta.WriteText(out1, "controlnet", `[{"attachment_id":"cn_1","control_type":"canny"}]`)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "both.png")
	if err := os.WriteFile(p, out2, 0o644); err != nil {
		t.Fatal(err)
	}

	got := runCmd(t, "read", p)
	if !strings.Contains(got, `"lcm"`) || !strings.Contains(got, `"controlnet"`) {
		t.Fatalf("read output missing one of lcm/controlnet: %q", got)
	}
	if !strings.Contains(got, "cn_1") {
		t.Fatalf("read output missing controlnet entry: %q", got)
	}
}

func TestReadErrorsWhenNoKnownChunks(t *testing.T) {
	path := pngWithText(t, "unrelated", `{"x":1}`)
	if _, err := runCmdMayFail(t, "read", path); err == nil {
		t.Fatal("expected error when no known metadata chunk present")
	}
}
```

`runCmdMayFail(t, args...) (string, error)` already exists in `cli/go/cmd/st/gen_test.go`
(same package `main`, visible from `read_test.go` without import) — use it as-is, no new
helper needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd cli/go && go test ./cmd/st/... -run 'TestReadPrintsLCMWrapped|TestReadPrintsControlNetMap|TestReadPrintsLCMAndControlNetTogether|TestReadErrorsWhenNoKnownChunks' -v`
Expected: FAIL — output still flat (no `"lcm"` wrapper key), and no error raised for the no-known-chunks case (current code errors only via `pngmeta.ReadLCM`'s specific message, but here we're asserting a general error which today's code coincidentally also produces — the wrapper-key assertions are what fail).

- [ ] **Step 3: Rewrite `runRead`**

Replace the entire body of `runRead` in `cli/go/cmd/st/read.go`:

```go
func runRead(cmd *cobra.Command, args []string) error {
	data, err := os.ReadFile(args[0])
	if err != nil {
		return err
	}

	chunks, err := pngmeta.Parse(data)
	if err != nil {
		return err
	}

	out := map[string]any{}
	if v, ok, err := chunks.FindLCM(); err != nil {
		return err
	} else if ok {
		out["lcm"] = v
	}
	if v, ok, err := chunks.FindControlNet(); err != nil {
		return err
	} else if ok {
		out["controlnet"] = v
	}
	if v, ok, err := chunks.FindControlNetMap(); err != nil {
		return err
	} else if ok {
		out["controlnet_map"] = v
	}

	if len(out) == 0 {
		return fmt.Errorf("no known metadata chunk (lcm, controlnet, controlnet_map) found in %s", args[0])
	}

	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), string(b))
	return nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd cli/go && go test ./cmd/st/... -v`
Expected: PASS — including `TestRecreateSeedsParams` (uses `buildGenParams`/`pngmeta.ReadLCM` directly, untouched by this task) and all read tests.

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/st/read.go cli/go/cmd/st/read_test.go
git commit -m "feat(cli): st read detects lcm/controlnet/controlnet_map chunks via one pngmeta.Parse walk, wraps output per chunk found"
```

---

### Task 4: Docs — README + USAGE

**Files:**
- Modify: `cli/go/README.md`
- Modify: `cli/go/USAGE.md`

**Interfaces:**
- None (docs only).

- [ ] **Step 1: Update the README one-liner**

In `cli/go/README.md`, replace:

```
| `st read <png>` | Print the `lcm` generation metadata embedded in a PNG |
```

with:

```
| `st read <png>` | Print embedded PNG metadata (`lcm`, `controlnet`, `controlnet_map`) |
```

- [ ] **Step 2: Update the USAGE.md example and prose**

In `cli/go/USAGE.md`, replace the "Reading PNG metadata" section:

```
## Reading PNG metadata

\`\`\`bash
st read images/out-0001.png
# {
#   "prompt": "a ceramic owl on a shelf, studio lighting",
#   "cfg": 2.5,
#   "steps": 8,
#   "seed": 3847291
# }
\`\`\`

Returns the raw `lcm` tEXt chunk as JSON. No server call; works offline.
```

with:

```
## Reading PNG metadata

\`\`\`bash
st read images/out-0001.png
# {
#   "lcm": {
#     "prompt": "a ceramic owl on a shelf, studio lighting",
#     "cfg": 2.5,
#     "steps": 8,
#     "seed": 3847291
#   }
# }

st read images/out-with-controlnet.png
# {
#   "lcm": { "prompt": "...", "seed": 42 },
#   "controlnet": [
#     { "attachment_id": "cn_1", "control_type": "canny", "generation": {...}, "source": {...} }
#   ]
# }

st read control_maps/canny.png
# { "controlnet_map": { "tool": "canny_map", "control_type": "canny", "source_width": 1024, "source_height": 1024 } }
\`\`\`

Detects whichever of the three known PNG tEXt chunks are present — `lcm` (generation
params), `controlnet` (per-attachment ControlNet provenance, present alongside `lcm`
whenever the generation used ControlNet), and `controlnet_map` (provenance on a
standalone control-map file) — and prints one JSON key per chunk found. No server
call; works offline.
```

- [ ] **Step 3: Commit**

```bash
git add cli/go/README.md cli/go/USAGE.md
git commit -m "docs(cli): update st read docs for lcm/controlnet/controlnet_map chunk detection"
```

---

## Self-Review

**Spec coverage:**
- `Parse`/`Chunks` single-walk API, `FindLCM`, `ReadLCM` reimplemented unchanged → Task 1. ✓
- `FindControlNet` (list) / `FindControlNetMap` (dict) → Task 2. ✓
- `runRead` detects all three, wraps output, errors on none/malformed → Task 3. ✓
- README + USAGE.md (example block AND prose line) → Task 4. ✓
- `TestReadPrintsLCM`→ tightened to assert wrapper key (renamed `TestReadPrintsLCMWrapped` for clarity, same coverage) → Task 3. ✓
- `TestRecreateSeedsParams` stays green untouched → Task 3 Step 5 explicitly checks it. ✓
- `TestWriteThenReadLCM`/`TestBakedParamsMapsToRequestFields` stay green untouched → Task 1 Step 4 runs the full package. ✓
- New package-level tests for co-occurring lcm+controlnet lookups via one `Chunks` value → Task 3's `TestReadPrintsLCMAndControlNetTogether` covers this at the CLI level (exercises `Parse` finding both from one call).

**Placeholder scan:** no TBD/TODO; every step carries concrete code and exact commands. Task 3 uses the existing `runCmdMayFail` helper (`cli/go/cmd/st/gen_test.go:110`, same package) rather than inventing a new one.

**Type consistency:** `Chunks`, `Parse(pngBytes []byte) (Chunks, error)`, `FindLCM() (map[string]any, bool, error)`, `FindControlNet() ([]any, bool, error)`, `FindControlNetMap() (map[string]any, bool, error)` are used identically across Tasks 1–3. `runRead` in Task 3 calls exactly the methods Tasks 1–2 define, with matching signatures.
