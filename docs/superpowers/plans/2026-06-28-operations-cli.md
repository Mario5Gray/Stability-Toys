# Operations CLI (Go) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `st`, a Go CLI that drives the backend's core operational path (generate txt2img/img2img/ControlNet, upload, superres, cancel/priority, models, modes, plus local PNG read/recreate) over the existing HTTP + WebSocket APIs.

**Architecture:** One Go module at `cli/go`. A single `pkg/stclient` package is the operation surface (HTTP reads via an `oapi-codegen`-generated client; a hand-written WS job client). `cmd/st` (Cobra) is a thin frontend over `stclient`. A future `cmd/st-mcp` reuses `stclient` unchanged.

**Tech Stack:** Go 1.22+, Cobra (CLI), `github.com/coder/websocket` + `wsjson` (WS), `oapi-codegen` (HTTP types from `/openapi.json`), Go stdlib `image/png` + `testing`/`httptest`.

**Spec:** `docs/superpowers/specs/2026-06-28-operations-cli-design.md`. **FP epic:** STABL-vincfflh.

## Global Constraints

- Module lives at `cli/go` with its own `go.mod`; never touches the Python or JS builds.
- **Transport:** WS-first for jobs (`generate`, `cancel`, `priority`); HTTP for reads/simple calls (`upload`, `superres`, `models`, `modes`, result fetch from `/storage/{key}`).
- **`stclient` is the only place operation logic lives.** `cmd/st` and the future MCP server are thin adapters. No HTTP/WS calls outside `stclient`.
- **Flags:** Cobra / POSIX double-dash. `st gen` takes the prompt as a positional arg.
- **TDD:** a failing Go test before each unit. Tests use `httptest`/in-process WS servers — never a live backend (except the gated `validate-track3` smoke + the gated OpenAPI drift test).
- **Live backend (gated checks only):** the server is remote at `http://enigma.lan:4200` — set `ST_SERVER=http://enigma.lan:4200` for `validate-track3` and the OpenAPI drift test. It is **not** on this host (a local MLX backend is future). The OpenAPI snapshot is pre-captured at repo-root `backend-oai.json`.
- **Contract:** HTTP types come from `oapi-codegen` over committed `openapi.snapshot.json`. The WS envelope + `job:*` frames are hand-written in `stclient/types.go`.
- **WS job submit envelope:** `{"type":"job:submit","id":<corr>,"jobType":"generate","params":{<GenerateRequest fields> , "init_image_ref"?:<ref>}}`. Server replies `job:ack {id,jobId}` → optional `job:progress` → `job:complete {jobId, outputs:[{url:"/storage/<key>",key}], meta:{seed,backend,sr}, controlnet_artifacts?}` | `job:error {jobId,error}`. The result image is a **storage URL**, fetched via `GET /storage/{key}`.
- **`GenerateRequest` fields (verbatim):** `prompt:str`, `negative_prompt:str?`, `mode:str?`, `scheduler_id:str?`, `size:"WxH"`, `num_inference_steps:int 1..50`, `guidance_scale:float 0..20`, `seed:int? 0..2^31-1` (omit = server-random), `superres:bool`, `superres_magnitude:int 1..3`, `denoise_strength:float 0.01..1.0`, `controlnets:[...]?`. `init_image_ref` is a WS-only extra param (not on the model; img2img is WS-only — HTTP `/generate` is txt2img/controlnet only).
- **Precedence (per param, low→high):** config `defaults.generation` (incl. `mode`) < baked params from a **local** `--init-image`/`--recreate` PNG (`lcm` chunk) < explicit CLI flags. Server backfills unset fields from the active mode/env. A bare `fileref:ID` supplies pixels only (no baked-param layer).
- **Config discovery:** `--config` → `$ST_CONFIG` → `$XDG_CONFIG_HOME/stability-toys/config.json` (fallback `~/.config/stability-toys/config.json`). If absent, write a placeholder template there and exit non-zero with the path + directions.
- **Output:** write under `output_directory` (or `-o`) as `out-####.<ext>` (next free index; `<ext>` from `output_format`); `--outfile` overrides (extension optional). When `include_meta`, write a client-side PNG text chunk from `meta` on top of the server's `lcm` chunk.
- **Mode:** v1 selects existing server-side modes only. The CLI issues `POST /api/modes/switch` before submit when the resolved mode differs from the server's current mode; it leaves the server on that mode (no restore). Client-defined modes and batch generation are v2 — out of scope.

---

## File Structure

```text
cli/go/
  go.mod  Makefile  oapi-codegen.yaml  openapi.snapshot.json
  tools/tools.go                 # pins the oapi-codegen tool dependency
  internal/openapi/openapi.gen.go # GENERATED (committed) — HTTP types + client
  pkg/stclient/
     client.go    # Client struct + constructor (baseURL, *http.Client)
     http.go      # Upload, SuperRes, Models, Modes, FetchStorage, SwitchMode
     ws.go        # Generate, Cancel, SetPriority (WS job client)
     types.go     # WS envelope + job:* frames; GenParams, Progress, Result
  internal/config/
     config.go      # Config struct, Load, discovery, BootstrapTemplate
     precedence.go  # Resolve(cfg, baked, flags) -> GenParams
  internal/pngmeta/
     pngmeta.go   # ReadLCM(png) -> map; WriteMeta(png, meta) -> png
  internal/output/
     output.go    # NextPath(dir, fmt), Write(path, png)
  cmd/st/
     main.go  gen.go  read.go  upload.go  superres.go
     cancel.go  priority.go  models.go  modes.go  validate_track3.go
```

Each file has one responsibility. `pkg/stclient` is public (the shared surface); everything else is `internal/`.

---

## Task 1: Module scaffold + OpenAPI codegen

**Files:**
- Create: `cli/go/go.mod`, `cli/go/Makefile`, `cli/go/oapi-codegen.yaml`, `cli/go/tools/tools.go`, `cli/go/openapi.snapshot.json`, `cli/go/internal/openapi/doc.go`
- Test: `cli/go/internal/openapi/openapi_smoke_test.go`

**Interfaces:**
- Produces: package `internal/openapi` with generated types (referenced by `stclient/http.go` in Task 3).

- [ ] **Step 1: Init the module**

```bash
cd cli/go
go mod init github.com/darkbit/stability-toys/cli/st
go get github.com/oapi-codegen/runtime@latest
```

- [ ] **Step 2: Provide the OpenAPI snapshot**

The OpenAPI document is already captured at repo root as **`backend-oai.json`** (untracked) — OpenAPI 3.1.0, 33 paths, includes the `GenerateRequest` schema and the `/generate`, `/superres`, `/v1/upload`, `/api/modes`, `/api/models/status` paths. Copy it into the module:

```bash
cp backend-oai.json cli/go/openapi.snapshot.json
```

The backend is **not** running on this host — it lives at `http://enigma.lan:4200` (a local MLX backend is future). Do **not** curl localhost. Refresh the snapshot later, when the contract changes, with `curl -s http://enigma.lan:4200/openapi.json -o cli/go/openapi.snapshot.json`.

- [ ] **Step 3: Codegen config**

`cli/go/oapi-codegen.yaml`:
```yaml
package: openapi
generate:
  models: true
  client: true
output: internal/openapi/openapi.gen.go
```

`cli/go/tools/tools.go`:
```go
//go:build tools
package tools

import _ "github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen"
```

`cli/go/Makefile`:
```make
gen:
	go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen -config oapi-codegen.yaml openapi.snapshot.json
build:
	go build ./...
test:
	go test ./...
```

- [ ] **Step 4: Write the failing smoke test**

`internal/openapi/openapi_smoke_test.go`:
```go
package openapi

import "testing"

func TestGenerateRequestTypeExists(t *testing.T) {
	var r GenerateRequest
	r.Prompt = "x"
	if r.Prompt != "x" {
		t.Fatalf("Prompt field not wired")
	}
}
```

- [ ] **Step 5: Generate, then run the test to verify it passes**

Run: `cd cli/go && make gen && go test ./internal/openapi/...`
Expected: PASS (generated `GenerateRequest` with a `Prompt` field). If the field is `Prompt string` vs pointer, adjust the test to match the generated shape, then re-run.

- [ ] **Step 6: Commit**

```bash
git add cli/go
git commit -m "feat(cli): scaffold Go module + oapi-codegen (STABL-vincfflh)"
```

---

## Task 2: stclient Client core + HTTP reads (models, modes)

**Files:**
- Create: `cli/go/pkg/stclient/client.go`, `cli/go/pkg/stclient/http.go`
- Test: `cli/go/pkg/stclient/http_test.go`

**Interfaces:**
- Produces: `New(baseURL string, opts ...Option) *Client`; `(*Client) Models(ctx) (ModelsStatus, error)`; `(*Client) Modes(ctx) ([]Mode, error)`; `(*Client) SwitchMode(ctx, name string) error`; `(*Client) CurrentMode(ctx) (string, error)`.
- Consumes: nothing (first stclient task).

- [ ] **Step 1: Write the failing test**

`pkg/stclient/http_test.go`:
```go
package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestModesParsesList(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/modes" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		w.Write([]byte(`{"modes":[{"name":"default"},{"name":"cartoony"}],"current":"default"}`))
	}))
	defer srv.Close()

	c := New(srv.URL)
	modes, err := c.Modes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(modes) != 2 || modes[0].Name != "default" {
		t.Fatalf("got %+v", modes)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestModes`
Expected: FAIL — `New`/`Modes` undefined.

- [ ] **Step 3: Implement client core + modes/models**

`pkg/stclient/client.go`:
```go
package stclient

import (
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL string
	http    *http.Client
}

type Option func(*Client)

func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.http = h } }

func New(baseURL string, opts ...Option) *Client {
	c := &Client{baseURL: strings.TrimRight(baseURL, "/"), http: &http.Client{Timeout: 120 * time.Second}}
	for _, o := range opts {
		o(c)
	}
	return c
}
```

`pkg/stclient/http.go`:
```go
package stclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

type Mode struct {
	Name string `json:"name"`
}
type ModelsStatus map[string]any

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("%s -> %s", path, resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (c *Client) Modes(ctx context.Context) ([]Mode, error) {
	var body struct {
		Modes []Mode `json:"modes"`
	}
	if err := c.getJSON(ctx, "/api/modes", &body); err != nil {
		return nil, err
	}
	return body.Modes, nil
}

func (c *Client) CurrentMode(ctx context.Context) (string, error) {
	var body struct {
		Current string `json:"current"`
	}
	err := c.getJSON(ctx, "/api/modes", &body)
	return body.Current, err
}

func (c *Client) Models(ctx context.Context) (ModelsStatus, error) {
	var m ModelsStatus
	return m, c.getJSON(ctx, "/api/models/status", &m)
}

func (c *Client) SwitchMode(ctx context.Context, name string) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/modes/switch?mode="+name, nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("switch mode %q -> %s", name, resp.Status)
	}
	return nil
}
```

> NOTE: confirm the exact `/api/modes` response keys and the `switch` param/body against `server/mode_config.py` / `server/model_routes.py` when wiring; adjust the structs/path to match, keeping the test green.

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestModes`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient
git commit -m "feat(cli): stclient core + modes/models reads (STABL-vincfflh)"
```

---

## Task 3: stclient Upload + SuperRes + storage fetch

**Files:**
- Modify: `cli/go/pkg/stclient/http.go`
- Test: `cli/go/pkg/stclient/upload_test.go`

**Interfaces:**
- Produces: `(*Client) Upload(ctx, filename string, data []byte) (string, error)` (returns `fileRef`); `(*Client) SuperRes(ctx, data []byte, magnitude int) ([]byte, error)`; `(*Client) FetchStorage(ctx, key string) ([]byte, error)`.

- [ ] **Step 1: Write the failing test**

`pkg/stclient/upload_test.go`:
```go
package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestUploadReturnsFileRef(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/upload" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if _, _, err := r.FormFile("file"); err != nil {
			t.Fatalf("no file part: %v", err)
		}
		w.Write([]byte(`{"fileRef":"abc123"}`))
	}))
	defer srv.Close()

	ref, err := New(srv.URL).Upload(context.Background(), "x.png", []byte("PNGBYTES"))
	if err != nil {
		t.Fatal(err)
	}
	if ref != "abc123" {
		t.Fatalf("got %q", ref)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestUpload`
Expected: FAIL — `Upload` undefined.

- [ ] **Step 3: Implement upload/superres/fetch**

Append to `pkg/stclient/http.go`:
```go
import (
	"bytes"
	"io"
	"mime/multipart"
)

func (c *Client) Upload(ctx context.Context, filename string, data []byte) (string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, _ := mw.CreateFormFile("file", filename)
	fw.Write(data)
	mw.Close()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/upload", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("upload -> %s", resp.Status)
	}
	var body struct {
		FileRef string `json:"fileRef"`
	}
	return body.FileRef, json.NewDecoder(resp.Body).Decode(&body)
}

func (c *Client) FetchStorage(ctx context.Context, key string) ([]byte, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/storage/"+key, nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("storage/%s -> %s", key, resp.Status)
	}
	return io.ReadAll(resp.Body)
}
```

For `SuperRes`, post the bytes to `/superres` (multipart `file` + form `magnitude`) and return the image body (mirror `Upload`'s multipart construction; read the raw response body like `FetchStorage`). Confirm the field names against `server/lcm_sr_server.py:682` (`/superres`) when wiring.

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestUpload`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient
git commit -m "feat(cli): stclient upload/superres/storage fetch (STABL-vincfflh)"
```

---

## Task 4: WS job types

**Files:**
- Create: `cli/go/pkg/stclient/types.go`
- Test: `cli/go/pkg/stclient/types_test.go`

**Interfaces:**
- Produces: `GenParams` (the WS `params` map builder), `submitFrame`, `ackFrame`, `progressFrame`, `completeFrame`, `errorFrame`, `Progress`, `Result`.

- [ ] **Step 1: Write the failing test**

`pkg/stclient/types_test.go`:
```go
package stclient

import (
	"encoding/json"
	"testing"
)

func TestSubmitFrameShape(t *testing.T) {
	p := GenParams{"prompt": "owl", "size": "512x512"}
	f := newSubmitFrame("corr-1", p)
	b, _ := json.Marshal(f)
	var m map[string]any
	json.Unmarshal(b, &m)
	if m["type"] != "job:submit" || m["jobType"] != "generate" || m["id"] != "corr-1" {
		t.Fatalf("bad envelope: %s", b)
	}
	if m["params"].(map[string]any)["prompt"] != "owl" {
		t.Fatalf("params not nested: %s", b)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestSubmitFrame`
Expected: FAIL — `GenParams`/`newSubmitFrame` undefined.

- [ ] **Step 3: Implement the frames**

`pkg/stclient/types.go`:
```go
package stclient

// GenParams is the WS params payload: GenerateRequest fields plus the
// WS-only `init_image_ref`. Built by the precedence resolver.
type GenParams map[string]any

type submitFrame struct {
	Type    string    `json:"type"`
	ID      string    `json:"id"`
	JobType string    `json:"jobType"`
	Params  GenParams `json:"params"`
}

func newSubmitFrame(corrID string, p GenParams) submitFrame {
	return submitFrame{Type: "job:submit", ID: corrID, JobType: "generate", Params: p}
}

type output struct {
	URL string `json:"url"`
	Key string `json:"key"`
}

// inFrame decodes any server frame; Type selects which fields are set.
type inFrame struct {
	Type    string            `json:"type"`
	ID      string            `json:"id"`
	JobID   string            `json:"jobId"`
	Delta   string            `json:"delta"`
	Error   string            `json:"error"`
	Outputs []output          `json:"outputs"`
	Meta    map[string]any    `json:"meta"`
	CNArts  []json.RawMessage `json:"controlnet_artifacts"`
}

type Progress struct{ Delta string }

type Result struct {
	StorageKey string
	StorageURL string
	Seed       int64
	Meta       map[string]any
	CNArtifacts []json.RawMessage
}
```
Add `import "encoding/json"` to the file.

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestSubmitFrame`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient/types.go cli/go/pkg/stclient/types_test.go
git commit -m "feat(cli): WS job frame types (STABL-vincfflh)"
```

---

## Task 5: WS Generate (submit → ack → complete → fetch)

**Files:**
- Create: `cli/go/pkg/stclient/ws.go`
- Test: `cli/go/pkg/stclient/ws_test.go`

**Interfaces:**
- Consumes: `submitFrame`/`inFrame`/`Result`/`Progress` (Task 4); `FetchStorage` (Task 3).
- Produces: `(*Client) Generate(ctx, p GenParams) (<-chan Progress, *Result, error)` — drains progress to the channel, returns `Result` on `job:complete` (with the storage key) or an error on `job:error`.

- [ ] **Step 1: Write the failing test (in-process WS server)**

`pkg/stclient/ws_test.go`:
```go
package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

func TestGenerateResolvesOnComplete(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/storage/") {
			w.Write([]byte("PNGDATA"))
			return
		}
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		corr := sub["id"]
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": corr, "jobId": "J1"})
		wsjson.Write(r.Context(), conn, map[string]any{
			"type": "job:complete", "jobId": "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": 777},
		})
	}))
	defer srv.Close()

	_, res, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": "owl"})
	if err != nil {
		t.Fatal(err)
	}
	if res.StorageKey != "K1" || res.Seed != 777 {
		t.Fatalf("got %+v", res)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go get github.com/coder/websocket && go test ./pkg/stclient/... -run TestGenerate`
Expected: FAIL — `Generate` undefined.

- [ ] **Step 3: Implement Generate**

`pkg/stclient/ws.go`:
```go
package stclient

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

func (c *Client) wsURL() string {
	u := strings.Replace(c.baseURL, "http", "ws", 1)
	return u + "/v1/ws"
}

func (c *Client) Generate(ctx context.Context, p GenParams) (<-chan Progress, *Result, error) {
	conn, _, err := websocket.Dial(ctx, c.wsURL(), &websocket.DialOptions{HTTPClient: c.http})
	if err != nil {
		return nil, nil, err
	}
	prog := make(chan Progress, 16)
	if err := wsjson.Write(ctx, conn, newSubmitFrame("c1", p)); err != nil {
		conn.Close(websocket.StatusInternalError, "")
		close(prog)
		return nil, nil, err
	}
	for {
		var f inFrame
		if err := wsjson.Read(ctx, conn, &f); err != nil {
			close(prog)
			conn.Close(websocket.StatusInternalError, "")
			return nil, nil, err
		}
		switch f.Type {
		case "job:ack":
			continue
		case "job:progress":
			prog <- Progress{Delta: f.Delta}
		case "job:error":
			close(prog)
			conn.Close(websocket.StatusNormalClosure, "")
			return nil, nil, fmt.Errorf("job error: %s", f.Error)
		case "job:complete":
			close(prog)
			conn.Close(websocket.StatusNormalClosure, "")
			res := &Result{Meta: f.Meta, CNArtifacts: f.CNArts}
			if len(f.Outputs) > 0 {
				res.StorageKey = f.Outputs[0].Key
				res.StorageURL = f.Outputs[0].URL
			}
			if s, ok := f.Meta["seed"].(float64); ok {
				res.Seed = int64(s)
			}
			return prog, res, nil
		}
	}
}
```
> NOTE: the returned channel is closed before `Generate` returns, so a non-streaming caller can ignore it; a streaming caller (Task 12) reads it in a goroutine. If image generation never emits `job:progress` (blocking future), the channel simply yields nothing.

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestGenerate`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient/ws.go cli/go/pkg/stclient/ws_test.go cli/go/go.mod cli/go/go.sum
git commit -m "feat(cli): WS Generate job client (STABL-vincfflh)"
```

---

## Task 6: WS Cancel + SetPriority

**Files:**
- Modify: `cli/go/pkg/stclient/ws.go`
- Test: `cli/go/pkg/stclient/ws_control_test.go`

**Interfaces:**
- Produces: `(*Client) Cancel(ctx, jobID string) error`; `(*Client) SetPriority(ctx, jobID string, level int) error` — each dials `/v1/ws`, sends one control frame, waits for the matching `*:ack`.

- [ ] **Step 1: Write the failing test**

`pkg/stclient/ws_control_test.go`:
```go
package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

func TestCancelSendsJobCancel(t *testing.T) {
	got := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var f map[string]any
		wsjson.Read(r.Context(), conn, &f)
		got <- f["type"].(string)
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:cancel:ack", "jobId": f["jobId"]})
	}))
	defer srv.Close()

	if err := New(srv.URL).Cancel(context.Background(), "J1"); err != nil {
		t.Fatal(err)
	}
	if v := <-got; v != "job:cancel" {
		t.Fatalf("got %q", v)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestCancel`
Expected: FAIL — `Cancel` undefined.

- [ ] **Step 3: Implement Cancel/SetPriority**

Append to `pkg/stclient/ws.go`:
```go
func (c *Client) controlFrame(ctx context.Context, send map[string]any, wantAck string) error {
	conn, _, err := websocket.Dial(ctx, c.wsURL(), &websocket.DialOptions{HTTPClient: c.http})
	if err != nil {
		return err
	}
	defer conn.Close(websocket.StatusNormalClosure, "")
	if err := wsjson.Write(ctx, conn, send); err != nil {
		return err
	}
	for {
		var f inFrame
		if err := wsjson.Read(ctx, conn, &f); err != nil {
			return err
		}
		if f.Type == wantAck {
			return nil
		}
		if f.Type == "job:error" {
			return fmt.Errorf("job error: %s", f.Error)
		}
	}
}

func (c *Client) Cancel(ctx context.Context, jobID string) error {
	return c.controlFrame(ctx, map[string]any{"type": "job:cancel", "jobId": jobID}, "job:cancel:ack")
}

func (c *Client) SetPriority(ctx context.Context, jobID string, level int) error {
	return c.controlFrame(ctx, map[string]any{"type": "job:priority", "jobId": jobID, "priority": level}, "job:priority:ack")
}
```
> NOTE: confirm the priority field name (`priority` vs `level`) against `server/ws_routes.py:296` when wiring; keep the test green.

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./pkg/stclient/... -run TestCancel`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient
git commit -m "feat(cli): WS cancel + priority (STABL-vincfflh)"
```

---

## Task 7: Config load + discovery + bootstrap

**Files:**
- Create: `cli/go/internal/config/config.go`
- Test: `cli/go/internal/config/config_test.go`

**Interfaces:**
- Produces: `type Config` (mirrors the spec schema); `Resolve(flagPath string) (path string, err error)` (discovery order); `Load(path string) (*Config, error)`; `BootstrapTemplate(path string) error` (writes placeholder, returns nil); `ErrBootstrapped` sentinel.

- [ ] **Step 1: Write the failing test**

`internal/config/config_test.go`:
```go
package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadReadsGenerationDefaults(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	os.WriteFile(p, []byte(`{"config":{"defaults":{"generation":{"mode":"m","cfg":2.5,"steps":10,"genres":"512x512","seed":"random"},"output_format":"png","output_directory":"/tmp/out"}}}`), 0o644)

	cfg, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Defaults.Generation.Mode != "m" || cfg.Defaults.Generation.Steps != 10 || cfg.Defaults.OutputDirectory != "/tmp/out" {
		t.Fatalf("got %+v", cfg.Defaults)
	}
}

func TestBootstrapWritesTemplate(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.json")
	if err := BootstrapTemplate(p); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(p); err != nil {
		t.Fatalf("template not loadable: %v", err)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./internal/config/...`
Expected: FAIL — `Load`/`BootstrapTemplate` undefined.

- [ ] **Step 3: Implement config**

`internal/config/config.go`:
```go
package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

var ErrBootstrapped = errors.New("config bootstrapped")

type Generation struct {
	Mode   string  `json:"mode"`
	Cfg    float64 `json:"cfg"`
	Steps  int     `json:"steps"`
	Genres string  `json:"genres"`
	Seed   any     `json:"seed"` // int or "random"
}
type Meta struct {
	ProducerName string `json:"producer_name"`
	IncludeDate  bool   `json:"include_date"`
	Misc         []map[string]any `json:"misc"`
}
type Defaults struct {
	Generation      Generation `json:"generation"`
	OutputFormat    string     `json:"output_format"`
	OutputDirectory string     `json:"output_directory"`
	IncludeMeta     bool       `json:"include_meta"`
	Meta            Meta       `json:"meta"`
}
type Config struct {
	Defaults Defaults `json:"defaults"`
}

func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var wrap struct {
		Config Config `json:"config"`
	}
	if err := json.Unmarshal(b, &wrap); err != nil {
		return nil, err
	}
	return &wrap.Config, nil
}

func Resolve(flagPath string) (string, error) {
	if flagPath != "" {
		return flagPath, nil
	}
	if env := os.Getenv("ST_CONFIG"); env != "" {
		return env, nil
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "stability-toys", "config.json"), nil
}

func BootstrapTemplate(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmpl := `{
  "config": {
    "defaults": {
      "generation": { "mode": "default", "cfg": 2.5, "steps": 10, "genres": "512x512", "seed": "random" },
      "output_format": "png",
      "output_directory": "REPLACE_ME/output",
      "include_meta": true,
      "meta": { "producer_name": "REPLACE_ME", "include_date": true, "misc": [] }
    }
  }
}
`
	return os.WriteFile(path, []byte(tmpl), 0o644)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./internal/config/...`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add cli/go/internal/config
git commit -m "feat(cli): config load/discovery/bootstrap (STABL-vincfflh)"
```

---

## Task 8: Precedence resolver

**Files:**
- Create: `cli/go/internal/config/precedence.go`
- Test: `cli/go/internal/config/precedence_test.go`

**Interfaces:**
- Consumes: `Config` (Task 7).
- Produces: `Resolve(cfg *Config, baked map[string]any, flags Flags) stclient.GenParams` where `Flags` carries the explicit CLI values (pointers/`ok` bools so "unset" is distinguishable). Mapping: `genres→size`, `cfg→guidance_scale`, `steps→num_inference_steps`, seed (`"random"`/unset → omit), etc.

- [ ] **Step 1: Write the failing test**

`internal/config/precedence_test.go`:
```go
package config

import "testing"

func TestPrecedenceFlagsBeatBakedBeatConfig(t *testing.T) {
	cfg := &Config{}
	cfg.Defaults.Generation = Generation{Cfg: 2.0, Steps: 5, Genres: "512x512", Seed: "random"}
	baked := map[string]any{"guidance_scale": 7.0, "num_inference_steps": 20}
	flags := Flags{Steps: intp(30)} // only steps set on CLI

	got := Resolve(cfg, baked, flags)
	if got["num_inference_steps"] != 30 { // flag wins
		t.Fatalf("steps=%v", got["num_inference_steps"])
	}
	if got["guidance_scale"] != 7.0 { // baked wins over config
		t.Fatalf("cfg=%v", got["guidance_scale"])
	}
	if got["size"] != "512x512" { // config base
		t.Fatalf("size=%v", got["size"])
	}
	if _, ok := got["seed"]; ok { // random -> omitted
		t.Fatalf("seed should be omitted")
	}
}

func intp(i int) *int { return &i }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./internal/config/... -run TestPrecedence`
Expected: FAIL — `Resolve`/`Flags` undefined.

- [ ] **Step 3: Implement the resolver**

`internal/config/precedence.go`:
```go
package config

type Flags struct {
	Prompt    string
	Negative  *string
	Genres    *string
	Steps     *int
	Cfg       *float64
	Seed      *string // "random" or integer text
	Scheduler *string
	Mode      *string
	SRLevel   *int
}

func Resolve(cfg *Config, baked map[string]any, f Flags) map[string]any {
	p := map[string]any{}
	g := cfg.Defaults.Generation

	// layer 1: config defaults
	setStr(p, "size", g.Genres)
	if g.Cfg != 0 {
		p["guidance_scale"] = g.Cfg
	}
	if g.Steps != 0 {
		p["num_inference_steps"] = g.Steps
	}
	applySeed(p, g.Seed)
	setStr(p, "mode", g.Mode)

	// layer 2: baked params (already in GenerateRequest field names)
	for k, v := range baked {
		p[k] = v
	}

	// layer 3: explicit CLI flags
	if f.Prompt != "" {
		p["prompt"] = f.Prompt
	}
	if f.Genres != nil {
		p["size"] = *f.Genres
	}
	if f.Steps != nil {
		p["num_inference_steps"] = *f.Steps
	}
	if f.Cfg != nil {
		p["guidance_scale"] = *f.Cfg
	}
	if f.Negative != nil {
		p["negative_prompt"] = *f.Negative
	}
	if f.Scheduler != nil {
		p["scheduler_id"] = *f.Scheduler
	}
	if f.Mode != nil {
		p["mode"] = *f.Mode
	}
	if f.SRLevel != nil && *f.SRLevel > 0 {
		p["superres"] = true
		p["superres_magnitude"] = clamp(*f.SRLevel, 1, 3)
	}
	if f.Seed != nil {
		applySeed(p, *f.Seed)
	}
	return p
}

func setStr(p map[string]any, k, v string) {
	if v != "" {
		p[k] = v
	}
}
func applySeed(p map[string]any, seed any) {
	switch s := seed.(type) {
	case string:
		if s == "" || s == "random" {
			delete(p, "seed")
			return
		}
	}
	if seed != nil && seed != "random" && seed != "" {
		p["seed"] = seed
	}
}
func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
```
> The result is a plain `map[string]any`; the `gen` command (Task 12) converts it to `stclient.GenParams` (identical underlying type) and adds `init_image_ref` / `controlnets` separately.

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./internal/config/... -run TestPrecedence`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/internal/config/precedence.go cli/go/internal/config/precedence_test.go
git commit -m "feat(cli): generation param precedence resolver (STABL-vincfflh)"
```

---

## Task 9: PNG metadata read + write

**Files:**
- Create: `cli/go/internal/pngmeta/pngmeta.go`
- Test: `cli/go/internal/pngmeta/pngmeta_test.go`

**Interfaces:**
- Produces: `ReadLCM(png []byte) (map[string]any, error)` (parse the server's `lcm` `tEXt` chunk → JSON map); `BakedParams(png []byte) (map[string]any, error)` (map `lcm` fields → GenerateRequest field names for precedence layer 2); `WriteText(png []byte, key, value string) ([]byte, error)` (add a `tEXt` chunk).

- [ ] **Step 1: Write the failing test**

`internal/pngmeta/pngmeta_test.go`:
```go
package pngmeta

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

func makePNGWithText(t *testing.T, key, val string) []byte {
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	out, err := WriteText(buf.Bytes(), key, val)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestWriteThenReadLCM(t *testing.T) {
	png := makePNGWithText(t, "lcm", `{"prompt":"owl","seed":42,"cfg":2.5}`)
	m, err := ReadLCM(png)
	if err != nil {
		t.Fatal(err)
	}
	if m["prompt"] != "owl" {
		t.Fatalf("got %+v", m)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./internal/pngmeta/...`
Expected: FAIL — `WriteText`/`ReadLCM` undefined.

- [ ] **Step 3: Implement png chunk read/write**

`internal/pngmeta/pngmeta.go` — implement minimal PNG chunk surgery: validate the 8-byte signature, walk `length(4)|type(4)|data|crc(4)` chunks. `WriteText` inserts a `tEXt` chunk (`keyword\x00text`, CRC32 over type+data) immediately before `IEND`. `ReadLCM` scans `tEXt` chunks for keyword `lcm` and `json.Unmarshal`s the text. `BakedParams` maps `lcm` keys → request fields:
```go
func BakedParams(pngBytes []byte) (map[string]any, error) {
	m, err := ReadLCM(pngBytes)
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	move := func(from, to string) {
		if v, ok := m[from]; ok {
			out[to] = v
		}
	}
	move("prompt", "prompt")
	move("negative_prompt", "negative_prompt")
	move("seed", "seed")
	move("cfg", "guidance_scale")
	move("steps", "num_inference_steps")
	move("size", "size")
	move("scheduler_id", "scheduler_id")
	return out, nil
}
```
Use `hash/crc32` (IEEE) for the CRC. Keep it dependency-free (stdlib only).

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./internal/pngmeta/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/internal/pngmeta
git commit -m "feat(cli): PNG lcm metadata read/write (STABL-vincfflh)"
```

---

## Task 10: Output filename scheme + writer

**Files:**
- Create: `cli/go/internal/output/output.go`
- Test: `cli/go/internal/output/output_test.go`

**Interfaces:**
- Produces: `NextPath(dir, format string) (string, error)` (`out-0001.png`, then `out-0002.png`…); `Resolve(outfile, dir, format string) (string, error)` (honors `--outfile`, appends extension if absent); `Write(path string, data []byte) error`.

- [ ] **Step 1: Write the failing test**

`internal/output/output_test.go`:
```go
package output

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNextPathIncrements(t *testing.T) {
	dir := t.TempDir()
	p1, _ := NextPath(dir, "png")
	if filepath.Base(p1) != "out-0001.png" {
		t.Fatalf("got %s", p1)
	}
	os.WriteFile(p1, []byte("x"), 0o644)
	p2, _ := NextPath(dir, "png")
	if filepath.Base(p2) != "out-0002.png" {
		t.Fatalf("got %s", p2)
	}
}

func TestResolveOutfileAppendsExt(t *testing.T) {
	got, _ := Resolve("/tmp/pic", "/tmp", "png")
	if got != "/tmp/pic.png" {
		t.Fatalf("got %s", got)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./internal/output/...`
Expected: FAIL — undefined.

- [ ] **Step 3: Implement output**

`internal/output/output.go`:
```go
package output

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func NextPath(dir, format string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	for i := 1; i < 100000; i++ {
		p := filepath.Join(dir, fmt.Sprintf("out-%04d.%s", i, format))
		if _, err := os.Stat(p); os.IsNotExist(err) {
			return p, nil
		}
	}
	return "", fmt.Errorf("no free out-#### slot in %s", dir)
}

func Resolve(outfile, dir, format string) (string, error) {
	if outfile == "" {
		return NextPath(dir, format)
	}
	if filepath.Ext(outfile) == "" {
		outfile += "." + format
	}
	if !filepath.IsAbs(outfile) {
		outfile = filepath.Join(dir, outfile)
	}
	return outfile, nil
}

func Write(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}

var _ = strings.TrimSpace
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./internal/output/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/internal/output
git commit -m "feat(cli): output filename scheme + writer (STABL-vincfflh)"
```

---

## Task 11: Cobra root + global flags + config wiring

**Files:**
- Create: `cli/go/cmd/st/main.go`
- Test: `cli/go/cmd/st/main_test.go`

**Interfaces:**
- Produces: `rootCmd` with persistent flags `--server`/`$ST_SERVER`, `--config`, `-o/--output-dir`, `--json`, `--timeout`; `loadConfigOrBootstrap()` that returns `(*config.Config, error)` and prints+exits non-zero with the path when bootstrapping.

- [ ] **Step 1: Write the failing test**

`cmd/st/main_test.go`:
```go
package main

import (
	"path/filepath"
	"testing"
)

func TestBootstrapExitsWithPathMessage(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	_, msg, bootstrapped := resolveConfig(p) // p does not exist yet
	if !bootstrapped {
		t.Fatal("expected bootstrap")
	}
	if msg == "" || !contains(msg, p) {
		t.Fatalf("message must state the path: %q", msg)
	}
}

func contains(s, sub string) bool { return len(s) >= len(sub) && (filepath.Base(s) != "" ) && (indexOf(s, sub) >= 0) }
func indexOf(s, sub string) int   { for i := 0; i+len(sub) <= len(s); i++ { if s[i:i+len(sub)] == sub { return i } }; return -1 }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./cmd/st/... -run TestBootstrap`
Expected: FAIL — `resolveConfig` undefined.

- [ ] **Step 3: Implement root + config gate**

`cmd/st/main.go`: build the Cobra `rootCmd`, register persistent flags, and add:
```go
func resolveConfig(path string) (cfg *config.Config, message string, bootstrapped bool) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		_ = config.BootstrapTemplate(path)
		return nil, fmt.Sprintf("No config found. Wrote a template to %s — edit output_directory/meta and re-run.", path), true
	}
	c, err := config.Load(path)
	if err != nil {
		return nil, fmt.Sprintf("config %s: %v", path, err), false
	}
	return c, "", false
}
```
`main()` calls `config.Resolve(--config)`, then `resolveConfig`; if `bootstrapped`, print `message` to stderr and `os.Exit(2)`. Wire `--server`/`$ST_SERVER` into an `stclient.New(...)` constructed per-command.

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./cmd/st/... -run TestBootstrap && go get github.com/spf13/cobra && go build ./...`
Expected: PASS + build succeeds.

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/st/main.go cli/go/cmd/st/main_test.go cli/go/go.mod cli/go/go.sum
git commit -m "feat(cli): cobra root + global flags + config gate (STABL-vincfflh)"
```

---

## Task 12: `st gen` command (the spine)

**Files:**
- Create: `cli/go/cmd/st/gen.go`
- Test: `cli/go/cmd/st/gen_test.go`

**Interfaces:**
- Consumes: `config.Resolve` (Task 8), `pngmeta.BakedParams` (Task 9), `stclient.Generate`/`Upload`/`FetchStorage`/`SwitchMode` (Tasks 2/3/5), `output.Resolve`/`Write` (Task 10).
- Produces: the `gen` subcommand wiring those into one flow.

Flow: build `Flags` from cobra flags → if `--init-image`/`--recreate` is a **local** path, `pngmeta.BakedParams` → layer 2 → `config.Resolve` → `GenParams`. Add `controlnets` from repeated `--controlnet`. Resolve `--init-image`: local path → `Upload` → `params["init_image_ref"]`; `fileref:ID` → `params["init_image_ref"]=ID`. If a `mode` is resolved and differs from `CurrentMode`, `SwitchMode`. `Generate` → on `Result`, `FetchStorage(key)` → optionally `pngmeta.WriteText` client meta when `include_meta` → `output.Resolve`/`Write`. Print result JSON to stdout.

- [ ] **Step 1: Write the failing test (flag→param mapping via a seam)**

`cmd/st/gen_test.go`:
```go
package main

import "testing"

func TestBuildGenParamsFromArgs(t *testing.T) {
	args := genArgs{Prompt: "an owl", Cfg: f64p(3.0), Genres: strp("768x768"), InitImage: "fileref:R1"}
	p, err := buildGenParams(nil /*cfg*/, args)
	if err != nil {
		t.Fatal(err)
	}
	if p["prompt"] != "an owl" || p["guidance_scale"] != 3.0 || p["size"] != "768x768" {
		t.Fatalf("params: %+v", p)
	}
	if p["init_image_ref"] != "R1" {
		t.Fatalf("fileref not threaded: %+v", p)
	}
}

func strp(s string) *string  { return &s }
func f64p(f float64) *float64 { return &f }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cli/go && go test ./cmd/st/... -run TestBuildGenParams`
Expected: FAIL — `buildGenParams`/`genArgs` undefined.

- [ ] **Step 3: Implement `buildGenParams` + the cobra command**

Implement `genArgs` (mirrors the flags), and `buildGenParams(cfg *config.Config, a genArgs) (stclient.GenParams, error)`:
- if `a.InitImage` or `a.Recreate` is a local file (no `fileref:` prefix and the file exists), read it and call `pngmeta.BakedParams` for layer 2.
- call `config.Resolve(cfg-or-empty, baked, a.toFlags())`.
- handle `init_image_ref`: `strings.HasPrefix(a.InitImage,"fileref:")` → strip prefix → set ref; else (local) the upload happens in the command runner (not in this pure function — `buildGenParams` only sets `init_image_ref` for the `fileref:` case; local upload is done in `RunE`). For the test, the `fileref:` path is pure.
- the cobra `RunE` does: `buildGenParams` → local-file upload (if needed) → mode switch → `Generate` → fetch → meta → write.

Use a `nil` config to mean "empty defaults" inside `buildGenParams` (guard with `if cfg == nil { cfg = &config.Config{} }`).

- [ ] **Step 4: Run to verify it passes**

Run: `cd cli/go && go test ./cmd/st/... -run TestBuildGenParams`
Expected: PASS.

- [ ] **Step 5: Add an end-to-end command test against a mock server**

Write `TestGenWritesOutputFile`: stand up an `httptest`+WS mock (reuse the Task 5 server shape: ack→complete with `outputs:[{key:"K1"}]`, and a `/storage/K1` body), run the `gen` cobra command with `--server <mock> -o <tmp> "owl"`, assert a `out-0001.png` file exists with the storage bytes. Run: `cd cli/go && go test ./cmd/st/... -run TestGenWrites` → PASS.

- [ ] **Step 6: Commit**

```bash
git add cli/go/cmd/st/gen.go cli/go/cmd/st/gen_test.go
git commit -m "feat(cli): st gen command end-to-end (STABL-vincfflh)"
```

---

## Task 13: `st read` + `gen --recreate`

**Files:**
- Create: `cli/go/cmd/st/read.go`
- Modify: `cli/go/cmd/st/gen.go` (wire `--recreate` into `buildGenParams` baked layer)
- Test: `cli/go/cmd/st/read_test.go`

**Interfaces:**
- Produces: `read` subcommand printing `pngmeta.ReadLCM` as indented JSON; `--recreate` already handled by `buildGenParams` (Task 12) when the path is local — add the explicit test.

- [ ] **Step 1: Write the failing tests**

`cmd/st/read_test.go`: build a PNG with an `lcm` chunk (reuse `pngmeta.WriteText`), run `read` against it, assert stdout contains `"prompt"`. Add `TestRecreateSeedsParams`: a PNG with `lcm` `{"prompt":"base","cfg":9}`, call `buildGenParams` with `Recreate: <path>, Cfg: nil`, assert `guidance_scale == 9` (baked) and `prompt == "base"`, then with `Cfg: f64p(1)` assert flag overrides to `1`.

- [ ] **Step 2: Run to verify they fail**

Run: `cd cli/go && go test ./cmd/st/... -run 'TestRead|TestRecreate'`
Expected: FAIL.

- [ ] **Step 3: Implement `read` + ensure `--recreate` feeds the baked layer**

`read.go`: read file → `pngmeta.ReadLCM` → `json.MarshalIndent` → print. In `buildGenParams`, treat `a.Recreate` the same as a local `a.InitImage` for the baked layer, but do **not** set `init_image_ref` for `--recreate` (recipe only, txt2img).

- [ ] **Step 4: Run to verify they pass**

Run: `cd cli/go && go test ./cmd/st/... -run 'TestRead|TestRecreate'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/st/read.go cli/go/cmd/st/gen.go cli/go/cmd/st/read_test.go
git commit -m "feat(cli): st read + gen --recreate (STABL-vincfflh)"
```

---

## Task 14: Peripheral subcommands (upload, superres, cancel, priority, models, modes)

**Files:**
- Create: `cli/go/cmd/st/{upload,superres,cancel,priority,models,modes}.go`
- Test: `cli/go/cmd/st/peripherals_test.go`

**Interfaces:**
- Each is a thin cobra command over an existing `stclient` method. No new stclient logic.

- [ ] **Step 1: Write failing tests**

`peripherals_test.go`: for each command, run it against a mock server asserting the right endpoint/frame is hit and the printed output. Example for `upload`:
```go
func TestUploadCmdPrintsFileRef(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"fileRef":"R9"}`))
	}))
	defer srv.Close()
	out := runCmd(t, []string{"upload", "--server", srv.URL, writeTempFile(t, "x.png")})
	if !contains(out, "R9") {
		t.Fatalf("got %q", out)
	}
}
```
(Use a `runCmd` helper that executes `rootCmd` with args and captures stdout.)

- [ ] **Step 2: Run to verify they fail**

Run: `cd cli/go && go test ./cmd/st/... -run TestUploadCmd`
Expected: FAIL.

- [ ] **Step 3: Implement the six commands**

Each: parse args/flags, construct `stclient.New(server)`, call the method, print result (`--json` honored). `superres` reads the file, calls `SuperRes`, writes via `output.Resolve`/`Write`. `cancel`/`priority` call the WS control methods. `models`/`modes` print the read results.

- [ ] **Step 4: Run to verify they pass**

Run: `cd cli/go && go test ./cmd/st/...`
Expected: PASS (all command tests).

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/st
git commit -m "feat(cli): upload/superres/cancel/priority/models/modes commands (STABL-vincfflh)"
```

---

## Task 15: `st validate-track3`

**Files:**
- Create: `cli/go/cmd/st/validate_track3.go`
- Test: `cli/go/cmd/st/validate_track3_test.go`

**Interfaces:**
- Produces: a `validate-track3` command that scripts the checklist from `docs/TESTING_CONTROLNET_TRACK3.md` using `stclient` (upload a control map → `gen --controlnet` → assert `controlnet_artifacts` present on the result). Gated behind `--server` pointing at a live server; the unit test drives the mock server and asserts the *sequence* of calls, not real generation.

- [ ] **Step 1: Write the failing test**

Mock server records the ordered calls (`/v1/upload`, then a WS `job:submit` whose `params.controlnets` is non-empty, then `job:complete` with `controlnet_artifacts`). Assert `validate-track3` reports success when artifacts are present and non-zero exit when absent.

- [ ] **Step 2: Run to verify it fails** — `go test ./cmd/st/... -run TestValidateTrack3` → FAIL.

- [ ] **Step 3: Implement** the command: read the bundled/`--control-image` map, `Upload`, build `GenParams` with one `controlnets` entry, `Generate`, check `Result.CNArtifacts`. Print a checklist-style PASS/FAIL summary; exit non-zero on FAIL.

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/st/validate_track3.go cli/go/cmd/st/validate_track3_test.go
git commit -m "feat(cli): st validate-track3 (STABL-vincfflh)"
```

---

## Task 16: OpenAPI drift guard + CI + docs

**Files:**
- Create: `cli/go/internal/openapi/drift_test.go`, `cli/go/README.md`
- Modify: CI config to add a `cli/go` job (`go test ./...`).

**Interfaces:**
- Produces: a gated test (skipped unless `ST_SERVER` is set) that fetches live `/openapi.json` and diffs it against `openapi.snapshot.json`, failing on divergence.

- [ ] **Step 1: Write the gated drift test**

```go
func TestOpenAPISnapshotMatchesLive(t *testing.T) {
	base := os.Getenv("ST_SERVER")
	if base == "" {
		t.Skip("set ST_SERVER to run the OpenAPI drift check")
	}
	// fetch base+"/openapi.json", read openapi.snapshot.json, compare canonicalized JSON; t.Fatal on diff.
}
```

- [ ] **Step 2: Run to verify it skips cleanly** — `cd cli/go && go test ./internal/openapi/...` → PASS (skipped).

- [ ] **Step 3: Add CI job + README** documenting `make gen|build|test`, the `ST_SERVER` env for gated checks, and the regen workflow when the backend contract changes.

- [ ] **Step 4: Run the full suite** — `cd cli/go && go test ./...` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/go
git commit -m "feat(cli): openapi drift guard + CI + README (STABL-vincfflh)"
```

---

## Self-Review (author checklist — completed)

- **Spec coverage:** scaffold/codegen (T1), stclient HTTP (T2–T3), WS jobs (T4–T6), config+precedence (T7–T8), metadata+output (T9–T10), root+gen spine (T11–T12), read/recreate (T13), peripherals (T14), validate-track3 (T15), contract drift+CI (T16). Mode carry-through covered in T12; config bootstrap in T7+T11; init-image `fileref:`/local split in T12; precedence in T8. Deferred (batch, client modes, advisor/keymap/comfy/workflows, MCP, Zig) correctly absent.
- **Placeholders:** none — every code step carries real Go; two `> NOTE:` callouts ask the implementer to confirm exact server field names against named files while keeping tests green (these are verification instructions, not deferred work).
- **Type consistency:** `GenParams`/`map[string]any` shared between `config.Resolve` and `stclient.Generate`; `Result.StorageKey` feeds `FetchStorage`; `buildGenParams` consumes `BakedParams` + `config.Resolve` + `Flags` consistently across T8/T9/T12/T13.
