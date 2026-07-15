# Upload Bucket Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent-driven development is **forbidden** in this repo (AGENTS.md). Task/issue tracking is via `fp` subissues, not markdown checklists (FP_AGENTS.md) — the `- [ ]` boxes are inline execution-step markers only.

**FP issue:** STABL-kcjkrpry
**Spec (authority):** `docs/superpowers/specs/2026-07-14-upload-bucket-routing-design.md`

**Goal:** Make `POST /v1/upload` honor the `type` field, routing files to `control_map`/`ref_image`/`upload` buckets with image validation for routed buckets, and surface the resolved bucket to `st upload --json`.

**Architecture:** A shared `image_metadata()` helper (extracted from `prepare_promotion`) validates + describes images. The endpoint reads `type` (Form), maps it via a local constant to a store bucket, validates routed buckets, and returns a typed `UploadResponse`. The OpenAPI snapshot is refreshed. `stclient` gains `UploadFile`/`UploadResult`; `st upload --json` shows the server-resolved bucket; `USAGE.md` is updated.

**Tech Stack:** FastAPI (`server/upload_routes.py`), PIL, existing `server/asset_store.py`; Go `pkg/stclient` + `cmd/st`.

## Global Constraints

- **Type→bucket table** (local constant, NOT derived from the ControlNet registry): `canny|depth|pose → control_map`, `image|ref → ref_image`, missing/unknown → `upload`.
- **Validation:** routed buckets (`control_map`/`ref_image`) require a decodable image (400 otherwise); the `upload` bucket stays lenient (any non-empty bytes; empty → 400 as today).
- **Response** (additive): `{fileRef, bucket, width?, height?}`; `bucket` always present, dims only for routed buckets. Declare a typed `UploadResponse` pydantic model **with `response_model_exclude_none=True` on the route** so `None` dims are omitted (not serialized as `null`) for the fallback `upload` bucket.
- **Durability is a consequence, not a policy change:** `control_map`/`ref_image` already have `ttl_s=None, persist=True`; no bucket policy is edited.
- **OpenAPI snapshot** (`cli/go/openapi.snapshot.json`) MUST be refreshed via a live backend + `make gen`; drift guard is `ST_SERVER`-gated.
- **`stclient.Upload` (ref-only) signature is preserved** so its callers (`upload.go`, `gen.go` control-image, `describe.go`) are untouched; it delegates to the new `UploadFile`.
- Python: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest`. Go: from `cli/go`. Commits reference STABL-kcjkrpry.

---

### Task 1: `image_metadata` helper

**Files:**
- Modify: `server/asset_store.py` (add `image_metadata`, refactor `prepare_promotion` ~line 46)
- Modify: `tests/test_asset_store.py` (add helper tests)

**Interfaces:**
- Produces (Tasks 2 rely on this): `image_metadata(data: bytes) -> dict[str, Any]` returning `{"media_type": str, "width": int, "height": int}`; raises `ValueError("asset is not a decodable image")` on non-image bytes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_asset_store.py`:

```python
import io
import pytest
from PIL import Image
from server.asset_store import image_metadata


def _png_bytes(w=7, h=5):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_metadata_reads_png_dimensions():
    meta = image_metadata(_png_bytes(7, 5))
    assert meta == {"media_type": "image/png", "width": 7, "height": 5}


def test_image_metadata_rejects_non_image():
    with pytest.raises(ValueError, match="not a decodable image"):
        image_metadata(b"definitely not an image")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_asset_store.py -k image_metadata -v`
Expected: FAIL — `ImportError: cannot import name 'image_metadata'`

- [ ] **Step 3: Implement the helper + refactor `prepare_promotion`**

In `server/asset_store.py`, add above `prepare_promotion`:

```python
def image_metadata(data: bytes) -> dict[str, Any]:
    """Validate `data` decodes as an image and return its descriptor.

    Raises ValueError if `data` is not a decodable image.
    """
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:
        raise ValueError("asset is not a decodable image") from exc
    # verify() leaves the image unusable; reopen to read format/size.
    img = Image.open(io.BytesIO(data))
    fmt = img.format or "PNG"
    media_type = Image.MIME.get(fmt, f"image/{fmt.lower()}")
    width, height = img.size
    return {"media_type": media_type, "width": width, "height": height}
```

Then replace the body of `prepare_promotion` to reuse it:

```python
def prepare_promotion(data: bytes, source_metadata: dict[str, Any], source_ref: str) -> dict[str, Any]:
    """Validate `data` decodes as an image, then return source metadata merged forward
    with promotion fields overlaid. Raises ValueError if `data` is not a decodable image."""
    meta = image_metadata(data)
    return {
        **source_metadata,
        "origin": "promoted",
        "source_asset_ref": source_ref,
        **meta,
    }
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_asset_store.py -v`
Expected: all PASS (new helper tests + existing promotion tests, which now route through the helper)

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "refactor(asset-store): extract image_metadata() helper from prepare_promotion (STABL-kcjkrpry) — next: upload routing"
```

---

### Task 2: `/v1/upload` bucket routing + typed response

**Files:**
- Modify: `server/upload_routes.py`
- Modify: `tests/test_upload_routes.py`

**Interfaces:**
- Consumes: Task 1's `image_metadata`; `get_store().write(bucket, data, metadata)`; `AssetEntry.bucket` from `resolve`.
- Produces: `POST /v1/upload` accepting an optional `type` Form field, returning `UploadResponse{fileRef, bucket, width?, height?}`; module constant `_TYPE_TO_BUCKET`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_upload_routes.py` (uses a real PNG for routed buckets; the existing `_clear_store` fixture resets buckets between tests):

```python
import io
from PIL import Image


def _png(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(type_label, data, filename="m.png"):
    files = {"file": (filename, data, "image/png")}
    fields = {"type": type_label} if type_label is not None else None
    return client.post("/v1/upload", files=files, data=fields)


def test_canny_routes_to_control_map_with_dimensions():
    resp = _upload("canny", _png(8, 6))
    assert resp.status_code == 200
    body = resp.json()
    assert body["bucket"] == "control_map"
    assert body["width"] == 8 and body["height"] == 6
    assert get_store().resolve(body["fileRef"]).bucket == "control_map"


@pytest.mark.parametrize("type_label, bucket", [
    ("canny", "control_map"),
    ("depth", "control_map"),
    ("pose", "control_map"),
    ("image", "ref_image"),
    ("ref", "ref_image"),
])
def test_routed_types_map_to_expected_bucket(type_label, bucket):
    resp = _upload(type_label, _png())
    assert resp.status_code == 200
    assert resp.json()["bucket"] == bucket
    assert get_store().resolve(resp.json()["fileRef"]).bucket == bucket


def test_unknown_type_falls_back_to_upload_bucket():
    resp = _upload("wat", _png())
    assert resp.status_code == 200
    body = resp.json()
    assert body["bucket"] == "upload"
    # upload bucket is unvalidated: dims are OMITTED entirely, not null
    # (response_model_exclude_none). Assert absence, not "absent or null".
    assert "width" not in body
    assert "height" not in body
    assert get_store().resolve(body["fileRef"]).bucket == "upload"


def test_no_type_uses_upload_bucket_backcompat():
    resp = _upload(None, b"not-an-image-bytes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bucket"] == "upload"
    assert isinstance(body["fileRef"], str)
    assert "width" not in body and "height" not in body


def test_control_map_rejects_non_image():
    resp = _upload("canny", b"this is not an image")
    assert resp.status_code == 400
    assert "control_map" in resp.json()["detail"]


def test_empty_upload_still_400():
    resp = _upload("canny", b"")
    assert resp.status_code == 400
    assert "Empty" in resp.json()["detail"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_upload_routes.py -k "route or bucket or non_image or backcompat or empty" -v`
Expected: FAIL — response has no `bucket` key / non-image `canny` returns 200 instead of 400

- [ ] **Step 3: Implement the endpoint**

Replace the endpoint in `server/upload_routes.py` (keep the module docstring, imports get `Form`, `Optional`, `BaseModel`, `image_metadata`):

```python
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from server.asset_store import get_store, image_metadata

# type label -> store bucket. Deliberately NOT derived from the ControlNet
# registry: upload must not depend on registry load.
_TYPE_TO_BUCKET = {
    "canny": "control_map",
    "depth": "control_map",
    "pose": "control_map",
    "image": "ref_image",
    "ref": "ref_image",
}
_VALIDATED_BUCKETS = {"control_map", "ref_image"}


class UploadResponse(BaseModel):
    fileRef: str
    bucket: str
    width: Optional[int] = None
    height: Optional[int] = None


@upload_router.post("/v1/upload", response_model=UploadResponse, response_model_exclude_none=True)
async def upload_temp_file(
    file: UploadFile = File(...),
    type: Optional[str] = Form(default=None),
) -> UploadResponse:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    bucket = _TYPE_TO_BUCKET.get((type or "").strip(), "upload")

    metadata = None
    width = height = None
    if bucket in _VALIDATED_BUCKETS:
        try:
            meta = image_metadata(data)
        except ValueError:
            raise HTTPException(400, f"{bucket} upload must be a decodable image")
        metadata = meta
        width, height = meta["width"], meta["height"]

    try:
        ref = get_store().write(bucket, data, metadata)
    except ValueError as exc:  # e.g. exceeds bucket byte budget
        raise HTTPException(400, str(exc))

    logger.info("Upload stored: %s -> bucket=%s (%d bytes)", ref, bucket, len(data))
    return UploadResponse(fileRef=ref, bucket=bucket, width=width, height=height)
```

Keep `resolve_file_ref` and `cleanup_uploads_loop` unchanged.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_upload_routes.py -v`
Expected: all PASS (new routing/validation tests + the pre-existing upload tests, which send no `type` and stay in the lenient `upload` bucket)

- [ ] **Step 5: Commit**

```bash
git add server/upload_routes.py tests/test_upload_routes.py
git commit -m "feat(upload): route /v1/upload by type to control_map/ref_image with validation + typed response (STABL-kcjkrpry) — next: openapi snapshot"
```

---

### Task 3: Refresh the OpenAPI snapshot

**Files:**
- Modify: `cli/go/openapi.snapshot.json` (regenerated, not hand-edited)
- Modify: `cli/go/internal/openapi/openapi.gen.go` (regenerated via `make gen`)

**Interfaces:**
- Consumes: the Task 2 endpoint (its `type` field + `UploadResponse` appear in the served `/openapi.json`).
- Produces: an up-to-date snapshot so the drift guard stays truthful.

> **Live-backend dependency:** the snapshot is captured verbatim from a running backend serving the **updated** endpoint. `/openapi.json` is served in any backend mode (it is the schema, not the model), so a locally started app suffices — CUDA is not required. If you cannot start the app in your environment, do **not** hand-edit the snapshot: refresh it on a machine that can run the app, commit that file, then run `make gen`. Flag this to the human if blocked.

- [ ] **Step 1: Start a backend and capture the snapshot**

With the app running and reachable at `$ST_SERVER` (e.g. `http://127.0.0.1:8000`):

```bash
cd cli/go
curl -s "$ST_SERVER/openapi.json" -o openapi.snapshot.json
```

- [ ] **Step 2: Confirm the new upload shape landed in the snapshot**

Run:
```bash
python -c "import json; d=json.load(open('openapi.snapshot.json')); print(json.dumps(d['paths']['/v1/upload']['post'].get('requestBody',{}), indent=0)[:400]); print('UploadResponse' in json.dumps(d))"
```
Expected: the request body shows a `type` property; `True` printed for `UploadResponse`.

- [ ] **Step 3: Regenerate the client types**

Run: `cd cli/go && make gen`
Expected: `internal/openapi/openapi.gen.go` updates; no error. `downspec` must not have mutated `openapi.snapshot.json` (git diff shows only additive upload changes there).

- [ ] **Step 4: Verify build + drift guard**

Run:
```bash
cd cli/go && go build ./... && ST_SERVER="$ST_SERVER" go test ./internal/openapi/ -run TestOpenAPISnapshotMatchesLive -v
```
Expected: build clean; drift test PASS (live spec == refreshed snapshot).

- [ ] **Step 5: Commit**

```bash
git add cli/go/openapi.snapshot.json cli/go/internal/openapi/openapi.gen.go
git commit -m "chore(openapi): refresh snapshot + regen for /v1/upload type routing (STABL-kcjkrpry) — next: stclient UploadFile"
```

---

### Task 4: `stclient.UploadFile` + `UploadResult`

**Files:**
- Modify: `cli/go/pkg/stclient/http.go` (add `UploadResult`/`UploadFile`, refactor `Upload`)
- Modify: `cli/go/pkg/stclient/upload_test.go`

**Interfaces:**
- Consumes: existing `multipartFile`, `Client`.
- Produces (Task 5 relies on this): `type UploadResult struct { Ref, Bucket string; Width, Height int }`; `func (c *Client) UploadFile(ctx context.Context, filename string, data []byte, typeLabel string) (UploadResult, error)`. `Upload(...) (string, error)` is preserved and delegates, returning `res.Ref`.

- [ ] **Step 1: Write the failing tests**

Append to `cli/go/pkg/stclient/upload_test.go` (mirror the existing upload test's httptest setup):

```go
func TestUploadFileReturnsResolvedBucketAndDims(t *testing.T) {
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		gotType = r.FormValue("type")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"fileRef":"R1","bucket":"control_map","width":8,"height":6}`))
	}))
	defer srv.Close()

	res, err := New(srv.URL).UploadFile(context.Background(), "m.png", []byte("data"), "canny")
	if err != nil {
		t.Fatal(err)
	}
	if gotType != "canny" {
		t.Fatalf("type field = %q", gotType)
	}
	if res.Ref != "R1" || res.Bucket != "control_map" || res.Width != 8 || res.Height != 6 {
		t.Fatalf("bad result: %+v", res)
	}
}

func TestUploadDelegatesAndReturnsRef(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"fileRef":"R2","bucket":"upload"}`))
	}))
	defer srv.Close()

	ref, err := New(srv.URL).Upload(context.Background(), "x.png", []byte("data"), "")
	if err != nil {
		t.Fatal(err)
	}
	if ref != "R2" {
		t.Fatalf("ref = %q", ref)
	}
}
```

Check `upload_test.go`'s existing imports; add `context`, `net/http`, `net/http/httptest` if not already present.

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd cli/go && go test ./pkg/stclient/ -run TestUpload -v`
Expected: compile FAIL — `c.UploadFile undefined` / `undefined: UploadResult`

- [ ] **Step 3: Implement**

In `cli/go/pkg/stclient/http.go`, replace the `Upload` method with:

```go
// UploadResult is the decoded /v1/upload response: the ref plus the
// server-resolved bucket and (for validated buckets) image dimensions.
type UploadResult struct {
	Ref    string
	Bucket string
	Width  int
	Height int
}

// UploadFile posts a file to /v1/upload with the given type label and returns
// the full result. The server maps the type label to a bucket.
func (c *Client) UploadFile(ctx context.Context, filename string, data []byte, typeLabel string) (UploadResult, error) {
	var fields map[string]string
	if typeLabel != "" {
		fields = map[string]string{"type": typeLabel}
	}
	buf, contentType, err := multipartFile(filename, data, fields)
	if err != nil {
		return UploadResult{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/upload", buf)
	if err != nil {
		return UploadResult{}, err
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := c.http.Do(req)
	if err != nil {
		return UploadResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return UploadResult{}, fmt.Errorf("upload -> %s", resp.Status)
	}
	var body struct {
		FileRef string `json:"fileRef"`
		Bucket  string `json:"bucket"`
		Width   int    `json:"width"`
		Height  int    `json:"height"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return UploadResult{}, err
	}
	return UploadResult{Ref: body.FileRef, Bucket: body.Bucket, Width: body.Width, Height: body.Height}, nil
}

// Upload posts a file and returns just the ref. Preserved for callers that do
// not need the resolved bucket; delegates to UploadFile.
func (c *Client) Upload(ctx context.Context, filename string, data []byte, bucket string) (string, error) {
	res, err := c.UploadFile(ctx, filename, data, bucket)
	if err != nil {
		return "", err
	}
	return res.Ref, nil
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd cli/go && go test ./pkg/stclient/ -v`
Expected: all PASS (new tests + existing upload/stclient tests)

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient/http.go cli/go/pkg/stclient/upload_test.go
git commit -m "feat(stclient): UploadFile/UploadResult surfacing resolved bucket; Upload delegates (STABL-kcjkrpry) — next: st upload --json + docs"
```

---

### Task 5: `st upload --json` resolved bucket + USAGE.md

**Files:**
- Modify: `cli/go/cmd/st/upload.go`
- Modify: `cli/go/cmd/st/upload_test.go` (create if absent)
- Modify: `cli/go/USAGE.md`

**Interfaces:**
- Consumes: Task 4's `Client.UploadFile`.

- [ ] **Step 1: Write the failing test**

Append to `cli/go/cmd/st/upload_test.go` (use the shared `runCmdCaptureWithStateRoot` harness from `gen_test.go`):

```go
func TestUploadJSONShowsServerResolvedBucket(t *testing.T) {
	dir := t.TempDir()
	img := filepath.Join(dir, "m.png")
	if err := os.WriteFile(img, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// client sent type=canny; server resolved it to control_map
		_, _ = w.Write([]byte(`{"fileRef":"R1","bucket":"control_map","width":8,"height":6}`))
	}))
	defer srv.Close()

	stdout, _, err := runCmdCaptureWithStateRoot(t, t.TempDir(),
		"upload", "canny:"+img, "--json", "--server", srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(stdout), &out); err != nil {
		t.Fatalf("bad json %q: %v", stdout, err)
	}
	if out["bucket"] != "control_map" || out["fileRef"] != "R1" {
		t.Fatalf("want server-resolved bucket, got %v", out)
	}
	if out["width"].(float64) != 8 {
		t.Fatalf("missing dims: %v", out)
	}
}
```

Add imports as needed (`encoding/json`, `net/http`, `net/http/httptest`, `os`, `path/filepath`).

- [ ] **Step 2: Run test, verify it fails**

Run: `cd cli/go && go test ./cmd/st/ -run TestUploadJSON -v`
Expected: FAIL — `--json` still emits the client label `"bucket":"canny"` (from the old `runUpload`)

- [ ] **Step 3: Implement**

Replace `runUpload` in `cli/go/cmd/st/upload.go`:

```go
func runUpload(cmd *cobra.Command, args []string) error {
	bucket, filePath := parseUploadArg(args[0])
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}
	res, err := newClient().UploadFile(cmd.Context(), filepath.Base(filePath), data, bucket)
	if err != nil {
		return err
	}
	if flagJSON {
		out := map[string]any{"fileRef": res.Ref, "bucket": res.Bucket}
		if res.Width > 0 || res.Height > 0 {
			out["width"] = res.Width
			out["height"] = res.Height
		}
		return emitJSON(cmd, out)
	}
	fmt.Fprintln(cmd.OutOrStdout(), res.Ref)
	return nil
}
```

(`bucket` here is the client-side *label*, passed as the `type` field; the JSON now reports `res.Bucket`, the server-resolved bucket.)

- [ ] **Step 4: Run test, verify it passes**

Run: `cd cli/go && go test ./cmd/st/ -v`
Expected: all PASS

- [ ] **Step 5: Update USAGE.md**

In `cli/go/USAGE.md`, replace the Upload section's `--json` example and the trailing note so they describe routing. The `--json` block becomes:

```markdown
# JSON output includes the server-resolved bucket (+ dims for control maps):
st upload canny:./control-map.png --json
# {
#   "bucket": "control_map",
#   "fileRef": "Rabc123def",
#   "height": 768,
#   "width": 768
# }
```

Replace the trailing paragraph with:

```markdown
The `type:path` prefix is split on the first `:`. The type now **routes** the
upload: `canny`/`depth`/`pose` land in the durable `control_map` bucket,
`image`/`ref` in `ref_image`, and any other or missing type in the ephemeral
`upload` bucket (5-minute TTL). Control-map and ref-image uploads are validated
as decodable images and are not time-expired, so their refs stay usable in a
later `st gen --control-ref` / `stcn`.
```

- [ ] **Step 6: Commit**

```bash
git add cli/go/cmd/st/upload.go cli/go/cmd/st/upload_test.go cli/go/USAGE.md
git commit -m "feat(st): upload --json reports server-resolved bucket; USAGE.md routing docs (STABL-kcjkrpry) — next: verification"
```

---

## Final Verification

- [ ] Python: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_upload_routes.py tests/test_asset_store.py -v` — all green.
- [ ] Go: `cd cli/go && go build ./... && go vet ./... && go test ./...` — all green; `gofmt -l pkg/stclient/ cmd/st/` empty.
- [ ] Drift: `ST_SERVER=<live> go test ./internal/openapi/ -run TestOpenAPISnapshotMatchesLive` — PASS (or explicitly flagged to human if no live backend was available for Task 3).
- [ ] `drift check` — report (do not relink) any stale anchors on `server/upload_routes.py` / `server/asset_store.py`; pre-existing deferral applies.
- [ ] FP comment on STABL-kcjkrpry per stopping-point policy.
