# Upload Bucket Routing — Design

**FP issue:** STABL-kcjkrpry
**Status:** Authority artifact for the upload bucket-routing track.

## Goal

Make `POST /v1/upload` a **solid upload → asset-ref** operative by honoring the
`type` label the client already sends: route the file to the store bucket that
matches its kind, validate it, and return the resolved bucket. Today the
endpoint ignores `type` and hardcodes the ephemeral `upload` bucket, so a
control map both lands in an opaque bucket and expires after 5 minutes — the
exact fragility that makes `upload → --control-ref/stcn` unreliable.

Success: `st upload canny:./map.png` stores the map in the durable `control_map`
bucket and returns a ref that survives long enough to use in a later
`st gen --control-ref canny:<ref>` or `stcn canny:<ref>`.

## Current state

- `POST /v1/upload` (`server/upload_routes.py`) reads only the `file` part and
  calls `get_store().write("upload", data)` — `type` is dropped. Response is
  `{"fileRef": "<uuid>"}`.
- The `upload` bucket policy has `ttl_s=300` (5-min TTL, purged by
  `cleanup_uploads_loop`); `control_map` has `ttl_s=None, persist=True`;
  `ref_image` has `ttl_s=None, persist=True` (`server/asset_store.py`).
- `st upload [type:]<file>` already sends `type` as a form field via
  `Client.Upload(ctx, filename, data, bucket)` and prints the bare ref (or
  `{"fileRef","bucket"}` with `--json`, where `bucket` is the *client's* label).

So the routing mechanism is entirely a server-side gap: the field arrives and is
ignored.

## Server: `/v1/upload` bucket routing

### Type → bucket mapping

A local constant in `server/upload_routes.py`:

```python
_TYPE_TO_BUCKET = {
    "canny": "control_map",
    "depth": "control_map",
    "pose":  "control_map",
    "image": "ref_image",
    "ref":   "ref_image",
}
# missing or unrecognized type -> "upload" (default, back-compat)
```

- The mapping is **decoupled from the ControlNet registry** on purpose. Upload
  must not depend on registry load — that coupling is what failed server
  startup on enigma (`ControlNet model path does not exist`). A small explicit
  table keeps uploads working regardless of registry/model availability.
- Missing or unrecognized `type` resolves to `upload`, preserving today's
  behavior for every existing caller (including non-image uploads).

### Validation

- When the resolved bucket is `control_map` or `ref_image`, the endpoint
  **decodes the bytes as an image before accepting**: `Image.open(...).verify()`,
  then re-open to read format/size. A non-image (or corrupt) payload returns
  `400` with a clear message naming the bucket. This reuses the exact pattern in
  `server/asset_store.py:prepare_promotion` (verify + `Image.MIME` +
  `img.size`); factor that image-metadata extraction into a shared helper (e.g.
  `image_metadata(data) -> {media_type, width, height}`) so both call sites use
  one implementation.
- The resolved `media_type`, `width`, `height` are stored as the asset's
  metadata (`get_store().write(bucket, data, metadata)`).
- The default `upload` bucket stays **lenient**: any non-empty bytes are
  accepted, unchanged (the existing empty-body `400` remains). No decode is
  attempted there, so current non-image upload uses are unaffected.

### Response

```json
{ "fileRef": "<uuid>", "bucket": "control_map", "width": 768, "height": 768 }
```

- `bucket` is always present (the resolved bucket). `width`/`height` are present
  only for validated (routed) buckets. This is **additive**: existing clients
  reading `fileRef` are unaffected.
- Empty upload still returns `400 "Empty upload"` before any routing.

### Errors

| Condition | Status | Body |
| --- | --- | --- |
| empty file | 400 | `Empty upload` (unchanged) |
| type routes to control_map/ref_image but bytes are not a decodable image | 400 | message naming the bucket and that it requires an image |
| write exceeds the bucket byte budget | 400 | the store's budget `ValueError` surfaced as 400 |

Unknown `type` is **not** an error — it falls back to `upload`.

### Durability (consequence, not a new policy)

Routing `canny/depth/pose` to `control_map` means the ref inherits
`ttl_s=None` (never time-expired) and `persist=True`. It is evicted only under
256 MB LRU budget pressure, and persists to the durable tier when a
`StorageProvider` is configured. This resolves the "ref expired before I used
it" pain as a side effect of routing — **no bucket-policy change is made**; the
existing `control_map` policy already provides this.

## Client: surface the resolved bucket

- Add `type UploadResult struct { Ref, Bucket string; Width, Height int }` and
  `func (c *Client) UploadFile(ctx, filename string, data []byte, typeLabel string) (UploadResult, error)`
  in `pkg/stclient`, decoding the full response.
- Refactor the existing `Client.Upload(...) (string, error)` to delegate to
  `UploadFile` and return `res.Ref`, so its current callers (control-image
  auto-upload, describe auto-upload) are unchanged.
- `st upload --json` uses `UploadFile` and emits
  `{"fileRef","bucket","width"?,"height"?}` reflecting the **server-resolved**
  bucket (not the raw client label), so the operator can confirm a control map
  actually landed in `control_map` (i.e. is durable). Plain `st upload` still
  prints the bare ref as a single token.

## Testing

- **Server endpoint** (`tests/test_upload_routes.py`, extend or create): routing
  table — `canny→control_map`, `depth→control_map`, `pose→control_map`,
  `image→ref_image`, `ref→ref_image`, no-type→`upload`, unknown-type→`upload`;
  validation — a non-image payload with `type=canny` returns 400, a real PNG
  returns 200 with `bucket=control_map` and correct `width`/`height`; response
  shape (`bucket` always present; dims only for routed buckets); back-compat —
  no `type` returns a `fileRef` in the `upload` bucket; empty body still 400.
  Assert the write landed in the resolved bucket via `get_store().buckets()` /
  `bucket_bytes`.
- **Image-metadata helper** unit test: decodable PNG → media_type/width/height;
  non-image bytes → raises.
- **Client** (`pkg/stclient`): `UploadFile` decodes `bucket`/dims from a
  mocked response; `Upload` still returns the ref. `st upload --json` surfaces
  the server-resolved bucket (httptest).

## Non-goals

- No bucket-policy (TTL / budget / persistence) changes — the routing *uses*
  the existing `control_map`/`ref_image` policies.
- No new buckets.
- No public asset-promotion endpoint.
- No rehydration of expired refs from local source paths.
- No change to how `type` is spelled on the client (`st upload canny:./m.png`
  stays; the client already sends `canny`).
- No coupling of the upload route to the ControlNet registry.

## Implementation order (input to the plan)

1. Shared `image_metadata(data)` helper (extract from `prepare_promotion`) with
   tests; refactor `prepare_promotion` to use it.
2. `/v1/upload` reads `type`, maps to a bucket, validates routed buckets,
   writes with metadata, returns `{fileRef,bucket,width?,height?}`; endpoint
   tests.
3. `stclient.UploadFile` + `UploadResult`; `Upload` delegates; `st upload
   --json` surfaces the resolved bucket; client tests.
