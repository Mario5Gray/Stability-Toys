# Gallery Advisor Implementation Plan

> **Execution mode:** run serially in one session. Avoid subagents unless explicitly requested.

**Goal:** Build a gallery-backed advisor that turns gallery image metadata into a reusable digest, persists editable advice per gallery, and applies that advice to the draft prompt with deterministic append/replace controls.

**Architecture:** Keep galleries and advisor state client-side in IndexedDB, add a narrow backend `/api/advisors/digest` route that validates typed evidence and calls an OpenAI-compatible LLM client, and render the advisor under `Negative Prompt Templates` in the existing options panel. The advisor identity is per `gallery_id`; mode config only contributes the runtime `maximum_len` constraint exposed through `/api/modes`.

**Tech Stack:** FastAPI, Pydantic, `httpx`, existing `ModeConfigManager`, React 19, Vitest, IndexedDB via browser APIs and `fake-indexeddb`.

---

## Status Refresh (2026-04-13)

- Active FP issue for the backend prerequisite is `STABL-grarbnxp` ("Add OpenAI-compatible chat completions backend client").
- Backend progress landed:
  - `server/mode_config.py` now supports `maximum_len` and per-mode `chat` config parsing/serialization.
  - `server/model_routes.py` now exposes `maximum_len` and `chat_enabled` in `/api/modes`.
  - `server/ws_routes.py` now accepts `jobType=chat`, enforces mode-scoped chat config (no env fallback), clamps `max_tokens` to mode `maximum_len`, and returns `job:complete` text outputs (plus streaming `job:progress` deltas).
  - `backends/chat_client.py` added as a minimal OpenAI-compatible client (`complete` + `stream`) with SSE guards for empty `choices` events.
  - `server/advisor_service.py` and `server/advisor_routes.py` now provide `POST /api/advisors/digest`, and `server/lcm_sr_server.py` includes `advisor_router`.
- For `STABL-grarbnxp`, backend scope is the active lane. Resume frontend advisor tasks only after backend chat plumbing is accepted.

---

## File Structure

### Backend

- Modify: `server/mode_config.py`
  Adds `maximum_len` to `ModeConfig`, parses it from `modes.yml`, persists it through `save_config()` and `to_dict()`.
- Modify: `server/model_routes.py`
  Exposes `maximum_len` through `/api/modes`.
- Create: `backends/chat_client.py`
  Minimal async OpenAI-compatible chat completions client for advisor analysis.
- Create: `server/advisor_service.py`
  Validates advisor digest requests, renders the analysis prompt, computes `evidence_fingerprint`, calls `ChatCompletionsClient`, returns `digest_text`.
- Create: `server/advisor_routes.py`
  FastAPI route layer for `POST /api/advisors/digest`.
- Modify: `server/lcm_sr_server.py`
  Includes `advisor_router`.
- Modify: `requirements.txt`
  Adds `httpx`.

### Backend Tests

- Modify: `tests/test_mode_config.py`
  Verifies `maximum_len` parsing/defaults/round-tripping.
- Modify: `tests/test_model_routes.py`
  Verifies `/api/modes` returns `maximum_len`.
- Create: `tests/test_chat_client.py`
  Verifies request shape and streaming-free completion behavior for the advisor client.
- Create: `tests/test_advisor_service.py`
  Verifies evidence hashing, prompt rendering, and digest generation.
- Create: `tests/test_advisor_routes.py`
  Verifies request validation and route response shape.

### Frontend

- Modify: `lcm-sr-ui/src/hooks/useGalleries.js`
  Adds `removeFromGallery()` and a lightweight per-gallery revision signal so advisor freshness can respond to membership changes.
- Modify: `lcm-sr-ui/src/hooks/useGalleries.test.jsx`
  Verifies removal and revision updates.
- Create: `lcm-sr-ui/src/hooks/useAdvisorState.js`
  IndexedDB persistence for advisor state by `gallery_id`.
- Create: `lcm-sr-ui/src/hooks/useAdvisorState.test.jsx`
  Verifies persistence, reload, and stale/error transitions.
- Create: `lcm-sr-ui/src/utils/advisorEvidence.js`
  Builds normalized evidence from gallery rows and computes a stable fingerprint.
- Create: `lcm-sr-ui/src/utils/advisorEvidence.test.mjs`
  Verifies evidence normalization and stable hashing input.
- Create: `lcm-sr-ui/src/hooks/useGalleryAdvisor.js`
  Orchestrates advisor state, digest rebuilds, auto-advice, and apply behavior.
- Create: `lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx`
  Verifies rebuild flow, status updates, and append/replace application.
- Create: `lcm-sr-ui/src/components/options/AdvisorPanel.jsx`
  Renders advisor controls and status UI.
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.jsx`
  Mounts `AdvisorPanel` directly under `Negative Prompt Templates`.
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`
  Verifies advisor rendering, `maximum_len` clamping, and apply controls.

---

### Task 1: Add `maximum_len` To Mode Config And `/api/modes`

**Files:**
- Modify: `server/mode_config.py`
- Modify: `server/model_routes.py`
- Modify: `tests/test_mode_config.py`
- Modify: `tests/test_model_routes.py`

- [ ] **Step 1: Reuse the existing maximum_len tests and deduplicate them if needed**

```python
# tests/test_mode_config.py
def test_mode_config_parses_maximum_len(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    default_size: 512x512
    maximum_len: 240
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    assert manager.get_mode("sdxl").maximum_len == 240


def test_mode_config_maximum_len_defaults_to_none(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sd15
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sd15:
    model: checkpoints/sd15/model.safetensors
    default_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    assert manager.get_mode("sd15").maximum_len is None
```

```python
# tests/test_model_routes.py
async def test_list_modes_includes_maximum_len():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sdxl",
        "resolution_sets": {"default": [{"size": "512x512", "aspect_ratio": "1:1"}]},
        "modes": {
            "sdxl": {
                "model": "checkpoints/sdxl/model.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 20,
                "default_guidance": 7.0,
                "maximum_len": 240,
                "negative_prompt_templates": {},
                "default_negative_prompt_template": None,
                "allow_custom_negative_prompt": False,
                "allowed_scheduler_ids": None,
                "default_scheduler_id": None,
                "resolution_set": "default",
                "resolution_options": [{"size": "512x512", "aspect_ratio": "1:1"}],
            }
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    assert data["modes"]["sdxl"]["maximum_len"] == 240
```

- [ ] **Step 2: Run the targeted backend tests to verify current failures**

Run:

```bash
python3 -m pytest tests/test_mode_config.py -k maximum_len -q
python3 -m pytest tests/test_model_routes.py -k maximum_len -q
```

Expected:

- at least one `maximum_len` mode-config test fails because `ModeConfig` does not expose/round-trip `maximum_len`
- route serialization test fails because `/api/modes` does not include `maximum_len`

- [ ] **Step 3: Implement `maximum_len` in config parsing and route serialization**

```python
# server/mode_config.py
@dataclass
class ModeConfig:
    name: str
    model: str
    loras: List[LoRAConfig] = field(default_factory=list)
    resolution_set: Optional[str] = None
    resolution_options: List[Dict[str, str]] = field(default_factory=list)
    default_size: str = "512x512"
    default_steps: int = 4
    default_guidance: float = 1.0
    maximum_len: Optional[int] = None
    ...

# inside _load_config()
mode = ModeConfig(
    name=mode_name,
    model=mode_data["model"],
    ...
    default_guidance=mode_data.get("default_guidance", 1.0),
    maximum_len=mode_data.get("maximum_len"),
    ...
)

# inside save_config()
if mode_data.get("maximum_len") is not None:
    mode_entry["maximum_len"] = mode_data.get("maximum_len")

# inside to_dict()
"maximum_len": mode.maximum_len,
```

```python
# server/model_routes.py
return {
    "default_mode": modes_dict["default_mode"],
    "resolution_sets": modes_dict.get("resolution_sets", {}),
    "modes": {
        name: {
            "model": mode_data["model"],
            "loras": mode_data["loras"],
            "default_size": mode_data["default_size"],
            "default_steps": mode_data["default_steps"],
            "default_guidance": mode_data["default_guidance"],
            "maximum_len": mode_data.get("maximum_len"),
            ...
        }
        for name, mode_data in modes_dict["modes"].items()
    },
}
```

- [ ] **Step 4: Run the targeted backend tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_mode_config.py -k maximum_len -q
python3 -m pytest tests/test_model_routes.py -k maximum_len -q
```

Expected:

- both commands pass

- [ ] **Step 5: Commit**

```bash
git add server/mode_config.py server/model_routes.py tests/test_mode_config.py tests/test_model_routes.py
git commit -m "feat: expose advisor maximum length in mode config"
```

### Task 2: Add The Advisor Digest Backend Route And Minimal LLM Client

**Files:**
- Modify: `requirements.txt`
- Modify: `server/lcm_sr_server.py`
- Create: `backends/chat_client.py`
- Create: `server/advisor_service.py`
- Create: `server/advisor_routes.py`
- Create: `tests/test_chat_client.py`
- Create: `tests/test_advisor_service.py`
- Create: `tests/test_advisor_routes.py`

- [ ] **Step 1: Write the failing backend tests for the client, service, and route**

```python
# tests/test_chat_client.py
import pytest
from types import SimpleNamespace

@pytest.mark.asyncio
async def test_chat_client_complete_posts_openai_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self): return None
        def json(self):
            return {"choices": [{"message": {"content": "digest text"}}]}

    class FakeAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def post(self, url, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("backends.chat_client.httpx.AsyncClient", FakeAsyncClient)

    from backends.chat_client import ChatCompletionsClient, ChatConfig

    client = ChatCompletionsClient(ChatConfig(endpoint="http://localhost:11434/v1", model="llama3.2"))
    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result == "digest text"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["model"] == "llama3.2"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
```

```python
# tests/test_advisor_service.py
def test_build_evidence_fingerprint_is_stable():
    from server.advisor_service import build_evidence_fingerprint

    evidence = {
        "version": 1,
        "gallery_id": "gal_1",
        "items": [{"cache_key": "a", "prompt": "cat", "steps": 10}],
    }

    assert build_evidence_fingerprint(evidence) == build_evidence_fingerprint(evidence)
```

```python
# tests/test_advisor_routes.py
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from server.lcm_sr_server import app

@pytest.mark.asyncio
async def test_advisor_digest_route_returns_digest():
    with patch("server.advisor_routes.generate_digest", new=AsyncMock(return_value={
        "digest_text": "stylized portrait",
        "meta": {"evidence_fingerprint": "sha256:abc", "model": "llama3.2"},
    })):
        client = TestClient(app)
        response = client.post("/api/advisors/digest", json={
            "gallery_id": "gal_1",
            "temperature": 0.4,
            "length_limit": 120,
            "evidence": {"version": 1, "gallery_id": "gal_1", "items": []},
        })

    assert response.status_code == 200
    assert response.json()["digest_text"] == "stylized portrait"
```

- [ ] **Step 2: Run the backend tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_chat_client.py tests/test_advisor_service.py tests/test_advisor_routes.py -q
```

Expected:

- import failures because `backends/chat_client.py`, `server/advisor_service.py`, and `server/advisor_routes.py` do not exist

- [ ] **Step 3: Add `httpx`, the chat client, the advisor service, and the route**

```python
# backends/chat_client.py
from dataclasses import dataclass
from typing import Optional
import os
import httpx

@dataclass
class ChatConfig:
    endpoint: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None

class ChatCompletionsClient:
    def __init__(self, config: ChatConfig):
        self.config = config
        self._api_key = os.environ.get(config.api_key_env, "")

    async def complete(self, messages: list[dict], *, max_tokens: int | None = None, temperature: float | None = None) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.endpoint.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            body = response.json()
        return body["choices"][0]["message"]["content"]
```

```python
# server/advisor_service.py
import hashlib
import json
from pydantic import BaseModel, Field
from backends.chat_client import ChatCompletionsClient, ChatConfig
from server.mode_config import get_mode_config

class EvidenceItem(BaseModel):
    cache_key: str
    added_at: int | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    size: str | None = None
    steps: int | None = None
    cfg: float | None = None
    scheduler_id: str | None = None
    seed: int | None = None
    superres_level: int | None = None
    metadata: dict = Field(default_factory=dict)

class EvidencePayload(BaseModel):
    version: int
    gallery_id: str
    items: list[EvidenceItem]

class AdvisorDigestRequest(BaseModel):
    gallery_id: str
    evidence: EvidencePayload
    temperature: float
    length_limit: int

def build_evidence_fingerprint(evidence: dict) -> str:
    normalized = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

async def generate_digest(request: AdvisorDigestRequest) -> dict:
    mode_name = get_mode_config().get_default_mode()
    mode = get_mode_config().get_mode(mode_name)
    chat_cfg = getattr(mode, "chat", None)
    if chat_cfg is None:
        raise ValueError("advisor digest requires chat configuration on the active mode")

    prompt = (
        "You are an image-style advisor. Analyze the evidence metadata and derive stable style themes, "
        "draft wording tendencies, and recurring generation parameter tendencies. "
        f"Limit the response to approximately {request.length_limit} words."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": request.evidence.model_dump_json()},
    ]

    client = ChatCompletionsClient(ChatConfig(
        endpoint=chat_cfg.endpoint,
        model=chat_cfg.model,
        api_key_env=chat_cfg.api_key_env,
        max_tokens=chat_cfg.max_tokens,
        temperature=chat_cfg.temperature,
        system_prompt=chat_cfg.system_prompt,
    ))
    digest_text = await client.complete(messages, temperature=request.temperature)
    return {
        "digest_text": digest_text,
        "meta": {
            "evidence_fingerprint": build_evidence_fingerprint(request.evidence.model_dump()),
            "model": chat_cfg.model,
        },
    }
```

```python
# server/advisor_routes.py
from fastapi import APIRouter, HTTPException
from server.advisor_service import AdvisorDigestRequest, generate_digest

router = APIRouter(prefix="/api/advisors", tags=["advisor"])

@router.post("/digest")
async def build_digest(request: AdvisorDigestRequest):
    try:
        return await generate_digest(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

```python
# server/lcm_sr_server.py
from server.advisor_routes import router as advisor_router
...
app.include_router(model_router)
app.include_router(telemetry_router)
app.include_router(workflow_router)
app.include_router(advisor_router)
```

```txt
# requirements.txt
httpx>=0.27.2
```

- [ ] **Step 4: Run the backend tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_chat_client.py tests/test_advisor_service.py tests/test_advisor_routes.py -q
```

Expected:

- all tests pass

- [ ] **Step 5: Commit**

```bash
git add requirements.txt backends/chat_client.py server/advisor_service.py server/advisor_routes.py server/lcm_sr_server.py tests/test_chat_client.py tests/test_advisor_service.py tests/test_advisor_routes.py
git commit -m "feat: add advisor digest backend service"
```

### Task 3: Add Gallery Mutation Tracking And Persisted Advisor State

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useGalleries.js`
- Modify: `lcm-sr-ui/src/hooks/useGalleries.test.jsx`
- Create: `lcm-sr-ui/src/hooks/useAdvisorState.js`
- Create: `lcm-sr-ui/src/hooks/useAdvisorState.test.jsx`

- [ ] **Step 1: Write the failing frontend tests for gallery revisions, removal, and advisor persistence**

```jsx
// lcm-sr-ui/src/hooks/useGalleries.test.jsx
it('removeFromGallery deletes a row by galleryId and cacheKey', async () => {
  const { result } = renderHook(() => useGalleries());
  await act(async () => { result.current.createGallery('Advisor'); });
  const galleryId = result.current.activeGalleryId;
  await act(async () => {
    await result.current.addToGallery('key_a', { serverImageUrl: null, params: {}, galleryId });
    await result.current.removeFromGallery(galleryId, 'key_a');
  });
  let items;
  await act(async () => { items = await result.current.getGalleryImages(galleryId); });
  expect(items).toEqual([]);
});

it('addToGallery bumps the gallery revision', async () => {
  const { result } = renderHook(() => useGalleries());
  await act(async () => { result.current.createGallery('Advisor'); });
  const galleryId = result.current.activeGalleryId;
  const before = result.current.getGalleryRevision(galleryId);
  await act(async () => {
    await result.current.addToGallery('key_a', { serverImageUrl: null, params: {}, galleryId });
  });
  expect(result.current.getGalleryRevision(galleryId)).toBeGreaterThan(before);
});
```

```jsx
// lcm-sr-ui/src/hooks/useAdvisorState.test.jsx
import 'fake-indexeddb/auto';
import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { useAdvisorState } from './useAdvisorState';

beforeEach(() => {
  localStorage.clear();
});

it('persists and reloads advisor state by gallery_id', async () => {
  const { result, rerender } = renderHook(({ galleryId }) => useAdvisorState(galleryId), {
    initialProps: { galleryId: 'gal_1' },
  });

  await act(async () => {
    await result.current.saveState({
      gallery_id: 'gal_1',
      digest_text: 'digest',
      advice_text: 'advice',
      status: 'fresh',
    });
  });

  rerender({ galleryId: 'gal_1' });
  await act(async () => { await result.current.reload(); });

  expect(result.current.state.digest_text).toBe('digest');
  expect(result.current.state.advice_text).toBe('advice');
});
```

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/hooks/useGalleries.test.jsx src/hooks/useAdvisorState.test.jsx
```

Expected:

- `removeFromGallery` and `getGalleryRevision` are undefined
- `useAdvisorState` import fails

- [ ] **Step 3: Implement gallery mutation tracking and advisor-state persistence**

```js
// lcm-sr-ui/src/hooks/useGalleries.js
const DB_VERSION = 2;
const STORE_NAME = 'gallery_items';

export function useGalleries() {
  const [galleries, setGalleries] = useState(() => loadGalleriesFromStorage());
  const [activeGalleryId, setActiveGalleryIdState] = useState(() => loadActiveFromStorage());
  const [galleryRevisions, setGalleryRevisions] = useState({});
  ...
  const bumpGalleryRevision = useCallback((galleryId) => {
    if (!galleryId) return;
    setGalleryRevisions((prev) => ({ ...prev, [galleryId]: (prev[galleryId] || 0) + 1 }));
  }, []);

  const removeFromGallery = useCallback(async (galleryId, cacheKey) => {
    if (!galleryId || !cacheKey) return;
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const rows = await promisifyRequest(store.index('galleryId').getAll(galleryId));
    await Promise.all(
      rows.filter((row) => row.cacheKey === cacheKey).map((row) => promisifyRequest(store.delete(row.id)))
    );
    bumpGalleryRevision(galleryId);
  }, [getDb, bumpGalleryRevision]);

  return {
    ...
    removeFromGallery,
    getGalleryRevision: (galleryId) => galleryRevisions[galleryId] || 0,
  };
}
```

```js
// lcm-sr-ui/src/hooks/useAdvisorState.js
import { useCallback, useEffect, useRef, useState } from 'react';

const DB_NAME = 'lcm-galleries';
const DB_VERSION = 2;
const STORE_NAME = 'advisor_states';

function openAdvisorDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'gallery_id' });
      }
    };
  });
}

export function useAdvisorState(galleryId) {
  const [state, setState] = useState(null);
  const dbRef = useRef(null);

  const getDb = useCallback(() => {
    if (!dbRef.current) dbRef.current = openAdvisorDb();
    return dbRef.current;
  }, []);

  const reload = useCallback(async () => {
    if (!galleryId) {
      setState(null);
      return null;
    }
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const next = await new Promise((resolve, reject) => {
      const req = tx.objectStore(STORE_NAME).get(galleryId);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
    setState(next);
    return next;
  }, [galleryId, getDb]);

  const saveState = useCallback(async (nextState) => {
    const db = await getDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    await new Promise((resolve, reject) => {
      const req = tx.objectStore(STORE_NAME).put(nextState);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    setState(nextState);
  }, [getDb]);

  useEffect(() => { void reload(); }, [reload]);

  return { state, setState, saveState, reload };
}
```

- [ ] **Step 4: Run the targeted frontend tests to verify they pass**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/hooks/useGalleries.test.jsx src/hooks/useAdvisorState.test.jsx
```

Expected:

- all tests pass

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useGalleries.js lcm-sr-ui/src/hooks/useGalleries.test.jsx lcm-sr-ui/src/hooks/useAdvisorState.js lcm-sr-ui/src/hooks/useAdvisorState.test.jsx
git commit -m "feat: persist advisor state and track gallery revisions"
```

### Task 4: Build Evidence And Orchestrate Advisor Rebuilds

**Files:**
- Create: `lcm-sr-ui/src/utils/advisorEvidence.js`
- Create: `lcm-sr-ui/src/utils/advisorEvidence.test.mjs`
- Create: `lcm-sr-ui/src/hooks/useGalleryAdvisor.js`
- Create: `lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx`

- [ ] **Step 1: Write the failing tests for evidence building and advisor orchestration**

```js
// lcm-sr-ui/src/utils/advisorEvidence.test.mjs
import { describe, expect, it } from 'vitest';
import { buildAdvisorEvidence } from './advisorEvidence';

describe('buildAdvisorEvidence', () => {
  it('normalizes gallery rows into versioned evidence', () => {
    const evidence = buildAdvisorEvidence('gal_1', [
      {
        cacheKey: 'abc',
        addedAt: 1712790000,
        params: { prompt: 'cat', negativePrompt: 'blur', size: '512x512', steps: 8, cfg: 2.5, schedulerId: 'euler', seed: 123 },
      },
    ]);

    expect(evidence).toEqual({
      version: 1,
      gallery_id: 'gal_1',
      items: [
        {
          cache_key: 'abc',
          added_at: 1712790000,
          prompt: 'cat',
          negative_prompt: 'blur',
          size: '512x512',
          steps: 8,
          cfg: 2.5,
          scheduler_id: 'euler',
          seed: 123,
          superres_level: null,
          metadata: {},
        },
      ],
    });
  });
});
```

```jsx
// lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx
import { renderHook, act, waitFor } from '@testing-library/react';
import { vi, expect, it } from 'vitest';
import { useGalleryAdvisor } from './useGalleryAdvisor';

const api = {
  fetchPost: vi.fn(),
};

it('rebuilds digest and seeds advice text when no edits exist', async () => {
  api.fetchPost.mockResolvedValue({
    digest_text: 'Painterly neon portrait',
    meta: { evidence_fingerprint: 'sha256:abc' },
  });

  const { result } = renderHook(() => useGalleryAdvisor({
    galleryId: 'gal_1',
    galleryRevision: 1,
    galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
    maximumLen: 240,
    api,
    advisorState: null,
    saveAdvisorState: vi.fn(),
    setDraftPrompt: vi.fn(),
  }));

  await act(async () => {
    await result.current.rebuildAdvisor();
  });

  await waitFor(() => expect(result.current.state.digest_text).toBe('Painterly neon portrait'));
  expect(result.current.state.advice_text).toBe('Painterly neon portrait');
});
```

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/utils/advisorEvidence.test.mjs src/hooks/useGalleryAdvisor.test.jsx
```

Expected:

- missing module failures for `advisorEvidence` and `useGalleryAdvisor`

- [ ] **Step 3: Implement evidence normalization and advisor orchestration**

```js
// lcm-sr-ui/src/utils/advisorEvidence.js
export function buildAdvisorEvidence(galleryId, rows) {
  return {
    version: 1,
    gallery_id: galleryId,
    items: (rows || []).map((row) => ({
      cache_key: row.cacheKey,
      added_at: row.addedAt ?? null,
      prompt: row.params?.prompt ?? null,
      negative_prompt: row.params?.negativePrompt ?? null,
      size: row.params?.size ?? null,
      steps: row.params?.steps ?? null,
      cfg: row.params?.cfg ?? null,
      scheduler_id: row.params?.schedulerId ?? null,
      seed: row.params?.seed ?? null,
      superres_level: row.params?.superresLevel ?? null,
      metadata: row.params?.metadata ?? {},
    })),
  };
}
```

```js
// lcm-sr-ui/src/hooks/useGalleryAdvisor.js
import { useCallback, useEffect, useMemo, useState } from 'react';
import { buildAdvisorEvidence } from '../utils/advisorEvidence';

export function useGalleryAdvisor({
  galleryId,
  galleryRevision,
  galleryImages,
  maximumLen,
  api,
  advisorState,
  saveAdvisorState,
  setDraftPrompt,
}) {
  const [state, setState] = useState(advisorState);

  useEffect(() => { setState(advisorState); }, [advisorState]);

  const evidence = useMemo(() => buildAdvisorEvidence(galleryId, galleryImages || []), [galleryId, galleryImages]);

  useEffect(() => {
    if (!state || !galleryId) return;
    if ((state.gallery_revision ?? 0) !== galleryRevision) {
      const next = { ...state, status: 'stale' };
      setState(next);
      void saveAdvisorState(next);
    }
  }, [galleryId, galleryRevision, saveAdvisorState, state]);

  const rebuildAdvisor = useCallback(async () => {
    const building = { ...(state || {}), gallery_id: galleryId, status: 'building' };
    setState(building);
    await saveAdvisorState(building);

    try {
      const response = await api.fetchPost('/api/advisors/digest', {
        gallery_id: galleryId,
        temperature: state?.temperature ?? 0.4,
        length_limit: Math.min(state?.length_limit ?? maximumLen ?? 0, maximumLen ?? 0),
        evidence,
      });

      const shouldReplaceAdvice = !state?.advice_text || state.advice_text === state.digest_text;
      const next = {
        ...(state || {}),
        gallery_id: galleryId,
        gallery_revision: galleryRevision,
        digest_text: response.digest_text,
        advice_text: shouldReplaceAdvice ? response.digest_text : state.advice_text,
        evidence_fingerprint: response.meta?.evidence_fingerprint ?? null,
        status: 'fresh',
        updated_at: Date.now(),
        error_message: null,
      };
      setState(next);
      await saveAdvisorState(next);
      return next;
    } catch (error) {
      const failed = {
        ...(state || {}),
        gallery_id: galleryId,
        status: 'error',
        error_message: error.message || 'Advisor rebuild failed',
      };
      setState(failed);
      await saveAdvisorState(failed);
      throw error;
    }
  }, [api, evidence, galleryId, galleryRevision, maximumLen, saveAdvisorState, state]);

  const applyAdvice = useCallback((mode) => {
    if (!state?.advice_text) return;
    setDraftPrompt((current) => {
      if (mode === 'replace') return state.advice_text;
      return current ? `${current}\n\n${state.advice_text}` : state.advice_text;
    });
  }, [setDraftPrompt, state?.advice_text]);

  return { state, setState, rebuildAdvisor, applyAdvice, evidence };
}
```

- [ ] **Step 4: Run the targeted frontend tests to verify they pass**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/utils/advisorEvidence.test.mjs src/hooks/useGalleryAdvisor.test.jsx
```

Expected:

- both tests pass

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/utils/advisorEvidence.js lcm-sr-ui/src/utils/advisorEvidence.test.mjs lcm-sr-ui/src/hooks/useGalleryAdvisor.js lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx
git commit -m "feat: add gallery advisor evidence and rebuild orchestration"
```

### Task 5: Add The Advisor Panel And Wire It Into `OptionsPanel`

**Files:**
- Create: `lcm-sr-ui/src/components/options/AdvisorPanel.jsx`
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.jsx`
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`

- [ ] **Step 1: Write the failing UI tests for advisor controls, status, and apply modes**

```jsx
// lcm-sr-ui/src/components/options/OptionsPanel.test.jsx
it('renders the advisor section under negative prompt controls when a gallery is active', async () => {
  const params = makeParams();
  render(
    <OptionsPanel
      params={params}
      inputImage={null}
      comfyInputImage={null}
      selectedParams={null}
      blurredSelectedParams={null}
      selectedMsgId={null}
      onClearSelection={vi.fn()}
      onApplyPromptDelta={vi.fn()}
      onApplySeedDelta={vi.fn()}
      onRerunSelected={vi.fn()}
      onPersistSelectedParams={vi.fn()}
      dreamState={{ isDreaming: false, temperature: 0.5, interval: 10, onStart: vi.fn(), onStop: vi.fn(), onGuide: vi.fn(), onTemperatureChange: vi.fn(), onIntervalChange: vi.fn() }}
      onSuperResUpload={vi.fn()}
      uploadFile={null}
      onUploadFileChange={vi.fn()}
      srMagnitude={1}
      onSrMagnitudeChange={vi.fn()}
      onSuperResSelected={vi.fn()}
      serverLabel="test"
      onRunComfy={vi.fn()}
      onClearCache={vi.fn()}
      getCacheStats={vi.fn().mockResolvedValue(null)}
      onClearHistory={vi.fn()}
      queueState={{ items: [] }}
      initImage={null}
      onClearInitImage={vi.fn()}
      denoiseStrength={0.75}
      onDenoiseStrengthChange={vi.fn()}
      modeState={makeModeState('SDXL', { maximum_len: 240, allow_custom_negative_prompt: true })}
      galleryState={{ galleries: [{ id: 'gal_1', name: 'Advisor' }], activeGalleryId: 'gal_1', setActiveGalleryId: vi.fn(), getGalleryImages: vi.fn().mockResolvedValue([]), getGalleryRevision: vi.fn(() => 0) }}
    />
  );

  expect(await screen.findByText('Advisor')).toBeTruthy();
  expect(screen.getByLabelText('Advisor length')).toHaveAttribute('max', '240');
});

it('offers append and replace apply modes', async () => {
  renderOptionsPanel(makeModeState('SDXL', { maximum_len: 240, allow_custom_negative_prompt: true }));
  expect(await screen.findByText('Apply Advice')).toBeTruthy();
  expect(screen.getByRole('option', { name: 'Append' })).toBeTruthy();
  expect(screen.getByRole('option', { name: 'Replace' })).toBeTruthy();
});
```

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/components/options/OptionsPanel.test.jsx
```

Expected:

- assertions fail because there is no advisor panel and no advisor length/apply controls

- [ ] **Step 3: Implement `AdvisorPanel` and mount it under negative prompt templates**

```jsx
// lcm-sr-ui/src/components/options/AdvisorPanel.jsx
import React from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

export function AdvisorPanel({
  state,
  maximumLen,
  onAutoAdviceChange,
  onTemperatureChange,
  onLengthChange,
  onAdviceChange,
  onResetToDigest,
  onRebuild,
  onApply,
  applyMode,
  onApplyModeChange,
}) {
  return (
    <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
      <Label>Advisor</Label>

      <label className="flex items-center justify-between text-sm">
        <span>Auto-Advice</span>
        <input
          aria-label="Auto advice"
          type="checkbox"
          checked={Boolean(state?.auto_advice)}
          onChange={(e) => onAutoAdviceChange(e.target.checked)}
        />
      </label>

      <div className="space-y-2">
        <Label htmlFor="advisor-temperature">Temperature</Label>
        <input id="advisor-temperature" aria-label="Advisor temperature" type="range" min="0" max="1" step="0.05" value={state?.temperature ?? 0.4} onChange={(e) => onTemperatureChange(Number(e.target.value))} />
      </div>

      <div className="space-y-2">
        <Label htmlFor="advisor-length">Length</Label>
        <input id="advisor-length" aria-label="Advisor length" type="range" min="0" max={maximumLen ?? 0} step="1" value={state?.length_limit ?? 0} onChange={(e) => onLengthChange(Number(e.target.value))} />
      </div>

      <div className="text-xs" data-status={state?.status || 'idle'}>
        {state?.status === 'building' ? 'Building digest…' : state?.status === 'error' ? state?.error_message || 'Advisor error' : state?.updated_at ? `Updated ${new Date(state.updated_at).toLocaleString()}` : 'No digest yet'}
      </div>

      <Textarea aria-label="Advisor advice" value={state?.advice_text ?? ''} onChange={(e) => onAdviceChange(e.target.value)} className="min-h-[120px] resize-none rounded-2xl" />

      <div className="space-y-2">
        <Label>Apply Mode</Label>
        <Select value={applyMode} onValueChange={onApplyModeChange}>
          <SelectTrigger aria-label="Apply advice mode">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="append">Append</SelectItem>
            <SelectItem value="replace">Replace</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex gap-2">
        <Button type="button" onClick={onRebuild}>Rebuild Advisor</Button>
        <Button type="button" variant="secondary" onClick={onResetToDigest}>Reset To Digest</Button>
        <Button type="button" onClick={() => onApply(applyMode)} disabled={!state?.advice_text}>Apply Advice</Button>
      </div>
    </div>
  );
}
```

```jsx
// inside lcm-sr-ui/src/components/options/OptionsPanel.jsx
import { createApiClient, createApiConfig } from '../../utils/api';
import { useAdvisorState } from '../../hooks/useAdvisorState';
import { useGalleryAdvisor } from '../../hooks/useGalleryAdvisor';
import { AdvisorPanel } from './AdvisorPanel';

...
const apiClientRef = useRef(null);
if (!apiClientRef.current) {
  apiClientRef.current = createApiClient(createApiConfig());
}
const advisorApi = apiClientRef.current;
const activeGalleryId = galleryState?.activeGalleryId ?? null;
const [galleryImages, setGalleryImages] = useState([]);
const galleryRevision = galleryState?.getGalleryRevision?.(activeGalleryId) ?? 0;
const { state: advisorState, saveState: saveAdvisorState } = useAdvisorState(activeGalleryId);
const [applyMode, setApplyMode] = useState('append');

useEffect(() => {
  if (!activeGalleryId || !galleryState?.getGalleryImages) {
    setGalleryImages([]);
    return;
  }
  void galleryState.getGalleryImages(activeGalleryId).then(setGalleryImages);
}, [activeGalleryId, galleryRevision, galleryState]);

const advisor = useGalleryAdvisor({
  galleryId: activeGalleryId,
  galleryRevision,
  galleryImages,
  maximumLen: resolvedMode?.maximum_len ?? 0,
  api: advisorApi,
  advisorState,
  saveAdvisorState,
  setDraftPrompt: params.setPrompt,
});
...
{(negativePromptOptions.length > 0 || resolvedMode?.allow_custom_negative_prompt || schedulerOptions.length > 0) && (
  <>
    <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
      ...
    </div>
    {activeGalleryId && (
      <AdvisorPanel
        state={advisor.state}
        maximumLen={resolvedMode?.maximum_len ?? 0}
        onAutoAdviceChange={(value) => advisor.setState((prev) => ({ ...(prev || {}), auto_advice: value }))}
        onTemperatureChange={(value) => advisor.setState((prev) => ({ ...(prev || {}), temperature: value }))}
        onLengthChange={(value) => advisor.setState((prev) => ({ ...(prev || {}), length_limit: value }))}
        onAdviceChange={(value) => advisor.setState((prev) => ({ ...(prev || {}), advice_text: value }))}
        onResetToDigest={() => advisor.setState((prev) => ({ ...(prev || {}), advice_text: prev?.digest_text || '' }))}
        onRebuild={advisor.rebuildAdvisor}
        onApply={advisor.applyAdvice}
        applyMode={applyMode}
        onApplyModeChange={setApplyMode}
      />
    )}
  </>
)}
```

- [ ] **Step 4: Run the targeted frontend tests to verify they pass**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/components/options/OptionsPanel.test.jsx
```

Expected:

- advisor panel tests pass

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/components/options/AdvisorPanel.jsx lcm-sr-ui/src/components/options/OptionsPanel.jsx lcm-sr-ui/src/components/options/OptionsPanel.test.jsx
git commit -m "feat: add gallery advisor panel to options UI"
```

### Task 6: Wire Freshness Updates And End-To-End Advisor Behavior

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useGalleryAdvisor.js`
- Modify: `lcm-sr-ui/src/hooks/useGalleries.js`
- Modify: `lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx`
- Modify: `lcm-sr-ui/src/hooks/useGalleries.test.jsx`

- [ ] **Step 1: Write the failing tests for stale transitions and preserved advice on rebuild**

```jsx
// lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx
it('marks advisor state stale when the gallery revision changes', async () => {
  const saveAdvisorState = vi.fn();
  const { result, rerender } = renderHook((props) => useGalleryAdvisor(props), {
    initialProps: {
      galleryId: 'gal_1',
      galleryRevision: 1,
      galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
      maximumLen: 240,
      api: { fetchPost: vi.fn() },
      advisorState: { gallery_id: 'gal_1', gallery_revision: 1, digest_text: 'digest', advice_text: 'digest', status: 'fresh' },
      saveAdvisorState,
      setDraftPrompt: vi.fn(),
    },
  });

  rerender({
    galleryId: 'gal_1',
    galleryRevision: 2,
    galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }, { cacheKey: 'xyz', addedAt: 2, params: { prompt: 'dog' } }],
    maximumLen: 240,
    api: { fetchPost: vi.fn() },
    advisorState: { gallery_id: 'gal_1', gallery_revision: 1, digest_text: 'digest', advice_text: 'digest', status: 'fresh' },
    saveAdvisorState,
    setDraftPrompt: vi.fn(),
  });

  await waitFor(() => expect(result.current.state.status).toBe('stale'));
});

it('preserves user-edited advice text when a rebuild returns a new digest', async () => {
  const api = { fetchPost: vi.fn().mockResolvedValue({ digest_text: 'new digest', meta: { evidence_fingerprint: 'sha256:new' } }) };
  const { result } = renderHook(() => useGalleryAdvisor({
    galleryId: 'gal_1',
    galleryRevision: 1,
    galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
    maximumLen: 240,
    api,
    advisorState: { gallery_id: 'gal_1', digest_text: 'old digest', advice_text: 'custom user advice', temperature: 0.4, length_limit: 120 },
    saveAdvisorState: vi.fn(),
    setDraftPrompt: vi.fn(),
  }));

  await act(async () => {
    await result.current.rebuildAdvisor();
  });

  expect(result.current.state.digest_text).toBe('new digest');
  expect(result.current.state.advice_text).toBe('custom user advice');
});
```

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/hooks/useGalleryAdvisor.test.jsx src/hooks/useGalleries.test.jsx
```

Expected:

- stale transition and preserved-advice assertions fail before the freshness logic is finished

- [ ] **Step 3: Finish the freshness logic and state preservation behavior**

```js
// lcm-sr-ui/src/hooks/useGalleryAdvisor.js
useEffect(() => {
  if (!galleryId || !state) return;
  if ((state.gallery_revision ?? 0) === galleryRevision) return;

  const next = {
    ...state,
    gallery_revision: galleryRevision,
    status: 'stale',
  };
  setState(next);
  void saveAdvisorState(next);
}, [galleryId, galleryRevision, saveAdvisorState, state]);

const rebuildAdvisor = useCallback(async () => {
  ...
  const isUserEdited = Boolean(state?.advice_text) && state.advice_text !== state.digest_text;
  const next = {
    ...(state || {}),
    gallery_id: galleryId,
    gallery_revision: galleryRevision,
    digest_text: response.digest_text,
    advice_text: isUserEdited ? state.advice_text : response.digest_text,
    status: 'fresh',
    updated_at: Date.now(),
    error_message: null,
  };
  ...
}, [...]);
```

```js
// lcm-sr-ui/src/hooks/useGalleries.js
const addToGallery = useCallback(async (cacheKey, payload) => {
  ...
  await promisifyRequest(rwTx.objectStore(STORE_NAME).put(row));
  bumpGalleryRevision(galleryId);
}, [getDb, bumpGalleryRevision]);
```

- [ ] **Step 4: Run the focused frontend tests and the key integration slice**

Run:

```bash
npm --prefix lcm-sr-ui test -- --run src/hooks/useGalleryAdvisor.test.jsx src/hooks/useGalleries.test.jsx src/components/options/OptionsPanel.test.jsx
python3 -m pytest tests/test_mode_config.py tests/test_model_routes.py tests/test_chat_client.py tests/test_advisor_service.py tests/test_advisor_routes.py -q
```

Expected:

- all listed frontend and backend tests pass

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useGalleryAdvisor.js lcm-sr-ui/src/hooks/useGalleries.js lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx lcm-sr-ui/src/hooks/useGalleries.test.jsx lcm-sr-ui/src/components/options/OptionsPanel.test.jsx tests/test_mode_config.py tests/test_model_routes.py tests/test_chat_client.py tests/test_advisor_service.py tests/test_advisor_routes.py
git commit -m "feat: finalize gallery advisor freshness and apply flow"
```

---

## Coverage Check

- `maximum_len` mode constraint: Task 1
- advisor digest backend and LLM transport: Task 2
- client-side `AdvisorState` persistence per `gallery_id`: Task 3
- typed `Evidence` shape and fingerprinting: Task 4
- advisor UI under `Negative Prompt Templates`: Task 5
- deterministic `Append` and `Replace` application to draft only: Task 5
- stale/fresh/building/error lifecycle and gallery mutation impact: Task 6

## Placeholder Scan

- No `TODO`, `TBD`, or deferred code references remain in the task steps.
- Every code-changing step includes concrete code to add or modify.
- Every test step includes an exact command.

## Type Consistency

- Backend request object names are consistent: `AdvisorDigestRequest`, `EvidencePayload`, `EvidenceItem`.
- Frontend object names are consistent: `AdvisorState`, `digest_text`, `advice_text`, `gallery_revision`, `maximumLen`.
- Apply modes are consistently `append` and `replace`.
