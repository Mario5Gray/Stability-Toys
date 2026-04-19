# Vision Chat Design

**Date:** 2026-04-18
**Status:** Approved

## Overview

Add vision capability to the main chat composer. An eye icon sits between the Send and Cancel buttons. When the user has an active (selected) image and the mode's chat delegate supports vision, clicking the eye icon resizes the image (longest-edge constrained), encodes it as base64, and sends it alongside the composer draft text as a multimodal chat job. Generation continues concurrently — vision jobs use the same async WS job system as text chat.

---

## Config

### `chat_delegates` — new optional fields

```yaml
chat_delegates:
  sdxl_advisor:
    connection: local_default
    model: "gemma3-1b"
    max_tokens: 750
    temperature: 0.4
    system_prompt: "You are a concise SDXL prompt advisor."
    vision: true
    vision_system_prompt: "You are a visual analyst. Describe what you see concisely."
    vision_default_prompt: "Describe this image."
    vision_resize: 512
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `vision` | bool | `false` | Enables multimodal requests to this delegate |
| `vision_system_prompt` | str | `null` | System prompt for vision requests; falls back to `system_prompt` if absent |
| `vision_default_prompt` | str | `"Describe this image."` | Prompt used when composer draft is empty |
| `vision_resize` | int | `512` | Longest-edge pixel limit; aspect ratio preserved |

### `ChatDelegateConfig` dataclass additions

```python
vision: bool = False
vision_system_prompt: Optional[str] = None
vision_default_prompt: str = "Describe this image."
vision_resize: int = 512
```

### `/api/modes` response

`model_routes.py` adds three fields to each mode entry:

```python
"vision_enabled": bool(mode_data.get("vision_enabled")),
"vision_resize": mode_data.get("vision_resize", 512),
"vision_default_prompt": mode_data.get("vision_default_prompt", "Describe this image."),
```

`to_dict()` in `mode_config.py` derives these from the resolved delegate:

```python
delegate = config.chat_delegates.get(mode.chat_delegate)
"vision_enabled": bool(delegate and delegate.vision),
"vision_resize": delegate.vision_resize if delegate else 512,
"vision_default_prompt": delegate.vision_default_prompt if delegate else "Describe this image.",
```

---

## Backend

### `backends/chat_client.py`

- Broaden `messages` type annotation from `List[Dict[str, str]]` to `List[Dict[str, Any]]` in `stream()`, `complete()`, and `_request_payload()`.
- No logic changes — list-content user messages already pass through the JSON payload untouched.

### `server/ws_routes.py`

#### `_run_chat` additions

```python
image_b64 = params.get("image_b64")  # base64 PNG string, no data: prefix

# Gate: reject vision request if delegate doesn't support it
if image_b64 and not chat_cfg.vision:
    await hub.send(client_id, {
        "type": "job:error",
        "jobId": job_id,
        "error": "Vision not enabled for this delegate",
    })
    return

# Select system prompt
effective_system = (
    chat_cfg.vision_system_prompt if image_b64 and chat_cfg.vision_system_prompt
    else chat_cfg.system_prompt
)

# Build user content
if image_b64:
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ]
else:
    user_content = prompt  # existing text-only path unchanged

messages = _build_chat_messages(user_content, effective_system)
```

#### `_build_chat_messages` signature

```python
def _build_chat_messages(
    prompt: Union[str, List[Dict[str, Any]]],
    system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
```

User message `content` field accepts `str` or `list` — no other changes.

---

## Frontend

### `lcm-sr-ui/src/utils/resizeImageToLongestEdge.js` (new)

Pure utility. Loads an image URL onto an offscreen canvas, scales so the longest edge equals `maxPx` (aspect ratio preserved), returns a base64 PNG string (no `data:` prefix).

```js
export async function resizeImageToLongestEdge(url, maxPx = 512) {
  // returns Promise<string>  — bare base64, no data: prefix
}
```

### `slashCtx` additions in `App.jsx`

```js
visionImageUrl: selectedMsg?.serverImageUrl || selectedMsg?.imageUrl || null,
visionEnabled: Boolean(modeState.activeMode?.vision_enabled),
visionResize: modeState.activeMode?.vision_resize ?? 512,
visionDefaultPrompt: modeState.activeMode?.vision_default_prompt ?? 'Describe this image.',
```

Added to the `useMemo` deps array: `selectedMsg`, `modeState.activeMode?.vision_enabled`, `modeState.activeMode?.vision_resize`.

### `MessageComposer.jsx`

**Eye icon** between Send and Cancel:

```jsx
import { Eye } from 'lucide-react';

<Button
  variant="ghost"
  size="icon"
  disabled={!slashCtx?.visionEnabled || !slashCtx?.visionImageUrl}
  onClick={handleVisionSend}
  title="Send image to vision model"
>
  <Eye className="h-4 w-4" />
</Button>
```

**`handleVisionSend`** (inside composer):

```js
const handleVisionSend = useCallback(async () => {
  const url = slashCtx.visionImageUrl;
  if (!url) return;

  const prompt = draft.trim() || (slashCtx.visionDefaultPrompt ?? 'Describe this image.');
  const maxPx = slashCtx.visionResize ?? 512;
  const image_b64 = await resizeImageToLongestEdge(url, maxPx);

  const userId = nowId();
  const assistantId = nowId();

  slashCtx.addMessage([
    { id: userId, role: 'user', kind: MESSAGE_KINDS.CHAT, text: prompt ?? '(image)', ts: Date.now() },
    { id: assistantId, role: 'assistant', kind: MESSAGE_KINDS.CHAT, text: '', streaming: true, jobId: null, ts: Date.now() },
  ]);

  // rAF delta coalescing (same pattern as chat.js)
  let rafBuffer = '', rafPending = false, terminated = false;
  const flushDelta = () => {
    const chunk = rafBuffer; rafBuffer = ''; rafPending = false;
    if (chunk && !terminated) slashCtx.updateMessage(assistantId, (prev) => ({ ...prev, text: prev.text + chunk }));
  };

  const handle = slashCtx.chatJob.start({
    prompt: prompt ?? '',
    image_b64,
    onAck: ({ jobId }) => slashCtx.updateMessage(assistantId, { jobId }),
    onDelta: (text) => {
      if (terminated) return;
      rafBuffer += text;
      if (!rafPending) { rafPending = true; requestAnimationFrame(flushDelta); }
    },
    onComplete: ({ text }) => { terminated = true; slashCtx.updateMessage(assistantId, { text, streaming: false, jobId: null }); },
    onError: (errMsg) => {
      terminated = true;
      slashCtx.updateMessage(assistantId, (prev) => ({
        ...prev, kind: MESSAGE_KINDS.ERROR, prevText: prev.text, text: errMsg, streaming: false, jobId: null,
      }));
    },
  });

  slashCtx.updateMessage(assistantId, { cancelHandle: handle });
  setDraft('');
}, [draft, slashCtx]);
```

### `useChatJob.js`

No changes — `image_b64` is forwarded through `params` in the WS `job:submit` message unchanged:

```js
wsClient.send({
  type: 'job:submit',
  id: corrId,
  jobType: 'chat',
  params: { prompt, image_b64, stream: true },  // image_b64 may be undefined
});
```

---

## Data Flow

```text
User: types question (optional) → clicks Eye
Composer:
  1. fetch visionImageUrl from slashCtx
  2. resizeImageToLongestEdge(url, visionResize) → base64
  3. addMessage [user bubble, assistant bubble (streaming)]
  4. chatJob.start({ prompt, image_b64 })
  5. clear draft

WS (useChatJob):
  job:submit { jobType:'chat', params:{ prompt, image_b64, stream:true } }
  → job:ack { jobId }
  → job:progress { delta } × N   ←── concurrent with generation jobs
  → job:complete { outputs:[{text}] }

Backend (_run_chat):
  1. detect image_b64 in params
  2. gate: vision flag on delegate
  3. select vision_system_prompt or system_prompt
  4. build multimodal user message
  5. stream via ChatCompletionsClient (unchanged)
```

---

## What Does Not Change

- `useChatJob.js` — no modifications
- `MessageBubble.jsx` — CHAT kind already renders streaming/complete/error states
- Generation pipeline — vision jobs are independent asyncio tasks, no contention
- Slash command registry — no new commands added

---

## Testing

- Unit test `resizeImageToLongestEdge`: landscape, portrait, square, already-small image (no upscale)
- Unit test `_build_chat_messages` with list content
- Integration: `_run_chat` with `image_b64` → multimodal message sent to client
- Integration: `_run_chat` with `image_b64` but `vision=False` delegate → `job:error`
- Config test: `ChatDelegateConfig` parses new fields; missing fields default correctly
