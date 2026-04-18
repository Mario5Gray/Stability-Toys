# Slash-command `/chat` in main chat — Design

**FP issue:** STABL-jhpayntq
**Parent:** STABL-grarbnxp (OpenAI-compatible chat completions backend)
**Related:** STABL-eaontsix (LLM config panel)
**Date:** 2026-04-18

## Summary

Add a slash-command framework to the main chat composer, with `/chat` as the first registered command. Typing `/chat <message>` dispatches a WebSocket `jobType=chat` to the active mode's configured chat backend and streams the assistant reply inline into the existing chat window. A hint popover surfaces available commands when the user types `/`. Extensible by design: future commands (`/gen`, `/seed`, `/help`) register into the same module without modifying the composer.

## Background

Backend chat plumbing is in place (STABL-grarbnxp):

- WebSocket `job:submit` accepts `jobType=chat` with `{prompt, stream, max_tokens?, temperature?}`.
- Streaming deltas arrive as `job:progress {delta}`; final aggregate as `job:complete {outputs:[{text}]}`.
- Backend resolves chat config from the global `chat.<mode>` mapping.
- `/api/modes` exposes `chat_enabled: bool` per mode (see `server/model_routes.py:162`).

The UI currently has no way to exercise chat. The existing main composer (`lcm-sr-ui/src/components/chat/MessageComposer.jsx`) sends every draft through `onSendPrompt`, which routes to image generation via `useImageGeneration`/`useComfyJobWs`. Chat turns and image-gen turns must coexist in the same chat window without colliding.

## Goals

- Dispatch `/chat <message>` through the existing WS chat job path.
- Render streaming deltas inline in the chat window as a new `CHAT` message kind.
- Provide an extensible slash-command registry so new commands add in one file.
- Hint popover listing available commands when `/` is typed, with keyboard nav.
- Gate `/chat` on `chat_enabled` for the active mode; surface a clear error when disabled.
- Handle `job:error` and cancellation gracefully.

## Non-goals

- Multi-turn conversation memory across turns (stateless single-shot first).
- Tool-use / function-calling.
- Persistence of chat transcripts beyond existing `localStorage` message store.
- Commands beyond `/chat` (framework only; additional commands tracked separately).
- Per-turn params UI (`temperature`, `max_tokens`). Backend applies config from `chat.<mode>`; STABL-eaontsix will add a config panel.

## Architecture

Four new modules and one edit:

| File | Role |
| ---- | ---- |
| `lcm-sr-ui/src/lib/slashCommands.js` | Pure-JS registry. `register`, `parse`, `list`. |
| `lcm-sr-ui/src/lib/slashCommands/chat.js` | Registers `/chat`. Handler owns streaming lifecycle. |
| `lcm-sr-ui/src/hooks/useChatJob.js` | WS correlation hook for `jobType=chat`. |
| `lcm-sr-ui/src/components/chat/SlashHintPopover.jsx` | Hint popover + keyboard nav. |
| `lcm-sr-ui/src/components/chat/MessageComposer.jsx` (edit) | Detect slash prefix, render popover, dispatch via registry. |

### `slashCommands.js` — registry

Pure JavaScript. No React.

```js
// Shape
register(name, {
  description: string,                   // shown in popover
  enabled: (ctx) => bool,                // optional; defaults to () => true
  disabledReason: (ctx) => string,       // optional; used for tooltip when enabled() is false
  handler: ({ args, ctx }) => void,
});

parse(input) -> { command, args, raw } | null;
// Parses only when input starts with "/". Splits at first whitespace.

list(ctx) -> Array<{ name, description, enabled, disabledReason }>;
// Resolves `enabled(ctx)` and, when false, `disabledReason(ctx)` for each command.
```

For `/chat`: `enabled(ctx)` returns `ctx.chatEnabled`; `disabledReason(ctx)` returns the string `Chat not enabled for mode <activeMode>`.

`ctx` is supplied by the composer at dispatch time:

```js
{
  addMessage, updateMessage,     // from useChatMessages
  activeMode,                    // string — current mode name
  chatEnabled,                   // bool — from /api/modes for activeMode
  chatJob,                       // useChatJob instance
  wsClient,                      // for job:cancel
}
```

### `slashCommands/chat.js` — `/chat` handler

Registers once at module load (imported from `App.jsx` or a bootstrap file).

Handler flow (see Data flow below).

### `useChatJob.js` — hook

Mirrors `useComfyJobWs.js:13` shape but for `jobType=chat`. Owns its own correlation id and subscription cleanup. Supports multiple in-flight chat jobs by allowing `start()` to return a handle with per-job unsubscribers.

API:

```js
const chatJob = useChatJob();
const handle = chatJob.start({
  prompt: string,
  onAck: ({ jobId }) => void,
  onDelta: (text) => void,
  onComplete: ({ text }) => void,
  onError: (errMessage) => void,
});
handle.cancel();   // sends job:cancel for this handle's jobId
```

Internally:

- Generate corrId via `nextCorrId()`.
- Subscribe to `job:ack`, `job:progress`, `job:complete`, `job:error` filtered by corrId/jobId.
- `send({type:"job:submit", id:corrId, jobType:"chat", prompt, stream:true})`.
- `cancel()` sends `{type:"job:cancel", jobId}` and clears subs.

### `SlashHintPopover.jsx`

Anchored above the composer. Visible when draft starts with `/` and caret is within the command token. Renders list from `slashCommands.list(ctx)` filtered by prefix.

- Arrow Up/Down: change highlighted entry.
- Tab or Enter: complete the command token (insert `<name> ` into draft). If entry is disabled, no-op.
- Esc: dismiss popover; Esc again clears draft only if popover already hidden (existing composer behavior preserved).
- Disabled entries render greyed with a tooltip showing the reason (e.g., `"chat not enabled for mode sd15"`).

### `MessageComposer.jsx` — edits

- Accept new props: `slashCtx` — the dispatch context object above.
- On `onChange`, detect `/` at start of draft. Toggle popover open/closed accordingly.
- On key events, if popover is open, route Tab/Enter/Arrow to it before falling through to existing handlers.
- `send()` becomes: if `parse(draft)` returns non-null, call `dispatchSlash(parsed, ctx)` instead of `onSendPrompt(draft)`. Clear draft on successful dispatch (not on disabled error — preserve so user can fix).
- Non-slash input path is unchanged: `onSendPrompt(text)` → existing generate flow.

`dispatchSlash` logic lives in the composer and is tiny:

```js
const entry = registry.get(parsed.command);
if (!entry) { ctx.addMessage(errorMsg(`Unknown command: /${parsed.command}`)); return; }
if (!entry.enabled(ctx)) { ctx.addMessage(errorMsg(`Chat not enabled for mode ${ctx.activeMode}`)); return; }
entry.handler({ args: parsed.args, ctx });
```

## Data flow — `/chat hello world`

1. User types `/chat hello world`. Popover opens on the first `/`; filters match `/chat`; disappears once user types past the command token.
2. User hits Enter. Composer parses → `{command:"chat", args:"hello world"}`. Entry is enabled. Handler runs.
3. Handler appends two messages:
   - User bubble: `{role:user, kind:CHAT, text:"hello world"}` (prefix stripped).
   - Assistant bubble: `{id:assistantId, role:assistant, kind:CHAT, text:"", streaming:true, jobId:null}`.
4. Handler calls `chatJob.start({prompt:"hello world", onAck, onDelta, onComplete, onError})`.
5. `onAck({jobId})` → `updateMessage(assistantId, {jobId})`.
6. `onDelta(text)` → appends to a per-job buffer. A `requestAnimationFrame` coalesces buffered deltas into a single `updateMessage(assistantId, prev => ({...prev, text: prev.text + buffered}))` per frame. Prevents layout thrash under fast deltas.
7. `onComplete({text})` → `updateMessage(assistantId, {text, streaming:false})`. Final text from `outputs[0].text` is authoritative and replaces accumulated deltas to guard against drift.
8. `onError(errMsg)` → `updateMessage(assistantId, prev => ({...prev, kind:ERROR, prevText: prev.text, text: errMsg, streaming:false}))`.
9. While `streaming:true`, `MessageBubble` renders a per-bubble cancel affordance (X button). Click calls `handle.cancel()`, which sends `job:cancel` and sets `kind:ERROR, text:"Canceled"`.

## Message kind: `CHAT`

Added to `lcm-sr-ui/src/utils/constants.js` alongside IMAGE / TEXT / PENDING / ERROR / SYSTEM.

`MessageBubble.jsx` gains a branch:

- User chat bubble: styled like existing user text but with a small `chat` glyph to distinguish from prompt/image messages.
- Assistant chat bubble: streams text; shows a typing indicator while `streaming:true`; shows cancel button while `streaming:true && jobId`.
- Not selectable as image params (existing `selectedParams` memoization already gates on `kind:IMAGE`).

Persistence: the existing `useChatMessages` localStorage path serializes `CHAT` kind as plain text (no blob handling). On reload, `streaming:true` bubbles are fixed up to `streaming:false, kind:ERROR, text:"Interrupted"` during rehydrate to avoid stuck spinners.

## Error handling

- `job:error` → ERROR bubble, partial stream preserved on `prevText`.
- Disabled mode (typed anyway, popover bypassed) → inline ERROR, no WS send, draft preserved.
- Unknown command → inline ERROR `"Unknown command: /<name>"`.
- Empty args (`/chat` alone) → no dispatch; draft preserved; popover stays visible.
- WS disconnect mid-stream → existing `wsClient` reconnect logic applies; a disconnect callback flips any `streaming:true` chat bubbles to ERROR `"Connection lost"`.
- Mode switch mid-stream → current stream is jobId-bound and finishes; new turns use the new mode.

## Testing

Vitest, colocated per existing convention.

- `lcm-sr-ui/src/lib/slashCommands.test.js` — register/parse/list, unknown command, disabled resolution, arg parsing (whitespace, multiline, empty).
- `lcm-sr-ui/src/hooks/useChatJob.test.js` — mock `wsClient`; assert submit payload, ack→delta→complete dispatch, error path, cancel path, parallel jobs don't cross-talk.
- `lcm-sr-ui/src/components/chat/SlashHintPopover.test.jsx` — render, prefix filter, arrow+Tab+Enter+Esc, disabled entry not completable, tooltip content.
- `lcm-sr-ui/src/components/chat/MessageComposer.test.jsx` (extend) — `/chat foo` bypasses `onSendPrompt`, routes through registry; disabled mode surfaces inline error and preserves draft; Shift+Enter still sends non-slash input via existing path.
- `lcm-sr-ui/src/lib/slashCommands/chat.test.js` — handler adds user+assistant bubbles, `onDelta` coalescing, `onComplete` replaces text with authoritative `outputs[0].text`, `onError` preserves partial as `prevText`, cancel wiring.

## Acceptance criteria

- Typing `/chat hello` in the main prompt sends a chat job and renders the streaming reply inline.
- When the active mode has `chat_enabled=false`, `/chat` surfaces a clear ERROR bubble instead of hanging.
- Slash registry allows future commands to be added without touching `MessageComposer.jsx`.
- Streaming is smooth: deltas coalesced per animation frame; no layout thrash.
- `job:error` frames render an inline ERROR state with partial text preserved.
- `/` typed alone opens hint popover listing available commands with keyboard nav.

## Open questions

None at spec time. Config-panel integration (temperature/max_tokens surface) is out of scope and handled in STABL-eaontsix.
