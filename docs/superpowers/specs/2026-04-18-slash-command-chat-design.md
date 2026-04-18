# Slash-command `/chat` in main chat — Design

**FP issue:** STABL-jhpayntq
**Parent:** STABL-grarbnxp (OpenAI-compatible chat completions backend)
**Related:** STABL-eaontsix (LLM config panel)
**Date:** 2026-04-18

## Summary

Add a slash-command framework to the main chat composer, with `/chat` as the first registered command. Typing `/chat <message>` dispatches a WebSocket `jobType=chat` to the active mode's configured chat backend, streams the assistant reply inline into the existing chat window, and switches the composer into a persistent `chat` routing mode. In `chat` routing mode, subsequent plain-text sends also route to the LLM until the user exits back to `generate` mode through `/gen` or an explicit UI toggle. A hint popover surfaces available commands when the user types `/`. Extensible by design: future commands register into the same module without modifying the composer.

## Background

Backend chat plumbing is in place (STABL-grarbnxp):

- WebSocket `job:submit` accepts `jobType=chat` with `params: {prompt, stream, max_tokens?, temperature?, system_prompt?, model?}`.
- Streaming deltas arrive as `job:progress {delta}`; final aggregate as `job:complete {outputs:[{text}]}`.
- Backend resolves chat config from the global `chat.<mode>` mapping.
- `/api/modes` exposes `chat_enabled: bool` per mode (see `server/model_routes.py:162`).

The UI currently has no way to exercise chat. The existing main composer (`lcm-sr-ui/src/components/chat/MessageComposer.jsx`) sends every draft through `onSendPrompt`, which routes to image generation via `useImageGeneration`/`useComfyJobWs`. Chat turns and image-gen turns must coexist in the same chat window without colliding.

## Goals

- Dispatch `/chat <message>` through the existing WS chat job path.
- Switch the composer into `chat` routing mode after a successful `/chat` send.
- Route plain-text followup sends through chat while the composer is in `chat` routing mode.
- Provide an explicit path back to image generation mode through `/gen` and a visible mode toggle.
- Render streaming deltas inline in the chat window as a new `CHAT` message kind.
- Provide an extensible slash-command registry so new commands add in one file.
- Hint popover listing available commands when `/` is typed, with keyboard nav.
- Gate `/chat` on `chat_enabled` for the active mode; surface a clear error when disabled.
- Handle `job:error` and cancellation gracefully.

## Non-goals

- Multi-turn conversation memory across turns. `chat` routing mode is stateless for now: each send includes only the current prompt.
- Tool-use / function-calling.
- Persistence of chat transcripts beyond existing `localStorage` message store.
- Persistence of composer routing mode across reload.
- Per-turn params UI (`temperature`, `max_tokens`). Backend applies config from `chat.<mode>`; STABL-eaontsix will add a config panel.
- Backend chat session objects or transcript-to-request assembly.

## Architecture

New modules and edits:

| File | Role |
| ---- | ---- |
| `lcm-sr-ui/src/App.jsx` (edit) | Owns composer `inputMode` state (`generate` or `chat`) and passes routing context into chat UI. |
| `lcm-sr-ui/src/components/chat/ChatContainer.jsx` (edit) | Threads composer mode props and per-message cancel callbacks into child components. |
| `lcm-sr-ui/src/lib/slashCommands.js` | Pure-JS registry. `register`, `parse`, `list`. |
| `lcm-sr-ui/src/lib/slashCommands/chat.js` | Registers `/chat`. Handler sends one chat turn and requests `inputMode='chat'` on success. |
| `lcm-sr-ui/src/lib/slashCommands/gen.js` | Registers `/gen` (and alias `/generate`). Handler sends one image-generation prompt and requests `inputMode='generate'` on success. |
| `lcm-sr-ui/src/hooks/useChatJob.js` | WS correlation hook for `jobType=chat`. |
| `lcm-sr-ui/src/components/chat/SlashHintPopover.jsx` | Hint popover + keyboard nav. |
| `lcm-sr-ui/src/components/chat/MessageComposer.jsx` (edit) | Detect slash prefix, render popover, dispatch via registry, and route plain sends according to `inputMode`. |
| `lcm-sr-ui/src/components/chat/MessageBubble.jsx` (edit) | Adds `CHAT` bubble rendering and per-bubble cancel UI for streaming assistant chat responses. |
| `lcm-sr-ui/src/hooks/useChatMessages.js` (edit) | Persists `CHAT` messages and fixes up interrupted streaming chat bubbles during rehydrate. |

### Composer state model

`App.jsx` owns a small explicit state machine:

```js
inputMode: 'generate' | 'chat'
```

Rules:

- Initial state is `generate`.
- `/chat <message>` sends one chat turn, then sets `inputMode = 'chat'` if dispatch succeeds.
- Plain text while `inputMode === 'chat'` routes to chat (stateless one-shot per send).
- `/gen <message>` sends one generation turn, then sets `inputMode = 'generate'` if dispatch succeeds.
- Clicking the mode toggle switches routing mode without sending.
- Reload resets to `generate`.

### `slashCommands.js` — registry

Pure JavaScript. No React.

```js
// Shape
register(name, {
  description: string,                   // shown in popover
  enabled: (ctx) => bool,                // optional; defaults to () => true
  disabledReason: (ctx) => string,       // optional; used for tooltip when enabled() is false
  handler: ({ args, ctx }) => Promise<boolean> | boolean,
});

parse(input) -> { command, args, raw } | null;
// Parses only when input starts with "/". Splits at first whitespace.

list(ctx) -> Array<{ name, description, enabled, disabledReason }>;
// Resolves `enabled(ctx)` and, when false, `disabledReason(ctx)` for each command.
```

For `/chat`: `enabled(ctx)` returns `ctx.chatEnabled`; `disabledReason(ctx)` returns the string `Chat not enabled for mode <activeMode>`.

For `/gen`: `enabled(ctx)` always returns `true`.

`ctx` is supplied by the composer at dispatch time:

```js
{
  addMessage, updateMessage,     // from useChatMessages
  createErrorMessage,            // from useChatMessages
  activeMode,                    // string — current mode name
  chatEnabled,                   // bool — from /api/modes for activeMode
  inputMode, setInputMode,       // from App
  chatJob,                       // useChatJob instance
  runGenerate,                   // existing image-generation entry point
  wsConnected,                   // preflight to avoid dropped sends
}
```

### `slashCommands/chat.js` — `/chat` handler

Registers once at module load.

Handler responsibilities:

- Reject empty args.
- Reject when `ctx.chatEnabled === false`.
- Reject when `ctx.wsConnected === false` to avoid creating a stuck streaming bubble after a dropped send.
- Append user + assistant `CHAT` bubbles.
- Start the chat job.
- Flip to `inputMode='chat'` only after submit succeeds.

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
- If `wsClient.connected` is false, fail immediately before any message mutation.
- `send({type:"job:submit", id:corrId, jobType:"chat", params:{prompt, stream:true}})`.
- `cancel()` sends `{type:"job:cancel", jobId}` and clears subs.
- Subscribe to socket state changes and fail active chat handles with `"Connection lost"` if the socket disconnects mid-stream.

### `SlashHintPopover.jsx`

Anchored above the composer. Visible when draft starts with `/` and caret is within the command token. Renders list from `slashCommands.list(ctx)` filtered by prefix.

- Arrow Up/Down: change highlighted entry.
- Tab or Enter: complete the command token (insert `<name> ` into draft). If entry is disabled, no-op.
- Enter is intercepted only while the caret is still inside the slash-command token. Outside that token, normal submit behavior applies.
- Disabled entries render greyed with a tooltip showing the reason (e.g., `"chat not enabled for mode sd15"`).

### `MessageComposer.jsx` — edits

- Accept new props: `slashCtx` — the dispatch context object above.
- Accept `inputMode` and render a visible mode indicator/toggle (`Chat mode` / `Generate mode`).
- On `onChange`, detect `/` at start of draft. Toggle popover open/closed accordingly.
- Preserve existing send keybindings:
  - `Shift+Enter` in the textarea still submits.
  - `Ctrl/Cmd+Enter` in the main chat window still submits.
- On key events, if popover is open and caret is in the command token, route Tab/Enter/Arrow to it before falling through to existing handlers.
- `send()` becomes: if `parse(draft)` returns non-null, call `dispatchSlash(parsed, ctx)` instead of `onSendPrompt(draft)`. Clear draft on successful dispatch (not on disabled error — preserve so user can fix).
- Non-slash input path depends on `inputMode`:
  - `generate` → existing `onSendPrompt(text)` image flow.
  - `chat` → `dispatchPlainChat(text, ctx)`.

`dispatchSlash` logic lives in the composer and is tiny:

```js
const entry = registry.get(parsed.command);
if (!entry) { ctx.addMessage(errorMsg(`Unknown command: /${parsed.command}`)); return; }
if (!entry.enabled(ctx)) { ctx.addMessage(errorMsg(`Chat not enabled for mode ${ctx.activeMode}`)); return; }
entry.handler({ args: parsed.args, ctx });
```

Plain-send routing:

```js
if (ctx.inputMode === 'chat') {
  dispatchPlainChat(text, ctx);
} else {
  onSendPrompt(text);
}
```

## Data flow — `/chat hello world`

1. User types `/chat hello world`. Popover opens on the first `/`; filters match `/chat`; disappears once user types past the command token.
2. User hits Enter. Composer parses → `{command:"chat", args:"hello world"}`. Entry is enabled. Handler runs.
3. Handler verifies `ctx.chatEnabled` and `ctx.wsConnected`. If either preflight fails, it appends an inline ERROR bubble, preserves the draft, and does not switch modes.
4. Handler appends two messages:
   - User bubble: `{role:user, kind:CHAT, text:"hello world"}` (prefix stripped).
   - Assistant bubble: `{id:assistantId, role:assistant, kind:CHAT, text:"", streaming:true, jobId:null}`.
5. Handler calls `chatJob.start({prompt:"hello world", onAck, onDelta, onComplete, onError})`.
6. If `start()` succeeds, handler sets `inputMode='chat'`.
7. `onAck({jobId})` → `updateMessage(assistantId, {jobId})`.
8. `onDelta(text)` → appends to a per-job buffer. A `requestAnimationFrame` coalesces buffered deltas into a single `updateMessage(assistantId, prev => ({...prev, text: prev.text + buffered}))` per frame. Prevents layout thrash under fast deltas.
9. `onComplete({text})` → `updateMessage(assistantId, {text, streaming:false})`. Final text from `outputs[0].text` is authoritative and replaces accumulated deltas to guard against drift.
10. `onError(errMsg)` → `updateMessage(assistantId, prev => ({...prev, kind:ERROR, prevText: prev.text, text: errMsg, streaming:false}))`.
11. While `streaming:true`, `MessageBubble` renders a per-bubble cancel affordance (X button). Click calls `handle.cancel()`.

## Data flow — followup plain message in chat mode

1. Composer is already in `inputMode='chat'`.
2. User types `tell me more about the scheduler profile`.
3. User submits via existing send keys or Send button.
4. Composer bypasses image generation and calls the same plain-chat dispatcher used by `/chat`.
5. The turn is still stateless: only the current prompt is sent to the backend.

## Data flow — `/gen cinematic fox`

1. User is in `inputMode='chat'` and types `/gen cinematic fox`.
2. Composer dispatches the slash command through the registry.
3. Handler calls the existing image-generation path with the stripped prompt.
4. On successful dispatch, handler sets `inputMode='generate'`.

## Message kind: `CHAT`

Added to `lcm-sr-ui/src/utils/constants.js` alongside IMAGE / TEXT / PENDING / ERROR / SYSTEM.

`MessageBubble.jsx` gains a branch:

- User chat bubble: styled like existing user text but with a small `chat` glyph to distinguish from prompt/image messages.
- Assistant chat bubble: streams text; shows a typing indicator while `streaming:true`; shows cancel button while `streaming:true && jobId`.
- Not selectable as image params (existing `selectedParams` memoization already gates on `kind:IMAGE`).

Persistence: the existing `useChatMessages` localStorage path serializes `CHAT` kind as plain text (no blob handling). On reload, `streaming:true` bubbles are fixed up to `streaming:false, kind:ERROR, text:"Interrupted"` during rehydrate to avoid stuck spinners. Composer `inputMode` is not persisted; reload always returns to `generate`.

## Error handling

- `job:error` → ERROR bubble, partial stream preserved on `prevText`.
- Disabled mode (typed anyway, popover bypassed) → inline ERROR, no WS send, draft preserved.
- WS disconnected before submit → inline ERROR, no bubble pair created, draft preserved, mode unchanged.
- Unknown command → inline ERROR `"Unknown command: /<name>"`.
- Empty args (`/chat` alone) → no dispatch; draft preserved; popover stays visible.
- WS disconnect mid-stream → disconnect callback flips any `streaming:true` chat bubbles to ERROR `"Connection lost"`.
- Failed `/chat` start → remain in the previous routing mode.
- Chat turn fails after routing mode has already been entered → current bubble errors, but composer remains in `chat` mode.
- Mode switch mid-stream → current stream is jobId-bound and finishes; new turns use the current composer routing mode.
- Active model mode changes while composer is in `chat` mode → next send re-checks `chat_enabled`; no implicit composer mode reset.

## Testing

Vitest, colocated per existing convention.

- `lcm-sr-ui/src/lib/slashCommands.test.js` — register/parse/list, unknown command, disabled resolution, arg parsing (whitespace, multiline, empty).
- `lcm-sr-ui/src/hooks/useChatJob.test.js` — mock `wsClient`; assert submit payload, ack→delta→complete dispatch, error path, cancel path, parallel jobs don't cross-talk.
- `lcm-sr-ui/src/components/chat/SlashHintPopover.test.jsx` — render, prefix filter, arrow+Tab+Enter+Esc, disabled entry not completable, tooltip content.
- `lcm-sr-ui/src/components/chat/MessageComposer.test.jsx` (new or extend) — `/chat foo` bypasses `onSendPrompt`, successful dispatch flips to `chat` mode, plain send in `chat` mode bypasses generation, `/gen foo` flips back to `generate`, disabled mode preserves draft, and existing `Shift+Enter` / `Ctrl/Cmd+Enter` submit behavior remains intact.
- `lcm-sr-ui/src/lib/slashCommands/chat.test.js` — handler adds user+assistant bubbles, `onDelta` coalescing, `onComplete` replaces text with authoritative `outputs[0].text`, `onError` preserves partial as `prevText`, mode flips only after successful start, disconnected preflight does not flip mode.
- `lcm-sr-ui/src/lib/slashCommands/gen.test.js` — handler routes one prompt through image generation and flips mode back to `generate`.
- `lcm-sr-ui/src/hooks/useChatMessages.test.js` — persisted `CHAT` messages survive reload; interrupted streaming chat bubbles rehydrate as ERROR; composer mode is not restored from storage.

## Acceptance criteria

- Typing `/chat hello` in the main prompt sends a chat job, renders the streaming reply inline, and leaves the composer in `chat` mode.
- After `/chat hello`, the next plain message routes to chat without requiring another slash command.
- `/gen hello` or the visible mode toggle returns the composer to `generate` mode.
- When the active mode has `chat_enabled=false`, `/chat` surfaces a clear ERROR bubble instead of hanging.
- When the socket is disconnected before submit, `/chat` surfaces a clear ERROR bubble, preserves the draft, and does not switch modes.
- Slash registry allows future commands to be added without touching `MessageComposer.jsx`.
- Streaming is smooth: deltas coalesced per animation frame; no layout thrash.
- `job:error` frames render an inline ERROR state with partial text preserved.
- `/` typed alone opens hint popover listing available commands with keyboard nav.
- Reload resets the composer to `generate` mode even if the previous session was in `chat` mode.

## Open questions

None at spec time. Config-panel integration (temperature/max_tokens surface) is out of scope and handled in STABL-eaontsix. True multi-turn history remains a follow-up feature and is intentionally excluded from this design.
