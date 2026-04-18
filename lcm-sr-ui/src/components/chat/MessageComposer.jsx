// src/components/chat/MessageComposer.jsx
import React from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { MessageSquare, Image } from "lucide-react";
import { SlashHintPopover } from "./SlashHintPopover";
import { parse, list, get } from "../../lib/slashCommands";

// Ensure slash command handlers are registered
import "../../lib/slashCommands/chat.js";
import "../../lib/slashCommands/gen.js";

/**
 * Determine whether the caret is still within the command token
 * (i.e. before the first whitespace after the leading slash).
 */
function caretInCommandToken(value, selectionStart) {
  if (!value.startsWith('/')) return false;
  const spaceIdx = value.indexOf(' ');
  if (spaceIdx === -1) return selectionStart <= value.length;
  return selectionStart <= spaceIdx;
}

export function MessageComposer({
  onSendPrompt,
  onCancelAll,
  onKeyDown: externalKeyDown,
  onFocus,
  inflightCount,
  serverLabel,
  // New props for slash-command / chat routing
  slashCtx,      // dispatch context object
  inputMode,     // 'generate' | 'chat'
  onSetInputMode,
}) {
  const [draft, setDraft] = React.useState("");
  const [popoverOpen, setPopoverOpen] = React.useState(false);
  const [popoverIdx, setPopoverIdx] = React.useState(0);
  const textareaRef = React.useRef(null);

  // Build filtered popover items whenever draft changes
  const popoverItems = React.useMemo(() => {
    if (!draft.startsWith('/') || !slashCtx) return [];
    const parsed = parse(draft);
    const prefix = parsed ? parsed.command : draft.slice(1).split(' ')[0];
    return list(slashCtx).filter((item) =>
      item.name.startsWith(prefix.toLowerCase())
    );
  }, [draft, slashCtx]);

  // Auto-open/close popover based on draft content and caret position
  const handleChange = React.useCallback((e) => {
    const value = e.target.value;
    setDraft(value);
    if (value.startsWith('/') && slashCtx) {
      const inToken = caretInCommandToken(value, e.target.selectionStart);
      setPopoverOpen(inToken);
      setPopoverIdx(0);
    } else {
      setPopoverOpen(false);
    }
  }, [slashCtx]);

  const closePopover = React.useCallback(() => setPopoverOpen(false), []);

  /** Complete the command token in the draft. */
  const completeCommand = React.useCallback((name) => {
    setDraft(`/${name} `);
    setPopoverOpen(false);
    textareaRef.current?.focus();
  }, []);

  /**
   * Dispatch a plain chat turn (inputMode === 'chat', no slash prefix).
   */
  const dispatchPlainChat = React.useCallback((text) => {
    if (!slashCtx) return;
    // Reuse /chat handler with synthetic parsed args
    const chatEntry = get('chat');
    if (!chatEntry) return;
    chatEntry.handler({ args: text, ctx: slashCtx });
  }, [slashCtx]);

  const send = React.useCallback(() => {
    const text = draft.trim();
    if (!text) return;

    // Slash command path
    if (text.startsWith('/') && slashCtx) {
      const parsed = parse(text);
      if (!parsed) {
        // Just "/" alone or parse failed — keep popover open
        return;
      }
      if (!parsed.args) {
        // Command with no args — keep draft, let user type args
        return;
      }
      const entry = get(parsed.command);
      if (!entry) {
        slashCtx.addMessage(slashCtx.createErrorMessage(`Unknown command: /${parsed.command}`));
        setDraft('');
        return;
      }
      if (!entry.enabled(slashCtx)) {
        slashCtx.addMessage(
          slashCtx.createErrorMessage(
            entry.disabledReason ? entry.disabledReason(slashCtx) : `/${parsed.command} is unavailable`
          )
        );
        // Preserve draft so user can see the failed command
        return;
      }
      const ok = entry.handler({ args: parsed.args, ctx: slashCtx });
      if (ok !== false) setDraft('');
      return;
    }

    // Plain chat mode routing
    if (inputMode === 'chat' && slashCtx) {
      dispatchPlainChat(text);
      setDraft('');
      return;
    }

    // Default: image generation
    onSendPrompt?.(text);
    // Note: intentionally not clearing draft — existing behavior
  }, [draft, slashCtx, inputMode, dispatchPlainChat, onSendPrompt]);

  const handleKeyDown = React.useCallback((e) => {
    // Route arrow/tab/enter to popover while it's open and caret is in command token
    if (popoverOpen && popoverItems.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setPopoverIdx((i) => Math.min(i + 1, popoverItems.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setPopoverIdx((i) => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Tab' || (e.key === 'Enter' && caretInCommandToken(draft, textareaRef.current?.selectionStart ?? 0))) {
        e.preventDefault();
        const item = popoverItems[popoverIdx];
        if (item?.enabled) completeCommand(item.name);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        closePopover();
        return;
      }
    }

    // External handler (Ctrl/Cmd+Enter from App)
    externalKeyDown?.(e);

    // Shift+Enter submits
    if (e.shiftKey && e.key === 'Enter') {
      e.preventDefault();
      send();
    }
  }, [popoverOpen, popoverItems, popoverIdx, draft, completeCommand, closePopover, externalKeyDown, send]);

  const modeLabel = inputMode === 'chat' ? 'Chat mode' : 'Generate mode';
  const ModeIcon = inputMode === 'chat' ? MessageSquare : Image;

  return (
    <div className="p-2 md:p-2 option-panel-area">
      {/* Mode indicator + toggle */}
      {slashCtx && (
        <div className="flex items-center gap-2 px-1 mb-1">
          <button
            type="button"
            onClick={() => onSetInputMode?.(inputMode === 'chat' ? 'generate' : 'chat')}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors rounded px-1.5 py-0.5 hover:bg-muted"
            title={`Switch to ${inputMode === 'chat' ? 'generate' : 'chat'} mode`}
          >
            <ModeIcon className="h-3 w-3" />
            {modeLabel}
          </button>
        </div>
      )}

      <div className="relative flex items-center rounded-base bg-neutral-secondary-soft">
        <SlashHintPopover
          items={popoverOpen ? popoverItems : []}
          activeIndex={popoverIdx}
          onSelect={completeCommand}
          onClose={closePopover}
        />

        <Textarea
          ref={textareaRef}
          value={draft}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={onFocus}
          placeholder={
            inputMode === 'chat'
              ? 'Chat with the model… (Shift+Enter to send, /gen to generate)'
              : 'Describe what you want to generate… (Shift+Enter to send)'
          }
          className="mx-4 bg-neutral-primary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 placeholder:text-body"
        />

        <div className="flex flex-col mt-2 gap-2">
          <Button
            onClick={send}
            disabled={!draft.trim()}
            className="relative overflow-hidden"
          >
            Send
          </Button>

          {onCancelAll ? (
            <Button variant="secondary" onClick={onCancelAll} type="button">
              Cancel
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
