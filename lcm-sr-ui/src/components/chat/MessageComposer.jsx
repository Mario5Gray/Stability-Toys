// src/components/chat/MessageComposer.jsx
import React from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

export function MessageComposer({
  onSendPrompt,
  onCancelAll,
  onKeyDown,
  onFocus,
}) {
  const [draft, setDraft] = React.useState("");

  const send = React.useCallback(() => {
    const text = draft.trim();
    if (!text) return;
    onSendPrompt?.(text);
    // optional:
    // setDraft("");
  }, [draft, onSendPrompt]);

  return (
    <div className="p-2 md:p-2 option-panel-area">
      <div className="flex items-center rounded-base bg-neutral-secondary-soft">
        <Textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            onKeyDown?.(e);
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
              e.preventDefault();
              send();
            }
          }}
          onFocus={onFocus}
          placeholder="Describe what you want to generate…"
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