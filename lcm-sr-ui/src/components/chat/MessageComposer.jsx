// src/components/chat/MessageComposer.jsx

import React from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Send, Square } from 'lucide-react';

/**
 * Message composer component - textarea + send/cancel buttons.
 */
export function MessageComposer({
  prompt,
  onPromptChange,
  onSend,
  onCancelAll,
  onKeyDown,
  inflightCount,
  disabled,
  currentParams,
  serverLabel,
}) {
  console.log('MessageComposer rendering', { prompt, disabled }); // DEBUG

  return (
    <div className="p-3 md:p-4 border-t">
      <div className="flex gap-2">
        <Textarea
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Describe what you want to generate…"
          className="min-h-[52px] max-h-[200px] resize-none rounded-2xl"
        />
        <div className="flex flex-col gap-2">
          <Button
            className="rounded-2xl"
            onClick={onSend}
            disabled={disabled}
          >
            <Send className="mr-2 h-4 w-4" />
            Send
          </Button>

          <Button
            variant="outline"
            className="rounded-2xl"
            onClick={onCancelAll}
            disabled={inflightCount === 0}
            title="Cancel all in-flight requests"
          >
            <Square className="mr-2 h-4 w-4" />
            Cancel
          </Button>
        </div>
      </div>

      {/* Current params summary */}
      {currentParams && (
        <div className="mt-2 text-xs text-muted-foreground">
          Current: size {currentParams.size} · steps {currentParams.steps} · CFG{' '}
          {Number(currentParams.cfg).toFixed(1)} · seed{' '}
          {currentParams.seedMode === 'random' ? 'random' : currentParams.seed}
          {currentParams.superresLevel > 0
            ? ` · SR ${currentParams.superresLevel}`
            : ''}{' '}
          · {serverLabel}
        </div>
      )}
    </div>
  );
}