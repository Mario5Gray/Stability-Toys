// src/components/chat/ChatContainer.jsx

import React from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ChatHeader } from './ChatHeader';
import { MessageComposer } from './MessageComposer';
import { MessageBubble } from './MessageBubble';

/**
 * Main chat container component.
 * Displays messages, handles scrolling, and provides composition UI.
 */
export function ChatContainer({
  messages,
  selectedMsgId,
  onToggleSelect,
  onCancelRequest,
  setMsgRef,
  scroll,
  composer,
  inflightCount,
  isDreaming,
  srLevel,
  onCopyPrompt,
  copied,
  serverLabel,
}) {
  return (
    <Card className="overflow-hidden rounded-2xl shadow-sm h-full flex flex-col">
      <ChatHeader
        inflightCount={inflightCount}
        isDreaming={isDreaming}
        srLevel={srLevel}
        onCopyPrompt={onCopyPrompt}
        copied={copied}
      />

      <CardContent className="flex flex-1 flex-col p-0 min-h-0">
        {/* Scrollable messages */}
        <div
          ref={scroll.chatScrollRef}
          onScroll={scroll.onChatScroll}
          className="flex-1 overflow-y-auto p-4 md:p-6 min-h-0"
        >
          <div className="space-y-4">
            {messages.map((msg) => (
              <div key={msg.id} ref={setMsgRef(msg.id)}>
                <MessageBubble
                  msg={msg}
                  isSelected={msg.id === selectedMsgId}
                  onSelect={() => onToggleSelect(msg.id)}
                  onCancel={
                    msg.kind === 'pending'
                      ? () => onCancelRequest(msg.id)
                      : null
                  }
                />
              </div>
            ))}
            <div ref={scroll.chatBottomRef} />
          </div>
        </div>

        <Separator />

        {/* Message composer - THIS IS THE SEND BUTTON SECTION */}
        <MessageComposer
          prompt={composer.prompt}
          onPromptChange={composer.onPromptChange}
          onSend={composer.onSend}
          onCancelAll={composer.onCancelAll}
          onKeyDown={composer.onKeyDown}
          inflightCount={inflightCount}
          disabled={composer.disabled}
          currentParams={composer.currentParams}
          serverLabel={serverLabel}
        />
      </CardContent>
    </Card>
  );
}