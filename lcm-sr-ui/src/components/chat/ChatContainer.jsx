// src/components/chat/ChatContainer.jsx

import React from 'react';
import ScrollToBottom from 'react-scroll-to-bottom';
import { Card, CardContent } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ChatHeader } from './ChatHeader';
import { MessageComposer } from './MessageComposer';
import { MessageBubble } from './MessageBubble';

/**
 * Main chat container component.
 * Uses react-scroll-to-bottom for automatic sticky-bottom behavior.
 */
export function ChatContainer({
  messages,
  selectedMsgId,
  onToggleSelect,
  onCancelRequest,
  setMsgRef,
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
        {/* Scrollable messages with sticky-bottom behavior */}
        <ScrollToBottom
          className="flex-1 min-h-0"
          scrollViewClassName="p-4 md:p-6"
          followButtonClassName="scroll-to-bottom-button"
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
          </div>
        </ScrollToBottom>

        <Separator />

        {/* Message composer */}
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
