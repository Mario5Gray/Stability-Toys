// src/components/chat/ChatContainer.jsx
import React from "react";
import ScrollToBottom from "react-scroll-to-bottom";
import { Card, CardContent } from "@/components/ui/card";
import { MessageComposer } from "./MessageComposer";
import { MessageBubble } from "./MessageBubble";

export function ChatContainer({
  messages,
  selectedMsgId,
  onToggleSelect,
  onCancelRequest,
  setMsgRef,
  composer, // { onSendPrompt, onCancelAll, onKeyDown, onFocus }
  inflightCount,
  isDreaming,
  dreamMessageId,
  onDreamSave,
  onDreamHistoryPrev,
  onDreamHistoryNext,
  onDreamHistoryLive,
  serverLabel,
}) {
  return (
    <Card className="option-panel-area overflow-hidden rounded-xl shadow-sm h-full flex flex-col">
      <CardContent className="flex flex-1 flex-col p-0 min-h-0">
        <ScrollToBottom
          className="flex-1 min-h-0"
          scrollViewClassName="p-0 md:p-0"
          followButtonClassName="scroll-to-bottom-button"
        >
          <div className="relative">
            <div className="sticky top-0 z-10 text-center py-1 bg-background/80 backdrop-blur-sm">
              <div className="text-xs text-white text-muted-foreground">[]</div>
            </div>

            <div className="space-y-4">
              {messages.map((msg) => (
                <div key={msg.id} ref={setMsgRef(msg.id)}>
                  <MessageBubble
                    msg={msg}
                    isSelected={msg.id === selectedMsgId}
                    onSelect={() => onToggleSelect(msg.id)}
                    onCancel={msg.kind === "pending" ? () => onCancelRequest(msg.id) : null}
                    isDreamMessage={isDreaming && msg.id === dreamMessageId}
                    hasDreamHistory={msg.imageHistory?.length > 1}
                    onDreamSave={onDreamSave}
                    onDreamHistoryPrev={() => onDreamHistoryPrev?.(msg)}
                    onDreamHistoryNext={() => onDreamHistoryNext?.(msg)}
                    onDreamHistoryLive={() => onDreamHistoryLive?.(msg)}
                  />
                </div>
              ))}
            </div>
          </div>
        </ScrollToBottom>

        <MessageComposer
          onSendPrompt={composer?.onSendPrompt}
          onCancelAll={composer?.onCancelAll}
          onKeyDown={composer?.onKeyDown}
          onFocus={composer?.onFocus}
          inflightCount={inflightCount}
          serverLabel={serverLabel}
        />
      </CardContent>
    </Card>
  );
}