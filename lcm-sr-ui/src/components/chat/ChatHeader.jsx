// src/components/chat/ChatHeader.jsx

import React from 'react';
import { CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Sparkles } from 'lucide-react';
import { BADGE_LABELS, UI_MESSAGES } from '../../utils/constants';

/**
 * Chat header component with status badges.
 * 
 * @param {object} props
 * @param {number} props.inflightCount - Number of in-flight requests
 * @param {boolean} props.isDreaming - Whether dream mode is active
 * @param {number} props.srLevel - Current SR level
 * @param {function} props.onCopyPrompt - Copy prompt callback
 * @param {boolean} props.copied - Whether prompt was copied
 */
export function ChatHeader({
  inflightCount,
  isDreaming,
  srLevel,
  onCopyPrompt,
  copied,
}) {
  return (
    <CardHeader className="border-b">
      <div className="flex items-center justify-between gap-3">
        <CardTitle className="text-xl">LCM + SR Chat</CardTitle>

        <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
          <Badge variant="secondary">{BADGE_LABELS.ENDPOINT}</Badge>
          <Badge variant="secondary">{BADGE_LABELS.FORMAT}</Badge>

          {/* Copy prompt badge */}
          <button
            type="button"
            onClick={onCopyPrompt}
            className="inline-flex"
            title={UI_MESSAGES.COPY_PROMPT_TIP}
          >
            <Badge
              variant="outline"
              className="bg-background cursor-pointer hover:bg-muted transition-colors"
            >
              {copied ? UI_MESSAGES.COPIED : UI_MESSAGES.COPY_PROMPT}
            </Badge>
          </button>

          {/* SR badge */}
          {srLevel > 0 ? (
            <Badge>SR {srLevel}</Badge>
          ) : (
            <Badge variant="outline">{BADGE_LABELS.SR_OFF}</Badge>
          )}

          {/* Dream mode badge */}
          {isDreaming && (
            <Badge className="gap-1 animate-pulse bg-gradient-to-r from-purple-600 to-pink-600">
              <Sparkles className="h-3 w-3" />
              Dreaming
            </Badge>
          )}

          {/* In-flight count */}
          {inflightCount > 0 && (
            <Badge variant="secondary">{inflightCount} running</Badge>
          )}
        </div>
      </div>

      <div className="text-sm text-muted-foreground">
        {UI_MESSAGES.KEYBOARD_TIP}
      </div>
    </CardHeader>
  );
}