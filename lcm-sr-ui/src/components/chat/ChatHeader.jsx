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
 * @param {string} props.frontendVersion - Frontend build version
 * @param {string | null | undefined} props.backendVersion - Backend runtime version
 * @param {function} props.onCopyPrompt - Copy prompt callback
 * @param {boolean} props.copied - Whether prompt was copied
 */
export function ChatHeader({
  inflightCount,
  isDreaming,
  srLevel,
  frontendVersion,
  backendVersion,
  onCopyPrompt,
  copied,
}) {
  const apiVersionLabel = backendVersion?.trim() ? backendVersion : '...';

  return (
    <CardHeader className="border-b">
      <div className="flex items-center justify-between gap-1">
        <CardTitle className="text-xl">LCM + SR Chat</CardTitle>

        <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
{/*          <Badge variant="secondary">{BADGE_LABELS.ENDPOINT}</Badge>
          <Badge variant="secondary">{BADGE_LABELS.FORMAT}</Badge>
*/}
          {/* SR badge */}
          {srLevel > 0 ? (
            <Badge>SR {srLevel}</Badge>
          ) : (
            <Badge variant="outline">{BADGE_LABELS.SR_OFF}</Badge>
          )}

          <Badge variant="outline">UI {frontendVersion}</Badge>
          <Badge variant="outline">API {apiVersionLabel}</Badge>


          {/* In-flight count */}
{/*          {inflightCount > 0 ? (
            <Badge className="gap-1 animate-pulse bg-gradient-to-r from-white-200 to-blue-200">InFlight</Badge>
          ) : (
            <Badge variant="secondary">Inflight</Badge>
          )}
*/}
          {/* Dream mode badge */}
          {isDreaming && (
            <Badge className="gap-1 animate-pulse bg-gradient-to-r from-purple-600 to-pink-600">
              <Sparkles className="h-1 w-1" />
              Dreaming
            </Badge>
          )}
        </div>
      </div>

      <div className="text-sm text-muted-foreground">
        {UI_MESSAGES.KEYBOARD_TIP}
      </div>
    </CardHeader>
  );
}
