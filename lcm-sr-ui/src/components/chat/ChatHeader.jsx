// src/components/chat/ChatHeader.jsx

import React from 'react';
import { SurfaceHeader } from '@/components/ui/SurfaceHeader';
import { BADGE_LABELS, UI_MESSAGES } from '../../utils/constants';

export function ChatHeader({
  srLevel,
  frontendVersion,
  backendVersion,
}) {
  const apiVersionLabel = backendVersion?.trim() ? backendVersion : '...';

  const srChip = srLevel > 0
    ? { label: `SR ${srLevel}` }
    : { label: BADGE_LABELS.SR_OFF, variant: 'outline' };

  const chips = [
    srChip,
    { label: `UI ${frontendVersion}`, variant: 'outline' },
    { label: `API ${apiVersionLabel}`, variant: 'outline' },
  ];

  return (
    <SurfaceHeader
      title="[ N Ξ U R Ø   C Λ N V Λ S ]"
      chips={chips}
      summary={UI_MESSAGES.KEYBOARD_TIP}
    />
  );
}
