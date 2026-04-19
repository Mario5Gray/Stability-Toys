import React from 'react';
import { CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// chips: [{ label: string, variant?: string }]
export function SurfaceHeader({ title, chips = [], summary }) {
  return (
    <CardHeader className="border-b">
      <div className="flex items-center justify-between gap-1 flex-wrap">
        <CardTitle className="text-xl">{title}</CardTitle>
        {chips.length > 0 && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
            {chips.map((chip) => (
              <Badge key={chip.label} variant={chip.variant ?? 'secondary'}>
                {chip.label}
              </Badge>
            ))}
          </div>
        )}
      </div>
      {summary && <div className="text-sm text-muted-foreground">{summary}</div>}
    </CardHeader>
  );
}
