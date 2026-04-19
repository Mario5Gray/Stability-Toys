import React from 'react';
import { Button } from '@/components/ui/button';

function ActionButton({ action, variant, className = '' }) {
  return (
    <Button
      type="button"
      variant={variant}
      disabled={Boolean(action.disabled)}
      onClick={action.onClick}
      className={`flex flex-col items-center gap-0.5 h-auto py-2.5 px-3 ${className}`}
    >
      <div className="flex items-center gap-1.5">
        {action.icon && <span className="shrink-0">{action.icon}</span>}
        <span className="font-medium text-sm">{action.label}</span>
      </div>
      {action.subtext && (
        <span className="text-xs opacity-70 font-normal">{action.subtext}</span>
      )}
    </Button>
  );
}

// primary: { icon?, label, subtext?, onClick, disabled? }
// secondary: [{ icon?, label, subtext?, onClick, disabled? }]
export function PanelActionBar({ primary, secondary = [] }) {
  return (
    <div className="border-t bg-muted/30 px-4 py-3 space-y-2">
      {secondary.length > 0 && (
        <div className="flex gap-2">
          {secondary.map((action) => (
            <ActionButton key={action.label} action={action} variant="secondary" className="flex-1" />
          ))}
        </div>
      )}
      {primary && (
        <ActionButton action={primary} variant="default" className="w-full" />
      )}
    </div>
  );
}
