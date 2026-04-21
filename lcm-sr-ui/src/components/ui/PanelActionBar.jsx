import React from 'react';
import { Button } from '@/components/ui/button';

function ActionButton({ action, variant, className = '' }) {
  return (
    <Button
      type="button"
      variant={variant}
      disabled={Boolean(action.disabled)}
      onClick={action.onClick}
      title={action.subtext}
      className={`relative group flex items-center gap-1.5 h-9 px-3 ${className}`}
    >
      {action.icon && <span className="shrink-0">{action.icon}</span>}
      <span className="font-medium text-sm">{action.label}</span>
      {action.subtext && (
        <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity z-50">
          {action.subtext}
        </span>
      )}
    </Button>
  );
}

export function PanelActionBar({ primary, secondary = [] }) {
  return (
    <div className="border-t px-3 py-2 flex gap-2 items-center">
      {secondary.map((action) => (
        <ActionButton key={action.label} action={action} variant="secondary" />
      ))}
      {primary && (
        <ActionButton action={primary} variant="default" className="ml-auto" />
      )}
    </div>
  );
}
