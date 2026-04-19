import React from 'react';
import { X } from 'lucide-react';
import { useOperationsStore } from '../../contexts/OperationsContext';

const TONE_CLASSES = {
  active:   'bg-indigo-50  border-indigo-200  dark:bg-indigo-950/30  dark:border-indigo-800',
  complete: 'bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-800',
  error:    'bg-red-50     border-red-200     dark:bg-red-950/30     dark:border-red-800',
  idle:     'bg-muted border-border',
};

function OperationItem({ op }) {
  const toneClass = TONE_CLASSES[op.tone] ?? TONE_CLASSES.idle;
  const isActive = op.tone === 'active';

  return (
    <div className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm ${toneClass}`}>
      {op.icon && (
        <span className={isActive ? 'animate-pulse shrink-0' : 'shrink-0'}>{op.icon}</span>
      )}
      <div className="flex flex-1 items-center gap-2 min-w-0">
        <span className="font-medium truncate">{op.text}</span>
        {op.detail && (
          <span className="text-xs text-muted-foreground truncate">{op.detail}</span>
        )}
        {op.progress && (
          <span className="text-xs text-muted-foreground shrink-0">
            {op.progress.current} / {op.progress.total}
          </span>
        )}
      </div>
      {op.cancellable && op.cancelFn && (
        <button
          type="button"
          onClick={op.cancelFn}
          className="shrink-0 p-0.5 rounded hover:bg-black/10 transition-colors"
          aria-label="Cancel"
        >
          <X className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}

export function PendingOperationsPane() {
  const { operations, order } = useOperationsStore();
  if (order.length === 0) return null;

  return (
    <div className="flex flex-col gap-1 px-3 py-1.5">
      {order.map((key) => {
        const op = operations.get(key);
        if (!op) return null;
        return <OperationItem key={key} op={op} />;
      })}
    </div>
  );
}
