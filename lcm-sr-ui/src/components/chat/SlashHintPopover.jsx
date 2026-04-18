// src/components/chat/SlashHintPopover.jsx — Slash-command hint popover
//
// Appears above the composer when the draft starts with "/" and the caret
// is within the command token. Supports keyboard nav (Up/Down/Tab/Enter/Esc).

import React, { useEffect, useRef } from 'react';

/**
 * @param {object}   props
 * @param {Array}    props.items        - From slashCommands.list(ctx), filtered by prefix
 * @param {number}   props.activeIndex  - Currently highlighted row index
 * @param {function} props.onSelect     - (name) => void — complete the command token
 * @param {function} props.onClose      - Close without selecting
 */
export function SlashHintPopover({ items, activeIndex, onSelect, onClose }) {
  const listRef = useRef(null);

  // Scroll active item into view
  useEffect(() => {
    const el = listRef.current?.children[activeIndex];
    el?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex]);

  if (!items || items.length === 0) return null;

  return (
    <div
      className="absolute bottom-full left-0 right-0 mb-1 z-50 rounded-xl border bg-background shadow-lg overflow-hidden"
      role="listbox"
      aria-label="Slash commands"
    >
      <ul ref={listRef} className="py-1 max-h-48 overflow-y-auto">
        {items.map((item, idx) => (
          <li
            key={item.name}
            role="option"
            aria-selected={idx === activeIndex}
            aria-disabled={!item.enabled}
            title={!item.enabled && item.disabledReason ? item.disabledReason : undefined}
            className={
              'flex items-center gap-3 px-3 py-2 text-sm cursor-default select-none ' +
              (idx === activeIndex ? 'bg-accent text-accent-foreground' : '') +
              (item.enabled ? ' hover:bg-accent/60' : ' opacity-40 cursor-not-allowed')
            }
            onMouseDown={(e) => {
              e.preventDefault(); // keep textarea focused
              if (item.enabled) onSelect(item.name);
            }}
          >
            <span className="font-mono font-medium text-primary">/{item.name}</span>
            <span className="text-muted-foreground truncate">{item.description}</span>
            {!item.enabled && item.disabledReason && (
              <span className="ml-auto text-xs text-destructive truncate shrink-0 max-w-[140px]">
                {item.disabledReason}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
