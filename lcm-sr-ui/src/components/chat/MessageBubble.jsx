// src/components/chat/MessageBubble.jsx

import React from 'react';
import { X, Loader2 } from 'lucide-react';
import { MESSAGE_ROLES, MESSAGE_KINDS } from '../../utils/constants';

/**
 * Pill component for displaying metadata tags.
 */
function Pill({ label, dark = false }) {
  if (!label) return null;
  return (
    <span
      className={
        'inline-flex items-center rounded-full px-2 py-0.5 text-[11px] ' +
        (dark
          ? 'bg-black/20 text-white/90'
          : 'bg-background/60 text-foreground border')
      }
    >
      {label}
    </span>
  );
}

/**
 * Chat message bubble component.
 * Displays user/assistant messages with support for text, images, pending, and error states.
 * 
 * @param {object} props
 * @param {object} props.msg - Message object
 * @param {boolean} props.isSelected - Whether this message is selected
 * @param {function} props.onSelect - Selection callback
 * @param {function} [props.onCancel] - Cancel callback for pending messages
 */
export function MessageBubble({ msg, isSelected, onSelect, onCancel }) {
  const isUser = msg.role === MESSAGE_ROLES.USER;

  const bubbleColor =
    isUser
      ? 'bg-primary text-primary-foreground'
      : msg.kind === MESSAGE_KINDS.ERROR
      ? 'bg-destructive text-destructive-foreground'
      : 'bg-muted';

  const selectedRing = isSelected ? 'ring-2 ring-primary ring-offset-2' : 'ring-0';
  const clickable = msg.kind === MESSAGE_KINDS.IMAGE ? 'cursor-pointer hover:ring-1 hover:ring-primary/30' : '';

  return (
    <div
      className={'flex w-full ' + (isUser ? 'justify-end' : 'justify-start')}
    >
      <div
        className={
          'max-w-[92%] rounded-2xl px-4 py-3 shadow-sm transition-all ' +
          bubbleColor +
          ' ' +
          selectedRing +
          ' ' +
          clickable
        }
        onClick={() => {
          if (msg.kind === MESSAGE_KINDS.IMAGE) {
            onSelect?.();
          }
        }}
        title={msg.kind === MESSAGE_KINDS.IMAGE ? 'Click to select and edit' : undefined}
      >
        {/* Top row: text + optional cancel */}
        <div className="flex items-start gap-3">
          <div className="flex-1 whitespace-pre-wrap text-sm leading-relaxed">
            {msg.text}
          </div>

          {msg.kind === MESSAGE_KINDS.PENDING && onCancel ? (
            <button
              className="opacity-70 hover:opacity-100 transition-opacity"
              onClick={(e) => {
                e.stopPropagation();
                onCancel();
              }}
              title="Cancel this request"
              aria-label="Cancel"
              type="button"
            >
              <X className="h-4 w-4" />
            </button>
          ) : null}
        </div>

        {/* Pending footer */}
        {msg.kind === MESSAGE_KINDS.PENDING ? (
          <div className="mt-2 flex items-center gap-2 text-xs opacity-80">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>Working…</span>
            {msg.meta?.request?.apiBase ? (
              <span className="ml-auto opacity-70 truncate max-w-[200px]">
                {msg.meta.request.apiBase}
              </span>
            ) : null}
          </div>
        ) : null}

        {/* Image */}
        {msg.kind === MESSAGE_KINDS.IMAGE && msg.imageUrl ? (
          <div className="mt-3">
            <img
              src={msg.imageUrl}
              alt="generation"
              className="max-h-[520px] w-auto rounded-xl border bg-background shadow-sm"
              loading="lazy"
              onClick={(e) => {
                e.stopPropagation();
                onSelect?.();
              }}
            />

            {/* Metadata pills + download */}
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
              {msg.params?.seed !== undefined && (
                <Pill label={`seed ${msg.params.seed}`} />
              )}
              {msg.params?.size && <Pill label={`${msg.params.size}`} />}
              {Number.isFinite(msg.params?.steps) && (
                <Pill label={`${msg.params.steps} steps`} />
              )}
              {Number.isFinite(msg.params?.cfg) && (
                <Pill label={`cfg ${Number(msg.params.cfg).toFixed(1)}`} />
              )}
              {msg.params?.superresLevel ? (
                <Pill label={`SR ${msg.params.superresLevel}`} />
              ) : null}
              {msg.meta?.backend ? (
                <Pill label={msg.meta.backend} />
              ) : null}

              <a
                className="ml-auto underline hover:no-underline"
                href={msg.imageUrl}
                download={`lcm_${msg.params?.seed ?? 'image'}.png`}
                onClick={(e) => e.stopPropagation()}
              >
                Download
              </a>
            </div>
          </div>
        ) : null}

        {/* User meta pills (for text messages) */}
        {isUser && msg.meta && msg.kind === MESSAGE_KINDS.TEXT ? (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs opacity-90">
            {msg.meta.size ? <Pill label={`${msg.meta.size}`} dark /> : null}
            {Number.isFinite(msg.meta.steps) ? (
              <Pill label={`${msg.meta.steps} steps`} dark />
            ) : null}
            {Number.isFinite(msg.meta.cfg) ? (
              <Pill label={`cfg ${Number(msg.meta.cfg).toFixed(1)}`} dark />
            ) : null}
            <Pill
              label={
                msg.meta.seedMode === 'random'
                  ? 'seed random'
                  : `seed ${msg.meta.seed ?? '?'}`
              }
              dark
            />
            {msg.meta.superres && <Pill label="SR on" dark />}
          </div>
        ) : null}
      </div>
    </div>
  );
}