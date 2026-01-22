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
 * Parse size string (e.g., "512x512") into width and height.
 */
function parseSize(sizeStr) {
  if (!sizeStr) return { width: 512, height: 512 };
  const match = sizeStr.match(/^(\d+)x(\d+)$/i);
  if (!match) return { width: 512, height: 512 };
  return { width: parseInt(match[1], 10), height: parseInt(match[2], 10) };
}

/**
 * Placeholder component for pending image generations.
 * Maintains aspect ratio matching expected output size.
 */
function ImagePlaceholder({ size, onCancel }) {
  const { width, height } = parseSize(size);

  // Scale down to fit within max dimensions while preserving aspect ratio
  const maxWidth = 400;
  const maxHeight = 520;
  const scale = Math.min(maxWidth / width, maxHeight / height, 1);
  const displayWidth = Math.round(width * scale);
  const displayHeight = Math.round(height * scale);

  return (
    <div className="flex justify-start w-full">
      <div
        className="image-placeholder relative"
        style={{ width: displayWidth, height: displayHeight }}
      >
        <div className="flex flex-col items-center justify-center gap-2 text-muted-foreground">
          <Loader2 className="h-8 w-8 animate-spin opacity-50" />
          <span className="text-xs opacity-70">Generating…</span>
        </div>
        {onCancel ? (
          <button
            className="absolute top-2 right-2 p-1 rounded-full bg-background/80 opacity-70 hover:opacity-100 transition-opacity"
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
    </div>
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

  // First-time pending generation (no image yet) - show placeholder instead of bubble
  const isFirstPending = msg.kind === MESSAGE_KINDS.PENDING && !msg.imageUrl;
  if (isFirstPending) {
    return (
      <ImagePlaceholder
        size={msg.meta?.request?.size}
        onCancel={onCancel}
      />
    );
  }

  // Image-only messages (no text) get minimal styling
  const isImageOnly = (msg.kind === MESSAGE_KINDS.IMAGE || msg.isRegenerating) && !msg.text;

  const bubbleColor =
    isUser
      ? 'bg-primary text-primary-foreground'
      : msg.kind === MESSAGE_KINDS.ERROR
      ? 'bg-destructive text-destructive-foreground'
      : isImageOnly
      ? '' // No background for image-only
      : 'bg-muted';

  const selectedRing = isSelected ? 'ring-2 ring-primary ring-offset-2' : 'ring-0';
  const clickable = msg.kind === MESSAGE_KINDS.IMAGE || msg.isRegenerating
    ? 'cursor-pointer hover:ring-1 hover:ring-primary/30'
    : '';

  // Different wrapper styles for image-only vs text messages
  const wrapperClass = isImageOnly
    ? 'max-w-[92%] transition-all ' + selectedRing + ' ' + clickable
    : 'max-w-[92%] rounded-2xl px-4 py-3 shadow-sm transition-all ' + bubbleColor + ' ' + selectedRing + ' ' + clickable;

  return (
    <div
      className={'flex w-full ' + (isUser ? 'justify-end' : 'justify-start')}
    >
      <div
        className={wrapperClass}
        onClick={() => {
          if (msg.kind === MESSAGE_KINDS.IMAGE || msg.isRegenerating) {
            onSelect?.();
          }
        }}
        title={msg.kind === MESSAGE_KINDS.IMAGE || msg.isRegenerating ? 'Click to select and edit' : undefined}
      >
        {/* Text content (for text messages, errors, and image captions) */}
        {msg.text ? (
          <div className="whitespace-pre-wrap text-sm leading-relaxed">
            {msg.text}
          </div>
        ) : null}

        {/* Image */}
        {(msg.kind === MESSAGE_KINDS.IMAGE || msg.isRegenerating) && msg.imageUrl ? (
          <div className={msg.text ? 'mt-3' : ''}>
            <div className={msg.isRegenerating ? 'regenerating-border inline-block' : 'inline-block'}>
              <img
                src={msg.imageUrl}
                alt="generation"
                className={
                  'max-h-[520px] w-auto rounded-xl border bg-background shadow-sm' +
                  (msg.isRegenerating ? ' opacity-80' : '')
                }
                loading="lazy"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect?.();
                }}
              />
            </div>
            {msg.isRegenerating ? (
              <div className="mt-2 flex items-center gap-2 text-xs opacity-80">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Regenerating…</span>
              </div>
            ) : null}

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