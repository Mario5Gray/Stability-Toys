// src/components/chat/MessageBubble.jsx

import React from 'react';
import { X, Loader2, ChevronLeft, ChevronRight, Radio } from 'lucide-react';
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
        'border backdrop-blur-sm transition-all ' +
        (dark
          ? 'bg-black/30 text-white/90 border-white/10 shadow-sm ' +
            'shadow-sm hover:shadow-md active:shadow-sm active:translate-y-[0.5px]'
          : 'bg-background/70 text-foreground border-border/60 shadow-sm' +
            'transition-all duration-150'
          )
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
function ImagePlaceholder({ size, onCancel, queuePosition }) {
  const { width, height } = parseSize(size);

  // Scale down to fit within max dimensions while preserving aspect ratio
  const maxWidth = 400;
  const maxHeight = 520;
  const scale = Math.min(maxWidth / width, maxHeight / height, 1);
  const displayWidth = Math.round(width * scale);
  const displayHeight = Math.round(height * scale);

  return (
    <div className="flex justify-center w-full">
      <div
        className="image-placeholder relative outline outline-2 outline-red-500"
        style={{ width: displayWidth, height: displayHeight }}
      >
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
          <Loader2 className="h-8 w-8 animate-spin opacity-50" />
          <span className="text-xs opacity-70">
            {queuePosition > 0 ? `Queued (#${queuePosition})` : 'Generating…'}
          </span>
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
 * @param {boolean} [props.isDreamMessage] - Whether this is the active dream message
 * @param {boolean} [props.hasDreamHistory] - Whether this message has browsable history
 * @param {function} [props.onDreamSave] - Callback to save dream and continue (double-click)
 * @param {function} [props.onDreamHistoryPrev] - Go to previous in history
 * @param {function} [props.onDreamHistoryNext] - Go to next in history
 * @param {function} [props.onDreamHistoryLive] - Go to latest (live)
 */
export function MessageBubble({
  msg,
  isSelected,
  onSelect,
  onCancel,
  isDreamMessage,
  hasDreamHistory,
  onDreamSave,
  onDreamHistoryPrev,
  onDreamHistoryNext,
  onDreamHistoryLive,
}) {
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

  // Selection ring for non-image messages only (images use CSS glow via image-frame)
  const selectedRing = isSelected && !isImageOnly ? 'ring-2 ring-primary ring-offset-2' : '';
  const clickable = msg.kind === MESSAGE_KINDS.IMAGE || msg.isRegenerating
    ? 'cursor-pointer'
    : '';

  // Different wrapper styles for image-only vs text messages
  const wrapperClass = isImageOnly
    ? 'max-w-[100%] transition-all' + clickable
    : 'max-w-[92%] rounded-2xl px-4 py-3 shadow-sm transition-all ' + bubbleColor + ' ' + selectedRing + ' ' + clickable;

  return (
      <div
    className={`
      flex items-start gap-0 max-w-[100%]
      rounded-2xl items-center justify-center 
      px-2 py-0
      ${isSelected ? "bg-violet-200 dark:bg-ping/3" : ""}
      transition-colors
    `}
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
            {/* Image frame with buffer space for indicators */}
            <div
              className={
                'image-frame inline-block' +
                (isSelected ? ' selected' : '')
              }
            >
              <div
                className={
                  'inline-block relative rounded-xl' +
                  (msg.isRegenerating ? ' image-generating-border' : '') +
                  (msg.hasError && !msg.isRegenerating ? ' image-error-border' : '') +
                  (isSelected && !msg.hasError && !msg.isRegenerating ? ' image-selected-border' : '')
                }
              >
                <img
                  src={msg.imageUrl}
                  alt="generation"
                  className={
                    'max-h-[640px] w-auto rounded-xl bg-background' +
                    (msg.isRegenerating ? ' opacity-60' : '') +
                    (isDreamMessage ? ' cursor-pointer' : '')
                  }
                  loading="lazy"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect?.();
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    if (isDreamMessage && onDreamSave) {
                      onDreamSave(msg);
                    }
                  }}
                />
                {/* Regenerating overlay - floats above image, no layout shift */}
                {msg.isRegenerating && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="flex items-center gap-2 bg-black/60 text-white text-sm px-3 py-1.5 rounded-full backdrop-blur-sm">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Regenerating</span>
                    </div>
                  </div>
                )}
                {/* Error indicator tooltip */}
                {msg.hasError && msg.errorText && (
                  <div
                    className="absolute bottom-2 left-2 right-2 bg-red-900/90 text-white text-xs px-2 py-1 rounded backdrop-blur-sm truncate"
                    title={msg.errorText}
                  >
                    {msg.errorText}
                  </div>
                )}
                {/* Dream mode indicator */}
                {isDreamMessage && !msg.isRegenerating && (
                  <div className="absolute top-2 right-2 bg-purple-600/80 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                    dreaming
                  </div>
                )}
              </div>
            </div>

            {/* Dream history navigation */}
            {hasDreamHistory && (
              <div className="mt-2 flex items-center gap-2">
                <button
                  onClick={(e) => { e.stopPropagation(); onDreamHistoryPrev?.(); }}
                  disabled={msg.historyIndex === 0}
                  className="p-1 rounded hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
                  title="Previous"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <span className="text-xs text-muted-foreground min-w-[60px] text-center">
                  {(msg.historyIndex ?? 0) + 1} / {msg.imageHistory?.length ?? 0}
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); onDreamHistoryNext?.(); }}
                  disabled={msg.historyIndex === (msg.imageHistory?.length ?? 1) - 1}
                  className="p-1 rounded hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
                  title="Next"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
                {!isDreamMessage && msg.historyIndex !== (msg.imageHistory?.length ?? 1) - 1 && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onDreamHistoryLive?.(); }}
                    className="p-1 rounded hover:bg-muted text-purple-500"
                    title="Go to latest"
                  >
                    <Radio className="h-4 w-4" />
                  </button>
                )}
              </div>
            )}

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