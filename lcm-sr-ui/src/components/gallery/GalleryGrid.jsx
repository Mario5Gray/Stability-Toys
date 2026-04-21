// src/components/gallery/GalleryGrid.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';

const PAGE_SIZE = 20;
const COLS = 5;

const PLACEHOLDER_SRC =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

function GalleryThumbnail({
  item,
  resolveImageUrl,
  onOpenViewer,
  onToggle,
  onRange,
  onZoom,
  selected,
  isAnchor,
}) {
  const [url, setUrl] = useState(null);
  const urlRef = useRef(null);
  const clickTimerRef = useRef(null);

  useEffect(() => {
    let active = true;
    resolveImageUrl(item).then((resolved) => {
      if (active) {
        urlRef.current = resolved;
        setUrl(resolved);
      }
    });
    return () => { active = false; };
  }, [item.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleClick(e) {
    if (clickTimerRef.current) clearTimeout(clickTimerRef.current);
    const shift = e.shiftKey;
    const mod = e.metaKey || e.ctrlKey;
    clickTimerRef.current = setTimeout(() => {
      if (shift) onRange?.(item.id);
      else onToggle?.(item.id, { shift, mod });
    }, 180);
  }

  function handleDoubleClick() {
    if (clickTimerRef.current) clearTimeout(clickTimerRef.current);
    clickTimerRef.current = null;
    onZoom?.(item);
  }

  function handleKeyDown(e) {
    if (e.key === ' ' && urlRef.current) {
      e.preventDefault();
      window.open(urlRef.current, '_blank');
    }
  }

  const ringClass = selected
    ? 'ring-2 ring-indigo-300'
    : isAnchor
      ? 'ring-2 ring-indigo-100'
      : '';

  return (
    <div
      role="gridcell"
      aria-selected={selected ? 'true' : 'false'}
      data-gallery-cell
      tabIndex={0}
      className={`relative w-32 h-32 rounded-md overflow-hidden cursor-pointer bg-muted focus:outline-none focus:ring-2 focus:ring-primary transition-transform duration-150 motion-reduce:transition-none hover:scale-[1.08] motion-reduce:hover:scale-100 ${ringClass}`}
      onKeyDown={handleKeyDown}
    >
      <img
        src={url ?? PLACEHOLDER_SRC}
        alt={item.params?.prompt ?? ''}
        className={`w-full h-full object-cover${url ? '' : ' opacity-0'}`}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
      />
      {selected && (
        <div
          aria-hidden="true"
          className="absolute top-1 left-1 rounded-full bg-indigo-300 text-indigo-900 h-5 w-5 flex items-center justify-center text-xs font-bold"
        >
          ✓
        </div>
      )}
      {!url && (
        <div
          aria-hidden="true"
          className="absolute inset-0 w-full h-full bg-muted flex items-center justify-center text-xs text-muted-foreground pointer-events-none"
        >
          …
        </div>
      )}
    </div>
  );
}

export function GalleryGrid({
  items,
  resolveImageUrl,
  onOpenViewer,
  onToggle,
  onRange,
  onZoom,
  onDeleteAction,
  onSelectAll,
  onDeselectAll,
  selectedIds,
  anchorId,
  keymap,
}) {
  const [page, setPage] = useState(0);
  const gridRef = useRef(null);

  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));

  useEffect(() => {
    setPage(0);
  }, [items.length]);

  const pageItems = items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const focusCellIndex = useCallback((idx) => {
    const cells = gridRef.current?.querySelectorAll('[data-gallery-cell]');
    if (!cells) return;
    const clamped = Math.max(0, Math.min(cells.length - 1, idx));
    cells[clamped]?.focus();
  }, []);

  function handleGridKeyDown(e) {
    if (!keymap) return;
    const cells = Array.from(gridRef.current?.querySelectorAll('[data-gallery-cell]') ?? []);
    const idx = cells.indexOf(document.activeElement);
    if (idx < 0) return;
    if (keymap.matches('right', e) || keymap.matches('next', e)) { e.preventDefault(); focusCellIndex(idx + 1); return; }
    if (keymap.matches('left', e)  || keymap.matches('prev', e)) { e.preventDefault(); focusCellIndex(idx - 1); return; }
    if (keymap.matches('down', e)) { e.preventDefault(); focusCellIndex(idx + COLS); return; }
    if (keymap.matches('up', e))   { e.preventDefault(); focusCellIndex(idx - COLS); return; }
    if (keymap.matches('select_all', e)) { e.preventDefault(); onSelectAll?.(); return; }
    if (keymap.matches('deselect_all', e)) { e.preventDefault(); onDeselectAll?.(); return; }
    if (keymap.matches('delete', e) || keymap.matches('delete_alt', e)) {
      e.preventDefault();
      const selected = selectedIds && selectedIds.size > 0 ? [...selectedIds] : [pageItems[idx]?.id].filter(Boolean);
      if (selected.length > 0) onDeleteAction?.(selected);
      return;
    }
    if (keymap.matches('zoom', e)) { e.preventDefault(); if (pageItems[idx]) onZoom?.(pageItems[idx]); return; }
  }

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
        No images in this gallery yet
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div ref={gridRef} role="grid" tabIndex={-1} onKeyDown={handleGridKeyDown} className="grid gap-2 overflow-visible" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)` }}>
        {pageItems.map((item) => (
          <GalleryThumbnail
            key={item.id}
            item={item}
            resolveImageUrl={resolveImageUrl}
            onOpenViewer={onOpenViewer}
            onToggle={onToggle}
            onRange={onRange}
            onZoom={onZoom}
            selected={selectedIds?.has(item.id) ?? false}
            isAnchor={anchorId === item.id}
          />
        ))}
      </div>

      <div className="flex items-center justify-center gap-4 text-sm">
        <button
          type="button"
          className="px-3 py-5 rounded border disabled:opacity-40 disabled:cursor-not-allowed hover:bg-muted"
          disabled={page === 0}
          onClick={() => setPage((p) => p - 1)}
          aria-label="Prev"
        >
          Prev
        </button>
        <span>Page {page + 1} of {totalPages}</span>
        <button
          type="button"
          className="px-3 py-1 rounded border disabled:opacity-40 disabled:cursor-not-allowed hover:bg-muted"
          disabled={page === totalPages - 1}
          onClick={() => setPage((p) => p + 1)}
          aria-label="Next"
        >
          Next
        </button>
      </div>
    </div>
  );
}
