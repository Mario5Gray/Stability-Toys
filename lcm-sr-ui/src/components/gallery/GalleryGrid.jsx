// src/components/gallery/GalleryGrid.jsx
import React, { useState, useEffect, useRef } from 'react';

const PAGE_SIZE = 20;

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
    ? 'ring-2 ring-primary'
    : isAnchor
      ? 'ring-2 ring-primary/40'
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
          className="absolute top-1 left-1 rounded-full bg-primary text-primary-foreground h-5 w-5 flex items-center justify-center text-xs"
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
  selectedIds,
  anchorId,
}) {
  const [page, setPage] = useState(0);

  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));

  useEffect(() => {
    setPage(0);
  }, [items.length]);

  const pageItems = items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
        No images in this gallery yet
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div role="grid" className="grid gap-2" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
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
          className="px-3 py-1 rounded border disabled:opacity-40 disabled:cursor-not-allowed hover:bg-muted"
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
