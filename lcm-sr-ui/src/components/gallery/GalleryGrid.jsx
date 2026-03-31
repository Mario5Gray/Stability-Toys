// src/components/gallery/GalleryGrid.jsx
import React, { useState, useEffect, useRef } from 'react';

const PAGE_SIZE = 20;

// 1x1 transparent PNG used as placeholder src so <img> is always in the DOM
const PLACEHOLDER_SRC =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

function GalleryThumbnail({ item, resolveImageUrl, onOpenViewer }) {
  const [url, setUrl] = useState(null);
  const urlRef = useRef(null);

  useEffect(() => {
    let active = true;
    resolveImageUrl(item).then((resolved) => {
      if (active) {
        urlRef.current = resolved;
        setUrl(resolved);
      }
    });
    return () => { active = false; };
  // resolveImageUrl is excluded: callers must ensure a stable reference (useCallback)
  // Re-running the effect on item.id change is sufficient for gallery item rendering
  }, [item.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleKeyDown(e) {
    if (e.key === ' ' && urlRef.current) {
      e.preventDefault();
      window.open(urlRef.current, '_blank');
    }
  }

  return (
    <div
      data-gallery-cell
      tabIndex={0}
      className="relative w-32 h-32 rounded-md overflow-hidden cursor-pointer bg-muted focus:outline-none focus:ring-2 focus:ring-primary"
      onKeyDown={handleKeyDown}
    >
      <img
        src={url ?? PLACEHOLDER_SRC}
        alt={item.params?.prompt ?? ''}
        className={`w-full h-full object-cover${url ? '' : ' opacity-0'}`}
        onClick={() => onOpenViewer(item)}
      />
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

export function GalleryGrid({ items, resolveImageUrl, onOpenViewer }) {
  const [page, setPage] = useState(0);

  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));

  // resets page when item count changes (e.g. parent re-fetches gallery)
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
      <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
        {pageItems.map((item) => (
          <GalleryThumbnail
            key={item.id}
            item={item}
            resolveImageUrl={resolveImageUrl}
            onOpenViewer={onOpenViewer}
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
