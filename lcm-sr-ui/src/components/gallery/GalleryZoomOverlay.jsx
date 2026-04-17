import React, { useEffect, useState, useRef } from 'react';
import { X } from 'lucide-react';

export function GalleryZoomOverlay({ item, resolveImageUrl, onClose }) {
  const [url, setUrl] = useState(null);
  const frameRef = useRef(null);

  useEffect(() => {
    let active = true;
    resolveImageUrl(item).then((resolved) => {
      if (active) setUrl(resolved);
    });
    return () => { active = false; };
  }, [item.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleBackdropMouseDown(e) {
    if (!frameRef.current) return;
    if (!frameRef.current.contains(e.target)) onClose?.();
  }

  return (
    <div
      data-testid="zoom-backdrop"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onMouseDown={handleBackdropMouseDown}
    >
      <div
        ref={frameRef}
        className="relative rounded-md shadow-2xl bg-background p-2"
      >
        <button
          type="button"
          aria-label="Close zoom"
          onClick={onClose}
          className="absolute -top-2 -right-2 rounded-full bg-background border p-1 hover:bg-muted"
        >
          <X className="h-4 w-4" />
        </button>
        {url ? (
          <img
            src={url}
            alt={item.params?.prompt ?? ''}
            style={{ maxWidth: '50vw', maxHeight: '50vh' }}
            className="object-contain block"
          />
        ) : (
          <div style={{ width: '25vw', height: '25vh' }} className="bg-muted rounded" />
        )}
      </div>
    </div>
  );
}
