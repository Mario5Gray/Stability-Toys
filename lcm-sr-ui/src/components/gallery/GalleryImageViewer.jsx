// src/components/gallery/GalleryImageViewer.jsx
import React, { useState, useEffect, useRef } from 'react';
import { ChevronLeft } from 'lucide-react';

export function GalleryImageViewer({ item, resolveImageUrl, onBack, onWindowOpen }) {
  const [url, setUrl] = useState(null);
  const urlRef = useRef(null);
  const containerRef = useRef(null);
  const [metaVisible, setMetaVisible] = useState(false);

  useEffect(() => {
    let active = true;
    resolveImageUrl(item).then((resolved) => {
      if (active) {
        urlRef.current = resolved;
        setUrl(resolved);
      }
    }).catch(() => {
      if (active) setUrl(null);
    });
    return () => { active = false; };
  }, [item.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === ' ' && urlRef.current) {
        e.preventDefault();
        const win = window.open(urlRef.current, '_blank');
        onWindowOpen?.(win);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onWindowOpen]);

  function handleMouseMove(e) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const threshold = rect.top + rect.height * 0.8;
    setMetaVisible(e.clientY >= threshold);
  }

  const { prompt, seed, size, steps, cfg, backend } = item.params ?? {};

  return (
    <div className="relative flex flex-col items-center justify-center h-full w-full">
      <button
        type="button"
        aria-label="Back"
        onClick={onBack}
        className="absolute top-2 left-2 z-10 p-1 rounded-full bg-background/80 hover:bg-background transition-colors"
      >
        <ChevronLeft className="h-5 w-5" />
      </button>

      <div
        ref={containerRef}
        data-testid="viewer-container"
        className="relative max-w-full max-h-full"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setMetaVisible(false)}
      >
        {url ? (
          <img
            src={url}
            alt={prompt ?? ''}
            className="max-w-full max-h-[80vh] object-contain rounded"
          />
        ) : (
          <div className="w-64 h-64 bg-muted rounded flex items-center justify-center text-muted-foreground text-sm">
            Loading…
          </div>
        )}

        <div
          data-testid="metadata-bar"
          className={
            'absolute bottom-0 left-0 right-0 bg-black/60 text-white text-xs p-2 space-y-0.5 transition-opacity duration-150 ' +
            (metaVisible ? 'opacity-100' : 'opacity-0 pointer-events-none')
          }
        >
          {prompt && <div><span className="opacity-60">prompt </span>{prompt}</div>}
          {seed !== undefined && <div><span className="opacity-60">seed </span>{seed}</div>}
          {size && <div><span className="opacity-60">size </span>{size}</div>}
          {steps !== undefined && <div><span className="opacity-60">steps </span>{steps}</div>}
          {cfg !== undefined && <div><span className="opacity-60">cfg </span>{cfg}</div>}
          {backend && <div><span className="opacity-60">backend </span>{backend}</div>}
          <div><span className="opacity-60">added </span>{new Date(item.addedAt).toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
}
