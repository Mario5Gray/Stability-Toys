// src/components/gallery/GalleryLightbox.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X } from 'lucide-react';
import { createCache } from '../../utils/cache';
import { GalleryGrid } from './GalleryGrid';
import { GalleryImageViewer } from './GalleryImageViewer';

export function GalleryLightbox({ galleryId, galleryName, getGalleryImages, onClose }) {
  const [items, setItems] = useState([]);
  const [viewerItem, setViewerItem] = useState(null);
  const [opacity, setOpacity] = useState(0.95);

  const cacheRef = useRef(null);
  const blobUrlsRef = useRef(new Map()); // cacheKey -> blobUrl
  const childWindowsRef = useRef([]);

  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  // Lazy-init the lcm-image-cache handle
  function getCache() {
    if (!cacheRef.current) cacheRef.current = createCache();
    return cacheRef.current;
  }

  // Fetch items on mount
  useEffect(() => {
    getGalleryImages(galleryId).then(setItems);
  }, [galleryId, getGalleryImages]);

  // Revoke blob URLs on unmount
  useEffect(() => {
    return () => {
      for (const url of blobUrlsRef.current.values()) {
        try { URL.revokeObjectURL(url); } catch {}
      }
    };
  }, []);

  // ESC key handler
  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') closeAll();
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // closeAll is excluded: onClose is accessed via onCloseRef (always current) — re-adding the listener on every render is unnecessary

  function closeAll() {
    for (const win of childWindowsRef.current) {
      try { if (win && !win.closed) win.close(); } catch {}
    }
    childWindowsRef.current = [];
    onCloseRef.current();
  }

  function handleWindowOpen(win) {
    if (win) childWindowsRef.current.push(win);
  }

  const resolveImageUrl = useCallback(async (item) => {
    const cache = getCache();
    if (item.cacheKey) {
      if (blobUrlsRef.current.has(item.cacheKey)) {
        return blobUrlsRef.current.get(item.cacheKey);
      }
      try {
        const entry = await cache.get(item.cacheKey);
        if (entry?.blob?.size > 0) {
          const blobUrl = URL.createObjectURL(entry.blob);
          blobUrlsRef.current.set(item.cacheKey, blobUrl);
          return blobUrl;
        }
      } catch {}
    }
    return item.serverImageUrl ?? null;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const displayName = (galleryName ?? '').slice(0, 16);

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{ backgroundColor: `rgba(0,0,0,${opacity})` }}
    >
      {/* Toolbar */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-white/10 text-white">
        <span className="font-medium truncate max-w-[160px]">{displayName}</span>
        <input
          type="range"
          min="0.7"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="w-28 accent-primary"
          aria-label="Background opacity"
        />
        {/* Reserved button slot — future additions go here */}
        <div className="flex-1" />
        <button
          type="button"
          aria-label="Close gallery"
          onClick={closeAll}
          className="p-1.5 rounded-full hover:bg-white/10 transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {viewerItem ? (
          <GalleryImageViewer
            item={viewerItem}
            resolveImageUrl={resolveImageUrl}
            onBack={() => setViewerItem(null)}
            onWindowOpen={handleWindowOpen}
          />
        ) : (
          <GalleryGrid
            items={items}
            resolveImageUrl={resolveImageUrl}
            onOpenViewer={setViewerItem}
          />
        )}
      </div>
    </div>
  );
}
