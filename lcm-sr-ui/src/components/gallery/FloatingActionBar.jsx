import React, { useState, useRef, useEffect } from 'react';
import { MoreHorizontal, X } from 'lucide-react';

export function FloatingActionBar({
  selectedCount,
  trashMode,
  onDelete,
  onRestore,
  onHardDelete,
  onClear,
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    function onDoc(e) {
      if (!ref.current) return;
      if (!ref.current.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  if (selectedCount < 1) return null;

  const fire = (handler) => () => {
    setMenuOpen(false);
    handler?.();
  };

  return (
    <div
      ref={ref}
      role="toolbar"
      aria-label="Gallery selection actions"
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 rounded-full bg-background/95 border shadow-lg px-3 py-1.5"
    >
      <span className="text-sm font-medium">
        {selectedCount} selected
      </span>
      <div className="relative">
        <button
          type="button"
          aria-label="Open action menu"
          className="p-1.5 rounded hover:bg-muted"
          onClick={() => setMenuOpen((v) => !v)}
        >
          <MoreHorizontal className="h-4 w-4" />
        </button>
        {menuOpen && (
          <div
            role="menu"
            className="absolute bottom-full mb-2 left-0 min-w-[180px] rounded-md border bg-background shadow-md py-1 text-sm"
          >
            {trashMode ? (
              <>
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full text-left px-3 py-1.5 hover:bg-muted"
                  onClick={fire(onRestore)}
                >
                  Restore
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full text-left px-3 py-1.5 hover:bg-muted text-destructive"
                  onClick={fire(onHardDelete)}
                >
                  Delete permanently
                </button>
              </>
            ) : (
              <button
                type="button"
                role="menuitem"
                className="block w-full text-left px-3 py-1.5 hover:bg-muted"
                onClick={fire(onDelete)}
              >
                Delete
              </button>
            )}
          </div>
        )}
      </div>
      <button
        type="button"
        aria-label="Clear selection"
        className="p-1.5 rounded hover:bg-muted"
        onClick={onClear}
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
