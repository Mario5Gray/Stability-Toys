// src/components/gallery/GalleryCreatePopover.jsx
import React, { useState, useRef, useEffect, useId } from 'react';
import { FolderPlus } from 'lucide-react';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { Label } from '../ui/label';

export function GalleryCreatePopover({ onCreateGallery }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const inputRef = useRef(null);
  const wrapperRef = useRef(null);
  const inputId = useId();

  useEffect(() => {
    if (open) {
      setName('');
      const id = setTimeout(() => inputRef.current?.focus(), 0);
      return () => clearTimeout(id);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function handleOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutside);
    return () => document.removeEventListener('mousedown', handleOutside);
  }, [open]);

  function handleConfirm() {
    const trimmed = name.trim();
    if (!trimmed) return;
    onCreateGallery(trimmed);
    setOpen(false);
  }

  return (
    <div className="relative" ref={wrapperRef}>
      <button
        type="button"
        aria-label="New gallery"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded-md hover:bg-muted transition-colors"
      >
        <FolderPlus className="h-4 w-4" />
        [+]
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 z-50 w-56 rounded-2xl border bg-indigo-100 dark:bg-zinc-900 shadow-xl p-3 space-y-2">
          <Label htmlFor={inputId}>Gallery name</Label>
          <Input
            id={inputId}
            ref={inputRef}
            value={name}
            maxLength={16}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleConfirm();
              if (e.key === 'Escape') setOpen(false);
            }}
            placeholder="e.g. Nature"
            className="rounded-2xl"
          />
          <Button type="button" size="sm" className="w-full" onClick={handleConfirm}>
            Create
          </Button>
        </div>
      )}
    </div>
  );
}
