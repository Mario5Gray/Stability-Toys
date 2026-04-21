// src/components/ui/InitImageDropzone.jsx
import React, { useMemo } from 'react';
import { useDropzone } from 'react-dropzone';

/**
 * Drag-and-drop wrapper that sets an init image for img2img generation.
 * Apply around any surface: chat main, message composer, init image panel, etc.
 *
 * Props:
 *  - onInitImageSelect(file: File) — called with the dropped image file
 *  - children
 *  - className — optional extra classes on the wrapper div
 */
export function InitImageDropzone({ onInitImageSelect, children, className }) {
  const onDrop = useMemo(
    () => async (acceptedFiles) => {
      if (!acceptedFiles.length || !onInitImageSelect) return;
      try {
        await onInitImageSelect(acceptedFiles[0]);
      } catch (e) {
        console.error('[InitImageDropzone] failed:', e);
      }
    },
    [onInitImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    multiple: false,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] },
    noClick: true,
    noKeyboard: true,
  });

  return (
    <div {...getRootProps()} className={`relative ${className ?? ''}`}>
      <input {...getInputProps()} />

      {isDragActive && (
        <div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center rounded-xl">
          <div
            className={[
              'rounded-2xl px-6 py-4 text-sm shadow-lg backdrop-blur border',
              isDragReject
                ? 'bg-destructive/20 border-destructive text-destructive-foreground'
                : 'bg-black/60 border-indigo-400/60 text-white',
            ].join(' ')}
          >
            {isDragReject ? (
              <div className="flex flex-col items-center gap-1">
                <div className="font-medium">Unsupported file</div>
                <div className="opacity-80">Drop a .png, .jpg, or .webp image</div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-1">
                <div className="font-medium">Drop to set init image</div>
                <div className="opacity-80">Used as starting point for generation</div>
              </div>
            )}
          </div>
        </div>
      )}

      {children}
    </div>
  );
}
