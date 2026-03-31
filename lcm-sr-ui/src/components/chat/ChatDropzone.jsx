// src/components/chat/ChatDropzone.jsx
import React, { useMemo } from "react";
import { useDropzone } from "react-dropzone";
import { useDropIngest } from "@/hooks/useDropIngest";

/**
 * Wraps children and enables drag-drop import of images.
 *
 * Props:
 *  - addMessage
 *  - setSelectedMsgId
 *  - setUploadFile (optional; recommended so drop selects image, not upload)
 *  - onInitImageSelect (optional; persists the init image for img2img generation)
 *  - children
 */
export function ChatDropzone({
  addMessage,
  setSelectedMsgId,
  setUploadFile,
  onInitImageSelect,
  children,
}) {
  const { ingestFiles } = useDropIngest({
    addMessage,
    setSelectedMsgId,
    setUploadFile,
  });

  const onDrop = useMemo(
    () => async (acceptedFiles) => {
      try {
        // Set the first dropped file as init image for img2img
        if (acceptedFiles.length > 0 && onInitImageSelect) {
          const file = acceptedFiles[0];
          try {
            await onInitImageSelect(file);
          } catch (persistError) {
            console.error("[ChatDropzone] init image persistence failed:", persistError);
          }
        }
        // Also run ingest (extracts embedded params from PNG metadata if present)
        await ingestFiles(acceptedFiles);
      } catch (e) {
        console.error("[ChatDropzone] ingest failed:", e);
      }
    },
    [ingestFiles, onInitImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    multiple: true,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
    },
    // Prevent the browser from opening the file
    noClick: true,
    noKeyboard: true,
  });

  return (
    <div {...getRootProps()} className="relative h-full w-full">
      <input {...getInputProps()} />

      {/* Overlay */}
      {isDragActive ? (
        <div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center">
          <div
            className={[
              "rounded-2xl px-6 py-4 text-sm shadow-lg backdrop-blur",
              "border",
              isDragReject
                ? "bg-destructive/20 border-destructive text-destructive-foreground"
                : "bg-black/50 border-white/20 text-white",
            ].join(" ")}
          >
            {isDragReject ? (
              <div className="flex flex-col items-center gap-1">
                <div className="font-medium">Unsupported file</div>
                <div className="opacity-80">Drop an image file (.png, .jpg, .webp)</div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-1">
                <div className="font-medium">Drop to set init image</div>
                <div className="opacity-80">Used as starting point for generation</div>
              </div>
            )}
          </div>
        </div>
      ) : null}

      {children}
    </div>
  );
}
