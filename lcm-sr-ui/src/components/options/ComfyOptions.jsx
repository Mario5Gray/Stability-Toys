// src/components/options/ComfyOptions.jsx
import React, { useState, useCallback, useEffect } from "react";

import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Sparkles, Pause } from "lucide-react";

import { CSS_CLASSES } from "../../utils/constants";
import { NumberStepper } from "@/components/ui/NumberStepper";
import { wsClient } from "@/lib/wsClient";
import { jobQueue } from "@/lib/jobQueue";

/**
 * ComfyOptions — now delegates to jobQueue via onRunComfy callback.
 * Progress is tracked by listening to WS job:progress events.
 */
export function ComfyOptions({
  inputImage,
  workflowId = "LCM_CYBERPONY_XL",
  defaultCfg = 0.0,
  defaultSteps = 8,
  defaultDenoise = 0.03,
  onRunComfy,
  queueState,
}) {
  // Controls
  const [comfyCfg, setComfyCFG] = useState(defaultCfg);
  const [comfySteps, setComfySteps] = useState(defaultSteps);
  const [comfyDenoise, setComfyDenoise] = useState(defaultDenoise);

  // Manual upload fallback ONLY (not synced from selection)
  const [uploadFile, setUploadFile] = useState(null);

  // Progress from WS events (latest comfy job)
  const [fraction, setFraction] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [error, setError] = useState(null);

  // Track latest comfy job progress via WS events
  useEffect(() => {
    const unsubProgress = wsClient.on("job:progress", (msg) => {
      // Show progress for any comfy-sourced job
      if (msg.source === "comfy" || msg.jobType === "comfy") {
        setShowProgress(true);
        setFraction(Math.max(0, Math.min(1, Number(msg.progress?.fraction ?? 0))));
      }
    });

    const onComplete = (e) => {
      const job = e.detail?.job;
      if (job?.source === "comfy") {
        setShowProgress(false);
        setFraction(0);
      }
    };

    const onError = (e) => {
      const job = e.detail?.job;
      if (job?.source === "comfy") {
        setShowProgress(false);
        setFraction(0);
      }
    };

    jobQueue.addEventListener("complete", onComplete);
    jobQueue.addEventListener("error", onError);

    return () => {
      unsubProgress();
      jobQueue.removeEventListener("complete", onComplete);
      jobQueue.removeEventListener("error", onError);
    };
  }, []);

  // --- Run action ---
  const run = useCallback(async () => {
    setError(null);

    let inputImageFile = null;

    if (inputImage?.kind === "file") {
      inputImageFile = inputImage.file;
    } else if (inputImage?.kind === "url") {
      const res = await fetch(inputImage.url);
      if (!res.ok) throw new Error(`fetch image failed: ${res.status}`);
      const blob = await res.blob();
      inputImageFile = new File([blob], inputImage.filename || "input.png", {
        type: blob.type || "image/png",
      });
    } else {
      inputImageFile = uploadFile;
    }

    if (!inputImageFile) {
      setError(new Error("No input image selected."));
      return;
    }

    const paramsSnapshot = { cfg: comfyCfg, steps: comfySteps, denoise: comfyDenoise };
    setShowProgress(true);
    setFraction(0);

    onRunComfy?.({
      workflowId,
      params: paramsSnapshot,
      inputImageFile,
    });
  }, [
    onRunComfy,
    inputImage,
    uploadFile,
    workflowId,
    comfyCfg,
    comfySteps,
    comfyDenoise,
  ]);

  return (
    <div className="option-panel-area space-y-3 rounded-2xl border p-4">
      <Label className="text-base font-semibold">ComfyUI Workflow</Label>

      {/* Progress */}
      <div className="flex w-full gap-2 w-full bg-neutral-quaternary rounded-full h-2">
        {showProgress ? (
          <progress className="progress-slim w-full" value={fraction} max={1} />
        ) : null}
      </div>

      {/* Controls */}
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-2">
        <label className="w-10 text-xs text-muted-foreground">cfg</label>
        <NumberStepper value={comfyCfg} onChange={setComfyCFG} step={0.01} min={0} precision={2} />
        <span className="text-xs opacity-60">0.0–2.0</span>

        <label className="w-10 text-xs text-muted-foreground">steps</label>
        <NumberStepper value={comfySteps} onChange={setComfySteps} step={1} min={1} precision={1} />
        <span className="text-xs opacity-60">0–20</span>

        <label className="w-10 text-xs text-muted-foreground">denoise</label>
        <NumberStepper value={comfyDenoise} onChange={setComfyDenoise} step={0.01} min={0.0} precision={2} />
        <span className="text-xs opacity-60">0.0–0.5</span>
      </div>

      {/* Manual file input only (optional fallback) */}
      <div className="space-y-1">
        <Input
          type="file"
          accept="image/*"
          className={CSS_CLASSES.INPUT}
          onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
        />

        {error ? (
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {String(error.message || error)}
          </pre>
        ) : null}
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: 8 }}>
        <Button
          onClick={run}
          className="
            relative overflow-hidden
            border border-purple-400/40
            bg-gradient-to-br from-purple-500/90 to-pink-500/90
            text-white
            shadow-md
            hover:from-purple-500 hover:to-pink-500
            active:scale-[0.90]
            transition-all
          "
        >
          <Sparkles className="mr-2 h-4 w-4" />
          Run
          {queueState?.depth > 0 && (
            <span className="ml-2 inline-flex items-center justify-center h-5 min-w-[20px] rounded-full bg-white/25 text-[11px] font-semibold px-1.5">
              {queueState.depth}
            </span>
          )}
        </Button>

        <Button
          variant="outline"
          onClick={() => jobQueue.cancelAll()}
          className="border-red-400/40 text-red-500 hover:bg-red-500/10"
        >
          <Pause className="mr-2 h-4 w-4" />
          Stop
        </Button>
      </div>
    </div>
  );
}
