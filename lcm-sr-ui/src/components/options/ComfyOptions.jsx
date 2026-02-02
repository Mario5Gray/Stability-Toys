// src/components/options/ComfyOptions.jsx
import React, { useMemo, useEffect, useRef, useState, useCallback } from "react";

import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Sparkles, Pause } from "lucide-react";

import { CSS_CLASSES } from "../../utils/constants";
import { createComfyInvokerApi } from "@/lib/comfyInvokerApi";
import { useComfyJobWs } from "@/hooks/useComfyJobWs";
import { NumberStepper } from "@/components/ui/NumberStepper";

/**
 * ComfyOptions (no preview / no eager url->File conversion)
 * - No syncing inputImage into local file state
 * - URL->File conversion happens ONLY when Run is clicked
 * - Outputs are expected to be injected into chat via onOutputs, not displayed here
 */
export function ComfyOptions({
  inputImage,
  apiBase = "https://node2:4205",
  workflowId = "LCM_CYBERPONY_XL",
  defaultCfg = 0.0,
  defaultSteps = 8,
  defaultDenoise = 0.03,
  onStart,
  onDone,
  onError,
  onOutputs,
  onComfyStart,
  queueState,
}) {
  // API + job hook
  const api = useMemo(() => createComfyInvokerApi(apiBase), [apiBase]);
  const comfy = useComfyJobWs({ api });

  // Controls
  const [comfyCfg, setComfyCFG] = useState(defaultCfg);
  const [comfySteps, setComfySteps] = useState(defaultSteps);
  const [comfyDenoise, setComfyDenoise] = useState(defaultDenoise);

  // Manual upload fallback ONLY (not synced from selection)
  const [uploadFile, setUploadFile] = useState(null);

  // Snapshot run params so callbacks don’t “drift” if user changes sliders mid-job
  const lastRunRef = useRef(null);

  // Derived progress values
  const rawFraction = comfy.job?.progress?.fraction;
  const fraction = Math.max(0, Math.min(1, Number(rawFraction ?? 0)));
  const showProgress =
    comfy.state === "starting" ||
    comfy.state === "running" ||
    comfy.state === "done" ||
    !!comfy.jobId;

  // --- Run action ---
  const run = useCallback(async () => {
    onStart?.();

    let inputImageFile = null;

    if (inputImage?.kind === "file") {
      inputImageFile = inputImage.file;
    } else if (inputImage?.kind === "url") {
      // Convert URL->File ONLY at Run time (no preview, no prefetch)
      const res = await fetch(inputImage.url);
      if (!res.ok) throw new Error(`fetch image failed: ${res.status}`);
      const blob = await res.blob();
      inputImageFile = new File([blob], inputImage.filename || "input.png", {
        type: blob.type || "image/png",
      });
    } else {
      // Manual upload fallback
      inputImageFile = uploadFile;
    }

    if (!inputImageFile) {
      const err = new Error("No input image selected.");
      onError?.(err);
      throw err;
    }

    // Snapshot params eagerly (stable for this run)
    const paramsSnapshot = { cfg: comfyCfg, steps: comfySteps, denoise: comfyDenoise };
    lastRunRef.current = {
      workflowId,
      params: paramsSnapshot,
    };

    // Create pending message in chat
    onComfyStart?.();

    // Start via WS hook (upload + submit + progress via push)
    comfy.start({
      workflowId,
      params: paramsSnapshot,
      inputImageFile,
    });
  }, [
    onStart,
    onError,
    onComfyStart,
    inputImage,
    uploadFile,
    workflowId,
    comfyCfg,
    comfySteps,
    comfyDenoise,
    comfy,
  ]);

  // done callback
  useEffect(() => {
    if (comfy.state === "done") {
      onDone?.(comfy.job);
    }
  }, [comfy.state, comfy.job, onDone]);

  // error callback
  useEffect(() => {
    if (comfy.state === "error" && comfy.error) {
      onError?.(comfy.error);
    }
  }, [comfy.state, comfy.error, onError]);

  // outputs callback (when outputs arrive)
  useEffect(() => {
    if (comfy.state !== "done") return;
    if (!comfy.job?.outputs?.length) return;

    const snap = lastRunRef.current;
    const payloadParams = snap?.params ?? { cfg: comfyCfg, steps: comfySteps, denoise: comfyDenoise };
    const payloadWorkflowId = snap?.workflowId ?? workflowId;

    onOutputs?.({
      workflowId: payloadWorkflowId,
      params: payloadParams,
      outputs: comfy.job.outputs,
      job: comfy.job,
    });
  }, [comfy.state, comfy.job, onOutputs, workflowId]);

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

        {comfy.error ? (
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {String(comfy.error.message || comfy.error)}
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
          onClick={comfy.cancel}
          disabled={!comfy.isBusy}
          className="border-red-400/40 text-red-500 hover:bg-red-500/10"
        >
          <Pause className="mr-2 h-4 w-4" />
          Stop
        </Button>
      </div>

      {/* No preview grid here. Outputs land in chat via onOutputs. */}
    </div>
  );
}