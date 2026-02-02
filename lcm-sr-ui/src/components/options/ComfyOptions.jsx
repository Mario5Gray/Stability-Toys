// src/components/options/ComfyOptions.jsx
import React, { useMemo, useEffect, useRef, useState, useCallback } from "react";

import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Sparkles, Pause } from "lucide-react";

import { CSS_CLASSES } from "../../utils/constants";
import { createComfyInvokerApi } from "@/lib/comfyInvokerApi";
import { useComfyJobWs } from "@/hooks/useComfyJobWs";
import { urlToFile } from "@/utils/imageToFile";
import { NumberStepper, NumberStepperDebounced } from "@/components/ui/NumberStepper";

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
  const [cfg, setCfg] = useState(defaultCfg);
  const [steps, setSteps] = useState(defaultSteps);
  const [denoise, setDenoise] = useState(defaultDenoise);

  // Local file (either from selected chat image URL/file, or manual upload)
  const [file, setFile] = useState(null);

  // (Unused in your snippet, keeping for compatibility if you later wire it)
  const [pullComfyProgress, setPullComfyProgress] = useState(false);

  // Derived progress values (must be after comfy is defined)
  const rawFraction = comfy.job?.progress?.fraction;
  const fraction = Math.max(0, Math.min(1, Number(rawFraction ?? 0)));
  const showProgress =
    comfy.state === "starting" ||
    comfy.state === "running" ||
    comfy.state === "done" ||
    !!comfy.jobId;

  // --- Run action ---
  const run = useCallback(async () => {
    let inputImageFile = null;

    if (inputImage?.kind === "file") {
      inputImageFile = inputImage.file;
    } else if (inputImage?.kind === "url") {
      if (file) {
        inputImageFile = file;
      } else {
        const res = await fetch(inputImage.url);
        const blob = await res.blob();
        inputImageFile = new File([blob], inputImage.filename || "input.png", {
          type: blob.type || "image/png",
        });
      }
    } else {
      inputImageFile = file;
    }

    if (!inputImageFile) {
      const err = new Error("No input image selected.");
      onError?.(err);
      throw err;
    }

    // Snapshot params eagerly
    const snapshotPayload = {
      workflowId,
      params: { cfg, steps, denoise },
      inputImageFile,
    };

    // Create pending message
    onComfyStart?.();

    // Start via WS hook (upload + submit + progress via push)
    comfy.start(snapshotPayload);
  }, [onError, inputImage, file, workflowId, cfg, steps, denoise, comfy, onComfyStart]);

  // Debug mount/unmount (kept, harmless)
  useEffect(() => {
    console.log("ComfyOptions mounted");  
    return () => console.log("ComfyOptions unmounted");
  }, []);

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
    if (comfy.state === "done" && comfy.job?.outputs?.length) {
      onOutputs?.({
        workflowId,
        params: { cfg, steps, denoise },
        outputs: comfy.job.outputs,
        job: comfy.job,
      });
    }
  }, [comfy.state, comfy.job, onOutputs]);

  // Keep file in sync with selected chat image (url/file)
  const lastKeyRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
  
    async function sync() {
      const key = inputImage?.key ?? null;

      // If selection changed, clear stale file immediately
      if (key !== lastKeyRef.current) {
        lastKeyRef.current = key;
        setFile(null);
      }

      if (!inputImage) return;

      if (inputImage.kind === "file") {
        setFile(inputImage.file);
        return;
      }

      if (inputImage.kind === "url") {
        try {
          // urlToFile likely uses fetch internally; do it once here.
          const t0= performance.now();
          _url = inputImage.url
          _name = inputImage.filename || "input.png"
          const f = await urlToFile(_url, _name);
          const t1 = performance.now();
          console.log("[imageToFile] total", (t1 - t0).toFixed(1), "ms", { url, name });
          if (!cancelled) setFile(f);
        } catch (err) {
          if (!cancelled) {
            console.error("Failed to load input image:", err);
            setFile(null);
          }
        }
      }
    }

    sync();
    return () => {
      cancelled = true;
    };
  }, [inputImage, cfg, denoise, steps]);

  return (
    <div className="option-panel-area space-y-3 rounded-2xl border p-4">
      <Label className="text-base font-semibold">ComfyUI Workflow</Label>

      {/* Progress */}
      <div className="flex w-full gap-2 w-full bg-neutral-quaternary rounded-full h-2">
        {showProgress ? (
          rawFraction == null ? (
            // Indeterminate (no value attr) => animated bar in DaisyUI
            <progress className="progress-slim w-full" value={fraction} max={1}/>
          ) : (
            <progress className="progress-slim w-full" value={fraction} max={1} />
          )
        ) : null}
      </div>

{/*
      <div className="text-xs opacity-70">
        status={String(comfy.job?.status)} frac={String(comfy.job?.progress?.fraction)} seen=
        {String(comfy.job?.progress?.nodes_seen)}/{String(comfy.job?.progress?.nodes_total)}
      </div>
*/}
      {/* CFG */}

      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-2">
        <label className="w-10 text-xs text-muted-foreground">cfg</label>

        <NumberStepper
          value={cfg}
          onChange={setCfg}
          step={0.01}
          min={0}
          precision={2}
        />      
        <span className="text-xs opacity-60">0.0–2.0</span>

      {/* Steps */}
        <label className=" w-10 text-xs text-muted-foreground">steps</label>
          <NumberStepper
            value={steps}
            onChange={setSteps}
            step={1}
            min={1}
            precision={1}
          /> 
      <span className="text-xs opacity-60">0-20</span>

      {/* Denoise */}
      
      <label className="w-10 text-xs text-muted-foreground">denoise</label>
        <NumberStepper
            value={denoise}
            onChange={setDenoise}
            step={0.01}
            min={0.00}
            precision={2}
          /> 
      <span className="text-xs opacity-60">0.0–0.5</span>
      </div>
      {/* Image selection */}
      <div className="space-y-1">      
        <Input
          type="file"
          accept="image/*"
          className={CSS_CLASSES.INPUT}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
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
          className="
            border-red-400/40
            text-red-500
            hover:bg-red-500/10
          "
        >
          <Pause className="mr-2 h-4 w-4" />
          Stop
        </Button>
      </div>

      {/* Outputs */}
      {comfy.job?.outputs?.length ? (
        <div style={{ display: "grid", gap: 8 }}>
          {comfy.job.outputs.map((o) => (
            <img
              key={o.id ?? `${comfy.jobId}:${o.type}:${o.subfolder ?? ""}:${o.filename ?? o.url}`}
              src={o.url}
              alt=""
              style={{ maxWidth: "100%" }}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}