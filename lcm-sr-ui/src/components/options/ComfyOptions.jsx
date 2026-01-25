// src/components/options/ComfyOptions.jsx
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Heart, Pause, Play } from 'lucide-react';
import { Input } from '@/components/ui/input';
import {CSS_CLASSES} from '../../utils/constants';
import { createComfyInvokerApi } from "@/lib/comfyInvokerApi";
import { useComfyJob } from "@/hooks/useComfyJob";
import React, { useMemo, useState, useCallback } from "react";

export function ComfyOptions({
  apiBase = "https://node2:4205",
  workflowId = "TRACKING-LCM-DIFFS",
  defaultCfg = 0.0,
  defaultSteps = 12,
  defaultDenoise = 0.13,
  onStart,
  onDone,
  onError,
}) {
  const api = useMemo(() => createComfyInvokerApi(apiBase), [apiBase]);
  const comfy = useComfyJob({ api });

  const [cfg, setCfg] = useState(defaultCfg);
  const [steps, setSteps] = useState(defaultSteps);
  const [denoise, setDenoise] = useState(defaultDenoise);
  const [file, setFile] = useState(null);
  console.log("OPTIONS PANEL BUILD STAMP", new Date().toISOString());
  const run = useCallback(async () => {
    try {
      onStart?.();

      const started = await comfy.start({
        workflowId,
        params: { cfg, steps, denoise },
        inputImageFile: file, // can be null
      });

      return started;
    } catch (e) {
      onError?.(e);
      throw e;
    }
  }, [cfg, steps, denoise, file, comfy, workflowId, onStart, onError]);

  // If you want a "done" callback when outputs arrive:
  // (cheap + safe: just watch comfy.state)
  React.useEffect(() => {
    if (comfy.state === "done") {
      onDone?.(comfy.job);
    }
  }, [comfy.state, comfy.job, onDone]);


  return (
    <div className="space-y-3 rounded-2xl border p-4 bg-gradient-to-br from-purple-50/50 to-pink-50/50 dark:from-purple-950/20 dark:to-pink-950/20">
      <Label className="text-base font-semibold">Send to ComfyUI Workflow</Label>
      <div>Status: {comfy.state}</div>

      <div className="space-y-1">
      <label>CFG</label>
        <input
          type="number"
          value={cfg}
          step="0.05"
          onChange={(e) => setCfg(parseFloat(e.target.value))}
        />
      </div>

      <div className="space-y-1">
      <label>Steps</label>
        <input
          type="number"
          value={steps}
          step="1"
          onChange={(e) => setSteps(parseInt(e.target.value, 10))}
        />
      </div>

      <label>
        Denoise
        <input
          type="number"
          value={denoise}
          step="0.05"
          onChange={(e) => setDenoise(parseFloat(e.target.value))}
        />
      </label>

      <div className="space-y-1">
              <Input
                type="file"
                accept="image/*"
                className={CSS_CLASSES.INPUT}
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
      {comfy.error ? (
        <pre style={{ whiteSpace: "pre-wrap" }}>
          {String(comfy.error.message || comfy.error)}
        </pre>
       ) : null}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <button disabled={comfy.isBusy} onClick={run}>Run</button>
        <button disabled={!comfy.isBusy} onClick={comfy.cancel}>Cancel</button>
      </div>


 {comfy.job?.outputs?.length ? (
        <div style={{ display: "grid", gap: 8 }}>
          {comfy.job.outputs.map((o) => (
            <img
              key={o.url ?? `${o.filename}-${o.subfolder ?? ""}`}
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