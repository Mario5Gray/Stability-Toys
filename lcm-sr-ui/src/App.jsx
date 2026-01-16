import React, { useMemo, useRef, useState, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Loader2, Send, Square, X } from "lucide-react";

/* -----------------------------
 * Utils
 * ----------------------------- */
function clampInt(n, lo, hi) {
  const x = Number.isFinite(n) ? n : lo;
  return Math.max(lo, Math.min(hi, Math.round(x)));
}

function eightDigitSeed() {
  return Math.floor(Math.random() * 100_000_000);
}

function safeJsonString(s) {
  return (s ?? "").toString();
}

function nowId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function parseApiBases(env) {
  if (!env) return [];
  return String(env)
    .split(/[;,]/g)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => s.replace(/\/+$/, ""));
}

function normalizeBase(s) {
  return String(s || "").trim().replace(/\/+$/, "");
}

/**
 * Read error from a previously-read ArrayBuffer.
 * IMPORTANT: never call res.json()/res.text() after res.arrayBuffer().
 */
function readErrorFromArrayBuffer(res, buf) {
  try {
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const text = new TextDecoder().decode(buf);

    if (ct.includes("application/json")) {
      try {
        const j = JSON.parse(text);
        return j?.detail ?? j?.error ?? text;
      } catch {
        return text;
      }
    }
    return text;
  } catch {
    return res.statusText || "Request failed";
  }
}

/**
 * Read response body exactly once -> object URL.
 * Fixes: "Body is disturbed or locked"
 */
async function responseToObjectURLStrict(res) {
  const buf = await res.arrayBuffer();

  if (!res.ok) {
    const detail = readErrorFromArrayBuffer(res, buf);
    throw new Error(`HTTP ${res.status}: ${detail || res.statusText}`);
  }

  const contentType = res.headers.get("content-type") || "image/png";
  const blob = new Blob([buf], { type: contentType });
  return URL.createObjectURL(blob);
}

async function copyToClipboard(text) {
  const t = String(text ?? "");
  try {
    await navigator.clipboard.writeText(t);
    return true;
  } catch {
    // Fallback for older / restricted contexts
    try {
      const ta = document.createElement("textarea");
      ta.value = t;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

export default function App() {
  const [prompt, setPrompt] = useState(
    "a cinematic photograph of a futuristic city at sunset"
  );

  // Upload/SR ingest
  const [uploadFile, setUploadFile] = useState(null); // File | null
  const [srMagnitude, setSrMagnitude] = useState(2); // 1..3

  // Right panel options
  const [size, setSize] = useState("512x512");
  const [steps, setSteps] = useState(4);
  const [cfg, setCfg] = useState(1.0);

  const [seedMode, setSeedMode] = useState("random"); // random | fixed
  const [seed, setSeed] = useState(() => String(eightDigitSeed()));

  // Super-Resolution level: 0=off, 1..4=on (UI). Backend magnitude max is 3.
  const [srLevel, setSrLevel] = useState(0); // 0..4

  // Chat messages
  const [messages, setMessages] = useState(() => [
    {
      id: nowId(),
      role: "assistant",
      kind: "system",
      text: "Ready. Type a prompt and hit Send to generate a PNG. Toggle SR if your server supports it.",
      ts: Date.now(),
    },
  ]);

  // Map<assistantMsgId, AbortController>
  const inflightRef = useRef(new Map());
  // Round-robin counter
  const rrRef = useRef(0);

  // Track blob: URLs so we can revoke them deterministically
  const blobUrlsRef = useRef(new Set());

  // Options scroll hint refs/state
  const optionsScrollRef = useRef(null);
  const [optionsCanScrollDown, setOptionsCanScrollDown] = useState(false);
  const [optionsCanScrollUp, setOptionsCanScrollUp] = useState(false);

  // Revoke object URLs on unmount
  useEffect(() => {
    return () => {
      for (const url of blobUrlsRef.current) {
        try {
          URL.revokeObjectURL(url);
        } catch {}
      }
      blobUrlsRef.current.clear();
    };
  }, []);

  const apiConfig = useMemo(() => {
    const vitePlural =
      typeof import.meta !== "undefined" &&
      import.meta.env &&
      import.meta.env.VITE_API_BASES;

    const viteSingle =
      typeof import.meta !== "undefined" &&
      import.meta.env &&
      import.meta.env.VITE_API_BASE;

    const craSingle =
      typeof process !== "undefined" &&
      process.env &&
      process.env.REACT_APP_API_BASE;

    const bases = parseApiBases(vitePlural || "");
    const single = normalizeBase(viteSingle || craSingle || "");

    return { bases, single };
  }, []);

  const pickApiBaseForRequest = useCallback(() => {
    const bases = apiConfig.bases.map(normalizeBase).filter(Boolean);
    if (bases.length > 0) {
      const idx = rrRef.current++ % bases.length;
      return bases[idx];
    }
    return normalizeBase(apiConfig.single);
  }, [apiConfig]);

  const inflightCount = useMemo(
    () => messages.reduce((n, m) => n + (m.kind === "pending" ? 1 : 0), 0),
    [messages]
  );

  const updateMessage = useCallback((id, patch) => {
    setMessages((m) => m.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  }, []);

  // Feature: click image -> restore settings
  const applyMessageMeta = useCallback((meta) => {
    if (!meta) return;

    if (typeof meta.size === "string" && /^\d+x\d+$/i.test(meta.size)) {
      setSize(meta.size);
    }
    if (Number.isFinite(meta.steps)) setSteps(clampInt(Number(meta.steps), 1, 50));
    if (Number.isFinite(meta.cfg)) setCfg(Number(meta.cfg));
    // If meta indicates SR, try to restore a level (prefer stored magnitude/pass count if present)
    if (meta?.superres) {
      const mag =
        Number(meta?.srMagnitude) ||
        Number(meta?.srPasses) ||
        (meta?.srScale ? 2 : 1);
      setSrLevel(clampInt(mag, 1, 4));
    } else if (typeof meta?.superres === "boolean") {
      setSrLevel(meta.superres ? 2 : 0);
    }

    if (meta.seed !== undefined && meta.seed !== null) {
      setSeedMode("fixed");
      setSeed(String(meta.seed));
    }
  }, []);

  // (3) options scroll hint updater
  const updateOptionsScrollHints = useCallback(() => {
    const el = optionsScrollRef.current;
    if (!el) return;

    const canDown = el.scrollHeight - el.scrollTop - el.clientHeight > 6;
    const canUp = el.scrollTop > 6;

    setOptionsCanScrollDown(canDown);
    setOptionsCanScrollUp(canUp);
  }, []);

  useEffect(() => {
    // initial + whenever layout changes (messages might change height)
    updateOptionsScrollHints();
  }, [updateOptionsScrollHints, messages.length]);

  const onSendSuperResUpload = useCallback(async () => {
    if (!uploadFile) return;

    const assistantId = nowId();
    const apiBaseForThisRequest = pickApiBaseForRequest();

    const userMsg = {
      id: nowId(),
      role: "user",
      kind: "text",
      text: `Super-res upload: ${uploadFile.name} (magnitude ${srMagnitude})`,
      meta: {
        ingest: "superres",
        filename: uploadFile.name,
        magnitude: srMagnitude,
      },
      ts: Date.now(),
    };

    const pendingMsg = {
      id: assistantId,
      role: "assistant",
      kind: "pending",
      text: "Super-resolving…",
      meta: {
        request: {
          apiBase: apiBaseForThisRequest || "(same origin)",
          endpoint: "/superres",
          magnitude: srMagnitude,
        },
      },
      ts: Date.now(),
    };

    setMessages((m) => [...m, userMsg, pendingMsg]);

    const controller = new AbortController();
    inflightRef.current.set(assistantId, controller);

    try {
      const fd = new FormData();
      fd.append("file", uploadFile);
      fd.append("magnitude", String(srMagnitude));
      fd.append("out_format", "png");
      fd.append("quality", "92");

      const res = await fetch(`${apiBaseForThisRequest}/superres`, {
        method: "POST",
        body: fd,
        signal: controller.signal,
      });

      const url = await responseToObjectURLStrict(res);
      blobUrlsRef.current.add(url);

      const magHdr = res.headers.get("X-SR-Magnitude") || String(srMagnitude);
      const passesHdr = res.headers.get("X-SR-Passes") || magHdr;
      const scaleHdr =
        res.headers.get("X-SR-Scale-Per-Pass") ||
        res.headers.get("X-SR-Scale") ||
        null;

      const backend =
        res.headers.get("X-LCM-Backend") ||
        res.headers.get("X-Backend") ||
        res.headers.get("X-Host") ||
        null;

      updateMessage(assistantId, {
        kind: "image",
        text: `Done (SR upload). Passes: ${passesHdr}${
          scaleHdr ? ` · scale/pass: ${scaleHdr}` : ""
        }`,
        imageUrl: url,
        meta: {
          superres: true,
          srMagnitude: Number(magHdr),
          srPasses: Number(passesHdr),
          srScale: scaleHdr,
          apiBase: apiBaseForThisRequest || "",
          backend,
          sourceUpload: uploadFile.name,
        },
      });
    } catch (err) {
      const msg =
        err?.name === "AbortError" ? "Canceled." : err?.message || String(err);
      updateMessage(assistantId, { kind: "error", text: msg });
    } finally {
      inflightRef.current.delete(assistantId);
    }
  }, [uploadFile, srMagnitude, pickApiBaseForRequest, updateMessage]);

  const onSend = useCallback(async () => {
    const p = safeJsonString(prompt).trim();
    if (!p) return;

    const reqSeed =
      seedMode === "random"
        ? eightDigitSeed()
        : clampInt(parseInt(seed || "0", 10), 0, 2 ** 31 - 1);

    const opts = {
      size,
      steps,
      cfg,
      seedMode,
      seed: reqSeed,
      superres: srLevel > 0,
      srLevel,
    };

    const userMsg = {
      id: nowId(),
      role: "user",
      kind: "text",
      text: p,
      meta: opts,
      ts: Date.now(),
    };

    const assistantId = nowId();
    const apiBaseForThisRequest = pickApiBaseForRequest();

    const pendingMsg = {
      id: assistantId,
      role: "assistant",
      kind: "pending",
      text: "Generating…",
      meta: {
        request: {
          apiBase: apiBaseForThisRequest || "(same origin)",
          ...opts,
        },
      },
      ts: Date.now(),
    };

    setMessages((m) => [...m, userMsg, pendingMsg]);

    const controller = new AbortController();
    inflightRef.current.set(assistantId, controller);

      const superresOn = srLevel > 0;

    try {
      const body = {
        prompt: p,
        size: opts.size,
        num_inference_steps: clampInt(opts.steps, 1, 50),
        guidance_scale: Math.max(0, Math.min(20, Number(opts.cfg) || 0)),
        seed: reqSeed,
        superres: superresOn,
        superres_format: "png",
        superres_quality: 92,      
        superres_magnitude: srLevel > 0 ? clampInt(Number(srLevel), 1, 3) : 1,
      };

      const res = await fetch(`${apiBaseForThisRequest}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      const url = await responseToObjectURLStrict(res);
      blobUrlsRef.current.add(url);

      const seedHdr = res.headers.get("X-Seed");
      const srHdr = res.headers.get("X-SuperRes");
      const srScale =
        res.headers.get("X-SR-Scale") ||
        res.headers.get("X-SR-Scale-Per-Pass") ||
        null;

      const backend =
        res.headers.get("X-LCM-Backend") ||
        res.headers.get("X-Backend") ||
        res.headers.get("X-Host") ||
        null;

      updateMessage(assistantId, {
        kind: "image",
        text: opts.superres
          ? `Done (SR ${srLevel}). Seed: ${seedHdr ?? reqSeed}`
          : `Done. Seed: ${seedHdr ?? reqSeed}`,
          imageUrl: url,
        meta: {
          seed: seedHdr ?? reqSeed,
          superres: srHdr === "1" || !!opts.superres,
          srScale: srScale ?? null,
          srMagnitude: clampInt(Number(srLevel), 1, 3),
          srLevel: Number(srLevel),
          size: opts.size,
          steps: opts.steps,
          cfg: opts.cfg,
          apiBase: apiBaseForThisRequest || "",
          backend
        },
      });

      if (seedMode === "random") setSeed(String(eightDigitSeed()));
    } catch (err) {
      const msg =
        err?.name === "AbortError" ? "Canceled." : err?.message || String(err);
      updateMessage(assistantId, { kind: "error", text: msg });
    } finally {
      inflightRef.current.delete(assistantId);
    }
  }, [
    prompt,
    seedMode,
    seed,
    size,
    steps,
    cfg,
    srLevel,
    pickApiBaseForRequest,
    updateMessage,
  ]);

  const cancelRequest = useCallback((id) => {
    const ctl = inflightRef.current.get(id);
    if (ctl) ctl.abort();
  }, []);

  const cancelAll = useCallback(() => {
    for (const [, ctl] of inflightRef.current.entries()) ctl.abort();
  }, []);

  const onKeyDown = useCallback(
    (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        onSend();
      }
    },
    [onSend]
  );

  // (4) Copy prompt badge/button
  const [copied, setCopied] = useState(false);
  const onCopyPrompt = useCallback(async () => {
    const text = safeJsonString(prompt).trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 900);
    } catch {
      // fallback
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 900);
      } catch {}
    }
  }, [prompt]);

  const serverLabel =
    apiConfig.bases.length > 0
      ? `RR (${apiConfig.bases.length} backends)`
      : apiConfig.single
      ? apiConfig.single
      : "(same origin)";

  // ---- style helpers ----
  const selectTriggerClass =
    "rounded-2xl bg-background text-foreground shadow-sm border";
  // (2) Force opaque dropdown surface (white) so background never shows through
  const selectContentClass =
    "!bg-white !text-black !opacity-100 shadow-xl border border-black/10";
  const selectItemClass =
    "!bg-white !text-black data-[highlighted]:!bg-black/5";
  const inputClass = "rounded-2xl bg-background text-foreground shadow-sm border";

  const sliderClass =
    "[&_[data-orientation=horizontal]]:bg-muted " +
    "[&_[role=slider]]:bg-foreground " +
    "[&_[role=slider]]:border-foreground " +
    "[&_[role=slider]]:shadow";

  return (
    <div className="h-screen overflow-hidden bg-background text-foreground">
      <div className="mx-auto max-w-6xl p-4 md:p-6 h-full">
        <div className="grid h-full grid-cols-1 gap-4 md:grid-cols-[1fr_360px]">
          {/* Main chat */}
          <Card className="overflow-hidden rounded-2xl shadow-sm h-full flex flex-col">
            <CardHeader className="border-b">
              <div className="flex items-center justify-between gap-3">
                <CardTitle className="text-xl">LCM + SR Chat</CardTitle>

                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Badge variant="secondary">/generate</Badge>
                  <Badge variant="secondary">PNG</Badge>

                  {/* (4) Copy prompt badge */}
                  <button
                    type="button"
                    onClick={onCopyPrompt}
                    className="inline-flex"
                    title="Copy current prompt to clipboard"
                  >
                    <Badge variant="outline" className="bg-background">
                      {copied ? "Copied!" : "Copy prompt"}
                    </Badge>
                  </button>

                  {srLevel > 0 ? <Badge>SR {srLevel}</Badge> : <Badge variant="outline">SR off</Badge>}
                  {inflightCount > 0 ? (
                    <Badge variant="secondary">{inflightCount} running</Badge>
                  ) : null}
                </div>
              </div>

              <div className="text-sm text-muted-foreground">
                Tip: press <span className="font-medium">Ctrl/⌘ + Enter</span> to send.
              </div>
            </CardHeader>

            {/* (1) chat uses h-full */}
            <CardContent className="flex flex-1 flex-col p-0 min-h-0">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 md:p-6 min-h-0">
                <div className="space-y-4">
                  {messages.map((m) => (
                    <MessageBubble
                      key={m.id}
                      msg={m}
                      onCancel={m.kind === "pending" ? () => cancelRequest(m.id) : null}
                      onPickMeta={m.kind === "image" ? () => applyMessageMeta(m.meta) : null}
                    />
                  ))}
                </div>
              </div>

              <Separator />

              {/* Composer */}
              <div className="p-3 md:p-4">
                <div className="flex gap-2">
                  <Textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={onKeyDown}
                    placeholder="Describe what you want to generate…"
                    className="min-h-[52px] resize-none rounded-2xl"
                  />
                  <div className="flex flex-col gap-2">
                    <Button className="rounded-2xl" onClick={onSend} disabled={!prompt.trim()}>
                      <Send className="mr-2 h-4 w-4" />
                      Send
                    </Button>

                    <Button
                      variant="outline"
                      className="rounded-2xl"
                      onClick={cancelAll}
                      disabled={inflightCount === 0}
                      title="Cancel all in-flight requests"
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Cancel all
                    </Button>
                  </div>
                </div>

                <div className="mt-2 text-xs text-muted-foreground">
                  Current: size {size} · steps {steps} · CFG {cfg.toFixed(1)} · seed{" "}
                  {seedMode === "random" ? "random" : seed}
                  {srLevel > 0 ? ` · SR ${srLevel}` : ""} · {serverLabel}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Right-side options */}
          <Card className="rounded-2xl shadow-sm h-full flex flex-col overflow-hidden">
            <CardHeader className="border-b">
              <CardTitle className="text-lg">Options</CardTitle>
              <div className="text-sm text-muted-foreground">Generation parameters</div>
            </CardHeader>

            {/* (3) scroll hint overlay */}
            <div className="relative flex-1 min-h-0">
              {(optionsCanScrollDown || optionsCanScrollUp) ? (
                <div className="mt-2 text-xs text-muted-foreground">
                  More {optionsCanScrollUp ? "↑" : "↓"} (scroll)
                </div>
              ) : null}
              <CardContent
                ref={optionsScrollRef}
                onScroll={updateOptionsScrollHints}
                className="h-full overflow-y-auto space-y-5 p-4 md:p-5 pt-10"
              >
                <div className="space-y-2">
                  <Label>Size</Label>
                  <Select value={size} onValueChange={setSize}>
                    <SelectTrigger className={selectTriggerClass}>
                      <SelectValue placeholder="Select size" />
                    </SelectTrigger>
                    <SelectContent className={selectContentClass}>
                      <SelectItem className={selectItemClass} value="256x256">
                        256×256
                      </SelectItem>
                      <SelectItem className={selectItemClass} value="512x512">
                        512×512
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Steps</Label>
                    <span className="text-sm text-muted-foreground">{steps}</span>
                  </div>
                  <Slider
                    className={sliderClass}
                    value={[steps]}
                    min={1}
                    max={12}
                    step={1}
                    onValueChange={(v) => setSteps(v[0] ?? 4)}
                  />
                  <div className="text-xs text-muted-foreground">
                    Server allows up to 50; typical LCM is 2–8.
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>CFG</Label>
                    <span className="text-sm text-muted-foreground">{cfg.toFixed(1)}</span>
                  </div>
                  <Slider
                    className={sliderClass}
                    value={[cfg]}
                    min={0}
                    max={5}
                    step={0.1}
                    onValueChange={(v) => setCfg(Number(v[0] ?? 1.0))}
                  />
                  <div className="text-xs text-muted-foreground">LCM commonly uses ~1.0.</div>
                </div>

                <Separator />

                <div className="space-y-3">
                  <Label>Seed</Label>
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <Select value={seedMode} onValueChange={setSeedMode}>
                        <SelectTrigger className={selectTriggerClass}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className={selectContentClass}>
                          <SelectItem className={selectItemClass} value="random">
                            Random
                          </SelectItem>
                          <SelectItem className={selectItemClass} value="fixed">
                            Fixed
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button
                      variant="outline"
                      className="rounded-2xl"
                      onClick={() => setSeed(String(eightDigitSeed()))}
                      title="Generate a new random seed"
                    >
                      Randomize
                    </Button>
                  </div>

                  <div className="flex items-center gap-2">
                    <Input
                      value={seed}
                      onChange={(e) => {
                        const v = (e.target.value || "").replace(/[^\d]/g, "").slice(0, 10);
                        setSeed(v);
                      }}
                      disabled={seedMode !== "fixed"}
                      className={inputClass}
                      inputMode="numeric"
                      placeholder="seed"
                    />
                  </div>

                  <div className="text-xs text-muted-foreground">
                    When Random: a new seed is chosen per request. When Fixed: the seed field is used.
                  </div>
                </div>

                <Separator />

                {/* (6) toggle above upload section */}
                <div className="space-y-2 rounded-2xl border p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium">Super-Resolution</div>
                      <div className="text-xs text-muted-foreground">
                        0 = off · 1–3 = passes
                      </div>
                    </div>

                    <div className="text-sm text-muted-foreground tabular-nums">
                      {srLevel === 0 ? "Off" : `Level ${srLevel}`}
                    </div>
                  </div>

                  <Slider
                    className={sliderClass}
                    value={[srLevel]}
                    min={0}
                    max={4}
                    step={1}
                    onValueChange={(v) => setSrLevel(v?.[0] ?? 0)}
                  />

                  <div className="flex justify-between text-[11px] text-muted-foreground">
                    <span>0</span><span>1</span><span>2</span><span>3</span><span>4</span>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="font-medium">Super-resolution an uploaded image</div>

                  {/* (7) Magnitude above Image file */}
                  <div className="space-y-2">
                    <Label>Magnitude</Label>
                    <Select
                      value={String(srMagnitude)}
                      onValueChange={(v) => setSrMagnitude(Number(v))}
                    >
                      <SelectTrigger className={selectTriggerClass}>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className={selectContentClass}>
                        <SelectItem className={selectItemClass} value="1">
                          1 (1 pass)
                        </SelectItem>
                        <SelectItem className={selectItemClass} value="2">
                          2 (default)
                        </SelectItem>
                        <SelectItem className={selectItemClass} value="3">
                          3 (3 passes)
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="text-xs text-muted-foreground">
                      Magnitude = number of SR passes. Default is 2.
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Image file</Label>
                    <Input
                      type="file"
                      accept="image/*"
                      className={inputClass}
                      onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                    />
                    <div className="text-xs text-muted-foreground">
                      {uploadFile ? `Selected: ${uploadFile.name}` : "Choose a JPG/PNG/WebP/etc."}
                    </div>
                  </div>

                  <Button
                    className="w-full rounded-2xl"
                    onClick={onSendSuperResUpload}
                    disabled={!uploadFile}
                    title={!uploadFile ? "Pick an image first" : "Upload and super-resolve"}
                  >
                    <Send className="mr-2 h-4 w-4" />
                    Super-res uploaded image
                  </Button>
                </div>

                <Separator />

                <div className="rounded-2xl bg-muted/40 p-3 text-xs text-muted-foreground">
                  <div className="font-medium text-foreground">Server base</div>
                  <div className="mt-1 break-all">{serverLabel}</div>
                  <div className="mt-2">Output: PNG only (per UI spec)</div>

                  {apiConfig.bases.length > 0 ? (
                    <div className="mt-2">
                      <div className="font-medium text-foreground">Backends</div>
                      <div className="mt-1 space-y-1">
                        {apiConfig.bases.map((b) => (
                          <div key={b} className="break-all">
                            {b}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              </CardContent>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ msg, onCancel, onPickMeta }) {
  const isUser = msg.role === "user";
  const bubble = isUser
    ? "bg-primary text-primary-foreground"
    : msg.kind === "error"
    ? "bg-destructive text-destructive-foreground"
    : "bg-muted";

  return (
    <div className={"flex w-full " + (isUser ? "justify-end" : "justify-start")}>
      <div className={"max-w-[92%] rounded-2xl px-4 py-3 shadow-sm " + bubble}>
        <div className="flex items-start gap-3">
          <div className="flex-1 whitespace-pre-wrap text-sm leading-relaxed">
            {msg.text}
          </div>

          {msg.kind === "pending" && onCancel ? (
            <button
              className="opacity-80 hover:opacity-100 transition"
              onClick={onCancel}
              title="Cancel this request"
              aria-label="Cancel"
              type="button"
            >
              <X className="h-4 w-4" />
            </button>
          ) : null}
        </div>

        {msg.kind === "pending" && (
          <div className="mt-2 flex items-center gap-2 text-xs opacity-80">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Working…
            {msg.meta?.request?.apiBase ? (
              <span className="ml-auto opacity-70">{msg.meta.request.apiBase}</span>
            ) : null}
          </div>
        )}

        {msg.kind === "image" && msg.imageUrl && (
          <div className="mt-3">
            <img
              src={msg.imageUrl}
              alt="generation"
              className={
                "max-h-[520px] w-auto rounded-2xl border bg-background " +
                (onPickMeta ? "cursor-pointer hover:opacity-95" : "")
              }
              loading="lazy"
              onClick={() => onPickMeta?.()}
              title={onPickMeta ? "Click to load these settings" : undefined}
            />

            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
              <Pill label={`seed ${msg.meta?.seed ?? "?"}`} />
              <Pill label={msg.meta?.size ? `size ${msg.meta.size}` : ""} />
              <Pill label={Number.isFinite(msg.meta?.steps) ? `steps ${msg.meta.steps}` : ""} />
              <Pill label={Number.isFinite(msg.meta?.cfg) ? `cfg ${Number(msg.meta.cfg).toFixed(1)}` : ""} />
              {msg.meta?.superres ? <Pill label={`SR ${msg.meta?.srScale ?? ""}`.trim()} /> : null}
              {msg.meta?.backend ? <Pill label={`backend ${msg.meta.backend}`} /> : null}

              <a
                className="ml-auto underline"
                href={msg.imageUrl}
                download={`lcm_${msg.meta?.seed ?? "image"}.png`}
              >
                Download
              </a>
            </div>
          </div>
        )}

        {msg.role === "user" && msg.meta && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs opacity-90">
            <Pill label={`size ${msg.meta.size}`} dark />
            <Pill label={`steps ${msg.meta.steps}`} dark />
            <Pill label={`cfg ${Number(msg.meta.cfg).toFixed(1)}`} dark />
            <Pill
              label={msg.meta.seedMode === "random" ? "seed random" : `seed ${msg.meta.seed}`}
              dark
            />
            {msg.meta.superres ? <Pill label="SR on" dark /> : <Pill label="SR off" dark />}

            <button
              type="button"
              className="ml-auto rounded-full border border-white/20 bg-black/20 px-2 py-0.5 text-white/90 hover:bg-black/30"
              onClick={async () => {
                const ok = await copyToClipboard(msg.text);
                // optional: tiny feedback without adding state
                if (!ok) alert("Copy failed (clipboard permission).");
              }}
              title="Copy this prompt"
              aria-label="Copy prompt"
            >
              Copy prompt
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function Pill({ label, dark }) {
  if (!label) return null;
  return (
    <span
      className={
        "inline-flex items-center rounded-full px-2 py-0.5 " +
        (dark ? "bg-black/20 text-white/90" : "bg-background/60 text-foreground border")
      }
    >
      {label}
    </span>
  );
}