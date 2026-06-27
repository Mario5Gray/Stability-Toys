// src/hooks/useDropIngest.js
import { useCallback } from "react";
import { uuidv4 } from "@/utils/uuid";

/**
 * Minimal constants fallback (in case you don't export these)
 * If you have MESSAGE_KINDS/MESSAGE_ROLES, pass them in config or swap imports.
 */
const ROLE_ASSISTANT = "assistant";
const KIND_IMAGE = "image";

/**
 * Build params for your selection pipeline.
 * IMPORTANT: do NOT set "undefined" strings; omit keys when unknown.
 */
function buildParamsFromMeta(meta) {
  const p = {};
  if (meta?.prompt) p.prompt = String(meta.prompt);

  if (meta?.size && /^\d+x\d+$/i.test(String(meta.size))) {
    p.size = String(meta.size);
  }

  if (Number.isFinite(meta?.steps)) p.steps = Number(meta.steps);
  if (Number.isFinite(meta?.cfg)) p.cfg = Number(meta.cfg);

  // seed -> fixed mode
  if (meta?.seed != null && Number.isFinite(Number(meta.seed))) {
    p.seedMode = "fixed";
    p.seed = Number(meta.seed);
  }

  if (Number.isFinite(meta?.superresLevel)) p.superresLevel = Number(meta.superresLevel);

  return p;
}

/**
 * Parse A1111-style "parameters" block if present.
 * Typical:
 *   <prompt>
 *   Negative prompt: <neg>
 *   Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123, Size: 512x512, ...
 */
function parseA1111ParametersBlock(text) {
  if (!text || typeof text !== "string") return null;

  const out = {};
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);

  // Prompt: everything until "Negative prompt:" or "Steps:"
  let promptLines = [];
  let i = 0;
  for (; i < lines.length; i++) {
    const l = lines[i];
    if (/^negative prompt\s*:/i.test(l) || /^steps\s*:/i.test(l)) break;
    promptLines.push(l);
  }
  const prompt = promptLines.join("\n").trim();
  if (prompt) out.prompt = prompt;

  // Negative prompt
  let neg = "";
  for (; i < lines.length; i++) {
    const l = lines[i];
    if (/^negative prompt\s*:/i.test(l)) {
      neg = l.replace(/^negative prompt\s*:\s*/i, "");
      // Sometimes neg spans multiple lines until "Steps:"
      for (let j = i + 1; j < lines.length; j++) {
        if (/^steps\s*:/i.test(lines[j])) {
          i = j - 1;
          break;
        }
        neg += (neg ? "\n" : "") + lines[j];
        i = j;
      }
      break;
    }
    if (/^steps\s*:/i.test(l)) break;
  }
  if (neg) out.negative = neg;

  // Key-value tail (often on "Steps:" line)
  const tailLine = lines.find((l) => /^steps\s*:/i.test(l)) || "";
  const tail = tailLine.replace(/^steps\s*:\s*/i, "Steps: ").trim();
  const kvText = tail ? tail : lines[lines.length - 1] || "";

  // Parse comma-separated key: value
  const parts = kvText.split(",").map((p) => p.trim()).filter(Boolean);
  for (const part of parts) {
    const m = part.match(/^([^:]+)\s*:\s*(.+)$/);
    if (!m) continue;
    const key = m[1].trim().toLowerCase();
    const val = m[2].trim();

    if (key === "steps") {
      const n = Number(val);
      if (Number.isFinite(n)) out.steps = n;
    } else if (key === "cfg scale" || key === "cfg") {
      const n = Number(val);
      if (Number.isFinite(n)) out.cfg = n;
    } else if (key === "seed") {
      const n = Number(val);
      if (Number.isFinite(n)) out.seed = n;
    } else if (key === "size") {
      if (/^\d+x\d+$/i.test(val)) out.size = val;
    } else if (key === "denoising strength" || key === "denoise") {
      const n = Number(val);
      if (Number.isFinite(n)) out.denoise = n;
    }
  }

  return out;
}

/**
 * Parse PNG text chunks (tEXt, iTXt, zTXt(best-effort, no inflate))
 * Returns map: { key: value }
 */
function parsePngTextChunks(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer);

  // PNG signature
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < sig.length; i++) {
    if (u8[i] !== sig[i]) return {};
  }

  let off = 8;
  const out = {};

  const readU32 = (o) =>
    (u8[o] << 24) | (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3];

  const readType = (o) =>
    String.fromCharCode(u8[o], u8[o + 1], u8[o + 2], u8[o + 3]);

  const readLatin1 = (start, end) =>
    String.fromCharCode(...u8.slice(start, end));

  const readUtf8 = (start, end) => {
    try {
      return new TextDecoder("utf-8", { fatal: false }).decode(u8.slice(start, end));
    } catch {
      return readLatin1(start, end);
    }
  };

  while (off + 8 < u8.length) {
    const len = readU32(off);
    const type = readType(off + 4);
    const dataStart = off + 8;
    const dataEnd = dataStart + len;
    // CRC is 4 bytes after dataEnd; ignore
    off = dataEnd + 4;

    if (dataEnd > u8.length) break;

    if (type === "tEXt") {
      // keyword\0text (latin1)
      const nul = u8.indexOf(0, dataStart);
      if (nul > -1 && nul < dataEnd) {
        const k = readLatin1(dataStart, nul);
        const v = readLatin1(nul + 1, dataEnd);
        if (k) out[k] = v;
      }
    } else if (type === "iTXt") {
      // keyword\0 compressionFlag compressionMethod languageTag\0 translatedKeyword\0 text(utf8)
      let p = dataStart;

      const nul1 = u8.indexOf(0, p);
      if (nul1 < 0 || nul1 >= dataEnd) continue;
      const keyword = readLatin1(p, nul1);
      p = nul1 + 1;

      const compressionFlag = u8[p]; p += 1;
      const compressionMethod = u8[p]; p += 1;

      const nul2 = u8.indexOf(0, p);
      if (nul2 < 0 || nul2 >= dataEnd) continue;
      // languageTag ignored
      p = nul2 + 1;

      const nul3 = u8.indexOf(0, p);
      if (nul3 < 0 || nul3 >= dataEnd) continue;
      // translatedKeyword ignored
      p = nul3 + 1;

      // If compressed, we skip (no inflate here)
      if (compressionFlag === 1) {
        // best effort: store marker
        if (keyword) out[keyword] = "[compressed iTXt not decoded]";
        continue;
      }

      const text = readUtf8(p, dataEnd);
      if (keyword) out[keyword] = text;
    } else if (type === "zTXt") {
      // keyword\0 compressionMethod compressedText
      // Without inflate dependency, we can't decode; just note it.
      const nul = u8.indexOf(0, dataStart);
      if (nul > -1 && nul + 2 <= dataEnd) {
        const k = readLatin1(dataStart, nul);
        if (k) out[k] = "[compressed zTXt not decoded]";
      }
    }

    if (type === "IEND") break;
  }

  return out;
}

/**
 * Try to interpret common metadata shapes produced by SD tools.
 */
function coerceMeta(textMap) {
  const meta = { raw: textMap };

  // Common keys seen in SD pipelines
  // - "parameters" (A1111)
  // - "prompt" / "negative_prompt"
  // - "workflow" / "comfy" etc (we can just keep raw)
  const paramsBlock =
    textMap?.parameters ||
    textMap?.Parameters ||
    textMap?.PARAMETERS ||
    null;

  if (paramsBlock && typeof paramsBlock === "string") {
    const parsed = parseA1111ParametersBlock(paramsBlock);
    if (parsed) Object.assign(meta, parsed);
  }

  // Direct fields (best-effort)
  if (textMap?.prompt) meta.prompt = meta.prompt ?? textMap.prompt;
  if (textMap?.Prompt) meta.prompt = meta.prompt ?? textMap.Prompt;

  if (textMap?.negative_prompt) meta.negative = meta.negative ?? textMap.negative_prompt;
  if (textMap?.NegativePrompt) meta.negative = meta.negative ?? textMap.NegativePrompt;

  // If some pipeline stores JSON in a known key, try it
  const jsonish = textMap?.lcm || textMap?.LCM || textMap?.generation || null;
  if (jsonish && typeof jsonish === "string") {
    try {
      const obj = JSON.parse(jsonish);
      if (obj && typeof obj === "object") meta.json = obj;
    } catch {
      // ignore
    }
  }

  // Pull from meta.json if it matches your schema
  const j = meta.json;
  if (j) {
    if (j.prompt && !meta.prompt) meta.prompt = j.prompt;
    if (j.size && !meta.size) meta.size = j.size;
    if (Number.isFinite(j.steps) && !Number.isFinite(meta.steps)) meta.steps = j.steps;
    if (Number.isFinite(j.cfg) && !Number.isFinite(meta.cfg)) meta.cfg = j.cfg;
    if (Number.isFinite(j.seed) && meta.seed == null) meta.seed = j.seed;
    if (Number.isFinite(j.superresLevel) && !Number.isFinite(meta.superresLevel)) meta.superresLevel = j.superresLevel;
  }

  return meta;
}

/**
 * Hook: ingest dropped files into chat.
 *
 * @param {object} cfg
 * @param {(msgOrMsgs:any)=>void} cfg.addMessage
 * @param {(id:string)=>void} cfg.setSelectedMsgId
 * @param {(fileOrNull:any)=>void} [cfg.setUploadFile]
 * @param {(file:File)=>void|Promise<void>} [cfg.onInitImageSelect] - promotes the dropped file to the active img2img init image
 */
export function useDropIngest({ addMessage, setSelectedMsgId, setUploadFile, onInitImageSelect }) {
  const ingestFiles = useCallback(
    async (files) => {
      if (!files?.length) return;

      for (const file of files) {
        // only png for now
        const isPng =
          file.type === "image/png" ||
          /\.png$/i.test(file.name || "");

        if (!isPng) continue;

        // Create a blob URL for display
        const imageUrl = URL.createObjectURL(file);

        // Extract metadata (best-effort)
        let textMap = {};
        try {
          const buf = await file.arrayBuffer();
          textMap = parsePngTextChunks(buf);
        } catch {
          textMap = {};
        }
        const meta = coerceMeta(textMap);
        const params = buildParamsFromMeta(meta);
        console.log(textMap);
        const id = uuidv4();

        addMessage({
          id,
          role: ROLE_ASSISTANT,
          kind: KIND_IMAGE,
          imageUrl,
          params,
          meta: {
            backend: "lcm:drop",
            source: "dragdrop",
            filename: file.name,
            mime: file.type,
            // Keep raw metadata available for debugging / later parsing improvements
            pngText: textMap,
            negative: meta.negative || undefined,
          },
          ts: Date.now(),
        });

        // Selecting the message is the "event" that populates OptionsPanel
        setSelectedMsgId?.(id);

        // Ensure selected image wins over uploadFile in your inputImage memo
        setUploadFile?.(null);

        // Arm img2img: promote the dropped file to the active init image so a
        // subsequent prompt edit regenerates from it instead of txt2img.
        await onInitImageSelect?.(file);
      }
    },
    [addMessage, setSelectedMsgId, setUploadFile, onInitImageSelect]
  );

  return { ingestFiles };
}