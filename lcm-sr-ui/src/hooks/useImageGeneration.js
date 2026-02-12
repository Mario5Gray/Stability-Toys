// src/hooks/useImageGeneration.js

import { useCallback, useRef, useEffect, useState } from 'react';
import { createApiClient, createApiConfig } from '../utils/api';
import { createCache, generateCacheKey } from '../utils/cache';
import { eightDigitSeed, clampInt, safeJsonString, nowId } from '../utils/helpers';
import {
  STEPS_CONFIG,
  CFG_CONFIG,
  SR_CONFIG,
  SEED_CONFIG,
  SEED_MODES,
  UI_MESSAGES,
  MESSAGE_KINDS,
  MESSAGE_ROLES,
  ABORT_ERROR_NAME,
} from '../utils/constants';
import { jobQueue, PRIORITY } from '../lib/jobQueue';
import { generateViaWsWithRetry } from '../lib/generateRunnerWs';
import { createComfyRunnerWs } from '../lib/comfyRunnerWs';
import { wsClient } from '../lib/wsClient';

/**
 * Dream mode modifiers - stochastic variations for exploration.
 */
const DREAM_MODIFIERS = [
  'dramatic lighting', 'soft lighting', 'golden hour', 'rim light',
  'volumetric light', 'backlighting', 'studio lighting', 'natural light',
  'misty', 'foggy', 'hazy', 'atmospheric', 'ethereal', 'moody',
  'wide angle', 'telephoto', 'shallow depth of field', 'bokeh',
  'cinematic composition', 'rule of thirds', 'symmetrical', 'dynamic angle',
  'highly detailed', 'painterly', 'photorealistic', 'stylized',
  'film grain', 'vintage', 'modern', 'minimalist',
  'warm tones', 'cool tones', 'vibrant colors', 'muted colors',
  'monochromatic', 'high contrast', 'desaturated',
  'intricate details', 'sharp focus', 'soft focus', 'textured',
];

function dreamVariation(basePrompt, temperature = 0.3) {
  const base = basePrompt.trim();
  const numMods = Math.floor(Math.random() * (1 + temperature * 3)) + 1;

  const mods = [];
  const available = [...DREAM_MODIFIERS];
  for (let i = 0; i < numMods && available.length > 0; i++) {
    const idx = Math.floor(Math.random() * available.length);
    mods.push(available.splice(idx, 1)[0]);
  }

  return mods.length > 0 ? `${base}, ${mods.join(', ')}` : base;
}

function mutateParams(baseParams, temperature = 0.3) {
  const mutations = { ...baseParams };

  if (Math.random() < temperature) {
    const delta = Math.floor(baseParams.steps * 0.2 * (Math.random() - 0.5));
    mutations.steps = clampInt(
      baseParams.steps + delta,
      STEPS_CONFIG.MIN,
      STEPS_CONFIG.MAX
    );
  }

  if (Math.random() < temperature) {
    const delta = baseParams.cfg * 0.3 * (Math.random() - 0.5);
    mutations.cfg = Math.max(0, Math.min(CFG_CONFIG.MAX, baseParams.cfg + delta));
  }

  mutations.seed = eightDigitSeed();
  mutations.seedMode = SEED_MODES.FIXED;

  return mutations;
}

/**
 * Background hydration:
 * - We store meta-only immediately (server URL + key)
 * - Then we fetch bytes in the background, and cache.set(blob, mergeMetadata:true)
 * - Cache emits "hydrated" and we can swap the UI to a blob URL (fast + offline-ish)
 */
async function hydrateCacheEntry({ cache, cacheKey, serverImageUrl, signal }) {
  if (!cache || !cacheKey || !serverImageUrl) return;

  const res = await fetch(serverImageUrl, { signal });
  if (!res.ok) throw new Error(`hydrate fetch failed: ${res.status}`);
  const blob = await res.blob();

  // Merge metadata so we preserve serverImageUrl/serverImageKey/etc.
  await cache.set(cacheKey, blob, {}, { mergeMetadata: true });

  return { blobSize: blob.size };
}

/**
 * Hook for image generation and super-resolution operations.
 */
export function useImageGeneration(addMessage, updateMessage, setSelectedMsgId) {
  // API client and cache (created once)
  const apiClientRef = useRef(null);
  const cacheRef = useRef(null);

  // Comfy runner (created once)
  const comfyRunnerRef = useRef(null);
  if (!comfyRunnerRef.current) comfyRunnerRef.current = createComfyRunnerWs();

  // Track which messages correspond to which cache keys
  const cacheKeyToMsgIdsRef = useRef(new Map()); // cacheKey -> Set(msgId)
  const msgIdToCacheKeyRef = useRef(new Map()); // msgId -> cacheKey
  const inflightHydrationsRef = useRef(new Set()); // cacheKey -> hydration in progress
  const blobUrlByCacheKeyRef = useRef(new Map()); // cacheKey -> blobUrl (for cleanup)

  // Dream mode state
  const [isDreaming, setIsDreaming] = useState(false);
  const [dreamTemperature, setDreamTemperature] = useState(0.3);
  const [dreamInterval, setDreamInterval] = useState(5000);
  const [dreamMessageId, setDreamMessageId] = useState(null);
  const dreamTimerRef = useRef(null);
  const dreamParamsRef = useRef(null);
  const dreamHistoryByMsgIdRef = useRef(new Map());
  const dreamTemperatureRef = useRef(dreamTemperature);
  const dreamIntervalRef = useRef(dreamInterval);
  const dreamMessageIdRef = useRef(dreamMessageId);

  useEffect(() => {
    dreamTemperatureRef.current = dreamTemperature;
  }, [dreamTemperature]);

  useEffect(() => {
    dreamIntervalRef.current = dreamInterval;
  }, [dreamInterval]);

  useEffect(() => {
    dreamMessageIdRef.current = dreamMessageId;
  }, [dreamMessageId]);

  // Initialize cache and API client
  if (!cacheRef.current) cacheRef.current = createCache();
  if (!apiClientRef.current) {
    const config = createApiConfig();
    apiClientRef.current = createApiClient(config);
  }

  const cache = cacheRef.current;
  const api = apiClientRef.current;

  /**
   * Cleanup on unmount:
   * - api.cleanup
   * - dream interval
   * - blob URL revokes created by us
   * - cache event listener
   */
  useEffect(() => {
    const onHydrated = (e) => {
      const { key: cacheKey } = e.detail || {};
      if (!cacheKey) return;

      // Read the hydrated entry and swap message URLs to blob URL
      (async () => {
        try {
          const entry = await cache.get(cacheKey);
          if (!entry?.blob || entry.blob.size === 0) return;

          // Create / reuse blob URL for this cache key
          let blobUrl = blobUrlByCacheKeyRef.current.get(cacheKey);
          if (!blobUrl) {
            blobUrl = URL.createObjectURL(entry.blob);
            blobUrlByCacheKeyRef.current.set(cacheKey, blobUrl);
          }

          // Update all messages that reference this cacheKey to use blobUrl
          const msgIds = cacheKeyToMsgIdsRef.current.get(cacheKey);
          if (msgIds && msgIds.size > 0) {
            for (const msgId of msgIds) {
              updateMessage(msgId, {
                imageUrl: blobUrl,
                // keep serverImageUrl/serverImageKey in message for persistence/debug
              });
            }
          }
        } catch (err) {
          console.warn('[Cache] hydrated handler failed:', err);
        } finally {
          inflightHydrationsRef.current.delete(cacheKey);
        }
      })();
    };

    // Attach cache event listener (if supported)
    if (cache?.addEventListener) {
      cache.addEventListener('hydrated', onHydrated);
    }

    return () => {
      api.cleanup();
      if (dreamTimerRef.current) clearInterval(dreamTimerRef.current);

      // Revoke any blob URLs we created
      for (const url of blobUrlByCacheKeyRef.current.values()) {
        try { URL.revokeObjectURL(url); } catch {}
      }
      blobUrlByCacheKeyRef.current.clear();

      if (cache?.removeEventListener) {
        cache.removeEventListener('hydrated', onHydrated);
      }
    };
  }, [api, cache, updateMessage]);

  /**
   * Schedule hydration in the background (deduped).
   */
  const scheduleHydration = useCallback((cacheKey, serverImageUrl) => {
    if (!cache || !cacheKey || !serverImageUrl) return;
    if (inflightHydrationsRef.current.has(cacheKey)) return;

    inflightHydrationsRef.current.add(cacheKey);

    jobQueue.enqueue({
      priority: PRIORITY.BACKGROUND,
      source: 'cache-hydrate',
      payload: { cacheKey, serverImageUrl },
      meta: {},
      runner: async (payload, signal) => {
        try {
          await hydrateCacheEntry({
            cache,
            cacheKey: payload.cacheKey,
            serverImageUrl: payload.serverImageUrl,
            signal,
          });
        } catch (err) {
          // Don't poison; just allow retry later
          inflightHydrationsRef.current.delete(payload.cacheKey);
          console.warn('[Cache] hydrate failed:', err);
        }
      },
    });
  }, [cache]);

  /**
   * Remember that msgId corresponds to cacheKey (so we can swap URL on hydrate).
   */
  const linkMsgToCacheKey = useCallback((msgId, cacheKey) => {
    if (!msgId || !cacheKey) return;
    const prevKey = msgIdToCacheKeyRef.current.get(msgId);
    if (prevKey && prevKey !== cacheKey) {
      const prevSet = cacheKeyToMsgIdsRef.current.get(prevKey);
      if (prevSet) {
        prevSet.delete(msgId);
        if (prevSet.size === 0) cacheKeyToMsgIdsRef.current.delete(prevKey);
      }
    }
    let set = cacheKeyToMsgIdsRef.current.get(cacheKey);
    if (!set) {
      set = new Set();
      cacheKeyToMsgIdsRef.current.set(cacheKey, set);
    }
    set.add(msgId);
    msgIdToCacheKeyRef.current.set(msgId, cacheKey);
  }, []);

  /**
   * Generate an image with the specified parameters.
   * Enqueues onto the job queue instead of calling API directly.
   */
  const runGenerate = useCallback(
    (params) => {
      const {
        prompt: promptParam,
        size: sizeParam,
        steps: stepsParam,
        cfg: cfgParam,
        superresLevel: srLevelParam,
        seedMode: seedModeParam,
        seed: seedParam,
        targetMessageId,
        skipAutoSelect = false,
        isDream = false,
      } = params;

      const p = safeJsonString(promptParam).trim();
      if (!p) return;

      const useSize = sizeParam;
      const useSteps = clampInt(Number(stepsParam), STEPS_CONFIG.MIN, STEPS_CONFIG.MAX);
      const useCfg = Math.max(0, Math.min(CFG_CONFIG.ABSOLUTE_MAX, Number(cfgParam) || 0));
      const useSrLevel = clampInt(Number(srLevelParam), SR_CONFIG.MIN, SR_CONFIG.BACKEND_MAX);
      const useSeedMode = seedModeParam;
      const useSeedValue = seedParam;

      const reqSeed =
        useSeedMode === SEED_MODES.RANDOM
          ? eightDigitSeed()
          : clampInt(parseInt(String(useSeedValue ?? '0'), 10), 0, SEED_CONFIG.MAX);

      const superresOn = useSrLevel > 0;
      const assistantId = targetMessageId ?? nowId();

      if (targetMessageId) {
        updateMessage(targetMessageId, { isRegenerating: true, text: null });
      } else {
        addMessage({
          id: assistantId,
          role: MESSAGE_ROLES.ASSISTANT,
          kind: MESSAGE_KINDS.PENDING,
          text: null,
          meta: {
            request: {
              apiBase: '(pending)',
              endpoint: '/generate',
              size: useSize,
              steps: useSteps,
              cfg: useCfg,
              seed: reqSeed,
              superres: superresOn,
              superres_magnitude: superresOn ? useSrLevel : 1,
            },
          },
          ts: Date.now(),
        });
      }

      const apiRef = api;

      const jobId = jobQueue.enqueue({
        priority: isDream ? PRIORITY.BACKGROUND : PRIORITY.NORMAL,
        source: isDream ? 'dream' : 'generate',
        payload: {
          prompt: p,
          size: useSize,
          steps: useSteps,
          cfg: useCfg,
          seed: reqSeed,
          superres: superresOn,
          superresLevel: useSrLevel,
          assistantId,
          targetMessageId,
          skipAutoSelect,
        },
        meta: {},
        runner: async (payload, signal) => {
          const cacheParams = {
            prompt: payload.prompt,
            size: payload.size,
            steps: payload.steps,
            cfg: payload.cfg,
            seed: payload.seed,
            superres: payload.superres,
            superresLevel: payload.superresLevel,
          };

          const cacheKey = generateCacheKey(cacheParams);
          linkMsgToCacheKey(payload.assistantId, cacheKey);

          let result = null;

          if (wsClient.connected) {
            // 1) Try cache first
            if (cache) {
              const cached = await cache.get(cacheKey);
              if (cached) {
                // Prefer blob URL (fast), otherwise server URL (instant), otherwise nothing
                const blobUrl =
                  cached.blob?.size > 0 ? URL.createObjectURL(cached.blob) : null;

                const serverUrl = cached.metadata?.serverImageUrl || null;

                if (blobUrl || serverUrl) {
                  result = {
                    imageUrl: blobUrl || serverUrl,
                    serverImageUrl: serverUrl,
                    serverImageKey: cached.metadata?.serverImageKey || null,
                    metadata: cached.metadata || {},
                    fromCache: true,
                  };
                }

                // If meta-only, schedule hydration
                if ((!cached.blob || cached.blob.size === 0) && serverUrl) {
                  scheduleHydration(cacheKey, serverUrl);
                }
              }
            }

            // 2) If not in cache, generate via WS
            if (!result) {
              result = await generateViaWsWithRetry(payload, signal);

              // 3) Store meta-only immediately, then hydrate in background
              if (cache && result?.serverImageUrl) {
                // better: call cache.setMetaOnly if present, fallback to set
                if (typeof cache.setMetaOnly === 'function') {
                  cache.setMetaOnly(cacheKey, {
                    ...result.metadata,
                    serverImageUrl: result.serverImageUrl,
                    serverImageKey: result.serverImageKey,
                  });
                } else {
                  cache.set(cacheKey, new Blob([]), {
                    ...result.metadata,
                    serverImageUrl: result.serverImageUrl,
                    serverImageKey: result.serverImageKey,
                  });
                }

                scheduleHydration(cacheKey, result.serverImageUrl);
              }
            }
          } else {
            // HTTP fallback — api.generate handles its own caching
            result = await apiRef.generate(cacheParams, payload.assistantId);
          }

          // --- Update message with result (server URL immediately, blob will swap later on "hydrated") ---
          const msgUpdate = {
            kind: MESSAGE_KINDS.IMAGE,
            isRegenerating: false,
            hasError: false,
            errorText: null,
            text: null,
            imageUrl: result?.imageUrl || null,
            serverImageUrl: result?.serverImageUrl || null,
            serverImageKey: result?.serverImageKey || null,
            params: {
              prompt: payload.prompt,
              size: payload.size,
              steps: payload.steps,
              cfg: payload.cfg,
              seedMode: SEED_MODES.FIXED,
              seed: result?.metadata?.seed ?? payload.seed,
              superresLevel: payload.superresLevel,
            },
            meta: {
              backend: result?.metadata?.backend,
              apiBase: result?.metadata?.apiBase,
              superres: result?.metadata?.superres,
              srScale: result?.metadata?.srScale,
              cacheKey,
            },
            
          };

          // Dream history accumulation: push each result into the ref and attach to message
          if (isDream) {
            const historyEntry = {
              imageUrl: result?.imageUrl || null,
              serverImageUrl: result?.serverImageUrl || null,
              serverImageKey: result?.serverImageKey || null,
              params: msgUpdate.params,
              meta: msgUpdate.meta,
            };
            const existingHistory =
              dreamHistoryByMsgIdRef.current.get(payload.assistantId) || [];
            const nextHistory = [...existingHistory, historyEntry];
            dreamHistoryByMsgIdRef.current.set(payload.assistantId, nextHistory);
            msgUpdate.imageHistory = nextHistory;
            msgUpdate.historyIndex = Math.max(0, nextHistory.length - 1);
          }

          updateMessage(payload.assistantId, msgUpdate);

          if (!payload.skipAutoSelect) setSelectedMsgId(payload.assistantId);

          return {
            imageKey: result?.serverImageKey || null,
            cacheKey,
            dimensions: payload.size,
          };
        },
      });

      // Per-job error handling (unchanged)
      const onError = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);

        const err = e.detail.error;
        const errMsg =
          err?.name === ABORT_ERROR_NAME ? UI_MESSAGES.CANCELED : err?.message || String(err);

        const savedParams = {
          prompt: promptParam,
          size: sizeParam,
          steps: stepsParam,
          cfg: cfgParam,
          superresLevel: srLevelParam,
          seedMode: seedModeParam,
          seed: seedParam,
          targetMessageId: assistantId,
        };

        updateMessage(assistantId, {
          kind: targetMessageId ? MESSAGE_KINDS.IMAGE : MESSAGE_KINDS.ERROR,
          isRegenerating: false,
          hasError: true,
          errorText: errMsg,
          text: targetMessageId ? null : errMsg,
          retryParams: savedParams,
        });
      };

      const onCancel = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);

        updateMessage(assistantId, {
          kind: targetMessageId ? MESSAGE_KINDS.IMAGE : MESSAGE_KINDS.ERROR,
          isRegenerating: false,
          hasError: true,
          errorText: UI_MESSAGES.CANCELED,
          text: targetMessageId ? null : UI_MESSAGES.CANCELED,
        });
      };

      const onComplete = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);
        jobQueue.removeEventListener('complete', onComplete);
      };

      jobQueue.addEventListener('error', onError);
      jobQueue.addEventListener('cancel', onCancel);
      jobQueue.addEventListener('complete', onComplete);

      return assistantId;
    },
    [api, cache, addMessage, updateMessage, setSelectedMsgId, scheduleHydration, linkMsgToCacheKey]
  );

  /**
   * Run a ComfyUI workflow through the job queue.
   * If targetMessageId is provided, updates that message and appends to its imageHistory.
   * Otherwise creates a new pending message.
   * Optional: skipAutoSelect to avoid UI focus changes on completion.
   */
  const runComfy = useCallback(
    ({ workflowId, params, inputImageFile, targetMessageId, existingHistory, skipAutoSelect }) => {
      const assistantId = targetMessageId ?? nowId();

      if (targetMessageId) {
        updateMessage(targetMessageId, { isRegenerating: true });
      } else {
        addMessage({
          id: assistantId,
          role: MESSAGE_ROLES.ASSISTANT,
          kind: MESSAGE_KINDS.PENDING,
          text: null,
          meta: { backend: 'comfy' },
          ts: Date.now(),
        });
      }

      const comfyRunner = comfyRunnerRef.current;
      // Snapshot current history so concurrent jobs don't clobber each other
      const priorHistory = existingHistory ? [...existingHistory] : [];

      const jobId = jobQueue.enqueue({
        priority: PRIORITY.NORMAL,
        source: 'comfy',
        payload: {
          workflowId,
          params,
          inputImageFile,
          assistantId,
          targetMessageId,
          priorHistory,
          skipAutoSelect: !!skipAutoSelect,
        },
        meta: {},
        runner: async (payload, signal) => {
          const result = await comfyRunner(
            {
              workflowId: payload.workflowId,
              params: payload.params,
              inputImageFile: payload.inputImageFile,
            },
            signal
          );

          const firstOutput = result?.outputs?.[0];
          const newUrl = firstOutput?.url || null;
          const entryParams = { ...payload.params, workflowId: payload.workflowId };
          const entryMeta = { backend: `comfy:${payload.workflowId}` };

          const historyEntry = {
            imageUrl: newUrl,
            serverImageUrl: newUrl,
            params: entryParams,
            meta: entryMeta,
          };

          const newHistory = [...payload.priorHistory, historyEntry];

          updateMessage(payload.assistantId, (msg) => {
            const currentHistory = msg?.imageHistory || [];
            const mergedHistory =
              currentHistory.length > payload.priorHistory.length
                ? [...currentHistory, historyEntry]
                : newHistory;
            const nextParams =
              payload.targetMessageId && msg?.params ? msg.params : entryParams;
            return {
              ...msg,
              kind: MESSAGE_KINDS.IMAGE,
              isRegenerating: false,
              imageUrl: newUrl,
              params: nextParams,
              meta: entryMeta,
              imageHistory: mergedHistory,
              historyIndex: mergedHistory.length - 1,
            };
          });

          if (!payload.skipAutoSelect) {
            setSelectedMsgId(payload.assistantId);
          }
          return result;
        },
      });

      // Error / cancel handlers
      const onError = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);

        const err = e.detail.error;
        const errMsg =
          err?.name === ABORT_ERROR_NAME ? UI_MESSAGES.CANCELED : err?.message || String(err);

        updateMessage(assistantId, {
          kind: targetMessageId ? MESSAGE_KINDS.IMAGE : MESSAGE_KINDS.ERROR,
          isRegenerating: false,
          hasError: true,
          errorText: errMsg,
          text: targetMessageId ? null : errMsg,
        });
      };

      const onCancel = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);

        updateMessage(assistantId, {
          kind: targetMessageId ? MESSAGE_KINDS.IMAGE : MESSAGE_KINDS.ERROR,
          isRegenerating: false,
          hasError: true,
          errorText: UI_MESSAGES.CANCELED,
          text: targetMessageId ? null : UI_MESSAGES.CANCELED,
        });
      };

      const onComplete = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);
        jobQueue.removeEventListener('complete', onComplete);
      };

      jobQueue.addEventListener('error', onError);
      jobQueue.addEventListener('cancel', onCancel);
      jobQueue.addEventListener('complete', onComplete);

      return assistantId;
    },
    [addMessage, updateMessage, setSelectedMsgId]
  );

  /**
   * Upload and super-resolve an image. (left as-is)
   */
  const runSuperResUpload = useCallback(
    async (file, magnitude) => {
      if (!file) return;

      const assistantId = nowId();

      const userMsg = {
        id: nowId(),
        role: MESSAGE_ROLES.USER,
        kind: MESSAGE_KINDS.TEXT,
        text: `Super-res upload: ${file.name} (magnitude ${magnitude})`,
        meta: { ingest: 'superres', filename: file.name, magnitude },
        ts: Date.now(),
      };

      const pendingMsg = {
        id: assistantId,
        role: MESSAGE_ROLES.ASSISTANT,
        kind: MESSAGE_KINDS.PENDING,
        text: UI_MESSAGES.SUPER_RESOLVING,
        meta: { request: { endpoint: '/superres', magnitude } },
        ts: Date.now(),
      };

      addMessage([userMsg, pendingMsg]);

      try {
        const result = await api.superResolve({ file, magnitude }, assistantId);

        updateMessage(assistantId, {
          kind: MESSAGE_KINDS.IMAGE,
          text: `Done (SR upload x${result.metadata.passes}).`,
          imageUrl: result.imageUrl,
          meta: {
            backend: result.metadata.backend,
            apiBase: result.metadata.apiBase,
            superres: true,
            srScale: result.metadata.scale,
          },
        });
      } catch (err) {
        const msg =
          err?.name === ABORT_ERROR_NAME ? UI_MESSAGES.CANCELED : err?.message || String(err);
        updateMessage(assistantId, { kind: MESSAGE_KINDS.ERROR, text: msg });
      }
    },
    [api, addMessage, updateMessage]
  );

  /**
   * Start dream mode.
   */
  const runDreamCycle = useCallback(
    (targetId = null) => {
      const baseParams = dreamParamsRef.current;
      if (!baseParams) return null;
      const temp = dreamTemperatureRef.current;
      const nextParams = mutateParams(baseParams, temp);
      nextParams.prompt = dreamVariation(baseParams.prompt, temp);
      nextParams.skipAutoSelect = true;
      nextParams.isDream = true;
      if (targetId) nextParams.targetMessageId = targetId;
      return runGenerate(nextParams);
    },
    [runGenerate]
  );

  const restartDreamInterval = useCallback(() => {
    if (dreamTimerRef.current) clearInterval(dreamTimerRef.current);
    dreamTimerRef.current = setInterval(() => {
      runDreamCycle(dreamMessageIdRef.current);
    }, dreamIntervalRef.current);
  }, [runDreamCycle]);

  const startDreaming = useCallback(
    (baseParams) => {
      if (dreamTimerRef.current) clearInterval(dreamTimerRef.current);

      dreamParamsRef.current = { ...baseParams };
      setIsDreaming(true);

      // First dream: create a fresh message
      const firstId = runDreamCycle(null);
      setDreamMessageId(firstId);
      dreamMessageIdRef.current = firstId;

      // Subsequent dreams paint into the same message
      restartDreamInterval();
    },
    [runDreamCycle, restartDreamInterval]
  );

  const stopDreaming = useCallback(() => {
    if (dreamTimerRef.current) {
      clearInterval(dreamTimerRef.current);
      dreamTimerRef.current = null;
    }
    setIsDreaming(false);
    setDreamMessageId(null);
    dreamParamsRef.current = null;
  }, []);

  const guideDream = useCallback((newBaseParams) => {
    if (!isDreaming) return;
    dreamParamsRef.current = { ...newBaseParams };
  }, [isDreaming]);

  /**
   * Save current dream bubble and start a fresh one on the next tick.
   */
  const saveDreamAndContinue = useCallback(() => {
    if (!isDreaming) return;
    setDreamMessageId(null);
    dreamMessageIdRef.current = null;

    // Restart the interval so the next tick creates a fresh message
    if (dreamTimerRef.current) clearInterval(dreamTimerRef.current);
    const baseParams = dreamParamsRef.current;
    if (!baseParams) return;

    // Immediately fire one to create the new message
    const newId = runDreamCycle(null);
    setDreamMessageId(newId);
    if (newId) dreamHistoryByMsgIdRef.current.set(newId, []);
    dreamMessageIdRef.current = newId;

    // Now fix the interval to paint into the new message
    restartDreamInterval();
  }, [isDreaming, runDreamCycle, restartDreamInterval]);

  useEffect(() => {
    if (!isDreaming) return;
    restartDreamInterval();
  }, [isDreaming, dreamInterval, restartDreamInterval]);

  /**
   * Dream history navigation: go to previous image.
   */
  const getImageFromCache = useCallback(async (params) => {
    if (!cache) return null;
    try {
      const key = generateCacheKey(params);
      const entry = await cache.get(key);

      const serverUrl = entry?.metadata?.serverImageUrl;
      if (serverUrl) {
        // Return immediately for display; hydration keeps blob cache warm
        scheduleHydration(key, serverUrl);
        return serverUrl;
      }

      if (entry?.blob && entry.blob.size > 0) {
        return URL.createObjectURL(entry.blob);
      }
    } catch (err) {
      console.warn('[Cache] getImageFromCache failed:', err);
    }
    return null;
  }, [cache, scheduleHydration]);

  const dreamHistoryPrev = useCallback(async (msg) => {
    if (!msg.imageHistory?.length) return;
    const newIdx = Math.max(0, (msg.historyIndex ?? 0) - 1);
    const entry = msg.imageHistory[newIdx];
    let imageUrl = entry.imageUrl || entry.serverImageUrl || null;
    if (!imageUrl && entry?.params) {
      imageUrl = await getImageFromCache(entry.params);
    }
    updateMessage(msg.id, {
      imageUrl,
      historyIndex: newIdx,
      serverImageUrl: entry.serverImageUrl,
      serverImageKey: entry.serverImageKey,
      needsReload: !imageUrl,
    });
  }, [updateMessage, getImageFromCache]);

  /**
   * Dream history navigation: go to next image.
   */
  const dreamHistoryNext = useCallback(async (msg) => {
    if (!msg.imageHistory?.length) return;
    const maxIdx = msg.imageHistory.length - 1;
    const newIdx = Math.min(maxIdx, (msg.historyIndex ?? 0) + 1);
    const entry = msg.imageHistory[newIdx];
    let imageUrl = entry.imageUrl || entry.serverImageUrl || null;
    if (!imageUrl && entry?.params) {
      imageUrl = await getImageFromCache(entry.params);
    }
    updateMessage(msg.id, {
      imageUrl,
      historyIndex: newIdx,
      serverImageUrl: entry.serverImageUrl,
      serverImageKey: entry.serverImageKey,
      needsReload: !imageUrl,
    });
  }, [updateMessage, getImageFromCache]);

  /**
   * Dream history navigation: jump to the latest (most recent) image.
   */
  const dreamHistoryLive = useCallback(async (msg) => {
    if (!msg.imageHistory?.length) return;
    const lastIdx = msg.imageHistory.length - 1;
    const entry = msg.imageHistory[lastIdx];
    let imageUrl = entry.imageUrl || entry.serverImageUrl || null;
    if (!imageUrl && entry?.params) {
      imageUrl = await getImageFromCache(entry.params);
    }
    updateMessage(msg.id, {
      imageUrl,
      historyIndex: lastIdx,
      serverImageUrl: entry.serverImageUrl,
      serverImageKey: entry.serverImageKey,
      needsReload: !imageUrl,
    });
  }, [updateMessage, getImageFromCache]);

  const cancelRequest = useCallback((id) => api.cancel(id), [api]);

  const cancelAll = useCallback(() => {
    api.cancelAll();
    stopDreaming();
  }, [api, stopDreaming]);

  const serverLabel =
    api.config.bases.length > 0
      ? `RR (${api.config.bases.length} backends)`
      : api.config.single || '(same origin)';

  /**
   * Get an image from cache by params.
   * - If blob is present: returns blob URL
   * - If meta-only: schedules hydration + returns null
   */
  const getCacheStats = useCallback(async () => {
    if (!cache) return { enabled: false };
    return await cache.stats();
  }, [cache]);

  const clearCache = useCallback(async () => {
    if (cache) {
      await cache.clear();
      console.log('[Cache] Cleared');
    }
  }, [cache]);

  const cleanupMessage = useCallback((msgId) => {
    if (!msgId) return;
    dreamHistoryByMsgIdRef.current.delete(msgId);
    msgIdToCacheKeyRef.current.delete(msgId);
    for (const [cacheKey, msgIds] of cacheKeyToMsgIdsRef.current.entries()) {
      if (msgIds.delete(msgId) && msgIds.size === 0) {
        cacheKeyToMsgIdsRef.current.delete(cacheKey);
      }
    }
  }, []);

  return {
    // Generation
    runGenerate,
    runComfy,
    runSuperResUpload,

    // Cancellation
    cancelRequest,
    cancelAll,

    // Dream mode
    isDreaming,
    startDreaming,
    stopDreaming,
    guideDream,
    dreamTemperature,
    setDreamTemperature,
    dreamInterval,
    setDreamInterval,
    dreamMessageId,
    saveDreamAndContinue,
    dreamHistoryPrev,
    dreamHistoryNext,
    dreamHistoryLive,

    // Cache
    getImageFromCache,
    getCacheStats,
    clearCache,
    cleanupMessage,

    // State
    inflightCount: api.inflightCount,
    serverLabel,
  };
}
