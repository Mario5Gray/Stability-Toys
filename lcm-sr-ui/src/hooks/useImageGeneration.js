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
import { generateViaWs } from '../lib/generateRunnerWs';
import { wsClient } from '../lib/wsClient';

/**
 * Dream mode modifiers - stochastic variations for exploration.
 */
const DREAM_MODIFIERS = [
  // Lighting
  'dramatic lighting', 'soft lighting', 'golden hour', 'rim light', 
  'volumetric light', 'backlighting', 'studio lighting', 'natural light',
  
  // Atmosphere
  'misty', 'foggy', 'hazy', 'atmospheric', 'ethereal', 'moody',
  
  // Camera/Composition
  'wide angle', 'telephoto', 'shallow depth of field', 'bokeh',
  'cinematic composition', 'rule of thirds', 'symmetrical', 'dynamic angle',
  
  // Style
  'highly detailed', 'painterly', 'photorealistic', 'stylized',
  'film grain', 'vintage', 'modern', 'minimalist',
  
  // Color
  'warm tones', 'cool tones', 'vibrant colors', 'muted colors',
  'monochromatic', 'high contrast', 'desaturated',
  
  // Detail
  'intricate details', 'sharp focus', 'soft focus', 'textured',
];

/**
 * Generate a dream variation of a prompt.
 * @param {string} basePrompt - Original prompt
 * @param {number} temperature - How wild (0-1, default 0.3)
 * @returns {string} Modified prompt with random additions
 */
function dreamVariation(basePrompt, temperature = 0.3) {
  const base = basePrompt.trim();
  
  // Number of modifiers scales with temperature
  const numMods = Math.floor(Math.random() * (1 + temperature * 3)) + 1;
  
  // Pick random modifiers
  const mods = [];
  const available = [...DREAM_MODIFIERS];
  for (let i = 0; i < numMods && available.length > 0; i++) {
    const idx = Math.floor(Math.random() * available.length);
    mods.push(available.splice(idx, 1)[0]);
  }
  
  return mods.length > 0 ? `${base}, ${mods.join(', ')}` : base;
}

/**
 * Generate parameter mutations for dream mode.
 * @param {object} baseParams - Base generation parameters
 * @param {number} temperature - Mutation strength (0-1)
 * @returns {object} Mutated parameters
 */
function mutateParams(baseParams, temperature = 0.3) {
  const mutations = { ...baseParams };
  
  // Randomly vary steps (±20%)
  if (Math.random() < temperature) {
    const delta = Math.floor(baseParams.steps * 0.2 * (Math.random() - 0.5));
    mutations.steps = clampInt(
      baseParams.steps + delta,
      STEPS_CONFIG.MIN,
      STEPS_CONFIG.MAX
    );
  }
  
  // Randomly vary CFG (±30%)
  if (Math.random() < temperature) {
    const delta = baseParams.cfg * 0.3 * (Math.random() - 0.5);
    mutations.cfg = Math.max(0, Math.min(CFG_CONFIG.MAX, baseParams.cfg + delta));
  }
  
  // New random seed always
  mutations.seed = eightDigitSeed();
  mutations.seedMode = SEED_MODES.FIXED;
  
  return mutations;
}

/**
 * Hook for image generation and super-resolution operations.
 * Includes "dream mode" for stochastic exploration.
 * 
 * @param {function} addMessage - Function to add messages to chat
 * @param {function} updateMessage - Function to update a message
 * @param {function} setSelectedMsgId - Function to set selected message ID
 * @returns {object} Generation functions and state
 */
export function useImageGeneration(addMessage, updateMessage, setSelectedMsgId) {
  // API client and cache (created once)
  const apiClientRef = useRef(null);
  const cacheRef = useRef(null);

  // Dream mode state
  const [isDreaming, setIsDreaming] = useState(false);
  const [dreamTemperature, setDreamTemperature] = useState(0.3); // 0-1
  const [dreamInterval, setDreamInterval] = useState(5000); // ms between dreams
  const [guideImage, setGuideImage] = useState(null); // { url, prompt } when user clicks "Guide Dream"
  const dreamTimerRef = useRef(null);
  const dreamParamsRef = useRef(null);
  
  // Initialize cache and API client
  if (!cacheRef.current) {
    cacheRef.current = createCache();
  }
  if (!apiClientRef.current) {
    const config = createApiConfig();
    apiClientRef.current = createApiClient(config);
  }

  const cache = cacheRef.current;
  const api = apiClientRef.current;

  /**
   * Cleanup blob URLs and abort requests on unmount.
   */
  useEffect(() => {
    return () => {
      api.cleanup();
      if (dreamTimerRef.current) {
        clearInterval(dreamTimerRef.current);
      }
    };
  }, [api]);

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

      // Validate prompt
      const p = safeJsonString(promptParam).trim();
      if (!p) return;

      // Resolve parameters with clamping (snapshot eagerly)
      const useSize = sizeParam;
      const useSteps = clampInt(Number(stepsParam), STEPS_CONFIG.MIN, STEPS_CONFIG.MAX);
      const useCfg = Math.max(0, Math.min(CFG_CONFIG.ABSOLUTE_MAX, Number(cfgParam) || 0));
      const useSrLevel = clampInt(
        Number(srLevelParam),
        SR_CONFIG.MIN,
        SR_CONFIG.BACKEND_MAX
      );
      const useSeedMode = seedModeParam;
      const useSeedValue = seedParam;

      const reqSeed =
        useSeedMode === SEED_MODES.RANDOM
          ? eightDigitSeed()
          : clampInt(parseInt(String(useSeedValue ?? '0'), 10), 0, SEED_CONFIG.MAX);

      const superresOn = useSrLevel > 0;

      // Determine if this is in-place regen or new generation
      const assistantId = targetMessageId ?? nowId();

      if (targetMessageId) {
        updateMessage(targetMessageId, {
          isRegenerating: true,
          text: null,
        });
      } else {
        const pendingMsg = {
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
        };
        addMessage(pendingMsg);
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
          const result = wsClient.connected
            ? await generateViaWs(payload, signal)
            : await apiRef.generate(
                {
                  prompt: payload.prompt,
                  size: payload.size,
                  steps: payload.steps,
                  cfg: payload.cfg,
                  seed: payload.seed,
                  superres: payload.superres,
                  superresLevel: payload.superresLevel,
                },
                payload.assistantId
              );

          // Update message with result
          updateMessage(payload.assistantId, {
            kind: MESSAGE_KINDS.IMAGE,
            isRegenerating: false,
            hasError: false,
            errorText: null,
            text: null,
            imageUrl: result.imageUrl,
            serverImageUrl: result.serverImageUrl || null,
            serverImageKey: result.serverImageKey || null,
            params: {
              prompt: payload.prompt,
              size: payload.size,
              steps: payload.steps,
              cfg: payload.cfg,
              seedMode: SEED_MODES.FIXED,
              seed: result.metadata.seed,
              superresLevel: payload.superresLevel,
            },
            meta: {
              backend: result.metadata.backend,
              apiBase: result.metadata.apiBase,
              superres: result.metadata.superres,
              srScale: result.metadata.srScale,
            },
          });

          if (!payload.skipAutoSelect) {
            setSelectedMsgId(payload.assistantId);
          }

          return {
            imageKey: result.serverImageKey || null,
            cacheKey: null,
            dimensions: payload.size,
          };
        },
      });

      // Listen for error on this specific job
      const onError = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);
        const err = e.detail.error;
        const errMsg =
          err?.name === ABORT_ERROR_NAME
            ? UI_MESSAGES.CANCELED
            : err?.message || String(err);
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
      jobQueue.addEventListener('error', onError);
      jobQueue.addEventListener('cancel', onCancel);

      // Clean up listeners on complete too
      const onComplete = (e) => {
        if (e.detail?.job?.id !== jobId) return;
        jobQueue.removeEventListener('error', onError);
        jobQueue.removeEventListener('cancel', onCancel);
        jobQueue.removeEventListener('complete', onComplete);
      };
      jobQueue.addEventListener('complete', onComplete);

      return assistantId;
    },
    [api, addMessage, updateMessage, setSelectedMsgId]
  );

  /**
   * Upload and super-resolve an image.
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
        meta: {
          ingest: 'superres',
          filename: file.name,
          magnitude,
        },
        ts: Date.now(),
      };

      const pendingMsg = {
        id: assistantId,
        role: MESSAGE_ROLES.ASSISTANT,
        kind: MESSAGE_KINDS.PENDING,
        text: UI_MESSAGES.SUPER_RESOLVING,
        meta: {
          request: {
            endpoint: '/superres',
            magnitude,
          },
        },
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
          err?.name === ABORT_ERROR_NAME
            ? UI_MESSAGES.CANCELED
            : err?.message || String(err);
        updateMessage(assistantId, {
          kind: MESSAGE_KINDS.ERROR,
          text: msg,
        });
      }
    },
    [api, addMessage, updateMessage]
  );

  /**
   * Start dream mode - continuously generate variations.
   * @param {object} baseParams - Starting parameters
   */
  const startDreaming = useCallback(
    (baseParams) => {
      // Stop any existing dream
      if (dreamTimerRef.current) {
        clearInterval(dreamTimerRef.current);
      }

      // Store base params
      dreamParamsRef.current = { ...baseParams };
      setIsDreaming(true);

      // Generate first dream immediately (skip auto-select to not disturb user's view)
      const dreamParams = mutateParams(baseParams, dreamTemperature);
      dreamParams.prompt = dreamVariation(baseParams.prompt, dreamTemperature);
      dreamParams.skipAutoSelect = true;
      runGenerate(dreamParams);

      // Schedule recurring dreams
      dreamTimerRef.current = setInterval(() => {
        const nextParams = mutateParams(dreamParamsRef.current, dreamTemperature);
        nextParams.prompt = dreamVariation(
          dreamParamsRef.current.prompt,
          dreamTemperature
        );
        nextParams.skipAutoSelect = true; // Don't yank the view
        runGenerate(nextParams);
      }, dreamInterval);
    },
    [dreamTemperature, dreamInterval, runGenerate]
  );

  /**
   * Stop dream mode.
   */
  const stopDreaming = useCallback(() => {
    if (dreamTimerRef.current) {
      clearInterval(dreamTimerRef.current);
      dreamTimerRef.current = null;
    }
    setIsDreaming(false);
    dreamParamsRef.current = null;
  }, []);

  /**
   * Guide the dream - update base params based on user feedback.
   * @param {object} newBaseParams - New parameters to dream from
   */
  const guideDream = useCallback(
    (newBaseParams) => {
      if (!isDreaming) return;
      
      // Update the base params that dreams mutate from
      dreamParamsRef.current = { ...newBaseParams };
    },
    [isDreaming]
  );

  /**
   * Cancel a specific request.
   */
  const cancelRequest = useCallback(
    (id) => {
      return api.cancel(id);
    },
    [api]
  );

  /**
   * Cancel all in-flight requests.
   */
  const cancelAll = useCallback(() => {
    api.cancelAll();
    stopDreaming(); // Also stop dreaming
  }, [api, stopDreaming]);

  /**
   * Get server label for UI display.
   */
  const serverLabel = api.config.bases.length > 0
    ? `RR (${api.config.bases.length} backends)`
    : api.config.single || '(same origin)';

  /**
   * Get an image from cache by params.
   * Returns blob URL if found, null otherwise.
   */
  const getImageFromCache = useCallback(async (params) => {
    if (!cache) return null;
    try {
      const key = generateCacheKey(params);
      const entry = await cache.get(key);
      if (entry?.blob) {
        return URL.createObjectURL(entry.blob);
      }
    } catch (err) {
      console.warn('[Cache] getImageFromCache failed:', err);
    }
    return null;
  }, [cache]);

  /**
   * Get cache stats.
   */
  const getCacheStats = useCallback(async () => {
    if (!cache) return { enabled: false };
    return await cache.stats();
  }, [cache]);

  /**
   * Clear the image cache.
   */
  const clearCache = useCallback(async () => {
    if (cache) {
      await cache.clear();
      console.log('[Cache] Cleared');
    }
  }, [cache]);

  return {
    // Generation
    runGenerate,
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

    // Cache
    getImageFromCache,
    getCacheStats,
    clearCache,

    // State
    inflightCount: api.inflightCount,
    serverLabel,
  };
}