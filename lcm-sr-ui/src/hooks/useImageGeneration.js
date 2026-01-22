// src/hooks/useImageGeneration.js

import { useCallback, useRef, useEffect, useState } from 'react';
import { createApiClient, createApiConfig } from '../utils/api';
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
  // API client (created once)
  const apiClientRef = useRef(null);
  
  // Dream mode state
  const [isDreaming, setIsDreaming] = useState(false);
  const [dreamTemperature, setDreamTemperature] = useState(0.3); // 0-1
  const [dreamInterval, setDreamInterval] = useState(5000); // ms between dreams
  const dreamTimerRef = useRef(null);
  const dreamParamsRef = useRef(null);
  
  // Initialize API client
  if (!apiClientRef.current) {
    const config = createApiConfig();
    apiClientRef.current = createApiClient(config);
  }

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
   */
  const runGenerate = useCallback(
    async (params) => {
      const {
        prompt: promptParam,
        size: sizeParam,
        steps: stepsParam,
        cfg: cfgParam,
        superresLevel: srLevelParam,
        seedMode: seedModeParam,
        seed: seedParam,
        targetMessageId,
        skipAutoSelect = false, // Don't auto-select after generation (used by dream mode)
      } = params;

      // Validate prompt
      const p = safeJsonString(promptParam).trim();
      if (!p) return;

      // Resolve parameters with clamping
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
        // In-place regeneration - keep the image visible with regenerating flag
        // Don't set text - use overlay indicator instead to avoid layout shift
        updateMessage(targetMessageId, {
          isRegenerating: true,
          text: null,
        });
      } else {
        // New generation - add pending message only (no user text bubble)
        const pendingMsg = {
          id: assistantId,
          role: MESSAGE_ROLES.ASSISTANT,
          kind: MESSAGE_KINDS.PENDING,
          text: null, // No text - just show placeholder
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

      try {
        const result = await api.generate(
          {
            prompt: p,
            size: useSize,
            steps: useSteps,
            cfg: useCfg,
            seed: reqSeed,
            superres: superresOn,
            superresLevel: useSrLevel,
          },
          assistantId
        );

        // Update message with result (no text - just the image)
        updateMessage(assistantId, {
          kind: MESSAGE_KINDS.IMAGE,
          isRegenerating: false,
          text: null,
          imageUrl: result.imageUrl,
          params: {
            prompt: p,
            size: useSize,
            steps: useSteps,
            cfg: useCfg,
            seedMode: SEED_MODES.FIXED,
            seed: result.metadata.seed,
            superresLevel: useSrLevel,
          },
          meta: {
            backend: result.metadata.backend,
            apiBase: result.metadata.apiBase,
            superres: result.metadata.superres,
            srScale: result.metadata.srScale,
          },
        });

        // Auto-select the new image (unless suppressed, e.g., during dream mode)
        if (!skipAutoSelect) {
          setSelectedMsgId(assistantId);
        }

        return assistantId; // Return message ID for dream mode
      } catch (err) {
        const errMsg =
          err?.name === ABORT_ERROR_NAME
            ? UI_MESSAGES.CANCELED
            : err?.message || String(err);
        updateMessage(assistantId, {
          kind: MESSAGE_KINDS.ERROR,
          isRegenerating: false,
          text: errMsg,
        });
        return null;
      }
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
    
    // State
    inflightCount: api.inflightCount,
    serverLabel,
  };
}