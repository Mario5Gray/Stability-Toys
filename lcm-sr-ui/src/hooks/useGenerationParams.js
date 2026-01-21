// src/hooks/useGenerationParams.js

import { useState, useMemo, useCallback, useRef } from 'react';
import { eightDigitSeed, clampInt, safeJsonString } from '../utils/helpers';
import {
  DEFAULT_PROMPT,
  DEFAULT_SIZE,
  STEPS_CONFIG,
  CFG_CONFIG,
  SR_CONFIG,
  SEED_MODES,
  DEBOUNCE_CONFIG,
} from '../utils/constants';

/**
 * Hook for managing image generation parameters.
 * Handles both draft parameters and selected image parameters.
 * Supports in-place regeneration with debouncing.
 * 
 * @param {object|null} selectedParams - Parameters from selected message
 * @param {function} patchSelectedParams - Function to update selected params
 * @param {function} runGenerate - Function to trigger generation
 * @param {string|null} selectedMsgId - Selected message ID
 * @returns {object} Parameters and setters
 * 
 * @example
 * const params = useGenerationParams(
 *   selectedParams,
 *   patchSelectedParams,
 *   runGenerate,
 *   selectedMsgId
 * );
 * 
 * // Use effective values in UI
 * <input value={params.effective.prompt} onChange={e => params.setPrompt(e.target.value)} />
 */
export function useGenerationParams(
  selectedParams,
  patchSelectedParams,
  runGenerate,
  selectedMsgId
) {
  // Draft parameters (when no image is selected)
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [size, setSize] = useState(DEFAULT_SIZE);
  const [steps, setSteps] = useState(STEPS_CONFIG.DEFAULT);
  const [cfg, setCfg] = useState(CFG_CONFIG.DEFAULT);
  const [srLevel, setSrLevel] = useState(SR_CONFIG.DEFAULT);
  const [seedMode, setSeedMode] = useState(SEED_MODES.RANDOM);
  const [seed, setSeed] = useState(() => String(eightDigitSeed()));

  // Debounce timer for regeneration
  const regenTimerRef = useRef(null);

  /**
   * Default values object (from draft state).
   */
  const DEFAULTS = useMemo(
    () => ({
      prompt,
      size,
      steps,
      cfg,
      seedMode,
      seed: seedMode === SEED_MODES.FIXED ? Number(seed || 0) : null,
      superresLevel: srLevel,
    }),
    [prompt, size, steps, cfg, seedMode, seed, srLevel]
  );

  /**
   * Effective values - either from selected message or defaults.
   * These are clamped/validated versions.
   */
  const effective = useMemo(() => {
    const src = selectedParams ?? DEFAULTS;

    return {
      prompt: String(src.prompt ?? DEFAULTS.prompt),
      size: String(src.size ?? DEFAULTS.size),
      steps: clampInt(Number(src.steps ?? DEFAULTS.steps), STEPS_CONFIG.MIN, STEPS_CONFIG.MAX),
      cfg: Number.isFinite(Number(src.cfg)) ? Number(src.cfg) : Number(DEFAULTS.cfg),
      seedMode: src.seedMode ?? DEFAULTS.seedMode,
      seed: src.seed ?? DEFAULTS.seed,
      superresLevel: clampInt(
        Number(src.superresLevel ?? DEFAULTS.superresLevel),
        SR_CONFIG.MIN,
        SR_CONFIG.BACKEND_MAX
      ),
    };
  }, [selectedParams, DEFAULTS]);

  /**
   * Schedule a regeneration with debouncing.
   * Updates params immediately, then regenerates after delay.
   */
  const scheduleRegenSelected = useCallback(
    (patch) => {
      if (!selectedMsgId || !selectedParams) return;

      // Update params immediately (UI responds instantly)
      patchSelectedParams(patch);

      // Debounce real regen
      window.clearTimeout(regenTimerRef.current);
      regenTimerRef.current = window.setTimeout(() => {
        const next = { ...selectedParams, ...patch };
        runGenerate({
          ...next,
          seedMode: SEED_MODES.FIXED,
          seed: next.seed,
          superresLevel: next.superresLevel ?? 0,
          targetMessageId: selectedMsgId, // In-place update
        });
      }, DEBOUNCE_CONFIG.REGEN_DELAY_MS);
    },
    [selectedMsgId, selectedParams, patchSelectedParams, runGenerate]
  );

  /**
   * Set prompt - updates draft or triggers regen if selected.
   */
  const setPromptEffective = useCallback(
    (value) => {
      const v = safeJsonString(value);
      if (selectedParams) {
        scheduleRegenSelected({ prompt: v });
      } else {
        setPrompt(v);
      }
    },
    [selectedParams, scheduleRegenSelected]
  );

  /**
   * Set size - updates draft or triggers regen if selected.
   */
  const setSizeEffective = useCallback(
    (value) => {
      if (selectedParams) {
        scheduleRegenSelected({ size: value });
      } else {
        setSize(value);
      }
    },
    [selectedParams, scheduleRegenSelected]
  );

  /**
   * Set steps - updates draft or triggers regen if selected.
   */
  const setStepsEffective = useCallback(
    (value) => {
      const clamped = clampInt(Number(value), STEPS_CONFIG.MIN, STEPS_CONFIG.MAX);
      if (selectedParams) {
        scheduleRegenSelected({ steps: clamped });
      } else {
        setSteps(clamped);
      }
    },
    [selectedParams, scheduleRegenSelected]
  );

  /**
   * Set CFG - updates draft or triggers regen if selected.
   */
  const setCfgEffective = useCallback(
    (value) => {
      const cfg = Number(value);
      if (selectedParams) {
        scheduleRegenSelected({ cfg });
      } else {
        setCfg(cfg);
      }
    },
    [selectedParams, scheduleRegenSelected]
  );

  /**
   * Set SR level - updates draft or triggers regen if selected.
   */
  const setSrLevelEffective = useCallback(
    (value) => {
      const level = clampInt(Number(value), SR_CONFIG.MIN, SR_CONFIG.MAX);
      if (selectedParams) {
        scheduleRegenSelected({ superresLevel: level });
      } else {
        setSrLevel(level);
      }
    },
    [selectedParams, scheduleRegenSelected]
  );

  /**
   * Randomize seed value.
   */
  const randomizeSeed = useCallback(() => {
    setSeed(String(eightDigitSeed()));
  }, []);

  /**
   * Apply a prompt delta (append text to existing prompt).
   */
  const applyPromptDelta = useCallback(
    (delta) => {
      if (!selectedParams) return;
      const base = String(selectedParams.prompt || "").trim();
      const next = base ? `${base}, ${delta}` : delta;
      patchSelectedParams({ prompt: next });
    },
    [selectedParams, patchSelectedParams]
  );

  /**
   * Reset all parameters to defaults.
   */
  const resetToDefaults = useCallback(() => {
    setPrompt(DEFAULT_PROMPT);
    setSize(DEFAULT_SIZE);
    setSteps(STEPS_CONFIG.DEFAULT);
    setCfg(CFG_CONFIG.DEFAULT);
    setSrLevel(SR_CONFIG.DEFAULT);
    setSeedMode(SEED_MODES.RANDOM);
    setSeed(String(eightDigitSeed()));
  }, []);

  /**
   * Load parameters from message metadata.
   * Used when clicking on a previous generation to restore settings.
   */
  const loadFromMetadata = useCallback((meta) => {
    if (!meta) return;

    if (typeof meta.size === 'string' && /^\d+x\d+$/i.test(meta.size)) {
      setSize(meta.size);
    }
    if (Number.isFinite(meta.steps)) {
      setSteps(clampInt(Number(meta.steps), STEPS_CONFIG.MIN, STEPS_CONFIG.MAX));
    }
    if (Number.isFinite(meta.cfg)) {
      setCfg(Number(meta.cfg));
    }

    // Restore SR level from metadata
    if (meta?.superres) {
      const mag =
        Number(meta?.srMagnitude) ||
        Number(meta?.srPasses) ||
        (meta?.srScale ? SR_CONFIG.DEFAULT_MAGNITUDE : 1);
      setSrLevel(clampInt(mag, SR_CONFIG.MIN, SR_CONFIG.MAX));
    } else if (typeof meta?.superres === 'boolean') {
      setSrLevel(meta.superres ? SR_CONFIG.DEFAULT_MAGNITUDE : 0);
    }

    // Restore seed
    if (meta.seed !== undefined && meta.seed !== null) {
      setSeedMode(SEED_MODES.FIXED);
      setSeed(String(meta.seed));
    }
  }, []);

  return {
    // Raw draft state
    draft: {
      prompt,
      size,
      steps,
      cfg,
      srLevel,
      seedMode,
      seed,
    },

    // Effective values (selected or draft, validated)
    effective,

    // Setters (smart - update draft or trigger regen)
    setPrompt: setPromptEffective,
    setSize: setSizeEffective,
    setSteps: setStepsEffective,
    setCfg: setCfgEffective,
    setSrLevel: setSrLevelEffective,
    setSeedMode,
    setSeed,

    // Utilities
    randomizeSeed,
    applyPromptDelta,
    resetToDefaults,
    loadFromMetadata,

    // Direct draft setters (bypass smart logic)
    setPromptDirect: setPrompt,
    setSizeDirect: setSize,
    setStepsDirect: setSteps,
    setCfgDirect: setCfg,
    setSrLevelDirect: setSrLevel,
  };
}