// src/hooks/useLatentExploration.js

import { useState, useCallback, useRef } from 'react';
import { clampInt } from '../utils/helpers';
import { STEPS_CONFIG, CFG_CONFIG, SEED_MODES } from '../utils/constants';

/**
 * Latent space exploration strategies based on LCM behavior.
 * 
 * Strategy 1: "Latent Lock" (0 steps, high CFG)
 * - Steps: 0
 * - CFG: 4-20
 * - Captures prompt semantic in latent space
 * - Next iteration "remembers" this latent state
 * 
 * Strategy 2: "Latent Cousins" (many steps, zero CFG, low denoise)
 * - Steps: 7+
 * - CFG: 0 (implicit guidance disabled)
 * - Denoise: <1.0
 * - Explores latent neighborhood of seed
 * - Produces detailed, prompt-sticky variations
 */

/**
 * Latent exploration presets.
 */
export const LATENT_PRESETS = {
  // Initial prompt encoding into latent space
  LATENT_LOCK: {
    name: 'Latent Lock',
    description: 'Encode prompt into latent space (0 steps, high CFG)',
    steps: 0,
    cfg: 8.0,
    denoise: 1.0,
    passCount: 1,
  },
  
  // Explore latent neighborhood with detail
  COUSINS_DETAILED: {
    name: 'Latent Cousins (Detailed)',
    description: 'Detailed variations in latent neighborhood',
    steps: 10,
    cfg: 0.0,
    denoise: 0.7,
    passCount: 1,
  },
  
  // Subtle latent variations
  COUSINS_SUBTLE: {
    name: 'Latent Cousins (Subtle)',
    description: 'Subtle variations staying close to seed',
    steps: 7,
    cfg: 0.0,
    denoise: 0.5,
    passCount: 1,
  },
  
  // Aggressive latent exploration
  COUSINS_WILD: {
    name: 'Latent Cousins (Wild)',
    description: 'Wild exploration of latent space',
    steps: 15,
    cfg: 0.0,
    denoise: 0.9,
    passCount: 2,
  },
  
  // Progressive refinement
  PROGRESSIVE: {
    name: 'Progressive Refinement',
    description: 'Lock then refine with multiple passes',
    steps: 8,
    cfg: 0.5,
    denoise: 0.8,
    passCount: 3,
  },
};

/**
 * Hook for latent space exploration and multi-pass rendering.
 * Exploits LCM's ability to hold semantic information across iterations.
 * 
 * @param {function} runGenerate - Generation function from useImageGeneration
 * @returns {object} Latent exploration state and controls
 * 
 * @example
 * const latent = useLatentExploration(runGenerate);
 * 
 * // Lock a prompt into latent space
 * await latent.lockPrompt({ prompt: "...", seed: 12345678 });
 * 
 * // Explore latent cousins
 * await latent.exploreCousins({ 
 *   seed: 12345678, 
 *   denoise: 0.7, 
 *   steps: 10 
 * });
 */
export function useLatentExploration(runGenerate) {
  // Current exploration mode
  const [explorationMode, setExplorationMode] = useState('standard'); // standard | latent_lock | cousins
  
  // Multi-pass configuration
  const [multiPassEnabled, setMultiPassEnabled] = useState(false);
  const [passCount, setPassCount] = useState(1);
  const [denoiseStrength, setDenoiseStrength] = useState(0.7);
  
  // Latent chain tracking (for multi-pass)
  const latentChainRef = useRef([]);
  
  // Current preset
  const [activePreset, setActivePreset] = useState(null);

  /**
   * Lock a prompt into latent space using 0-step high-CFG generation.
   * This creates a strong semantic encoding that persists.
   * 
   * @param {object} params - Base parameters
   * @param {string} params.prompt - Prompt to lock
   * @param {number} params.seed - Seed for reproducibility
   * @param {string} params.size - Image size
   * @returns {Promise<string>} Message ID of locked latent
   */
  const lockPrompt = useCallback(
    async (params) => {
      const { prompt, seed, size = '512x512' } = params;

      const lockParams = {
        prompt,
        seed,
        size,
        steps: 0, // Zero steps = latent encoding only
        cfg: 8.0, // High CFG = strong prompt adherence
        seedMode: SEED_MODES.FIXED,
        superresLevel: 0,
      };

      const msgId = await runGenerate(lockParams);
      
      // Track this as the start of a latent chain
      latentChainRef.current = [
        {
          msgId,
          params: lockParams,
          type: 'lock',
          timestamp: Date.now(),
        },
      ];

      return msgId;
    },
    [runGenerate]
  );

  /**
   * Explore "latent cousins" - variations in the neighborhood of a seed.
   * Uses many steps, zero CFG, and controlled denoise for detail.
   * 
   * @param {object} params - Exploration parameters
   * @param {string} params.prompt - Base prompt
   * @param {number} params.seed - Seed to explore around
   * @param {number} [params.denoise=0.7] - Denoise strength (0-1)
   * @param {number} [params.steps=10] - Inference steps
   * @param {string} [params.size='512x512'] - Image size
   * @returns {Promise<string>} Message ID
   */
  const exploreCousins = useCallback(
    async (params) => {
      const {
        prompt,
        seed,
        denoise = 0.7,
        steps = 10,
        size = '512x512',
      } = params;

      const cousinParams = {
        prompt,
        seed,
        size,
        steps: clampInt(steps, 7, STEPS_CONFIG.MAX), // Min 7 for cousins
        cfg: 0.0, // Zero CFG = implicit guidance off
        seedMode: SEED_MODES.FIXED,
        superresLevel: 0,
        // Note: denoise would need backend support
        denoise, // Pass this through if your backend supports it
      };

      const msgId = await runGenerate(cousinParams);

      latentChainRef.current.push({
        msgId,
        params: cousinParams,
        type: 'cousin',
        timestamp: Date.now(),
      });

      return msgId;
    },
    [runGenerate]
  );

  /**
   * Multi-pass refinement - run multiple passes with controlled denoise.
   * Each pass builds on the previous latent state.
   * 
   * @param {object} params - Base parameters
   * @param {number} [numPasses=3] - Number of passes
   * @param {number} [denoisePerPass=0.8] - Denoise per pass
   * @returns {Promise<string[]>} Array of message IDs
   */
  const multiPassRefine = useCallback(
    async (params, numPasses = 3, denoisePerPass = 0.8) => {
      const {
        prompt,
        seed,
        size = '512x512',
        steps = 8,
        cfg = 0.5,
      } = params;

      const messageIds = [];
      latentChainRef.current = [];

      for (let pass = 0; pass < numPasses; pass++) {
        const passParams = {
          prompt,
          seed,
          size,
          steps,
          cfg,
          seedMode: SEED_MODES.FIXED,
          superresLevel: 0,
          denoise: denoisePerPass,
          passNumber: pass + 1,
          totalPasses: numPasses,
        };

        const msgId = await runGenerate(passParams);
        messageIds.push(msgId);

        latentChainRef.current.push({
          msgId,
          params: passParams,
          type: 'refinement',
          pass: pass + 1,
          timestamp: Date.now(),
        });

        // Small delay between passes to avoid overwhelming backend
        if (pass < numPasses - 1) {
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      }

      return messageIds;
    },
    [runGenerate]
  );

  /**
   * Apply a preset exploration strategy.
   * 
   * @param {string} presetKey - Key from LATENT_PRESETS
   * @param {object} baseParams - Base parameters (prompt, seed, size)
   * @returns {Promise<string|string[]>} Message ID(s)
   */
  const applyPreset = useCallback(
    async (presetKey, baseParams) => {
      const preset = LATENT_PRESETS[presetKey];
      if (!preset) {
        throw new Error(`Unknown preset: ${presetKey}`);
      }

      setActivePreset(presetKey);

      const params = {
        ...baseParams,
        steps: preset.steps,
        cfg: preset.cfg,
        denoise: preset.denoise,
      };

      if (preset.passCount > 1) {
        return await multiPassRefine(
          params,
          preset.passCount,
          preset.denoise
        );
      } else if (preset.steps === 0) {
        return await lockPrompt(params);
      } else if (preset.cfg === 0) {
        return await exploreCousins(params);
      } else {
        return await runGenerate({
          ...params,
          seedMode: SEED_MODES.FIXED,
          superresLevel: 0,
        });
      }
    },
    [runGenerate, lockPrompt, exploreCousins, multiPassRefine]
  );

  /**
   * Get the latent chain history.
   * Useful for visualizing the exploration path.
   */
  const getLatentChain = useCallback(() => {
    return [...latentChainRef.current];
  }, []);

  /**
   * Clear latent chain history.
   */
  const clearLatentChain = useCallback(() => {
    latentChainRef.current = [];
  }, []);

  /**
   * Generate a "latent walk" - smoothly interpolate through latent space.
   * Creates a sequence from seed A to seed B.
   * 
   * @param {object} params
   * @param {number} params.seedStart - Starting seed
   * @param {number} params.seedEnd - Ending seed
   * @param {number} [params.steps=5] - Number of interpolation steps
   * @param {string} params.prompt - Prompt to use
   * @returns {Promise<string[]>} Array of message IDs
   */
  const latentWalk = useCallback(
    async (params) => {
      const {
        seedStart,
        seedEnd,
        steps: walkSteps = 5,
        prompt,
        size = '512x512',
        renderSteps = 10,
        cfg = 0.0,
      } = params;

      const messageIds = [];
      latentChainRef.current = [];

      for (let i = 0; i <= walkSteps; i++) {
        // Linear interpolation between seeds (simplified)
        // In practice, you'd need backend support for true latent interpolation
        const t = i / walkSteps;
        const interpolatedSeed = Math.round(
          seedStart + (seedEnd - seedStart) * t
        );

        const walkParams = {
          prompt,
          seed: interpolatedSeed,
          size,
          steps: renderSteps,
          cfg,
          seedMode: SEED_MODES.FIXED,
          superresLevel: 0,
          denoise: 0.7,
        };

        const msgId = await runGenerate(walkParams);
        messageIds.push(msgId);

        latentChainRef.current.push({
          msgId,
          params: walkParams,
          type: 'walk',
          interpolationStep: i,
          totalSteps: walkSteps,
          timestamp: Date.now(),
        });

        // Delay between walk steps
        if (i < walkSteps) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      }

      return messageIds;
    },
    [runGenerate]
  );

  return {
    // Modes
    explorationMode,
    setExplorationMode,

    // Multi-pass controls
    multiPassEnabled,
    setMultiPassEnabled,
    passCount,
    setPassCount,
    denoiseStrength,
    setDenoiseStrength,

    // Preset management
    activePreset,
    applyPreset,
    presets: LATENT_PRESETS,

    // Exploration functions
    lockPrompt,
    exploreCousins,
    multiPassRefine,
    latentWalk,

    // History
    getLatentChain,
    clearLatentChain,
    chainLength: latentChainRef.current.length,
  };
}