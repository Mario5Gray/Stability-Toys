// src/utils/api.js

import {
  API_ENDPOINTS,
  FILE_CONFIG,
  RESPONSE_HEADERS,
  CONTENT_TYPES,
} from './constants';
import {
  responseToObjectURLStrict,
  extractSRMetadata,
  extractGenerationMetadata,
  parseApiBases,
  normalizeBase,
  
} from './helpers';

/* ============================================================================
 * API CONFIGURATION
 * ========================================================================== */

/**
 * Create API configuration from environment variables.
 * Checks Vite and CRA environment variable patterns.
 * 
 * @returns {object} API configuration with bases and single URL
 * @returns {string[]} return.bases - Array of base URLs for round-robin
 * @returns {string} return.single - Single base URL fallback
 * 
 * @example
 * const config = createApiConfig();
 * // => { bases: ["http://api1.com", "http://api2.com"], single: "" }
 */
export function createApiConfig() {
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


  // Import parseApiBases and normalizeBase from helpers
  (async () => {
    const { parseApiBases, normalizeBase } = await import('./helpers');
  })();


  const bases = parseApiBases(vitePlural || "");
  const single = normalizeBase(viteSingle || craSingle || "");

  return { bases, single };
}

/**
 * Round-robin selector for API base URLs.
 * Creates a stateful picker that cycles through available bases.
 * 
 * @param {object} apiConfig - API configuration object
 * @returns {function} Function that returns next API base URL
 * 
 * @example
 * const picker = createRoundRobinPicker(config);
 * picker() // => "http://api1.com"
 * picker() // => "http://api2.com"
 * picker() // => "http://api1.com" (cycles back)
 */
export function createRoundRobinPicker(apiConfig) {
  let counter = 0;
  (async () => {
    const { normalizeBase } = await import('./helpers');
  })();

  return function pickApiBase() {
    const bases = apiConfig.bases.map(normalizeBase).filter(Boolean);
    if (bases.length > 0) {
      const idx = counter++ % bases.length;
      return bases[idx];
    }
    return normalizeBase(apiConfig.single);
  };
}

/* ============================================================================
 * IMAGE GENERATION API
 * ========================================================================== */

/**
 * Request parameters for image generation.
 * @typedef {object} GenerateParams
 * @property {string} prompt - Text prompt for generation
 * @property {string} size - Image size (e.g., "512x512")
 * @property {number} steps - Number of inference steps (0 for latent lock)
 * @property {number} cfg - Guidance scale (CFG, 0 for implicit off)
 * @property {number} seed - Random seed
 * @property {boolean} superres - Enable super-resolution
 * @property {number} superresLevel - SR magnitude (1-3)
 * @property {number} [denoise=1.0] - Denoise strength (0.1-1.0)
 * @property {number} [passNumber] - Current pass number (for multi-pass)
 * @property {number} [totalPasses] - Total number of passes
 */

/**
 * Response from image generation API.
 * @typedef {object} GenerateResponse
 * @property {string} imageUrl - Blob URL of generated image
 * @property {object} metadata - Response metadata
 * @property {number} metadata.seed - Seed used for generation
 * @property {boolean} metadata.superres - Whether SR was applied
 * @property {string|null} metadata.srScale - SR scale factor
 * @property {string|null} metadata.backend - Backend server identifier
 * @property {string} metadata.apiBase - API base URL used
 */

/**
 * Call the /generate endpoint to create an image.
 * 
 * @param {string} apiBase - Base URL for the API
 * @param {GenerateParams} params - Generation parameters
 * @param {AbortSignal} [signal] - Optional abort signal
 * @returns {Promise<GenerateResponse>} Generated image URL and metadata
 * @throws {Error} If request fails or response is not OK
 * 
 * @example
 * const result = await generateImage(
 *   "https://api.example.com",
 *   {
 *     prompt: "a sunset",
 *     size: "512x512",
 *     steps: 4,
 *     cfg: 1.0,
 *     seed: 12345678,
 *     superres: true,
 *     superresLevel: 2
 *   },
 *   controller.signal
 * );
 * 
 * imageElement.src = result.imageUrl;
 * console.log("Seed used:", result.metadata.seed);
 */
export async function generateImage(apiBase, params, signal = null) {
  const {
    prompt,
    size,
    steps,
    cfg,
    seed,
    superres,
    superresLevel,
    denoise = 1.0,
    passNumber,
    totalPasses,
  } = params;

  const body = {
    prompt,
    size,
    num_inference_steps: steps,
    guidance_scale: cfg,
    seed,
    superres,
    superres_format: FILE_CONFIG.SUPERRES_FORMAT,
    superres_quality: FILE_CONFIG.QUALITY,
    superres_magnitude: superres ? superresLevel : 1,
  };

  // Add denoise if not default (backend compatibility)
  if (denoise !== 1.0) {
    body.denoise_strength = denoise;
  }

  // Add pass info if multi-pass (backend compatibility)
  if (passNumber !== undefined && totalPasses !== undefined) {
    body.pass_number = passNumber;
    body.total_passes = totalPasses;
  }

  const res = await fetch(`${apiBase}${API_ENDPOINTS.GENERATE}`, {
    method: "POST",
    headers: { "Content-Type": CONTENT_TYPES.JSON },
    body: JSON.stringify(body),
    signal,
  });

  const imageUrl = await responseToObjectURLStrict(res);
  const metadata = extractGenerationMetadata(res, seed);

  return {
    imageUrl,
    metadata: {
      ...metadata,
      apiBase: apiBase || "",
      denoise,
      passNumber,
      totalPasses,
    },
  };
}

/* ============================================================================
 * SUPER-RESOLUTION API
 * ========================================================================== */

/**
 * Request parameters for super-resolution upload.
 * @typedef {object} SuperResParams
 * @property {File} file - Image file to super-resolve
 * @property {number} magnitude - SR magnitude (1-3, number of passes)
 */

/**
 * Response from super-resolution API.
 * @typedef {object} SuperResResponse
 * @property {string} imageUrl - Blob URL of super-resolved image
 * @property {object} metadata - Response metadata
 * @property {string} metadata.magnitude - SR magnitude applied
 * @property {string} metadata.passes - Number of SR passes
 * @property {string|null} metadata.scale - SR scale factor
 * @property {string|null} metadata.backend - Backend server identifier
 * @property {string} metadata.apiBase - API base URL used
 */

/**
 * Upload an image for super-resolution processing.
 * 
 * @param {string} apiBase - Base URL for the API
 * @param {SuperResParams} params - Super-resolution parameters
 * @param {AbortSignal} [signal] - Optional abort signal
 * @returns {Promise<SuperResResponse>} Super-resolved image URL and metadata
 * @throws {Error} If request fails or response is not OK
 * 
 * @example
 * const fileInput = document.querySelector('input[type="file"]');
 * const file = fileInput.files[0];
 * 
 * const result = await superResolveImage(
 *   "https://api.example.com",
 *   { file, magnitude: 2 },
 *   controller.signal
 * );
 * 
 * imageElement.src = result.imageUrl;
 * console.log("Applied", result.metadata.passes, "SR passes");
 */
export async function superResolveImage(apiBase, params, signal = null) {
  const { file, magnitude } = params;

  const fd = new FormData();
  fd.append("file", file);
  fd.append("magnitude", String(magnitude));
  fd.append("out_format", FILE_CONFIG.OUTPUT_FORMAT);
  fd.append("quality", String(FILE_CONFIG.QUALITY));

  const res = await fetch(`${apiBase}${API_ENDPOINTS.SUPERRES}`, {
    method: "POST",
    body: fd,
    signal,
  });

  const imageUrl = await responseToObjectURLStrict(res);
  const srMeta = extractSRMetadata(res, magnitude);

  return {
    imageUrl,
    metadata: {
      magnitude: srMeta.magnitude,
      passes: srMeta.passes,
      scale: srMeta.scale,
      backend: srMeta.backend,
      apiBase: apiBase || "",
    },
  };
}

/* ============================================================================
 * BLOB URL MANAGEMENT
 * ========================================================================== */

/**
 * Blob URL manager for tracking and cleanup.
 * Prevents memory leaks by revoking object URLs when done.
 * 
 * @returns {object} Manager with add/clear/cleanup methods
 * 
 * @example
 * const blobManager = createBlobUrlManager();
 * 
 * const url = URL.createObjectURL(blob);
 * blobManager.add(url);
 * 
 * // Later, cleanup all tracked URLs
 * blobManager.cleanup();
 */
export function createBlobUrlManager() {
  const urls = new Set();

  return {
    /**
     * Track a blob URL for later cleanup.
     * @param {string} url - Blob URL to track
     */
    add(url) {
      urls.add(url);
    },

    /**
     * Remove a specific URL from tracking (doesn't revoke it).
     * @param {string} url - URL to stop tracking
     */
    remove(url) {
      urls.delete(url);
    },

    /**
     * Revoke all tracked blob URLs and clear the set.
     */
    cleanup() {
      for (const url of urls) {
        try {
          URL.revokeObjectURL(url);
        } catch (err) {
          console.warn("Failed to revoke blob URL:", err);
        }
      }
      urls.clear();
    },

    /**
     * Get count of tracked URLs.
     * @returns {number}
     */
    get size() {
      return urls.size;
    },
  };
}

/* ============================================================================
 * ABORT CONTROLLER MANAGEMENT
 * ========================================================================== */

/**
 * Manages AbortControllers for in-flight requests.
 * Allows canceling individual or all requests.
 * 
 * @returns {object} Manager with add/abort/abortAll/cleanup methods
 * 
 * @example
 * const abortManager = createAbortManager();
 * 
 * const controller = new AbortController();
 * abortManager.add("msg-123", controller);
 * 
 * // Cancel specific request
 * abortManager.abort("msg-123");
 * 
 * // Cancel all requests
 * abortManager.abortAll();
 */
export function createAbortManager() {
  const controllers = new Map();

  return {
    /**
     * Track an AbortController for a request.
     * @param {string} id - Request identifier
     * @param {AbortController} controller - AbortController instance
     */
    add(id, controller) {
      controllers.set(id, controller);
    },

    /**
     * Abort a specific request and remove it.
     * @param {string} id - Request identifier
     * @returns {boolean} True if request was found and aborted
     */
    abort(id) {
      const controller = controllers.get(id);
      if (controller) {
        controller.abort();
        controllers.delete(id);
        return true;
      }
      return false;
    },

    /**
     * Abort all tracked requests.
     */
    abortAll() {
      for (const [id, controller] of controllers.entries()) {
        controller.abort();
      }
      controllers.clear();
    },

    /**
     * Remove a controller without aborting (called after request completes).
     * @param {string} id - Request identifier
     */
    remove(id) {
      controllers.delete(id);
    },

    /**
     * Get count of in-flight requests.
     * @returns {number}
     */
    get size() {
      return controllers.size;
    },

    /**
     * Check if a specific request is in-flight.
     * @param {string} id - Request identifier
     * @returns {boolean}
     */
    has(id) {
      return controllers.has(id);
    },
  };
}

/* ============================================================================
 * HIGH-LEVEL API CLIENT
 * ========================================================================== */

/**
 * Create a complete API client with all functionality.
 * Combines generation, SR, blob management, and abort handling.
 * 
 * @param {object} apiConfig - API configuration from createApiConfig()
 * @returns {object} API client with all methods
 * 
 * @example
 * const config = createApiConfig();
 * const api = createApiClient(config);
 * 
 * // Generate image
 * const result = await api.generate({
 *   prompt: "a sunset",
 *   size: "512x512",
 *   steps: 4,
 *   cfg: 1.0,
 *   seed: 12345678,
 *   superres: false,
 *   superresLevel: 0
 * });
 * 
 * // Super-resolve uploaded image
 * const srResult = await api.superResolve({ file, magnitude: 2 });
 * 
 * // Cleanup on unmount
 * api.cleanup();
 */
export function createApiClient(apiConfig) {
  const pickApiBase = createRoundRobinPicker(apiConfig);
  const blobManager = createBlobUrlManager();
  const abortManager = createAbortManager();

  return {
    /**
     * Generate an image with automatic API base selection and tracking.
     */
    async generate(params, requestId = null) {
      const apiBase = pickApiBase();
      const controller = new AbortController();
      
      if (requestId) {
        abortManager.add(requestId, controller);
      }

      try {
        const result = await generateImage(apiBase, params, controller.signal);
        blobManager.add(result.imageUrl);
        return result;
      } finally {
        if (requestId) {
          abortManager.remove(requestId);
        }
      }
    },

    /**
     * Super-resolve an uploaded image with automatic API base selection.
     */
    async superResolve(params, requestId = null) {
      const apiBase = pickApiBase();
      const controller = new AbortController();
      
      if (requestId) {
        abortManager.add(requestId, controller);
      }

      try {
        const result = await superResolveImage(apiBase, params, controller.signal);
        blobManager.add(result.imageUrl);
        return result;
      } finally {
        if (requestId) {
          abortManager.remove(requestId);
        }
      }
    },

    /**
     * Cancel a specific request.
     */
    cancel(requestId) {
      return abortManager.abort(requestId);
    },

    /**
     * Cancel all in-flight requests.
     */
    cancelAll() {
      abortManager.abortAll();
    },

    /**
     * Get count of in-flight requests.
     */
    get inflightCount() {
      return abortManager.size;
    },

    /**
     * Cleanup all blob URLs and abort controllers.
     * Call this on component unmount.
     */
    cleanup() {
      abortManager.abortAll();
      blobManager.cleanup();
    },

    /**
     * Get current API configuration.
     */
    get config() {
      return apiConfig;
    },
  };
}