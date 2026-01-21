// src/utils/helpers.js

import {
  SEED_CONFIG,
  FALLBACK_TEXTAREA_STYLES,
  REGEX_PATTERNS,
} from './constants';

/* ============================================================================
 * NUMERIC UTILITIES
 * ========================================================================== */

/**
 * Clamp a number to an integer within [lo, hi] range.
 * If n is not finite, returns lo as default.
 * 
 * @param {number} n - The number to clamp
 * @param {number} lo - Minimum value (inclusive)
 * @param {number} hi - Maximum value (inclusive)
 * @returns {number} Clamped integer value
 * 
 * @example
 * clampInt(5.7, 1, 10)   // => 6
 * clampInt(15, 1, 10)    // => 10
 * clampInt(NaN, 1, 10)   // => 1
 */
export function clampInt(n, lo, hi) {
  const x = Number.isFinite(n) ? n : lo;
  return Math.max(lo, Math.min(hi, Math.round(x)));
}

/**
 * Generate a random 8-digit seed number (0 to 99,999,999).
 * 
 * @returns {number} Random integer between 0 and SEED_CONFIG.RANDOM_MAX
 * 
 * @example
 * eightDigitSeed() // => 42857391
 */
export function eightDigitSeed() {
  return Math.floor(Math.random() * SEED_CONFIG.RANDOM_MAX);
}

/* ============================================================================
 * STRING UTILITIES
 * ========================================================================== */

/**
 * Safely convert any value to a string, handling null/undefined.
 * 
 * @param {*} s - Value to convert
 * @returns {string} String representation, empty string for null/undefined
 * 
 * @example
 * safeJsonString(null)      // => ""
 * safeJsonString(undefined) // => ""
 * safeJsonString(123)       // => "123"
 * safeJsonString("hello")   // => "hello"
 */
export function safeJsonString(s) {
  return (s ?? "").toString();
}

/**
 * Normalize a URL base by trimming and removing trailing slashes.
 * 
 * @param {string} s - URL base to normalize
 * @returns {string} Normalized URL base without trailing slashes
 * 
 * @example
 * normalizeBase("https://api.example.com///") // => "https://api.example.com"
 * normalizeBase("  /api/v1/  ")                // => "/api/v1"
 */
export function normalizeBase(s) {
  return String(s || "").trim().replace(REGEX_PATTERNS.TRAILING_SLASH, "");
}

/* ============================================================================
 * ID GENERATION
 * ========================================================================== */

/**
 * Generate a unique ID using timestamp + random hex.
 * 
 * @returns {string} Unique ID in format "{timestamp}-{random_hex}"
 * 
 * @example
 * nowId() // => "1705843200000-a3f5c2d9e1b4"
 */
export function nowId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/* ============================================================================
 * API CONFIGURATION PARSING
 * ========================================================================== */

/**
 * Parse API base URLs from environment variable string.
 * Splits on semicolons/commas, trims, filters empty, removes trailing slashes.
 * 
 * @param {string} env - Environment variable value with delimited URLs
 * @returns {string[]} Array of normalized base URLs
 * 
 * @example
 * parseApiBases("http://api1.com/; http://api2.com,http://api3.com/")
 * // => ["http://api1.com", "http://api2.com", "http://api3.com"]
 */
export function parseApiBases(env) {
  if (!env) return [];
  return String(env)
    .split(REGEX_PATTERNS.ENV_SPLIT)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => s.replace(REGEX_PATTERNS.TRAILING_SLASH, ""));
}

/* ============================================================================
 * ERROR HANDLING
 * ========================================================================== */

/**
 * Read error message from a previously-read ArrayBuffer.
 * IMPORTANT: never call res.json()/res.text() after res.arrayBuffer().
 * 
 * @param {Response} res - Fetch Response object
 * @param {ArrayBuffer} buf - Pre-read response body as ArrayBuffer
 * @returns {string} Error message extracted from response
 * 
 * @example
 * const buf = await res.arrayBuffer();
 * const error = readErrorFromArrayBuffer(res, buf);
 * console.error(error); // => "Invalid API key"
 */
export function readErrorFromArrayBuffer(res, buf) {
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

/* ============================================================================
 * RESPONSE PROCESSING
 * ========================================================================== */

/**
 * Read response body exactly once and convert to object URL.
 * Fixes: "Body is disturbed or locked" error.
 * Throws if response is not OK.
 * 
 * @param {Response} res - Fetch Response object
 * @returns {Promise<string>} Object URL for the response blob
 * @throws {Error} If response status is not OK
 * 
 * @example
 * const res = await fetch('/api/image');
 * const url = await responseToObjectURLStrict(res);
 * imgElement.src = url; // Use the blob URL
 */
export async function responseToObjectURLStrict(res) {
  const buf = await res.arrayBuffer();

  if (!res.ok) {
    const detail = readErrorFromArrayBuffer(res, buf);
    throw new Error(`HTTP ${res.status}: ${detail || res.statusText}`);
  }

  const contentType = res.headers.get("content-type") || "image/png";
  const blob = new Blob([buf], { type: contentType });
  return URL.createObjectURL(blob);
}

/* ============================================================================
 * CLIPBOARD UTILITIES
 * ========================================================================== */

/**
 * Copy text to clipboard with fallback for older browsers.
 * 
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} True if successful, false otherwise
 * 
 * @example
 * const success = await copyToClipboard("Hello World");
 * if (success) {
 *   console.log("Copied!");
 * }
 */
export async function copyToClipboard(text) {
  const t = String(text ?? "");
  try {
    await navigator.clipboard.writeText(t);
    return true;
  } catch {
    // Fallback for older / restricted contexts
    try {
      const ta = document.createElement("textarea");
      ta.value = t;
      ta.style.position = FALLBACK_TEXTAREA_STYLES.position;
      ta.style.left = FALLBACK_TEXTAREA_STYLES.left;
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

/* ============================================================================
 * SCROLL UTILITIES
 * ========================================================================== */

/**
 * Check if a scrollable element is near the bottom.
 * 
 * @param {HTMLElement|null} el - Scrollable element to check
 * @param {number} thresholdPx - Pixel threshold for "near bottom" (default: 80)
 * @returns {boolean} True if near bottom or element is null
 * 
 * @example
 * const chatDiv = document.querySelector('.chat');
 * if (isNearBottom(chatDiv)) {
 *   // Auto-scroll to new messages
 * }
 */
export function isNearBottom(el, thresholdPx = 80) {
  if (!el) return true;
  return el.scrollHeight - el.scrollTop - el.clientHeight < thresholdPx;
}

/* ============================================================================
 * VALIDATION UTILITIES
 * ========================================================================== */

/**
 * Validate if a string matches the size format (e.g., "512x512").
 * 
 * @param {string} size - Size string to validate
 * @returns {boolean} True if format is valid
 * 
 * @example
 * isValidSizeFormat("512x512")   // => true
 * isValidSizeFormat("512X768")   // => true (case insensitive)
 * isValidSizeFormat("512-512")   // => false
 * isValidSizeFormat("invalid")   // => false
 */
export function isValidSizeFormat(size) {
  return typeof size === "string" && REGEX_PATTERNS.SIZE_FORMAT.test(size);
}

/**
 * Sanitize seed input to only digits, max length.
 * 
 * @param {string} value - Input value
 * @returns {string} Sanitized string with only digits, max 10 chars
 * 
 * @example
 * sanitizeSeedInput("123abc456")  // => "123456"
 * sanitizeSeedInput("12345678901") // => "1234567890" (truncated to 10)
 */
export function sanitizeSeedInput(value) {
  return (value || "")
    .replace(REGEX_PATTERNS.NON_DIGIT, "")
    .slice(0, SEED_CONFIG.MAX_INPUT_LENGTH);
}

/* ============================================================================
 * FORMAT UTILITIES
 * ========================================================================== */

/**
 * Format size string with multiplication symbol.
 * 
 * @param {string} size - Size in format "512x512"
 * @returns {string} Formatted size "512×512"
 * 
 * @example
 * formatSizeDisplay("512x512") // => "512×512"
 * formatSizeDisplay("1024X768") // => "1024×768"
 */
export function formatSizeDisplay(size) {
  return String(size).replace(/x/i, "×");
}

/**
 * Strip JSON markdown fences from a string.
 * Used when parsing JSON responses that might include ```json markers.
 * 
 * @param {string} text - Text potentially containing ```json fences
 * @returns {string} Clean text without fences
 * 
 * @example
 * stripJsonFences("```json\n{\"key\": \"value\"}\n```")
 * // => "{\"key\": \"value\"}"
 */
export function stripJsonFences(text) {
  return text.replace(REGEX_PATTERNS.JSON_FENCE, "").trim();
}

/* ============================================================================
 * SERVER LABEL FORMATTING
 * ========================================================================== */

/**
 * Generate a display label for API configuration.
 * 
 * @param {object} apiConfig - API configuration object
 * @param {string[]} apiConfig.bases - Array of base URLs (for round-robin)
 * @param {string} apiConfig.single - Single base URL
 * @returns {string} Human-readable server label
 * 
 * @example
 * getServerLabel({ bases: ["api1", "api2"], single: "" })
 * // => "RR (2 backends)"
 * 
 * getServerLabel({ bases: [], single: "https://api.example.com" })
 * // => "https://api.example.com"
 * 
 * getServerLabel({ bases: [], single: "" })
 * // => "(same origin)"
 */
export function getServerLabel(apiConfig) {
  if (apiConfig.bases.length > 0) {
    return `RR (${apiConfig.bases.length} backends)`;
  }
  return apiConfig.single || "(same origin)";
}

/* ============================================================================
 * METADATA UTILITIES
 * ========================================================================== */

/**
 * Extract SR metadata from response headers.
 * 
 * @param {Response} res - Fetch Response object
 * @param {number} defaultMagnitude - Default magnitude if headers not present
 * @returns {object} SR metadata object
 * 
 * @example
 * const srMeta = extractSRMetadata(response, 2);
 * // => { magnitude: "2", passes: "2", scale: "1.5", backend: "cuda" }
 */
export function extractSRMetadata(res, defaultMagnitude) {
  const magnitude = res.headers.get("X-SR-Magnitude") || String(defaultMagnitude);
  const passes = res.headers.get("X-SR-Passes") || magnitude;
  const scale = 
    res.headers.get("X-SR-Scale-Per-Pass") ||
    res.headers.get("X-SR-Scale") ||
    null;
  const backend =
    res.headers.get("X-LCM-Backend") ||
    res.headers.get("X-Backend") ||
    res.headers.get("X-Host") ||
    null;

  return { magnitude, passes, scale, backend };
}

/**
 * Extract generation metadata from response headers.
 * 
 * @param {Response} res - Fetch Response object
 * @param {number} requestedSeed - The seed sent in the request
 * @returns {object} Generation metadata
 * 
 * @example
 * const meta = extractGenerationMetadata(response, 12345678);
 * // => { seed: 12345678, superres: true, srScale: "1.5", backend: "cuda" }
 */
export function extractGenerationMetadata(res, requestedSeed) {
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

  return {
    seed: Number(seedHdr ?? requestedSeed),
    superres: srHdr === "1",
    srScale,
    backend,
  };
}