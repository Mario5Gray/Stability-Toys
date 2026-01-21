// src/utils/constants.js

/* ============================================================================
 * SIZE OPTIONS
 * ========================================================================== */
export const SIZE_OPTIONS = [
  "256x256",
  "512x512",
  "640x360",
  "768x768",
  "960x540",
  "1024x1024",
  "512x768",
  "768x512",
];

export const DEFAULT_SIZE = "512x512";

/* ============================================================================
 * GENERATION PARAMETERS
 * ========================================================================== */
export const STEPS_CONFIG = {
  MIN: 0, // LCM allows 0 steps for latent lock
  MAX: 20,
  DEFAULT: 4,
  SERVER_MAX: 50, // Server allows up to 50
  LCM_TYPICAL_MIN: 2,
  LCM_TYPICAL_MAX: 8,
  LATENT_LOCK: 0, // Zero steps for latent encoding
  LATENT_COUSINS_MIN: 7, // Minimum for cousin exploration
};

export const CFG_CONFIG = {
  MIN: 0,
  MAX: 5,
  DEFAULT: 1.0,
  STEP: 0.1,
  LCM_TYPICAL: 1.0,
  ABSOLUTE_MAX: 20, // Used in runGenerate clamp
  LATENT_LOCK: 8.0, // High CFG for latent lock
  LATENT_COUSINS: 0.0, // Zero CFG for cousin exploration
};

export const DENOISE_CONFIG = {
  MIN: 0.1,
  MAX: 1.0,
  DEFAULT: 1.0,
  STEP: 0.05,
  COUSINS_DETAILED: 0.7,
  COUSINS_SUBTLE: 0.5,
  COUSINS_WILD: 0.9,
};

/* ============================================================================
 * SUPER-RESOLUTION
 * ========================================================================== */
export const SR_CONFIG = {
  MIN: 0,
  MAX: 4,
  DEFAULT: 0,
  BACKEND_MAX: 3, // Backend magnitude max is 3
  DEFAULT_MAGNITUDE: 2,
};

export const SR_MAGNITUDE_OPTIONS = [
  { value: "1", label: "1 (1 pass)" },
  { value: "2", label: "2 (default)" },
  { value: "3", label: "3 (3 passes)" },
];

/* ============================================================================
 * SEED CONFIGURATION
 * ========================================================================== */
export const SEED_CONFIG = {
  MIN: 0,
  MAX: 2 ** 31 - 1,
  DIGIT_COUNT: 8,
  RANDOM_MAX: 100_000_000,
  MAX_INPUT_LENGTH: 10,
};

export const SEED_MODES = {
  RANDOM: "random",
  FIXED: "fixed",
};

/* ============================================================================
 * PROMPT DELTAS (Quick Enhancement Buttons)
 * ========================================================================== */
export const PROMPT_DELTAS = [
  "more cinematic",
  "shallow depth of field",
  "soft rim light",
  "film grain",
  "wider shot",
  "more detail",
  "simpler background",
];

/* ============================================================================
 * UI MESSAGES & LABELS
 * ========================================================================== */
export const UI_MESSAGES = {
  INITIAL_SYSTEM: "Ready. Type a prompt and hit Send to generate a PNG. Toggle SR if your server supports it.",
  GENERATING: "Generating…",
  REGENERATING: "Regenerating…",
  SUPER_RESOLVING: "Super-resolving…",
  CANCELED: "Canceled.",
  COPY_PROMPT_TIP: "Copy current prompt to clipboard",
  COPIED: "Copied!",
  COPY_PROMPT: "Copy prompt",
  KEYBOARD_TIP: "Tip: press Ctrl/⌘ + Enter to send.",
  SELECT_IMAGE_TIP: "Tip: click an image to select it. Sliders will edit that image's settings.",
  CLICK_TO_SELECT: "Click to select",
  CLICK_TO_LOAD_SETTINGS: "Click to load these settings",
  MORE_SCROLL: "More",
};

/* ============================================================================
 * FILE FORMATS & UPLOAD
 * ========================================================================== */
export const FILE_CONFIG = {
  OUTPUT_FORMAT: "png",
  SUPERRES_FORMAT: "png",
  QUALITY: 92,
  ACCEPT_TYPES: "image/*",
};

/* ============================================================================
 * API ENDPOINTS
 * ========================================================================== */
export const API_ENDPOINTS = {
  GENERATE: "/generate",
  SUPERRES: "/superres",
};

/* ============================================================================
 * HTTP HEADERS
 * ========================================================================== */
export const RESPONSE_HEADERS = {
  SEED: "X-Seed",
  SUPERRES: "X-SuperRes",
  SR_PASSES: "X-SR-Passes",
  SR_SCALE: "X-SR-Scale",
  SR_SCALE_PER_PASS: "X-SR-Scale-Per-Pass",
  SR_MAGNITUDE: "X-SR-Magnitude",
  BACKEND: "X-LCM-Backend",
  BACKEND_ALT: "X-Backend",
  HOST: "X-Host",
  CONTENT_TYPE: "content-type",
};

/* ============================================================================
 * SCROLL BEHAVIOR
 * ========================================================================== */
export const SCROLL_CONFIG = {
  NEAR_BOTTOM_THRESHOLD_PX: 80,
  HINT_THRESHOLD_PX: 6,
  BEHAVIOR: "smooth",
  BLOCK_CENTER: "center",
  BLOCK_END: "end",
};

/* ============================================================================
 * DEBOUNCE TIMERS
 * ========================================================================== */
export const DEBOUNCE_CONFIG = {
  REGEN_DELAY_MS: 180,
  COPY_FEEDBACK_MS: 900,
};

/* ============================================================================
 * CSS CLASSES (Reusable)
 * ========================================================================== */
export const CSS_CLASSES = {
  SELECT_TRIGGER: "rounded-2xl bg-background text-foreground shadow-sm border",
  SELECT_CONTENT: "!bg-white !text-black !opacity-100 shadow-xl border border-black/10",
  SELECT_ITEM: "!bg-white !text-black data-[highlighted]:!bg-black/5",
  INPUT: "rounded-2xl bg-background text-foreground shadow-sm border",
  SLIDER: 
    "[&_[data-orientation=horizontal]]:bg-muted " +
    "[&_[role=slider]]:bg-foreground " +
    "[&_[role=slider]]:border-foreground " +
    "[&_[role=slider]]:shadow",
};

/* ============================================================================
 * ENVIRONMENT VARIABLE KEYS
 * ========================================================================== */
export const ENV_KEYS = {
  VITE_API_BASES: "VITE_API_BASES",
  VITE_API_BASE: "VITE_API_BASE",
  REACT_APP_API_BASE: "REACT_APP_API_BASE",
};

/* ============================================================================
 * BADGE LABELS
 * ========================================================================== */
export const BADGE_LABELS = {
  ENDPOINT: "/generate",
  FORMAT: "PNG",
  SR_OFF: "SR off",
};

/* ============================================================================
 * DEFAULT PROMPTS
 * ========================================================================== */
export const DEFAULT_PROMPT = "a cinematic photograph of a futuristic city at sunset";

/* ============================================================================
 * ERROR MESSAGES
 * ========================================================================== */
export const ERROR_MESSAGES = {
  REQUEST_FAILED: "Request failed",
  BODY_DISTURBED: "Body is disturbed or locked",
  READ_FAILED: "Read failed",
};

/* ============================================================================
 * FALLBACK TEXTAREA STYLES (for copyToClipboard)
 * ========================================================================== */
export const FALLBACK_TEXTAREA_STYLES = {
  position: "fixed",
  left: "-9999px",
};

/* ============================================================================
 * MESSAGE KINDS (Type Constants)
 * ========================================================================== */
export const MESSAGE_KINDS = {
  SYSTEM: "system",
  TEXT: "text",
  IMAGE: "image",
  PENDING: "pending",
  ERROR: "error",
};

export const MESSAGE_ROLES = {
  USER: "user",
  ASSISTANT: "assistant",
};

/* ============================================================================
 * INGEST TYPES (for meta.ingest)
 * ========================================================================== */
export const INGEST_TYPES = {
  SUPERRES: "superres",
};

/* ============================================================================
 * REGEX PATTERNS
 * ========================================================================== */
export const REGEX_PATTERNS = {
  SIZE_FORMAT: /^\d+x\d+$/i,
  NON_DIGIT: /[^\d]/g,
  ENV_SPLIT: /[;,]/g,
  TRAILING_SLASH: /\/+$/,
  JSON_FENCE: /```json|```/g,
};

/* ============================================================================
 * CONTENT TYPE CHECKS
 * ========================================================================== */
export const CONTENT_TYPES = {
  JSON: "application/json",
  PNG: "image/png",
};

/* ============================================================================
 * ABORT ERROR NAME
 * ========================================================================== */
export const ABORT_ERROR_NAME = "AbortError";

/* ============================================================================
 * SERVER LABEL FORMATTING
 * ========================================================================== */
export const SERVER_LABELS = {
  SAME_ORIGIN: "(same origin)",
  ROUND_ROBIN_PREFIX: "RR",
};

/* ============================================================================
 * NUMERIC LIMITS
 * ========================================================================== */
export const NUMERIC_LIMITS = {
  SEED_DIGIT_MAX: 100_000_000,
  SEED_INT_MAX: 2 ** 31 - 1,
};