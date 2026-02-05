// src/utils/otelTelemetry.js
/* Minimal OTLP/WS JSON exporter for browser-side telemetry. */

import { wsClient } from '../lib/wsClient';
import { uuidv4 } from './uuid';

const DEFAULT_FLUSH_MS = 3000;
const MAX_BATCH = 50;
const MAX_QUEUE = 500;

function getEnv(name, fallback = '') {
  try {
    // Vite env
    return (import.meta?.env && import.meta.env[name]) || fallback;
  } catch {
    return fallback;
  }
}

const OTEL_ENABLED =
  String(getEnv('VITE_OTEL_ENABLED', 'true')).toLowerCase() !== 'false';
const OTEL_NAME_PREFIX = getEnv('VITE_OTEL_NAME_PREFIX', 'ui');
const OTEL_SAMPLE_RATE = Math.max(
  0,
  Math.min(1, Number(getEnv('VITE_OTEL_SAMPLE_RATE', '1')))
);
const OTEL_BATCH_MS = Math.max(250, Number(getEnv('VITE_OTEL_BATCH_MS', DEFAULT_FLUSH_MS)));
const OTEL_BATCH_MAX = Math.max(1, Number(getEnv('VITE_OTEL_BATCH_MAX', MAX_BATCH)));
const OTEL_QUEUE_MAX = Math.max(50, Number(getEnv('VITE_OTEL_QUEUE_MAX', MAX_QUEUE)));

const OTEL_PROXY =
  getEnv('VITE_OTEL_PROXY_ENDPOINT', '').trim();
const OTEL_ENDPOINT =
  getEnv('VITE_OTEL_EXPORTER_OTLP_ENDPOINT', '').trim();

const OTEL_SERVICE_NAME = getEnv('VITE_OTEL_SERVICE_NAME', 'lcm-sr-ui');
const OTEL_SCOPE_NAME = getEnv('VITE_OTEL_SCOPE_NAME', 'ui-telemetry');

// Kept for compatibility; we currently send via WebSocket proxy.
void OTEL_PROXY;
void OTEL_ENDPOINT;

function toHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function newTraceId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return toHex(bytes);
}

function newSpanId() {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return toHex(bytes);
}

function nowUnixNano() {
  return BigInt(Date.now()) * 1000000n;
}

function toAttributeValue(value) {
  if (typeof value === 'string') return { stringValue: value };
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return { intValue: value };
    return { doubleValue: value };
  }
  if (typeof value === 'boolean') return { boolValue: value };
  if (value == null) return { stringValue: '' };
  return { stringValue: String(value) };
}

function attrsToOtel(attrs = {}) {
  return Object.entries(attrs).map(([key, value]) => ({
    key,
    value: toAttributeValue(value),
  }));
}

let sessionId = null;
function getSessionId() {
  if (sessionId) return sessionId;
  try {
    const key = 'lcm-ui-session-id';
    const existing = localStorage.getItem(key);
    if (existing) {
      sessionId = existing;
      return sessionId;
    }
    const next = uuidv4();
    localStorage.setItem(key, next);
    sessionId = next;
    return sessionId;
  } catch {
    sessionId = uuidv4();
    return sessionId;
  }
}

const queue = [];
let flushTimer = null;
let wsReady = wsClient.connected;
let sampledSession = null;

wsClient.addEventListener('statechange', (e) => {
  wsReady = e.detail?.state === 'connected';
  if (wsReady) flush();
});

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(() => {
    flushTimer = null;
    flush();
  }, OTEL_BATCH_MS);
}

function flush() {
  if (!OTEL_ENABLED || queue.length === 0) return;
  if (!wsReady) return;

  const batch = queue.splice(0, OTEL_BATCH_MAX);
  const payload = {
    resourceSpans: [
      {
        resource: {
          attributes: attrsToOtel({
            'service.name': OTEL_SERVICE_NAME,
            'service.instance.id': getSessionId(),
          }),
        },
        scopeSpans: [
          {
            scope: { name: OTEL_SCOPE_NAME, version: '1.0.0' },
            spans: batch,
          },
        ],
      },
    ],
  };

  wsClient.send({
    type: 'telemetry:otlp',
    payload,
    contentType: 'application/json',
  });
}

function shouldSample() {
  if (sampledSession !== null) return sampledSession;
  if (OTEL_SAMPLE_RATE >= 1) {
    sampledSession = true;
    return true;
  }
  if (OTEL_SAMPLE_RATE <= 0) {
    sampledSession = false;
    return false;
  }
  sampledSession = Math.random() < OTEL_SAMPLE_RATE;
  return sampledSession;
}

function formatName(name) {
  const prefix = OTEL_NAME_PREFIX ? `${OTEL_NAME_PREFIX}.` : '';
  return `${prefix}${name}`;
}

export function emitJobEvent(eventName, attrs = {}) {
  if (!OTEL_ENABLED || !shouldSample()) return;
  const time = nowUnixNano();
  const span = {
    traceId: newTraceId(),
    spanId: newSpanId(),
    name: formatName(`job.${eventName}`),
    kind: 1, // INTERNAL
    startTimeUnixNano: String(time),
    endTimeUnixNano: String(time),
    attributes: attrsToOtel({
      'event.name': eventName,
      'event.domain': 'job',
      'ui.session_id': getSessionId(),
      ...attrs,
    }),
  };

  queue.push(span);
  if (queue.length > OTEL_QUEUE_MAX) {
    queue.splice(0, queue.length - OTEL_QUEUE_MAX);
  }
  if (queue.length >= OTEL_BATCH_MAX) {
    flush();
  } else {
    scheduleFlush();
  }
}

// TODO: add HTTP piggybacking for environments without WS telemetry.

export function emitUiEvent(eventName, attrs = {}) {
  if (!OTEL_ENABLED || !shouldSample()) return;
  const time = nowUnixNano();
  const span = {
    traceId: newTraceId(),
    spanId: newSpanId(),
    name: formatName(eventName),
    kind: 1,
    startTimeUnixNano: String(time),
    endTimeUnixNano: String(time),
    attributes: attrsToOtel({
      'event.name': eventName,
      'event.domain': 'ui',
      'ui.session_id': getSessionId(),
      ...attrs,
    }),
  };

  queue.push(span);
  if (queue.length > OTEL_QUEUE_MAX) {
    queue.splice(0, queue.length - OTEL_QUEUE_MAX);
  }
  if (queue.length >= OTEL_BATCH_MAX) {
    flush();
  } else {
    scheduleFlush();
  }
}
