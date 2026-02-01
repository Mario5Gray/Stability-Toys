// src/hooks/useWsSubscription.js — Subscribe to specific WS message types

import { useEffect, useRef } from 'react';
import { wsClient } from '../lib/wsClient';

/**
 * Subscribe to WS messages of a specific type.
 * Callback is stable-ref'd so it won't cause re-subscribe on every render.
 *
 * @param {string} type - Message type (e.g. "job:complete", "job:progress")
 * @param {function} callback - Called with the full message object
 */
export function useWsSubscription(type, callback) {
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    return wsClient.on(type, (msg) => cbRef.current(msg));
  }, [type]);
}

/**
 * Subscribe to ALL WS messages (useful for debugging or routing).
 *
 * @param {function} callback - Called with every message
 */
export function useWsMessages(callback) {
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    const handler = (e) => cbRef.current(e.detail);
    wsClient.addEventListener('message', handler);
    return () => wsClient.removeEventListener('message', handler);
  }, []);
}
