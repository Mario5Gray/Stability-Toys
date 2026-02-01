// src/hooks/useWs.js — React binding for wsClient
//
// Mirrors useJobQueue.js pattern: useSyncExternalStore wrapper.

import { useSyncExternalStore, useCallback, useEffect } from 'react';
import { wsClient } from '../lib/wsClient';

const subscribe = (cb) => wsClient.subscribe(cb);
const getSnapshot = () => wsClient.getSnapshot();

/**
 * React hook for WebSocket connection state + send.
 *
 * @param {boolean} [autoConnect=true] - Connect on mount
 * @returns {{ state, systemStatus, connected, send, connect, disconnect }}
 */
export function useWs(autoConnect = true) {
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  useEffect(() => {
    if (autoConnect) {
      wsClient.connect();
    }
    // Don't disconnect on unmount — singleton persists across re-renders.
    // Only disconnect on explicit call.
  }, [autoConnect]);

  const send = useCallback((msg) => wsClient.send(msg), []);
  const connect = useCallback((url) => wsClient.connect(url), []);
  const disconnect = useCallback(() => wsClient.disconnect(), []);

  return {
    ...snap,
    connected: snap.state === 'connected',
    send,
    connect,
    disconnect,
  };
}
