// src/lib/wsClient.js — Singleton WebSocket Client
//
// Auto-reconnecting WS with typed message dispatch.
// Follows the same subscribe/getSnapshot pattern as jobQueue.js
// for useSyncExternalStore compatibility.

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 15000;
const PING_INTERVAL_MS = 25000;

let _corrId = 1;

/** Generate a short correlation ID for request/response pairing. */
export function nextCorrId() {
  return `c${_corrId++}`;
}

class WSClient extends EventTarget {
  constructor() {
    super();
    this._ws = null;
    this._url = null;
    this._state = 'disconnected'; // disconnected | connecting | connected
    this._reconnectTimer = null;
    this._pingTimer = null;
    this._reconnectDelay = RECONNECT_BASE_MS;
    this._snapshot = null;
    this._systemStatus = null;
    this._intentionalClose = false;
  }

  // ---------- lifecycle ----------

  /**
   * Connect to the WS endpoint.
   * @param {string} [url] - ws:// or wss:// URL. Defaults to same-origin /v1/ws.
   */
  connect(url) {    
    if (this._ws && (this._state === 'connected' || this._state === 'connecting')) {
      return; // already connected/connecting
    }

    if (!url) {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      url = `${proto}//${location.host}/v1/ws`;
    }

    this._url = url;
    this._intentionalClose = false;
    this._openSocket();
  }

  disconnect() {
    this._intentionalClose = true;
    clearTimeout(this._reconnectTimer);
    clearInterval(this._pingTimer);
    if (this._ws) {
      this._ws.close(1000, 'client disconnect');
      this._ws = null;
    }
    this._setState('disconnected');
  }

  // ---------- send ----------

  /**
   * Send a JSON message. Returns the correlation ID.
   * @param {object} msg - Message envelope (must have `type`).
   * @returns {string} correlation ID
   */
  send(msg) {
    const id = msg.id || nextCorrId();
    const envelope = { ...msg, id };
    
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      const jsonReq = JSON.stringify(envelope)
      console.log("[WS] sending now request: " + jsonReq) ;
      this._ws.send(jsonReq);
    } else {
      console.warn('[WS] send while not connected, dropping:', envelope.type);
    }
    return id;
  }

  // ---------- subscribe (useSyncExternalStore) ----------

  subscribe(callback) {
    const handler = () => callback();
    for (const evt of ['statechange', 'message', 'status']) {
      this.addEventListener(evt, handler);
    }
    return () => {
      for (const evt of ['statechange', 'message', 'status']) {
        this.removeEventListener(evt, handler);
      }
    };
  }

  getSnapshot() {
    if (!this._snapshot) {
      this._snapshot = Object.freeze({
        state: this._state,
        systemStatus: this._systemStatus,
      });
    }
    return this._snapshot;
  }

  // ---------- typed subscriptions ----------

  /**
   * Subscribe to messages of a specific type.
   * @param {string} type - Message type (e.g. "job:complete")
   * @param {function} callback - Called with the message payload
   * @returns {function} unsubscribe
   */
  on(type, callback) {
    const handler = (e) => {
      if (e.detail?.type === type) callback(e.detail);
    };
    this.addEventListener('message', handler);
    return () => this.removeEventListener('message', handler);
  }

  /**
   * Wait for a single message matching a predicate.
   * @param {function} predicate - (msg) => boolean
   * @param {number} [timeoutMs=60000] - Timeout
   * @returns {Promise<object>}
   */
  waitFor(predicate, timeoutMs = 60000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        unsub();
        reject(new Error('WS waitFor timeout'));
      }, timeoutMs);

      const unsub = this.on('*_all', (msg) => {
        // We need a different approach — listen on raw message event
      });

      // Use raw listener
      const handler = (e) => {
        const msg = e.detail;
        if (msg && predicate(msg)) {
          clearTimeout(timer);
          this.removeEventListener('message', handler);
          resolve(msg);
        }
      };
      this.addEventListener('message', handler);
    });
  }

  // ---------- getters ----------

  get connected() { return this._state === 'connected'; }
  get state() { return this._state; }
  get systemStatus() { return this._systemStatus; }

  // ---------- internals ----------

  _openSocket() {
    this._setState('connecting');

    const ws = new WebSocket(this._url);
    this._ws = ws;
    
    ws.onopen = () => {
      this._reconnectDelay = RECONNECT_BASE_MS;
      this._setState('connected');
      this._startPing();
      this._emit('statechange', { state: 'connected' });
    };

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      // Handle system:status internally
      if (msg.type === 'system:status') {
        this._systemStatus = Object.freeze(msg);
        this._invalidate();
        this._emit('status', msg);
        return;
      }

      // Handle pong silently
      if (msg.type === 'pong') return;

      // Dispatch to listeners
      this._emit('message', msg);
    };

    ws.onclose = (event) => {
      this._ws = null;
      clearInterval(this._pingTimer);
      this._setState('disconnected');

      if (!this._intentionalClose) {
        this._scheduleReconnect();
      }
    };

    ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  _scheduleReconnect() {
    clearTimeout(this._reconnectTimer);
    const delay = Math.min(this._reconnectDelay, RECONNECT_MAX_MS);
    this._reconnectDelay = Math.min(delay * 1.5, RECONNECT_MAX_MS);
    this._reconnectTimer = setTimeout(() => {
      if (this._url && !this._intentionalClose) {
        this._openSocket();
      }
    }, delay);
  }

  _startPing() {
    clearInterval(this._pingTimer);
    this._pingTimer = setInterval(() => {
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, PING_INTERVAL_MS);
  }

  _setState(s) {
    if (this._state === s) return;
    this._state = s;
    this._invalidate();
    this._emit('statechange', { state: s });
  }

  _invalidate() {
    this._snapshot = null;
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

export const wsClient = new WSClient();
