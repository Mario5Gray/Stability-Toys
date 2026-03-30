// test-setup.js — vitest global setup
// Node 25 provides a built-in global localStorage that lacks the full Storage
// interface (no clear/getItem/setItem/removeItem). This file replaces it with
// a proper in-memory implementation so tests can use localStorage normally.

function makeInMemoryStorage() {
  let store = {};
  return {
    get length() { return Object.keys(store).length; },
    clear() { store = {}; },
    getItem(key) { return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null; },
    setItem(key, value) { store[String(key)] = String(value); },
    removeItem(key) { delete store[key]; },
    key(n) { return Object.keys(store)[n] ?? null; },
  };
}

if (typeof global !== 'undefined') {
  Object.defineProperty(global, 'localStorage', {
    value: makeInMemoryStorage(),
    writable: true,
    configurable: true,
  });
  Object.defineProperty(global, 'sessionStorage', {
    value: makeInMemoryStorage(),
    writable: true,
    configurable: true,
  });
}
