import React, { createContext, useCallback, useContext, useReducer, useRef } from 'react';

const COMPLETE_LINGER_MS = 2000;
const ERROR_LINGER_MS = 5000;

function operationsReducer(state, action) {
  switch (action.type) {
    case 'UPSERT': {
      const { record } = action;
      const exists = state.operations.has(record.key);
      const next = new Map(state.operations);
      next.set(record.key, { ...(next.get(record.key) ?? {}), ...record });
      return {
        operations: next,
        order: exists ? state.order : [...state.order, record.key],
      };
    }
    case 'REMOVE': {
      const next = new Map(state.operations);
      next.delete(action.key);
      return { operations: next, order: state.order.filter((k) => k !== action.key) };
    }
    default:
      return state;
  }
}

const OperationsStoreContext = createContext(null);
const OperationsDispatchContext = createContext(null);

export function OperationsProvider({ children }) {
  const [state, rawDispatch] = useReducer(operationsReducer, {
    operations: new Map(),
    order: [],
  });
  const expiryTimers = useRef(new Map());

  const scheduleRemoval = useCallback((key, delayMs) => {
    const existing = expiryTimers.current.get(key);
    if (existing) clearTimeout(existing);
    const id = setTimeout(() => {
      rawDispatch({ type: 'REMOVE', key });
      expiryTimers.current.delete(key);
    }, delayMs);
    expiryTimers.current.set(key, id);
  }, []);

  const dispatch = useCallback((action) => {
    rawDispatch(action);
    if (action.type === 'UPSERT') {
      const { tone, key } = action.record;
      if (tone === 'complete') {
        scheduleRemoval(key, COMPLETE_LINGER_MS);
      } else if (tone === 'error') {
        scheduleRemoval(key, ERROR_LINGER_MS);
      } else if (tone != null) {
        const existing = expiryTimers.current.get(key);
        if (existing) { clearTimeout(existing); expiryTimers.current.delete(key); }
      }
    } else if (action.type === 'REMOVE') {
      const existing = expiryTimers.current.get(action.key);
      if (existing) { clearTimeout(existing); expiryTimers.current.delete(action.key); }
    }
  }, [scheduleRemoval]);

  return (
    <OperationsStoreContext.Provider value={state}>
      <OperationsDispatchContext.Provider value={dispatch}>
        {children}
      </OperationsDispatchContext.Provider>
    </OperationsStoreContext.Provider>
  );
}

export function useOperationsStore() {
  return useContext(OperationsStoreContext);
}

export function useOperationsController() {
  const dispatch = useContext(OperationsDispatchContext);

  const start = useCallback((init) => {
    const cancelFn =
      init.cancellable && typeof init.cancelFn === 'function'
        ? () => {
            init.cancelFn();
            dispatch({ type: 'REMOVE', key: init.key });
          }
        : null;

    const record = {
      key: init.key,
      kind: init.kind ?? 'generic',
      icon: init.icon ?? null,
      tone: init.tone ?? 'active',
      text: init.text ?? '',
      detail: init.detail ?? null,
      progress: init.progress ?? null,
      cancellable: Boolean(cancelFn),
      cancelFn,
      createdAt: Date.now(),
    };
    dispatch({ type: 'UPSERT', record });

    const handle = {
      setText:     (text)     => dispatch({ type: 'UPSERT', record: { key: record.key, text } }),
      setDetail:   (detail)   => dispatch({ type: 'UPSERT', record: { key: record.key, detail } }),
      setProgress: (progress) => dispatch({ type: 'UPSERT', record: { key: record.key, progress } }),
      setTone:     (tone)     => dispatch({ type: 'UPSERT', record: { key: record.key, tone } }),
      complete: ({ text } = {}) =>
        dispatch({
          type: 'UPSERT',
          record: { key: record.key, tone: 'complete', text: text ?? 'Done', detail: null, progress: null },
        }),
      error: ({ text } = {}) =>
        dispatch({ type: 'UPSERT', record: { key: record.key, tone: 'error', text: text ?? 'Error' } }),
      remove: () => dispatch({ type: 'REMOVE', key: record.key }),
    };

    if (cancelFn) handle.cancel = cancelFn;
    return handle;
  }, [dispatch]);

  return { start };
}
