// src/hooks/useJobQueue.js — React Binding for JobQueue

import { useSyncExternalStore, useCallback } from 'react';
import { jobQueue, PRIORITY } from '../lib/jobQueue';

const subscribe = (cb) => jobQueue.subscribe(cb);
const getSnapshot = () => jobQueue.getSnapshot();

export function useJobQueue() {
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  const enqueue = useCallback(
    (opts) => jobQueue.enqueue(opts),
    []
  );

  const cancel = useCallback(
    (id) => jobQueue.cancel(id),
    []
  );

  const cancelAll = useCallback(
    () => jobQueue.cancelAll(),
    []
  );

  return {
    ...state,
    enqueue,
    cancel,
    cancelAll,
  };
}

export { PRIORITY };
