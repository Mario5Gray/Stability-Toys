import { useCallback, useEffect, useRef, useState } from 'react';
import { createApiClient, createApiConfig } from '../utils/api';
import { getActiveMode } from '../utils/generationControls';

const STATUS_POLL_INTERVAL_MS = 5000;

export function useModeConfig() {
  const [config, setConfig] = useState(null);
  const [runtimeStatus, setRuntimeStatus] = useState(null);
  const [error, setError] = useState(null);
  const [isSwitching, setIsSwitching] = useState(false);
  const apiClientRef = useRef(null);

  if (!apiClientRef.current) {
    apiClientRef.current = createApiClient(createApiConfig());
  }

  const api = apiClientRef.current;

  const refreshStatus = useCallback(async () => {
    try {
      const statusRes = await api.fetchGet('/api/models/status');
      setRuntimeStatus(statusRes || null);
      setError(null);
      return statusRes;
    } catch (e) {
      setRuntimeStatus(null);
      setError(e.message || 'Failed to load runtime status');
      throw e;
    }
  }, [api]);

  const loadModes = useCallback(async () => {
    try {
      const modesRes = await api.fetchGet('/api/modes');
      setConfig({
        default_mode: modesRes.default_mode,
        resolution_sets: modesRes.resolution_sets,
        modes: modesRes.modes,
      });
    } catch (e) {
      setError(e.message || 'Failed to load modes');
      return;
    }

    try {
      await refreshStatus();
      setError(null);
    } catch {
      // refreshStatus already recorded the runtime status failure.
    }
  }, [api, refreshStatus]);

  useEffect(() => {
    loadModes();
  }, [loadModes]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshStatus();
    }, STATUS_POLL_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, [refreshStatus]);

  const switchMode = useCallback(
    async (name) => {
      const currentRuntimeMode = runtimeStatus?.current_mode ?? null;
      if (!config || (currentRuntimeMode && name === currentRuntimeMode)) return;
      setIsSwitching(true);
      try {
        await api.fetchPost('/api/modes/switch', { mode: name });
        await refreshStatus();
        setError(null);
      } catch (e) {
        setError(e.message || 'Failed to switch mode');
        throw e;
      } finally {
        setIsSwitching(false);
      }
    },
    [api, config, refreshStatus, runtimeStatus?.current_mode]
  );

  const defaultModeName = config?.default_mode || null;
  const activeModeName = runtimeStatus?.current_mode ?? null;
  const activeMode = getActiveMode(config, activeModeName);
  const isLoaded = Boolean(runtimeStatus?.is_loaded);

  const reloadActiveModel = useCallback(async () => {
    try {
      const result = await api.fetchPost('/api/models/reload');
      setError(null);
      return result;
    } catch (e) {
      setError(e.message || 'Failed to reload active model');
      throw e;
    }
  }, [api]);

  const freeVram = useCallback(async () => {
    try {
      const result = await api.fetchPost('/api/models/free-vram');
      setError(null);
      return result;
    } catch (e) {
      setError(e.message || 'Failed to free VRAM');
      throw e;
    }
  }, [api]);

  return {
    config,
    defaultModeName,
    activeModeName,
    activeMode,
    runtimeStatus,
    isLoaded,
    isSwitching,
    error,
    loadModes,
    refreshStatus,
    switchMode,
    reloadActiveModel,
    freeVram,
  };
}
