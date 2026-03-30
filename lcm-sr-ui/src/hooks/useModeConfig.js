import { useCallback, useEffect, useRef, useState } from 'react';
import { createApiClient, createApiConfig } from '../utils/api';
import { getActiveMode } from '../utils/generationControls';

export function useModeConfig() {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);
  const [isSwitching, setIsSwitching] = useState(false);
  const apiClientRef = useRef(null);

  if (!apiClientRef.current) {
    apiClientRef.current = createApiClient(createApiConfig());
  }

  const api = apiClientRef.current;

  const loadModes = useCallback(async () => {
    try {
      const modesRes = await api.fetchGet('/api/modes');
      setConfig({
        default_mode: modesRes.default_mode,
        modes: modesRes.modes,
      });
      setError(null);
    } catch (e) {
      setError(e.message || 'Failed to load modes');
    }
  }, [api]);

  useEffect(() => {
    loadModes();
  }, [loadModes]);

  const switchMode = useCallback(
    async (name) => {
      if (!config || name === config.default_mode) return;
      setIsSwitching(true);
      try {
        await api.fetchPost('/api/modes/switch', { mode: name });
        setConfig((prev) => (prev ? { ...prev, default_mode: name } : prev));
        setError(null);
      } catch (e) {
        setError(e.message || 'Failed to switch mode');
        throw e;
      } finally {
        setIsSwitching(false);
      }
    },
    [api, config]
  );

  const activeModeName = config?.default_mode || null;
  const activeMode = getActiveMode(config, activeModeName);

  return {
    config,
    activeModeName,
    activeMode,
    isSwitching,
    error,
    loadModes,
    switchMode,
  };
}
