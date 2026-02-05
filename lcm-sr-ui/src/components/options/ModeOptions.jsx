import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../ui/select';
import { createApiClient, createApiConfig } from '../../utils/api';
import { CSS_CLASSES } from '../../utils/constants';

export default function ModeOptions() {
  const [config, setConfig] = useState(null); // { default_mode, modes }
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isSwitching, setIsSwitching] = useState(false);
  const apiClientRef = useRef(null);
  
  if (!apiClientRef.current) {
    const apiConfig = createApiConfig();
    apiClientRef.current = createApiClient(apiConfig);
  }

  const api = apiClientRef.current;

  const load = useCallback(async () => {
    try {
      const modesRes = await api.fetchGet('/api/modes');

      setConfig({
        default_mode: modesRes.default_mode,
        modes: modesRes.modes,
      });
      
      setError(null);
    } catch (e) {
      console.error('Fetching modes aborted:', e);
      setError(e.message || 'Failed to load modes');
    }
  }, [api]);

  useEffect(() => { load(); }, [load]);

  const setDefaultMode = async (name) => {
    if (!config) return;
    if (name === config.default_mode) return;

    try {
      setIsSwitching(true);
      await api.fetchPost('/api/modes/switch', {
        mode: name,
      });

      setSuccess(`"${name}" mode is now active.`);
      setTimeout(() => setSuccess(null), 2000);
      setConfig((prev) => (prev ? { ...prev, default_mode: name } : prev));
    } catch (e) {
      setError(e.message || 'Failed to switch mode');
    } finally {
      setIsSwitching(false);
    }
  };

  if (!config) {
    return <div className="p-4 text-center text-muted-foreground">Mode selection unavailable.</div>;
  }

  const modeNames = Object.keys(config.modes);
  return (
    <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
      <div className="space-y-1">
        <span className="text-sm font-medium">Mode</span>
        <Select value={config.default_mode} onValueChange={setDefaultMode} disabled={isSwitching}>
          <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
            <SelectValue placeholder="Select Mode" />
          </SelectTrigger>
          <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
            {modeNames.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {error && <div className="text-xs text-red-600">{error}</div>}
        {success && <div className="text-xs text-green-700">{success}</div>}
      </div>
    </div>
  );
}
