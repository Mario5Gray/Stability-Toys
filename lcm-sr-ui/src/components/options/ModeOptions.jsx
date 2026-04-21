import React, { useState } from 'react';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../ui/select';
import { CSS_CLASSES } from '../../utils/constants';
import { emitUiEvent } from '../../utils/otelTelemetry';

export default function ModeOptions({
  modeConfig,
  activeModeName,
  onModeChange,
  isSwitching = false,
  error = null,
}) {
  const [success, setSuccess] = useState(null);

  const setDefaultMode = async (name) => {
    if (!modeConfig) return;
    if (name === activeModeName) return;

    try {
      emitUiEvent('mode.select', {
        'ui.component': 'ModeOptions',
        'ui.action': 'select',
        'ui.value': name,
      });
      await onModeChange?.(name);
      setSuccess(`"${name}" mode is now active.`);
      setTimeout(() => setSuccess(null), 2000);
    } catch (_e) {
      // parent owns persistent error state
    }
  };

  if (!modeConfig) {
    return <div className="p-4 text-center text-muted-foreground">Mode selection unavailable.</div>;
  }

  const modeNames = Object.keys(modeConfig.modes || {});

  return (
    <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
      <div className="space-y-1">
        <span className="text-sm font-medium optionTop">Mode</span>
        <Select value={activeModeName} onValueChange={setDefaultMode} disabled={isSwitching}>
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
