import React from 'react';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Check, RefreshCw, RotateCcw } from 'lucide-react';
import { PanelActionBar } from '@/components/ui/PanelActionBar';

export function AdvisorPanel({
  state,
  maximumLen,
  onAutoAdviceChange,
  onTemperatureChange,
  onLengthChange,
  onAdviceChange,
  onResetToDigest,
  onRebuild,
  onApply,
  applyMode,
  onApplyModeChange,
}) {
  const hasMaximumLen = Number.isFinite(Number(maximumLen)) && Number(maximumLen) > 0;

  const applySubtext = applyMode === 'replace' ? 'Replace prompt' : 'Append to prompt';

  const statusText =
    state?.status === 'error'
      ? state?.error_message || 'Advisor error'
      : state?.updated_at
        ? `Updated ${new Date(state.updated_at).toLocaleString()}`
        : 'No digest yet';

  return (
    <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
      <Label className="optionTop">Advisor</Label>

      <label className="flex items-center justify-between text-sm">
        <span>Auto-Advice</span>
          <Switch aria-label="Auto advice"
            checked={Boolean(state?.auto_advice)}
            onCheckedChange={(e) => onAutoAdviceChange(e.target.checked)}
          />
      </label>

      <div className="space-y-2">
        <Label>Temperature</Label>
        <Slider
          aria-label="Advisor temperature"
          min={0}
          max={1}
          step={0.05}
          value={[state?.temperature ?? 0.4]}
          onValueChange={([v]) => onTemperatureChange(v)}
        />
      </div>

      {hasMaximumLen && (
        <div className="space-y-2">
          <Label>Length</Label>
          <Slider
            aria-label="Advisor length"
            min={0}
            max={maximumLen}
            step={1}
            value={[state?.length_limit ?? 0]}
            onValueChange={([v]) => onLengthChange(v)}
          />
        </div>
      )}

      <div className="text-xs text-muted-foreground" data-status={state?.status || 'idle'}>
        {statusText}
      </div>

      <Textarea
        aria-label="Advisor advice"
        value={state?.advice_text ?? ''}
        onChange={(e) => onAdviceChange(e.target.value)}
        className="min-h-[120px] resize-none rounded-2xl"
      />

      {/* Apply mode grouped with the Apply action */}
      <div className="flex items-center gap-2 text-sm">
        <Label htmlFor="apply-mode-select" className="shrink-0">Apply as</Label>
        <select
          id="apply-mode-select"
          aria-label="Apply advice mode"
          value={applyMode}
          onChange={(e) => onApplyModeChange(e.target.value)}
          className="flex-1 h-8 rounded-lg border border-gray-300 bg-white text-gray-800 px-2 py-1 text-sm dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100"
        >
          <option value="append">Append to prompt</option>
          <option value="replace">Replace prompt</option>
        </select>
      </div>

      <PanelActionBar
        primary={{
          icon: <Check className="h-4 w-4" />,
          label: 'Apply',
          subtext: applySubtext,
          onClick: () => onApply(applyMode),
          disabled: !state?.advice_text,
        }}
        secondary={[
          {
            icon: <RefreshCw className="h-4 w-4" />,
            label: 'Rebuild',
            subtext: 'Refresh digest from gallery',
            onClick: onRebuild,
          },
          {
            icon: <RotateCcw className="h-4 w-4" />,
            label: 'Reset',
            subtext: 'Restore digest text',
            onClick: onResetToDigest,
          },
        ]}
      />
    </div>
  );
}
