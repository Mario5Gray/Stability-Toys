import React from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';

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

  return (
    <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
      <Label>Advisor</Label>

      <label className="flex items-center justify-between text-sm">
        <span>Auto-Advice</span>
        <input
          aria-label="Auto advice"
          type="checkbox"
          checked={Boolean(state?.auto_advice)}
          onChange={(e) => onAutoAdviceChange(e.target.checked)}
        />
      </label>

      <div className="space-y-2">
        <Label htmlFor="advisor-temperature">Temperature</Label>
        <input
          id="advisor-temperature"
          aria-label="Advisor temperature"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={state?.temperature ?? 0.4}
          onChange={(e) => onTemperatureChange(Number(e.target.value))}
        />
      </div>

      {hasMaximumLen && (
        <div className="space-y-2">
          <Label htmlFor="advisor-length">Length</Label>
          <input
            id="advisor-length"
            aria-label="Advisor length"
            type="range"
            min="0"
            max={maximumLen}
            step="1"
            value={state?.length_limit ?? 0}
            onChange={(e) => onLengthChange(Number(e.target.value))}
          />
        </div>
      )}

      <div className="text-xs" data-status={state?.status || 'idle'}>
        {state?.status === 'building'
          ? 'Building digest...'
          : state?.status === 'error'
            ? state?.error_message || 'Advisor error'
            : state?.updated_at
              ? `Updated ${new Date(state.updated_at).toLocaleString()}`
              : 'No digest yet'}
      </div>

      <Textarea
        aria-label="Advisor advice"
        value={state?.advice_text ?? ''}
        onChange={(e) => onAdviceChange(e.target.value)}
        className="min-h-[120px] resize-none rounded-2xl"
      />

      <div className="space-y-2">
        <Label>Apply Mode</Label>
        <select
          aria-label="Apply advice mode"
          value={applyMode}
          onChange={(e) => onApplyModeChange(e.target.value)}
          className="h-10 w-full rounded-2xl border border-input bg-background px-3 py-2 text-sm"
        >
          <option value="append">Append</option>
          <option value="replace">Replace</option>
        </select>
      </div>

      <div className="flex gap-2">
        <Button type="button" onClick={onRebuild}>
          Rebuild Advisor
        </Button>
        <Button type="button" variant="secondary" onClick={onResetToDigest}>
          Reset To Digest
        </Button>
        <Button type="button" onClick={() => onApply(applyMode)} disabled={!state?.advice_text}>
          Apply Advice
        </Button>
      </div>
    </div>
  );
}
