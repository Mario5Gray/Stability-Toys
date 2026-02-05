// src/components/options/DreamControls.jsx

import React, { useState } from 'react';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Heart, Pause, Play } from 'lucide-react';
import { emitUiEvent } from '@/utils/otelTelemetry';

/**
 * Dream Mode controls component.
 * Allows users to enable stochastic image generation with guided evolution.
 * 
 * @param {object} props
 * @param {boolean} props.isDreaming - Whether dream mode is active
 * @param {number} props.dreamTemperature - Mutation strength (0-1)
 * @param {number} props.dreamInterval - MS between dreams
 * @param {function} props.onStartDreaming - Start dream mode callback
 * @param {function} props.onStopDreaming - Stop dream mode callback
 * @param {function} props.onGuideDream - Guide dream toward params callback
 * @param {function} props.onTemperatureChange - Temperature change callback
 * @param {function} props.onIntervalChange - Interval change callback
 * @param {object|null} props.selectedParams - Currently selected image params
 * @param {object} props.baseParams - Base params to start dreaming from
 */
export function DreamControls({
  isDreaming,
  dreamTemperature,
  dreamInterval,
  onStartDreaming,
  onStopDreaming,
  onGuideDream,
  onTemperatureChange,
  onIntervalChange,
  selectedParams,
  baseParams,
}) {
  const [isGuided, setIsGuided] = useState(false);

  const handleGuideDream = () => {
    onGuideDream(selectedParams);
    setIsGuided(true);
    setTimeout(() => setIsGuided(false), 1500);
  };

  const temperatureLabel =
    dreamTemperature < 0.3
      ? 'Subtle variations'
      : dreamTemperature < 0.6
      ? 'Moderate exploration'
      : 'Wild experimentation';

  const intervalSeconds = Math.round(dreamInterval / 1000);

  return (
    <div className="option-panel-area space-y-3 rounded-2xl border p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-purple-600" />
          <Label className="text-base font-semibold">Dream Mode</Label>
          {isDreaming && (
            <Badge variant="secondary" className="animate-pulse">
              Dreaming...
            </Badge>
          )}
        </div>
        <Switch
          checked={isDreaming}
          onCheckedChange={(checked) => {
            emitUiEvent('dream.toggle', {
              'ui.component': 'DreamControls',
              'ui.action': 'toggle',
              'ui.value': checked ? 'on' : 'off',
            });
            if (checked) {
              onStartDreaming(baseParams);
            } else {
              onStopDreaming();
            }
          }}
        />
      </div>

      {/* Description */}
      <p className="text-xs text-muted-foreground">
        Let the AI continuously explore variations. Guide it toward what looks good to you.
      </p>

      {isDreaming && (
        <>
          {/* Temperature Control */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Temperature</Label>
              <span className="text-xs text-muted-foreground tabular-nums">
                {dreamTemperature.toFixed(1)}
              </span>
            </div>
            <Slider
              value={[dreamTemperature]}
              min={0}
              max={1}
              step={0.1}
              onValueChange={([v]) => onTemperatureChange(v)}
              onValueChangeEnd={([v]) =>
                emitUiEvent('dream.temperature.change', {
                  'ui.component': 'DreamControls',
                  'ui.action': 'change',
                  'ui.value': v,
                })
              }
              className="[&_[data-orientation=horizontal]]:bg-gradient-to-r [&_[data-orientation=horizontal]]:from-blue-200 [&_[data-orientation=horizontal]]:to-red-200"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>Focused</span>
              <span className="font-medium">{temperatureLabel}</span>
              <span>Chaotic</span>
            </div>
          </div>

          {/* Interval Control */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Dream Interval</Label>
              <span className="text-xs text-muted-foreground tabular-nums">
                {intervalSeconds}s
              </span>
            </div>
            <Slider
              value={[intervalSeconds]}
              min={2}
              max={30}
              step={1}
              onValueChange={([v]) => onIntervalChange(v * 1000)}
              onValueChangeEnd={([v]) =>
                emitUiEvent('dream.interval.change', {
                  'ui.component': 'DreamControls',
                  'ui.action': 'change',
                  'ui.value': v,
                })
              }
            />
            <div className="text-xs text-muted-foreground">
              Generate a new variation every {intervalSeconds} second{intervalSeconds !== 1 ? 's' : ''}
            </div>
          </div>

          {/* Guide Button (only show if image is selected) */}
          {selectedParams && (
            <Button
              className={`w-full gap-2 transition-all duration-200 ${
                isGuided
                  ? 'bg-green-500 hover:bg-green-500 scale-95'
                  : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'
              }`}
              onClick={handleGuideDream}
              onMouseDown={() =>
                emitUiEvent('dream.guide.click', {
                  'ui.component': 'DreamControls',
                  'ui.action': 'click',
                })
              }
            >
              <Heart className={`h-4 w-4 ${isGuided ? 'fill-current animate-pulse' : ''}`} />
              {isGuided ? 'Guided!' : 'Guide Dream Toward This'}
            </Button>
          )}

          {/* Quick Actions */}
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
              onClick={onStopDreaming}
              onMouseDown={() =>
                emitUiEvent('dream.pause.click', {
                  'ui.component': 'DreamControls',
                  'ui.action': 'click',
                })
              }
            >
              <Pause className="mr-1 h-3 w-3" />
              Pause
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
              onClick={() => {
                onStopDreaming();
                setTimeout(() => onStartDreaming(baseParams), 100);
              }}
              onMouseDown={() =>
                emitUiEvent('dream.restart.click', {
                  'ui.component': 'DreamControls',
                  'ui.action': 'click',
                })
              }
            >
              <Play className="mr-1 h-3 w-3" />
              Restart
            </Button>
          </div>

          {/* Tips */}
          <div className="rounded-lg bg-purple-100/50 dark:bg-purple-900/20 p-2 text-xs text-purple-900 dark:text-purple-100">
            <strong>💡 Tip:</strong> Click images you like, then "Guide Dream" to evolve in that direction
          </div>
        </>
      )}
    </div>
  );
}
