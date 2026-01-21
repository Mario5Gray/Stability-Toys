// src/components/options/LatentControls.jsx

import React, { useState } from 'react';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  Lock, 
  Users, 
  Layers, 
  Route,
  Info,
  Zap,
} from 'lucide-react';
import { CSS_CLASSES } from '../../utils/constants';

/**
 * Latent space exploration controls.
 * Provides UI for LCM's advanced latent behaviors.
 * 
 * @param {object} props
 * @param {object} props.latentState - Latent exploration state from useLatentExploration
 * @param {object} props.currentParams - Current generation parameters
 * @param {boolean} props.hasSelectedImage - Whether an image is selected
 */
export function LatentControls({ latentState, currentParams, hasSelectedImage }) {
  const [selectedPreset, setSelectedPreset] = useState('COUSINS_DETAILED');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleApplyPreset = async () => {
    if (!currentParams.prompt?.trim()) {
      alert('Enter a prompt first');
      return;
    }

    await latentState.applyPreset(selectedPreset, {
      prompt: currentParams.prompt,
      seed: currentParams.seed || Math.floor(Math.random() * 100000000),
      size: currentParams.size,
    });
  };

  const handleLatentWalk = async () => {
    if (!currentParams.prompt?.trim()) {
      alert('Enter a prompt first');
      return;
    }

    const seedStart = Math.floor(Math.random() * 100000000);
    const seedEnd = Math.floor(Math.random() * 100000000);

    await latentState.latentWalk({
      seedStart,
      seedEnd,
      steps: 5,
      prompt: currentParams.prompt,
      size: currentParams.size,
    });
  };

  return (
    <div className="space-y-3 rounded-2xl border p-4 bg-gradient-to-br from-orange-50/50 to-amber-50/50 dark:from-orange-950/20 dark:to-amber-950/20">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-orange-600" />
          <Label className="text-base font-semibold">Latent Explorer</Label>
          {latentState.chainLength > 0 && (
            <Badge variant="secondary" className="text-xs">
              {latentState.chainLength} in chain
            </Badge>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <Info className="h-3 w-3" />
        </Button>
      </div>

      {/* Explainer */}
      <div className="text-xs text-muted-foreground">
        Exploit LCM's latent space persistence for advanced control
      </div>

      {showAdvanced && (
        <div className="rounded-lg bg-orange-100/50 dark:bg-orange-900/20 p-3 text-xs space-y-2">
          <div>
            <strong>📍 Latent Lock (0 steps, high CFG):</strong> Encodes prompt into latent space. Next iteration "remembers" this.
          </div>
          <div>
            <strong>👥 Latent Cousins (many steps, CFG=0, denoise&lt;1):</strong> Explores neighborhood of a seed. Extremely detailed, prompt-sticky.
          </div>
          <div>
            <strong>🎯 Multi-pass:</strong> Progressive refinement using latent persistence.
          </div>
        </div>
      )}

      <Separator />

      {/* Preset Selection */}
      <div className="space-y-2">
        <Label>Exploration Preset</Label>
        <Select value={selectedPreset} onValueChange={setSelectedPreset}>
          <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
            {Object.entries(latentState.presets).map(([key, preset]) => (
              <SelectItem
                key={key}
                value={key}
                className={CSS_CLASSES.SELECT_ITEM}
              >
                <div className="flex flex-col items-start">
                  <span className="font-medium">{preset.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {preset.description}
                  </span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Preset Details */}
      {selectedPreset && latentState.presets[selectedPreset] && (
        <div className="rounded-lg bg-muted/50 p-2 text-xs space-y-1">
          <div className="flex justify-between">
            <span>Steps:</span>
            <span className="font-mono">
              {latentState.presets[selectedPreset].steps}
            </span>
          </div>
          <div className="flex justify-between">
            <span>CFG:</span>
            <span className="font-mono">
              {latentState.presets[selectedPreset].cfg.toFixed(1)}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Denoise:</span>
            <span className="font-mono">
              {latentState.presets[selectedPreset].denoise.toFixed(1)}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Passes:</span>
            <span className="font-mono">
              {latentState.presets[selectedPreset].passCount}
            </span>
          </div>
        </div>
      )}

      {/* Apply Preset Button */}
      <Button
        className="w-full gap-2"
        onClick={handleApplyPreset}
        disabled={!currentParams.prompt?.trim()}
      >
        <Zap className="h-4 w-4" />
        Apply Preset
      </Button>

      <Separator />

      {/* Quick Actions */}
      <div className="space-y-2">
        <Label className="text-sm">Quick Actions</Label>
        
        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            onClick={async () => {
              await latentState.lockPrompt({
                prompt: currentParams.prompt,
                seed: currentParams.seed || Math.floor(Math.random() * 100000000),
                size: currentParams.size,
              });
            }}
            disabled={!currentParams.prompt?.trim()}
          >
            <Lock className="h-3 w-3" />
            Lock
          </Button>

          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            onClick={async () => {
              await latentState.exploreCousins({
                prompt: currentParams.prompt,
                seed: currentParams.seed || Math.floor(Math.random() * 100000000),
                size: currentParams.size,
                denoise: 0.7,
                steps: 10,
              });
            }}
            disabled={!currentParams.prompt?.trim()}
          >
            <Users className="h-3 w-3" />
            Cousins
          </Button>

          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            onClick={async () => {
              await latentState.multiPassRefine(
                {
                  prompt: currentParams.prompt,
                  seed: currentParams.seed || Math.floor(Math.random() * 100000000),
                  size: currentParams.size,
                },
                3,
                0.8
              );
            }}
            disabled={!currentParams.prompt?.trim()}
          >
            <Layers className="h-3 w-3" />
            Multi-Pass
          </Button>

          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            onClick={handleLatentWalk}
            disabled={!currentParams.prompt?.trim()}
          >
            <Route className="h-3 w-3" />
            Walk
          </Button>
        </div>
      </div>

      {/* Multi-Pass Advanced Controls */}
      {showAdvanced && (
        <>
          <Separator />
          <div className="space-y-3">
            <Label className="text-sm">Multi-Pass Settings</Label>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Pass Count</Label>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {latentState.passCount}
                </span>
              </div>
              <Slider
                value={[latentState.passCount]}
                min={1}
                max={5}
                step={1}
                onValueChange={([v]) => latentState.setPassCount(v)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Denoise Strength</Label>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {latentState.denoiseStrength.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[latentState.denoiseStrength]}
                min={0.1}
                max={1.0}
                step={0.05}
                onValueChange={([v]) => latentState.setDenoiseStrength(v)}
              />
              <div className="text-xs text-muted-foreground">
                Lower = stay closer to latent, Higher = more variation
              </div>
            </div>
          </div>
        </>
      )}

      {/* Chain Status */}
      {latentState.chainLength > 0 && (
        <>
          <Separator />
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">
              Latent chain: {latentState.chainLength} generations
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs"
              onClick={latentState.clearLatentChain}
            >
              Clear
            </Button>
          </div>
        </>
      )}

      {/* Tips */}
      <div className="rounded-lg bg-orange-100/50 dark:bg-orange-900/20 p-2 text-xs text-orange-900 dark:text-orange-100">
        <strong>💡 Pro tip:</strong> Use "Lock" to encode a prompt, then "Cousins" to explore variations. The latent state persists!
      </div>
    </div>
  );
}