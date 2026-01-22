// src/components/options/OptionsPanel.jsx

import React, { useRef, useCallback, useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { useDebounceValue } from 'usehooks-ts';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Send } from 'lucide-react';
import { DreamControls } from './DreamControls';
import { SelectedImageControls } from './SelectedImageControls';
import {
  SIZE_OPTIONS,
  STEPS_CONFIG,
  CFG_CONFIG,
  SR_CONFIG,
  SR_MAGNITUDE_OPTIONS,
  SEED_MODES,
  CSS_CLASSES,
  SCROLL_CONFIG,
} from '../../utils/constants';
import { formatSizeDisplay, sanitizeSeedInput } from '../../utils/helpers';

/**
 * Options panel component - right sidebar with all generation controls.
 * 
 * @param {object} props
 * @param {object} props.params - Generation parameters
 * @param {object|null} props.selectedParams - Selected image params
 * @param {function} props.onClearSelection - Clear selection callback
 * @param {function} props.onApplyPromptDelta - Apply prompt delta callback
 * @param {function} props.onRerunSelected - Rerun selected callback
 * @param {object} props.dreamState - Dream mode state
 * @param {function} props.onSuperResUpload - SR upload callback
 * @param {File|null} props.uploadFile - Selected file for SR
 * @param {function} props.onUploadFileChange - File change callback
 * @param {number} props.srMagnitude - SR magnitude
 * @param {function} props.onSrMagnitudeChange - SR magnitude change callback
 * @param {string} props.serverLabel - Server label for display
 */
export function OptionsPanel({
  params,
  selectedParams,
  selectedMsgId,
  onClearSelection,
  onApplyPromptDelta,
  onRerunSelected,
  dreamState,
  onSuperResUpload,
  uploadFile,
  onUploadFileChange,
  srMagnitude,
  onSrMagnitudeChange,
  serverLabel,
}) {
  const optionsScrollRef = useRef(null);
  const [canScrollDown, setCanScrollDown] = useState(false);
  const [canScrollUp, setCanScrollUp] = useState(false);

  // Track selected image ID to sync only on selection change
  const prevSelectedId = useRef(null);

  // Local state for prompt
  const [localPrompt, setLocalPrompt] = useState(params.draft.prompt);
  const [debouncedPrompt] = useDebounceValue(localPrompt, 500);

  // Local state for controls (no debounce - immediate feedback, push on change)
  const [localSteps, setLocalSteps] = useState(params.effective.steps);
  const [localCfg, setLocalCfg] = useState(params.effective.cfg);
  const [localSrLevel, setLocalSrLevel] = useState(params.effective.superresLevel);

  // Sync local state when selection changes (including select/deselect)
  useEffect(() => {
    const currentId = selectedMsgId ?? null;
    if (currentId !== prevSelectedId.current) {
      prevSelectedId.current = currentId;

      if (selectedParams) {
        // Selected an image - sync from its params
        setLocalPrompt(selectedParams.prompt ?? params.draft.prompt);
        setLocalSteps(selectedParams.steps ?? params.effective.steps);
        setLocalCfg(selectedParams.cfg ?? params.effective.cfg);
        setLocalSrLevel(selectedParams.superresLevel ?? params.effective.superresLevel);
      } else {
        // Deselected - sync from draft params
        setLocalPrompt(params.draft.prompt);
        setLocalSteps(params.effective.steps);
        setLocalCfg(params.effective.cfg);
        setLocalSrLevel(params.effective.superresLevel);
      }
    }
  }, [selectedMsgId, selectedParams, params.draft.prompt, params.effective.steps, params.effective.cfg, params.effective.superresLevel]);

  // Push debounced prompt to parent
  useEffect(() => {
    if (debouncedPrompt !== params.draft.prompt) {
      params.setPrompt(debouncedPrompt);
    }
  }, [debouncedPrompt]);

  // Handlers that update local state AND push to parent
  const handleStepsChange = (v) => {
    setLocalSteps(v);
    params.setSteps(v);
  };

  const handleCfgChange = (v) => {
    setLocalCfg(v);
    params.setCfg(v);
  };

  const handleSrLevelChange = (v) => {
    setLocalSrLevel(v);
    params.setSrLevel(v);
  };

  const updateScrollHints = useCallback(() => {
    const el = optionsScrollRef.current;
    if (!el) return;

    const down =
      el.scrollHeight - el.scrollTop - el.clientHeight >
      SCROLL_CONFIG.HINT_THRESHOLD_PX;
    const up = el.scrollTop > SCROLL_CONFIG.HINT_THRESHOLD_PX;

    setCanScrollDown(down);
    setCanScrollUp(up);
  }, []);

  useEffect(() => {
    updateScrollHints();
  }, [updateScrollHints]);

  return (
    <Card className="rounded-2xl shadow-sm h-full flex flex-col overflow-hidden">
      <CardHeader className="border-b">
        <CardTitle className="text-lg">Options</CardTitle>
        <div className="text-sm text-muted-foreground">
          Generation parameters
        </div>
      </CardHeader>

      {/* Scroll container with hint overlay */}
      <div className="relative flex-1 min-h-0">
        {(canScrollDown || canScrollUp) && (
          <div className="absolute top-0 left-0 right-0 z-10 text-center py-1 bg-background/80 backdrop-blur-sm">
            <div className="text-xs text-muted-foreground">
              More {canScrollUp ? '↑' : '↓'} (scroll)
            </div>
          </div>
        )}

        <CardContent
          ref={optionsScrollRef}
          onScroll={updateScrollHints}
          className="h-full overflow-y-auto space-y-2 p-4 md:p-5"
        >

          {/* Dream Mode */}
          <DreamControls
            isDreaming={dreamState.isDreaming}
            dreamTemperature={dreamState.temperature}
            dreamInterval={dreamState.interval}
            onStartDreaming={dreamState.onStart}
            onStopDreaming={dreamState.onStop}
            onGuideDream={dreamState.onGuide}
            onTemperatureChange={dreamState.onTemperatureChange}
            onIntervalChange={dreamState.onIntervalChange}
            selectedParams={selectedParams}
            baseParams={params.effective}
          />
          
          <Separator />

          {/* Prompt */}
          <div className="space-y-1">
            <Label>
              {selectedParams ? 'Selected image prompt' : 'Draft prompt'}
            </Label>
            <Textarea
              value={localPrompt}
              onChange={(e) => setLocalPrompt(e.target.value)}
              className="min-h-[90px] resize-none rounded-2xl"
              placeholder="Describe what you want to generate…"
            />
          </div>

          {/* Steps - Segmented Control */}
          <div className="space-y-2">
            <Label>Steps</Label>
            <div
              className="relative flex rounded-xl p-0.5 overflow-hidden"
              style={{ background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%)' }}
            >
              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20].map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => handleStepsChange(v)}
                  className={
                    'flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ' +
                    (localSteps === v
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-white/90 hover:bg-white/20')
                  }
                >
                  {v}
                </button>
              ))}
            </div>
            <div className="text-xs text-muted-foreground">
              LCM typical: {STEPS_CONFIG.LCM_TYPICAL_MIN}–{STEPS_CONFIG.LCM_TYPICAL_MAX}. 0 = latent lock.
            </div>
          </div>

          {/* CFG - Segmented Control */}
          <div className="space-y-2">
            <Label>CFG (Guidance)</Label>
            <div
              className="relative flex rounded-xl p-0.5 overflow-hidden"
              style={{ background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%)' }}
            >
              {[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0].map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => handleCfgChange(v)}
                  className={
                    'flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ' +
                    (localCfg === v
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-white/90 hover:bg-white/20')
                  }
                >
                  {v.toFixed(1)}
                </button>
              ))}
            </div>
            <div className="text-xs text-muted-foreground">
              LCM typical: ~{CFG_CONFIG.LCM_TYPICAL}. Higher = stronger prompt adherence.
            </div>
          </div>

          <Separator />

          {/* Seed */}
          <div className="space-y-1">
            <Label>Seed</Label>
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <Select
                  value={params.draft.seedMode}
                  onValueChange={params.setSeedMode}
                >
                  <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
                    <SelectItem
                      className={CSS_CLASSES.SELECT_ITEM}
                      value={SEED_MODES.RANDOM}
                    >
                      Random
                    </SelectItem>
                    <SelectItem
                      className={CSS_CLASSES.SELECT_ITEM}
                      value={SEED_MODES.FIXED}
                    >
                      Fixed
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button
                variant="outline"
                className="rounded-2xl"
                onClick={params.randomizeSeed}
                title="Generate a new random seed"
              >
                Randomize
              </Button>
            </div>

            <Input
              value={params.draft.seed}
              onChange={(e) => {
                const sanitized = sanitizeSeedInput(e.target.value);
                params.setSeed(sanitized);
              }}
              disabled={params.draft.seedMode !== SEED_MODES.FIXED}
              className={CSS_CLASSES.INPUT}
              inputMode="numeric"
              placeholder="seed"
            />

            <div className="text-xs text-muted-foreground">
              When Random: a new seed is chosen per request. When Fixed: the seed
              field is used.
            </div>
          </div>
          {/* Size */}
          <div className="space-y-1">
            <Label>Size</Label>
            <Select value={params.effective.size} onValueChange={params.setSize}>
              <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
                <SelectValue placeholder="Select size" />
              </SelectTrigger>
              <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
                {SIZE_OPTIONS.map((s) => (
                  <SelectItem
                    key={s}
                    className={CSS_CLASSES.SELECT_ITEM}
                    value={s}
                  >
                    {formatSizeDisplay(s)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Separator />

          {/* Selected Image Controls */}
          {selectedParams ? (
            <SelectedImageControls
              selectedParams={selectedParams}
              onClear={onClearSelection}
              onApplyDelta={onApplyPromptDelta}
              onRerun={onRerunSelected}
            />
          ) : (
            <div className="text-xs text-muted-foreground rounded-lg bg-muted/50 p-3">
              💡 Tip: Click an image to select it. Sliders will edit that image's
              settings and regenerate live.
            </div>
          )}

          <Separator />

          {/* Super-Resolution - Segmented Control */}
          <div className="space-y-2">
            <Label>Super-Resolution</Label>
            <div
              className="relative flex rounded-xl p-0.5 overflow-hidden"
              style={{ background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%)' }}
            >
              {[
                { v: 0, label: 'Off' },
                { v: 1, label: '1×' },
                { v: 2, label: '2×' },
                { v: 3, label: '3×' },
                { v: 4, label: '4×' },
              ].map(({ v, label }) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => handleSrLevelChange(v)}
                  className={
                    'flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ' +
                    (localSrLevel === v
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-white/90 hover:bg-white/20')
                  }
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="text-xs text-muted-foreground">
              Number of upscale passes. Higher = more detail, slower.
            </div>
          </div>

          {/* Super-Resolution Upload */}
          <div className="space-y-3">
            <div className="font-medium">Super-resolve an uploaded image</div>

            {/* Magnitude */}
            <div className="space-y-2">
              <Label>Magnitude</Label>
              <Select
                value={String(srMagnitude)}
                onValueChange={(v) => onSrMagnitudeChange(Number(v))}
              >
                <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
                  {SR_MAGNITUDE_OPTIONS.map((opt) => (
                    <SelectItem
                      key={opt.value}
                      className={CSS_CLASSES.SELECT_ITEM}
                      value={opt.value}
                    >
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="text-xs text-muted-foreground">
                Magnitude = number of SR passes. Default is 2.
              </div>
            </div>

            {/* File Input */}
            <div className="space-y-2">
              <Label>Image file</Label>
              <Input
                type="file"
                accept="image/*"
                className={CSS_CLASSES.INPUT}
                onChange={(e) => onUploadFileChange(e.target.files?.[0] || null)}
              />
              <div className="text-xs text-muted-foreground">
                {uploadFile
                  ? `Selected: ${uploadFile.name}`
                  : 'Choose a JPG/PNG/WebP/etc.'}
              </div>
            </div>

            <Button
              className="w-full rounded-2xl"
              onClick={onSuperResUpload}
              disabled={!uploadFile}
              title={!uploadFile ? 'Pick an image first' : 'Upload and super-resolve'}
            >
              <Send className="mr-2 h-4 w-4" />
              Super-res uploaded image
            </Button>
          </div>

          <Separator />

          {/* Server Info */}
          <div className="rounded-2xl bg-muted/40 p-3 text-xs text-muted-foreground">
            <div className="font-medium text-foreground">Server base</div>
            <div className="mt-1 break-all">{serverLabel}</div>
            <div className="mt-2">Output: PNG only (per UI spec)</div>
          </div>
        </CardContent>
      </div>
    </Card>
  );
}