// src/components/options/OptionsPanel.jsx

import React, { useRef, useCallback, useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
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
          className="h-full overflow-y-auto space-y-5 p-4 md:p-5"
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
          <div className="space-y-2">
            <Label>
              {selectedParams ? 'Selected image prompt' : 'Draft prompt'}
            </Label>
            <Textarea
              value={params.effective.prompt}
              onChange={(e) => params.setPrompt(e.target.value)}
              className="min-h-[90px] resize-none rounded-2xl"
              placeholder="Describe what you want to generate…"
            />
          </div>

          {/* Steps */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <Label>Steps</Label>
              <span className="text-sm text-muted-foreground tabular-nums">
                {params.effective.steps}
              </span>
            </div>
            <Slider
              value={[params.effective.steps]}
              min={STEPS_CONFIG.MIN}
              max={STEPS_CONFIG.MAX}
              step={1}
              onValueChange={([v]) => params.setSteps(v)}
              className={CSS_CLASSES.SLIDER}
            />
            <div className="text-xs text-muted-foreground">
              Server allows up to {STEPS_CONFIG.SERVER_MAX}; typical LCM is{' '}
              {STEPS_CONFIG.LCM_TYPICAL_MIN}–{STEPS_CONFIG.LCM_TYPICAL_MAX}.
            </div>
          </div>

          {/* CFG */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <Label>CFG (Guidance Scale)</Label>
              <span className="text-sm text-muted-foreground tabular-nums">
                {Number(params.effective.cfg).toFixed(1)}
              </span>
            </div>
            <Slider
              value={[params.effective.cfg]}
              min={CFG_CONFIG.MIN}
              max={CFG_CONFIG.MAX}
              step={CFG_CONFIG.STEP}
              onValueChange={([v]) => params.setCfg(v)}
              className={CSS_CLASSES.SLIDER}
            />
            <div className="text-xs text-muted-foreground">
              LCM commonly uses ~{CFG_CONFIG.LCM_TYPICAL}.
            </div>
          </div>

          <Separator />

          {/* Seed */}
          <div className="space-y-3">
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

          {/* Super-Resolution Toggle */}
          <div className="space-y-2 rounded-2xl border p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="font-medium">Super-Resolution</div>
                <div className="text-xs text-muted-foreground">
                  0 = off · 1–3 = passes
                </div>
              </div>
              <div className="text-sm text-muted-foreground tabular-nums">
                {params.effective.superresLevel === 0
                  ? 'Off'
                  : `Level ${params.effective.superresLevel}`}
              </div>
            </div>

            <Slider
              className={CSS_CLASSES.SLIDER}
              value={[params.effective.superresLevel]}
              min={SR_CONFIG.MIN}
              max={SR_CONFIG.MAX}
              step={1}
              onValueChange={([v]) => params.setSrLevel(v)}
            />

            <div className="flex justify-between text-[11px] text-muted-foreground">
              <span>0</span>
              <span>1</span>
              <span>2</span>
              <span>3</span>
              <span>4</span>
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

          {/* Size */}
          <div className="space-y-2">
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