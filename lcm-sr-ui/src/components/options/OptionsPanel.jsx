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
import { Send, Trash2, Combine} from 'lucide-react';
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
import { ComfyOptions } from "./ComfyOptions";

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
  inputImage,
  comfyInputImage,
  selectedParams,
  selectedMsgId,
  onClearSelection,
  onApplyPromptDelta,
  onApplySeedDelta,
  onRerunSelected,
  dreamState,
  onSuperResUpload,
  uploadFile,
  onUploadFileChange,
  srMagnitude,
  onSrMagnitudeChange,
  serverLabel,
  onRunComfy,
  comfyIsBusy,
  comfyState,
  comfyJob,
  comfyError,
  onCancelComfy,
  onClearCache,
  getCacheStats,
  onClearHistory,
  onComfyOutputs,
  onComfyStart,
  queueState,
}) {
  const optionsScrollRef = useRef(null);
  const [canScrollDown, setCanScrollDown] = useState(false);
  const [canScrollUp, setCanScrollUp] = useState(false);

  // Cache stats state
  const [cacheStats, setCacheStats] = useState(null);
  const [isClearing, setIsClearing] = useState(false);

  // Track selected image ID to sync only on selection change
  const prevSelectedId = useRef(null);
  // Flag to skip prompt push during selection sync
  const isSyncingSelection = useRef(false);

  // Local state for prompt
  const [localPrompt, setLocalPrompt] = useState(params.draft.prompt);
  const [debouncedPrompt] = useDebounceValue(localPrompt, 500);

  // Local state for controls (no debounce - immediate feedback, push on change)
  const [localSteps, setLocalSteps] = useState(params.effective.steps);
  // Split CFG into base (whole number) and fine (decimal 0-9 representing 0.0-0.9)
  const [localCfgBase, setLocalCfgBase] = useState(Math.floor(params.effective.cfg));
  const [localCfgFine, setLocalCfgFine] = useState(Math.round((params.effective.cfg % 1) * 10));
  const [localSrLevel, setLocalSrLevel] = useState(params.effective.superresLevel);
  // Seed modifier sign: 1 for positive, -1 for negative
  const [seedSign, setSeedSign] = useState(1);

  // Combined CFG value for display
  const localCfg = localCfgBase + localCfgFine / 10;

  // Sync local state when selection changes (including select/deselect)
  useEffect(() => {
    const currentId = selectedMsgId ?? null;
    if (currentId !== prevSelectedId.current) {
      prevSelectedId.current = currentId;
      // Mark that we're syncing to prevent prompt push from triggering regen
      isSyncingSelection.current = true;

      if (selectedParams) {
        // Selected an image - sync from its params
        const cfg = selectedParams.cfg ?? params.effective.cfg;
        setLocalPrompt(selectedParams.prompt ?? params.draft.prompt);
        setLocalSteps(selectedParams.steps ?? params.effective.steps);
        setLocalCfgBase(Math.floor(cfg));
        setLocalCfgFine(Math.round((cfg % 1) * 10));
        setLocalSrLevel(selectedParams.superresLevel ?? params.effective.superresLevel);
      } else {
        // Deselected - sync from draft params
        const cfg = params.effective.cfg;
        setLocalPrompt(params.draft.prompt);
        setLocalSteps(params.effective.steps);
        setLocalCfgBase(Math.floor(cfg));
        setLocalCfgFine(Math.round((cfg % 1) * 10));
        setLocalSrLevel(params.effective.superresLevel);
      }

      // Clear sync flag after debounce settles (must exceed useDebounceValue delay)
      setTimeout(() => {
        isSyncingSelection.current = false;
      }, 600);
    }
  }, [selectedMsgId, selectedParams, params.draft.prompt, params.effective.steps, params.effective.cfg, params.effective.superresLevel]);

  // Push debounced prompt to parent (skip during selection sync to avoid regen)
  useEffect(() => {
    if (isSyncingSelection.current) return;
    if (debouncedPrompt !== params.draft.prompt) {
      params.setPrompt(debouncedPrompt);
    }
  }, [debouncedPrompt]);

  // Handlers that update local state AND push to parent
  const handleStepsChange = (v) => {
    setLocalSteps(v);
    params.setSteps(v);
  };

  const handleCfgBaseChange = (v) => {
    setLocalCfgBase(v);
    params.setCfg(v + localCfgFine / 10);
  };

  const handleCfgFineChange = (v) => {
    setLocalCfgFine(v);
    params.setCfg(localCfgBase + v / 10);
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

  // Fetch cache stats on mount and after clearing
  const refreshCacheStats = useCallback(async () => {
    if (getCacheStats) {
      const stats = await getCacheStats();
      setCacheStats(stats);
    }
  }, [getCacheStats]);

  useEffect(() => {
    refreshCacheStats();
  }, [refreshCacheStats]);

  // Handle clear cache and history
  const handleClearCache = useCallback(async () => {
    
    setIsClearing(true);
    try {
      await onClearCache();
      onClearHistory();
      await refreshCacheStats();
    } finally {
      setIsClearing(false);
    }
  }, [onClearCache, onClearHistory, refreshCacheStats]);

  // Format bytes for display
  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  function ComfyGenerator() {
    const api = useMemo(() => createComfyInvokerApi("/api"), []);
    const comfy = useComfyJob({ api });

    const [options, setOptions] = useState({
      cfg: 0.35,
      steps: 12,
      denoise: 0.3,
    });
  };

  return (
    <Card className="rounded-2xl shadow-sm h-full flex flex-col overflow-hidden">
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

          {/* Prompt, steps, cfg, seed, resolution */}
          <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
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
            <div className="flex items-center justify-between">
              <Combine className="mr-2 h-4 w-4" /><Label className="text-xs">Steps</Label>
              <span className="text-xs text-muted-foreground tabular-nums">{localSteps}</span>
            </div>
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
                    'flex-1 py-1 text-[10px] font-medium rounded-lg transition-all ' +
                    (localSteps === v
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-white/90 hover:bg-white/20')
                  }
                >
                  {v % 2 === 0 ? v : '|'}
                </button>
              ))}
            </div>
{/*            <div className="text-xs text-muted-foreground">
              LCM typical: {STEPS_CONFIG.LCM_TYPICAL_MIN}–{STEPS_CONFIG.LCM_TYPICAL_MAX}. 0 = latent lock.
            </div>
*/}          </div>

          {/* CFG - Segmented Control */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">CFG (Guidance)</Label>
              <span className="text-xs text-muted-foreground tabular-nums">{localCfg.toFixed(1)}</span>
            </div>
            <div
              className="relative flex rounded-xl p-0.5 overflow-hidden"
              style={{ background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%)' }}
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => handleCfgBaseChange(v)}
                  className={
                    'flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ' +
                    (localCfgBase === v
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-white/90 hover:bg-white/20')
                  }
                >
                  {v}
                </button>
              ))}
            </div>

            {/* CFG Fine-tune (0.0 - 0.9) */}
            <div
              className="relative flex rounded-xl p-0.5 overflow-hidden"
              style={{ background: 'linear-gradient(135deg, #6d28d9 0%, #7c3aed 50%, #8b5cf6 100%)' }}
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => handleCfgFineChange(v)}
                  className={
                    'flex-1 py-1 text-[10px] font-medium rounded-lg transition-all ' +
                    (localCfgFine === v
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-white/80 hover:bg-white/20')
                  }
                >
                  {v % 2 === 0 ? `.${v}` : '|'}
                </button>
              ))}
            </div>
            {selectedParams && (
              <div className="space-y-2 mt-3">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Seed Modifier</Label>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    current: {selectedParams.seed}
                  </span>
                </div>
                <div
                  className="relative flex rounded-xl p-0.5 overflow-hidden"
                  style={{ background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%)' }}
                >
                  {/* Sign toggle button */}
                  <button
                    type="button"
                    onClick={() => setSeedSign(s => s * -1)}
                    className={
                      'flex-1 py-1.5 text-xs font-bold rounded-lg transition-all ' +
                      'bg-white/20 text-white hover:bg-white/30'
                    }
                  >
                    {seedSign > 0 ? '+/−' : '−/+'}
                  </button>
                  {[
                    { delta: 1, label: '1' },
                    { delta: 10, label: '10' },
                    { delta: 100, label: '100' },
                    { delta: 1000, label: '1k' },
                    { delta: 10000, label: '10k' },
                  ].map(({ delta, label }) => (
                    <button
                      key={delta}
                      type="button"
                      onClick={() => onApplySeedDelta?.(delta * seedSign)}
                      className="flex-1 py-1.5 text-xs font-medium rounded-lg transition-all text-white/90 hover:bg-white/20 active:bg-white active:text-purple-700"
                    >
                      {seedSign > 0 ? '+' : '−'}{label}
                    </button>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground">
                  {seedSign > 0 ? 'Add to' : 'Subtract from'} current seed and regenerate.
                </div>
              </div>
            )}
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
          </div>
          </div>
          <Separator />

          {/* Seed */}
          <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
            <Label>Next Seed</Label>
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

            {/* Seed Modifier - only when image is selected */}
          </div>

          {/* Super-Resolution - Segmented Control */}

          
          <Separator />

          <ComfyOptions   onOutputs={onComfyOutputs}
                          inputImage={comfyInputImage}
                          onComfyStart={onComfyStart}
                          queueState={queueState}
          />
          
          <Separator />

          {/* Super-Resolution Upload */}
          <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
            <div className="font-medium">Super-resolve an uploaded image</div>

            {/* Magnitude */}
            <div className="space-y-2">
              <Label># Passes</Label>
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

          {/* Cache Management */}
          <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
            <div className="font-medium">Image Cache</div>
            {cacheStats && (
              <div className="rounded-2xl bg-muted/40 p-3 text-xs text-muted-foreground space-y-1">
                <div className="flex justify-between">
                  <span>Entries:</span>
                  <span className="tabular-nums">{cacheStats.entries ?? 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Size:</span>
                  <span className="tabular-nums">{formatBytes(cacheStats.bytes)}</span>
                </div>
                {cacheStats.maxBytes && (
                  <div className="flex justify-between">
                    <span>Usage:</span>
                    <span className="tabular-nums">
                      {((cacheStats.utilizationBytes ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            )}
            <Button
              variant="outline"
              className="w-full rounded-2xl"
              onClick={handleClearCache}
              disabled={isClearing}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              {isClearing ? 'Clearing...' : 'Clear Messages'}
            </Button>
            <div className="text-xs text-muted-foreground">
              Clears all messages and locally cached images from browser storage.
            </div>
          </div>
        </CardContent>
      </div>
    </Card>
  );
}