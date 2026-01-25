// src/App.jsx

import React, { useMemo, useState, useCallback, useEffect } from 'react';
import { useChatMessages } from './hooks/useChatMessages';
import { useGenerationParams } from './hooks/useGenerationParams';
import { useImageGeneration } from './hooks/useImageGeneration';
import { ChatContainer } from './components/chat/ChatContainer';
import { OptionsPanel } from './components/options/OptionsPanel';
import { copyToClipboard } from './utils/helpers';
import { SR_CONFIG } from './utils/constants';


export default function App() {
  // ============================================================================
  // STATE MANAGEMENT VIA HOOKS
  // ============================================================================

  // Chat messages and selection
  const chatState = useChatMessages();
  const {
    messages,
    selectedMsgId,
    selectedMsg,
    selectedParams,
    addMessage,
    updateMessage,
    toggleSelectMsg,
    clearSelection,
    setSelectedMsgId,
    patchSelectedParams,
    setMsgRef,
    clearHistory,
  } = chatState;

  // Image generation (includes dream mode)
  const generation = useImageGeneration(addMessage, updateMessage, setSelectedMsgId);
  const {
    runGenerate,
    runSuperResUpload,
    cancelRequest,
    cancelAll,
    isDreaming,
    startDreaming,
    stopDreaming,
    guideDream,
    saveDreamAndContinue,
    dreamMessageId,
    dreamHistoryPrev,
    dreamHistoryNext,
    dreamHistoryLive,
    dreamTemperature,
    setDreamTemperature,
    dreamInterval,
    setDreamInterval,
    inflightCount,
    serverLabel,
    getImageFromCache,
    getCacheStats,
    clearCache,
  } = generation;


  // Reload cached images on startup (only for blob URLs without server URL)
  useEffect(() => {
    const reloadCachedImages = async () => {
      // Only reload messages that need it AND don't have a server URL
      const needsReload = messages.filter(
        (m) =>
          m.kind === 'image' &&
          m.needsReload &&
          m.params &&
          !m.serverImageUrl
      );
      if (needsReload.length === 0) return;

      console.log(`[App] Reloading ${needsReload.length} images from local cache...`);

      for (const msg of needsReload) {
        const imageUrl = await getImageFromCache(msg.params);
        if (imageUrl) {
          updateMessage(msg.id, { imageUrl, needsReload: false });
          console.log(`[App] Reloaded ${msg.id.slice(0, 8)}`);
        } else {
          updateMessage(msg.id, { needsReload: false, cacheExpired: true });
          console.log(`[App] Cache miss for ${msg.id.slice(0, 8)}`);
        }
      }
    };

    reloadCachedImages();
    // Only run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Generation parameters (draft + selected)
  const params = useGenerationParams(
    selectedParams,
    patchSelectedParams,
    runGenerate,
    selectedMsgId
  );
  
  // ============================================================================
  // LOCAL UI STATE
  // ============================================================================

  // Super-resolution upload
  const [uploadFile, setUploadFile] = useState(null);
  const [srMagnitude, setSrMagnitude] = useState(SR_CONFIG.DEFAULT_MAGNITUDE);

  // Copy feedback
  const [copied, setCopied] = useState(false);

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  /**
   * Send a new generation request.
   */
  const onSend = useCallback(() => {
    // If an image is selected, don't auto-generate (user controls via sliders)
    if (selectedParams) return;

    runGenerate({
      prompt: params.effective.prompt,
      size: params.effective.size,
      steps: params.effective.steps,
      cfg: params.effective.cfg,
      seedMode: params.draft.seedMode,
      seed: params.draft.seed,
      superresLevel: params.effective.superresLevel,
    });
  }, [selectedParams, runGenerate, params]);

  /**
   * Re-run the currently selected image with its params.
   */
  const onRerunSelected = useCallback(() => {
    if (!selectedParams) return;

    runGenerate({
      prompt: selectedParams.prompt,
      size: selectedParams.size,
      steps: selectedParams.steps,
      cfg: selectedParams.cfg,
      seedMode: 'fixed',
      seed: selectedParams.seed,
      superresLevel: selectedParams.superresLevel ?? 0,
    });
  }, [selectedParams, runGenerate]);

  /**
   * Apply a prompt delta to selected image.
   */
  const onApplyPromptDelta = useCallback(
    (delta) => {
      if (!selectedParams) return;
      const base = String(selectedParams.prompt || '').trim();
      const next = base ? `${base}, ${delta}` : delta;
      patchSelectedParams({ prompt: next });
    },
    [selectedParams, patchSelectedParams]
  );

  /**
   * Apply a seed delta to selected image and regenerate.
   */
  const onApplySeedDelta = useCallback(
    (delta) => {
      if (!selectedParams) return;
      const currentSeed = Number(selectedParams.seed) || 0;
      const newSeed = currentSeed + delta;
      // Trigger regeneration with new seed
      runGenerate({
        prompt: selectedParams.prompt,
        size: selectedParams.size,
        steps: selectedParams.steps,
        cfg: selectedParams.cfg,
        seedMode: 'fixed',
        seed: newSeed,
        superresLevel: selectedParams.superresLevel ?? 0,
        targetMessageId: selectedMsgId,
      });
    },
    [selectedParams, selectedMsgId, runGenerate]
  );

  /**
   * Handle super-resolution upload.
   */
  const onSuperResUpload = useCallback(() => {
    if (!uploadFile) return;
    runSuperResUpload(uploadFile, srMagnitude);
  }, [uploadFile, srMagnitude, runSuperResUpload]);

  /**
   * Copy current prompt to clipboard.
   */
  const onCopyPrompt = useCallback(async () => {
    const text = String(params.effective.prompt || '').trim();
    if (!text) return;

    const success = await copyToClipboard(text);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 900);
    }
  }, [params.effective.prompt]);

  /**
   * Handle Ctrl/Cmd + Enter to send.
   */
  const onKeyDown = useCallback(
    (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        onSend();
      }
    },
    [onSend]
  );


  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className="h-screen overflow-hidden bg-background text-foreground">
      <div className="mx-auto max-w-6xl p-4 md:p-6 h-full">
        <div className="grid h-full grid-cols-1 gap-4 md:grid-cols-[1fr_360px]">
          {/* Chat Panel */}
          <ChatContainer
            messages={messages}
            selectedMsgId={selectedMsgId}
            onToggleSelect={toggleSelectMsg}
            onCancelRequest={cancelRequest}
            setMsgRef={setMsgRef}
            composer={{
              prompt: params.draft.prompt,
              onPromptChange: params.setPromptDirect,
              onSend,
              onCancelAll: cancelAll,
              onKeyDown,
              onFocus: clearSelection,
              disabled: !params.effective.prompt.trim(),
              currentParams: {
                size: params.effective.size,
                steps: params.effective.steps,
                cfg: params.effective.cfg,
                seedMode: params.draft.seedMode,
                seed: params.draft.seed,
                superresLevel: params.effective.superresLevel,
              },
            }}
            inflightCount={inflightCount}
            isDreaming={isDreaming}
            dreamMessageId={dreamMessageId}
            onDreamSave={saveDreamAndContinue}
            onDreamHistoryPrev={dreamHistoryPrev}
            onDreamHistoryNext={dreamHistoryNext}
            onDreamHistoryLive={dreamHistoryLive}
            srLevel={params.effective.superresLevel}
            onCopyPrompt={onCopyPrompt}
            copied={copied}
            serverLabel={serverLabel}
          />

          {/* Options Panel */}
          <OptionsPanel
            params={params}
            selectedParams={selectedParams}
            selectedMsgId={selectedMsgId}
            onClearSelection={clearSelection}
            onApplyPromptDelta={onApplyPromptDelta}
            onApplySeedDelta={onApplySeedDelta}
            onRerunSelected={onRerunSelected}
            dreamState={{
              isDreaming,
              temperature: dreamTemperature,
              interval: dreamInterval,
              onStart: startDreaming,
              onStop: stopDreaming,
              onGuide: guideDream,
              onTemperatureChange: setDreamTemperature,
              onIntervalChange: setDreamInterval,
            }}
            onSuperResUpload={onSuperResUpload}
            uploadFile={uploadFile}
            onUploadFileChange={setUploadFile}
            srMagnitude={srMagnitude}
            onSrMagnitudeChange={setSrMagnitude}
            serverLabel={serverLabel}
            onClearCache={clearCache}
            getCacheStats={getCacheStats}
            onClearHistory={clearHistory}

          />
        </div>
      </div>
    </div>
  );
}