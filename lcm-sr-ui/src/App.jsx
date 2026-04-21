// src/App.jsx

import React, { useMemo, useState, useCallback, useEffect, useRef } from 'react';
import { ChatDropzone } from "./components/chat/ChatDropzone";
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { useChatMessages } from './hooks/useChatMessages';
import { useGenerationParams } from './hooks/useGenerationParams';
import { useImageGeneration } from './hooks/useImageGeneration';
import { useModeConfig } from './hooks/useModeConfig';
import { ChatContainer } from './components/chat/ChatContainer';
import { OptionsPanel } from './components/options/OptionsPanel';
import { copyToClipboard } from './utils/helpers';
import {
  DEFAULT_IMG2IMG_DENOISE_STRENGTH,
  DEFAULT_SIZE,
  SR_CONFIG,
} from './utils/constants';
import { applyModeControlDefaultsToDraft } from './utils/generationControls';
import { MessageSquare, Settings, Folder } from 'lucide-react';
import ModeEditor from './components/config/ModeEditor';
import WorkflowEditor from './components/config/WorkflowEditor';
import { useWs } from './hooks/useWs';
import { useJobQueue } from './hooks/useJobQueue';
import { useChatJob } from './hooks/useChatJob';
import { emitUiEvent } from './utils/otelTelemetry';
import {
  clearActiveSource,
  loadActiveSource,
  saveSource,
  setActiveSourceId,
} from './utils/img2imgSourceStore';
import { useGalleries } from './hooks/useGalleries';
import { GalleryCreatePopover } from './components/gallery/GalleryCreatePopover';
import { GalleryLightbox } from './components/gallery/GalleryLightbox';
import { getFrontendVersion } from './utils/version';


export async function fetchBlobFromCandidates(candidateUrls) {
  const urls = Array.from(new Set((candidateUrls || []).filter(Boolean)));
  let lastError = null;

  for (const candidateUrl of urls) {
    try {
      const response = await fetch(candidateUrl);
      if (!response.ok) {
        lastError = new Error(`Failed to fetch init image: ${response.status}`);
        continue;
      }
      return {
        blob: await response.blob(),
        resolvedUrl: candidateUrl,
      };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error('Failed to fetch init image from available candidates');
}

export function buildSelectedSeedDeltaPayload(
  selectedParams,
  selectedMsgId,
  delta,
  initImageFile = null,
  denoiseStrength = null
) {
  if (!selectedParams) return null;
  const currentSeed = Number(selectedParams.seed) || 0;
  const newSeed = currentSeed + delta;

  return {
    prompt: selectedParams.prompt,
    negativePrompt: selectedParams.negativePrompt,
    schedulerId: selectedParams.schedulerId,
    size: selectedParams.size,
    steps: selectedParams.steps,
    cfg: selectedParams.cfg,
    seedMode: 'fixed',
    seed: newSeed,
    superresLevel: selectedParams.superresLevel ?? 0,
    initImageFile: initImageFile || null,
    denoiseStrength:
      denoiseStrength ?? selectedParams.denoiseStrength ?? null,
    targetMessageId: selectedMsgId,
  };
}

export function getModeDefaultsSyncPlan(modeState, draft, lastAppliedDraftDefaults = null) {
  const configModes = modeState?.config?.modes || {};
  const mode =
    modeState?.activeMode ||
    (modeState?.activeModeName ? configModes[modeState.activeModeName] : null) ||
    (modeState?.config?.default_mode ? configModes[modeState.config.default_mode] : null) ||
    null;

  if (!mode || !draft) return null;

  const comparableDraft = {
    size: draft.size ?? DEFAULT_SIZE,
    negativePrompt: draft.negativePrompt ?? '',
    schedulerId: draft.schedulerId ?? null,
  };
  const baselineDraft = lastAppliedDraftDefaults || {
    size: DEFAULT_SIZE,
    negativePrompt: '',
    schedulerId: null,
  };
  const canSync =
    comparableDraft.size === baselineDraft.size &&
    comparableDraft.negativePrompt === baselineDraft.negativePrompt &&
    comparableDraft.schedulerId === baselineDraft.schedulerId;

  if (!canSync) return null;

  const nextDraftDefaults = applyModeControlDefaultsToDraft(comparableDraft, mode);
  const draftDefaults = {
    size: nextDraftDefaults.size || DEFAULT_SIZE,
    negativePrompt: nextDraftDefaults.negativePrompt || '',
    schedulerId: nextDraftDefaults.schedulerId || null,
  };

  if (
    lastAppliedDraftDefaults &&
    draftDefaults.size === lastAppliedDraftDefaults.size &&
    draftDefaults.negativePrompt === lastAppliedDraftDefaults.negativePrompt &&
    draftDefaults.schedulerId === lastAppliedDraftDefaults.schedulerId
  ) {
    return null;
  }

  return {
    mode,
    draftDefaults,
  };
}

export default function App() {
  const ws = useWs(); // auto-connect WS singleton on mount
  const queueState = useJobQueue();
  const chatJob = useChatJob();
  const [inputMode, setInputMode] = useState('generate'); // 'generate' | 'chat'
  const galleryState = useGalleries();
  const frontendVersion = useMemo(() => getFrontendVersion(), []);
  const [openGalleryId, setOpenGalleryId] = useState(null);
  const [trashOpen, setTrashOpen] = useState(false);
  const didReportRender = useRef(false);

  useEffect(() => {
    if (didReportRender.current) return;
    didReportRender.current = true;
    const start = window.__appStartTime;
    if (typeof start === 'number') {
      emitUiEvent('ui.render.app', {
        'ui.component': 'App',
        'ui.metric': 'first_render_ms',
        'ui.value': Math.max(0, performance.now() - start),
      });
    }
  }, []);
  // ============================================================================
  // STATE MANAGEMENT VIA HOOKS
  // ============================================================================

  // Tab navigation
  const [activeTab, setActiveTab] = useState('chat'); // chat | dreams

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
    deleteMessage,
    createErrorMessage,
  } = chatState;
  const handleGenerationAutoSelect = useCallback(
    (id) => {
      setSelectedMsgId(id);
    },
    [setSelectedMsgId]
  );

  // Image generation (includes dream mode)
  const generation = useImageGeneration(addMessage, updateMessage, handleGenerationAutoSelect);
  const {
    runGenerate,
    runComfy,
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
    cleanupMessage,
  } = generation;


  // Reload cached images on startup (only for blob URLs without server URL)
  useEffect(() => {
    console.log("Reloading cached images");
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

      for (const msg of needsReload) {
        console.log("[app] fetching image from cache ")
        const imageUrl = await getImageFromCache(msg.params);
        console.log("[app] fetched image from cache ")
        if (imageUrl) {
          updateMessage(msg.id, { imageUrl, needsReload: false });
        } else {
          updateMessage(msg.id, { needsReload: false, cacheExpired: true });
        }
      }
    };

    reloadCachedImages();
    // Only run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onImageDisplayed = useCallback((msg) => {
    if (!msg?.id) return;
    updateMessage(msg.id, {
      imageDisplayed: true,
      imageDisplayedAt: Date.now(),
      imageLoadError: false,
    });
  }, [updateMessage]);

  const onImageError = useCallback(async (msg) => {
    if (!msg?.id) return;

    if (msg.imageRetryAttempted) {
      updateMessage(msg.id, { imageLoadError: true, imageDisplayed: false });
      return;
    }

    updateMessage(msg.id, { imageLoadError: true, imageDisplayed: false, imageRetryAttempted: true });

    if (msg.serverImageUrl) {
      updateMessage(msg.id, { imageUrl: msg.serverImageUrl });
      return;
    }

    if (msg.params) {
      const imageUrl = await getImageFromCache(msg.params);
      if (imageUrl) {
        updateMessage(msg.id, { imageUrl, needsReload: false });
      } else {
        updateMessage(msg.id, { needsReload: true });
      }
    }
  }, [getImageFromCache, updateMessage]);  

  // ============================================================================
  // LOCAL UI STATE
  // ============================================================================

  // Super-resolution upload
  const [uploadFile, setUploadFile] = useState(null);
  const [srMagnitude, setSrMagnitude] = useState(SR_CONFIG.DEFAULT_MAGNITUDE);

  // Init image for img2img generation
  const [initImage, setInitImage] = useState(null);
  const [sourceDefaultDenoiseStrength, setSourceDefaultDenoiseStrength] = useState(
    DEFAULT_IMG2IMG_DENOISE_STRENGTH
  );
  const initImageRef = useRef(initImage);

  useEffect(() => {
    initImageRef.current = initImage;
  }, [initImage]);

  const updateInitImage = useCallback((nextInitImage) => {
    setInitImage((current) => {
      if (
        current?.objectUrl &&
        (!nextInitImage || current.objectUrl !== nextInitImage.objectUrl)
      ) {
        URL.revokeObjectURL(current.objectUrl);
      }
      return nextInitImage;
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    const restoreActiveInitImage = async () => {
      const restored = await loadActiveSource();
      if (!restored || cancelled) return;

      const file = new File([restored.blob], restored.filename, {
        type: restored.mimeType || restored.blob?.type || 'application/octet-stream',
      });
      const objectUrl = URL.createObjectURL(restored.blob);

      updateInitImage({
        sourceId: restored.id,
        originType: restored.originType,
        originMessageId: restored.originMessageId ?? null,
        file,
        objectUrl,
        filename: restored.filename,
        cacheKey: restored.cacheKey ?? null,
        serverImageUrl: restored.serverImageUrl ?? null,
      });
      const nextDefault =
        restored.defaultDenoiseStrength ?? DEFAULT_IMG2IMG_DENOISE_STRENGTH;
      setSourceDefaultDenoiseStrength(nextDefault);
    };

    restoreActiveInitImage();

    return () => {
      cancelled = true;
    };
  }, [updateInitImage]);

  useEffect(() => {
    return () => {
      if (initImage?.objectUrl) {
        URL.revokeObjectURL(initImage.objectUrl);
      }
    };
  }, [initImage]);

  const persistInitImageSelection = useCallback(
    async (file) => {
      if (!file) return;

      const row = await saveSource({
        originType: 'upload',
        blob: file,
        mimeType: file.type,
        filename: file.name,
        defaultDenoiseStrength: DEFAULT_IMG2IMG_DENOISE_STRENGTH,
      });

      setActiveSourceId(row.id);
      const nextDefault = row.defaultDenoiseStrength ?? DEFAULT_IMG2IMG_DENOISE_STRENGTH;
      setSourceDefaultDenoiseStrength(nextDefault);
      updateInitImage({
        sourceId: row.id,
        originType: row.originType,
        originMessageId: null,
        file,
        objectUrl: URL.createObjectURL(file),
        filename: row.filename,
        cacheKey: row.cacheKey ?? null,
        serverImageUrl: row.serverImageUrl ?? null,
      });
    },
    [updateInitImage]
  );

  const clearInitImage = useCallback(async () => {
    updateInitImage(null);
    setSourceDefaultDenoiseStrength(DEFAULT_IMG2IMG_DENOISE_STRENGTH);
    await clearActiveSource();
  }, [updateInitImage]);


  // Copy feedback
  const [copied, setCopied] = useState(false);
  const [blurredSelection, setBlurredSelection] = useState(null);
  const modeState = useModeConfig();

  // Generation parameters (draft + selected)
  const params = useGenerationParams(
    selectedParams,
    patchSelectedParams,
    runGenerate,
    selectedMsgId,
    initImage?.file || null,
    sourceDefaultDenoiseStrength
  );
  const lastAppliedModeDraftDefaultsRef = useRef(null);

  useEffect(() => {
    const syncPlan = getModeDefaultsSyncPlan(
      modeState,
      params.draft,
      lastAppliedModeDraftDefaultsRef.current
    );
    if (!syncPlan) return;
    params.applyModeControlDefaults(syncPlan.mode);
    lastAppliedModeDraftDefaultsRef.current = syncPlan.draftDefaults;
  }, [
    modeState.activeMode,
    modeState.activeModeName,
    modeState.config,
    params,
  ]); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist edits on selection change so message state reflects current controls
  const persistSelectedParams = useCallback((id, patch) => {
    if (!id) return;
    updateMessage(id, (msg) => {
      if (!msg || msg.kind !== 'image') return msg;
      const next = { ...msg, params: { ...(msg.params || {}), ...patch } };
      if (Array.isArray(msg.imageHistory) && msg.imageHistory.length > 0) {
        const lastIdx = msg.imageHistory.length - 1;
        const currentIdx = msg.historyIndex ?? lastIdx;
        if (currentIdx === lastIdx) {
          const entry = msg.imageHistory[lastIdx];
          return {
            ...next,
            imageUrl: entry?.serverImageUrl || entry?.imageUrl || next.imageUrl,
            serverImageUrl: entry?.serverImageUrl || next.serverImageUrl,
            serverImageKey: entry?.serverImageKey || next.serverImageKey,
          };
        }
      }
      return next;
    });
  }, [updateMessage]);

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
      negativePrompt: params.effective.negativePrompt,
      schedulerId: params.effective.schedulerId,
      size: params.effective.size,
      steps: params.effective.steps,
      cfg: params.effective.cfg,
      seedMode: params.draft.seedMode,
      seed: params.draft.seed,
      superresLevel: params.effective.superresLevel,
      initImageFile: initImage?.file || null,
      denoiseStrength: params.draft.denoiseStrength,
    });
  }, [selectedParams, runGenerate, params, initImage]);

  // Debounced img2img: fire a generation whenever initImage or denoiseStrength changes.
  // Snapshot params at the moment of change so the timer closure is never stale.
  const initImageTimerRef = useRef(null);
  useEffect(() => {
    if (!initImage) {
      window.clearTimeout(initImageTimerRef.current);
      return;
    }
    const snapshot = {
      prompt: params.effective.prompt,
      negativePrompt: params.effective.negativePrompt,
      schedulerId: params.effective.schedulerId,
      size: params.effective.size,
      steps: params.effective.steps,
      cfg: params.effective.cfg,
      seedMode: params.draft.seedMode,
      seed: params.draft.seed,
      superresLevel: params.effective.superresLevel,
      initImageFile: initImage.file,
      denoiseStrength: params.draft.denoiseStrength,
    };
    window.clearTimeout(initImageTimerRef.current);
    initImageTimerRef.current = window.setTimeout(() => runGenerate(snapshot), 180);
    return () => window.clearTimeout(initImageTimerRef.current);
  }, [initImage, params.draft.denoiseStrength]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Re-run the currently selected image with its params.
   */
  const onRerunSelected = useCallback(() => {
    if (!selectedParams) return;

    runGenerate({
      prompt: selectedParams.prompt,
      negativePrompt: selectedParams.negativePrompt,
      schedulerId: selectedParams.schedulerId,
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
      const payload = buildSelectedSeedDeltaPayload(
        selectedParams,
        selectedMsgId,
        delta,
        initImage?.file || null,
        params.effective.denoiseStrength
      );
      if (!payload) return;
      runGenerate(payload);
    },
    [selectedParams, selectedMsgId, initImage, params.effective.denoiseStrength, runGenerate]
  );

  /* explicit selectedmessage state */
  const selectedImage = useMemo(() => {

    if (!selectedMsg) return null;
    const url = selectedMsg.imageUrl;
    if (!url) return null;

    const image_filename = `chat_${selectedMsg.id}.png`
    
    return {
      kind: "url",
      url,
      filename: image_filename,
      source: "chat",
      key: selectedMsg.id,
    };
  }, [selectedMsg]);

  /**
   * Handle super-resolution upload.
   */
  const onSuperResUpload = useCallback(() => {
    if (!uploadFile) return;
    runSuperResUpload(uploadFile, srMagnitude);
  }, [uploadFile, srMagnitude, runSuperResUpload]);

  const onSuperResSelected = useCallback(async () => {
    const url = selectedMsg?.serverImageUrl || selectedMsg?.imageUrl;
    if (!url) return;
    const response = await fetch(url);
    const blob = await response.blob();
    const file = new File([blob], `sr_${selectedMsg.id}.png`, { type: blob.type || 'image/png' });
    runSuperResUpload(file, srMagnitude);
  }, [selectedMsg, srMagnitude, runSuperResUpload]);

  const onAddToGallery = useCallback(async (cacheKey, { serverImageUrl, params: imgParams }) => {
    if (!galleryState.activeGalleryId || !cacheKey) return;
    await galleryState.addToGallery(cacheKey, {
      serverImageUrl,
      params: imgParams,
      galleryId: galleryState.activeGalleryId,
    });
  }, [galleryState]);

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

  // would be better to have a Dispatcher that could track
  // files to context (generate vs comfy)
  const selectedChatImage = useMemo(() => {
    console.log ( (selectedMsg?.serverImageUrl || "foo" )+ "," + (selectedMsg?.imageUrl || "bar" ));
    if (!selectedMsg?.imageUrl) return null;
    return {
      kind: "url",
      url: selectedMsg.serverImageUrl || selectedMsg.imageUrl,
      serverUrl: selectedMsg.serverImageUrl || null,
      cacheKey: selectedMsg.meta?.cacheKey || null,
      filename: `chat_${selectedMsg.id}.png`,
      source: "chat",
      key: selectedMsg.id,
    };
  }, [selectedMsg]);

  const uploadImage = useMemo(() => {
    if (!uploadFile) return null;
    return { kind: "file", file: uploadFile, source: "upload" };
  }, [uploadFile]);

  const generatorInputImage = useMemo(() => {
    // generator prefers explicit upload if present, else selected chat image
      return uploadImage ?? selectedChatImage ?? null;
  }, [uploadImage, selectedChatImage]);

  const handleComposerFocus = useCallback(() => {
    if (selectedMsgId && selectedParams) {
      setBlurredSelection({
        msgId: selectedMsgId,
        params: selectedParams,
      });
    }
    clearSelection();
  }, [selectedMsgId, selectedParams, clearSelection]);

  const handleToggleSelectMsg = useCallback((id) => {
    setBlurredSelection(null);
    toggleSelectMsg(id);
  }, [toggleSelectMsg]);

  const defaultComposer = useMemo(() => ({
    onSendPrompt: (promptText) => {
      const text = String(promptText || "").trim();
      if (!text) return;

      runGenerate({
        prompt: text,
        negativePrompt: params.effective.negativePrompt,
        schedulerId: params.effective.schedulerId,
        size: params.effective.size,
        steps: params.effective.steps,
        cfg: params.effective.cfg,
        seedMode: params.draft.seedMode,
        seed: params.draft.seed,
        superresLevel: params.effective.superresLevel,
        initImageFile: initImage?.file || null,
        denoiseStrength: params.draft.denoiseStrength,
      });
    },
    onCancelAll: cancelAll,
    onKeyDown,
    onFocus: handleComposerFocus,
  }), [
    runGenerate,
    params.effective.size,
    params.effective.steps,
    params.effective.cfg,
    params.draft.seedMode,
    params.draft.seed,
    params.effective.superresLevel,
    params.draft.denoiseStrength,
    initImage,
    cancelAll,
    onKeyDown,
    handleComposerFocus,
  ]);
  // Slash-command dispatch context — passed down into MessageComposer via ChatContainer
  const slashCtx = useMemo(() => ({
    addMessage,
    updateMessage,
    createErrorMessage,
    activeMode: modeState.activeModeName ?? null,
    chatEnabled: Boolean(modeState.activeMode?.chat_enabled),
    inputMode,
    setInputMode,
    chatJob,
    runGenerate,
    onSendPrompt: defaultComposer.onSendPrompt,
    wsConnected: ws.connected,
  }), [
    addMessage,
    updateMessage,
    createErrorMessage,
    modeState.activeModeName,
    modeState.activeMode,
    inputMode,
    chatJob,
    runGenerate,
    defaultComposer.onSendPrompt,
    ws.connected,
  ]);

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
<div className="h-screen overflow-hidden bg-indigo-200 bg-background text-foreground">
  <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
    {/* Tab Navigation */}
    <div className="border-b px-4 flex items-center">
      <TabsList className="h-12">
        <TabsTrigger value="chat" className="gap-2">
          <MessageSquare className="h-4 w-4" />
          Main Chat
        </TabsTrigger>
        <TabsTrigger value="config" className="gap-2">
          <Settings className="h-4 w-4" />
          Configuration
        </TabsTrigger>
      </TabsList>

      {/* Gallery controls — siblings of TabsList, not children */}
      <div className="flex items-center gap-1 ml-2">
        <GalleryCreatePopover onCreateGallery={galleryState.createGallery} />
        {galleryState.galleries.map((g) => (
          <button
            key={g.id}
            type="button"
            onClick={() => setOpenGalleryId(g.id)}
            className="inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded-md hover:bg-muted transition-colors truncate max-w-[120px]"
            title={g.name}
          >
            <Folder className="h-4 w-4 shrink-0" />
            {g.name}
          </button>
        ))}
      </div>
    </div>

{/* Tab Content */}
<div className="flex-1 overflow-hidden">
  {/* Main Chat Tab */}

      <TabsContent value="chat" className="h-full m-0"> 
          <ChatDropzone
        addMessage={addMessage}
        setSelectedMsgId={setSelectedMsgId}
        setUploadFile={setUploadFile}
      >       
      <div className="mx-auto max-w-6xl p-4 md:p-6 h-full">
        <div className="grid h-full grid-cols-1 gap-4 md:grid-cols-[1fr_360px]">
          {/* Chat Panel */}
          <ChatContainer
            messages={messages}
            selectedMsgId={selectedMsgId}
            blurredSelectedMsgId={blurredSelection?.msgId ?? null}
            onToggleSelect={handleToggleSelectMsg}
            onCancelRequest={(id) => { cleanupMessage(id); cancelRequest(id); deleteMessage(id); }}
            setMsgRef={setMsgRef}
            composer={defaultComposer}
            inflightCount={inflightCount}
            isDreaming={isDreaming}
            dreamMessageId={dreamMessageId}
            onDreamSave={saveDreamAndContinue}
            onDreamHistoryPrev={dreamHistoryPrev}
            onDreamHistoryNext={dreamHistoryNext}
            onDreamHistoryLive={dreamHistoryLive}
            onRetry={(msg) => { if (msg.retryParams) runGenerate(msg.retryParams); }}
            srLevel={params.effective.superresLevel}
            frontendVersion={frontendVersion}
            backendVersion={modeState.runtimeStatus?.backend_version}
            onCopyPrompt={onCopyPrompt}
            copied={copied}
            serverLabel={serverLabel}
            onImageDisplayed={onImageDisplayed}
            onImageError={onImageError}
            activeGalleryId={galleryState.activeGalleryId}
            onAddToGallery={onAddToGallery}
            slashCtx={slashCtx}
            inputMode={inputMode}
            onSetInputMode={setInputMode}
          />

          {/* Options Panel */}
          <OptionsPanel
            inputImage={generatorInputImage}
            comfyInputImage={selectedChatImage}
            params={params}
            selectedParams={selectedParams}
            blurredSelectedParams={blurredSelection?.params ?? null}
            selectedMsgId={selectedMsgId}
            onClearSelection={clearSelection}
            onApplyPromptDelta={onApplyPromptDelta}
            onApplySeedDelta={onApplySeedDelta}
            onRerunSelected={onRerunSelected}
            onPersistSelectedParams={persistSelectedParams}
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
            onSuperResSelected={onSuperResSelected}
            uploadFile={uploadFile}
            onUploadFileChange={setUploadFile}
            srMagnitude={srMagnitude}
            onSrMagnitudeChange={setSrMagnitude}
            serverLabel={serverLabel}
            onClearCache={clearCache}
            getCacheStats={getCacheStats}
            onClearHistory={clearHistory}
            initImage={initImage}
            onClearInitImage={clearInitImage}
            denoiseStrength={params.effective.denoiseStrength}
            onDenoiseStrengthChange={params.setDenoiseStrength}
            modeState={modeState}
            onRunComfy={(payload) => {
              const history = selectedMsg?.imageHistory || [];
              // Bootstrap history with current image if message exists but has no history
              if (selectedMsg && history.length === 0 && selectedMsg.imageUrl) {
                history.push({
                  imageUrl: selectedMsg.imageUrl,
                  serverImageUrl: selectedMsg.serverImageUrl,
                  params: selectedMsg.params,
                  meta: selectedMsg.meta,
                });
              }
              runComfy({
                ...payload,
                targetMessageId: selectedMsgId || undefined,
                existingHistory: history,
                skipAutoSelect: true,
              });
            }}
            queueState={queueState}
            galleryState={galleryState}
            onOpenTrash={() => setTrashOpen(true)}
          />
        </div>
      </div>
      </ChatDropzone>
    </TabsContent>
  

          {/* Configuration Tab */}
          <TabsContent value="config" className="h-full m-0 overflow-auto">
            <ModeEditor modeState={modeState} />

            <WorkflowEditor />
          </TabsContent>
  
        </div>
        {openGalleryId && (
          <GalleryLightbox
            galleryId={openGalleryId}
            galleryName={galleryState.galleries.find((g) => g.id === openGalleryId)?.name ?? ''}
            getGalleryImages={galleryState.getGalleryImages}
            onClose={() => setOpenGalleryId(null)}
            onMoveToTrash={galleryState.moveToTrash}
            onRestoreFromTrash={galleryState.restoreFromTrash}
            onHardDelete={galleryState.hardDelete}
          />
        )}
        {trashOpen && (
          <GalleryLightbox
            galleryId={galleryState.TRASH_GALLERY_ID}
            galleryName="Trash"
            trashMode
            getGalleryImages={galleryState.getTrashItems}
            onClose={() => setTrashOpen(false)}
            onMoveToTrash={galleryState.moveToTrash}
            onRestoreFromTrash={galleryState.restoreFromTrash}
            onHardDelete={galleryState.hardDelete}
          />
        )}
      </Tabs>

    </div>
  );
}
