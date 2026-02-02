// src/App.jsx

import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react';
import { ChatDropzone } from "./components/chat/ChatDropzone";
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { useChatMessages } from './hooks/useChatMessages';
import { useGenerationParams } from './hooks/useGenerationParams';
import { useImageGeneration } from './hooks/useImageGeneration';
import { ChatContainer } from './components/chat/ChatContainer';
import { OptionsPanel } from './components/options/OptionsPanel';
import { copyToClipboard } from './utils/helpers';
import { SR_CONFIG } from './utils/constants';
import { MessageSquare, Sparkles } from 'lucide-react';
import { uuidv4 } from "@/utils/uuid";
import { useWs } from './hooks/useWs';
import { useJobQueue } from './hooks/useJobQueue';

export default function App() {
  useWs(); // auto-connect WS singleton on mount
  const queueState = useJobQueue();
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

  // API base from env

  const apiBase = "";

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
   *  Handling comfyui image to chat message 
   */
  const pendingIdRef = useRef(null);
  
  
  const onComfyStart = useCallback(() => {
    const id = uuidv4();
    pendingIdRef.current = id;

    addMessage({
      id,
      role: "assistant",
      kind: "pending",
      meta: { backend: "comfy" },
    });
  }, [addMessage]);

  /*
    Handle ComfyUI image->chat
  */
  const onComfyOutputs = useCallback(({ workflowId, params, outputs }) => {
    // if you only care about first output:
    const first = outputs?.[0];
    if (!first) return;

    const id = pendingIdRef.current;
    if (id) {
      updateMessage(id, {
        kind: "image",
        imageUrl: first.url,
        params: { ...params, workflowId },
        meta: { backend: `comfy:${workflowId}` },
      });
      pendingIdRef.current = null;
    } else {
      // fallback if no pending
      addMessage({ role:"assistant", kind:"image", imageUrl:first.url, params:{...params, workflowId}, meta:{backend:`comfy:${workflowId}`}});
    }
  }, [addMessage, updateMessage]);

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

  // would be better to have a Dispatcher that could track
  // files to context (generate vs comfy)
  const selectedChatImage = useMemo(() => {
    console.log ( (selectedMsg?.serverImageUrl || "foo" )+ "," + (selectedMsg?.imageUrl || "bar" ));
    if (!selectedMsg?.imageUrl) return null;
    return {
      kind: "url",
      url: selectedMsg.imageUrl,
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

  const defaultComposer = useMemo(() => ({
    onSendPrompt: (promptText) => {
      const text = String(promptText || "").trim();
      if (!text) return;

      runGenerate({
        prompt: text,
        size: params.effective.size,
        steps: params.effective.steps,
        cfg: params.effective.cfg,
        seedMode: params.draft.seedMode,
        seed: params.draft.seed,
        superresLevel: params.effective.superresLevel,
      });
    },
    onCancelAll: cancelAll,
    onKeyDown,
    onFocus: clearSelection,
  }), [
    runGenerate,
    params.effective.size,
    params.effective.steps,
    params.effective.cfg,
    params.draft.seedMode,
    params.draft.seed,
    params.effective.superresLevel,
    cancelAll,
    onKeyDown,
    clearSelection,
  ]);
  // ============================================================================
  // RENDER
  // ============================================================================

  return (
<div className="h-screen overflow-hidden bg-indigo-200 bg-background text-foreground">
  <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
    {/* Tab Navigation */}
    <div className="border-b px-4">
      <TabsList className="h-12">
        <TabsTrigger value="chat" className="gap-2">
          <MessageSquare className="h-4 w-4" />
          Main Chat
        </TabsTrigger>
        <TabsTrigger value="dreams" className="gap-2">        
          Save Gallery
        </TabsTrigger>
      </TabsList>
    </div>

{/* Tab Content */}
<div className="flex-1 overflow-hidden">
  {/* Main Chat Tab */}
<ChatDropzone
  addMessage={addMessage}
  setSelectedMsgId={setSelectedMsgId}
  setUploadFile={setUploadFile}
>
  <TabsContent value="chat" className="h-full m-0">        
      <div className="mx-auto max-w-6xl p-4 md:p-6 h-full">
        <div className="grid h-full grid-cols-1 gap-4 md:grid-cols-[1fr_360px]">
          {/* Chat Panel */}
          <ChatContainer
            messages={messages}
            selectedMsgId={selectedMsgId}
            onToggleSelect={toggleSelectMsg}
            onCancelRequest={(id) => { cancelRequest(id); deleteMessage(id); }}
            setMsgRef={setMsgRef}
            composer={defaultComposer}
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
            inputImage={generatorInputImage}
            comfyInputImage={selectedChatImage}
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
            onComfyOutputs={onComfyOutputs}
            onComfyStart={onComfyStart}
            queueState={queueState}
          />
        </div>
      </div>
    </TabsContent>
  </ChatDropzone>
        </div>
      </Tabs>

    </div>
  );
}