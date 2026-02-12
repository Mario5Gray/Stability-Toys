// src/hooks/useChatMessages.js

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { nowId } from '../utils/helpers';
import { UI_MESSAGES, MESSAGE_KINDS, MESSAGE_ROLES } from '../utils/constants';

const STORAGE_KEY = 'lcm-chat-messages';

/**
 * Load messages from localStorage.
 * Server URLs are persistent and don't need reload.
 * Blob URLs need reload from cache.
 * Returns null if not found or invalid.
 */
function loadPersistedMessages() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return null;
    const parsed = JSON.parse(saved);
    if (!Array.isArray(parsed) || parsed.length === 0) return null;

    return parsed.map((msg) => {
      if (msg.kind === MESSAGE_KINDS.IMAGE) {
        const nextHistory = Array.isArray(msg.imageHistory)
          ? msg.imageHistory.map((entry) => {
              if (entry?.serverImageUrl) {
                return { ...entry, imageUrl: entry.serverImageUrl };
              }
              if (entry?.imageUrl?.startsWith('blob:')) {
                return { ...entry, imageUrl: null };
              }
              return entry;
            })
          : msg.imageHistory;
        // Server URLs (http/https) are persistent - use directly
        if (msg.imageUrl?.startsWith('http')) {
          return { ...msg, imageHistory: nextHistory };
        }
        // Server URL stored separately - use it
        if (msg.serverImageUrl) {
          return { ...msg, imageUrl: msg.serverImageUrl, imageHistory: nextHistory };
        }
        // Blob URLs or missing - need reload from client cache
        return { ...msg, imageUrl: null, needsReload: true, imageHistory: nextHistory };
      }
      return msg;
    });
  } catch (err) {
    console.warn('[Chat] Failed to load persisted messages:', err);
    return null;
  }
}

/**
 * Save messages to localStorage.
 * Server URLs persist, blob URLs are stripped.
 */
function persistMessages(messages) {
  try {
    const toSave = messages.map((msg) => {
      if (msg.kind === MESSAGE_KINDS.IMAGE) {
        const history = Array.isArray(msg.imageHistory)
          ? msg.imageHistory.map((entry) => {
              if (entry?.serverImageUrl) {
                return { ...entry, imageUrl: entry.serverImageUrl };
              }
              if (entry?.imageUrl?.startsWith('blob:')) {
                return { ...entry, imageUrl: null };
              }
              return entry;
            })
          : msg.imageHistory;
        // Server URLs are persistent - use as primary imageUrl
        if (msg.serverImageUrl) {
          return { ...msg, imageUrl: msg.serverImageUrl, imageHistory: history };
        }
        // Blob URLs don't persist - mark for reload
        if (msg.imageUrl?.startsWith('blob:')) {
          return { ...msg, imageUrl: null, needsReload: true, imageHistory: history };
        }
        return { ...msg, imageHistory: history };
      }
      return msg;
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
  } catch (err) {
    console.warn('[Chat] Failed to persist messages:', err);
  }
}

/**
 * Hook for managing chat messages and selection state.
 * Handles message CRUD operations, selection, and parameter updates.
 * 
 * @returns {object} Message state and operations
 * 
 * @example
 * const {
 *   messages,
 *   selectedMsgId,
 *   selectedMsg,
 *   selectedParams,
 *   addMessage,
 *   updateMessage,
 *   toggleSelectMsg,
 *   patchSelectedParams,
 *   setMsgRef,
 * } = useChatMessages();
 */
export function useChatMessages() {
  // Message list - load from localStorage or use initial system message
  const [messages, setMessages] = useState(() => {
    const persisted = loadPersistedMessages();
    if (persisted) {
      console.log(`[Chat] Restored ${persisted.length} messages from storage`);
      return persisted;
    }
    return [
      {
        id: nowId(),
        role: MESSAGE_ROLES.ASSISTANT,
        kind: MESSAGE_KINDS.SYSTEM,
        text: UI_MESSAGES.INITIAL_SYSTEM,
        ts: Date.now(),
      },
    ];
  });

  // Currently selected message ID (for editing params)
  const [selectedMsgId, setSelectedMsgId] = useState(null);

  // Persist messages to localStorage on change
  useEffect(() => {
    persistMessages(messages);
  }, [messages]);

  // Refs to DOM elements for each message (for scrolling)
  const msgRefs = useRef(new Map());

  /**
   * Add one or more messages to the chat.
   * @param {object|object[]} newMessages - Single message or array of messages
   */
  const addMessage = useCallback((newMessages) => {
    const msgs = Array.isArray(newMessages) ? newMessages : [newMessages];
    setMessages((prev) => [...prev, ...msgs]);
  }, []);

  /**
   * Update a specific message by ID with partial data.
   * @param {string} id - Message ID to update
   * @param {object|function} patch - Partial message data to merge or updater fn
   */
  const updateMessage = useCallback((id, patch) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== id) return msg;
        const next = typeof patch === 'function' ? patch(msg) : { ...msg, ...patch };
        return next;
      })
    );
  }, []);

  /**
   * Delete a message by ID.
   * @param {string} id - Message ID to delete
   */
  const deleteMessage = useCallback((id) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== id));
  }, []);

  /**
   * Toggle message selection (select if not selected, deselect if selected).
   * @param {string} id - Message ID to toggle
   */
  const toggleSelectMsg = useCallback((id) => {
    console.log("currentMSGID: " + (id ? id : 0))
    setSelectedMsgId((current) => (current === id ? null : id));
  }, []);

  /**
   * Clear message selection.
   */
  const clearSelection = useCallback(() => {
    setSelectedMsgId(null);
  }, []);

  /**
   * Set a ref callback for a message element (for scrolling).
   * @param {string} id - Message ID
   * @returns {function} Ref callback
   */
  const setMsgRef = useCallback((id) => (el) => {
    if (!el) {
      msgRefs.current.delete(id);
    } else {
      msgRefs.current.set(id, el);
    }
  }, []);

  /**
   * Get the currently selected message object.
   */
  const selectedMsg = useMemo(() => {
    return messages.find((m) => m.id === selectedMsgId) || null;
  }, [messages, selectedMsgId]);

  /**
   * Get the params object from the selected message (if it's an image).
   */
  const selectedParams = useMemo(() => {
    if (selectedMsg?.kind === MESSAGE_KINDS.IMAGE) {
      return selectedMsg.params || null;
    }
    return null;
  }, [selectedMsg]);

  /**
   * Update params for the currently selected message.
   * Only works if selected message is an image with params.
   * @param {object} patch - Partial params to merge
   */
  const patchSelectedParams = useCallback(
    (patch) => {
      if (!selectedMsg || selectedMsg.kind !== MESSAGE_KINDS.IMAGE) {
        return;
      }
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === selectedMsg.id
            ? { ...msg, params: { ...(msg.params || {}), ...patch } }
            : msg
        )
      );
    },
    [selectedMsg]
  );

  /**
   * Count messages of a specific kind.
   * @param {string} kind - Message kind to count
   * @returns {number} Count of messages with that kind
   */
  const countMessagesByKind = useCallback(
    (kind) => {
      return messages.filter((m) => m.kind === kind).length;
    },
    [messages]
  );

  /**
   * Get the DOM element for a specific message.
   * @param {string} id - Message ID
   * @returns {HTMLElement|null} DOM element or null
   */
  const getMessageElement = useCallback((id) => {
    return msgRefs.current.get(id) || null;
  }, []);

  /**
   * Create a user message object.
   * @param {string} text - Message text
   * @param {object} [meta] - Optional metadata
   * @returns {object} User message object
   */
  const createUserMessage = useCallback((text, meta = {}) => {
    return {
      id: nowId(),
      role: MESSAGE_ROLES.USER,
      kind: MESSAGE_KINDS.TEXT,
      text,
      meta,
      ts: Date.now(),
    };
  }, []);

  /**
   * Create a pending assistant message object.
   * @param {string} [text] - Message text (default: "Generating…")
   * @param {object} [meta] - Optional metadata
   * @returns {object} Pending message object
   */
  const createPendingMessage = useCallback((text = UI_MESSAGES.GENERATING, meta = {}) => {
    return {
      id: nowId(),
      role: MESSAGE_ROLES.ASSISTANT,
      kind: MESSAGE_KINDS.PENDING,
      text,
      meta,
      ts: Date.now(),
    };
  }, []);

  /**
   * Create an error message object.
   * @param {string} text - Error message text
   * @param {object} [meta] - Optional metadata
   * @returns {object} Error message object
   */
  const createErrorMessage = useCallback((text, meta = {}) => {
    return {
      id: nowId(),
      role: MESSAGE_ROLES.ASSISTANT,
      kind: MESSAGE_KINDS.ERROR,
      text,
      meta,
      ts: Date.now(),
    };
  }, []);

  /**
   *  Handle keyboard input for 'delete', and 'undo'
   **/
  function isTypingTarget(el) {
    if (!el) return false;
    const tag = el.tagName?.toLowerCase();
    return (
      tag === "input" ||
      tag === "textarea" ||
      tag === "select" ||
      el.isContentEditable
    );
  }

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return; // avoid shortcuts
      if (isTypingTarget(e.target)) return;

      if (!selectedMsgId) return;

      e.preventDefault();

      // implement this however your chat state stores messages
      // Option A: you already have a helper:
      //deleteMessage(selectedMsgId);

      // Option B: filter it out:
      setMessages((prev) => prev.filter((m) => m.id !== selectedMsgId));

      clearSelection?.(); // or setSelectedMsgId(null)
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedMsgId, clearSelection /*, deleteMessage, setMessages */]);


  /**
   * Reload images from cache for messages marked with needsReload.
   * Call this on app init with the cache instance.
   * @param {object} cache - Cache with get() method
   * @param {function} keyFn - Function to generate cache key from params
   */
  const reloadImagesFromCache = useCallback(async (cache, keyFn) => {
    if (!cache) return;

    const toReload = messages.filter(
      (m) => m.kind === MESSAGE_KINDS.IMAGE && m.needsReload && m.params
    );

    if (toReload.length === 0) return;

    console.log(`[Chat] Reloading ${toReload.length} images from cache...`);

    for (const msg of toReload) {
      try {
        const cacheKey = keyFn(msg.params);
        const cached = await cache.get(cacheKey);

        if (cached?.blob) {
          const imageUrl = URL.createObjectURL(cached.blob);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === msg.id
                ? { ...m, imageUrl, needsReload: false }
                : m
            )
          );
          console.log(`[Chat] Reloaded image for message ${msg.id.slice(0, 8)}`);
        } else {
          // Not in cache - mark as unavailable
          setMessages((prev) =>
            prev.map((m) =>
              m.id === msg.id
                ? { ...m, needsReload: false, cacheExpired: true }
                : m
            )
          );
        }
      } catch (err) {
        console.warn(`[Chat] Failed to reload image ${msg.id}:`, err);
      }
    }
  }, [messages]);

  /**
   * Clear all chat history and reset to initial state.
   */
  const clearHistory = useCallback(() => {
    setMessages([
      {
        id: nowId(),
        role: MESSAGE_ROLES.ASSISTANT,
        kind: MESSAGE_KINDS.SYSTEM,
        text: UI_MESSAGES.INITIAL_SYSTEM,
        ts: Date.now(),
      },
    ]);
    setSelectedMsgId(null);
    localStorage.removeItem(STORAGE_KEY);
    console.log('[Chat] History cleared');
  }, []);

  return {
    // State
    messages,
    selectedMsgId,
    selectedMsg,
    selectedParams,
    msgRefs,

    // CRUD operations
    addMessage,
    updateMessage,
    deleteMessage,

    // Selection
    toggleSelectMsg,
    clearSelection,
    setSelectedMsgId,

    // Params manipulation
    patchSelectedParams,

    // Refs
    setMsgRef,
    getMessageElement,

    // Utilities
    countMessagesByKind,
    createUserMessage,
    createPendingMessage,
    createErrorMessage,

    // Persistence
    reloadImagesFromCache,
    clearHistory,
  };
}
