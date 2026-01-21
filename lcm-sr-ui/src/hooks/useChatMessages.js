// src/hooks/useChatMessages.js

import { useState, useCallback, useMemo, useRef } from 'react';
import { nowId } from '../utils/helpers';
import { UI_MESSAGES, MESSAGE_KINDS, MESSAGE_ROLES } from '../utils/constants';

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
  // Message list with initial system message
  const [messages, setMessages] = useState(() => [
    {
      id: nowId(),
      role: MESSAGE_ROLES.ASSISTANT,
      kind: MESSAGE_KINDS.SYSTEM,
      text: UI_MESSAGES.INITIAL_SYSTEM,
      ts: Date.now(),
    },
  ]);

  // Currently selected message ID (for editing params)
  const [selectedMsgId, setSelectedMsgId] = useState(null);

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
   * @param {object} patch - Partial message data to merge
   */
  const updateMessage = useCallback((id, patch) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg))
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
  };
}