// src/hooks/useScrollManagement.js

import { useRef, useCallback, useEffect } from 'react';
import { isNearBottom } from '../utils/helpers';
import { SCROLL_CONFIG } from '../utils/constants';

/**
 * Hook for managing auto-scroll behavior in chat.
 * Handles scroll detection, auto-scrolling, and selection-based centering.
 * 
 * @param {object[]} messages - Array of chat messages
 * @param {string|null} selectedMsgId - Currently selected message ID
 * @param {Map} msgRefs - Map of message IDs to DOM elements
 * @returns {object} Scroll refs and handlers
 * 
 * @example
 * const {
 *   chatScrollRef,
 *   chatBottomRef,
 *   onChatScroll,
 * } = useScrollManagement(messages, selectedMsgId, msgRefs);
 * 
 * <div ref={chatScrollRef} onScroll={onChatScroll}>
 *   {messages.map(...)}
 *   <div ref={chatBottomRef} />
 * </div>
 */
export function useScrollManagement(messages, selectedMsgId, msgRefs) {
  // Ref to the scrollable container
  const chatScrollRef = useRef(null);
  
  // Ref to the bottom sentinel element
  const chatBottomRef = useRef(null);
  
  // Track whether user is near bottom (for auto-scroll decision)
  const autoScrollRef = useRef(true);

  /**
   * Handle scroll event - update auto-scroll flag.
   */
  const onChatScroll = useCallback(() => {
    const el = chatScrollRef.current;
    autoScrollRef.current = isNearBottom(el, SCROLL_CONFIG.NEAR_BOTTOM_THRESHOLD_PX);
  }, []);

  /**
   * Scroll to a specific message element.
   * @param {string} messageId - Message ID to scroll to
   * @param {string} [block] - Scroll alignment (default: "center")
   */
  const scrollToMessage = useCallback(
    (messageId, block = SCROLL_CONFIG.BLOCK_CENTER) => {
      const el = msgRefs.current.get(messageId);
      if (el) {
        el.scrollIntoView({ block, behavior: SCROLL_CONFIG.BEHAVIOR });
      }
    },
    [msgRefs]
  );

  /**
   * Scroll to bottom of chat.
   */
  const scrollToBottom = useCallback(() => {
    chatBottomRef.current?.scrollIntoView({
      block: SCROLL_CONFIG.BLOCK_END,
      behavior: SCROLL_CONFIG.BEHAVIOR,
    });
  }, []);

  /**
   * Force enable auto-scroll (useful after user actions).
   */
  const enableAutoScroll = useCallback(() => {
    autoScrollRef.current = true;
    scrollToBottom();
  }, [scrollToBottom]);

  /**
   * Auto-scroll effect - runs when messages change or selection changes.
   */
  useEffect(() => {
    const container = chatScrollRef.current;
    if (!container) return;

    // If a message is selected, keep it centered
    if (selectedMsgId) {
      const el = msgRefs.current.get(selectedMsgId);
      if (el) {
        el.scrollIntoView({
          block: SCROLL_CONFIG.BLOCK_CENTER,
          behavior: SCROLL_CONFIG.BEHAVIOR,
        });
      }
      return;
    }

    // No selection: auto-scroll only if user is near bottom
    if (autoScrollRef.current) {
      chatBottomRef.current?.scrollIntoView({
        block: SCROLL_CONFIG.BLOCK_END,
        behavior: SCROLL_CONFIG.BEHAVIOR,
      });
    }
  }, [messages.length, selectedMsgId, msgRefs]);

  return {
    chatScrollRef,
    chatBottomRef,
    onChatScroll,
    scrollToMessage,
    scrollToBottom,
    enableAutoScroll,
    autoScrollRef,
  };
}