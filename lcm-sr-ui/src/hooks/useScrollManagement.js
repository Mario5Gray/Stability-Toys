// src/hooks/useScrollManagement.js

import { useRef, useCallback, useEffect } from 'react';
import { SCROLL_CONFIG } from '../utils/constants';

/**
 * Hook for "sticky bottom" scroll behavior.
 *
 * Simple rule:
 * - If user is at the bottom, stay stuck to bottom (auto-scroll on new content)
 * - If user scrolls away, stay where they are (no auto-scroll)
 * - Clicking "scroll to bottom" re-enables sticky behavior
 *
 * @param {object[]} messages - Array of chat messages
 * @param {string|null} selectedMsgId - Currently selected message ID
 * @param {Map} msgRefs - Map of message IDs to DOM elements
 * @returns {object} Scroll refs and handlers
 */
export function useScrollManagement(messages, selectedMsgId, msgRefs) {
  // Ref to the scrollable container
  const chatScrollRef = useRef(null);

  // Ref to the bottom sentinel element
  const chatBottomRef = useRef(null);

  // Track if we're "stuck" to bottom - starts true
  const isStuckToBottom = useRef(true);

  // Track the last selected message to scroll to it once
  const lastSelectedRef = useRef(null);

  /**
   * Check if scroll position is at the bottom.
   */
  const checkIfAtBottom = useCallback((el) => {
    if (!el) return true;
    // Within 20px of bottom counts as "at bottom"
    return el.scrollHeight - el.scrollTop - el.clientHeight < 20;
  }, []);

  /**
   * Handle scroll event - update sticky state.
   */
  const onChatScroll = useCallback(() => {
    const el = chatScrollRef.current;
    if (!el) return;

    // Update sticky state based on scroll position
    isStuckToBottom.current = checkIfAtBottom(el);
  }, [checkIfAtBottom]);

  /**
   * Scroll to a specific message element.
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
   * Scroll to bottom by setting scrollTop directly.
   */
  const scrollToBottom = useCallback((smooth = true) => {
    const el = chatScrollRef.current;
    if (!el) return;

    if (smooth) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    } else {
      el.scrollTop = el.scrollHeight;
    }
  }, []);

  /**
   * Force enable sticky-to-bottom (for the scroll button).
   */
  const enableAutoScroll = useCallback(() => {
    isStuckToBottom.current = true;
    scrollToBottom(true); // smooth scroll when user clicks button
  }, [scrollToBottom]);

  /**
   * Scroll to newly selected message (one-time).
   */
  useEffect(() => {
    if (selectedMsgId && selectedMsgId !== lastSelectedRef.current) {
      lastSelectedRef.current = selectedMsgId;
      const el = msgRefs.current.get(selectedMsgId);
      if (el) {
        el.scrollIntoView({
          block: SCROLL_CONFIG.BLOCK_CENTER,
          behavior: SCROLL_CONFIG.BEHAVIOR,
        });
      }
    } else if (!selectedMsgId) {
      lastSelectedRef.current = null;
    }
  }, [selectedMsgId, msgRefs]);

  /**
   * Sticky bottom effect - when messages change, scroll to bottom if stuck.
   */
  useEffect(() => {
    if (!isStuckToBottom.current) return;

    const el = chatScrollRef.current;
    if (!el) return;

    // Use requestAnimationFrame to ensure DOM has updated
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }, [messages.length]);

  return {
    chatScrollRef,
    chatBottomRef,
    onChatScroll,
    scrollToMessage,
    scrollToBottom,
    enableAutoScroll,
    autoScrollRef: isStuckToBottom, // Expose for external checks
  };
}
