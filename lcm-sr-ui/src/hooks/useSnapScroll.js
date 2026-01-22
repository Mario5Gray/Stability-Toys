// src/hooks/useSnapScroll.js

import { useRef, useCallback, useEffect } from 'react';

/**
 * Snap-to-image scroll behavior with physics-based resistance.
 * Images "lock" at center with magnetic divots.
 * 
 * @param {object[]} messages - Array of messages
 * @param {Map} msgRefs - Map of message IDs to DOM elements
 * @returns {object} Scroll control functions
 */
export function useSnapScroll(messages, msgRefs) {
  const scrollContainerRef = useRef(null);
  const velocityRef = useRef(0);
  const lastScrollTimeRef = useRef(0);
  const lastScrollTopRef = useRef(0);
  const isScrollingRef = useRef(false);
  const snapTimeoutRef = useRef(null);
  const currentSnapIndexRef = useRef(0);

  // Physics constants
  const SNAP_THRESHOLD = 0.3; // How close to center to trigger snap (0-1)
  const RESISTANCE_STRENGTH = 0.7; // How strong the center resistance (0-1)
  const VELOCITY_THRESHOLD = 50; // px/s - below this, trigger snap
  const SNAP_DURATION = 400; // ms - snap animation duration
  const FLICK_THRESHOLD = 800; // px/s - "hard scroll" velocity

  /**
   * Get all image message elements in scroll order.
   */
  const getImageElements = useCallback(() => {
    return messages
      .filter(m => m.kind === 'image')
      .map(m => ({
        id: m.id,
        element: msgRefs.current.get(m.id),
      }))
      .filter(item => item.element);
  }, [messages, msgRefs]);

  /**
   * Get distance from element center to viewport center.
   * Returns { distance, ratio } where ratio is -1 to 1.
   */
  const getDistanceFromCenter = useCallback((element) => {
    if (!scrollContainerRef.current) return { distance: Infinity, ratio: 0 };

    const container = scrollContainerRef.current;
    const containerRect = container.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();

    const containerCenter = containerRect.top + containerRect.height / 2;
    const elementCenter = elementRect.top + elementRect.height / 2;
    
    const distance = elementCenter - containerCenter;
    const ratio = distance / (containerRect.height / 2); // -1 to 1

    return { distance, ratio };
  }, []);

  /**
   * Find the closest image to center.
   */
  const findClosestImage = useCallback(() => {
    const images = getImageElements();
    if (images.length === 0) return null;

    let closest = null;
    let minDistance = Infinity;

    images.forEach((img, index) => {
      const { distance } = getDistanceFromCenter(img.element);
      const absDistance = Math.abs(distance);
      
      if (absDistance < minDistance) {
        minDistance = absDistance;
        closest = { ...img, index, distance: absDistance };
      }
    });

    return closest;
  }, [getImageElements, getDistanceFromCenter]);

  /**
   * Snap to specific image with smooth animation.
   */
  const snapToImage = useCallback((element, force = false) => {
    if (!scrollContainerRef.current || !element) return;

    const container = scrollContainerRef.current;
    const containerRect = container.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();

    const containerCenter = containerRect.top + containerRect.height / 2;
    const elementCenter = elementRect.top + elementRect.height / 2;
    const offset = elementCenter - containerCenter;

    // Smooth scroll to center
    container.scrollBy({
      top: offset,
      behavior: force ? 'auto' : 'smooth',
    });
  }, []);

  /**
   * Calculate resistance factor based on distance from center.
   * Returns 0-1, where 1 = maximum resistance at center.
   */
  const getResistanceFactor = useCallback((ratio) => {
    const absRatio = Math.abs(ratio);
    
    // Quadratic falloff: resistance strongest near center
    const resistance = Math.max(0, 1 - absRatio * absRatio);
    
    return resistance * RESISTANCE_STRENGTH;
  }, []);

  /**
   * Handle scroll events - calculate velocity and resistance.
   */
  const onScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;

    const now = Date.now();
    const currentScrollTop = scrollContainerRef.current.scrollTop;
    const timeDelta = now - lastScrollTimeRef.current;
    
    if (timeDelta > 0) {
      const scrollDelta = currentScrollTop - lastScrollTopRef.current;
      velocityRef.current = Math.abs(scrollDelta / timeDelta) * 1000; // px/s
    }

    lastScrollTimeRef.current = now;
    lastScrollTopRef.current = currentScrollTop;
    isScrollingRef.current = true;

    // Clear existing snap timeout
    if (snapTimeoutRef.current) {
      clearTimeout(snapTimeoutRef.current);
    }

    // Apply resistance (visual feedback via CSS custom property)
    const closest = findClosestImage();
    if (closest) {
      const { ratio } = getDistanceFromCenter(closest.element);
      const resistance = getResistanceFactor(ratio);
      
      // Set CSS variable for visual feedback (optional)
      scrollContainerRef.current.style.setProperty(
        '--scroll-resistance',
        resistance
      );
    }

    // Schedule snap check after scrolling stops
    snapTimeoutRef.current = setTimeout(() => {
      isScrollingRef.current = false;
      
      // If velocity is low, snap to nearest
      if (velocityRef.current < VELOCITY_THRESHOLD) {
        const closest = findClosestImage();
        if (closest) {
          const { ratio } = getDistanceFromCenter(closest.element);
          
          // Only snap if reasonably close to center
          if (Math.abs(ratio) < SNAP_THRESHOLD) {
            snapToImage(closest.element);
            currentSnapIndexRef.current = closest.index;
          }
        }
      } else if (velocityRef.current > FLICK_THRESHOLD) {
        // Hard flick: jump to next/prev image
        const direction = lastScrollTopRef.current > 
          (scrollContainerRef.current?.scrollTop || 0) ? -1 : 1;
        
        const images = getImageElements();
        const nextIndex = currentSnapIndexRef.current + direction;
        
        if (nextIndex >= 0 && nextIndex < images.length) {
          snapToImage(images[nextIndex].element);
          currentSnapIndexRef.current = nextIndex;
        }
      }
      
      velocityRef.current = 0;
    }, 150); // Wait 150ms after last scroll event
  }, [
    findClosestImage,
    getDistanceFromCenter,
    getResistanceFactor,
    snapToImage,
    getImageElements,
  ]);

  /**
   * Snap to specific message by ID.
   */
  const snapToMessage = useCallback((messageId) => {
    const element = msgRefs.current.get(messageId);
    if (element) {
      snapToImage(element, false);
      
      // Update current index
      const images = getImageElements();
      const index = images.findIndex(img => img.id === messageId);
      if (index !== -1) {
        currentSnapIndexRef.current = index;
      }
    }
  }, [msgRefs, snapToImage, getImageElements]);

  /**
   * Navigate to next image.
   */
  const snapToNext = useCallback(() => {
    const images = getImageElements();
    const nextIndex = Math.min(currentSnapIndexRef.current + 1, images.length - 1);
    
    if (images[nextIndex]) {
      snapToImage(images[nextIndex].element);
      currentSnapIndexRef.current = nextIndex;
    }
  }, [getImageElements, snapToImage]);

  /**
   * Navigate to previous image.
   */
  const snapToPrevious = useCallback(() => {
    const images = getImageElements();
    const prevIndex = Math.max(currentSnapIndexRef.current - 1, 0);
    
    if (images[prevIndex]) {
      snapToImage(images[prevIndex].element);
      currentSnapIndexRef.current = prevIndex;
    }
  }, [getImageElements, snapToImage]);

  /**
   * Auto-snap to latest image when new one arrives.
   */
  useEffect(() => {
    const images = getImageElements();
    if (images.length === 0) return;

    // If we're at or near the bottom, snap to new image
    const container = scrollContainerRef.current;
    if (!container) return;

    const isNearBottom = 
      container.scrollHeight - container.scrollTop - container.clientHeight < 100;

    if (isNearBottom) {
      const lastImage = images[images.length - 1];
      if (lastImage) {
        // Small delay to ensure image is rendered
        setTimeout(() => {
          snapToImage(lastImage.element, true);
          currentSnapIndexRef.current = images.length - 1;
        }, 100);
      }
    }
  }, [messages.length, getImageElements, snapToImage]);

  /**
   * Keyboard navigation.
   */
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return; // Don't interfere with text input
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        snapToNext();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        snapToPrevious();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [snapToNext, snapToPrevious]);

  /**
   * Cleanup.
   */
  useEffect(() => {
    return () => {
      if (snapTimeoutRef.current) {
        clearTimeout(snapTimeoutRef.current);
      }
    };
  }, []);

  return {
    scrollContainerRef,
    onScroll,
    snapToMessage,
    snapToNext,
    snapToPrevious,
    currentSnapIndex: currentSnapIndexRef.current,
  };
}