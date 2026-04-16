// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useKeymap } from './useKeymap';

function mockFetchDefaults(keymap) {
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => ({ keymap }),
  }));
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  delete globalThis.fetch;
});

describe('useKeymap', () => {
  it('fetches defaults and matches keydown events by code', async () => {
    mockFetchDefaults({
      delete: { code: 'Backspace', label: 'Delete' },
      next: { code: 'ArrowRight', label: 'Next' },
    });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    const event = { code: 'Backspace', metaKey: false, ctrlKey: false, shiftKey: false, altKey: false };
    expect(result.current.matches('delete', event)).toBe(true);
    expect(result.current.matches('next', event)).toBe(false);
  });

  it('applies localStorage override over server defaults', async () => {
    mockFetchDefaults({ delete: { code: 'Backspace', label: 'Delete' } });
    localStorage.setItem(
      'lcm-keymap-overrides',
      JSON.stringify({ delete: { code: 'KeyD', label: 'Delete' } }),
    );

    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    expect(result.current.matches('delete', { code: 'KeyD', metaKey: false, ctrlKey: false })).toBe(true);
    expect(result.current.matches('delete', { code: 'Backspace', metaKey: false, ctrlKey: false })).toBe(false);
  });

  it('matches modifier bindings using platform modifier (meta on mac, ctrl elsewhere)', async () => {
    mockFetchDefaults({ select_all: { code: 'KeyA', mod: 'mod', label: 'Select all' } });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    expect(result.current.matches('select_all', { code: 'KeyA', metaKey: true, ctrlKey: false })).toBe(true);
    expect(result.current.matches('select_all', { code: 'KeyA', metaKey: false, ctrlKey: true })).toBe(true);
    expect(result.current.matches('select_all', { code: 'KeyA', metaKey: false, ctrlKey: false })).toBe(false);
  });

  it('setBinding writes to localStorage and updates matcher', async () => {
    mockFetchDefaults({ delete: { code: 'Backspace', label: 'Delete' } });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    await act(async () => {
      result.current.setBinding('delete', 'KeyX');
    });

    const stored = JSON.parse(localStorage.getItem('lcm-keymap-overrides'));
    expect(stored.delete.code).toBe('KeyX');
    expect(result.current.matches('delete', { code: 'KeyX', metaKey: false, ctrlKey: false })).toBe(true);
  });

  it('falls back to hardcoded defaults when fetch fails', async () => {
    globalThis.fetch = vi.fn(async () => { throw new Error('network'); });
    const { result } = renderHook(() => useKeymap());
    await waitFor(() => expect(result.current.ready).toBe(true));

    expect(result.current.matches('delete', { code: 'Backspace', metaKey: false, ctrlKey: false })).toBe(true);
    expect(result.current.matches('deselect_all', { code: 'Escape', metaKey: false, ctrlKey: false })).toBe(true);
  });
});
