// @vitest-environment jsdom

import { renderHook, waitFor, act } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useModeConfig } from './useModeConfig';

const api = vi.hoisted(() => ({
  client: {
    fetchGet: vi.fn(),
    fetchPost: vi.fn(),
  },
}));

vi.mock('../utils/api', () => ({
  createApiClient: vi.fn(() => api.client),
  createApiConfig: vi.fn(() => ({})),
}));

afterEach(() => {
  vi.clearAllMocks();
});

describe('useModeConfig', () => {
  it('keeps runtime active mode separate from the default mode and loads runtime status', async () => {
    api.client.fetchGet.mockImplementation(async (endpoint) => {
      if (endpoint === '/api/modes') {
        return {
          default_mode: 'cinematic',
          modes: {
            cinematic: { model: 'base-cinematic' },
            portrait: { model: 'base-portrait' },
          },
        };
      }

      if (endpoint === '/api/models/status') {
        return {
          current_mode: 'portrait',
          is_loaded: true,
        };
      }

      throw new Error(`Unexpected endpoint: ${endpoint}`);
    });

    const { result } = renderHook(() => useModeConfig());

    await waitFor(() => expect(result.current.config).not.toBeNull());
    await waitFor(() => expect(result.current.activeModeName).toBe('portrait'));

    expect(result.current.defaultModeName).toBe('cinematic');
    expect(result.current.activeMode).toEqual({ model: 'base-portrait' });
    expect(api.client.fetchGet).toHaveBeenCalledWith('/api/modes');
    expect(api.client.fetchGet).toHaveBeenCalledWith('/api/models/status');
  });

  it('refreshes runtime status after a successful mode switch', async () => {
    api.client.fetchGet.mockImplementation(async (endpoint) => {
      if (endpoint === '/api/modes') {
        return {
          default_mode: 'cinematic',
          modes: {
            cinematic: { model: 'base-cinematic' },
            portrait: { model: 'base-portrait' },
          },
        };
      }

      if (endpoint === '/api/models/status') {
        return {
          current_mode: 'cinematic',
          is_loaded: true,
        };
      }

      throw new Error(`Unexpected endpoint: ${endpoint}`);
    });
    api.client.fetchPost.mockResolvedValue({ status: 'ok' });

    const { result } = renderHook(() => useModeConfig());

    await waitFor(() => expect(result.current.config).not.toBeNull());

    await act(async () => {
      await result.current.switchMode('portrait');
    });

    await waitFor(() => expect(api.client.fetchGet).toHaveBeenCalledTimes(3));
  });
});
