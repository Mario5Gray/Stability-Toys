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
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('useModeConfig', () => {
  it('posts a switch for the default mode when runtime status is unavailable', async () => {
    let statusCalls = 0;
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
        statusCalls += 1;
        if (statusCalls === 1) {
          throw new Error('status unavailable');
        }
        return {
          current_mode: 'cinematic',
          is_loaded: true,
          backend_version: 'abc1234',
        };
      }

      throw new Error(`Unexpected endpoint: ${endpoint}`);
    });
    api.client.fetchPost.mockResolvedValue({ status: 'ok' });

    const { result } = renderHook(() => useModeConfig());

    await waitFor(() => expect(result.current.defaultModeName).toBe('cinematic'));

    await act(async () => {
      await result.current.switchMode(result.current.defaultModeName);
    });

    expect(api.client.fetchPost).toHaveBeenCalledWith('/api/modes/switch', { mode: 'cinematic' });
  });

  it('keeps default mode separate when runtime status is missing current_mode', async () => {
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
          is_loaded: false,
          backend_version: 'abc1234',
        };
      }

      throw new Error(`Unexpected endpoint: ${endpoint}`);
    });

    const { result } = renderHook(() => useModeConfig());

    await waitFor(() => expect(result.current.config).not.toBeNull());

    expect(result.current.defaultModeName).toBe('cinematic');
    expect(result.current.activeModeName).toBeNull();
    expect(result.current.activeMode).toBeNull();
  });

  it('keeps default mode separate when runtime status fetch fails', async () => {
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
        throw new Error('status unavailable');
      }

      throw new Error(`Unexpected endpoint: ${endpoint}`);
    });

    const { result } = renderHook(() => useModeConfig());

    await waitFor(() => expect(result.current.config).not.toBeNull());

    expect(result.current.defaultModeName).toBe('cinematic');
    expect(result.current.activeModeName).toBeNull();
    expect(result.current.activeMode).toBeNull();
  });

  it('polls runtime status and updates the active mode after time advances', async () => {
    vi.useFakeTimers();
    const statuses = [
      {
        current_mode: 'cinematic',
        is_loaded: true,
        backend_version: 'abc1234',
      },
      {
        current_mode: 'portrait',
        is_loaded: false,
        backend_version: 'abc1234',
      },
    ];

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
        return statuses[Math.min(statuses.length - 1, api.client.fetchGet.mock.calls.filter((call) => call[0] === '/api/models/status').length - 1)];
      }

      throw new Error(`Unexpected endpoint: ${endpoint}`);
    });

    const { result } = renderHook(() => useModeConfig());

    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    expect(result.current.activeModeName).toBe('portrait');
    expect(result.current.defaultModeName).toBe('cinematic');
    expect(result.current.activeMode).toEqual({ model: 'base-portrait' });
  });

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
          backend_version: 'abc1234',
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

  it('loads backend_version into runtimeStatus from /api/models/status', async () => {
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
          backend_version: 'abc1234',
        };
      }

      throw new Error(`Unexpected endpoint: ${endpoint}`);
    });

    const { result } = renderHook(() => useModeConfig());

    await waitFor(() => expect(result.current.runtimeStatus?.backend_version).toBe('abc1234'));

    expect(result.current.runtimeStatus).toMatchObject({
      current_mode: 'portrait',
      is_loaded: true,
      backend_version: 'abc1234',
    });
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
          backend_version: 'abc1234',
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
