// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ModeEditor from './ModeEditor';

const api = vi.hoisted(() => ({
  client: {
    fetchGet: vi.fn(),
    fetchPost: vi.fn(),
    fetchPut: vi.fn(),
    fetchDelete: vi.fn(),
  },
}));

vi.mock('../../utils/api', () => ({
  createApiClient: vi.fn(() => api.client),
  createApiConfig: vi.fn(() => ({})),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderModeEditor(modeState) {
  api.client.fetchGet.mockImplementation(async (endpoint) => {
    if (endpoint === '/api/modes') {
      return {
        default_mode: 'cinematic',
        resolution_sets: {
          square: [{ size: '1024x1024', aspect_ratio: '1:1' }],
          portrait: [{ size: '832x1216', aspect_ratio: '13:19' }],
        },
        modes: {
          cinematic: {
            model: 'model-a',
            default_size: '512x512',
            default_steps: 8,
            default_guidance: 1.5,
            resolution_set: 'square',
            scheduler_policy: 'karras',
            prompt_policy_negative: 'required',
          },
          portrait: {
            model: 'model-b',
            default_size: '768x768',
            default_steps: 10,
            default_guidance: 2.0,
            resolution_set: 'portrait',
            scheduler_policy: 'ancestral',
            prompt_policy_negative: 'optional',
          },
        },
      };
    }

    if (endpoint === '/api/inventory/models') {
      return {
        model_root: '/models',
        models: ['model-a', 'model-b'],
      };
    }

    if (endpoint === '/api/inventory/loras') {
      return {
        lora_root: '/loras',
        loras: ['lora-a'],
      };
    }

    throw new Error(`Unexpected endpoint: ${endpoint}`);
  });

  return render(<ModeEditor modeState={modeState} />);
}

describe('ModeEditor runtime controls', () => {
  it('refreshes shared mode state after changing the default mode', async () => {
    const modeState = {
      config: null,
      defaultModeName: 'cinematic',
      activeModeName: 'cinematic',
      activeMode: {
        model: 'model-a',
      },
      isLoaded: true,
      error: null,
      loadModes: vi.fn().mockResolvedValue(undefined),
      refreshStatus: vi.fn().mockResolvedValue(undefined),
      reloadActiveModel: vi.fn().mockResolvedValue(undefined),
      freeVram: vi.fn().mockResolvedValue(undefined),
      switchMode: vi.fn(),
    };

    renderModeEditor(modeState);

    fireEvent.click(await screen.findByRole('button', { name: /Set as default/i }));

    await waitFor(() => expect(api.client.fetchPut).toHaveBeenCalled());
    expect(api.client.fetchPut).toHaveBeenCalledWith('/api/modes', {
      model_root: '/models',
      lora_root: '/loras',
      default_mode: 'portrait',
      resolution_sets: {
        square: [{ size: '1024x1024', aspect_ratio: '1:1' }],
        portrait: [{ size: '832x1216', aspect_ratio: '13:19' }],
      },
      modes: {
        cinematic: {
          model: 'model-a',
          default_size: '512x512',
          default_steps: 8,
          default_guidance: 1.5,
          resolution_set: 'square',
          scheduler_policy: 'karras',
          prompt_policy_negative: 'required',
        },
        portrait: {
          model: 'model-b',
          default_size: '768x768',
          default_steps: 10,
          default_guidance: 2.0,
          resolution_set: 'portrait',
          scheduler_policy: 'ancestral',
          prompt_policy_negative: 'optional',
        },
      },
    });
    await waitFor(() => expect(modeState.loadModes).toHaveBeenCalledTimes(1));
  });

  it('preserves top-level resolution_sets and existing mode metadata when saving an edited mode', async () => {
    const modeState = {
      config: null,
      defaultModeName: 'cinematic',
      activeModeName: 'cinematic',
      activeMode: {
        model: 'model-a',
      },
      isLoaded: true,
      error: null,
      loadModes: vi.fn().mockResolvedValue(undefined),
      refreshStatus: vi.fn().mockResolvedValue(undefined),
      reloadActiveModel: vi.fn().mockResolvedValue(undefined),
      freeVram: vi.fn().mockResolvedValue(undefined),
      switchMode: vi.fn(),
    };

    renderModeEditor(modeState);

    fireEvent.click((await screen.findAllByRole('button', { name: /Edit/i }))[0]);
    fireEvent.click(await screen.findByRole('button', { name: /^Save$/i }));

    await waitFor(() => expect(api.client.fetchPut).toHaveBeenCalled());
    expect(api.client.fetchPut).toHaveBeenCalledWith('/api/modes', {
      model_root: '/models',
      lora_root: '/loras',
      default_mode: 'cinematic',
      resolution_sets: {
        square: [{ size: '1024x1024', aspect_ratio: '1:1' }],
        portrait: [{ size: '832x1216', aspect_ratio: '13:19' }],
      },
      modes: {
        cinematic: {
          model: 'model-a',
          loras: [],
          default_size: '512x512',
          default_steps: 8,
          default_guidance: 1.5,
          resolution_set: 'square',
          scheduler_policy: 'karras',
          prompt_policy_negative: 'required',
        },
        portrait: {
          model: 'model-b',
          default_size: '768x768',
          default_steps: 10,
          default_guidance: 2.0,
          resolution_set: 'portrait',
          scheduler_policy: 'ancestral',
          prompt_policy_negative: 'optional',
        },
      },
    });
    await waitFor(() => expect(modeState.loadModes).toHaveBeenCalledTimes(1));
  });

  it('shows runtime status and refreshes it after reloading the active model', async () => {
    const modeState = {
      config: null,
      defaultModeName: 'cinematic',
      activeModeName: 'portrait',
      activeMode: {
        model: 'model-b',
      },
      isLoaded: true,
      error: null,
      loadModes: vi.fn(),
      refreshStatus: vi.fn().mockResolvedValue(undefined),
      reloadActiveModel: vi.fn().mockResolvedValue(undefined),
      freeVram: vi.fn().mockResolvedValue(undefined),
      switchMode: vi.fn(),
    };

    renderModeEditor(modeState);

    await waitFor(() => {
      expect(screen.getByTestId('runtime-active-mode')).toHaveTextContent('Active mode: portrait');
    });
    expect(screen.getByTestId('runtime-loaded-state')).toHaveTextContent('Loaded: yes');

    fireEvent.click(await screen.findByRole('button', { name: /Reload Active Model/i }));

    await waitFor(() => expect(modeState.reloadActiveModel).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(modeState.refreshStatus).toHaveBeenCalledTimes(1));
  });

  it('surfaces free VRAM failures inline', async () => {
    const modeState = {
      config: null,
      defaultModeName: 'cinematic',
      activeModeName: 'cinematic',
      activeMode: {
        model: 'model-a',
      },
      isLoaded: false,
      error: null,
      loadModes: vi.fn(),
      refreshStatus: vi.fn().mockResolvedValue(undefined),
      reloadActiveModel: vi.fn().mockResolvedValue(undefined),
      freeVram: vi.fn().mockRejectedValue(new Error('VRAM cleanup failed')),
      switchMode: vi.fn(),
    };

    renderModeEditor(modeState);

    fireEvent.click(await screen.findByRole('button', { name: /Free VRAM/i }));

    await waitFor(() => expect(modeState.freeVram).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/VRAM cleanup failed/i)).toBeTruthy();
  });
});
