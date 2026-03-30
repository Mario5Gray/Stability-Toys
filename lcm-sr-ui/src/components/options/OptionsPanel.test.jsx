// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OptionsPanel } from './OptionsPanel';

if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
}

if (!Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = () => {};
}

if (!Element.prototype.releasePointerCapture) {
  Element.prototype.releasePointerCapture = () => {};
}

afterEach(() => {
  cleanup();
});

function makeParams(overrides = {}) {
  return {
    draft: {
      prompt: 'draft prompt',
      negativePrompt: '',
      seedMode: 'random',
      seed: '',
      ...overrides.draft,
    },
    effective: {
      prompt: 'draft prompt',
      size: '512x512',
      steps: 8,
      cfg: 2.8,
      superresLevel: 1,
      negativePrompt: '',
      schedulerId: null,
      seedMode: 'random',
      seed: null,
      denoiseStrength: 0.75,
      ...overrides.effective,
    },
    setPrompt: vi.fn(),
    setSize: vi.fn(),
    setSteps: vi.fn(),
    setCfg: vi.fn(),
    setSrLevel: vi.fn(),
    setNegativePrompt: vi.fn(),
    setSchedulerId: vi.fn(),
    setSeedMode: vi.fn(),
    setSeed: vi.fn(),
    randomizeSeed: vi.fn(),
    ...overrides,
  };
}

function makeModeState(activeModeName, activeMode) {
  return {
    config: {
      default_mode: activeModeName,
      modes: {
        [activeModeName]: activeMode,
      },
    },
    activeModeName,
    activeMode,
    isSwitching: false,
    error: null,
    switchMode: vi.fn(),
  };
}

function renderOptionsPanel(modeState, params = makeParams()) {
  return render(
    <OptionsPanel
      params={params}
      inputImage={null}
      comfyInputImage={null}
      selectedParams={null}
      blurredSelectedParams={null}
      selectedMsgId={null}
      onClearSelection={vi.fn()}
      onApplyPromptDelta={vi.fn()}
      onApplySeedDelta={vi.fn()}
      onRerunSelected={vi.fn()}
      onPersistSelectedParams={vi.fn()}
      dreamState={{
        isDreaming: false,
        temperature: 0.5,
        interval: 10,
        onStart: vi.fn(),
        onStop: vi.fn(),
        onGuide: vi.fn(),
        onTemperatureChange: vi.fn(),
        onIntervalChange: vi.fn(),
      }}
      onSuperResUpload={vi.fn()}
      uploadFile={null}
      onUploadFileChange={vi.fn()}
      srMagnitude={1}
      onSrMagnitudeChange={vi.fn()}
      onSuperResSelected={vi.fn()}
      serverLabel="test"
      onRunComfy={vi.fn()}
      onClearCache={vi.fn()}
      getCacheStats={vi.fn().mockResolvedValue(null)}
      onClearHistory={vi.fn()}
      queueState={{ items: [] }}
      initImage={null}
      onClearInitImage={vi.fn()}
      denoiseStrength={0.75}
      onDenoiseStrengthChange={vi.fn()}
      modeState={modeState}
    />
  );
}

describe('OptionsPanel mode-driven controls', () => {
  it('forwards negative prompt edits through the selected-image params path', () => {
    const params = makeParams({
      effective: {
        negativePrompt: 'blurry',
        schedulerId: 'ddim',
      },
    });
    const modeState = makeModeState('cinematic', {
      negative_prompt_templates: {
        cinematic: 'blurry',
      },
      allow_custom_negative_prompt: true,
      allowed_scheduler_ids: ['ddim'],
      default_scheduler_id: 'ddim',
    });

    render(
      <OptionsPanel
        params={params}
        inputImage={null}
        comfyInputImage={null}
        selectedParams={{
          prompt: 'portrait',
          negativePrompt: 'blurry',
          schedulerId: 'ddim',
          size: '512x512',
          steps: 8,
          cfg: 2.8,
          superresLevel: 1,
        }}
        blurredSelectedParams={null}
        selectedMsgId="msg-1"
        onClearSelection={vi.fn()}
        onApplyPromptDelta={vi.fn()}
        onApplySeedDelta={vi.fn()}
        onRerunSelected={vi.fn()}
        onPersistSelectedParams={vi.fn()}
        dreamState={{
          isDreaming: false,
          temperature: 0.5,
          interval: 10,
          onStart: vi.fn(),
          onStop: vi.fn(),
          onGuide: vi.fn(),
          onTemperatureChange: vi.fn(),
          onIntervalChange: vi.fn(),
        }}
        onSuperResUpload={vi.fn()}
        uploadFile={null}
        onUploadFileChange={vi.fn()}
        srMagnitude={1}
        onSrMagnitudeChange={vi.fn()}
        onSuperResSelected={vi.fn()}
        serverLabel="test"
        onRunComfy={vi.fn()}
        onClearCache={vi.fn()}
        getCacheStats={vi.fn().mockResolvedValue(null)}
        onClearHistory={vi.fn()}
        queueState={{ items: [] }}
        initImage={null}
        onClearInitImage={vi.fn()}
        denoiseStrength={0.75}
        onDenoiseStrengthChange={vi.fn()}
        modeState={modeState}
      />
    );

    fireEvent.change(screen.getByLabelText('Negative prompt'), {
      target: { value: 'washed out' },
    });

    expect(params.setNegativePrompt).toHaveBeenCalledWith('washed out');
  });

  it('refreshes negative prompt and sampler controls when the active mode changes', async () => {
    const cinematicMode = {
      negative_prompt_templates: {
        cinematic: 'blurry, low quality',
      },
      allow_custom_negative_prompt: true,
      allowed_scheduler_ids: ['lcm', 'ddim'],
      default_scheduler_id: 'lcm',
    };

    const portraitMode = {
      negative_prompt_templates: {
        portrait: 'deformed, bad anatomy',
      },
      allow_custom_negative_prompt: false,
      allowed_scheduler_ids: ['euler'],
      default_scheduler_id: 'euler',
    };

    const view = renderOptionsPanel(
      makeModeState('cinematic', cinematicMode),
      makeParams({
        effective: {
          negativePrompt: cinematicMode.negative_prompt_templates.cinematic,
          schedulerId: 'lcm',
        },
      })
    );

    expect(screen.getByLabelText('Negative prompt')).toBeTruthy();
    expect(screen.getByLabelText('Negative prompt template')).toBeTruthy();
    expect(screen.getByLabelText('Sampler')).toBeTruthy();

    fireEvent.pointerDown(screen.getByLabelText('Negative prompt template'));
    expect((await screen.findAllByText('cinematic')).length).toBeGreaterThan(0);

    view.rerender(
      <OptionsPanel
        params={makeParams({
          effective: {
            negativePrompt: portraitMode.negative_prompt_templates.portrait,
            schedulerId: 'euler',
          },
        })}
        inputImage={null}
        comfyInputImage={null}
        selectedParams={null}
        blurredSelectedParams={null}
        selectedMsgId={null}
        onClearSelection={vi.fn()}
        onApplyPromptDelta={vi.fn()}
        onApplySeedDelta={vi.fn()}
        onRerunSelected={vi.fn()}
        onPersistSelectedParams={vi.fn()}
        dreamState={{
          isDreaming: false,
          temperature: 0.5,
          interval: 10,
          onStart: vi.fn(),
          onStop: vi.fn(),
          onGuide: vi.fn(),
          onTemperatureChange: vi.fn(),
          onIntervalChange: vi.fn(),
        }}
        onSuperResUpload={vi.fn()}
        uploadFile={null}
        onUploadFileChange={vi.fn()}
        srMagnitude={1}
        onSrMagnitudeChange={vi.fn()}
        onSuperResSelected={vi.fn()}
        serverLabel="test"
        onRunComfy={vi.fn()}
        onClearCache={vi.fn()}
        getCacheStats={vi.fn().mockResolvedValue(null)}
        onClearHistory={vi.fn()}
        queueState={{ items: [] }}
        initImage={null}
        onClearInitImage={vi.fn()}
        denoiseStrength={0.75}
        onDenoiseStrengthChange={vi.fn()}
        modeState={makeModeState('portrait', portraitMode)}
      />
    );

    await waitFor(() => {
      expect(screen.queryByLabelText('Negative prompt')).toBeNull();
    });

    fireEvent.pointerDown(screen.getByLabelText('Negative prompt template'));
    expect((await screen.findAllByText('portrait')).length).toBeGreaterThan(0);
    expect(screen.queryAllByText('cinematic')).toHaveLength(0);

    fireEvent.pointerDown(screen.getByLabelText('Sampler'));
    expect((await screen.findAllByText('euler')).length).toBeGreaterThan(0);
    expect(screen.queryAllByText('ddim')).toHaveLength(0);
    expect(screen.queryAllByText('lcm')).toHaveLength(0);
  });
});
