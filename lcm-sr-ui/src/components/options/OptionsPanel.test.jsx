// @vitest-environment jsdom

import React from 'react';
import { IDBFactory } from 'fake-indexeddb';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
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

if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

beforeEach(() => {
  globalThis.indexedDB = new IDBFactory();
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

function makeGalleryState(overrides = {}) {
  return {
    galleries: [{ id: 'gal_1', name: 'Advisor' }],
    activeGalleryId: 'gal_1',
    setActiveGalleryId: vi.fn(),
    getGalleryImages: vi.fn().mockResolvedValue([]),
    getGalleryRevision: vi.fn(() => 0),
    ...overrides,
  };
}

function renderOptionsPanel(modeState, params = makeParams(), extraProps = {}) {
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
      {...extraProps}
    />
  );
}

function openSizeSelect() {
  const sizeSection = screen.getByText('Size').closest('div');
  const trigger = sizeSection?.querySelector('[role="combobox"]');
  if (!trigger) {
    throw new Error('Size select trigger not found');
  }
  fireEvent.click(trigger);
  fireEvent.pointerDown(trigger);
}

describe('OptionsPanel mode-driven controls', () => {
  it('renders size labels with aspect ratio from the active mode', async () => {
    renderOptionsPanel(
      makeModeState('SDXL', {
        default_size: '1024x1024',
        resolution_options: [
          { size: '1024x1024', aspect_ratio: '1:1' },
          { size: '896x1152', aspect_ratio: '7:9' },
        ],
      })
    );

    openSizeSelect();

    expect(await screen.findByText('1024×1024 • 1:1')).toBeTruthy();
    expect(screen.getByText('896×1152 • 7:9')).toBeTruthy();
  });

  it('constrains the size dropdown viewport to five visible rows', async () => {
    renderOptionsPanel(
      makeModeState('SDXL', {
        default_size: '1024x1024',
        resolution_options: [
          { size: '1024x1024', aspect_ratio: '1:1' },
          { size: '896x1152', aspect_ratio: '7:9' },
          { size: '1152x896', aspect_ratio: '9:7' },
          { size: '1216x832', aspect_ratio: '19:13' },
          { size: '832x1216', aspect_ratio: '13:19' },
          { size: '1344x768', aspect_ratio: '7:4' },
        ],
      })
    );

    openSizeSelect();

    expect(await screen.findByText('1344×768 • 7:4')).toBeTruthy();
    expect(document.querySelector('.max-h-60.overflow-y-auto')).toBeTruthy();
  });

  it('falls back to loaded default mode size options when active mode is unavailable', async () => {
    renderOptionsPanel(
      {
        config: {
          default_mode: 'SDXL',
          modes: {
            SDXL: {
              default_size: '1024x1024',
              resolution_options: [
                { size: '1024x1024', aspect_ratio: '1:1' },
                { size: '896x1152', aspect_ratio: '7:9' },
              ],
            },
          },
        },
        activeModeName: null,
        activeMode: null,
        isSwitching: false,
        error: null,
        switchMode: vi.fn(),
      }
    );

    openSizeSelect();

    expect(await screen.findByText('1024×1024 • 1:1')).toBeTruthy();
    expect(screen.getByText('896×1152 • 7:9')).toBeTruthy();
  });

  it('uses logarithmic seed modifier steps when log mode is selected', () => {
    const onApplySeedDelta = vi.fn();

    render(
      <OptionsPanel
        params={makeParams()}
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
          seed: 12345678,
        }}
        blurredSelectedParams={null}
        selectedMsgId="msg-1"
        onClearSelection={vi.fn()}
        onApplyPromptDelta={vi.fn()}
        onApplySeedDelta={onApplySeedDelta}
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
        modeState={makeModeState('cinematic', {})}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: 'Log' }));
    fireEvent.click(screen.getByRole('button', { name: '+1M' }));

    expect(onApplySeedDelta).toHaveBeenCalledWith(1000000);
  });

  it('renders restored init image controls when an init image is provided', () => {
    render(
      <OptionsPanel
        params={makeParams()}
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
        initImage={{
          sourceId: 'src-1',
          file: new File(['x'], 'restored.png', { type: 'image/png' }),
          objectUrl: 'blob:restored',
          filename: 'restored.png',
        }}
        onClearInitImage={vi.fn()}
        denoiseStrength={0.5}
        onDenoiseStrengthChange={vi.fn()}
        modeState={makeModeState('cinematic', {})}
      />
    );

    expect(screen.getByText('Init Image')).toBeTruthy();
    expect(screen.getByText('restored.png')).toBeTruthy();
  });

  it('forwards denoise slider changes immediately when init image is present', () => {
    vi.useFakeTimers();
    const onDenoiseStrengthChange = vi.fn();

    render(
      <OptionsPanel
        params={makeParams()}
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
        initImage={{
          file: new File(['x'], 'init.png', { type: 'image/png' }),
          objectUrl: 'blob:init-image',
        }}
        onClearInitImage={vi.fn()}
        denoiseStrength={0.75}
        onDenoiseStrengthChange={onDenoiseStrengthChange}
        modeState={makeModeState('cinematic', {})}
      />
    );

    fireEvent.change(screen.getByRole('slider'), { target: { value: '42' } });

    expect(onDenoiseStrengthChange).toHaveBeenCalledWith(0.42);
  });

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

  it('renders the advisor section under negative prompt controls when a gallery is active', async () => {
    const params = makeParams();

    renderOptionsPanel(
      makeModeState('SDXL', {
        maximum_len: 240,
        allow_custom_negative_prompt: true,
      }),
      params,
      { galleryState: makeGalleryState() },
    );

    expect(await screen.findByLabelText('Advisor length')).toHaveAttribute('max', '240');
  });

  it('offers append and replace apply modes', async () => {
    renderOptionsPanel(
      makeModeState('SDXL', {
        maximum_len: 240,
        allow_custom_negative_prompt: true,
      }),
      makeParams(),
      { galleryState: makeGalleryState() },
    );

    expect(await screen.findByRole('button', { name: 'Apply Advice' })).toBeTruthy();

    fireEvent.pointerDown(screen.getByLabelText('Apply advice mode'));
    expect((await screen.findAllByText('Append')).length).toBeGreaterThan(0);
    expect((await screen.findAllByText('Replace')).length).toBeGreaterThan(0);
  });

  it('hides advisor length control when active mode has no maximum_len', async () => {
    renderOptionsPanel(
      makeModeState('SDXL', {
        allow_custom_negative_prompt: true,
      }),
      makeParams(),
      { galleryState: makeGalleryState() },
    );

    expect(await screen.findByRole('button', { name: 'Apply Advice' })).toBeTruthy();
    expect(screen.queryByLabelText('Advisor length')).toBeNull();
  });
});
