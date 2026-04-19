// @vitest-environment jsdom
import React from 'react';
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { OperationsProvider } from '../../contexts/OperationsContext';
import { AdvisorPanel } from './AdvisorPanel';

afterEach(() => cleanup());

function wrap(ui) {
  return render(ui, {
    wrapper: ({ children }) => <OperationsProvider>{children}</OperationsProvider>,
  });
}

const base = {
  state: { advice_text: 'Some advice', status: 'fresh', updated_at: Date.now() },
  maximumLen: 240,
  onAutoAdviceChange: vi.fn(),
  onTemperatureChange: vi.fn(),
  onLengthChange: vi.fn(),
  onAdviceChange: vi.fn(),
  onResetToDigest: vi.fn(),
  onRebuild: vi.fn(),
  onApply: vi.fn(),
  applyMode: 'append',
  onApplyModeChange: vi.fn(),
};

describe('AdvisorPanel', () => {
  it('renders Apply as a <button>', () => {
    wrap(<AdvisorPanel {...base} />);
    expect(screen.getByRole('button', { name: /apply/i })).toBeInTheDocument();
  });

  it('renders Rebuild and Reset as <button> elements', () => {
    wrap(<AdvisorPanel {...base} />);
    expect(screen.getByRole('button', { name: /rebuild/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset/i })).toBeInTheDocument();
  });

  it('Apply button is disabled when advice_text is empty', () => {
    wrap(<AdvisorPanel {...base} state={{ ...base.state, advice_text: '' }} />);
    expect(screen.getByRole('button', { name: /apply/i })).toBeDisabled();
  });

  it('does not render "Building digest..." text — no animated inline status', () => {
    wrap(<AdvisorPanel {...base} state={{ ...base.state, status: 'building' }} />);
    expect(screen.queryByText('Building digest...')).not.toBeInTheDocument();
  });

  it('shows last-updated timestamp when status is fresh', () => {
    wrap(<AdvisorPanel {...base} />);
    expect(screen.getByText(/updated/i)).toBeInTheDocument();
  });
});
