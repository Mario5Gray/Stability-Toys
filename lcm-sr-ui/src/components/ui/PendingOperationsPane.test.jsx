// @vitest-environment jsdom
import React, { useEffect } from 'react';
import { render, screen, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { OperationsProvider, useOperationsController } from '../../contexts/OperationsContext';
import { PendingOperationsPane } from './PendingOperationsPane';

afterEach(() => cleanup());

function ControlPanel({ onReady }) {
  const ctrl = useOperationsController();
  useEffect(() => { onReady(ctrl); }, []); // eslint-disable-line
  return null;
}

function Harness({ onReady }) {
  return (
    <OperationsProvider>
      <ControlPanel onReady={onReady} />
      <PendingOperationsPane />
    </OperationsProvider>
  );
}

describe('PendingOperationsPane', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('renders nothing when no operations are active', () => {
    const { container } = render(<OperationsProvider><PendingOperationsPane /></OperationsProvider>);
    expect(container.firstChild).toBeNull();
  });

  it('renders an active operation with text and detail', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => { ctrl.start({ key: 'adv:1', text: 'Rebuilding', detail: 'Analyzing evidence' }); });
    expect(screen.getByText('Rebuilding')).toBeInTheDocument();
    expect(screen.getByText('Analyzing evidence')).toBeInTheDocument();
  });

  it('renders progress value when set', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => {
      const handle = ctrl.start({ key: 'gen:1', text: 'Generating image' });
      handle.setProgress({ current: 8, total: 28 });
    });
    expect(screen.getByText('8 / 28')).toBeInTheDocument();
  });

  it('shows cancel button when cancelFn is provided', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => {
      ctrl.start({ key: 'gen:1', text: 'Generating', cancellable: true, cancelFn: vi.fn() });
    });
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
  });

  it('does not show cancel button when not cancellable', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => { ctrl.start({ key: 'adv:1', text: 'Working', cancellable: false }); });
    expect(screen.queryByRole('button', { name: /cancel/i })).not.toBeInTheDocument();
  });

  it('auto-removes after complete linger', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    let handle;
    act(() => { handle = ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.complete({ text: 'Done' }); });
    expect(screen.getByText('Done')).toBeInTheDocument();
    act(() => { vi.advanceTimersByTime(2001); });
    expect(screen.queryByText('Done')).not.toBeInTheDocument();
  });
});
