// @vitest-environment jsdom
import React from 'react';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { OperationsProvider, useOperationsStore, useOperationsController } from './OperationsContext';

afterEach(() => cleanup());

function wrapper({ children }) {
  return <OperationsProvider>{children}</OperationsProvider>;
}

function useStoreAndCtrl() {
  return { store: useOperationsStore(), ctrl: useOperationsController() };
}

describe('OperationsContext', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('start() creates an operation record', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    act(() => {
      result.current.ctrl.start({ key: 'adv:1', kind: 'advisor', text: 'Working' });
    });
    const op = result.current.store.operations.get('adv:1');
    expect(op).toBeDefined();
    expect(op.text).toBe('Working');
    expect(op.tone).toBe('active');
  });

  it('start() with same key upserts instead of creating a duplicate', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    act(() => {
      result.current.ctrl.start({ key: 'adv:1', text: 'First' });
      result.current.ctrl.start({ key: 'adv:1', text: 'Second' });
    });
    expect(result.current.store.order).toHaveLength(1);
    expect(result.current.store.operations.get('adv:1').text).toBe('Second');
  });

  it('handle.setText() updates text without changing tone', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.setText('Analyzing'); });
    expect(result.current.store.operations.get('adv:1').text).toBe('Analyzing');
    expect(result.current.store.operations.get('adv:1').tone).toBe('active');
  });

  it('handle.complete() sets tone to complete and auto-removes after 2s', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.complete({ text: 'Done' }); });
    expect(result.current.store.operations.get('adv:1').tone).toBe('complete');
    act(() => { vi.advanceTimersByTime(2001); });
    expect(result.current.store.operations.get('adv:1')).toBeUndefined();
  });

  it('handle.error() sets tone to error, still present at 2s, gone after 5s', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.error({ text: 'Failed' }); });
    expect(result.current.store.operations.get('adv:1').tone).toBe('error');
    act(() => { vi.advanceTimersByTime(2001); });
    expect(result.current.store.operations.get('adv:1')).toBeDefined();
    act(() => { vi.advanceTimersByTime(3001); });
    expect(result.current.store.operations.get('adv:1')).toBeUndefined();
  });

  it('handle.cancel() only exists when cancelFn is provided', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    const cancelFn = vi.fn();
    let handleA, handleB;
    act(() => {
      handleA = result.current.ctrl.start({ key: 'a', cancellable: false });
      handleB = result.current.ctrl.start({ key: 'b', cancellable: true, cancelFn });
    });
    expect(handleA.cancel).toBeUndefined();
    expect(handleB.cancel).toBeDefined();
    act(() => { handleB.cancel(); });
    expect(cancelFn).toHaveBeenCalledOnce();
    expect(result.current.store.operations.get('b')).toBeUndefined();
  });

  it('handle.remove() removes operation immediately', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.remove(); });
    expect(result.current.store.operations.get('adv:1')).toBeUndefined();
  });
});
