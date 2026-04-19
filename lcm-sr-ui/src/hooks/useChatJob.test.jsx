// @vitest-environment jsdom

import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useChatJob } from './useChatJob';

const wsMock = vi.hoisted(() => {
  const stateListeners = new Set();
  const typedListeners = new Map();
  let connected = true;

  const getListeners = (type) => {
    if (!typedListeners.has(type)) typedListeners.set(type, new Set());
    return typedListeners.get(type);
  };

  const client = {
    get connected() {
      return connected;
    },
    send: vi.fn(),
    on: vi.fn((type, callback) => {
      const listeners = getListeners(type);
      listeners.add(callback);
      return () => listeners.delete(callback);
    }),
    addEventListener: vi.fn((type, callback) => {
      if (type === 'statechange') stateListeners.add(callback);
    }),
    removeEventListener: vi.fn((type, callback) => {
      if (type === 'statechange') stateListeners.delete(callback);
    }),
  };

  return {
    client,
    nextCorrId: vi.fn(),
    reset() {
      connected = true;
      stateListeners.clear();
      typedListeners.clear();
      client.send.mockReset();
      client.on.mockClear();
      client.addEventListener.mockClear();
      client.removeEventListener.mockClear();
      this.nextCorrId.mockReset();
    },
    emit(type, message) {
      for (const callback of getListeners(type)) callback(message);
    },
    emitState(detail) {
      for (const callback of stateListeners) callback({ detail });
    },
    setConnected(value) {
      connected = value;
    },
  };
});

vi.mock('../lib/wsClient', () => ({
  wsClient: wsMock.client,
  nextCorrId: wsMock.nextCorrId,
}));

afterEach(() => {
  wsMock.reset();
});

describe('useChatJob', () => {
  it('sends cancel after ack when the user cancels before ack arrives', () => {
    wsMock.nextCorrId.mockReturnValue('c-chat-1');

    const onAck = vi.fn();
    const onError = vi.fn();

    const { result } = renderHook(() => useChatJob());

    let handle;
    act(() => {
      handle = result.current.start({
        prompt: 'hello',
        onAck,
        onDelta: vi.fn(),
        onComplete: vi.fn(),
        onError,
      });
    });

    expect(wsMock.client.send).toHaveBeenCalledWith({
      type: 'job:submit',
      id: 'c-chat-1',
      jobType: 'chat',
      params: { prompt: 'hello', stream: true },
    });

    act(() => {
      handle.cancel();
    });

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith('Cancelled');

    act(() => {
      wsMock.emit('job:ack', { id: 'c-chat-1', jobId: 'job-123' });
    });

    expect(onAck).not.toHaveBeenCalled();
    expect(wsMock.client.send).toHaveBeenLastCalledWith({
      type: 'job:cancel',
      jobId: 'job-123',
    });

    act(() => {
      wsMock.emit('job:error', { jobId: 'job-123', error: 'Cancelled by client' });
    });

    expect(onError).toHaveBeenCalledTimes(1);
  });
});
