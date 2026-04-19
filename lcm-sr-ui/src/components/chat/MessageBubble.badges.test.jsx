// @vitest-environment jsdom
import React from 'react';
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { MessageBubble } from './MessageBubble';

afterEach(() => cleanup());

const baseMsg = {
  id: 'msg_1',
  role: 'assistant',
  kind: 'image',
  imageUrl: 'data:image/png;base64,AA==',
  params: { prompt: 'cat', size: '512x512', steps: 20, cfg: 7, seed: 1, superresLevel: 0, seedMode: 'fixed' },
  meta: {},
  ts: Date.now(),
};

describe('MessageBubble — no animated badges', () => {
  it('does not render "dreaming" badge when isDreamMessage is true', () => {
    render(
      <MessageBubble
        msg={baseMsg}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        onCancel={null}
        isDreamMessage={true}
        hasDreamHistory={false}
        onDreamSave={vi.fn()}
        onDreamHistoryPrev={vi.fn()}
        onDreamHistoryNext={vi.fn()}
        onDreamHistoryLive={vi.fn()}
        onRetry={vi.fn()}
        activeGalleryId={null}
        onAddToGallery={vi.fn()}
      />,
    );
    expect(screen.queryByText('dreaming')).not.toBeInTheDocument();
  });

  it('does not render "generating" badge when isRegenerating is true', () => {
    render(
      <MessageBubble
        msg={{ ...baseMsg, isRegenerating: true }}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        onCancel={null}
        isDreamMessage={false}
        hasDreamHistory={false}
        onDreamSave={vi.fn()}
        onDreamHistoryPrev={vi.fn()}
        onDreamHistoryNext={vi.fn()}
        onDreamHistoryLive={vi.fn()}
        onRetry={vi.fn()}
        activeGalleryId={null}
        onAddToGallery={vi.fn()}
      />,
    );
    expect(screen.queryByText('generating')).not.toBeInTheDocument();
  });

  it('contains no animate-pulse elements on image bubbles', () => {
    const { container } = render(
      <MessageBubble
        msg={baseMsg}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        onCancel={null}
        isDreamMessage={true}
        hasDreamHistory={false}
        onDreamSave={vi.fn()}
        onDreamHistoryPrev={vi.fn()}
        onDreamHistoryNext={vi.fn()}
        onDreamHistoryLive={vi.fn()}
        onRetry={vi.fn()}
        activeGalleryId={null}
        onAddToGallery={vi.fn()}
      />,
    );
    expect(container.querySelector('.animate-pulse')).toBeNull();
  });
});
