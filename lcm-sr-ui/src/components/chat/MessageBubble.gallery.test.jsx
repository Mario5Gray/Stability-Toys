// @vitest-environment jsdom
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MessageBubble } from './MessageBubble';
import { MESSAGE_ROLES, MESSAGE_KINDS } from '../../utils/constants';

afterEach(() => cleanup());

function makeImageMsg(overrides = {}) {
  return {
    id: 'msg_1',
    role: MESSAGE_ROLES.ASSISTANT,
    kind: MESSAGE_KINDS.IMAGE,
    imageUrl: 'blob:http://localhost/fake',
    serverImageUrl: 'http://example.com/img.png',
    params: { prompt: 'cat', seed: 1 },
    meta: { cacheKey: 'abc123', backend: 'cuda' },
    ...overrides,
  };
}

describe('MessageBubble — Gallery pill', () => {
  it('renders the Gallery pill when onAddToGallery and cacheKey are present', () => {
    render(
      <MessageBubble
        msg={makeImageMsg()}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        activeGalleryId="gal_1"
        onAddToGallery={vi.fn()}
      />
    );
    expect(screen.getByTitle('Add to gallery')).toBeInTheDocument();
  });

  it('does not render Gallery pill when onAddToGallery is absent', () => {
    render(
      <MessageBubble
        msg={makeImageMsg()}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        activeGalleryId="gal_1"
      />
    );
    expect(screen.queryByTitle(/gallery/i)).not.toBeInTheDocument();
  });

  it('does not render Gallery pill when cacheKey is absent', () => {
    render(
      <MessageBubble
        msg={makeImageMsg({ meta: { backend: 'cuda' } })} // no cacheKey
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        activeGalleryId="gal_1"
        onAddToGallery={vi.fn()}
      />
    );
    expect(screen.queryByTitle(/gallery/i)).not.toBeInTheDocument();
  });

  it('is disabled and dimmed when activeGalleryId is null', () => {
    render(
      <MessageBubble
        msg={makeImageMsg()}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        activeGalleryId={null}
        onAddToGallery={vi.fn()}
      />
    );
    const btn = screen.getByTitle('Select a gallery first');
    expect(btn).toBeDisabled();
    expect(btn.className).toContain('opacity-40');
  });

  it('calls onAddToGallery with cacheKey and image info on click', () => {
    const onAdd = vi.fn();
    render(
      <MessageBubble
        msg={makeImageMsg()}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={vi.fn()}
        activeGalleryId="gal_1"
        onAddToGallery={onAdd}
      />
    );
    fireEvent.click(screen.getByTitle('Add to gallery'));
    expect(onAdd).toHaveBeenCalledWith('abc123', {
      serverImageUrl: 'http://example.com/img.png',
      params: { prompt: 'cat', seed: 1 },
    });
  });
});
