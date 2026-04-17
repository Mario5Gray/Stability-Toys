// @vitest-environment jsdom
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GalleryZoomOverlay } from './GalleryZoomOverlay';

afterEach(cleanup);

function item() {
  return { id: 'id_1', cacheKey: 'k1', serverImageUrl: 'http://example.com/a.png', params: { prompt: 'x' }, addedAt: 1 };
}

describe('GalleryZoomOverlay', () => {
  it('renders image at 50vw/50vh bounds and closes on Close button', async () => {
    const onClose = vi.fn();
    await act(async () => {
      render(
        <GalleryZoomOverlay
          item={item()}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onClose={onClose}
        />,
      );
    });
    const img = await screen.findByAltText('x');
    expect(img.style.maxWidth).toBe('50vw');
    expect(img.style.maxHeight).toBe('50vh');
    fireEvent.click(screen.getByRole('button', { name: /close zoom/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it('closes on click outside the image frame', async () => {
    const onClose = vi.fn();
    await act(async () => {
      render(
        <GalleryZoomOverlay
          item={item()}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onClose={onClose}
        />,
      );
    });
    fireEvent.mouseDown(screen.getByTestId('zoom-backdrop'));
    expect(onClose).toHaveBeenCalled();
  });
});
