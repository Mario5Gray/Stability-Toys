// @vitest-environment jsdom
import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, expect, it, vi, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import { GalleryImageViewer } from './GalleryImageViewer';

afterEach(cleanup);

const item = {
  id: 'row_1',
  galleryId: 'gal_1',
  cacheKey: 'key_abc',
  serverImageUrl: 'http://example.com/img.png',
  params: { prompt: 'a cat', seed: 42, size: '512x512', steps: 20, cfg: 7.5 },
  addedAt: 1711670000000,
};

const resolve = (i) => Promise.resolve(i.serverImageUrl);

describe('GalleryImageViewer', () => {
  it('renders the image after URL resolves', async () => {
    await act(async () => {
      render(
        <GalleryImageViewer
          item={item}
          resolveImageUrl={resolve}
          onBack={vi.fn()}
          onWindowOpen={vi.fn()}
        />
      );
    });
    expect(screen.getByRole('img')).toHaveAttribute('src', 'http://example.com/img.png');
  });

  it('calls onBack when back button is clicked', async () => {
    const onBack = vi.fn();
    await act(async () => {
      render(
        <GalleryImageViewer
          item={item}
          resolveImageUrl={resolve}
          onBack={onBack}
          onWindowOpen={vi.fn()}
        />
      );
    });
    fireEvent.click(screen.getByRole('button', { name: /back/i }));
    expect(onBack).toHaveBeenCalled();
  });

  it('metadata bar is hidden by default', async () => {
    await act(async () => {
      render(
        <GalleryImageViewer
          item={item}
          resolveImageUrl={resolve}
          onBack={vi.fn()}
          onWindowOpen={vi.fn()}
        />
      );
    });
    const metaBar = screen.getByTestId('metadata-bar');
    expect(metaBar.className).toContain('opacity-0');
  });

  it('metadata bar becomes visible when pointer moves into lower 20%', async () => {
    await act(async () => {
      render(
        <GalleryImageViewer
          item={item}
          resolveImageUrl={resolve}
          onBack={vi.fn()}
          onWindowOpen={vi.fn()}
        />
      );
    });
    const container = screen.getByTestId('viewer-container');
    // Simulate getBoundingClientRect returning a 500px tall rect
    vi.spyOn(container, 'getBoundingClientRect').mockReturnValue({
      top: 0, bottom: 500, left: 0, right: 500, height: 500, width: 500,
    });
    // Move pointer into lower 20% (clientY > 400 = 80% of 500)
    fireEvent.mouseMove(container, { clientY: 420 });
    expect(screen.getByTestId('metadata-bar').className).toContain('opacity-100');
  });

  it('spacebar calls window.open and onWindowOpen with result', async () => {
    const mockWin = { close: vi.fn() };
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(mockWin);
    const onWindowOpen = vi.fn();

    await act(async () => {
      render(
        <GalleryImageViewer
          item={item}
          resolveImageUrl={resolve}
          onBack={vi.fn()}
          onWindowOpen={onWindowOpen}
        />
      );
    });

    fireEvent.keyDown(document, { key: ' ' });
    expect(openSpy).toHaveBeenCalledWith('http://example.com/img.png', '_blank');
    expect(onWindowOpen).toHaveBeenCalledWith(mockWin);
    openSpy.mockRestore();
  });
});
