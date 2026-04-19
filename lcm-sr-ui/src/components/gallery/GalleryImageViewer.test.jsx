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

  it('backend field renders when present in params', async () => {
    const itemWithBackend = {
      ...item,
      params: { ...item.params, backend: 'cuda' },
    };

    await act(async () => {
      render(
        <GalleryImageViewer
          item={itemWithBackend}
          resolveImageUrl={resolve}
          onBack={vi.fn()}
          onWindowOpen={vi.fn()}
        />
      );
    });

    expect(screen.getByTestId('metadata-bar').textContent).toContain('cuda');
  });

  it('shows placeholder when resolveImageUrl returns null', async () => {
    await act(async () => {
      render(
        <GalleryImageViewer
          item={item}
          resolveImageUrl={() => Promise.resolve(null)}
          onBack={vi.fn()}
          onWindowOpen={vi.fn()}
        />
      );
    });
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('addedAt is formatted via toLocaleString', async () => {
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

    const expected = new Date(item.addedAt).toLocaleString();
    expect(screen.getByTestId('metadata-bar').textContent).toContain(expected);
  });

  it('pointer-events-none is present on hidden metadata bar', async () => {
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
    expect(metaBar.className).toContain('pointer-events-none');
  });

  it('renders an Open in new tab button that calls window.open with resolved url', async () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue({ closed: false });
    const onWindowOpen = vi.fn();
    await act(async () => {
      render(
        <GalleryImageViewer
          item={item}
          resolveImageUrl={resolve}
          onBack={vi.fn()}
          onWindowOpen={onWindowOpen}
        />,
      );
    });
    await screen.findByAltText('a cat');
    fireEvent.click(screen.getByRole('button', { name: /open in new tab/i }));
    expect(openSpy).toHaveBeenCalledWith('http://example.com/img.png', '_blank');
    expect(onWindowOpen).toHaveBeenCalled();
    openSpy.mockRestore();
  });
});

describe('GalleryImageViewer — keyboard navigation', () => {
  const makeKeymap = () => ({
    matches: (action, e) => ({
      next: e.code === 'ArrowRight',
      prev: e.code === 'ArrowLeft',
      delete: e.code === 'Backspace',
      delete_alt: e.code === 'Delete',
      close: e.code === 'Escape',
      open_new_tab: e.code === 'Space',
    }[action] ?? false),
  });

  it('ArrowRight calls onNext', async () => {
    const onNext = vi.fn();
    await act(async () => {
      render(
        <GalleryImageViewer
          item={{ id: 'id_1', serverImageUrl: 'x', params: { prompt: 'p' }, addedAt: 1 }}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onBack={vi.fn()}
          onNext={onNext}
          onPrev={vi.fn()}
          onDelete={vi.fn()}
          keymap={makeKeymap()}
          onWindowOpen={vi.fn()}
        />,
      );
    });
    fireEvent.keyDown(document, { code: 'ArrowRight' });
    expect(onNext).toHaveBeenCalled();
  });

  it('Backspace calls onDelete', async () => {
    const onDelete = vi.fn();
    await act(async () => {
      render(
        <GalleryImageViewer
          item={{ id: 'id_1', serverImageUrl: 'x', params: { prompt: 'p' }, addedAt: 1 }}
          resolveImageUrl={(it) => Promise.resolve(it.serverImageUrl)}
          onBack={vi.fn()}
          onNext={vi.fn()}
          onPrev={vi.fn()}
          onDelete={onDelete}
          keymap={makeKeymap()}
          onWindowOpen={vi.fn()}
        />,
      );
    });
    fireEvent.keyDown(document, { code: 'Backspace' });
    expect(onDelete).toHaveBeenCalled();
  });
});
