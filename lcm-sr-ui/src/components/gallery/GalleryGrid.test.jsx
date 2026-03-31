// @vitest-environment jsdom
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GalleryGrid } from './GalleryGrid';

afterEach(() => {
  cleanup();
});

function makeItem(n, override = {}) {
  return {
    id: `id_${n}`,
    galleryId: 'gal_1',
    cacheKey: `key_${n}`,
    serverImageUrl: `http://example.com/img${n}.png`,
    params: { prompt: `item ${n}`, seed: n },
    addedAt: 1000 * n,
    ...override,
  };
}

const resolve = (item) => Promise.resolve(item.serverImageUrl);
const resolveNull = () => Promise.resolve(null);

describe('GalleryGrid', () => {
  it('shows empty state when items is empty', () => {
    render(<GalleryGrid items={[]} resolveImageUrl={resolve} onOpenViewer={vi.fn()} />);
    expect(screen.getByText(/no images in this gallery yet/i)).toBeInTheDocument();
  });

  it('renders thumbnail cells for each item on the first page', async () => {
    const items = Array.from({ length: 5 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} resolveImageUrl={resolve} onOpenViewer={vi.fn()} />);
    });
    const imgs = screen.getAllByRole('img');
    expect(imgs).toHaveLength(5);
  });

  it('paginates — only shows 20 items per page', async () => {
    const items = Array.from({ length: 25 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} resolveImageUrl={resolve} onOpenViewer={vi.fn()} />);
    });
    expect(screen.getAllByRole('img')).toHaveLength(20);
    expect(screen.getByText(/page 1 of 2/i)).toBeInTheDocument();
  });

  it('Next button advances to page 2', async () => {
    const items = Array.from({ length: 25 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} resolveImageUrl={resolve} onOpenViewer={vi.fn()} />);
    });
    fireEvent.click(screen.getByRole('button', { name: /next/i }));
    expect(screen.getByText(/page 2 of 2/i)).toBeInTheDocument();
    expect(screen.getAllByRole('img')).toHaveLength(5);
  });

  it('Prev button is disabled on first page', async () => {
    const items = Array.from({ length: 5 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} resolveImageUrl={resolve} onOpenViewer={vi.fn()} />);
    });
    expect(screen.getByRole('button', { name: /prev/i })).toBeDisabled();
  });

  it('calls onOpenViewer when a thumbnail is clicked', async () => {
    const onOpen = vi.fn();
    const items = [makeItem(0)];
    await act(async () => {
      render(<GalleryGrid items={items} resolveImageUrl={resolve} onOpenViewer={onOpen} />);
    });
    fireEvent.click(screen.getByRole('img'));
    expect(onOpen).toHaveBeenCalledWith(items[0]);
  });

  it('Space key on a thumbnail opens window.open with the resolved URL', async () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const items = [makeItem(0)];
    await act(async () => {
      render(<GalleryGrid items={items} resolveImageUrl={resolve} onOpenViewer={vi.fn()} />);
    });
    const cell = screen.getByRole('img').closest('[data-gallery-cell]');
    fireEvent.keyDown(cell, { key: ' ' });
    expect(openSpy).toHaveBeenCalledWith('http://example.com/img0.png', '_blank');
    openSpy.mockRestore();
  });

  it('Space key does nothing when resolvedUrl is null', async () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const items = [makeItem(0, { serverImageUrl: null })];
    await act(async () => {
      render(<GalleryGrid items={items} resolveImageUrl={resolveNull} onOpenViewer={vi.fn()} />);
    });
    const cell = document.querySelector('[data-gallery-cell]');
    fireEvent.keyDown(cell, { key: ' ' });
    expect(openSpy).not.toHaveBeenCalled();
    openSpy.mockRestore();
  });
});
