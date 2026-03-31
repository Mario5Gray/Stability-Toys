// @vitest-environment jsdom
import 'fake-indexeddb/auto';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, expect, it, vi, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import { GalleryLightbox } from './GalleryLightbox';

afterEach(cleanup);

const items = [
  {
    id: 'r1', galleryId: 'gal_1', cacheKey: 'k1',
    serverImageUrl: 'http://example.com/1.png',
    params: { prompt: 'cat', seed: 1 }, addedAt: 2000,
  },
  {
    id: 'r2', galleryId: 'gal_1', cacheKey: 'k2',
    serverImageUrl: 'http://example.com/2.png',
    params: { prompt: 'dog', seed: 2 }, addedAt: 1000,
  },
];

const getGalleryImages = vi.fn().mockResolvedValue(items);

describe('GalleryLightbox', () => {
  it('renders the gallery name in the toolbar', async () => {
    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="gal_1"
          galleryName="Nature"
          getGalleryImages={getGalleryImages}
          onClose={vi.fn()}
        />
      );
    });
    expect(screen.getByText('Nature')).toBeInTheDocument();
  });

  it('renders the close button', async () => {
    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="gal_1"
          galleryName="Nature"
          getGalleryImages={getGalleryImages}
          onClose={vi.fn()}
        />
      );
    });
    expect(screen.getByRole('button', { name: /close/i })).toBeInTheDocument();
  });

  it('calls onClose when the X button is clicked', async () => {
    const onClose = vi.fn();
    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="gal_1"
          galleryName="Nature"
          getGalleryImages={getGalleryImages}
          onClose={onClose}
        />
      );
    });
    fireEvent.click(screen.getByRole('button', { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when ESC is pressed', async () => {
    const onClose = vi.fn();
    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="gal_1"
          galleryName="Nature"
          getGalleryImages={getGalleryImages}
          onClose={onClose}
        />
      );
    });
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  it('renders an opacity range slider in the toolbar', async () => {
    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="gal_1"
          galleryName="Nature"
          getGalleryImages={getGalleryImages}
          onClose={vi.fn()}
        />
      );
    });
    const slider = screen.getByRole('slider');
    expect(slider).toHaveAttribute('min', '0.7');
    expect(slider).toHaveAttribute('max', '1');
  });

});
