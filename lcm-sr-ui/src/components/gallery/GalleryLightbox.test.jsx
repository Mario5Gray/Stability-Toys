// @vitest-environment jsdom
import 'fake-indexeddb/auto';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
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

describe('GalleryLightbox — selection action bar', () => {
  it('renders the action bar after selecting a thumbnail and fires onMoveToTrash on Delete', async () => {
    const getImages = vi.fn(async () => items);
    const onMoveToTrash = vi.fn(async () => {});

    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="gal_1"
          galleryName="Alpha"
          getGalleryImages={getImages}
          onClose={vi.fn()}
          onMoveToTrash={onMoveToTrash}
          onRestoreFromTrash={vi.fn()}
          onHardDelete={vi.fn()}
        />,
      );
    });

    const firstImg = screen.getAllByRole('img')[0];
    fireEvent.click(firstImg);
    await new Promise((r) => setTimeout(r, 200));
    expect(screen.getByText('1 selected')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /^delete$/i }));
    expect(onMoveToTrash).toHaveBeenCalledWith(['r1']);
  });

  it('in trash mode, menu shows Restore and Delete permanently', async () => {
    const trashItems = [
      { id: 'id_1', galleryId: '__trash__', cacheKey: 'k1', serverImageUrl: 'x', params: { prompt: 'trashed' }, addedAt: 1 },
    ];
    const getImages = vi.fn(async () => trashItems);
    const onRestore = vi.fn(async () => {});
    const onHardDelete = vi.fn(async () => {});

    await act(async () => {
      render(
        <GalleryLightbox
          galleryId="__trash__"
          galleryName="Trash"
          trashMode
          getGalleryImages={getImages}
          onClose={vi.fn()}
          onMoveToTrash={vi.fn()}
          onRestoreFromTrash={onRestore}
          onHardDelete={onHardDelete}
        />,
      );
    });

    await waitFor(() => expect(screen.getAllByRole('img').length).toBeGreaterThan(0));
    fireEvent.click(screen.getAllByRole('img')[0]);
    await new Promise((r) => setTimeout(r, 200));
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    expect(screen.getByRole('menuitem', { name: /restore/i })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /delete permanently/i })).toBeInTheDocument();
  });
});
