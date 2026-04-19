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

const defaultProps = {
  resolveImageUrl: resolve,
  onOpenViewer: vi.fn(),
  onToggle: vi.fn(),
  onRange: vi.fn(),
  onZoom: vi.fn(),
  selectedIds: new Set(),
  anchorId: null,
};

describe('GalleryGrid', () => {
  it('shows empty state when items is empty', () => {
    render(<GalleryGrid items={[]} {...defaultProps} />);
    expect(screen.getByText(/no images in this gallery yet/i)).toBeInTheDocument();
  });

  it('renders thumbnail cells for each item on the first page', async () => {
    const items = Array.from({ length: 5 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} {...defaultProps} />);
    });
    const imgs = screen.getAllByRole('img');
    expect(imgs).toHaveLength(5);
  });

  it('paginates — only shows 20 items per page', async () => {
    const items = Array.from({ length: 25 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} {...defaultProps} />);
    });
    expect(screen.getAllByRole('img')).toHaveLength(20);
    expect(screen.getByText(/page 1 of 2/i)).toBeInTheDocument();
  });

  it('Next button advances to page 2', async () => {
    const items = Array.from({ length: 25 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} {...defaultProps} />);
    });
    fireEvent.click(screen.getByRole('button', { name: /next/i }));
    expect(screen.getByText(/page 2 of 2/i)).toBeInTheDocument();
    expect(screen.getAllByRole('img')).toHaveLength(5);
  });

  it('Prev button is disabled on first page', async () => {
    const items = Array.from({ length: 5 }, (_, i) => makeItem(i));
    await act(async () => {
      render(<GalleryGrid items={items} {...defaultProps} />);
    });
    expect(screen.getByRole('button', { name: /prev/i })).toBeDisabled();
  });

  it('Space key on a thumbnail opens window.open with the resolved URL', async () => {
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
    const items = [makeItem(0)];
    await act(async () => {
      render(<GalleryGrid items={items} {...defaultProps} />);
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
      render(<GalleryGrid items={items} {...defaultProps} resolveImageUrl={resolveNull} />);
    });
    const cell = document.querySelector('[data-gallery-cell]');
    fireEvent.keyDown(cell, { key: ' ' });
    expect(openSpy).not.toHaveBeenCalled();
    openSpy.mockRestore();
  });
});

describe('GalleryGrid — selection', () => {
  it('click on thumbnail calls onToggle with item id', async () => {
    vi.useFakeTimers();
    const items = Array.from({ length: 3 }, (_, i) => makeItem(i));
    const onToggle = vi.fn();
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          {...defaultProps}
          onToggle={onToggle}
        />,
      );
    });
    fireEvent.click(screen.getAllByRole('img')[1]);
    vi.advanceTimersByTime(200);
    expect(onToggle).toHaveBeenCalledWith('id_1', { shift: false, mod: false });
    vi.useRealTimers();
  });

  it('shift+click calls onRange', async () => {
    vi.useFakeTimers();
    const items = Array.from({ length: 3 }, (_, i) => makeItem(i));
    const onRange = vi.fn();
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          {...defaultProps}
          onRange={onRange}
        />,
      );
    });
    fireEvent.click(screen.getAllByRole('img')[2], { shiftKey: true });
    vi.advanceTimersByTime(200);
    expect(onRange).toHaveBeenCalledWith('id_2');
    vi.useRealTimers();
  });

  it('double-click calls onZoom', async () => {
    const items = [makeItem(0)];
    const onZoom = vi.fn();
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          {...defaultProps}
          onZoom={onZoom}
        />,
      );
    });
    fireEvent.doubleClick(screen.getAllByRole('img')[0]);
    expect(onZoom).toHaveBeenCalledWith(items[0]);
  });

  it('selected items get an aria-selected=true attribute', async () => {
    const items = Array.from({ length: 2 }, (_, i) => makeItem(i));
    await act(async () => {
      render(
        <GalleryGrid
          items={items}
          {...defaultProps}
          selectedIds={new Set(['id_1'])}
          anchorId={'id_1'}
        />,
      );
    });
    const cells = screen.getAllByRole('gridcell');
    expect(cells[1].getAttribute('aria-selected')).toBe('true');
    expect(cells[0].getAttribute('aria-selected')).toBe('false');
  });
});

describe('GalleryGrid — keyboard navigation', () => {
  function makeKeymap() {
    return {
      matches: (action, e) => {
        const map = {
          right: e.code === 'ArrowRight',
          next: e.code === 'ArrowRight',
          left: e.code === 'ArrowLeft',
          prev: e.code === 'ArrowLeft',
          down: e.code === 'ArrowDown',
          up: e.code === 'ArrowUp',
          delete: e.code === 'Backspace',
          delete_alt: e.code === 'Delete',
          select_all: e.code === 'KeyA' && (e.metaKey || e.ctrlKey),
          deselect_all: e.code === 'Escape',
          zoom: e.code === 'Enter',
          open_new_tab: e.code === 'Space',
        };
        return map[action] ?? false;
      },
    };
  }

  function renderGrid(overrides = {}) {
    const items = Array.from({ length: 10 }, (_, i) => makeItem(i));
    const props = {
      items, resolveImageUrl: resolve, onOpenViewer: vi.fn(),
      onToggle: vi.fn(), onRange: vi.fn(), onZoom: vi.fn(),
      onDeleteAction: vi.fn(), onSelectAll: vi.fn(), onDeselectAll: vi.fn(),
      selectedIds: new Set(), anchorId: null,
      keymap: makeKeymap(),
      ...overrides,
    };
    render(<GalleryGrid {...props} />);
    return props;
  }

  it('ArrowRight moves focus to the next cell', async () => {
    renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'ArrowRight' });
    expect(document.activeElement).toBe(screen.getAllByRole('gridcell')[1]);
  });

  it('Backspace calls onDeleteAction with selection', async () => {
    const selected = new Set(['id_2']);
    const props = renderGrid({ selectedIds: selected });
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'Backspace' });
    expect(props.onDeleteAction).toHaveBeenCalledWith(['id_2']);
  });

  it('Backspace with empty selection deletes focused cell', async () => {
    const props = renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'Backspace' });
    expect(props.onDeleteAction).toHaveBeenCalledWith(['id_0']);
  });

  it('Cmd+A triggers select all', async () => {
    const props = renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'KeyA', metaKey: true });
    expect(props.onSelectAll).toHaveBeenCalled();
  });

  it('Escape triggers deselect all', async () => {
    const props = renderGrid();
    await act(async () => {
      screen.getAllByRole('gridcell')[0].focus();
    });
    fireEvent.keyDown(document.activeElement, { code: 'Escape' });
    expect(props.onDeselectAll).toHaveBeenCalled();
  });
});
