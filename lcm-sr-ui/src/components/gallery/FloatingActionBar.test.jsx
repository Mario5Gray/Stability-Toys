// @vitest-environment jsdom
import { cleanup, render, screen, fireEvent } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { FloatingActionBar } from './FloatingActionBar';

afterEach(() => {
  cleanup();
});

describe('FloatingActionBar', () => {
  it('does not render when selection is empty', () => {
    const { container } = render(
      <FloatingActionBar selectedCount={0} trashMode={false} onDelete={vi.fn()} onClear={vi.fn()} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders count and delete button in normal context', () => {
    const onDelete = vi.fn();
    render(
      <FloatingActionBar selectedCount={2} trashMode={false} onDelete={onDelete} onClear={vi.fn()} />,
    );
    expect(screen.getByText('2 selected')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /delete/i }));
    expect(onDelete).toHaveBeenCalled();
  });

  it('renders Restore + Delete permanently in trash context', () => {
    const onRestore = vi.fn();
    const onHardDelete = vi.fn();
    render(
      <FloatingActionBar
        selectedCount={1}
        trashMode
        onRestore={onRestore}
        onHardDelete={onHardDelete}
        onClear={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /restore/i }));
    expect(onRestore).toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    fireEvent.click(screen.getByRole('menuitem', { name: /delete permanently/i }));
    expect(onHardDelete).toHaveBeenCalled();
  });

  it('clear button calls onClear', () => {
    const onClear = vi.fn();
    render(
      <FloatingActionBar selectedCount={1} trashMode={false} onDelete={vi.fn()} onClear={onClear} />,
    );
    fireEvent.click(screen.getByRole('button', { name: /clear selection/i }));
    expect(onClear).toHaveBeenCalled();
  });
});
