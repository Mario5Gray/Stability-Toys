// @vitest-environment jsdom
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GalleryCreatePopover } from './GalleryCreatePopover';

afterEach(() => {
  cleanup();
});

describe('GalleryCreatePopover', () => {
  it('renders the [+] trigger button', () => {
    render(<GalleryCreatePopover onCreateGallery={vi.fn()} />);
    expect(screen.getByRole('button', { name: /new gallery/i })).toBeInTheDocument();
  });

  it('shows the name input after clicking the trigger', () => {
    render(<GalleryCreatePopover onCreateGallery={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /new gallery/i }));
    expect(screen.getByLabelText(/gallery name/i)).toBeInTheDocument();
  });

  it('calls onCreateGallery and closes when Enter is pressed', () => {
    const onCreate = vi.fn();
    render(<GalleryCreatePopover onCreateGallery={onCreate} />);
    fireEvent.click(screen.getByRole('button', { name: /new gallery/i }));
    const input = screen.getByLabelText(/gallery name/i);
    fireEvent.change(input, { target: { value: 'Nature' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onCreate).toHaveBeenCalledWith('Nature');
    expect(screen.queryByLabelText(/gallery name/i)).not.toBeInTheDocument();
  });

  it('calls onCreateGallery when Create button is clicked', () => {
    const onCreate = vi.fn();
    render(<GalleryCreatePopover onCreateGallery={onCreate} />);
    fireEvent.click(screen.getByRole('button', { name: /new gallery/i }));
    fireEvent.change(screen.getByLabelText(/gallery name/i), { target: { value: 'Portraits' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    expect(onCreate).toHaveBeenCalledWith('Portraits');
  });

  it('does not call onCreateGallery when name is empty', () => {
    const onCreate = vi.fn();
    render(<GalleryCreatePopover onCreateGallery={onCreate} />);
    fireEvent.click(screen.getByRole('button', { name: /new gallery/i }));
    fireEvent.keyDown(screen.getByLabelText(/gallery name/i), { key: 'Enter' });
    expect(onCreate).not.toHaveBeenCalled();
  });
});
