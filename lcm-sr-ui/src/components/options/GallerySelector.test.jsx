// @vitest-environment jsdom
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GallerySelector } from './OptionsPanel';

afterEach(cleanup);

// jsdom does not implement scrollIntoView; Radix UI Select needs it when opening
window.HTMLElement.prototype.scrollIntoView = vi.fn();

const galleries = [
  { id: 'gal_1', name: 'Nature', createdAt: 1000 },
  { id: 'gal_2', name: 'Portraits', createdAt: 2000 },
];

describe('GallerySelector', () => {
  it('renders the label "Active Gallery"', () => {
    render(
      <GallerySelector
        galleries={galleries}
        activeGalleryId={null}
        setActiveGalleryId={vi.fn()}
      />
    );
    expect(screen.getByText(/active gallery/i)).toBeInTheDocument();
  });

  it('renders "None" as the first option', () => {
    render(
      <GallerySelector
        galleries={galleries}
        activeGalleryId={null}
        setActiveGalleryId={vi.fn()}
      />
    );
    expect(screen.getByText('None')).toBeInTheDocument();
  });

  it('renders each gallery name', () => {
    render(
      <GallerySelector
        galleries={galleries}
        activeGalleryId={null}
        setActiveGalleryId={vi.fn()}
      />
    );
    // Open the dropdown so Radix renders items into the portal
    fireEvent.click(screen.getByRole('combobox'));
    expect(screen.getByText('Nature')).toBeInTheDocument();
    expect(screen.getByText('Portraits')).toBeInTheDocument();
  });
});
