// @vitest-environment jsdom
import React from 'react';
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup } from '@testing-library/react';
import { SurfaceHeader } from './SurfaceHeader';

afterEach(() => cleanup());

describe('SurfaceHeader', () => {
  it('renders title', () => {
    render(<SurfaceHeader title="LCM + SR Chat" />);
    expect(screen.getByText('LCM + SR Chat')).toBeInTheDocument();
  });

  it('renders chip labels', () => {
    render(
      <SurfaceHeader
        title="Chat"
        chips={[
          { label: 'UI abc1234', variant: 'outline' },
          { label: 'API 1.0.0',  variant: 'outline' },
        ]}
      />
    );
    expect(screen.getByText('UI abc1234')).toBeInTheDocument();
    expect(screen.getByText('API 1.0.0')).toBeInTheDocument();
  });

  it('renders summary text', () => {
    render(<SurfaceHeader title="Chat" summary="Tip: press Ctrl + Enter to send." />);
    expect(screen.getByText('Tip: press Ctrl + Enter to send.')).toBeInTheDocument();
  });

  it('contains no animate-pulse elements — header must be calm', () => {
    const { container } = render(
      <SurfaceHeader title="Chat" chips={[{ label: 'UI abc' }]} summary="Some tip" />
    );
    expect(container.querySelector('.animate-pulse')).toBeNull();
  });
});
