// @vitest-environment jsdom
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { PanelActionBar } from './PanelActionBar';

afterEach(() => cleanup());

const primary = { label: 'Apply', subtext: 'Append to prompt', onClick: vi.fn() };
const secondary = [
  { label: 'Rebuild', subtext: 'Refresh digest from gallery', onClick: vi.fn() },
  { label: 'Reset',   subtext: 'Restore digest text',         onClick: vi.fn() },
];

describe('PanelActionBar', () => {
  it('renders primary action label and subtext', () => {
    render(<PanelActionBar primary={primary} />);
    expect(screen.getByText('Apply')).toBeInTheDocument();
    expect(screen.getByText('Append to prompt')).toBeInTheDocument();
  });

  it('renders secondary actions', () => {
    render(<PanelActionBar primary={primary} secondary={secondary} />);
    expect(screen.getByText('Rebuild')).toBeInTheDocument();
    expect(screen.getByText('Reset')).toBeInTheDocument();
  });

  it('primary is a <button> element at rest — identifiable without hover', () => {
    render(<PanelActionBar primary={primary} />);
    const btn = screen.getByRole('button', { name: /apply/i });
    expect(btn.tagName).toBe('BUTTON');
  });

  it('primary onClick fires on click', () => {
    const onClick = vi.fn();
    render(<PanelActionBar primary={{ ...primary, onClick }} />);
    fireEvent.click(screen.getByRole('button', { name: /apply/i }));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('disabled primary is a disabled control that still renders', () => {
    render(<PanelActionBar primary={{ ...primary, disabled: true }} />);
    const btn = screen.getByRole('button', { name: /apply/i });
    expect(btn).toBeDisabled();
  });

  it('secondary action labels are visible', () => {
    render(<PanelActionBar primary={primary} secondary={secondary} />);
    expect(screen.getByText('Rebuild')).toBeVisible();
    expect(screen.getByText('Reset')).toBeVisible();
  });
});
