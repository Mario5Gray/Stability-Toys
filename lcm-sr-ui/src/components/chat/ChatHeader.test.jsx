// @vitest-environment jsdom

import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { ChatHeader } from './ChatHeader';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

describe('ChatHeader', () => {
  it('renders a neutral api placeholder before runtime status loads', () => {
    render(
      <ChatHeader
        inflightCount={0}
        isDreaming={false}
        srLevel={0}
        frontendVersion="abc1234"
      />
    );

    expect(screen.getByText('UI abc1234')).toBeInTheDocument();
    expect(screen.getByText('API ...')).toBeInTheDocument();
  });

  it('renders frontend and backend version badges', () => {
    render(
      <ChatHeader
        inflightCount={0}
        isDreaming={false}
        srLevel={0}
        frontendVersion="abc1234"
        backendVersion="abc1234"
      />
    );

    expect(screen.getByText('UI abc1234')).toBeInTheDocument();
    expect(screen.getByText('API abc1234')).toBeInTheDocument();
  });
});
