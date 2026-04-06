// @vitest-environment jsdom

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ChatHeader } from './ChatHeader';

describe('ChatHeader', () => {
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
