// src/lib/slashCommands/gen.js — /gen slash command
//
// Sends one image-generation prompt through the existing runGenerate path
// and switches the composer back to generate routing mode.

import { register } from '../slashCommands.js';

function genHandler({ args, ctx }) {
  if (!args) return false;
  ctx.runGenerate({ prompt: args });
  ctx.setInputMode('generate');
  return true;
}

register('gen', {
  description: 'Generate an image',
  enabled: () => true,
  handler: genHandler,
});

register('generate', {
  description: 'Generate an image (alias for /gen)',
  enabled: () => true,
  handler: genHandler,
});
