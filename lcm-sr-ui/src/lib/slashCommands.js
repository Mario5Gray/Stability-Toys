// src/lib/slashCommands.js — Slash-command registry
//
// Pure JS. No React. Register handlers once at module load; dispatch from composer.

const registry = new Map();

/**
 * Register a slash command.
 * @param {string} name - Command name without leading slash (e.g. "chat")
 * @param {{ description: string, enabled?: (ctx) => bool, disabledReason?: (ctx) => string, handler: ({args, ctx}) => bool }} def
 */
export function register(name, def) {
  registry.set(name, def);
}

/**
 * Parse a slash command from raw input.
 * Returns null if input does not start with "/".
 * @param {string} input
 * @returns {{ command: string, args: string, raw: string } | null}
 */
export function parse(input) {
  if (!input || !input.startsWith('/')) return null;
  const m = input.match(/^\/(\S+)(?:\s+([\s\S]*))?$/);
  if (!m) return null;
  return { command: m[1], args: (m[2] || '').trim(), raw: input };
}

/**
 * List all registered commands, resolving enabled/disabledReason against ctx.
 * @param {object} ctx - Dispatch context
 * @returns {Array<{ name: string, description: string, enabled: bool, disabledReason: string|null }>}
 */
export function list(ctx) {
  return Array.from(registry.entries()).map(([name, def]) => {
    const enabled = def.enabled ? def.enabled(ctx) : true;
    const disabledReason = !enabled && def.disabledReason ? def.disabledReason(ctx) : null;
    return { name, description: def.description, enabled, disabledReason };
  });
}

/**
 * Get a registered command by name.
 * @param {string} name
 * @returns {object|null}
 */
export function get(name) {
  return registry.get(name) ?? null;
}
