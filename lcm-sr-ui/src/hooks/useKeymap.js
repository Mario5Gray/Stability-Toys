import { useCallback, useEffect, useMemo, useState } from 'react';

const OVERRIDES_KEY = 'lcm-keymap-overrides';

const HARDCODED_FALLBACK = {
  next:         { code: 'ArrowRight', label: 'Next' },
  prev:         { code: 'ArrowLeft',  label: 'Previous' },
  up:           { code: 'ArrowUp',    label: 'Up' },
  down:         { code: 'ArrowDown',  label: 'Down' },
  delete:       { code: 'Backspace',  label: 'Delete' },
  delete_alt:   { code: 'Delete',     label: 'Delete' },
  select_all:   { code: 'KeyA', mod: 'mod', label: 'Select all' },
  deselect_all: { code: 'Escape',     label: 'Deselect' },
  close:        { code: 'Escape',     label: 'Close' },
  zoom:         { code: 'Enter',      label: 'Zoom overlay' },
  open_new_tab: { code: 'Space',      label: 'Open in new tab' },
};

function readOverrides() {
  try {
    const raw = localStorage.getItem(OVERRIDES_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeOverrides(overrides) {
  try {
    localStorage.setItem(OVERRIDES_KEY, JSON.stringify(overrides));
  } catch {
    // quota or storage disabled — keep in-memory state
  }
}

function modPressed(event) {
  return Boolean(event.metaKey || event.ctrlKey);
}

export function useKeymap() {
  const [defaults, setDefaults] = useState(HARDCODED_FALLBACK);
  const [overrides, setOverrides] = useState(() => readOverrides());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    fetch('/api/keymap/defaults')
      .then((res) => (res.ok ? res.json() : { keymap: {} }))
      .then((body) => {
        if (!active) return;
        const serverMap = body && typeof body.keymap === 'object' ? body.keymap : {};
        if (Object.keys(serverMap).length > 0) {
          setDefaults({ ...HARDCODED_FALLBACK, ...serverMap });
        }
      })
      .catch(() => {})
      .finally(() => { if (active) setReady(true); });
    return () => { active = false; };
  }, []);

  const merged = useMemo(() => ({ ...defaults, ...overrides }), [defaults, overrides]);

  const matches = useCallback((action, event) => {
    const binding = merged[action];
    if (!binding) return false;
    if (event.code !== binding.code) return false;
    if (binding.mod === 'mod') return modPressed(event);
    return !modPressed(event);
  }, [merged]);

  const bindingOf = useCallback((action) => merged[action] ?? null, [merged]);

  const setBinding = useCallback((action, code, mod) => {
    const next = { ...overrides, [action]: { code, ...(mod ? { mod } : {}), label: merged[action]?.label ?? action } };
    setOverrides(next);
    writeOverrides(next);
  }, [overrides, merged]);

  return { ready, matches, bindingOf, setBinding };
}
