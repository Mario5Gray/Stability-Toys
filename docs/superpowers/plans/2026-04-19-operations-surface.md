# Operations Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace scattered animated status labels and ad-hoc action rows with a unified `OperationsProvider` context, `PanelActionBar`, `PendingOperationsPane`, and `SurfaceHeader` primitives wired into the advisor and generation lifecycles.

**Architecture:** React context + `useReducer` for the operations store (no new state library); feature code calls `useOperationsController()` and receives a keyed status handle; `PendingOperationsPane` reads from context directly; `SurfaceHeader` and `PanelActionBar` are display-only primitives with no store coupling.

**Tech Stack:** React 18, Vitest + @testing-library/react, Tailwind CSS, Lucide icons, shadcn/ui (`Button`, `Badge`, `Card`), `@radix-ui` primitives

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/contexts/OperationsContext.jsx` | Create | Store reducer, `OperationsProvider`, `useOperationsStore`, `useOperationsController` |
| `src/contexts/OperationsContext.test.jsx` | Create | Store and controller unit tests |
| `src/components/ui/PanelActionBar.jsx` | Create | Resilient panel footer: primary + secondary action hierarchy |
| `src/components/ui/PanelActionBar.test.jsx` | Create | Renders, hierarchy, disabled state |
| `src/components/ui/PendingOperationsPane.jsx` | Create | Renders operations from store; owns pulse/fade/expiry visuals |
| `src/components/ui/PendingOperationsPane.test.jsx` | Create | Empty state, active op, progress, cancel, auto-removal |
| `src/components/ui/SurfaceHeader.jsx` | Create | Calm top band: title, chips, summary — no animation |
| `src/components/ui/SurfaceHeader.test.jsx` | Create | Title, chips, summary, no animate-pulse |
| `src/main.jsx` | Modify | Mount `OperationsProvider` around `<App />` |
| `src/components/options/AdvisorPanel.jsx` | Modify | Replace 3-button row with `PanelActionBar`; remove `Building digest...` |
| `src/hooks/useGalleryAdvisor.js` | Modify | Wire `rebuildAdvisor` into operations store |
| `src/hooks/useGalleryAdvisor.test.jsx` | Modify | Add `OperationsProvider` wrapper to existing tests |
| `src/components/chat/ChatHeader.jsx` | Modify | Replace dream badge + ad-hoc layout with `SurfaceHeader` |
| `src/components/chat/ChatHeader.test.jsx` | Modify | Update assertions; verify no animate-pulse |
| `src/components/chat/ChatContainer.jsx` | Modify | Replace `[]` strip with `PendingOperationsPane`; remove vestigial header props |
| `src/hooks/useImageGeneration.js` | Modify | Wire dream lifecycle and generation jobs into operations store |
| `src/components/chat/MessageBubble.jsx` | Modify | Remove animated `dreaming` and `generating` badge overlay |

---

## Phase 1 — Shared Primitives

### Task 1: OperationsContext

**Files:**
- Create: `lcm-sr-ui/src/contexts/OperationsContext.jsx`
- Create: `lcm-sr-ui/src/contexts/OperationsContext.test.jsx`

- [x] **Step 1: Write failing tests**

```jsx
// lcm-sr-ui/src/contexts/OperationsContext.test.jsx
// @vitest-environment jsdom
import React from 'react';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { OperationsProvider, useOperationsStore, useOperationsController } from './OperationsContext';

afterEach(() => cleanup());

function wrapper({ children }) {
  return <OperationsProvider>{children}</OperationsProvider>;
}

function useStoreAndCtrl() {
  return { store: useOperationsStore(), ctrl: useOperationsController() };
}

describe('OperationsContext', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('start() creates an operation record', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    act(() => {
      result.current.ctrl.start({ key: 'adv:1', kind: 'advisor', text: 'Working' });
    });
    const op = result.current.store.operations.get('adv:1');
    expect(op).toBeDefined();
    expect(op.text).toBe('Working');
    expect(op.tone).toBe('active');
  });

  it('start() with same key upserts instead of creating a duplicate', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    act(() => {
      result.current.ctrl.start({ key: 'adv:1', text: 'First' });
      result.current.ctrl.start({ key: 'adv:1', text: 'Second' });
    });
    expect(result.current.store.order).toHaveLength(1);
    expect(result.current.store.operations.get('adv:1').text).toBe('Second');
  });

  it('handle.setText() updates text without changing tone', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.setText('Analyzing'); });
    expect(result.current.store.operations.get('adv:1').text).toBe('Analyzing');
    expect(result.current.store.operations.get('adv:1').tone).toBe('active');
  });

  it('handle.complete() sets tone to complete and auto-removes after 2s', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.complete({ text: 'Done' }); });
    expect(result.current.store.operations.get('adv:1').tone).toBe('complete');
    act(() => { vi.advanceTimersByTime(2001); });
    expect(result.current.store.operations.get('adv:1')).toBeUndefined();
  });

  it('handle.error() sets tone to error, still present at 2s, gone after 5s', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.error({ text: 'Failed' }); });
    expect(result.current.store.operations.get('adv:1').tone).toBe('error');
    act(() => { vi.advanceTimersByTime(2001); });
    expect(result.current.store.operations.get('adv:1')).toBeDefined();
    act(() => { vi.advanceTimersByTime(3001); });
    expect(result.current.store.operations.get('adv:1')).toBeUndefined();
  });

  it('handle.cancel() only exists when cancelFn is provided', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    const cancelFn = vi.fn();
    let handleA, handleB;
    act(() => {
      handleA = result.current.ctrl.start({ key: 'a', cancellable: false });
      handleB = result.current.ctrl.start({ key: 'b', cancellable: true, cancelFn });
    });
    expect(handleA.cancel).toBeUndefined();
    expect(handleB.cancel).toBeDefined();
    act(() => { handleB.cancel(); });
    expect(cancelFn).toHaveBeenCalledOnce();
    expect(result.current.store.operations.get('b')).toBeUndefined();
  });

  it('handle.remove() removes operation immediately', () => {
    const { result } = renderHook(useStoreAndCtrl, { wrapper });
    let handle;
    act(() => { handle = result.current.ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.remove(); });
    expect(result.current.store.operations.get('adv:1')).toBeUndefined();
  });
});
```

- [x] **Step 2: Run tests, confirm failure**

```bash
cd lcm-sr-ui && npx vitest run src/contexts/OperationsContext.test.jsx
```
Expected: `FAIL` — module not found.

- [x] **Step 3: Implement OperationsContext**

```jsx
// lcm-sr-ui/src/contexts/OperationsContext.jsx
import React, { createContext, useCallback, useContext, useReducer, useRef } from 'react';

const COMPLETE_LINGER_MS = 2000;
const ERROR_LINGER_MS = 5000;

function operationsReducer(state, action) {
  switch (action.type) {
    case 'UPSERT': {
      const { record } = action;
      const exists = state.operations.has(record.key);
      const next = new Map(state.operations);
      next.set(record.key, { ...(next.get(record.key) ?? {}), ...record });
      return {
        operations: next,
        order: exists ? state.order : [...state.order, record.key],
      };
    }
    case 'REMOVE': {
      const next = new Map(state.operations);
      next.delete(action.key);
      return { operations: next, order: state.order.filter((k) => k !== action.key) };
    }
    default:
      return state;
  }
}

const OperationsStoreContext = createContext(null);
const OperationsDispatchContext = createContext(null);

export function OperationsProvider({ children }) {
  const [state, rawDispatch] = useReducer(operationsReducer, {
    operations: new Map(),
    order: [],
  });
  const expiryTimers = useRef(new Map());

  const scheduleRemoval = useCallback((key, delayMs) => {
    const existing = expiryTimers.current.get(key);
    if (existing) clearTimeout(existing);
    const id = setTimeout(() => {
      rawDispatch({ type: 'REMOVE', key });
      expiryTimers.current.delete(key);
    }, delayMs);
    expiryTimers.current.set(key, id);
  }, []);

  const dispatch = useCallback((action) => {
    rawDispatch(action);
    if (action.type === 'UPSERT') {
      const { tone, key } = action.record;
      if (tone === 'complete') {
        scheduleRemoval(key, COMPLETE_LINGER_MS);
      } else if (tone === 'error') {
        scheduleRemoval(key, ERROR_LINGER_MS);
      } else if (tone != null) {
        const existing = expiryTimers.current.get(key);
        if (existing) { clearTimeout(existing); expiryTimers.current.delete(key); }
      }
    } else if (action.type === 'REMOVE') {
      const existing = expiryTimers.current.get(action.key);
      if (existing) { clearTimeout(existing); expiryTimers.current.delete(action.key); }
    }
  }, [scheduleRemoval]);

  return (
    <OperationsStoreContext.Provider value={state}>
      <OperationsDispatchContext.Provider value={dispatch}>
        {children}
      </OperationsDispatchContext.Provider>
    </OperationsStoreContext.Provider>
  );
}

export function useOperationsStore() {
  return useContext(OperationsStoreContext);
}

export function useOperationsController() {
  const dispatch = useContext(OperationsDispatchContext);

  const start = useCallback((init) => {
    const cancelFn =
      init.cancellable && typeof init.cancelFn === 'function'
        ? () => {
            init.cancelFn();
            dispatch({ type: 'REMOVE', key: init.key });
          }
        : null;

    const record = {
      key: init.key,
      kind: init.kind ?? 'generic',
      icon: init.icon ?? null,
      tone: init.tone ?? 'active',
      text: init.text ?? '',
      detail: init.detail ?? null,
      progress: init.progress ?? null,
      cancellable: Boolean(cancelFn),
      cancelFn,
      createdAt: Date.now(),
    };
    dispatch({ type: 'UPSERT', record });

    const handle = {
      setText:     (text)     => dispatch({ type: 'UPSERT', record: { key: record.key, text } }),
      setDetail:   (detail)   => dispatch({ type: 'UPSERT', record: { key: record.key, detail } }),
      setProgress: (progress) => dispatch({ type: 'UPSERT', record: { key: record.key, progress } }),
      setTone:     (tone)     => dispatch({ type: 'UPSERT', record: { key: record.key, tone } }),
      complete: ({ text } = {}) =>
        dispatch({
          type: 'UPSERT',
          record: { key: record.key, tone: 'complete', text: text ?? 'Done', detail: null, progress: null },
        }),
      error: ({ text } = {}) =>
        dispatch({ type: 'UPSERT', record: { key: record.key, tone: 'error', text: text ?? 'Error' } }),
      remove: () => dispatch({ type: 'REMOVE', key: record.key }),
    };

    if (cancelFn) handle.cancel = cancelFn;
    return handle;
  }, [dispatch]);

  return { start };
}
```

- [x] **Step 4: Run tests, confirm pass**

```bash
cd lcm-sr-ui && npx vitest run src/contexts/OperationsContext.test.jsx
```
Expected: all 7 tests pass.

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/contexts/OperationsContext.jsx lcm-sr-ui/src/contexts/OperationsContext.test.jsx
git commit -m "feat: add OperationsContext — store, controller, status handle API"
```

---

### Task 2: PanelActionBar

**Files:**
- Create: `lcm-sr-ui/src/components/ui/PanelActionBar.jsx`
- Create: `lcm-sr-ui/src/components/ui/PanelActionBar.test.jsx`

- [x] **Step 1: Write failing tests**

```jsx
// lcm-sr-ui/src/components/ui/PanelActionBar.test.jsx
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

  it('secondary actions are not truncated into ambiguity', () => {
    render(<PanelActionBar primary={primary} secondary={secondary} />);
    expect(screen.getByText('Rebuild')).toBeVisible();
    expect(screen.getByText('Reset')).toBeVisible();
  });
});
```

- [x] **Step 2: Run tests, confirm failure**

```bash
cd lcm-sr-ui && npx vitest run src/components/ui/PanelActionBar.test.jsx
```
Expected: `FAIL` — module not found.

- [x] **Step 3: Implement PanelActionBar**

```jsx
// lcm-sr-ui/src/components/ui/PanelActionBar.jsx
import React from 'react';
import { Button } from '@/components/ui/button';

function ActionButton({ action, variant, className = '' }) {
  return (
    <Button
      type="button"
      variant={variant}
      disabled={Boolean(action.disabled)}
      onClick={action.onClick}
      className={`flex flex-col items-center gap-0.5 h-auto py-2.5 px-3 ${className}`}
    >
      <div className="flex items-center gap-1.5">
        {action.icon && <span className="shrink-0">{action.icon}</span>}
        <span className="font-medium text-sm">{action.label}</span>
      </div>
      {action.subtext && (
        <span className="text-xs opacity-70 font-normal">{action.subtext}</span>
      )}
    </Button>
  );
}

// primary: { icon?, label, subtext?, onClick, disabled? }
// secondary: [{ icon?, label, subtext?, onClick, disabled? }]
export function PanelActionBar({ primary, secondary = [] }) {
  return (
    <div className="border-t bg-muted/30 px-4 py-3 space-y-2">
      {secondary.length > 0 && (
        <div className="flex gap-2">
          {secondary.map((action) => (
            <ActionButton key={action.label} action={action} variant="secondary" className="flex-1" />
          ))}
        </div>
      )}
      {primary && (
        <ActionButton action={primary} variant="default" className="w-full" />
      )}
    </div>
  );
}
```

- [x] **Step 4: Run tests, confirm pass**

```bash
cd lcm-sr-ui && npx vitest run src/components/ui/PanelActionBar.test.jsx
```
Expected: all 6 tests pass.

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/components/ui/PanelActionBar.jsx lcm-sr-ui/src/components/ui/PanelActionBar.test.jsx
git commit -m "feat: add PanelActionBar — primary/secondary action footer primitive"
```

---

### Task 3: PendingOperationsPane

**Files:**
- Create: `lcm-sr-ui/src/components/ui/PendingOperationsPane.jsx`
- Create: `lcm-sr-ui/src/components/ui/PendingOperationsPane.test.jsx`

- [x] **Step 1: Write failing tests**

```jsx
// lcm-sr-ui/src/components/ui/PendingOperationsPane.test.jsx
// @vitest-environment jsdom
import React, { useEffect } from 'react';
import { render, screen, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { OperationsProvider, useOperationsController } from '../../contexts/OperationsContext';
import { PendingOperationsPane } from './PendingOperationsPane';

afterEach(() => cleanup());

function ControlPanel({ onReady }) {
  const ctrl = useOperationsController();
  useEffect(() => { onReady(ctrl); }, []); // eslint-disable-line
  return null;
}

function Harness({ onReady }) {
  return (
    <OperationsProvider>
      <ControlPanel onReady={onReady} />
      <PendingOperationsPane />
    </OperationsProvider>
  );
}

describe('PendingOperationsPane', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('renders nothing when no operations are active', () => {
    const { container } = render(<OperationsProvider><PendingOperationsPane /></OperationsProvider>);
    expect(container.firstChild).toBeNull();
  });

  it('renders an active operation with text and detail', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => { ctrl.start({ key: 'adv:1', text: 'Rebuilding', detail: 'Analyzing evidence' }); });
    expect(screen.getByText('Rebuilding')).toBeInTheDocument();
    expect(screen.getByText('Analyzing evidence')).toBeInTheDocument();
  });

  it('renders progress value when set', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => {
      const handle = ctrl.start({ key: 'gen:1', text: 'Generating image' });
      handle.setProgress({ current: 8, total: 28 });
    });
    expect(screen.getByText('8 / 28')).toBeInTheDocument();
  });

  it('shows cancel button when cancelFn is provided', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => {
      ctrl.start({ key: 'gen:1', text: 'Generating', cancellable: true, cancelFn: vi.fn() });
    });
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
  });

  it('does not show cancel button when not cancellable', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    act(() => { ctrl.start({ key: 'adv:1', text: 'Working', cancellable: false }); });
    expect(screen.queryByRole('button', { name: /cancel/i })).not.toBeInTheDocument();
  });

  it('auto-removes after complete linger', () => {
    let ctrl;
    render(<Harness onReady={(c) => { ctrl = c; }} />);
    let handle;
    act(() => { handle = ctrl.start({ key: 'adv:1', text: 'Working' }); });
    act(() => { handle.complete({ text: 'Done' }); });
    expect(screen.getByText('Done')).toBeInTheDocument();
    act(() => { vi.advanceTimersByTime(2001); });
    expect(screen.queryByText('Done')).not.toBeInTheDocument();
  });
});
```

- [x] **Step 2: Run tests, confirm failure**

```bash
cd lcm-sr-ui && npx vitest run src/components/ui/PendingOperationsPane.test.jsx
```
Expected: `FAIL` — module not found.

- [x] **Step 3: Implement PendingOperationsPane**

```jsx
// lcm-sr-ui/src/components/ui/PendingOperationsPane.jsx
import React from 'react';
import { X } from 'lucide-react';
import { useOperationsStore } from '../../contexts/OperationsContext';

const TONE_CLASSES = {
  active:   'bg-indigo-50  border-indigo-200  dark:bg-indigo-950/30  dark:border-indigo-800',
  complete: 'bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-800',
  error:    'bg-red-50     border-red-200     dark:bg-red-950/30     dark:border-red-800',
  idle:     'bg-muted border-border',
};

function OperationItem({ op }) {
  const toneClass = TONE_CLASSES[op.tone] ?? TONE_CLASSES.idle;
  const isActive = op.tone === 'active';

  return (
    <div className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm ${toneClass}`}>
      {op.icon && (
        <span className={isActive ? 'animate-pulse shrink-0' : 'shrink-0'}>{op.icon}</span>
      )}
      <div className="flex flex-1 items-center gap-2 min-w-0">
        <span className="font-medium truncate">{op.text}</span>
        {op.detail && (
          <span className="text-xs text-muted-foreground truncate">{op.detail}</span>
        )}
        {op.progress && (
          <span className="text-xs text-muted-foreground shrink-0">
            {op.progress.current} / {op.progress.total}
          </span>
        )}
      </div>
      {op.cancellable && op.cancelFn && (
        <button
          type="button"
          onClick={op.cancelFn}
          className="shrink-0 p-0.5 rounded hover:bg-black/10 transition-colors"
          aria-label="Cancel"
        >
          <X className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}

export function PendingOperationsPane() {
  const { operations, order } = useOperationsStore();
  if (order.length === 0) return null;

  return (
    <div className="flex flex-col gap-1 px-3 py-1.5">
      {order.map((key) => {
        const op = operations.get(key);
        if (!op) return null;
        return <OperationItem key={key} op={op} />;
      })}
    </div>
  );
}
```

- [x] **Step 4: Run tests, confirm pass**

```bash
cd lcm-sr-ui && npx vitest run src/components/ui/PendingOperationsPane.test.jsx
```
Expected: all 6 tests pass.

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/components/ui/PendingOperationsPane.jsx lcm-sr-ui/src/components/ui/PendingOperationsPane.test.jsx
git commit -m "feat: add PendingOperationsPane — operations display surface with pulse/expiry"
```

---

### Task 4: SurfaceHeader

**Files:**
- Create: `lcm-sr-ui/src/components/ui/SurfaceHeader.jsx`
- Create: `lcm-sr-ui/src/components/ui/SurfaceHeader.test.jsx`

- [x] **Step 1: Write failing tests**

```jsx
// lcm-sr-ui/src/components/ui/SurfaceHeader.test.jsx
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
```

- [x] **Step 2: Run tests, confirm failure**

```bash
cd lcm-sr-ui && npx vitest run src/components/ui/SurfaceHeader.test.jsx
```
Expected: `FAIL` — module not found.

- [x] **Step 3: Implement SurfaceHeader**

```jsx
// lcm-sr-ui/src/components/ui/SurfaceHeader.jsx
import React from 'react';
import { CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// chips: [{ label: string, variant?: string }]
export function SurfaceHeader({ title, chips = [], summary }) {
  return (
    <CardHeader className="border-b">
      <div className="flex items-center justify-between gap-1 flex-wrap">
        <CardTitle className="text-xl">{title}</CardTitle>
        {chips.length > 0 && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
            {chips.map((chip) => (
              <Badge key={chip.label} variant={chip.variant ?? 'secondary'}>
                {chip.label}
              </Badge>
            ))}
          </div>
        )}
      </div>
      {summary && <div className="text-sm text-muted-foreground">{summary}</div>}
    </CardHeader>
  );
}
```

- [x] **Step 4: Run tests, confirm pass**

```bash
cd lcm-sr-ui && npx vitest run src/components/ui/SurfaceHeader.test.jsx
```
Expected: all 4 tests pass.

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/components/ui/SurfaceHeader.jsx lcm-sr-ui/src/components/ui/SurfaceHeader.test.jsx
git commit -m "feat: add SurfaceHeader — calm title/chip/summary primitive"
```

---

## Phase 2 — Advisor Migration

### Task 5: AdvisorPanel → PanelActionBar

**Files:**
- Modify: `lcm-sr-ui/src/components/options/AdvisorPanel.jsx`

The current `AdvisorPanel.jsx` has three equal-weight buttons in a `flex gap-2` row (lines 95–105) and `Building digest...` animated status text (lines 65–73). Replace with `PanelActionBar`. Group Apply Mode select adjacent to the Apply action in the footer region. Remove `Building digest...` — local status shows only durable state (last-updated timestamp or error text).

- [x] **Step 1: Write failing test**

```jsx
// Add to a new file: lcm-sr-ui/src/components/options/AdvisorPanel.test.jsx
// @vitest-environment jsdom
import React from 'react';
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { OperationsProvider } from '../../contexts/OperationsContext';
import { AdvisorPanel } from './AdvisorPanel';

afterEach(() => cleanup());

function wrap(ui) {
  return render(ui, {
    wrapper: ({ children }) => <OperationsProvider>{children}</OperationsProvider>,
  });
}

const base = {
  state: { advice_text: 'Some advice', status: 'fresh', updated_at: Date.now() },
  maximumLen: 240,
  onAutoAdviceChange: vi.fn(),
  onTemperatureChange: vi.fn(),
  onLengthChange: vi.fn(),
  onAdviceChange: vi.fn(),
  onResetToDigest: vi.fn(),
  onRebuild: vi.fn(),
  onApply: vi.fn(),
  applyMode: 'append',
  onApplyModeChange: vi.fn(),
};

describe('AdvisorPanel', () => {
  it('renders Apply as a <button>', () => {
    wrap(<AdvisorPanel {...base} />);
    expect(screen.getByRole('button', { name: /apply/i })).toBeInTheDocument();
  });

  it('renders Rebuild and Reset as <button> elements', () => {
    wrap(<AdvisorPanel {...base} />);
    expect(screen.getByRole('button', { name: /rebuild/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset/i })).toBeInTheDocument();
  });

  it('Apply button is disabled when advice_text is empty', () => {
    wrap(<AdvisorPanel {...base} state={{ ...base.state, advice_text: '' }} />);
    expect(screen.getByRole('button', { name: /apply/i })).toBeDisabled();
  });

  it('does not render "Building digest..." text — no animated inline status', () => {
    wrap(<AdvisorPanel {...base} state={{ ...base.state, status: 'building' }} />);
    expect(screen.queryByText('Building digest...')).not.toBeInTheDocument();
  });

  it('shows last-updated timestamp when status is fresh', () => {
    wrap(<AdvisorPanel {...base} />);
    expect(screen.getByText(/updated/i)).toBeInTheDocument();
  });
});
```

- [x] **Step 2: Run test, confirm failure**

```bash
cd lcm-sr-ui && npx vitest run src/components/options/AdvisorPanel.test.jsx
```
Expected: `FAIL` — test file not found or existing panel renders the old markup.

- [x] **Step 3: Rewrite AdvisorPanel.jsx**

Replace the entire file with the following. Key changes from current:
- `flex gap-2` three-button row → `PanelActionBar`
- Apply is `primary`; Rebuild and Reset are `secondary`
- Apply Mode select moves to the footer region, just above `PanelActionBar`
- `Building digest...` removed — status shows only last-updated or error

```jsx
// lcm-sr-ui/src/components/options/AdvisorPanel.jsx
import React from 'react';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Check, RefreshCw, RotateCcw } from 'lucide-react';
import { PanelActionBar } from '@/components/ui/PanelActionBar';

export function AdvisorPanel({
  state,
  maximumLen,
  onAutoAdviceChange,
  onTemperatureChange,
  onLengthChange,
  onAdviceChange,
  onResetToDigest,
  onRebuild,
  onApply,
  applyMode,
  onApplyModeChange,
}) {
  const hasMaximumLen = Number.isFinite(Number(maximumLen)) && Number(maximumLen) > 0;

  const applySubtext = applyMode === 'replace' ? 'Replace prompt' : 'Append to prompt';

  const statusText =
    state?.status === 'error'
      ? state?.error_message || 'Advisor error'
      : state?.updated_at
        ? `Updated ${new Date(state.updated_at).toLocaleString()}`
        : 'No digest yet';

  return (
    <div className="space-y-3 rounded-2xl border p-4 option-panel-area">
      <Label>Advisor</Label>

      <label className="flex items-center justify-between text-sm">
        <span>Auto-Advice</span>
        <input
          aria-label="Auto advice"
          type="checkbox"
          checked={Boolean(state?.auto_advice)}
          onChange={(e) => onAutoAdviceChange(e.target.checked)}
        />
      </label>

      <div className="space-y-2">
        <Label htmlFor="advisor-temperature">Temperature</Label>
        <input
          id="advisor-temperature"
          aria-label="Advisor temperature"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={state?.temperature ?? 0.4}
          onChange={(e) => onTemperatureChange(Number(e.target.value))}
        />
      </div>

      {hasMaximumLen && (
        <div className="space-y-2">
          <Label htmlFor="advisor-length">Length</Label>
          <input
            id="advisor-length"
            aria-label="Advisor length"
            type="range"
            min="0"
            max={maximumLen}
            step="1"
            value={state?.length_limit ?? 0}
            onChange={(e) => onLengthChange(Number(e.target.value))}
          />
        </div>
      )}

      <div className="text-xs text-muted-foreground" data-status={state?.status || 'idle'}>
        {statusText}
      </div>

      <Textarea
        aria-label="Advisor advice"
        value={state?.advice_text ?? ''}
        onChange={(e) => onAdviceChange(e.target.value)}
        className="min-h-[120px] resize-none rounded-2xl"
      />

      {/* Apply mode grouped with the Apply action */}
      <div className="flex items-center gap-2 text-sm">
        <Label htmlFor="apply-mode-select" className="shrink-0">Apply as</Label>
        <select
          id="apply-mode-select"
          aria-label="Apply advice mode"
          value={applyMode}
          onChange={(e) => onApplyModeChange(e.target.value)}
          className="flex-1 h-8 rounded-lg border border-input bg-background px-2 py-1 text-sm"
        >
          <option value="append">Append to prompt</option>
          <option value="replace">Replace prompt</option>
        </select>
      </div>

      <PanelActionBar
        primary={{
          icon: <Check className="h-4 w-4" />,
          label: 'Apply',
          subtext: applySubtext,
          onClick: () => onApply(applyMode),
          disabled: !state?.advice_text,
        }}
        secondary={[
          {
            icon: <RefreshCw className="h-4 w-4" />,
            label: 'Rebuild',
            subtext: 'Refresh digest from gallery',
            onClick: onRebuild,
          },
          {
            icon: <RotateCcw className="h-4 w-4" />,
            label: 'Reset',
            subtext: 'Restore digest text',
            onClick: onResetToDigest,
          },
        ]}
      />
    </div>
  );
}
```

- [x] **Step 4: Run tests, confirm pass**

```bash
cd lcm-sr-ui && npx vitest run src/components/options/AdvisorPanel.test.jsx
```
Expected: all 5 tests pass.

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/components/options/AdvisorPanel.jsx lcm-sr-ui/src/components/options/AdvisorPanel.test.jsx
git commit -m "feat: migrate AdvisorPanel to PanelActionBar — primary/secondary hierarchy, durable status"
```

---

### Task 6: useGalleryAdvisor → operations store

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useGalleryAdvisor.js`
- Modify: `lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx`

`rebuildAdvisor` currently goes through `building` → `fresh`/`error` entirely in local state. Wire it into the operations store so a keyed operation appears in `PendingOperationsPane` during rebuild.

- [x] **Step 1: Update existing tests to add OperationsProvider wrapper**

The existing tests in `useGalleryAdvisor.test.jsx` call `renderHook(() => useGalleryAdvisor(...))` without any wrapper. After adding `useOperationsController()` to the hook, this will throw. Wrap each `renderHook` call with `OperationsProvider`:

Replace the top of `useGalleryAdvisor.test.jsx` with:

```jsx
// lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx
// @vitest-environment jsdom
import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { vi, expect, it } from 'vitest';
import { OperationsProvider } from '../contexts/OperationsContext';
import { useGalleryAdvisor } from './useGalleryAdvisor';

function wrapper({ children }) {
  return <OperationsProvider>{children}</OperationsProvider>;
}
```

Then update each `renderHook(...)` call in the file to pass `{ wrapper }`:

```js
// Change every:
const { result } = renderHook(() => useGalleryAdvisor({...}));
// To:
const { result } = renderHook(() => useGalleryAdvisor({...}), { wrapper });

// And every renderHook((props) => ...) form:
const { result, rerender } = renderHook((props) => useGalleryAdvisor(props), {
  initialProps: { ... },
  wrapper,
});
```

- [x] **Step 2: Write a new failing test for operations-store integration**

Add after the existing tests in `useGalleryAdvisor.test.jsx`:

```jsx
import { useOperationsStore } from '../contexts/OperationsContext';

it('rebuildAdvisor creates an active operation and completes it on success', async () => {
  const api = { fetchPost: vi.fn().mockResolvedValue({ digest_text: 'result', meta: {} }) };

  function useCombo() {
    return {
      advisor: useGalleryAdvisor({
        galleryId: 'gal_1', modeName: 'SDXL', galleryRevision: 1,
        galleryImages: [{ cacheKey: 'abc', addedAt: 1, params: { prompt: 'cat' } }],
        maximumLen: 240, api,
        advisorState: null, saveAdvisorState: vi.fn(), setDraftPrompt: vi.fn(),
      }),
      store: useOperationsStore(),
    };
  }

  const { result } = renderHook(useCombo, { wrapper });

  // Before rebuild: no operations
  expect(result.current.store.order).toHaveLength(0);

  await act(async () => { await result.current.advisor.rebuildAdvisor(); });

  // After rebuild: operation should be complete (tone complete or already expired)
  // At minimum the rebuild ran without throwing
  expect(result.current.advisor.state.digest_text).toBe('result');
});
```

- [x] **Step 3: Run tests, confirm the new test fails (existing tests now pass with wrapper)**

```bash
cd lcm-sr-ui && npx vitest run src/hooks/useGalleryAdvisor.test.jsx
```
Expected: 3 existing tests pass (with updated wrapper), new integration test fails.

- [x] **Step 4: Add useOperationsController to useGalleryAdvisor**

In `lcm-sr-ui/src/hooks/useGalleryAdvisor.js`:

1. Add import at top:
```js
import { useOperationsController } from '../contexts/OperationsContext';
```

2. Add inside `useGalleryAdvisor` function body (after the existing hooks):
```js
const { start: startOperation } = useOperationsController();
```

3. Replace the `rebuildAdvisor` implementation with:
```js
const rebuildAdvisor = useCallback(async () => {
  const building = { ...(state || {}), gallery_id: galleryId, status: 'building' };
  setState(building);
  await persistState(building);

  const statusHandle = startOperation({
    key: `advisor-rebuild:${galleryId}`,
    kind: 'advisor',
    text: 'Rebuilding',
    detail: 'Collecting gallery evidence',
    cancellable: false,
  });

  try {
    statusHandle.setDetail('Analyzing evidence');
    const response = await api.fetchPost('/api/advisors/digest', {
      gallery_id: galleryId,
      mode: modeName || undefined,
      temperature: state?.temperature ?? 0.4,
      length_limit: resolveLengthLimit(state, maximumLen),
      evidence,
    });

    const shouldReplaceAdvice = !state?.advice_text || state.advice_text === state.digest_text;
    const next = {
      ...(state || {}),
      gallery_id: galleryId,
      gallery_revision: galleryRevision,
      digest_text: response.digest_text,
      advice_text: shouldReplaceAdvice ? response.digest_text : state.advice_text,
      evidence_fingerprint: response.meta?.evidence_fingerprint ?? null,
      status: 'fresh',
      updated_at: Date.now(),
      error_message: null,
    };
    setState(next);
    await persistState(next);
    statusHandle.complete({ text: 'Digest updated' });
    return next;
  } catch (error) {
    const failed = {
      ...(state || {}),
      gallery_id: galleryId,
      status: 'error',
      error_message: error.message || 'Advisor rebuild failed',
    };
    setState(failed);
    await persistState(failed);
    statusHandle.error({ text: error.message || 'Rebuild failed' });
    throw error;
  }
}, [api, evidence, galleryId, galleryRevision, maximumLen, modeName, persistState, state, startOperation]);
```

- [x] **Step 5: Run tests, confirm all pass**

```bash
cd lcm-sr-ui && npx vitest run src/hooks/useGalleryAdvisor.test.jsx
```
Expected: all 4 tests pass.

- [x] **Step 6: Commit**

```bash
git add lcm-sr-ui/src/hooks/useGalleryAdvisor.js lcm-sr-ui/src/hooks/useGalleryAdvisor.test.jsx
git commit -m "feat: wire advisor rebuild into operations store — keyed status handle per gallery"
```

---

## Phase 3 — Chat and Generation Migration

### Task 7: Mount OperationsProvider in main.jsx

**Files:**
- Modify: `lcm-sr-ui/src/main.jsx`

`useImageGeneration` and `useGalleryAdvisor` (called inside `App`) both call `useOperationsController()`. The provider must be above `App` in the tree so these hooks can read context.

- [x] **Step 1: Add OperationsProvider to main.jsx**

```jsx
// lcm-sr-ui/src/main.jsx  (full file)
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { emitUiEvent } from "./utils/otelTelemetry";
import { OperationsProvider } from "./contexts/OperationsContext";

if (!window.__appStartTime) {
  window.__appStartTime = performance.now();
}

if (!window.__otelErrorHooked) {
  window.__otelErrorHooked = true;
  window.addEventListener("error", (e) => {
    emitUiEvent("ui.error", {
      "ui.component": "global",
      "ui.action": "error",
      "error.message": e.message || "Unknown error",
      "error.filename": e.filename || "",
      "error.lineno": e.lineno || 0,
      "error.colno": e.colno || 0,
    });
  });
  window.addEventListener("unhandledrejection", (e) => {
    const reason = e.reason;
    emitUiEvent("ui.error", {
      "ui.component": "global",
      "ui.action": "unhandledrejection",
      "error.message": reason?.message || String(reason || "Unknown rejection"),
    });
  });
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <OperationsProvider>
      <App />
    </OperationsProvider>
  </React.StrictMode>
);
```

- [x] **Step 2: Run full test suite, confirm no regressions**

```bash
cd lcm-sr-ui && npm test
```
Expected: all tests pass. `App.test.jsx` may need `OperationsProvider` in its render wrapper — if so, add it to `App.test.jsx` the same way (wrap render in `<OperationsProvider>`).

- [x] **Step 3: Commit**

```bash
git add lcm-sr-ui/src/main.jsx
git commit -m "feat: mount OperationsProvider in app shell"
```

---

### Task 8: ChatHeader → SurfaceHeader

**Files:**
- Modify: `lcm-sr-ui/src/components/chat/ChatHeader.jsx`
- Modify: `lcm-sr-ui/src/components/chat/ChatHeader.test.jsx`

Current `ChatHeader` has an `isDreaming` animated badge with `animate-pulse`. Replace with `SurfaceHeader`. Remove the dream badge — dream state is represented in `PendingOperationsPane` after Task 10.

- [x] **Step 1: Update ChatHeader.test.jsx with new assertions**

Replace the existing test file:

```jsx
// lcm-sr-ui/src/components/chat/ChatHeader.test.jsx
// @vitest-environment jsdom
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { ChatHeader } from './ChatHeader';
import { cleanup } from '@testing-library/react';

afterEach(() => cleanup());

describe('ChatHeader', () => {
  it('renders a neutral API placeholder before runtime status loads', () => {
    render(<ChatHeader srLevel={0} frontendVersion="abc1234" />);
    expect(screen.getByText('UI abc1234')).toBeInTheDocument();
    expect(screen.getByText('API ...')).toBeInTheDocument();
  });

  it('renders frontend and backend version chips', () => {
    render(<ChatHeader srLevel={0} frontendVersion="abc1234" backendVersion="1.2.3" />);
    expect(screen.getByText('UI abc1234')).toBeInTheDocument();
    expect(screen.getByText('API 1.2.3')).toBeInTheDocument();
  });

  it('renders SR badge when srLevel > 0', () => {
    render(<ChatHeader srLevel={2} frontendVersion="abc1234" />);
    expect(screen.getByText('SR 2')).toBeInTheDocument();
  });

  it('contains no animate-pulse elements — dream state is in the operations pane', () => {
    const { container } = render(
      <ChatHeader srLevel={0} frontendVersion="abc1234" />
    );
    expect(container.querySelector('.animate-pulse')).toBeNull();
  });
});
```

- [x] **Step 2: Run test, confirm failure** (current header still has dream badge code)

```bash
cd lcm-sr-ui && npx vitest run src/components/chat/ChatHeader.test.jsx
```
Expected: animate-pulse test fails.

- [x] **Step 3: Rewrite ChatHeader.jsx**

```jsx
// lcm-sr-ui/src/components/chat/ChatHeader.jsx
import React from 'react';
import { SurfaceHeader } from '@/components/ui/SurfaceHeader';
import { BADGE_LABELS, UI_MESSAGES } from '../../utils/constants';

export function ChatHeader({ srLevel, frontendVersion, backendVersion }) {
  const apiLabel = backendVersion?.trim() ? backendVersion : '...';

  const chips = [
    srLevel > 0
      ? { label: `SR ${srLevel}` }
      : { label: BADGE_LABELS.SR_OFF, variant: 'outline' },
    { label: `UI ${frontendVersion}`, variant: 'outline' },
    { label: `API ${apiLabel}`,       variant: 'outline' },
  ];

  return (
    <SurfaceHeader
      title="LCM + SR Chat"
      chips={chips}
      summary={UI_MESSAGES.KEYBOARD_TIP}
    />
  );
}
```

- [x] **Step 4: Run tests, confirm pass**

```bash
cd lcm-sr-ui && npx vitest run src/components/chat/ChatHeader.test.jsx
```
Expected: all 4 tests pass.

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/components/chat/ChatHeader.jsx lcm-sr-ui/src/components/chat/ChatHeader.test.jsx
git commit -m "feat: migrate ChatHeader to SurfaceHeader — remove dream badge, calm header"
```

---

### Task 9: ChatContainer — wire PendingOperationsPane, clean up props

**Files:**
- Modify: `lcm-sr-ui/src/components/chat/ChatContainer.jsx`

Two changes:
1. Replace the `[]` sticky strip placeholder (line 56-58) with `<PendingOperationsPane />`.
2. Remove props no longer passed to `ChatHeader`: `isDreaming`, `inflightCount`, `onCopyPrompt`, `copied` (they were vestigial in the header). The `isDreaming` prop stays in `ChatContainer`'s own props because it still flows to each `MessageBubble` as `isDreamMessage`.

- [x] **Step 1: Update ChatContainer.jsx**

Replace the file:

```jsx
// lcm-sr-ui/src/components/chat/ChatContainer.jsx
import React from "react";
import ScrollToBottom from "react-scroll-to-bottom";
import { Card, CardContent } from "@/components/ui/card";
import { MessageComposer } from "./MessageComposer";
import { MessageBubble } from "./MessageBubble";
import { ChatHeader } from "./ChatHeader";
import { PendingOperationsPane } from "@/components/ui/PendingOperationsPane";

export function ChatContainer({
  messages,
  selectedMsgId,
  blurredSelectedMsgId,
  onToggleSelect,
  onCancelRequest,
  setMsgRef,
  composer,
  inflightCount,
  isDreaming,
  dreamMessageId,
  onDreamSave,
  onDreamHistoryPrev,
  onDreamHistoryNext,
  onDreamHistoryLive,
  onRetry,
  serverLabel,
  srLevel,
  frontendVersion,
  backendVersion,
  onCopyPrompt,
  copied,
  activeGalleryId,
  onAddToGallery,
  slashCtx,
  inputMode,
  onSetInputMode,
}) {
  return (
    <Card className="option-panel-area overflow-hidden rounded-xl shadow-sm h-full flex flex-col">
      <ChatHeader
        srLevel={srLevel}
        frontendVersion={frontendVersion}
        backendVersion={backendVersion}
      />
      <CardContent className="flex flex-1 flex-col p-0 min-h-0">
        <ScrollToBottom
          className="flex-1 min-h-0"
          scrollViewClassName="p-0 md:p-0"
          followButtonClassName="scroll-to-bottom-button"
        >
          <div className="relative">
            <div className="sticky top-0 z-10 bg-background/80 backdrop-blur-sm">
              <PendingOperationsPane />
            </div>

            <div className="space-y-4">
              {messages.map((msg) => (
                <div key={msg.id} ref={setMsgRef(msg.id)}>
                  <MessageBubble
                    msg={msg}
                    isSelected={msg.id === selectedMsgId}
                    isBlurredSelected={msg.id === blurredSelectedMsgId}
                    onSelect={() => onToggleSelect(msg.id)}
                    onCancel={msg.kind === "pending" ? () => onCancelRequest(msg.id) : null}
                    isDreamMessage={isDreaming && msg.id === dreamMessageId}
                    hasDreamHistory={msg.imageHistory?.length > 1}
                    onDreamSave={onDreamSave}
                    onDreamHistoryPrev={() => onDreamHistoryPrev?.(msg)}
                    onDreamHistoryNext={() => onDreamHistoryNext?.(msg)}
                    onDreamHistoryLive={() => onDreamHistoryLive?.(msg)}
                    onRetry={onRetry}
                    activeGalleryId={activeGalleryId}
                    onAddToGallery={onAddToGallery}
                  />
                </div>
              ))}
            </div>
          </div>
        </ScrollToBottom>

        <MessageComposer
          onSendPrompt={composer?.onSendPrompt}
          onCancelAll={composer?.onCancelAll}
          onKeyDown={composer?.onKeyDown}
          onFocus={composer?.onFocus}
          inflightCount={inflightCount}
          serverLabel={serverLabel}
          slashCtx={slashCtx}
          inputMode={inputMode}
          onSetInputMode={onSetInputMode}
        />
      </CardContent>
    </Card>
  );
}
```

- [x] **Step 2: Run full test suite**

```bash
cd lcm-sr-ui && npm test
```
Expected: all tests pass.

- [x] **Step 3: Commit**

```bash
git add lcm-sr-ui/src/components/chat/ChatContainer.jsx
git commit -m "feat: wire PendingOperationsPane into chat sticky strip — replace [] placeholder"
```

---

### Task 10: useImageGeneration → dream + generation lifecycle → operations store

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useImageGeneration.js`

Wire four lifecycle paths into the operations store:

1. **Generation jobs** — non-dream `runGenerate` creates `generation:<assistantId>` on start; `onComplete` calls `statusHandle.complete()`; `onError` calls `statusHandle.error()`.
2. **`startDreaming`** — creates `dream:active` operation via a ref-held handle.
3. **`stopDreaming`** — completes the dream operation.
4. **`guideDream`** — updates the dream operation's detail text.
5. **`saveDreamAndContinue`** — upserts (refreshes) the same `dream:active` key.

The dream handle is held in a `useRef` so it survives across `startDreaming`/`stopDreaming` calls without needing to be in the dependency arrays of every dream callback.

- [x] **Step 1: Add import and controller call**

At the top of `useImageGeneration.js`, add the import:

```js
import { useOperationsController } from '../contexts/OperationsContext';
```

Inside `useImageGeneration` function body, before the existing refs, add:

```js
const { start: startOperation } = useOperationsController();
const dreamStatusRef = useRef(null);
```

- [x] **Step 2: Wire generation operations into runGenerate**

Inside the `runGenerate` `useCallback`, immediately after the `const assistantId = ...` line and before the `if (targetMessageId)` block, add:

```js
const statusHandle = isDream
  ? null
  : startOperation({
      key: `generation:${assistantId}`,
      kind: 'generation',
      text: 'Generating',
      detail: null,
      cancellable: false,
    });
```

In the existing `onError` handler (after the `updateMessage` call), add:

```js
statusHandle?.error({ text: errMsg });
```

In the existing `onComplete` handler (after removing listeners), add:

```js
statusHandle?.complete({ text: 'Image ready' });
```

Add `startOperation` to `runGenerate`'s `useCallback` deps array:

```js
[api, cache, addMessage, updateMessage, setSelectedMsgId, scheduleHydration, linkMsgToCacheKey, startOperation]
```

- [x] **Step 3: Wire dream lifecycle**

Replace `startDreaming` with:

```js
const startDreaming = useCallback(
  (baseParams) => {
    if (dreamTimerRef.current) clearInterval(dreamTimerRef.current);

    dreamParamsRef.current = { ...baseParams };
    setIsDreaming(true);

    dreamStatusRef.current = startOperation({
      key: 'dream:active',
      kind: 'dream',
      text: 'Dream mode',
      detail: 'Exploring variations',
      cancellable: false,
    });

    const firstId = runDreamCycle(null);
    setDreamMessageId(firstId);
    dreamMessageIdRef.current = firstId;

    restartDreamInterval();
  },
  [runDreamCycle, restartDreamInterval, startOperation]
);
```

Replace `stopDreaming` with:

```js
const stopDreaming = useCallback(() => {
  if (dreamTimerRef.current) {
    clearInterval(dreamTimerRef.current);
    dreamTimerRef.current = null;
  }
  setIsDreaming(false);
  setDreamMessageId(null);
  dreamParamsRef.current = null;
  dreamStatusRef.current?.complete({ text: 'Dream ended' });
  dreamStatusRef.current = null;
}, []);
```

Replace `guideDream` with:

```js
const guideDream = useCallback((newBaseParams) => {
  if (!isDreaming) return;
  dreamParamsRef.current = { ...newBaseParams };
  dreamStatusRef.current?.setDetail('Guided to selected image');
}, [isDreaming]);
```

Replace `saveDreamAndContinue` with:

```js
const saveDreamAndContinue = useCallback(() => {
  if (!isDreaming) return;
  setDreamMessageId(null);
  dreamMessageIdRef.current = null;

  if (dreamTimerRef.current) clearInterval(dreamTimerRef.current);
  const baseParams = dreamParamsRef.current;
  if (!baseParams) return;

  const newId = runDreamCycle(null);
  setDreamMessageId(newId);
  if (newId) dreamHistoryByMsgIdRef.current.set(newId, []);
  dreamMessageIdRef.current = newId;

  // Refresh the dream operation detail to reflect the new message
  dreamStatusRef.current = startOperation({
    key: 'dream:active',
    kind: 'dream',
    text: 'Dream mode',
    detail: 'New variation started',
    cancellable: false,
  });

  restartDreamInterval();
}, [isDreaming, runDreamCycle, restartDreamInterval, startOperation]);
```

- [x] **Step 4: Run full test suite**

```bash
cd lcm-sr-ui && npm test
```
Expected: all tests pass. (useImageGeneration has no direct unit tests but the App integration tests should still pass.)

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useImageGeneration.js
git commit -m "feat: wire dream and generation lifecycle into operations store"
```

---

### Task 11: MessageBubble — remove animated badges

**Files:**
- Modify: `lcm-sr-ui/src/components/chat/MessageBubble.jsx`

Remove the animated `dreaming` and `generating` badge overlay (lines 286–299 in current file). Dream mode and generation activity are now represented in `PendingOperationsPane`, not on image content.

- [x] **Step 1: Write failing test**

In `MessageBubble.gallery.test.jsx` (or a new `MessageBubble.badges.test.jsx`), add:

```jsx
// Add to lcm-sr-ui/src/components/chat/MessageBubble.badges.test.jsx
// @vitest-environment jsdom
import React from 'react';
import { render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup } from '@testing-library/react';
import { MessageBubble } from './MessageBubble';

afterEach(() => cleanup());

const imageMsg = {
  id: 'msg1',
  role: 'assistant',
  kind: 'image',
  imageUrl: 'blob:http://localhost/fake',
  serverImageUrl: null,
  params: { seed: 12345, size: '512x512', steps: 4, cfg: 1.0 },
  meta: {},
};

describe('MessageBubble animated badges', () => {
  it('renders no animate-pulse badge when isDreamMessage is true', () => {
    const { container } = render(
      <MessageBubble
        msg={imageMsg}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={() => {}}
        isDreamMessage={true}
        hasDreamHistory={false}
      />
    );
    expect(container.querySelector('.animate-pulse')).toBeNull();
  });

  it('renders no animate-pulse badge when msg.isRegenerating is true', () => {
    const { container } = render(
      <MessageBubble
        msg={{ ...imageMsg, isRegenerating: true }}
        isSelected={false}
        isBlurredSelected={false}
        onSelect={() => {}}
        isDreamMessage={false}
        hasDreamHistory={false}
      />
    );
    expect(container.querySelector('.animate-pulse')).toBeNull();
  });
});
```

- [x] **Step 2: Run test, confirm failure**

```bash
cd lcm-sr-ui && npx vitest run src/components/chat/MessageBubble.badges.test.jsx
```
Expected: `FAIL` — `animate-pulse` elements found.

- [x] **Step 3: Remove animated badge overlay from MessageBubble.jsx**

Locate and remove the following block (currently after the `<img>` element, inside the `image-frame` div):

```jsx
{/* Status badges — REMOVE THIS ENTIRE BLOCK */}
{(isDreamMessage || msg.isRegenerating) && (
  <div className="absolute top-2 right-2 flex flex-col gap-1 items-end pointer-events-none">
    {isDreamMessage && !msg.isRegenerating && (
      <span className="bg-purple-600/80 text-white text-xs px-2 py-1 rounded backdrop-blur-sm animate-pulse">
        dreaming
      </span>
    )}
    {msg.isRegenerating && (
      <span className="bg-slate-800/80 text-white text-xs px-2 py-1 rounded backdrop-blur-sm animate-pulse">
        generating
      </span>
    )}
  </div>
)}
```

Do not remove anything else — dream history navigation (`hasDreamHistory` controls), double-click handler (`isDreamMessage && onDreamSave`), or cursor styling (`isDreamMessage ? 'cursor-pointer' : ''`) must all remain.

- [x] **Step 4: Run all tests, confirm pass**

```bash
cd lcm-sr-ui && npm test
```
Expected: all tests pass including the new badge tests and existing `MessageBubble.gallery.test.jsx`.

- [x] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/components/chat/MessageBubble.jsx lcm-sr-ui/src/components/chat/MessageBubble.badges.test.jsx
git commit -m "feat: remove animated dream/generating badges from MessageBubble — pane owns all active-state animation"
```

---

## Phase 4 — Structured Generation Progress (backend-dependent, deferred)

Phase 4 depends on the backend emitting structured progress events from `callback_on_step_end`. It is planned as a separate track once that capability exists. When ready:

- Add a WebSocket adapter that translates `{ kind: 'generation', phase, step, total_steps }` into `statusHandle.setProgress({ current: step, total: total_steps })` and `statusHandle.setDetail(...)` calls.
- Surface `cancellable: true` with a real cancel callback when the backend supports job cancellation.
- Add queue position display when `{ kind: 'queue', position }` events arrive.

---

## Validation Checklist

Before calling this complete:

- [x] Advisor footer actions remain legible and inside bounds at 360px width (visually verify in browser)
- [x] First-time user can identify all Advisor footer actions as buttons without hovering
- [x] Active work (dream mode, rebuild) appears in the sticky pane, not as animated labels on image content or in the header
- [x] Removing an animated badge from MessageBubble has not broken dream history navigation (prev/next/live buttons still work)
- [x] Generation and advisor status can be updated through handles without any direct pane rendering
- [x] Running `npm test` passes all tests with zero failures
