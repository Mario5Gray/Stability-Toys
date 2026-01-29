// src/components/ui/NumberStepper.jsx
// src/components/common/NumberStepperDebounced.jsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * NumberStepperDebounced
 *
 * Behavior:
 * - Keeps an internal "draft" value that updates immediately as the user types/clicks.
 * - Calls onCommit ONLY when:
 *    - input blurs
 *    - +/- mouse/touch is released (mouse up / touch end / pointer up)
 *    - user presses Enter
 * - Optional: commits when arrow keys adjust and then key is released (handled via pointer-up for buttons; input keys commit on blur/Enter)
 *
 * Notes:
 * - Use onCommit to update expensive state (e.g., triggers inference refresh, API calls, etc.)
 * - If you still want live updates while typing, use a separate onChange prop (not included here on purpose).
 */
export function NumberStepperDebounced({
  value,
  onCommit,
  step = 1,
  min = -Infinity,
  max = Infinity,
  precision = null, // decimals, or null => integer
  widthClass = "w-24",
  inputClassName = "",
  buttonClassName = "px-2 py-1 rounded bg-indigo-700 hover:bg-zinc-600 text-white",
  disabled = false,
}) {
  const clamp = useCallback(
    (v) => Math.min(max, Math.max(min, v)),
    [min, max]
  );

  const normalize = useCallback(
    (v) => {
      if (!Number.isFinite(v)) return value;
      let out = clamp(v);
      if (precision != null) out = Number(out.toFixed(precision));
      else out = Math.trunc(out);
      return out;
    },
    [clamp, precision, value]
  );

  // Internal draft state
  const [draft, setDraft] = useState(value);

  // Keep draft in sync if parent value changes externally
  useEffect(() => {
    setDraft(value);
  }, [value]);

  const draftRef = useRef(draft);
  useEffect(() => {
    draftRef.current = draft;
  }, [draft]);

  const commit = useCallback(() => {
    const next = normalize(Number(draftRef.current));
    if (next !== value) onCommit(next);
    // Always snap draft to normalized form so UI stays tidy
    setDraft(next);
  }, [normalize, onCommit, value]);

  const bumpDraft = useCallback(
    (dir) => {
      setDraft((prev) => normalize(Number(prev) + dir * step));
    },
    [normalize, step]
  );

  // Commit on pointer up (mouse/touch/pen) anywhere after a button click/drag
  // This ensures: click + hold repeat patterns still end with one commit at release.
  const pendingPointerCommitRef = useRef(false);

  const beginPointerSequence = useCallback(() => {
    if (disabled) return;
    pendingPointerCommitRef.current = true;
    const onUp = () => {
      if (!pendingPointerCommitRef.current) return;
      pendingPointerCommitRef.current = false;
      commit();
      window.removeEventListener("pointerup", onUp, true);
      window.removeEventListener("pointercancel", onUp, true);
    };
    window.addEventListener("pointerup", onUp, true);
    window.addEventListener("pointercancel", onUp, true);
  }, [commit, disabled]);

  const decDisabled = disabled || draft - step < min;
  const incDisabled = disabled || draft + step > max;

  const inputClasses = useMemo(
    () =>
      `${widthClass} text-center px-2 py-1.5 rounded bg-violet-300 border border-zinc-700 text-black text-sm font-mono ${inputClassName}`,
    [widthClass, inputClassName]
  );

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        disabled={decDisabled}
        className={buttonClassName}
        onPointerDown={() => {
          beginPointerSequence();
          bumpDraft(-1);
        }}
      >
        –
      </button>

      <input
        type="number"
        step={step}
        value={draft}
        disabled={disabled}
        onChange={(e) => {
          // Update draft only; do NOT commit
          const n = Number(e.target.value);
          if (!Number.isFinite(n) && e.target.value !== "") return;
          setDraft(e.target.value === "" ? "" : n);
        }}
        onBlur={() => {
          if (disabled) return;
          commit();
        }}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === "Enter") {
            e.currentTarget.blur(); // triggers commit
          }
          // Optional: allow arrow keys to change draft without commit
          if (e.key === "ArrowUp") {
            e.preventDefault();
            setDraft((prev) => normalize(Number(prev) + step));
          }
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setDraft((prev) => normalize(Number(prev) - step));
          }
        }}
        className={inputClasses}
      />

      <button
        type="button"
        disabled={incDisabled}
        className={buttonClassName}
        onPointerDown={() => {
          beginPointerSequence();
          bumpDraft(1);
        }}
      >
        +
      </button>
    </div>
  );
}

export function NumberStepper({
  value,
  onChange,
  step = 1,
  min = -Infinity,
  max = Infinity,
  precision = null, // number of decimals, or null for integer
  widthClass = "w-24",
}) {
  const clamp = (v) => Math.min(max, Math.max(min, v));

  const format = (v) =>
    precision != null ? Number(v.toFixed(precision)) : Math.trunc(v);

  const bump = (dir) => {
    const next = clamp(value + dir * step);
    onChange(format(next));
  };

  const handleInput = (e) => {
    const n = Number(e.target.value);
    if (!Number.isFinite(n)) return;
    onChange(clamp(format(n)));
  };

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={() => bump(-1)}
        className="px-2 py-1 rounded bg-indigo-700 hover:bg-zinc-600 text-white"
      >
        –
      </button>

      <input
        type="number"
        step={step}
        value={value}
        onChange={handleInput}
        className={`${widthClass} text-center px-2 py-1.5 rounded bg-violet-300 border border-zinc-700 text-black text-sm font-mono`}
      />

      <button
        type="button"
        onClick={() => bump(1)}
        className="px-2 py-1 rounded bg-indigo-700 hover:bg-zinc-600 text-white"
      >
        +
      </button>
    </div>
  );
}