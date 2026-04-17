// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { useSelection } from './useSelection';

function items(n) {
  return Array.from({ length: n }, (_, i) => ({ id: `id_${i}` }));
}

describe('useSelection', () => {
  it('starts empty with null anchor', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    expect(result.current.selectedIds.size).toBe(0);
    expect(result.current.anchorId).toBeNull();
  });

  it('toggle adds then removes and updates anchor on add', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    act(() => { result.current.toggle('id_1'); });
    expect(result.current.selectedIds.has('id_1')).toBe(true);
    expect(result.current.anchorId).toBe('id_1');
    act(() => { result.current.toggle('id_1'); });
    expect(result.current.selectedIds.has('id_1')).toBe(false);
  });

  it('rangeTo selects contiguous ids from anchor to target in item order', () => {
    const { result } = renderHook(() => useSelection(items(5)));
    act(() => { result.current.toggle('id_1'); });
    act(() => { result.current.rangeTo('id_3'); });
    expect([...result.current.selectedIds].sort()).toEqual(['id_1', 'id_2', 'id_3']);
  });

  it('rangeTo works in reverse direction', () => {
    const { result } = renderHook(() => useSelection(items(5)));
    act(() => { result.current.toggle('id_3'); });
    act(() => { result.current.rangeTo('id_1'); });
    expect([...result.current.selectedIds].sort()).toEqual(['id_1', 'id_2', 'id_3']);
  });

  it('rangeTo with no anchor falls back to single-select', () => {
    const { result } = renderHook(() => useSelection(items(5)));
    act(() => { result.current.rangeTo('id_2'); });
    expect([...result.current.selectedIds]).toEqual(['id_2']);
  });

  it('selectAll selects every visible item', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    act(() => { result.current.selectAll(); });
    expect(result.current.selectedIds.size).toBe(3);
  });

  it('clear resets selection and anchor', () => {
    const { result } = renderHook(() => useSelection(items(3)));
    act(() => { result.current.toggle('id_1'); });
    act(() => { result.current.clear(); });
    expect(result.current.selectedIds.size).toBe(0);
    expect(result.current.anchorId).toBeNull();
  });

  it('resets when items array identity changes to a different gallery', () => {
    const first = items(3);
    const { result, rerender } = renderHook(({ list }) => useSelection(list), {
      initialProps: { list: first },
    });
    act(() => { result.current.toggle('id_1'); });
    expect(result.current.selectedIds.size).toBe(1);
    rerender({ list: [{ id: 'other_1' }] });
    expect(result.current.selectedIds.size).toBe(0);
  });
});
