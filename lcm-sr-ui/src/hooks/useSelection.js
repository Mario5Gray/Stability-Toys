import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export function useSelection(items) {
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [anchorId, setAnchorId] = useState(null);
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const itemsKey = useMemo(
    () => items.map((it) => it.id).join('|'),
    [items],
  );

  useEffect(() => {
    setSelectedIds(new Set());
    setAnchorId(null);
  }, [itemsKey]);

  const toggle = useCallback((id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setAnchorId((prev) => (prev === id ? prev : id));
  }, []);

  const rangeTo = useCallback((id) => {
    const list = itemsRef.current;
    setAnchorId((prevAnchor) => {
      if (!prevAnchor) {
        setSelectedIds(new Set([id]));
        return id;
      }
      const a = list.findIndex((it) => it.id === prevAnchor);
      const b = list.findIndex((it) => it.id === id);
      if (a === -1 || b === -1) {
        setSelectedIds(new Set([id]));
        return id;
      }
      const [lo, hi] = a < b ? [a, b] : [b, a];
      const next = new Set();
      for (let i = lo; i <= hi; i++) next.add(list[i].id);
      setSelectedIds(next);
      return prevAnchor;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(itemsRef.current.map((it) => it.id)));
  }, []);

  const clear = useCallback(() => {
    setSelectedIds(new Set());
    setAnchorId(null);
  }, []);

  return { selectedIds, anchorId, toggle, rangeTo, selectAll, clear };
}
