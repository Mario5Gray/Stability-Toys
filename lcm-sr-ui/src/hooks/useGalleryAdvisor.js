import { useCallback, useEffect, useMemo, useState } from 'react';
import { buildAdvisorEvidence } from '../utils/advisorEvidence';
import { useOperationsController } from '../contexts/OperationsContext';

function resolveLengthLimit(state, maximumLen) {
  const configured = Number(maximumLen);
  if (!Number.isFinite(configured) || configured <= 0) {
    const fallback = Number(state?.length_limit);
    return Number.isFinite(fallback) && fallback > 0 ? fallback : undefined;
  }
  const requested = Number(state?.length_limit);
  if (Number.isFinite(requested) && requested > 0) {
    return Math.min(requested, configured);
  }
  return configured;
}

export function useGalleryAdvisor({
  galleryId,
  modeName,
  galleryRevision,
  galleryImages,
  maximumLen,
  api,
  advisorState,
  saveAdvisorState,
  setDraftPrompt,
}) {
  const { start: startOperation } = useOperationsController();
  const [state, setState] = useState(advisorState);
  const persistState = useCallback(async (nextState) => {
    if (!saveAdvisorState) return;
    try {
      await saveAdvisorState(nextState);
    } catch (error) {
      console.warn('[useGalleryAdvisor] failed to persist advisor state:', error);
    }
  }, [saveAdvisorState]);

  useEffect(() => {
    if (!galleryId) {
      setState(null);
      return;
    }
    if (!advisorState) return;
    setState((prev) => {
      if (!prev) return advisorState;
      if (prev.gallery_id !== advisorState.gallery_id) return advisorState;
      const prevUpdated = Number(prev.updated_at || 0);
      const nextUpdated = Number(advisorState.updated_at || 0);
      if (nextUpdated > prevUpdated) return advisorState;
      return prev;
    });
  }, [advisorState, galleryId]);

  const evidence = useMemo(
    () => buildAdvisorEvidence(galleryId, galleryImages || []),
    [galleryId, galleryImages],
  );

  useEffect(() => {
    if (!galleryId) return;
    setState((prev) => {
      if (!prev) return prev;
      if ((prev.gallery_revision ?? 0) === galleryRevision || prev.status === 'stale') {
        return prev;
      }
      const next = { ...prev, gallery_revision: galleryRevision, status: 'stale' };
      void persistState(next);
      return next;
    });
  }, [galleryId, galleryRevision, persistState]);

  const rebuildAdvisor = useCallback(async () => {
    const building = { ...(state || {}), gallery_id: galleryId, status: 'building' };
    setState(building);
    await persistState(building);

    const statusHandle = startOperation({
      key: `advisor-rebuild:${galleryId}`,
      text: 'Building digest',
      tone: 'active',
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
      statusHandle.error({ text: error.message || 'Advisor rebuild failed' });
      throw error;
    }
  }, [api, evidence, galleryId, galleryRevision, maximumLen, modeName, persistState, startOperation, state]);

  const applyAdvice = useCallback((mode) => {
    if (!state?.advice_text) return;
    setDraftPrompt((current) => {
      if (mode === 'replace') return state.advice_text;
      return current ? `${current}\n\n${state.advice_text}` : state.advice_text;
    });
  }, [setDraftPrompt, state?.advice_text]);

  return { state, setState, rebuildAdvisor, applyAdvice, evidence };
}
