import { useCallback, useEffect, useMemo, useState } from 'react';
import { buildAdvisorEvidence } from '../utils/advisorEvidence';

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
  galleryRevision,
  galleryImages,
  maximumLen,
  api,
  advisorState,
  saveAdvisorState,
  setDraftPrompt,
}) {
  const [state, setState] = useState(advisorState);

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
    if (!state || !galleryId) return;
    if ((state.gallery_revision ?? 0) !== galleryRevision && state.status !== 'stale') {
      const next = { ...state, gallery_revision: galleryRevision, status: 'stale' };
      setState(next);
      if (saveAdvisorState) {
        void saveAdvisorState(next);
      }
    }
  }, [galleryId, galleryRevision, saveAdvisorState, state]);

  const rebuildAdvisor = useCallback(async () => {
    const building = { ...(state || {}), gallery_id: galleryId, status: 'building' };
    setState(building);
    if (saveAdvisorState) {
      await saveAdvisorState(building);
    }

    try {
      const response = await api.fetchPost('/api/advisors/digest', {
        gallery_id: galleryId,
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
      if (saveAdvisorState) {
        await saveAdvisorState(next);
      }
      return next;
    } catch (error) {
      const failed = {
        ...(state || {}),
        gallery_id: galleryId,
        status: 'error',
        error_message: error.message || 'Advisor rebuild failed',
      };
      setState(failed);
      if (saveAdvisorState) {
        await saveAdvisorState(failed);
      }
      throw error;
    }
  }, [api, evidence, galleryId, galleryRevision, maximumLen, saveAdvisorState, state]);

  const applyAdvice = useCallback((mode) => {
    if (!state?.advice_text) return;
    setDraftPrompt((current) => {
      if (mode === 'replace') return state.advice_text;
      return current ? `${current}\n\n${state.advice_text}` : state.advice_text;
    });
  }, [setDraftPrompt, state?.advice_text]);

  return { state, setState, rebuildAdvisor, applyAdvice, evidence };
}
