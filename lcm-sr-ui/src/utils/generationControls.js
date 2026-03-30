export const CUSTOM_NEGATIVE_PROMPT_ID = '__custom__';

export function getActiveMode(config, activeModeName) {
  if (!config || !activeModeName) return null;
  return config.modes?.[activeModeName] || null;
}

export function getNegativePromptTemplates(mode) {
  return mode?.negative_prompt_templates || {};
}

export function getNegativePromptTemplateOptions(mode) {
  return Object.entries(getNegativePromptTemplates(mode)).map(([value, prompt]) => ({
    value,
    label: value,
    prompt,
  }));
}

export function resolveNegativePromptTemplateId(mode, negativePrompt) {
  const normalized = String(negativePrompt || '').trim();
  const templates = getNegativePromptTemplates(mode);

  for (const [templateId, templatePrompt] of Object.entries(templates)) {
    if (normalized === String(templatePrompt || '').trim()) {
      return templateId;
    }
  }

  if (mode?.allow_custom_negative_prompt && normalized) {
    return CUSTOM_NEGATIVE_PROMPT_ID;
  }

  return '';
}

export function getSchedulerOptions(mode) {
  const allowed = Array.isArray(mode?.allowed_scheduler_ids) ? mode.allowed_scheduler_ids : [];
  return allowed.map((schedulerId) => ({
    value: schedulerId,
    label: schedulerId,
  }));
}

export function applyModeControlDefaultsToDraft(draft, mode) {
  const next = { ...draft };
  const templates = getNegativePromptTemplates(mode);
  const defaultTemplateId = mode?.default_negative_prompt_template || null;

  next.negativePrompt =
    defaultTemplateId && templates[defaultTemplateId]
      ? templates[defaultTemplateId]
      : '';

  const allowedSchedulers = Array.isArray(mode?.allowed_scheduler_ids) ? mode.allowed_scheduler_ids : [];
  if (mode?.default_scheduler_id) {
    next.schedulerId = mode.default_scheduler_id;
  } else if (allowedSchedulers.length === 1) {
    next.schedulerId = allowedSchedulers[0];
  } else {
    next.schedulerId = null;
  }

  return next;
}

export function buildGenerateWsParams(payload) {
  return {
    prompt: payload.prompt,
    negative_prompt: payload.negativePrompt || undefined,
    scheduler_id: payload.schedulerId || undefined,
    size: payload.size,
    steps: payload.steps,
    cfg: payload.cfg,
    seed: payload.seed,
    superres: payload.superres,
    superres_magnitude: payload.superresLevel || 1,
    init_image_ref: payload.initImageRef || undefined,
    denoise_strength: payload.initImageRef ? payload.denoiseStrength : undefined,
  };
}
