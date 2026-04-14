export function buildAdvisorEvidence(galleryId, rows) {
  const items = (rows || []).map((row) => ({
    cache_key: row.cacheKey,
    added_at: row.addedAt ?? null,
    prompt: row.params?.prompt ?? null,
    negative_prompt: row.params?.negativePrompt ?? null,
    size: row.params?.size ?? null,
    steps: row.params?.steps ?? null,
    cfg: row.params?.cfg ?? null,
    scheduler_id: row.params?.schedulerId ?? null,
    seed: row.params?.seed ?? null,
    superres_level: row.params?.superresLevel ?? null,
    metadata: row.params?.metadata ?? {},
  }));

  items.sort((a, b) => {
    const aKey = String(a.cache_key ?? '');
    const bKey = String(b.cache_key ?? '');
    if (aKey !== bKey) return aKey.localeCompare(bKey);
    return Number(a.added_at ?? 0) - Number(b.added_at ?? 0);
  });

  return {
    version: 1,
    gallery_id: galleryId,
    items,
  };
}
