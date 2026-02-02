// cacheHydrate.js
import { PRIORITY } from "@/lib/jobQueue";
import jobQueue from "@/lib/jobQueue";

const inflightHydrates = new Set();

/**
 * Hydrate a cache entry by fetching blob bytes from serverImageUrl.
 * Enqueues as BACKGROUND to avoid blocking UI.
 */
export function hydrateCacheEntry({ cache, cacheKey, serverImageUrl }) {
  if (!cache || !serverImageUrl) return;

  // avoid duplicate hydrations if user spam-clicks
  const hydrateJobId = `hydrate:${cacheKey}`;
  if (jobQueue.has?.(hydrateJobId)) return; // if you have this; else skip

    if (inflightHydrates.has(cacheKey)) return;
      inflightHydrates.add(cacheKey);

  jobQueue.enqueue({
    id: hydrateJobId, // optional if your queue supports custom ids
    priority: PRIORITY.BACKGROUND,
    source: "cache-hydrate",
    payload: { cacheKey, serverImageUrl },
    runner: async ({ cacheKey, serverImageUrl }, signal) => {
      try {        
        const t0 = performance.now();
        const res = await fetch(serverImageUrl, { signal, cache: "force-cache" });
        if (!res.ok) throw new Error(`hydrate fetch failed: ${res.status}`);
        const blob = await res.blob();

        // Persist blob + preserve metadata by merging
        const existing = await cache.get(cacheKey);
        const metadata = { ...(existing?.metadata || {}), hydratedAt: Date.now() };
        

        await cache.set(cacheKey, blob, metadata);

        // Optional: emit event so UI can swap from server URL to blob URL immediately
        cache.emit?.("hydrated", { cacheKey });
                console.log(
            "[Cache] hydrated",
            cacheKey,
            `${blob.size} bytes`,
            `${(performance.now() - t0).toFixed(1)}ms`
        );      
      } 
        finally 
      {
        inflightHydrates.delete(cacheKey);
      }


      return true;
    },
  });
}
