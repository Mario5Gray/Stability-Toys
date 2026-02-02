// utils/imageToFile.js

const inFlight = new Map(); // url -> Promise<File>

function extFromType(type) {
  return type === "image/jpeg" ? "jpg"
    : type === "image/webp" ? "webp"
    : "png";
}

function ensureExt(name, ext) {
  return name.includes(".") ? name : `${name}.${ext}`;
}

async function dataUrlToBlob(dataUrl) {
  // data:[<mediatype>][;base64],<data>
  const [header, data] = dataUrl.split(",");
  const isBase64 = header.includes(";base64");
  const mime = (header.match(/data:([^;]+)/)?.[1]) || "image/png";

  if (!isBase64) {
    // percent-encoded
    const text = decodeURIComponent(data);
    return new Blob([text], { type: mime });
  }

  const bin = atob(data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

/**
 * Convert a URL to a File.
 *
 * Optional opts:
 *  - signal: AbortSignal
 *  - resolveBlob: async (url, signal) => Blob | null   // for custom schemes like lcm_image:
 */
export async function urlToFile(url, filename = "chat.png", opts = {}) {
  const key = `${url}::${filename}`;

  if (inFlight.has(key)) return inFlight.get(key);

  const p = (async () => {
    // Fast paths
    if (!url) throw new Error("urlToFile: missing url");

    // data: URL — no fetch
    if (url.startsWith("data:")) {
      const blob = dataUrlToBlob(url);
      const ext = extFromType(blob.type);
      return new File([blob], ensureExt(filename, ext), { type: blob.type || "image/png" });
    }

    // Custom scheme hook (IMPORTANT for lcm_image:)
    if (opts.resolveBlob) {
      const maybe = await opts.resolveBlob(url, opts.signal);
      if (maybe instanceof Blob) {
        const ext = extFromType(maybe.type);
        return new File([maybe], ensureExt(filename, ext), { type: maybe.type || "image/png" });
      }
    }

    // Default: fetch (blob: URLs will still go through here)
    const t0 = performance.now();
    const res = await fetch(url, { signal: opts.signal });
    const t1 = performance.now();

    if (!res.ok) throw new Error(`fetch image failed: ${res.status}`);

    const blob = await res.blob();
    const t2 = performance.now();

    // Optional perf logging
     console.log(`[urlToFile] fetch=${(t1-t0).toFixed(1)}ms blob=${(t2-t1).toFixed(1)}ms url=${url}`);

    const ext = extFromType(blob.type);
    return new File([blob], ensureExt(filename, ext), { type: blob.type || "image/png" });
  })();

  inFlight.set(key, p);

  try {
    const file = await p;
    return file;
  } finally {
    inFlight.delete(key);
  }
}


//// utils/imageToFile.js
// export async function urlToFile(url, filename = "chat.png") {
//   const res = await fetch(url);
//   if (!res.ok) throw new Error(`fetch image failed: ${res.status}`);
//   const blob = await res.blob();

//   // try to preserve type
//   const ext = blob.type === "image/jpeg" ? "jpg"
//            : blob.type === "image/webp" ? "webp"
//            : "png";

//   const safeName = filename.includes(".") ? filename : `${filename}.${ext}`;
//   return new File([blob], safeName, { type: blob.type || "image/png" });
// }