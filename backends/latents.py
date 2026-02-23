"""Latent tensor utilities shared across RKNN and CUDA backends."""

import numpy as np


def extract_latents(res):
    """
    Best-effort extraction of latents from pipeline return object.
    Accepts dict-like objects or diffusers-like outputs.
    """
    if res is None:
        return None
    if isinstance(res, dict):
        for k in ("latents", "latent", "images"):
            if k in res and res[k] is not None:
                return res[k]
        return None
    # object with attributes
    for k in ("latents", "latent", "images"):
        if hasattr(res, k):
            v = getattr(res, k)
            if v is not None:
                return v
    return None


def latent_to_nchw(x) -> np.ndarray:
    """
    Convert latent tensor to numpy NCHW.
    Supports:
      - numpy arrays
      - lists
      - objects with .numpy()
      - torch tensors (via .detach().cpu().numpy() if available)
    """
    if x is None:
        raise ValueError("latent is None")

    # Torch tensor?
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.ndim != 4:
        raise ValueError(f"latent must be 4D, got shape={x.shape}")

    # NCHW
    if x.shape[1] == 4:
        return x

    # NHWC -> NCHW
    if x.shape[-1] == 4:
        return np.transpose(x, (0, 3, 1, 2))

    # Unknown layout, best effort: if one dim equals 4, move it to C
    if 4 in x.shape:
        c_axis = list(x.shape).index(4)
        if c_axis != 1:
            axes = list(range(4))
            axes.pop(c_axis)
            axes.insert(1, c_axis)
            return np.transpose(x, axes)

    raise ValueError(f"cannot interpret latent layout, shape={x.shape}")


def downsample_to_8x8_nchw(lat: np.ndarray) -> np.ndarray:
    """
    Downsample NCHW latent to [1,4,8,8] using block-mean if divisible.
    Falls back to nearest sampling if not divisible.
    """
    lat = np.asarray(lat)
    if lat.shape[0] != 1:
        lat = lat[:1]
    if lat.shape[1] != 4:
        raise ValueError(f"expected C=4, got shape={lat.shape}")

    _, _, h, w = lat.shape
    if h == 8 and w == 8:
        return lat

    # Block-average if divisible by 8
    if (h % 8 == 0) and (w % 8 == 0):
        bh = h // 8
        bw = w // 8
        # (1,4,8,bh,8,bw) -> mean over bh,bw
        return lat.reshape(1, 4, 8, bh, 8, bw).mean(axis=(3, 5))

    # Nearest sampling fallback
    ys = (np.linspace(0, h - 1, 8)).round().astype(np.int64)
    xs = (np.linspace(0, w - 1, 8)).round().astype(np.int64)
    return lat[:, :, ys][:, :, :, xs]
