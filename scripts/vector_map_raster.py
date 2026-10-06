"""Raster preparation for st-vector-map (spec 5, 6.1).

S2.2 walking skeleton (STABL-ascsgqha): mask luminance >= 128 and optional inversion only.
S2.3 (STABL-vjpnctjh) adds EXIF orientation, alpha selection, --max-res, bands, and masks.
Until then, input that needs those steps is rejected, not silently mis-prepared.
"""

import numpy as np
from PIL import Image

from vector_map_config import S23, deferred

FOREGROUND_LUMINANCE = 128
_EXIF_ORIENTATION = 0x0112


def prepare_mask(path, *, invert):
    """Return the boolean material layer. White (luminance >= 128) is material."""
    try:
        image = Image.open(path)
    except Image.DecompressionBombError as exc:
        # Subclasses Exception only. Re-raise as a processing failure (exit 1).
        raise ValueError(f"{path} exceeds the Pillow pixel limit: {exc}") from exc
    with image:
        if image.getexif().get(_EXIF_ORIENTATION, 1) != 1:
            raise deferred("EXIF orientation other than 1", S23)
        if "A" in image.getbands() or "transparency" in image.info:
            raise deferred("Alpha-bearing input", S23)
        material = np.asarray(image.convert("L")) >= FOREGROUND_LUMINANCE
    return ~material if invert else material
