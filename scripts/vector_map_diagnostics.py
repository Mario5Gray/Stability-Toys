"""Conservative square-coverage diagnostics. Outside canvas is background. STABL-vjpnctjh."""

import cv2
import numpy as np


def separate_solids(material):
    """Warn about unbacked islands. S3.4, STABL-gorlibrt. Each island becomes one solid."""
    count = _component_count(material)
    if count < 2:
        return ()
    return ({
        "level": "warning", "code": "multiple_solids", "components": count,
        "message": (
            f"The prepared silhouette has {count} disconnected components and no backing. "
            "The relief becomes that many separate solids, an approximation until S3.6 mesh "
            "validation. Add --backing-mm to join them."
        ),
    },)


def _component_count(material):
    """Count raster-connected foreground islands. Label 0 is the background."""
    count, _ = cv2.connectedComponents(material.astype(np.uint8), connectivity=8)
    return int(count) - 1


def feature_diagnostics(material):
    """Warn about possible thin material and gaps without changing input pixels."""
    diagnostics = []
    kernel = np.ones((4, 4), np.uint8)
    for name, selected, outside in (("material", material, 0), ("background", ~material, 1)):
        padded = np.pad(selected.astype(np.uint8), 3, constant_values=outside)
        # Erosion marks top-left corners. Reflected dilation covers each complete block.
        starts = cv2.erode(padded, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        covered = cv2.dilate(starts, kernel, anchor=(3, 3), borderType=cv2.BORDER_CONSTANT, borderValue=0)[3:-3, 3:-3]
        count = int(np.count_nonzero(selected & ~covered.astype(bool)))
        if count:
            diagnostics.append({
                "level": "warning", "code": f"thin_{name}", "pixels": count, "method": "square_coverage_4px",
                "message": (
                    f"Possible {name} features below four pixels: {count} pixels lack 4x4 square coverage. "
                    "Curved or diagonal boundaries can produce false positives. Geometry unchanged."
                ),
            })
    return tuple(diagnostics)
