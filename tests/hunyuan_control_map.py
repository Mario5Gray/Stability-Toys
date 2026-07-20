"""The one Hunyuan Canny control-map fixture (STABL-ichgkgno).

Shared deliberately between the live acceptance and `scripts/hunyuan_cn_probe.py`.
They previously carried separate hand-drawn maps, and the probe validated its
own while the acceptance ran a different, untested one — which cost a long
investigation into worker code that turned out to be correct. A probe that does
not exercise the acceptance's exact conditioning input cannot clear it, so there
is one fixture and both import it.

Geometry is the validated probe map: strokes inset well clear of the frame.
Border-to-border edges drive this Canny checkpoint into noise; see
`tests/test_hunyuan_control_map.py` for the guards.

Kept free of torch/diffusers imports so it loads in the probe and in off-GPU
unit tests alike.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def control_map_image(size: int = 1024) -> Image.Image:
    """Synthetic Canny-style map: inset box, gable, and a horizontal course."""
    img = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(img)
    scale = size / 1024

    def s(value: float) -> int:
        return int(round(value * scale))

    def stroke(width: int) -> int:
        return max(1, s(width))

    draw.rectangle((s(160), s(180), s(864), s(850)), outline="white", width=stroke(8))
    draw.line(
        (s(160), s(850), s(512), s(180), s(864), s(850)),
        fill="white",
        width=stroke(8),
    )
    draw.line((s(250), s(720), s(760), s(720)), fill="white", width=stroke(5))
    return img


def control_map_png(size: int = 1024) -> bytes:
    buf = io.BytesIO()
    control_map_image(size).save(buf, format="PNG")
    return buf.getvalue()
