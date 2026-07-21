"""Guards on the shared Hunyuan control-map fixture.

The Task-10 acceptance failed for days on a fixture nobody had validated: a
border-to-border box/X map that the Hunyuan Canny ControlNet turns into noise,
while the standalone probe was quietly validating a *different*, inset map. The
integration was never broken. These tests pin the properties that distinguish
the known-good fixture so the pathological shape cannot come back.

No CUDA, no model — pure image geometry.
"""

import io

from hunyuan_control_map import control_map_image, control_map_png
from PIL import Image


def test_control_map_is_requested_size():
    assert control_map_image(1024).size == (1024, 1024)
    assert control_map_image(512).size == (512, 512)


def test_control_map_png_round_trips():
    img = Image.open(io.BytesIO(control_map_png()))
    img.load()
    assert img.size == (1024, 1024)
    assert img.mode == "RGB"


def test_control_map_leaves_a_clear_margin():
    """No ink on the frame edge.

    Edge-to-edge strokes are the property that made the old fixture
    pathological for this checkpoint.
    """
    img = control_map_image(1024)
    pixels = img.load()
    size = img.size[0]

    edge_coords = []
    for i in range(size):
        edge_coords.extend([(i, 0), (i, size - 1), (0, i), (size - 1, i)])

    lit = [c for c in edge_coords if pixels[c] != (0, 0, 0)]
    assert not lit, f"control map touches the frame at {len(lit)} border pixels"


def test_control_map_has_actual_content():
    # A blank map would trivially satisfy the margin guard above.
    img = control_map_image(1024)
    lit = sum(1 for p in img.getdata() if p != (0, 0, 0))
    assert lit > 5000, f"control map has too little edge content ({lit} px)"


def test_control_map_is_deterministic():
    assert control_map_png() == control_map_png()
