"""Generate the S1.2 synthetic topology fixtures (STABL-snyaxjef).

Wrapper convention: white (255) is material, black (0) is background.
Features are at least 10 px wide, above filter_speckle=4. No shape touches another
only at a diagonal, so 4- and 8-connectivity give the same topology.

Run from the repo root to rewrite the committed PNGs:
    python -m tests.fixtures.vector_map.make_fixtures
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = (128, 128)
WHITE, BLACK = 255, 0

# name -> (material components, enclosed holes)
EXPECTED = {
    "donut": (1, 1),
    "nested_island": (2, 2),
    "separate_components": (3, 0),
    "border_touching": (2, 1),
    "asymmetric": (2, 0),
}


def _donut(draw):
    draw.ellipse([16, 16, 111, 111], fill=WHITE)
    draw.ellipse([44, 44, 83, 83], fill=BLACK)


def _nested_island(draw):
    # Ring, then an island in its hole, then a hole in the island.
    draw.ellipse([8, 8, 119, 119], fill=WHITE)
    draw.ellipse([28, 28, 99, 99], fill=BLACK)
    draw.ellipse([44, 44, 83, 83], fill=WHITE)
    draw.ellipse([56, 56, 71, 71], fill=BLACK)


def _separate_components(draw):
    draw.rectangle([10, 10, 40, 40], fill=WHITE)
    draw.ellipse([70, 20, 115, 65], fill=WHITE)
    draw.rectangle([20, 80, 110, 110], fill=WHITE)


def _border_touching(draw):
    # Touches the left, top and bottom edges, with one hole.
    draw.rectangle([0, 0, 63, 127], fill=WHITE)
    draw.rectangle([20, 40, 43, 87], fill=BLACK)
    # Touches the top-right corner.
    draw.rectangle([96, 0, 127, 40], fill=WHITE)


def _asymmetric(draw):
    # An L with a marker at top right. A mirror or flip changes the raster.
    draw.rectangle([16, 16, 39, 111], fill=WHITE)
    draw.rectangle([16, 88, 95, 111], fill=WHITE)
    draw.rectangle([96, 16, 111, 31], fill=WHITE)


_DRAW = {
    "donut": _donut,
    "nested_island": _nested_island,
    "separate_components": _separate_components,
    "border_touching": _border_touching,
    "asymmetric": _asymmetric,
}


def build(name):
    """Return the fixture as a uint8 array with values 0 and 255 only."""
    image = Image.new("L", SIZE, BLACK)
    _DRAW[name](ImageDraw.Draw(image))
    return np.asarray(image)


def main():
    out = Path(__file__).parent
    for name in EXPECTED:
        Image.fromarray(build(name)).save(out / f"{name}.png", optimize=False)
        print(out / f"{name}.png")


if __name__ == "__main__":
    main()
