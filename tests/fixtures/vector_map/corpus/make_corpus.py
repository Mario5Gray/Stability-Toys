"""Generate the S1.4 acceptance corpus v1 (STABL-cbwzjyky, spec 10).

Eight cases, two per required image class. Wrapper convention: white (255) is material.
Each case commits source.png (the processing-size input) and mask.png (the reviewed
intended geometry). The mask is traced. The render is compared with the same mask.

skimage sources come from the skimage 0.26.0 bundled data. Their licenses are in the
skimage docstrings and are copied into corpus.json. Original sources are drawn here.

derive_mask() reproduces every committed mask from the committed source.
A manual correction must be a deterministic patch operation in CASES.

Run from the repo root to rewrite the committed files:
    python -m tests.fixtures.vector_map.corpus.make_corpus
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from tests.fixtures.vector_map.corpus import metrics

MAX_SIDE = 256
SUPERSAMPLE = 4
CLASSES = ("flat_graphic", "line_art", "textured_photo", "pattern")
PROPERTIES = ("sparse", "dense", "high_contrast", "low_contrast", "holes", "disconnected")

# Contrast: |median source luma of material - median of background| / 255.
CONTRAST_LABELS = ((0.6, "high_contrast"), (0.35, "medium_contrast"), (0.0, "low_contrast"))
# Edge density: metrics.perimeter(mask) / mask pixel count.
DENSE_EDGE_DENSITY = 0.05

SKIMAGE = {
    "horse": (
        "https://openclipart.org/detail/158377/horse-by-marauder",
        "CC0 1.0, given by the owner Andreas Preuss (marauder). No copyright restrictions.",
    ),
    "coins": (
        "https://www.brooklynmuseum.org/opencollection/archives/image/51611",
        "Brooklyn Museum Collection. No known copyright restrictions.",
    ),
    "gravel": (
        "https://cc0textures.com/view.php?tex=Gravel04",
        "CC0 1.0 (CC0Textures).",
    ),
    "brick": (
        "https://cc0textures.com/view.php?tex=Bricks25",
        "CC0 1.0 (CC0Textures).",
    ),
}
ORIGINAL_LICENSE = "Original work drawn by make_corpus.py for this repository. Repository license."

# Derivation parameters. Steps run in the order derive_mask() documents.
CASES = {
    "horse": {
        "class": "flat_graphic",
        "properties": ["sparse", "high_contrast"],
        "source": "skimage",
        # max_hole_area fills a 3 px LANCZOS gap between tail hairs at x=22, y=153..155.
        "derive": {"blur_sigma": 0.0, "threshold": 128, "material": "light", "open_radius": 0, "close_radius": 0,
                   "min_area": 0, "max_hole_area": 8},
    },
    "badge": {
        "class": "flat_graphic",
        "properties": ["sparse", "high_contrast", "holes", "disconnected"],
        "source": "original",
        "derive": {"blur_sigma": 0.0, "threshold": 128, "material": "light", "open_radius": 0, "close_radius": 0,
                   "min_area": 0, "max_hole_area": 0},
    },
    "bracket": {
        "class": "line_art",
        "properties": ["sparse", "high_contrast", "holes", "disconnected"],
        "source": "original",
        "derive": {"blur_sigma": 0.0, "threshold": 128, "material": "dark", "open_radius": 0, "close_radius": 0,
                   "min_area": 0, "max_hole_area": 0},
    },
    "pcb": {
        "class": "line_art",
        "properties": ["dense", "high_contrast", "holes", "disconnected"],
        "source": "original",
        "derive": {"blur_sigma": 0.0, "threshold": 128, "material": "light", "open_radius": 0, "close_radius": 0,
                   "min_area": 0, "max_hole_area": 0},
    },
    "coins": {
        "class": "textured_photo",
        "properties": ["dense", "medium_contrast", "disconnected"],
        "source": "skimage",
        # The top-hat removes the bright illumination at the top left.
        # close_radius 2 bridges two adjacent coins. Radius 1 keeps all 24 separate.
        "derive": {"blur_sigma": 1.0, "tophat_size": 41, "threshold": 40, "material": "light", "open_radius": 1,
                   "close_radius": 1, "min_area": 40, "max_hole_area": 400},
    },
    "gravel": {
        "class": "textured_photo",
        "properties": ["dense", "low_contrast", "holes", "disconnected"],
        "source": "skimage",
        "derive": {"blur_sigma": 1.0, "threshold": 132, "material": "light", "open_radius": 1, "close_radius": 0,
                   "min_area": 12, "max_hole_area": 8},
    },
    "brick": {
        "class": "pattern",
        "properties": ["dense", "low_contrast", "holes"],
        "source": "skimage",
        # Material is the mortar network. The top-hat keeps thin bright lines and drops the shading.
        "derive": {"blur_sigma": 0.5, "tophat_size": 9, "threshold": 25, "material": "light", "open_radius": 0,
                   "close_radius": 0, "min_area": 30, "max_hole_area": 8},
    },
    "truchet": {
        "class": "pattern",
        "properties": ["dense", "low_contrast", "holes", "disconnected"],
        "source": "original",
        "derive": {"blur_sigma": 1.0, "threshold": 128, "material": "light", "open_radius": 0, "close_radius": 0,
                   "min_area": 8, "max_hole_area": 8},
    },
}
for _case in CASES.values():
    _case["derive"].setdefault("tophat_size", 0)
    _case["derive"].setdefault("patches", [])


# --- Sources ------------------------------------------------------------------


def _resize_long_side(image):
    width, height = image.size
    scale = MAX_SIDE / max(width, height)
    size = (round(width * scale), round(height * scale))
    return image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image


def _skimage_source(name):
    import skimage.data

    data = getattr(skimage.data, name)()
    if name == "horse":
        data = np.where(data, 0, 255).astype(np.uint8)  # skimage True is background.
    return np.asarray(_resize_long_side(Image.fromarray(data, mode="L")))


def _canvas(background, size=(MAX_SIDE, MAX_SIDE)):
    image = Image.new("L", (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE), background)
    return image, ImageDraw.Draw(image)


def _star(cx, cy, outer, inner, points=5):
    angles = np.pi / 2 + np.arange(2 * points) * np.pi / points
    radii = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    return [(cx + r * np.cos(a), cy - r * np.sin(a)) for r, a in zip(radii, angles)]


def _badge():
    ink, paper = 235, 20
    image, draw = _canvas(paper)
    draw.ellipse([112, 112, 912, 912], fill=ink)
    draw.ellipse([232, 232, 792, 792], fill=paper)
    draw.polygon(_star(512, 520, 240, 100), fill=ink)
    draw.ellipse([482, 490, 542, 550], fill=paper)  # a hole in the star island
    for cx, cy in ((90, 90), (934, 90), (90, 934), (934, 934)):
        draw.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], fill=ink)
    return image


def _bracket():
    ink, paper, stroke = 25, 245, 12
    image, draw = _canvas(paper)
    draw.rounded_rectangle([120, 220, 904, 840], radius=60, outline=ink, width=stroke)
    for cx in (320, 704):
        draw.ellipse([cx - 90, 440, cx + 90, 620], outline=ink, width=stroke)
        draw.rectangle([cx - 36, 526, cx + 36, 534], fill=ink)  # centre mark
        draw.rectangle([cx - 4, 494, cx + 4, 566], fill=ink)
    # Dimension line with extension lines and arrowheads, clear of the plate.
    draw.rectangle([120, 100, 128, 190], fill=ink)
    draw.rectangle([896, 100, 904, 190], fill=ink)
    draw.rectangle([128, 142, 896, 150], fill=ink)
    draw.polygon([(128, 146), (178, 126), (178, 166)], fill=ink)
    draw.polygon([(896, 146), (846, 126), (846, 166)], fill=ink)
    return image


def _pcb():
    copper, board = 200, 40
    image, draw = _canvas(board)
    rng = np.random.default_rng(1404)
    centres = [92 + 120 * i for i in range(8)]
    for y in centres:
        for x0, x1 in zip(centres, centres[1:]):
            if rng.random() < 0.45:
                draw.rectangle([x0, y - 7, x1, y + 7], fill=copper)
    for x in centres:
        for y0, y1 in zip(centres, centres[1:]):
            if rng.random() < 0.3:
                draw.rectangle([x - 7, y0, x + 7, y1], fill=copper)
    for y in centres:
        for x in centres:
            draw.ellipse([x - 34, y - 34, x + 34, y + 34], fill=copper)
            draw.ellipse([x - 13, y - 13, x + 13, y + 13], fill=board)
    return image


def _truchet():
    ink, paper, tile, stroke = 150, 105, 64, 16
    image, draw = _canvas(paper)
    rng = np.random.default_rng(1405)
    half = tile // 2
    for row in range(MAX_SIDE * SUPERSAMPLE // tile):
        for col in range(MAX_SIDE * SUPERSAMPLE // tile):
            x, y = col * tile, row * tile
            if rng.random() < 0.5:
                arcs = (((x - half, y - half, x + half, y + half), 0, 90),
                        ((x + half, y + half, x + 3 * half, y + 3 * half), 180, 270))
            else:
                arcs = (((x + half, y - half, x + 3 * half, y + half), 90, 180),
                        ((x - half, y + half, x + half, y + 3 * half), 270, 360))
            for box, start, end in arcs:
                draw.arc(box, start, end, fill=ink, width=stroke)
    return image


_ORIGINALS = {"badge": _badge, "bracket": _bracket, "pcb": _pcb, "truchet": _truchet}
_NOISE = {"truchet": (1406, 12)}  # case -> (seed, uniform amplitude) added after downsampling


def build_source(case):
    """Return the processing-size grayscale source as uint8."""
    if CASES[case]["source"] == "skimage":
        return _skimage_source(case)
    source = np.asarray(_ORIGINALS[case]().reduce(SUPERSAMPLE)).astype(np.int16)
    if case in _NOISE:
        seed, amplitude = _NOISE[case]
        source = source + np.random.default_rng(seed).integers(-amplitude, amplitude + 1, source.shape)
    return np.clip(source, 0, 255).astype(np.uint8)


# --- Mask derivation ----------------------------------------------------------


def _disk(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def _morph(material, op, radius):
    """Binary opening or closing with edge-replicated padding, so the canvas edge is not background."""
    padded = np.pad(material, radius, mode="edge")
    out = op(padded, structure=_disk(radius))
    return out[radius:-radius, radius:-radius]


def _remove_small(material, min_area):
    labels, count = ndimage.label(material)
    if not count:
        return material
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def _fill_small_holes(material, max_area):
    background, count = ndimage.label(~material)
    edge = set(np.unique(np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]])))
    sizes = np.bincount(background.ravel(), minlength=count + 1)
    fill = np.zeros(count + 1, bool)
    for label in range(1, count + 1):
        fill[label] = label not in edge and sizes[label] <= max_area
    return material | fill[background]


def _bridge_diagonals(material):
    """Bridge each 2x2 checker with one material pixel. Return (array, bridges)."""
    out = material.copy()
    bridges = 0
    while True:
        a, b = out[:-1, :-1], out[:-1, 1:]
        c, d = out[1:, :-1], out[1:, 1:]
        main = a & d & ~b & ~c  # [[1,0],[0,1]] -> set top-right
        anti = b & c & ~a & ~d  # [[0,1],[1,0]] -> set top-left
        rows, cols = np.nonzero(main | anti)
        if rows.size == 0:
            return out, bridges
        r, k = rows[0], cols[0]
        out[r, k + 1 if main[r, k] else k] = True
        bridges += 1


def _patch(material, patches):
    out = material.copy()
    for patch in patches:
        if patch["op"] != "fill":
            raise ValueError(f"Unknown patch op: {patch['op']!r}")
        x0, y0, x1, y1 = patch["rect"]
        out[y0:y1, x0:x1] = bool(patch["value"])
    return out


def derive_mask(case, source):
    """Return the uint8 0/255 mask for a committed source.

    Steps, in order:
    1. Gaussian blur (sigma, mode nearest) when blur_sigma > 0.
       White top-hat when tophat_size > 0: subtract a square grey opening (mode nearest).
       This removes uneven illumination wider than the structures.
    2. Threshold: light material is luma >= threshold. Dark material is luma < threshold.
    3. Opening, then closing, each with a disk and edge-replicated padding.
    4. Apply patch operations.
    5. Repeat until stable: remove 4-connected material components under min_area,
       fill enclosed 4-connected holes of max_hole_area or less, bridge 2x2 diagonal checkers.
    """
    params = CASES[case]["derive"]
    gray = np.asarray(source, float)
    if params["blur_sigma"] > 0:
        gray = ndimage.gaussian_filter(gray, params["blur_sigma"], mode="nearest")
    if params["tophat_size"]:
        size = (params["tophat_size"], params["tophat_size"])
        gray = gray - ndimage.grey_opening(gray, size=size, mode="nearest")
    material = gray >= params["threshold"] if params["material"] == "light" else gray < params["threshold"]
    if params["open_radius"]:
        material = _morph(material, ndimage.binary_opening, params["open_radius"])
    if params["close_radius"]:
        material = _morph(material, ndimage.binary_closing, params["close_radius"])
    material = _patch(material, params["patches"])
    for _ in range(100):
        before = material
        if params["min_area"]:
            material = _remove_small(material, params["min_area"])
        if params["max_hole_area"]:
            material = _fill_small_holes(material, params["max_hole_area"])
        material, _ = _bridge_diagonals(material)
        if np.array_equal(before, material):
            break
    else:
        raise RuntimeError(f"{case}: mask derivation did not converge.")
    return np.where(material, 255, 0).astype(np.uint8)


# --- Inventory ----------------------------------------------------------------


def contrast(source, material):
    return abs(float(np.median(source[material])) - float(np.median(source[~material]))) / 255.0


def contrast_label(value):
    return next(label for floor, label in CONTRAST_LABELS if value >= floor)


def density_label(value):
    return "dense" if value >= DENSE_EDGE_DENSITY else "sparse"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(case, source, mask, out, previous):
    spec = CASES[case]
    material = mask == 255
    if spec["source"] == "skimage":
        url, license_text = SKIMAGE[case]
        import skimage.data

        original = getattr(skimage.data, case)().shape
        source_info = {
            "origin": "skimage",
            "name": f"skimage.data.{case}",
            "skimage_version": version("scikit-image"),
            "upstream": url,
            "original_size": [original[1], original[0]],
            "resample": "PIL LANCZOS to long side 256",
        }
    else:
        license_text = ORIGINAL_LICENSE
        source_info = {
            "origin": "original",
            "name": f"make_corpus._{case}",
            "resample": f"drawn at {SUPERSAMPLE}x, PIL reduce({SUPERSAMPLE}) box filter",
        }
        if case in _NOISE:
            seed, amplitude = _NOISE[case]
            source_info["noise"] = f"numpy default_rng({seed}) uniform integers +/-{amplitude}"
    hashes = {"source": _sha256(out / "source.png"), "mask": _sha256(out / "mask.png")}
    review = previous.get("review") if previous.get("sha256", {}).get("mask") == hashes["mask"] else None
    edge = metrics.perimeter(material)
    return {
        "class": spec["class"],
        "properties": spec["properties"],
        "source": source_info,
        "license": license_text,
        "processing_size": [mask.shape[1], mask.shape[0]],
        "mask_provenance": {"method": "make_corpus.derive_mask", **spec["derive"]},
        "topology": list(metrics.topology(material)),
        "material_fraction": round(float(material.mean()), 4),
        "perimeter_px": edge,
        "edge_density": round(edge / material.size, 4),
        "contrast": round(contrast(source, material), 4),
        "sha256": hashes,
        "review": review or {"status": "pending"},
    }


def contact_sheet(path):
    """Write a review sheet: source, mask, overlay and inventory values for each case."""
    root = Path(__file__).parent
    cases = json.loads((root / "corpus.json").read_text())["cases"]
    tile, gap, text_width = MAX_SIDE, 8, 300
    sheet = Image.new("RGB", (3 * (tile + gap) + text_width, len(cases) * (tile + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (case, entry) in enumerate(sorted(cases.items())):
        source = np.asarray(Image.open(root / case / "source.png"))
        material = np.asarray(Image.open(root / case / "mask.png")) == 255
        edge = material ^ ndimage.binary_erosion(material, border_value=1)
        overlay = np.stack([source] * 3, axis=-1).astype(float) * 0.6
        overlay[material] = overlay[material] * 0.5 + np.array([0, 140, 255]) * 0.5
        overlay[edge] = (255, 40, 40)
        y = row * (tile + gap)
        for col, image in enumerate((Image.fromarray(source).convert("RGB"),
                                     Image.fromarray(np.where(material, 255, 0).astype(np.uint8)).convert("RGB"),
                                     Image.fromarray(overlay.astype(np.uint8)))):
            sheet.paste(image, (col * (tile + gap), y))
            draw.rectangle([col * (tile + gap), y, col * (tile + gap) + image.width - 1, y + image.height - 1],
                           outline=(128, 128, 128))
        components, holes = entry["topology"]
        lines = [
            case,
            entry["class"],
            f"{entry['processing_size'][0]}x{entry['processing_size'][1]} px",
            f"components {components}, holes {holes}",
            f"contrast {entry['contrast']}",
            f"edge density {entry['edge_density']}",
            f"material {entry['material_fraction']}",
            ", ".join(entry["properties"]),
            entry["source"]["name"],
            f"review: {entry['review']['status']}",
        ]
        draw.multiline_text((3 * (tile + gap) + 8, y + 8), "\n".join(lines), fill="black", spacing=6)
    sheet.save(path)


def main():
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--sheet":
        contact_sheet(sys.argv[2])
        return
    root = Path(__file__).parent
    index = root / "corpus.json"
    previous = json.loads(index.read_text())["cases"] if index.exists() else {}
    cases = {}
    for case in sorted(CASES):
        out = root / case
        out.mkdir(exist_ok=True)
        source = build_source(case)
        Image.fromarray(source, mode="L").save(out / "source.png", optimize=False)
        mask = derive_mask(case, np.asarray(Image.open(out / "source.png")))
        Image.fromarray(mask, mode="L").save(out / "mask.png", optimize=False)
        cases[case] = _entry(case, source, mask, out, previous.get(case, {}))
        print(case, cases[case]["topology"], cases[case]["contrast"], cases[case]["edge_density"])
    index.write_text(json.dumps({"version": 1, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
