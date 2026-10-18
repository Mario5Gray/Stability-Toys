# Vector-map operator guide

Use this guide to go from an image to an SVG, an OpenSCAD source, a slicer file, and an evidence review.
Flag, JSON, and manifest details live in the [CLI reference](../scripts/USAGE.md#st-vector-map). This guide does not repeat them.

## 1. Prerequisites

| Need | Value |
|---|---|
| Python | CPython 3.12. Proven on macOS arm64 and Fedora 44 x86_64. |
| Install | `make install-controlnet-scripts EXTRAS=vector` |
| OpenSCAD | 2021.01. Needed only to reproduce the committed CAD evidence. |
| Corpus | `tests/fixtures/vector_map/corpus/<case>/source.png` and `mask.png` |

Support claim. Only `st-vector-map` and `st-canny-map` conversion of a real image on CPython 3.12 is proven.
The depth, pose, and all extras install and run `--help`. Their inference is untested.
CPython 3.14 and later refuse vector conversion. CPython 3.10, 3.11, and 3.13 are unverified.
Do not read "pip installs it" as "supported". The [S4.3 regression report](superpowers/reports/2026-10-18-vector-map-s4-3-regression.md) holds the matrix.

OpenSCAD is not on `PATH` by default, and the default Mac install can be a later version. Select the 2021.01 binary explicitly:

```bash
export OPENSCAD=/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD
"$OPENSCAD" --version   # OpenSCAD version 2021.01
```

Later OpenSCAD versions fail some committed CAD tests. Use 2021.01 to reproduce evidence.

## 2. Choose an input kind

Every example uses the `badge` corpus case. Write outputs outside the repository.

```bash
C=tests/fixtures/vector_map/corpus/badge
W=$(mktemp -d)
```

| Kind | Use when | Command |
|---|---|---|
| `mask` | You have a black-and-white mask. | `st-vector-map $C/mask.png $W/mask.svg --input-kind mask --width-mm 100` |
| `edges` | You have an edge map. | `st-vector-map $W/edges.png $W/edges.svg --input-kind edges --width-mm 100 --line-width-mm 0.8` |
| `image` | You have a photo or graphic. | `st-vector-map $C/source.png $W/image.svg --input-kind image --mask $C/mask.png --width-mm 100` |

Make `$W/edges.png` with `st-canny-map $C/source.png $W/edges.png`.

Image mode needs exactly one silhouette method. Without one, the command exits 2.

| Method | Selects |
|---|---|
| `--mask PATH` | White pixels of a reviewed mask. The source sets the canvas only. |
| `--alpha` | Source alpha of 128 or more. |
| `--threshold N` | Source luminance of N or more. N is 0 to 255. |

Prefer a reviewed `--mask`. A threshold or alpha selection is a guess until you inspect the preview.

Rules that apply to all kinds:

- Select one physical size: `--width-mm` or `--height-mm`. The other side follows the aspect ratio.
- Default material is luminance of 128 or more. `--invert` reverses the selection.
- EXIF orientation applies first. An external mask must match the oriented source size.
- `--include-mask` limits material. `--exclude-mask` removes material and always wins.
- Pixels are not millimetres. The manifest records `canvas.mm_per_px`. The badge run uses 0.390625 mm per pixel.
- All layers share one canvas and origin. The tool never crops or centres a layer.
- `--line-width-mm` needs `--input-kind edges`. It is a nominal width, rounded up to an odd pixel count. At 0.390625 mm per pixel, a request of 0.8 mm gives 1.17 mm.

## 3. Build a layered relief

A relief has up to three roles: `silhouette`, `structure`, and `detail`. The list must include `silhouette`.

```bash
st-vector-map $C/source.png $W/relief.svg --input-kind image --mask $C/mask.png --width-mm 100 \
  --layers silhouette,structure,detail --preset relief-0.4 --export-scad --preview --debug-bundle --json
```

- **Reviewed maps versus Canny.** By default, Canny on the source proposes structure and detail. Treat that as a proposal.
  Use `--structure-map PATH` and `--detail-map PATH` for reviewed masks. A map replaces Canny for that role.
- **Preset.** `relief-0.4` fills fabrication values only. It never picks roles, so `--layers` stays explicit.
  The preset does not prove that a 0.4 mm nozzle can print the result.
- **Recipe.** `--recipe FILE` replays settings. The file needs `schema_version: 1`. Paths resolve relative to the recipe file.
- **Override order.** Built-in defaults, then preset, then recipe, then explicit CLI flags. The last one wins.
- **Preview.** `--preview` writes `<stem>.preview.png`. It shows the mask, the rendered SVG, and an overlay on the source.
  It checks the SVG trace. It does not prove CAD or print results.
- **JSON.** `--json` prints one result object. Fields are in the [JSON result](../scripts/USAGE.md#st-vector-map) section of the CLI reference.
- **Export.** `--export-scad` writes `<stem>.scad` beside the SVG. It never starts OpenSCAD.
  `--no-export-scad` overrides a recipe export request. It validates values and writes no source.
- **Replay.** `--debug-bundle` saves the prepared masks and `<stem>.debug/recipe.json`. Replay with
  `st-vector-map $W/replay.svg --recipe $W/relief.debug/recipe.json`.

A run publishes `<stem>.svg`, `<stem>.vector.json`, and `<stem>.layers/<role>.svg`. It adds `<stem>.scad`, the preview,
and the debug directory when requested. The manifest lists every other published artifact with a hash. It cannot list or hash itself.
Trust only artifacts that the current manifest lists with matching hashes.

The relief run warns when the silhouette has several disconnected parts and no backing. Add `--backing-mm` to join them.

## 4. SCAD assembly, heights, and Z intervals

The SCAD source imports each published `<stem>.layers/<role>.svg` with `center=false`.
It never imports the combined SVG. OpenSCAD resolves each relative import beside the SCAD file.
Keep the SCAD file and its `.layers` directory together.
`--center` adds one outer `translate` that centres the assembly on the XY origin. Imports stay uncentred.

Terms: backing B, silhouette thickness S, structure height H, detail height D.

| Element | Z interval |
|---|---|
| Backing, if B > 0 | `[0, B]` |
| Silhouette | `[B, B+S]` |
| Raised structure | `[B+S, B+S+H]` |
| Raised detail | `[B+S, B+S+D]` |
| Engraving removal | `[B+S-D, B+S]` |

- Engraved detail needs `0 < D < S`. Select it with `--detail-mode engraved`.
- Export needs a positive S. Each published role needs a positive height.
- Small overlaps of 0.01 mm join solids at support planes. Top heights stay exact.
- Without a backing, separate silhouette parts become separate solids. The backing spans the canvas and joins them.

Render and slice outside the repository. `*.stl` is not ignored by git.

```bash
"$OPENSCAD" -o "$TMPDIR/relief.stl" $W/relief.scad
```

## 5. Evidence

Four sources answer four different questions. None replaces another.

| Question | Source | Verdict |
|---|---|---|
| Did conversion publish an SVG? | CLI `--json` `evidence.conversion` | `converted`, `failed`, or `invalid` |
| Does the trace match the mask? | [S4.1 corpus report](superpowers/reports/2026-10-15-vector-map-s4-1-corpus.md). Per case: `<case>/<mode>/result.json`, field `fidelity.failures`. | 4 of 8 cases pass. Four fail. |
| Does OpenSCAD render it? | The same `result.json`, field `cad_passed` | 16 of 16 rows pass on OpenSCAD 2021.01 |
| Does it slice? | [S4.5 print trial](superpowers/reports/2026-10-17-vector-map-s4-5-print-trial.md), Slicer rows. Operator: Mario. | `badge/raised` PASS and `brick/engraved` PASS, by operator report |
| Does it print? | The same report, Print rows. Operator: Mario. | `badge/raised` PASS and `brick/engraved` PASS, by operator report |

The install and repeatability evidence is in the [S4.3 regression report](superpowers/reports/2026-10-18-vector-map-s4-3-regression.md).
CLI JSON always states `cad_render: not_run` and `physical_print: pending`. This stays true after external acceptance.
The CLI never reads the CAD or print reports.

The S4.5 print rows are `badge/raised` and `brick/engraved`. "Thickness and scale look correct" was a visual check, not a caliper measurement.
The print trial does not cover other cases.

## 6. Known limits

- **No boundary guarantee.** The tool gives no universal physical boundary-error guarantee. Check the preview and the fidelity record.
- **Thin features.** Features 1 to 2 processing pixels wide can be lost or changed. Make a feature at least four pixels wide in the mask.
  The command warns about possible features under four pixels. Curved or diagonal edges can give false positives.
- **Retained failures.** Bracket, brick, gravel, and horse fail the locked fidelity checks. Mario accepted them as documented limitations on 2026-10-15.
  They stay FAIL. See the [S4.1 corpus report](superpowers/reports/2026-10-15-vector-map-s4-1-corpus.md) for values.
- **No repair.** The tool does not repair self-intersections. The SVG dialect check does not prove that paths have none.
- **No bridges.** The tool adds no bridges between separate solids. Use `--backing-mm`.
- **Canny is a band.** Canny output is an edge band. It is not a centreline and not an inferred silhouette. A line width can widen it more.
- **Engraving.** Engraving needs `0 < D < S`. An engraved groove that opens through the outline stays open.
  A recess that meets the outline at one point can leave a small uncut square. The command reports `cad_recess_bridge`.
- **OpenSCAD SVG import.** OpenSCAD 2021.01 can lose material where two rings touch at one point, and it exits 0.
  The command moves contact points by at most 0.01 mm and reports `cad_pinch_closed` or `cad_pinch_unrepaired`.
  Read these warnings before you trust a render.
- **Topology fallback.** Research on a repair exists in `STABL-wuzmvgxm`. It is not a released feature.

## 7. Check this guide

```bash
st-vector-map --help          # flags match section 2 and section 3
git diff --check              # no whitespace errors
drift check                   # anchors for this guide are fresh
```

The botanical relief example in the design spec, section 13, is an illustration. It is not an acceptance recipe.
