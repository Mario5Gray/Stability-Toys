# S3.2 Canny default selection

Issue: `STABL-uifsadne`. Plan: [S3.2 cold plan](../plans/2026-10-09-vector-map-s3-2.md), "Canny default selection".
Script: `spikes/vector_map_canny_sweep.py --structure 100,200,3 --detail 50,100,3`.
Data: `tests/fixtures/vector_map/corpus/canny-sweep.json`. Contact sheets: [`2026-10-09-vector-map-s3-2-canny/`](2026-10-09-vector-map-s3-2-canny/).
Environment: Python 3.12, OpenCV 5.0.0, the `stability-toys` conda env.

## Selected defaults

| Role | low | high | blur | Source |
|---|---|---|---|---|
| structure (coarse) | 100 | 200 | 3 | `vector_map_layers.DEFAULT_CANNY["structure"]` |
| detail (fine) | 50 | 100 | 3 | `vector_map_layers.DEFAULT_CANNY["detail"]` |

The `canny_map.py` command default is 100/200 blur 0. Neither role copies it.

## Method

All eight reviewed sources under `tests/fixtures/vector_map/corpus` were used.
Each reviewed mask is the silhouette. Each case keeps its full oriented canvas.
The grid is threshold pairs (50,100), (75,150), (100,200), (125,250), each with blur 0, 3, and 5.
Each candidate ran through `vector_map_layers.source_rgb` and `canny_map.canny_edges`.
Gap closing was disabled.
Candidate bytes equal a direct `canny_map.canny_edges` call on the same prepared pixels in all 96 runs (`parity: true`).
The sweep records edge density, clipped material counts, and 8-connected components per candidate.
The sweep records detail counts again after removal of the selected structure.

## Clipped density by candidate

Clipped edge pixels divided by silhouette pixels.

| Candidate | badge | bracket | brick | coins | gravel | horse | pcb | truchet |
|---|---|---|---|---|---|---|---|---|
| 50/100 b0 | 0.0491 | 0.3489 | 0.4905 | 0.3610 | 0.2903 | 0.0414 | 0.2511 | 0.3067 |
| 50/100 b3 | 0.0507 | 0.3403 | 0.3840 | 0.3019 | 0.2561 | 0.0421 | 0.2392 | 0.2998 |
| 50/100 b5 | 0.0512 | 0.1727 | 0.2864 | 0.2411 | 0.2470 | 0.0429 | 0.1976 | 0.2901 |
| 75/150 b0 | 0.0491 | 0.3489 | 0.4898 | 0.3287 | 0.2721 | 0.0414 | 0.2511 | 0.2987 |
| 75/150 b3 | 0.0507 | 0.3403 | 0.3824 | 0.2408 | 0.2403 | 0.0421 | 0.2392 | 0.2976 |
| 75/150 b5 | 0.0512 | 0.1727 | 0.2835 | 0.1703 | 0.2274 | 0.0429 | 0.1976 | 0.2778 |
| 100/200 b0 | 0.0491 | 0.3489 | 0.4890 | 0.2830 | 0.2532 | 0.0414 | 0.2511 | 0.2939 |
| 100/200 b3 | 0.0507 | 0.3403 | 0.3793 | 0.1835 | 0.2177 | 0.0421 | 0.2392 | 0.2824 |
| 100/200 b5 | 0.0512 | 0.1724 | 0.2574 | 0.1306 | 0.1913 | 0.0429 | 0.1976 | 0.0000 |
| 125/250 b0 | 0.0491 | 0.3489 | 0.4882 | 0.2370 | 0.2356 | 0.0414 | 0.2511 | 0.2908 |
| 125/250 b3 | 0.0507 | 0.3403 | 0.3696 | 0.1418 | 0.1922 | 0.0421 | 0.2392 | 0.0000 |
| 125/250 b5 | 0.0512 | 0.1716 | 0.1070 | 0.1163 | 0.1453 | 0.0426 | 0.1976 | 0.0000 |

## Structure: 100/200 blur 3

Blur 5 is unsafe as a default.
It removes about half of the bracket's thin strokes at every threshold pair (1178 to 598 pixels).
It removes all truchet material at 100/200 and 125/250.
It breaks the brick pattern at 125/250 (5089 to 1473 pixels).
125/250 blur 3 also removes all truchet material.
100/200 blur 3 is the highest-threshold blurred candidate that keeps material in all eight cases.
On coins it keeps clean outlines and suppresses most interior relief that blur 0 keeps.

## Detail: 50/100 blur 3

Detail must have the same blur as structure.
With equal blur, lower thresholds give a superset of the structure edges.
The sweep confirms it: in all eight cases, 50/100 b3 pixels equal structure pixels plus detail-after-removal pixels.
Structure removal then leaves only the additional weaker edges.
With a different blur, the same outlines move by about one pixel.
Subtraction then leaves one-pixel slivers of structure in detail, not fine detail.

Detail pixels after removal of 100/200 b3 structure:

| Detail candidate | badge | bracket | brick | coins | gravel | horse | pcb | truchet |
|---|---|---|---|---|---|---|---|---|
| 50/100 b0 | 15 | 56 | 1604 | 3600 | 3235 | 29 | 264 | 1185 |
| 50/100 b3 | 0 | 0 | 64 | 2138 | 1024 | 0 | 0 | 339 |
| 75/150 b3 | 0 | 0 | 43 | 1034 | 601 | 0 | 0 | 296 |

50/100 b0 leaves 1604 brick pixels and 56 bracket pixels. Those cases have no fine detail. The pixels are offset outlines.
50/100 b3 leaves no detail on the four flat graphics. It keeps coin relief (2138) and gravel texture (1024).
50/100 b3 keeps more coin relief than 75/150 b3 (2138 against 1034).
[`defaults.png`](2026-10-09-vector-map-s3-2-canny/defaults.png) shows structure and detail after removal for each case.

## Per-image overrides

The defaults keep structure material in all eight cases. No case requires an override.
One optional override is recorded separately from the defaults:

| Case | Role | Override | Reason |
|---|---|---|---|
| coins | structure | 125/250 blur 5 | Outlines only. 100/200 b3 keeps some interior relief on several coins. |

Do not apply this override to brick. At 125/250 b5, brick structure falls from 5223 to 1473 pixels.

## Limits

- Flat graphics get an empty detail role at the defaults. `prepare_layers` reports it as `role_empty`.
- Brick and truchet detail at the defaults is isolated speckle from source noise (64 and 339 pixels). VTracer speckle filtering applies downstream.
- No gravel candidate isolates individual stones. Structure and detail both stay fragmentary.
- The corpus has 256-pixel processing sizes. The sweep did not measure larger processing sizes.
- These defaults are visual heuristics. They are not object recognition.
