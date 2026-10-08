# Sprint 2 acceptance: standalone st-vector-map

Issue: `STABL-ntgnbxci` (S2.8). Parent: `STABL-neizotrw`.
Executed: 2026-10-08. Base: `main` at `363b5ac`. Branch: `feat/ntgnbxci-acceptance`.
Plan: [remainder plan, S2.8](../plans/2026-10-07-vector-map-sprint-2-remainder.md). Contract: spec sections 8, 10 and 11.2.
Tests: [`tests/test_vector_map_acceptance.py`](../../../tests/test_vector_map_acceptance.py).
Evidence JSON: [`evidence-source.json`](2026-10-07-vector-map-sprint-2-acceptance/evidence-source.json), [`evidence-installed-commands.json`](2026-10-07-vector-map-sprint-2-acceptance/evidence-installed-commands.json), [`evidence-container.json`](2026-10-07-vector-map-sprint-2-acceptance/evidence-container.json).

This report records standalone conversion evidence only. It makes no CAD or physical-print claim.
Optional empty detail layers and relief composition remain Sprint 3 acceptance.

## Summary

| Claim | Result |
|---|---|
| Repeated real conversions give identical bytes | Pass. 21 cases, three runs each, one destination. SVG, manifest, preview and debug bytes equal. |
| Real `st-canny-map` output converts in edge mode | Pass. Eight corpus sources. All eight convert. Geometric checks are recorded, not gated. |
| Manifest hashes verify independently | Pass. Each run hashes every listed file with `hashlib` in the test. |
| Run one publishes by `os.link`. Runs two and three publish by `os.replace` | Pass. Every case: run one 5 links and 0 replaces. Runs two and three 0 links and 5 replaces. |
| Coordinates, composition and failure behaviour at CLI level | Pass. 18 checks below, plus 2 command equality checks. |
| Installed package converts without Torch | Pass. Fresh venv, `pip install "./scripts[vector]"`, no `torch` module. |
| Sibling entry points | Pass. `st-canny-map` and `st-resize-for-model` run. `st-depth-map` and `st-pose-map` exit 1 with the install hint when Torch is absent. |
| Focused and vector suites | Pass locally and in the native container. |
| Full isolated suite | **Exit 1 locally and in the container.** No failure is in vector-map code. See "Known failures". |

## Environments

| Name | Platform | Python | vtracer | resvg-py | Pillow | numpy | opencv |
|---|---|---|---|---|---|---|---|
| Local source | macOS 26.0.1 arm64, conda env `stability-toys` | 3.12.13 | 0.6.15 | 0.5.0 | 12.3.0 | 2.5.1 | 4.11.0.86 |
| Installed venv, console scripts only | macOS 26.0.1 arm64, fresh `python -m venv` | 3.12.13 | 0.6.15 | 0.5.0 | 12.3.0 | 2.5.3 | headless 5.0.0.93 |
| Native container | `docker-compose.test.yml` service `test`, Linux aarch64, CPU | 3.12.15 | 0.6.15 | 0.5.0 | 12.3.0 | 1.26.4 | 4.11.0.86 |

`st-controlnet-helpers` is 0.1.0 in all three. The installed venv resolved `opencv-python-headless` 5.0.0.93 because the vector extra sets only a floor.
The acceptance tests always run `vector_map.main()` from the source `scripts/` directory under the pytest interpreter. That covers the 21 repeated conversions and all 18 CLI checks.
`ST_VECTOR_MAP_BIN` changes only the subprocess checks: the 8 `st-canny-map` runs and the 2 command byte-equality checks.
Evidence JSON key `in_process` records the pytest interpreter and its versions. Key `command_versions` records the installed venv versions, read with the venv interpreter.

## Commands

Logs are in the session scratchpad and are not durable. The evidence JSON files above are the durable record.

| Command | Exit | Result | Log |
|---|---:|---|---|
| `python -m pytest tests/test_vector_map_acceptance.py -q` | 0 | 41 passed | `s28-acceptance.log` |
| `python -m pytest tests/test_vector_map_*.py -q -rs` | 0 | 876 passed, 0 skipped. OpenSCAD is present on this host | `s28-vector.log` |
| `ST_VECTOR_MAP_BIN=<venv>/bin python -m pytest tests/test_vector_map_acceptance.py` | 0 | 41 passed. Installed `st-canny-map` for the 8 edge sources and installed `st-vector-map` for the 2 command checks. The 21 repeated conversions ran from source under the conda interpreter | `evidence-installed-commands.json` |
| `docker compose -f docker-compose.test.yml build test` | 0 | Image built | `s28-container-build.log` |
| `docker compose -f docker-compose.test.yml run --rm test` | 1 | Cohort 1: 10 failed, 2298 passed, 27 skipped. Real-library cohort: 213 passed. Cohort 2: 1 skipped. Vector files: 862 passed, 14 skipped, 0 failed | `s28-container-run.log` |
| Container `python -m pytest tests/test_vector_map_acceptance.py` | 0 | 41 passed | `evidence-container.json` |
| `docker compose -f docker-compose.test.yml run --rm test` from the main checkout at `363b5ac` | 1 | Cohort 1: the same 10 failed, 2257 passed, 27 skipped. Real-library cohort: 213 passed. Cohort 2: 1 skipped | `s28-container-main.log` |
| `python -m tests.run tests/ -- -q -rs` | 1 | Cohort 1: 1 failed, 2325 passed, 9 skipped. Real-library cohort: 213 passed. Cohort 2: 1 skipped | `s28-full.log` |

Skips. Local cohort 1: 8 SDXL worker checks need CUDA, 1 check needs `env.custom`. Local cohort 2: 1 HunyuanDiT check needs CUDA.
Container cohort 1: 8 SDXL worker, 5 `env.custom` and 14 OpenSCAD checks. Container cohort 2: 1 HunyuanDiT check.
The 14 container OpenSCAD skips are existing CAD regression checks. They are not standalone acceptance. S2.8 adds no CAD gate.

## Repeatability

Each case ran three times through the console-script `main()` at one destination path, with `--preview --debug-bundle`.
Run one had no `--overwrite`. Runs two and three used `--overwrite`. The test saved all bundle bytes after each run, before the next run started.
It compared all three saved results. It did not compare the third run with itself.

Mask mode: `--input-kind mask --width-mm 100`. Edge mode: real `st-canny-map` output with default thresholds, then `--input-kind edges --width-mm 100 --line-width-mm 0.8`.
Corpus mask failures equal the locked S1.4 results at `filter_speckle` 4. Topology fixtures keep their expected component and hole counts.
Edge-mode geometric failures compare the vector render with the prepared edge band. They are measurements, not acceptance thresholds.
Run seconds are in-process wall time on the local host, including preview rendering.

| Mode | Case | Paths | Commands | Geometric failures | XOR px | Warnings | Run seconds (1, 2, 3) |
|---|---|---:|---:|---|---:|---|---|
| mask | asymmetric | 2 | 12 | none | 0 | none | 0.021, 0.013, 0.012 |
| mask | border_touching | 2 | 15 | none | 0 | none | 0.013, 0.013, 0.012 |
| mask | donut | 1 | 46 | none | 84 | none | 0.016, 0.014, 0.015 |
| mask | nested_island | 2 | 84 | none | 140 | none | 0.017, 0.015, 0.016 |
| mask | separate_components | 3 | 27 | none | 20 | none | 0.014, 0.013, 0.013 |
| mask | badge | 6 | 156 | none | 509 | thin_material, thin_background | 0.028, 0.027, 0.027 |
| mask | bracket | 6 | 147 | centroid | 240 | thin_material, thin_background | 0.024, 0.022, 0.022 |
| mask | brick | 1 | 703 | centroid, topology | 2807 | thin_material, thin_background | 0.034, 0.034, 0.033 |
| mask | coins | 24 | 341 | none | 374 | thin_material, thin_background | 0.025, 0.024, 0.024 |
| mask | gravel | 209 | 2808 | boundary, topology | 3944 | thin_material, thin_background | 0.054, 0.053, 0.054 |
| mask | horse | 1 | 132 | centroid, topology | 603 | thin_material, thin_background | 0.024, 0.023, 0.022 |
| mask | pcb | 30 | 1082 | none | 672 | thin_background | 0.030, 0.030, 0.029 |
| mask | truchet | 55 | 2354 | none | 3616 | thin_material, thin_background | 0.047, 0.051, 0.048 |
| edges | badge | 8 | 328 | none | 1043 | requested_width_below_four_pixels, thin_material, thin_background | 0.034, 0.031, 0.030 |
| edges | bracket | 6 | 172 | centroid | 425 | requested_width_below_four_pixels, thin_material, thin_background | 0.029, 0.025, 0.024 |
| edges | brick | 1 | 749 | centroid, topology | 2523 | requested_width_below_four_pixels, thin_material, thin_background, degenerate_subpaths | 0.038, 0.036, 0.034 |
| edges | coins | 25 | 1338 | bounds, scale, topology | 1143 | requested_width_below_four_pixels, thin_material, thin_background, degenerate_subpaths | 0.035, 0.032, 0.033 |
| edges | gravel | 1 | 3230 | topology | 1950 | requested_width_below_four_pixels, thin_material, thin_background, degenerate_subpaths | 0.050, 0.059, 0.048 |
| edges | horse | 1 | 238 | centroid, topology | 1061 | requested_width_below_four_pixels, thin_material, thin_background, degenerate_subpaths | 0.029, 0.026, 0.026 |
| edges | pcb | 94 | 2331 | topology | 2470 | requested_width_below_four_pixels, thin_material, thin_background | 0.041, 0.038, 0.038 |
| edges | truchet | 53 | 3808 | topology | 3759 | requested_width_below_four_pixels, thin_material, thin_background, degenerate_subpaths | 0.057, 0.054, 0.054 |

## Coordinates, composition and failures

| Check | Exit | Evidence |
|---|---:|---|
| EXIF orientation 6 | 0 | Original 120x128, oriented 128x120. Prepared material equals the oriented source. Vector XOR 0 px. Every flip is worse. |
| Constraint mask with other dimensions | 2 | Message names the oriented dimensions. No output files. |
| Common resize, `--max-res 64` with include mask | 0 | Processed 64x64, 1.5625 mm/px. Material and vector stay in the resized include region. |
| Physical rectangle, 200x100 px at `--width-mm 50` | 0 | Root 50 mm x 25 mm. Rendered rectangle 25 mm x 12.5 mm at origin (10 mm, 5 mm). |
| Asymmetric alignment | 0 | XOR 0 px. Left-right flip 3712 px, up-down flip 3200 px, transpose 5504 px. |
| Exclusion wins over inclusion | 0 | 1200 source material px in the excluded region. Prepared material and vector have none there. |
| Clipping after edge expansion, 6 mm line | 0 | Kernel 9 px, radius 4 px. Material and vector stay inside the include mask. |
| Empty material, black source | 1 | "The traced SVG is empty". No SVG, preview or manifest. Requested debug files retained. |
| Empty material, single-pixel islands only | 1 | `filter_speckle` 4 removes them. Same message. No SVG, preview or manifest. |
| Malformed upstream SVG (`script` element) | 1 | "unsupported element script". No bundle. |
| `--max-paths 2` | 1 | "path count at least 3 exceeds limit 2". No bundle. |
| `--max-svg-bytes 200` | 1 | "SVG size 509 bytes exceeds limit 200". No bundle. |
| `--max-path-commands 10` | 1 | "path command count at least 11 exceeds limit 10". No bundle. |
| Existing `out.svg`, `out.vector.json`, `out.preview.png` or `out.debug/mask.png` | 2 | Each refuses without `--overwrite`. The existing bytes stay unchanged. |
| Interrupted `--overwrite` (preview rename fails) | 1 | Manifest absent. The new SVG landed before the failure. Spec 8 says several renames are not atomic. |
| Real command equals in-process bundle (donut, asymmetric) | 0 | Byte-equal bundles from the source script and from the installed `st-vector-map`. |

Three mutations of raster preparation were each caught by exactly one check: no EXIF transpose, no clip after expansion, and inclusion over exclusion.

## Installed package and sibling entry points

```bash
python -m venv <scratch>/venv
<scratch>/venv/bin/pip install "./scripts[vector]"     # exit 0
<scratch>/venv/bin/python -c "import torch"            # ModuleNotFoundError
<scratch>/venv/bin/st-vector-map tests/fixtures/vector_map/donut.png <scratch>/inst/donut.svg \
  --input-kind mask --width-mm 40 --preview --json     # exit 0, status converted
<scratch>/venv/bin/st-canny-map --help                 # exit 0
<scratch>/venv/bin/st-resize-for-model --help          # exit 0
<scratch>/venv/bin/st-depth-map --help                 # exit 1: needs torch, install the depth extra
<scratch>/venv/bin/st-pose-map --help                  # exit 1: needs torch, install the pose extra
```

The depth and pose results are the S2.1 contract for a vector-only install. They are not failures.

## Byte identity across environments

| Comparison | SVG | Recipe | Debug mask PNG | Preview PNG | Manifest |
|---|---|---|---|---|---|
| Canny from installed venv (opencv 5.0.0.93) vs host (4.11.0.86). Conversions from source under the conda interpreter in both | equal, 21 of 21 | equal | equal | equal | mask equal. Edge differs: it records the per-session Canny path |
| Local macOS vs Linux container | equal, 21 of 21 | equal | differs | differs | differs (it records PNG hashes) |

PNG bytes differ across platforms. Decoded pixels were not compared. A different zlib build is the likely cause, but this run did not prove it.
Spec 8 does not promise identical bytes across native library versions. Repeatability holds within one environment.

## Known failures

Full isolated suite, local: `tests/test_entry_spans.py::test_an_UNHASHABLE_type_does_not_break_the_span`. It also fails on `main`. `STABL-fjmjolci` tracks it.

Native container, 10 failures:

- The same entry-span test.
- 8 tests fail with `FileNotFoundError`. They need repository files that `Dockerfile.test` does not copy into the image: `docs/observability-contract.md`, `env.dev` and `env.prod`. Affected tests: `test_log_format.py` (3), `test_logging_env_contract.py` (4), `test_metrics.py` (1). `STABL-tlixascy` tracks them.
- `test_log_levels.py::test_every_tracking_logger_actually_carries_LOG_LEVEL` fails with `assert 'DEBUG' == 'INFO'`. It is not a missing-file failure. It depends on test order and on `LOG_LEVEL=DEBUG`, which the container gets from `env.dev`.
  `test_env_accessor.py::test_a_QUOTED_LOG_LEVEL_does_not_break_dictConfig` deletes `LOG_LEVEL` and reloads `server.logging_config`, so `LOG_LEVEL` becomes `INFO`. The log-levels test keeps the `LOGGING_CONFIG` it imported at collection, which still has `DEBUG`.
  Reproduction: container, test alone, 1 passed. Container, QUOTED test first, 1 failed. Host with `LOG_LEVEL=DEBUG`, QUOTED test first, 1 failed. `STABL-gwrotiyb` tracks it.

Classification against `main`: `docker compose -f docker-compose.test.yml run --rm test` from the main checkout at `363b5ac` rebuilt the image from `main` and gave the same 10 failures, exit 1. Cohort 1: 10 failed, 2257 passed, 27 skipped. The 41-test difference in passes is the new acceptance file.

## Not proven

- CAD or physical-print acceptance. S2.8 has no CAD gate.
- Decoded PNG pixel identity across platforms.
- Byte identity for other native library versions, and for x86-64 or CUDA images.
- A full-suite pass. Both full runs exit 1 for the reasons above.
