# Vector-map S4.3 regression and installation report

Issue: `STABL-asohzavp`. Plan: `docs/superpowers/plans/2026-10-17-vector-map-s4-3.md`.
Code base: `main` at `d0a8405` (PR #120 merged) plus one test commit on `feat/s4-3-regression`.
Run date: 2026-10-18. Writer: Theta.

## Verdict

- Proven (alpha support claim): `st-vector-map` and `st-canny-map` convert a real image on CPython 3.12, on macOS arm64 and Fedora 44 x86_64.
- Installed only, not proven: depth, pose, and all. They install on both hosts and `--help` runs. Inference never ran. On the Mac, `pip check` fails for pose and all.
- Refused: production vector conversion on CPython 3.14. The refusal works on both hosts.
- Unverified: CPython 3.10, 3.11, 3.13, depth or pose inference, and depth, pose, or all on 3.14. No run exists for them.
- Two complete 16-row corpus runs agree byte for byte on every published artifact.
- `vtracer==1.0.0a4` is rejected for this release. Its API lacks `convert_raw_image_to_svg`.
- No physical print claim. CLI JSON keeps `physical_print: pending`.

## 1. Python policy (STABL-mlahbjjj)

The guard from PR #120 holds on real CPython 3.14 on both hosts (section 3.2).
It refuses before `import vtracer`. Stderr and JSON carry the plain message with no `VTracer ... layer` prefix.
I did not rerun the unguarded 0.6.15 call. The native crash stays documented only on enigma at 3.14.7.
The Mac refusal is precautionary. No native Mac crash was shown.

## 2. Repeatability

### 2.1 New test

`test_layered_conversion_repeats_across_independent_directories` in `tests/test_vector_map_acceptance.py`.
It converts one layered image into two empty output directories from shared inputs.
It requires identical bytes for every bundle file (combined SVG, three layer SVGs, manifest, preview, debug masks, recipe) and equal JSON status, counts, diagnostics, and evidence.
It passed on first valid run. The property already held. No product code changed.
My first draft used separate input directories and failed. The manifest records input paths as given. That is correct behavior, not a defect.
Existing tests already cover three runs at one destination for every mask and edge case.

Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts='' tests/test_vector_map_acceptance.py -q` → 42 passed.

### 2.2 Full corpus, two independent runs

Host: macOS 26.0.1 arm64, CPython 3.12.13 (conda env `stability-toys`), vtracer 0.6.15, resvg-py 0.5.0.
OpenSCAD: `OPENSCAD=/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD`, `--version` printed `OpenSCAD version 2021.01`.

```bash
python -m tests.vector_map_s4_1_corpus_probe --work-dir RUN1 --index-json run1.json --report run1.md
python -m tests.vector_map_s4_1_corpus_probe --work-dir RUN2 --index-json run2.json --compare-index run1.json --report run2.md
```

Both runs used empty directories outside the repository. Both exited 0.

| Check | Run 1 | Run 2 |
|---|---|---|
| Full selection (8 cases x 2 modes) | yes | yes |
| `evidence_complete` | true | true |
| `fidelity_matches_pin` | true | true |
| `fidelity_modes_agree` | true | true |
| Manifest hash check | 152 checked, 0 mismatches | 152 checked, 0 mismatches |
| CAD pass | 16 of 16 | 16 of 16 |
| Regeneration compare against run 1 | n/a | passed, no difference in any row |

The compare excludes `out.stl`, `area-*`, and `mesh.png`.
The four known fidelity failures reproduced unchanged: bracket (centroid), brick (centroid, topology), gravel (boundary, topology), horse (centroid, topology).
They are not regressions. S4.1 pinned them.

| Case | Mode | Fidelity failures | CAD | SVG sha256 (run 1) | Run 2 |
|---|---|---|---|---|---|
| badge | raised | none | pass | `14aa7d2a4857f8e5` | same |
| badge | engraved | none | pass | `14aa7d2a4857f8e5` | same |
| bracket | raised | centroid | pass | `a9f7da4e395ffd00` | same |
| bracket | engraved | centroid | pass | `a9f7da4e395ffd00` | same |
| brick | raised | centroid, topology | pass | `2a23a85856142aea` | same |
| brick | engraved | centroid, topology | pass | `2a23a85856142aea` | same |
| coins | raised | none | pass | `ebe710885a3b8fa7` | same |
| coins | engraved | none | pass | `ebe710885a3b8fa7` | same |
| gravel | raised | boundary, topology | pass | `1071b2c441d9146a` | same |
| gravel | engraved | boundary, topology | pass | `1071b2c441d9146a` | same |
| horse | raised | centroid, topology | pass | `7cd01393a4ad97dd` | same |
| horse | engraved | centroid, topology | pass | `7cd01393a4ad97dd` | same |
| pcb | raised | none | pass | `1b6388b710fa39a0` | same |
| pcb | engraved | none | pass | `1b6388b710fa39a0` | same |
| truchet | raised | none | pass | `d25c4decb947c418` | same |
| truchet | engraved | none | pass | `d25c4decb947c418` | same |

Full SVG hashes are in the run indexes. I did not commit the indexes.
S4.2 added vertices and evidence fields. I compared run 2 to run 1 only, not to the old S4.1 manifests.

## 3. Installation matrix

Durable records: `docs/superpowers/reports/2026-10-18-vector-map-s4-3-regression.evidence/` (see section 7). Each cell has its full `pip freeze`, `pip check` output, and tool output. `matrix/summary.json` holds the exit codes.

Method: `cell.sh` per cell. Fresh `venv`, then the exact command `python -m pip install './scripts[EXTRA]'` from a copy of the repository tree, then `pip freeze`, `pip check`, entry-point listing, Torch presence, and a real run of each tool the extra provides.
The copy lives outside the repository so `build/` and `egg-info` stay out of the work tree.
Test input: `tests/fixtures/vector_map/donut.png`. Vector command: `--input-kind mask --width-mm 100`.

### 3.1 CPython 3.12

Column meaning: Install is `pip install` exit. `pip check` is its exit. Run is a real conversion of the fixture. Help is `--help` only and proves nothing about inference.

| Host | Extra | Install | `pip check` | Torch | `st-vector-map` | `st-canny-map` | depth, pose `--help` |
|---|---|---|---|---|---|---|---|
| Mac arm64 | vector | ok | ok | absent | converts | converts | torch hint |
| Mac arm64 | canny | ok | ok | absent | refuses (no vtracer) | converts | torch hint |
| Mac arm64 | depth | ok | ok | present | n/a | converts | ok |
| Mac arm64 | pose | ok | **fails**: mediapipe 0.10.14 "not supported on this platform" | present | n/a | converts | ok |
| Mac arm64 | all | ok | **fails**: same mediapipe message | present | refuses (no vtracer) | converts | ok |
| Mac arm64 | `make ... EXTRAS=vector` | ok | ok | absent | converts | converts | torch hint |
| enigma x86_64 | vector | ok | ok | absent | converts | converts | torch hint |
| enigma x86_64 | canny | ok | ok | absent | refuses (no vtracer) | converts | torch hint |
| enigma x86_64 | depth | ok | ok | present | n/a | converts | ok |
| enigma x86_64 | pose | ok | ok | present | n/a | converts | ok |
| enigma x86_64 | all | ok | ok | present | refuses (no vtracer) | converts | ok |
| enigma x86_64 | `make ... EXTRAS=vector` | ok | ok | absent | converts | converts | torch hint |

Notes:

- `all` does not install VTracer. The refusal in those rows is the designed behavior.
- Mac `pose` and `all`: `pip check` **fails** with "mediapipe 0.10.14 is not supported on this platform". `import mediapipe` works and `st-pose-map --help` runs. I did not run pose inference. These two cells are not a support claim.
- Depth and pose ran only to `--help`. No model download or inference ran. Their output is unverified on both hosts. The depth and pose rows show install and help results only.
- `make install-controlnet-scripts EXTRAS=vector` matches the direct pip result on both hosts. Resolved packages are identical.
- Torch came from pip on both hosts. The documented conda-first Mac step was not needed. Torch 2.14.1 satisfied the `torch>=2.1` pin.
- Key versions: vtracer 0.6.15, resvg-py 0.5.0, opencv-python-headless 5.0.0.93, mediapipe 0.10.14, torch 2.14.1, transformers 5.20.1.
- Interpreters: Mac conda-forge CPython 3.12.15. enigma uv CPython 3.12.12 (Clang build).

### 3.2 CPython 3.14

| Host | Interpreter | Extra | Install | `st-vector-map` | Adapter call | `vtracer` imported | Files created | `st-canny-map` |
|---|---|---|---|---|---|---|---|---|
| Mac arm64 | conda-forge 3.14.8 | vector | ok | exit 1, refusal, `failed` JSON | `RuntimeError` refusal | no | none | converts |
| Mac arm64 | conda-forge 3.14.8 | canny | ok | exit 1 (no vtracer) | n/a | n/a | n/a | converts |
| enigma x86_64 | system 3.14.8 | vector | ok | exit 1, refusal, `failed` JSON | `RuntimeError` refusal | no | none | converts |
| enigma x86_64 | system 3.14.8 | canny | ok | exit 1 (no vtracer) | n/a | n/a | n/a | converts |

Every refusal ran with `python -X faulthandler`, `--debug-bundle --json`. Each exited with an ordinary code 1. No signal, no native crash, no output directory content.
depth, pose, and all on 3.14: not tested. Marked unverified.

## 4. `vtracer==1.0.0a4` (option 3)

Installed alone, with numpy and Pillow, in disposable venvs. Hosts: Mac conda 3.12.15 and 3.14.8, enigma uv 3.12.12 and system 3.14.8.

- Public API: `Config`, `convert_bytes`, `convert_file`, `convert_pixels`. `convert_raw_image_to_svg` is absent on all four.
- **Interface gate: failed.** The 0.6.15 adapter calls `convert_raw_image_to_svg`. The adapter cannot run on 1.0.0a4 without a migration.
- Supplementary smoke call, not required by the plan: `convert_pixels` on a 16x16 binary RGBA square, in a timed child with `-X faulthandler`, on all four cells. Each returned an SVG with one path. No crash on 3.14 on either host.
- Because the gate failed, I did not run the topology, adapter, CAD, or repeatability fixtures with the candidate.
- Verdict: reject for this release. A later change owns the migration and pin. It needs its own review. The smoke result suggests the 3.14 crash is fixed upstream, but one tiny call does not prove it.

## 5. Deviations and incidents

1. **Interpreters on the Mac.** Homebrew CPython 3.12.14 and 3.14.7 cannot run `pip install`. Their `pyexpat` fails to load: it needs a newer `libexpat` than `/usr/lib/libexpat.1.dylib` provides. `venv` also fails under the harness because `platform.mac_ver()` returns empty. I used conda-forge CPython 3.12.15 and 3.14.8 instead. The plan named Homebrew. Cost if wrong: the Mac row describes conda-forge builds, not Homebrew. The Task 1 child proof in PR #120 did use the Homebrew 3.14.7 child.
2. **enigma 3.14 is 3.14.8**, not 3.14.7. The documented crash was on 3.14.7. The guard covers all 3.14 and later.
3. **Shared env change, repaired.** My first matrix script fell back to the Miniforge base interpreter after `venv` failed. It reinstalled `st-controlnet-helpers` 0.1.0 there and added `vtracer` and `resvg-py`. I uninstalled `vtracer` and `resvg-py`. The base env now holds the current `st-controlnet-helpers` build (with `st-vector-map`). Before, it held an older build. The script now stops when `venv` fails.
4. **Rerun of the matrix.** The first pass had two harness faults: a wrong fixture path on enigma and a missing `Makefile.test` for the `make` cells. I reran every cell. The tables above come from the second pass only.
5. **Seven Mac CAD failures** reported earlier came from the wrong OpenSCAD (2026.10.11). With 2021.01 the corpus is green.

## 6. Limits

- CPython 3.10, 3.11, 3.13: unverified.
- Depth and pose inference, and `all` on 3.14: unverified.
- Two hosts only. Other platforms and Windows: unverified.
- Corpus CAD ran on the Mac only. enigma lacks a native OpenSCAD on `PATH`.
- Mac `pose`: `pip check` reports a platform message. Treat pose on Mac as installable, not proven.
- Print claims: none. See the S4.5 print report for the separate trial.

## 7. Durable evidence

Directory: `docs/superpowers/reports/2026-10-18-vector-map-s4-3-regression.evidence/`. Host paths and the login name are scrubbed. A scan found none.

- `corpus/run1.json`, `corpus/run2.json`: per row, the full SHA-256 and size of every artifact, plus source, mask, and manifest hashes, recipe, fidelity, and CAD verdict. Also host, OpenSCAD version, selection, manifest hash check, and run 2 regeneration compare. They omit work-directory paths and STL-adjacent paths.
- `matrix/<host>-<python>-<extra>/`: `freeze.txt` (full `pip freeze`), `check.txt`, `key.txt`, `entrypoints.txt`, `torch.txt`, and the tool, guard, and adapter output. `matrix/summary.json` lists exit codes for every cell. `file://` paths in `freeze.txt` are replaced by `<scratch>`.
- `candidate/<host>-cand-<python>/`: `freeze.txt`, `probe.out` (API listing), `smoke.out`, and exit codes.
- `candidate/wheels.json`: exact wheels and SHA-256.
  - `vtracer-1.0.0a4-cp38-abi3-macosx_11_0_arm64.whl`: `d120dcee58ec8c92a303b46d3cb141c9bdb891dc2234bebf67373e130fd2f974`
  - `vtracer-1.0.0a4-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`: `94fb2737bce7d66f82981de25cf9fd38685f02c5a0399354e536837c501a6007`
  - The hash is of the file `pip download` fetched on each host after the probe. The install logs show the same filename. The wheel is abi3, so one file serves 3.12 and 3.14 per platform.
- Gap: I did not keep the raw work directories or the STL files. Anyone can regenerate them with the commands in section 2.2.
