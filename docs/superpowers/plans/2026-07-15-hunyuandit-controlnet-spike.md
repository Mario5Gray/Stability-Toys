# HunyuanDiT ControlNet Spike Implementation Plan

> **For agentic workers:** Execute via superpowers:executing-plans inline. Do NOT
> use subagent-driven development (forbidden by AGENTS.md). Checkboxes are step
> markers, not tracking authority — waveplan/FP own task state (STABL-ichgkgno).

**Goal:** Prove the HunyuanDiT ControlNet stack imports, loads via the production
`from_pipe` shape, and generates a Canny-conditioned image on real CUDA, recording
dep pins and peak VRAM.

**Architecture:** One throwaway script in `spikes/` with three observable stages —
import gate, `from_pipe` composition, single generation + VRAM capture. No imports
from `server/` or `backends/`; the spike validates the upstream diffusers stack
only. Spec: `docs/superpowers/specs/2026-07-15-hunyuandit-controlnet-spike-design.md`.

**Tech Stack:** Python, diffusers (`HunyuanDiTPipeline`, `HunyuanDiT2DControlNetModel`,
`HunyuanDiTControlNetPipeline`), transformers (BERT + mT5 encoders), torch CUDA, PIL.

## Global Constraints

- FP issue: `STABL-ichgkgno`; every commit message includes it plus the next step.
- Throwaway code: lives under `spikes/`, never imported by production code; TDD is
  relaxed per spec — verification is staged observable runs, not unit suites.
- Local commands use Miniforge base: `source /Users/darkbit1001/miniforge3/bin/activate base`, then `python`.
- CUDA execution only via the split test container on a linux/amd64 + NVIDIA host:
  `docker compose -f docker-compose.test.yml build test-cuda` / `run --rm test-cuda`.
- Model pins (exact): base `Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers`, control
  `Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny`, `torch_dtype=fp16`,
  1024×1024, `use_resolution_binning=True`, control map passed as `control_image=`.
- Exit codes are the script's contract: `0` success, `2` import gate failed,
  `3` no CUDA device.
- Known local baseline (mac, Miniforge): diffusers 0.37.0 + transformers 5.10.2
  fails the import gate on `BertModel`. The mac run is expected to exit `2`; that
  is the local verification signal, not a defect.

---

### Task 1: Spike script — CLI + import gate

**Files:**
- Create: `spikes/hunyuandit_controlnet_spike.py`
- Create: `spikes/README.md`

**Interfaces:**
- Consumes: nothing (standalone).
- Produces: `python spikes/hunyuandit_controlnet_spike.py --imports-only` →
  prints `diffusers=<v> transformers=<v>`, then either `import gate: OK` (exit 0)
  or `IMPORT GATE FAILED: <exc>` (exit 2). Task 2 extends `main()` past the gate.

- [ ] **Step 1: Write the script with argparse and the import gate**

```python
#!/usr/bin/env python
"""Throwaway spike: prove HunyuanDiT ControlNet imports, composes via from_pipe,
and generates a Canny-conditioned image on CUDA.

FP: STABL-ichgkgno
Spec: docs/superpowers/specs/2026-07-15-hunyuandit-controlnet-spike-design.md

NOT production code. Must not import from server/ or backends/.
Exit codes: 0 success, 2 import gate failed, 3 no CUDA device.
"""
from __future__ import annotations

import argparse
import sys

BASE_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers"
CANNY_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny"


def stage(msg: str) -> None:
    print(f"[spike] {msg}", flush=True)


def import_gate() -> dict:
    """First pass gate: record pins, then import the HunyuanDiT ControlNet stack.

    Diffusers lazy-loads pipeline modules, so the from-import below is what
    triggers the transformers BertModel/T5EncoderModel resolution that broke
    locally (transformers 5.10.2). Exit 2 with the versions already printed.
    """
    import diffusers
    import transformers

    stage(f"diffusers={diffusers.__version__} transformers={transformers.__version__}")
    try:
        from diffusers import (  # noqa: F401
            HunyuanDiT2DControlNetModel,
            HunyuanDiTControlNetPipeline,
            HunyuanDiTPipeline,
        )
        from transformers import BertModel, T5EncoderModel  # noqa: F401
    except Exception as exc:  # broad by design: any import failure fails the gate
        stage(f"IMPORT GATE FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
    stage("import gate: OK")
    return {
        "base_pipeline_cls": HunyuanDiTPipeline,
        "controlnet_model_cls": HunyuanDiT2DControlNetModel,
        "controlnet_pipeline_cls": HunyuanDiTControlNetPipeline,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true",
                        help="run the import gate and exit")
    parser.add_argument("--control-map", default=None,
                        help="path to a Canny control-map PNG (default: synthesized)")
    parser.add_argument("--prompt", default="a photograph of a cat, high quality, detailed")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="spike_hunyuandit_out.png")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import_gate()
    if args.imports_only:
        return 0
    stage("generation path not implemented yet (Task 2)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Write `spikes/README.md`**

```markdown
# spikes/

Throwaway proof scripts. Nothing here is production code; nothing here may be
imported by `server/` or `backends/`. Each spike names its FP issue and spec.

- `hunyuandit_controlnet_spike.py` — STABL-ichgkgno; run recipe + pass criteria
  in `docs/superpowers/specs/2026-07-15-hunyuandit-controlnet-spike-design.md`.
```

- [ ] **Step 3: Verify the gate locally (expected failure = the signal)**

Run:
```bash
source /Users/darkbit1001/miniforge3/bin/activate base
cd /Users/darkbit1001/workspace/Stability-Toys
python spikes/hunyuandit_controlnet_spike.py --imports-only; echo "exit=$?"
```
Expected output (mac baseline):
```
[spike] diffusers=0.37.0 transformers=5.10.2
[spike] IMPORT GATE FAILED: RuntimeError: Failed to import ... BertModel ...
exit=2
```
Exit MUST be 2 (not an unhandled traceback), and both versions MUST print
before the failure line.

- [ ] **Step 4: Commit**

```bash
git add spikes/hunyuandit_controlnet_spike.py spikes/README.md
git commit -m "feat(spike): HunyuanDiT ControlNet spike import gate + CLI (STABL-ichgkgno) — next: from_pipe load + generation path"
```

---

### Task 2: Load, compose via from_pipe, generate, capture VRAM

**Files:**
- Modify: `spikes/hunyuandit_controlnet_spike.py` (extend `main()`, add functions)

**Interfaces:**
- Consumes: `import_gate()` return dict (`base_pipeline_cls`,
  `controlnet_model_cls`, `controlnet_pipeline_cls`) and `parse_args` fields
  (`control_map`, `prompt`, `steps`, `seed`, `out`) from Task 1.
- Produces: full-run behavior for Task 3 — on CUDA, saves `--out` PNG and prints
  `peak VRAM: <n> GiB`; exits 3 when CUDA is absent.

- [ ] **Step 1: Add control-map loading with a synthesized default**

Add after `import_gate()`:

```python
def load_control_map(path: str | None, size: tuple[int, int] = (1024, 1024)):
    """Load the Canny map, or synthesize a geometric edge image (white lines on
    black — a valid canny-style conditioning input) when no path is given."""
    from PIL import Image, ImageDraw

    if path is not None:
        img = Image.open(path).convert("RGB").resize(size)
        stage(f"control map: {path} -> {img.size}")
        return img
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], outline=255, width=4)
    draw.ellipse([w // 3, h // 3, 2 * w // 3, 2 * h // 3], outline=255, width=4)
    draw.line([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=255, width=4)
    stage("control map: synthesized geometric edges (rect + ellipse + diagonal)")
    return img.convert("RGB")
```

- [ ] **Step 2: Add the CUDA run — production `from_pipe` shape + VRAM capture**

Add after `load_control_map`, and replace `main()`:

```python
def run(classes: dict, args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        stage("no CUDA device — spike requires the test-cuda container on an NVIDIA host")
        return 3

    device = "cuda"
    dtype = torch.float16

    stage(f"loading base: {BASE_REPO} (fp16)")
    base = classes["base_pipeline_cls"].from_pretrained(BASE_REPO, torch_dtype=dtype)
    base.to(device)

    stage(f"loading controlnet: {CANNY_REPO} (fp16)")
    controlnet = classes["controlnet_model_cls"].from_pretrained(
        CANNY_REPO, torch_dtype=dtype
    )

    stage("composing via from_pipe (production load shape)")
    pipe = classes["controlnet_pipeline_cls"].from_pipe(base, controlnet=controlnet)
    pipe.to(device)

    control_image = load_control_map(args.control_map)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    torch.cuda.reset_peak_memory_stats()
    stage(f"generating: steps={args.steps} seed={args.seed} 1024x1024 binning=True")
    result = pipe(
        prompt=args.prompt,
        control_image=control_image,  # the one per-family kwarg divergence
        height=1024,
        width=1024,
        num_inference_steps=args.steps,
        use_resolution_binning=True,
        generator=generator,
    )
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    result.images[0].save(args.out)
    stage(f"saved: {args.out}")
    stage(f"peak VRAM: {peak_gib:.2f} GiB")
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    classes = import_gate()
    if args.imports_only:
        return 0
    return run(classes, args)
```

- [ ] **Step 3: Verify locally — compile + gate behavior unchanged**

Run:
```bash
source /Users/darkbit1001/miniforge3/bin/activate base
cd /Users/darkbit1001/workspace/Stability-Toys
python -m py_compile spikes/hunyuandit_controlnet_spike.py && echo "compile OK"
python spikes/hunyuandit_controlnet_spike.py --imports-only; echo "exit=$?"
```
Expected: `compile OK`; gate run still prints both versions and exits 2 on the
mac baseline.

- [ ] **Step 4: Verify the synthesized control map locally (PIL path only)**

Run:
```bash
python - <<'EOF'
import sys
sys.path.insert(0, "spikes")
from hunyuandit_controlnet_spike import load_control_map
img = load_control_map(None)
assert img.size == (1024, 1024) and img.mode == "RGB", (img.size, img.mode)
print("synth control map OK")
EOF
```
Expected: `[spike] control map: synthesized ...` then `synth control map OK`.

- [ ] **Step 5: Commit**

```bash
git add spikes/hunyuandit_controlnet_spike.py
git commit -m "feat(spike): HunyuanDiT from_pipe load + generation + VRAM capture (STABL-ichgkgno) — next: execute on NVIDIA host, record results"
```

---

### Task 3: Execute on the NVIDIA host and record the verdict

**Files:**
- Modify: none (execution + FP recording only)

**Interfaces:**
- Consumes: the complete script from Tasks 1–2; run recipe from the spec.
- Produces: pass/fail verdict + recorded pins/VRAM on `STABL-ichgkgno`, gating
  the full-family spec.

- [ ] **Step 1: Build and run in the test-cuda container on the remote host**

On the linux/amd64 + NVIDIA host, from the repo root:
```bash
docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
    python spikes/hunyuandit_controlnet_spike.py --out /tmp/spike_hunyuandit_out.png
```
Expected sequence:
```
[spike] diffusers=<v> transformers=<v>
[spike] import gate: OK            <- pass gate 1; exit 2 here = pin resolution is in-scope for the plan
[spike] loading base: ...
[spike] loading controlnet: ...
[spike] composing via from_pipe (production load shape)   <- pass gate 2 if no dtype/VAE error
[spike] control map: synthesized ...
[spike] generating: ...
[spike] saved: /tmp/spike_hunyuandit_out.png
[spike] peak VRAM: <n> GiB         <- pass gate 4
```
Copy the output PNG off the host and eyeball it: a coherent image whose
composition follows the rect/ellipse/diagonal edges (pass gate 3).

- [ ] **Step 2: Record the verdict on FP**

```bash
fp comment STABL-ichgkgno "SPIKE RESULT: import gate <OK/FAILED @ pins diffusers=X transformers=Y>; from_pipe <OK/error>; image <coherent-canny-conditioned/verdict>; peak VRAM <n> GiB.
GO/NO-GO: <go — open full-family spec per brainstorm ewgzdmdnfumoczdfyergbyghcwsggdij / no-go — reason + next action (e.g. pin resolution task)>."
```
Fill every angle-bracket field from the actual run output. If the import gate
failed in the container, the spike still delivered its highest-value finding:
dependency-pin resolution becomes an explicit work item for the full-family plan.

- [ ] **Step 3: Commit any run artifacts worth keeping (optional) and close out**

If the eyeballed PNG is worth preserving for the full-family spec discussion,
add it under `docs/superpowers/specs/assets/` and commit:
```bash
git add docs/superpowers/specs/assets/spike_hunyuandit_out.png
git commit -m "docs(spike): HunyuanDiT spike output artifact (STABL-ichgkgno) — next: full-family spec go/no-go"
```
Otherwise skip — the FP comment is the record of truth.

---

## Self-Review Notes

- Spec coverage: import gate (Task 1), from_pipe/base/control load + `control_image=`
  + 1024/binning + VRAM capture (Task 2), remote run recipe + pass criteria + FP
  recording (Task 3). Canny-only and record-only VRAM honored. Non-goals untouched.
- Exit-code contract stated once (Global Constraints) and implemented in Tasks 1–2.
- Type consistency: `import_gate() -> dict` keys match `run(classes, ...)` reads;
  `parse_args` fields match `run`'s uses.
