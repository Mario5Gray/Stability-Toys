# qrandom — quantum seeds with a gesture of intent

`qrandom` (invoked as `qrng`) fetches a seed from a source of genuine physical
randomness. It is a standalone toy: it lives at the repo root, depends only on
the Python standard library (plus optional `qiskit` for the IBM source), and is
**not** wired into the backend's generation-seed path. Nothing in the render
pipeline calls it.

It exists to answer a specific wish — *a seed that captures the randomness of a
moment* — with real quantum measurement rather than a clock, and to carry a
`--intent` string that is deliberately, provably discarded.

---

## Quick start

```console
$ qrng
Specify source of Quantum Randomness
  use --source anu | nist | ibm
  use --help for a guided help
  use --intent "string of intent"

$ qrng --source nist
10948757345347754095912007625123495526843168941517440012491173007537551597...

$ qrng --source anu --intent "let this render find its face"
14203391556002019284

$ qrng --source ibm --intent "collapse toward the image i mean" --json
{ ... }
```

Install the `qrng` command onto your PATH:

```bash
make install-qrng        # symlinks ~/.local/bin/qrng -> ./qrandom (edits stay live)
make install-qrng-ibm    # the above, plus qiskit for --source ibm
make install             # qrng alongside st + controlnet scripts
make uninstall-qrng      # remove the symlink
```

Because it is a symlink, edits to `./qrandom` take effect immediately — no
reinstall. Or alias it manually:

```bash
alias qrng='/Users/darkbit1001/workspace/Stability-Toys/qrandom'
```

---

## The three sources

| Source | Physics | Bits | Latency | Needs |
|--------|---------|------|---------|-------|
| `anu`  | Optical measurement of vacuum field fluctuations | 64  | instant | `ANU_API_KEY` (recommended) |
| `nist` | NIST Randomness Beacon quantum pulse, signed + timestamped | 512 | instant | nothing |
| `ibm`  | Real QPU: Hadamard + measurement collapse, debiased into a local pool | 64 | one job up front, then instant | `IBM_QUANTUM_TOKEN` + `qiskit` |

### `anu` — vacuum fluctuations

ANU's quantum RNG measures the quantum noise of a vacuum electromagnetic field.
The current API at `api.quantumnumbers.anu.edu.au` requires a key
(`ANU_API_KEY`); register at <https://quantumnumbers.anu.edu.au>. Without a key,
`qrng` falls back to the keyless legacy endpoint at `qrng.anu.edu.au`, which is
rate-limited and may refuse — you'll get a clear message telling you to set the
key.

### `nist` — the beacon

The [NIST Randomness Beacon](https://beacon.nist.gov) emits a 512-bit
full-entropy pulse every 60 seconds, generated from quantum measurements, then
signs and publicly archives it. `qrng --source nist` returns the most recent
pulse and, with `--json`, its `timestamp` and `pulseIndex` — the exact moment
those bits were minted.

Two honest caveats:

- **Public.** Anyone can look up the same value for any past minute. These seeds
  are auditable and provably tied to an instant, but not secret.
- **Minute-coarse.** Every call inside the same wall-clock minute returns the
  *same* pulse. For per-image uniqueness this is a liability; for
  "provably-from-this-moment" provenance it is the whole point.

### `ibm` — a real quantum computer, pooled

IBM Quantum jobs queue for **minutes to hours** on real hardware, so a per-call
round trip is impossible. `qrng` instead harvests in bulk and caches:

```text
   ┌─ one QPU job ──────────────────────────────┐
   │  H on every qubit → measure the collapse    │
   │  shots × num_qubits raw measured bits       │
   └──────────────────┬──────────────────────────┘
                      │  von Neumann debias
                      ▼
   ~/.cache/qrandom/ibm_pool.bin   (a pool of quantum bytes)
                      │
   qrng --source ibm  ├─ pop 8 bytes → 64-bit seed   (instant)
   qrng --source ibm  ├─ pop 8 bytes → 64-bit seed   (instant)
   qrng --source ibm  └─ pool empty → run a new job, refill
```

- **One job fills the pool; every subsequent call is free** until it drains.
- **`--fresh`** forces a new job even when the pool is warm — use it when you
  specifically want bits attested by a *this-run* QPU job.
- **Debiasing.** Real qubits are never exactly 50/50 (readout error, T1/T2
  decay), so the raw stream is biased. `qrng` applies the **von Neumann
  extractor** (`01→0`, `10→1`, discard `00`/`11`), which removes first-order
  bias *without a classical hash* — the pool stays purely quantum-derived, at
  the cost of discarding roughly three quarters of the raw bits. If a job
  returns bits too correlated to extract anything, you'll get a message
  suggesting more shots.
- **Simulators are refused.** `qrng` selects `least_busy(simulator=False)`. A
  local/classical simulator is pseudorandom and would defeat the purpose.

Setup:

```bash
pip install 'qiskit>=1.0' qiskit-ibm-runtime
export IBM_QUANTUM_TOKEN=...   # from https://quantum.ibm.com
qrng --source ibm --fresh --json
```

> **Access note.** IBM migrated to the IBM Quantum Platform on IBM Cloud; the
> free Open Plan has tight monthly QPU-time budgets. Harvest a large pool once
> (raise `QRANDOM_IBM_SHOTS`) rather than calling `--fresh` repeatedly.

---

## The intent, and why it is dropped

`--intent "..."` is a **performative gesture**, not an input to the result:

- **`anu` / `nist`** — the string is written into an `X-Intent` HTTP request
  header, sent with the call, and **read by no endpoint**. It crosses the wire
  and is dropped downstream. It loses its performance the instant it was typed.
- **`ibm`** — the string rides along as a **job tag**. IBM stores it in the
  classical job record; the qubits never encode it. The Hadamard-and-measure
  collapse is intent-blind.

Either way the seed that returns was chosen without the intent. It is never
folded into the number — the default output prints only the raw value, and no
code path mixes intent into it. If intent and source were ever aligned, it
happened at emission, at a resolution finer than anything here records.

This is the design, not an oversight: the intent is *sent* so it can be
*released*, and *dropped* so the randomness stays the universe's, not yours.

---

## Options

| Flag | Effect |
|------|--------|
| `--source anu\|nist\|ibm` | Choose the quantum source. Required. |
| `--intent "STRING"` | The performative intent (sent, then dropped). |
| `--json` | Emit the full record: value, hex, bits, and provenance. |
| `--sd-seed` | Reduce to a Stable Diffusion seed (see below). |
| `--shift N` | Slide the `--sd-seed` capture window up N bits (see below). |
| `--pick "P P ..."` | Capture specific bit positions in order, MSB-first (see below). |
| `--fresh` | `ibm` only — force a new QPU job instead of drawing from the pool. |
| `--help` | Guided help. |

Exit codes: `0` success · `1` no source given (prints the prologue) · `2`
source error (unreachable, missing key/token/dependency).

### `--sd-seed`

Stable Diffusion seed fields are **unsigned 32-bit**: `0 .. 4294967295`. That is
the range accepted by the torch/diffusers generator and the A1111 / ComfyUI seed
inputs. The quantum sources here return 64 bits (`anu`, `ibm`) or 512 bits
(`nist`), which overflow that field.

`--sd-seed` keeps the **low 32 quantum bits** (`value & 0xFFFFFFFF`). Those bits
are already uniformly random, so masking — not folding or hashing — is enough;
you get an SD-valid seed carrying real quantum entropy.

```console
$ qrng --source nist --sd-seed
1791483961

$ qrng --source anu --sd-seed --json
{
  "source": "anu",
  "bits": 64,
  "value": 14203391556002019284,
  "hex": "c51b...",
  "sd_seed": 2264977876
}
```

With `--json` the full quantum `value` is preserved and an `sd_seed` field is
added alongside it, so you keep the provenance while feeding the 32-bit seed to
your pipeline.

#### `--shift N` — slide the capture window

By default the low 32 bits are captured. `--shift N` right-shifts the source
value `N` bits first, sliding the 32-bit window **up** — so you can draw a
different, non-overlapping slice out of a wide source:

```text
  512-bit NIST value:  [ … | b127..b96 | b95..b64 | b63..b32 | b31..b0 ]
  --shift 0   captures                                        ^^^^^^^^^  (b31..b0)
  --shift 32  captures                             ^^^^^^^^^             (b63..b32)
  --shift 64  captures                  ^^^^^^^^^                        (b95..b64)
```

`sd_seed = (value >> N) & 0xFFFFFFFF`. This lets a single 512-bit `nist` pulse
yield up to 16 independent SD seeds (`--shift 0, 32, 64, … 480`), or a 64-bit
`anu`/`ibm` value yield two.

```console
$ qrng --source nist --sd-seed --shift 64
2938104771
```

Rules: `--shift` requires `--sd-seed`, must be `>= 0`, and must be less than the
source width (`512` for `nist`, `64` for `anu`/`ibm`) — otherwise there are no
bits left to capture and `qrng` exits non-zero. With `--json`, a non-zero shift
is recorded in a `"shift"` field.

### `--pick "P P ..."` — capture specific bit positions

Where `--shift` slides a contiguous 32-bit window, `--pick` selects **individual
bit positions in an arbitrary order** and packs them into the result. This is a
separate capture mode from `--sd-seed` (the two are mutually exclusive).

- **Indexing:** position `n` is the bit of significance `n` — bit `0` is the
  LSB — the same convention as `--shift` (`bit = (value >> n) & 1`).
- **Order & packing:** bits are read in the order listed and packed **MSB-first**
  — the *first* position becomes the top bit of the result. The result is as
  wide as the number of positions you list.
- **Positions may repeat**, and the separator may be spaces or commas.

```console
$ qrng --source nist --pick "15 0 1 2 3 4 14"
# reads bits 15,0,1,2,3,4,14 (MSB-first) -> a 7-bit value
```

Worked example on `value = 0x8000000000000001` (bit 63 = 1, bit 0 = 1, rest 0):

```console
$ qrng --source ibm --pick "63 0 62 1" --json
{ ..., "pick_positions": [63, 0, 62, 1], "pick_width": 4, "pick": 12 }
#   bit63=1, bit0=1, bit62=0, bit1=0  ->  0b1100  =  12
```

Rules: every position must be in `[0, width)` (`64` for `anu`/`ibm`, `512` for
`nist`); a non-integer token, an out-of-range position, or an empty list exits
non-zero. With `--json`, `pick_positions`, `pick_width`, and `pick` are added.

> **Future direction — IBM temporal picks.** For `--source ibm`, a richer job is
> possible: rather than picking positions out of one pooled measurement, write a
> QPU program that *measures the selected qubit positions repeatedly over a
> period of time* and assembles the seed from that temporal sequence. That is a
> distinct circuit/runtime design (per-position measurement schedule, not a
> single snapshot) and is not yet implemented — the current `--pick` operates on
> the already-harvested source bits for every source uniformly.

## Environment

| Variable | Meaning |
|----------|---------|
| `ANU_API_KEY` | Key for the current ANU API. Absent → keyless legacy fallback. |
| `IBM_QUANTUM_TOKEN` | API token for `--source ibm`. |
| `IBM_QUANTUM_CHANNEL` | IBM channel (default `ibm_quantum`). |
| `QRANDOM_IBM_SHOTS` | Shots per QPU job (default `512`). More shots → bigger pool. |
| `QRANDOM_CACHE` | Pool directory (default `~/.cache/qrandom`). |

---

## Provenance record (`--json`)

`nist`:

```json
{
  "source": "nist",
  "bits": 512,
  "value": 109487573453477540959120076251234955268431689415174400124911730075...,
  "hex": "…128 hex chars…",
  "timestamp": "2026-07-14T22:30:00.000Z",
  "index": 1859759
}
```

`ibm`:

```json
{
  "source": "ibm",
  "bits": 64,
  "value": 12370169555311111083,
  "hex": "abababababababab",
  "backend": "ibm_test",
  "harvested": "2026-07-14T00:00:00Z",
  "origin": "cached quantum pool"
}
```

---

## Not for cryptography

These sources are for *capturing the randomness of a moment* — seed variety,
provenance, the feel of a physical draw. They are **not** hardened key material:
`nist` is public and reproducible; the ANU legacy path and IBM pool have no
health-test guarantees on the delivered bytes. For unpredictable-to-an-adversary
seeds, use the OS CSPRNG (`secrets.randbits(64)`), which already folds hardware
quantum jitter into a fast, private, health-checked stream.
