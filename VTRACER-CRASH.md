# VTracer crash on CPython 3.14.7

## Summary

The VTracer crash is probably not the same crash as CPython issue
[#158312](https://github.com/python/cpython/issues/158312).

Both cases use Python 3.14.7 and end with `SIGSEGV`. Their resolved crash
frames, triggers, and reproduction evidence differ.

| Detail | CPython issue #158312 | VTracer crash |
|---|---|---|
| Resolved crash frame | `_DK_ENTRIES` during dictionary insertion | Nearest resolved frame is `PyBuffer_Release` |
| Corruption | Dictionary keys pointer equals `0x64` | Partly corrupt native stack |
| Trigger | Hermes startup and object initialization | `convert_raw_image_to_svg()` |
| Reproduction | One reported crash without a minimal case | Five deterministic crashes |
| Dependencies | Large application with many extensions | Minimal 16 by 16 PNG conversion |
| Python 3.12 | Not established | Passes on the same enigma host |

## CPython issue status

CPython maintainers have not confirmed issue #158312 as an interpreter bug.
Hermes loads many compiled extensions. A maintainer requested a minimal
reproduction because those extensions remain possible causes.

See the
[maintainer comment](https://github.com/python/cpython/issues/158312#issuecomment-5863919601).

## Enigma evidence

Enigma has five VTracer core dumps:

- Four pytest processes crashed.
- One minimal conversion process crashed.
- All five crashes occurred within 22 seconds.
- All five processes used Fedora Python 3.14.7.
- The minimal core's nearest resolved Python frame is `PyBuffer_Release`.
- The native stack is partly corrupt, so this frame does not prove the root cause.

The minimal reproduction converts a blank 16 by 16 PNG through
`vtracer.convert_raw_image_to_svg()`.

The same conversion and the complete fixture suite pass on enigma with uv
CPython 3.12.12 and `vtracer==0.6.15`.

## Likely boundary

VTracer 0.6.15 uses PyO3 0.19. Its Python binding accepts image bytes as a Rust
`Vec<u8>`. This places the failing call near Python's buffer-conversion boundary.

This evidence suggests a VTracer or PyO3 compatibility problem. It does not
prove one. A CPython 3.14 regression remains possible.

Sources:

- [VTracer 0.6.15 Python binding](https://github.com/visioncortex/vtracer/blob/0.6.15/cmdapp/src/python.rs)
- [VTracer 0.6.15 Cargo dependency](https://github.com/visioncortex/vtracer/blob/0.6.15/cmdapp/Cargo.toml)

Do not mark the VTracer crash as a duplicate of CPython issue #158312 without a
matching native stack or a shared minimal reproduction.

## Current impact

- Python 3.12 works on enigma.
- The project environment and container use Python 3.12.
- Python 3.14 support remains unresolved.
- STABL-mlahbjjj tracks the required release decision.

Before the standalone vector release, select one policy:

1. Limit the complete helper package to Python versions below 3.14.
2. Reject Python 3.14 in the vector adapter before VTracer loads.
3. Adopt a newer VTracer release after the complete fixture suite passes.

## Upstream diagnosis

A stronger upstream diagnosis should rebuild VTracer with a current PyO3
version. It should run the minimal reproduction against a Python 3.14 debug
build. Compare that native stack with the existing Fedora core dumps.
