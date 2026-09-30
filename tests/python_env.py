"""Check test interpreter identity without importing project dependencies."""

import os
from pathlib import Path
import shlex
import sys


def python_environment_error():
    """Return mismatch guidance. Shell activation does not prove interpreter identity."""
    default = Path.home() / 'miniforge3/envs/stability-toys'
    expected = Path(os.environ.get('ST_TEST_PYTHON_PREFIX', str(default))).expanduser().resolve()
    actual = Path(sys.prefix).resolve()
    if actual == expected:
        return None
    interpreter = expected / 'bin/python'
    return (
        'Wrong Python environment for Stability-Toys tests.\n'
        f'Interpreter: {sys.executable}\n'
        f'Actual prefix: {actual}\n'
        f'Expected prefix: {expected}\n'
        f'Run: {shlex.quote(str(interpreter))} -m tests.run tests/ -- -q\n'
        'Default host activation: conda activate stability-toys\n'
        'For another supported environment, set ST_TEST_PYTHON_PREFIX to its prefix.'
    )


if __name__ == '__main__':
    error = python_environment_error()
    if error:
        print(error, file=sys.stderr)
        raise SystemExit(4)
