"""STABL-bpsfmoke: server-runtime output goes through logging, not stdout.

A print bypasses level, formatter and structure, and lands in the middle of a
stream something downstream is parsing as JSON.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# server/superres_cli.py is EXCLUDED on purpose: it is a CLI and print IS its
# output contract. Converting it would be a regression, not a fix (spec 7.3).
EXCLUDED = {"server/superres_cli.py"}

IN_SCOPE = sorted(
    p for d in ("server", "backends")
    for p in ROOT.joinpath(d).rglob("*.py")
    if str(p.relative_to(ROOT)) not in EXCLUDED
)


@pytest.mark.parametrize("path", IN_SCOPE, ids=lambda p: str(p.name))
def test_no_print_calls(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not offenders, f"{path.relative_to(ROOT)} prints at lines {offenders}"


def test_the_exclusion_still_names_a_real_file():
    """An exclusion for a path that no longer exists silently widens the guard.
    If superres_cli.py is renamed, this fails and forces a decision."""
    for rel in EXCLUDED:
        assert ROOT.joinpath(rel).is_file(), f"excluded path {rel} does not exist"


def test_the_sweep_actually_covers_the_files_it_claims_to():
    """Guards against a glob that quietly matches nothing — a green suite that
    checked zero files looks identical to a clean one."""
    names = {str(p.relative_to(ROOT)) for p in IN_SCOPE}
    for expected in (
        "backends/cuda_worker.py",
        "backends/rknnlcm.py",
        "server/lcm_sr_server.py",
        "server/superres_service.py",
    ):
        assert expected in names
    assert "server/superres_cli.py" not in names
