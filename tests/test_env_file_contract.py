"""STABL-voqsoicx: what an env file may contain, per loader.

Quoting is HANDLED — `utils/env.py` accepts it, deliberately. `export ` is not,
and cannot be: `docker run --env-file` rejects the whole file before Python sees
it, so no amount of application-side tolerance helps.

Measured against docker 29.6.2 on a live daemon:

    value form          docker run --env-file      docker compose env_file
    BARE=a=1,b=2        works                      works
    QUOTED="a=1,b=2"    quotes kept literally      quotes stripped
    export X=y          WHOLE FILE REJECTED        works (export stripped)
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files reachable by `docker run --env-file` (runner.sh:9 and :12), which cannot
# tolerate `export `. env.prod is deliberately ABSENT: it carries two export lines
# and is loaded only by compose (docker-cuda.yml, docker-rknn.yml), where they are
# handled. Adding it here would fail for a problem it does not have; passing it to
# runner.sh is what would break, and the second test below is what would notice.
DOCKER_RUN_ENV_FILES = ["env.cuda", "env.custom", "env.dev", "env.rknn"]


@pytest.mark.parametrize("name", DOCKER_RUN_ENV_FILES)
def test_no_export_prefix_in_docker_run_env_files(name):
    """`docker run --env-file` fails the ENTIRE run on one `export` line:

        docker: --env-file: invalid env file (...): variable 'export X'
        contains whitespaces

    It does not skip the line or warn — the container never starts.
    """
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    offenders = [
        (i, line.rstrip())
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if line.startswith("export ")
    ]
    assert not offenders, (
        f"{name} has export-prefixed lines {offenders}; `docker run --env-file` "
        f"rejects the whole file and the container will not start"
    )


def test_the_file_list_still_matches_runner_sh():
    """Guards the guard. A file added to runner.sh but not to DOCKER_RUN_ENV_FILES
    is unguarded, and the parametrised test above would stay green while covering
    nothing."""
    runner = (ROOT / "runner.sh").read_text()
    referenced = set(re.findall(r"--env-file\s+(\S+)", runner))
    unguarded = referenced - set(DOCKER_RUN_ENV_FILES)
    assert not unguarded, (
        f"runner.sh passes these to `docker run --env-file` but the export guard "
        f"does not cover them: {sorted(unguarded)}"
    )


def test_env_prod_is_excluded_ON_PURPOSE_and_still_compose_only():
    """env.prod is the one file with `export` lines. That is safe only while
    compose is its sole loader — this states the condition rather than leaving a
    silent exclusion, so moving env.prod onto a `docker run` path fails here
    instead of at container start."""
    prod = ROOT / "env.prod"
    if not prod.is_file():
        pytest.skip("env.prod not present")

    has_export = any(
        line.startswith("export ") for line in prod.read_text().splitlines()
    )
    if not has_export:
        pytest.skip("env.prod no longer uses `export`; it could join the guard")

    runner = (ROOT / "runner.sh").read_text()
    assert "env.prod" not in re.findall(r"--env-file\s+(\S+)", runner), (
        "env.prod contains `export` lines AND is now passed to "
        "`docker run --env-file`. That combination does not start. Either drop "
        "the export prefixes or keep env.prod compose-only."
    )
