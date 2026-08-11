# Typed Env Accessor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not use sub-agent driven development** — `AGENTS.md` forbids it in this repo.

**Goal:** One place that reads environment variables, so a quoted value means the same thing regardless of which loader supplied it.

**Architecture:** `utils/env.py` exposes typed accessors, each with its own default quoting policy. Callers that handle list-shaped or logging values migrate to it. The env files are left alone — accepting quotes is the decision, not forbidding them.

**Tech Stack:** stdlib only.

**FP:** `STABL-voqsoicx`

## Global Constraints

- **Never raise.** These run at import time and inside `dataclass` field factories. A malformed value degrades to the default; it does not take down the process.
- **`env_bool` preserves the old truthiness EXACTLY**, oddities included. Moving a
  read must not change which deployments have a flag on. Normalisation is tracked
  separately so that decision is made on its own evidence.
- **Strip matching pairs only, one layer.** `"a,b"` → `a,b`. `"a` unchanged. `a"b` unchanged. `""` → empty string. `'a,b'` → `a,b`.
- **Do not un-quote the env files.** That was the rejected alternative. The wrapper is what makes them portable.
- **Do not migrate all 68 read sites.** Only values that are quoted today or are list-shaped.
- Python env: `conda activate stability-toys`; use `python`, not `python3`.

---

## The measurement this rests on

Run on enigma against a live daemon (docker 29.6.2):

| value form | `docker run --env-file` | `docker compose env_file` |
|---|---|---|
| `BARE=a=1,b=2` | works | works |
| `QUOTED="a=1,b=2"` | **quotes kept literally** | quotes stripped |
| `export X=y` | **whole file rejected, run fails** | works |

`env.dev` is loaded **both** ways — compose (`docker-compose.dev.yml`,
`docker-compose.test.yml`) and `docker run --env-file` (`runner.sh`). Its two
quoted values are correct under one loader and corrupt under the other at the same
time. That is the bug this closes.

---

## File Structure

| File | Responsibility |
|---|---|
| `utils/env.py` *(create)* | `Quotes` policy + typed accessors |
| `utils/request_logger.py` *(modify)* | 4 reads incl. its local `_env_bool` |
| `server/log_levels.py` *(modify)* | `LOG_LEVEL`, `LOG_LEVELS` |
| `server/log_format.py` *(modify)* | `LOG_FORMAT` |
| `tests/test_env_accessor.py` *(create)* | Unit + policy coverage |
| `tests/test_env_file_contract.py` *(create)* | The `export ` guard |
| `docs/observability-contract.md` *(modify)* | State the policy |

---

## Task 1: `utils/env.py`

**Files:**
- Create: `utils/env.py`
- Test: `tests/test_env_accessor.py`

**Interfaces:**
- Produces: `Quotes` (`ALLOW` / `WARN` / `LITERAL`), `unquote(value, policy, name) -> str`,
  `env_str(name, default="", *, quotes=Quotes.WARN) -> str`,
  `env_list(name, default="", *, sep=",", quotes=Quotes.ALLOW) -> list[str]`,
  `env_int(name, default, *, quotes=Quotes.ALLOW) -> int`,
  `env_bool(name, default=True, *, quotes=Quotes.ALLOW) -> bool`.

### Why the policy differs by accessor

A comma-separated value is the one a human instinctively quotes, so `env_list`
accepts quotes **silently** — that is the whole point of the wrapper. A quoted
*scalar* is almost always a mistake, so `env_str` strips it but **says so**: it
works, and the operator finds out. `LITERAL` exists for the value that genuinely
begins and ends with a quote character, which is otherwise unrepresentable.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_env_accessor.py
"""STABL-voqsoicx: one place that reads the environment."""
import logging

import pytest

from utils import env as envmod
from utils.env import Quotes, env_bool, env_int, env_list, env_str, unquote


# --- unquote ------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('"a,b"', "a,b"),
    ("'a,b'", "a,b"),
    ("a,b", "a,b"),
    ('""', ""),
    ("''", ""),
])
def test_matching_pairs_are_stripped(raw, expected):
    assert unquote(raw, Quotes.ALLOW, "X") == expected


@pytest.mark.parametrize("raw", ['"a', 'a"', "'a", 'a"b', '"a\'', "", " "])
def test_UNMATCHED_or_interior_quotes_are_left_ALONE(raw):
    """Only a matching outer pair is a quoting artifact. Anything else is data,
    and mangling it would be worse than the bug being fixed."""
    assert unquote(raw, Quotes.ALLOW, "X") == raw


def test_only_ONE_layer_is_stripped():
    """A doubly-quoted value almost certainly means the inner pair is data."""
    assert unquote('""a,b""', Quotes.ALLOW, "X") == '"a,b"'


def test_LITERAL_strips_nothing():
    assert unquote('"a,b"', Quotes.LITERAL, "X") == '"a,b"'


def test_WARN_strips_AND_names_the_variable(caplog):
    with caplog.at_level(logging.WARNING, logger="utils.env"):
        assert unquote('"a"', Quotes.WARN, "MY_VAR") == "a"
    assert "MY_VAR" in caplog.text


def test_ALLOW_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="utils.env"):
        unquote('"a"', Quotes.ALLOW, "MY_VAR")
    assert caplog.text == ""


# --- accessors ----------------------------------------------------------------

def test_env_str_returns_the_default_when_unset(monkeypatch):
    monkeypatch.delenv("ST_TEST", raising=False)
    assert env_str("ST_TEST", "fallback") == "fallback"


def test_env_str_WARNS_on_a_quoted_scalar(monkeypatch, caplog):
    monkeypatch.setenv("ST_TEST", '"value"')
    with caplog.at_level(logging.WARNING, logger="utils.env"):
        assert env_str("ST_TEST") == "value"
    assert "ST_TEST" in caplog.text


def test_env_list_splits_strips_and_drops_empties(monkeypatch):
    monkeypatch.setenv("ST_TEST", " a , b ,, c ")
    assert env_list("ST_TEST") == ["a", "b", "c"]


def test_env_list_accepts_a_quoted_value_SILENTLY(monkeypatch, caplog):
    """The case that is broken today under docker run --env-file."""
    monkeypatch.setenv("ST_TEST", '"content-type,host"')
    with caplog.at_level(logging.WARNING, logger="utils.env"):
        assert env_list("ST_TEST") == ["content-type", "host"]
    assert caplog.text == ""


def test_env_list_uses_the_default_string_when_unset(monkeypatch):
    monkeypatch.delenv("ST_TEST", raising=False)
    assert env_list("ST_TEST", "a,b") == ["a", "b"]


def test_env_list_of_an_empty_value_is_EMPTY_not_a_blank_entry(monkeypatch):
    monkeypatch.setenv("ST_TEST", "")
    assert env_list("ST_TEST") == []


def test_env_int_parses_a_quoted_number(monkeypatch):
    monkeypatch.setenv("ST_TEST", '"8192"')
    assert env_int("ST_TEST", 10) == 8192


def test_env_int_falls_back_and_WARNS_on_junk(monkeypatch, caplog):
    """Never raises: this runs inside dataclass field factories at import."""
    monkeypatch.setenv("ST_TEST", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="utils.env"):
        assert env_int("ST_TEST", 10) == 10
    assert "ST_TEST" in caplog.text


# IMPROVED DURING EXECUTION, on review feedback: do NOT hand-copy the truth table.
# utils/request_logger._env_bool still exists until Task 2 deletes it, so compare
# against the LIVE function — a copied table is a second source of truth that
# drifts silently, and these cases are precisely the ones a "tidy-up" would flip.
#
# Task 2 deletes _env_bool. At that point the oracle freezes into the test file,
# and it is provably faithful because this test passed against the real function
# first. A companion test asserts _env_bool still exists, so its removal fails
# loudly rather than turning this into a no-op.
#
# The surprising rows — '', '  ', ' false ', 'off', 'OFF', 'FALSE', 'NO' — are all
# TRUE under the old semantics and are in the parametrisation deliberately.
@pytest.mark.parametrize("raw", [
    "1", "0", "true", "false", "False", "no", "No", "yes",
    "", "  ", " false ", "off", "OFF", "FALSE", "NO",
])
def test_env_bool_matches_the_OLD_semantics_EXACTLY(monkeypatch, raw):
    """This migration moves the read; it does not change which deployments have a
    flag on. All three current flags are `1` everywhere, so a divergence would be
    inert TODAY and would bite the first time someone wrote `LOG_REQUESTS=` or
    `off` — and env.live-test:27 shows empty values do get written in these files.
    """
    monkeypatch.setenv("ST_TEST", raw)
    assert env_bool("ST_TEST", True) is _old_env_bool(raw)


def test_env_bool_ignores_quotes_before_comparing(monkeypatch):
    """The one intended difference from the old function: it never saw a quoted
    value correctly under `docker run --env-file`."""
    monkeypatch.setenv("ST_TEST", '"false"')
    assert env_bool("ST_TEST", True) is False


def test_env_bool_returns_the_default_when_unset(monkeypatch):
    monkeypatch.delenv("ST_TEST", raising=False)
    assert env_bool("ST_TEST", False) is False


def test_nothing_here_raises_on_hostile_input(monkeypatch):
    # NOT "\x00": os.environ rejects embedded nulls, so monkeypatch.setenv would
    # raise before the accessor ran — the test would be exercising the harness,
    # not this module.
    for raw in ['"', "''", '"""', " ", ",", "=", '"a', "a"]:
        monkeypatch.setenv("ST_TEST", raw)
        env_str("ST_TEST")
        env_list("ST_TEST")
        env_int("ST_TEST", 1)
        env_bool("ST_TEST", True)
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_env_accessor.py -q`
Expected: `ModuleNotFoundError: No module named 'utils.env'`.

- [x] **Step 3: Write the implementation**

```python
# utils/env.py
"""Typed environment access with an explicit quoting policy (STABL-voqsoicx).

The two loaders this repo uses disagree about quotes. Measured on a live daemon:

    value form          docker run --env-file      docker compose env_file
    BARE=a=1,b=2        works                      works
    QUOTED="a=1,b=2"    QUOTES KEPT LITERALLY      quotes stripped
    export X=y          WHOLE FILE REJECTED        works

`env.dev` is loaded BOTH ways — compose for the dev and test containers,
`docker run --env-file` from `runner.sh` — so its quoted values are correct under
one loader and corrupt under the other at the same time. Rather than forbid quotes
in the files, accept them here and be explicit about where.

Nothing in this module raises. It is called at import time and inside dataclass
field factories; a bad value degrades to the default and says so.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

_PAIRS = ('"', "'")


class Quotes(Enum):
    """What an accessor does with a matching outer quote pair."""

    ALLOW = "allow"      # strip, silently — for values humans naturally quote
    WARN = "warn"        # strip, and name the variable
    LITERAL = "literal"  # strip nothing; the quotes are data


def unquote(value: str, policy: Quotes, name: str) -> str:
    """Remove ONE layer of matching outer quotes, per policy.

    Matching pairs only, and only one layer: `"a,b"` -> `a,b`, but `"a` and `a"b`
    are left alone because an unmatched or interior quote is data, and mangling it
    would be worse than the bug this fixes.
    """
    if policy is Quotes.LITERAL or len(value) < 2:
        return value
    if value[0] not in _PAIRS or value[-1] != value[0]:
        return value
    stripped = value[1:-1]
    if policy is Quotes.WARN:
        logger.warning(
            "[env] %s was quoted; the quotes were removed. Env files here are "
            "loaded by both `docker compose` (strips quotes) and `docker run "
            "--env-file` (does not), so an unquoted value is the portable form.",
            name,
        )
    return stripped


def env_str(name: str, default: str = "", *, quotes: Quotes = Quotes.WARN) -> str:
    """A scalar. Quoted input is honoured but reported — it is nearly always a
    mistake, and a silent fix teaches the wrong lesson."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return unquote(raw, quotes, name)


def env_list(
    name: str,
    default: str = "",
    *,
    sep: str = ",",
    quotes: Quotes = Quotes.ALLOW,
) -> List[str]:
    """A separated list. Quoted input is accepted SILENTLY: a comma-separated
    value is the one a human instinctively quotes, and accepting it is the point
    of this module. Empty entries are dropped."""
    raw = os.environ.get(name)
    source = default if raw is None else unquote(raw, quotes, name)
    return [part.strip() for part in source.split(sep) if part.strip()]


def env_int(name: str, default: int, *, quotes: Quotes = Quotes.ALLOW) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(unquote(raw, quotes, name).strip())
    except (TypeError, ValueError):
        logger.warning("[env] %s=%r is not an integer; using %r", name, raw, default)
        return default


# VERBATIM from utils/request_logger.py's original `_env_bool`. Not a set, not
# lowercased, not stripped — those would all be improvements, and every one of
# them changes which deployments have a flag on.
#
# Measured divergences if this were "tidied" to {"0","false","no","off",""} with
# .strip().lower():
#
#     ''  '  '  ' false '  'off'  'OFF'  'FALSE'  'NO'
#
# all flip from TRUE to FALSE. Empty is the dangerous one: `LOG_REQUESTS=` is
# currently ON, and env.live-test:27 shows empty values do get written in these
# files. Normalising is tracked separately so that decision is made on its own
# evidence, not smuggled through a refactor (STABL-voqsoicx).
_FALSE_VERBATIM = ("0", "false", "False", "no", "No")


def env_bool(name: str, default: bool = True, *, quotes: Quotes = Quotes.ALLOW) -> bool:
    """Everything except 0/false/False/no/No is true — including empty, 'off',
    'FALSE' and 'NO'.

    Those last four are surprising, and deliberately preserved: this module's job
    in the migration is to move the read, not to change which deployments have a
    flag on. See the note on _FALSE_VERBATIM.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return unquote(raw, quotes, name) not in _FALSE_VERBATIM
```

- [x] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_env_accessor.py -q`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add utils/env.py tests/test_env_accessor.py
git commit -m "feat(env): typed accessors with a per-method quoting policy (STABL-voqsoicx) — next: Task 2, migrate the readers"
```

---

## Task 2: Migrate the readers that matter

**Files:**
- Modify: `utils/request_logger.py`, `server/log_levels.py`, `server/log_format.py`
- Test: `tests/test_env_accessor.py` (extend)

**Interfaces:** consumes Task 1.

Only values that are quoted today or are list-shaped. The other ~60 read sites stay
as they are — a migration nobody needs is churn that has to be reviewed.

- [x] **Step 1: Write the failing test**

```python
# append to tests/test_env_accessor.py
def test_request_logger_reads_a_QUOTED_allowlist_correctly(monkeypatch):
    """The live bug: env.dev quotes this value and runner.sh passes the quotes
    through, so the first entry was `"content-type` and the last `host"`."""
    monkeypatch.setenv("LOG_HEADER_ALLOWLIST", '"content-type,host"')
    from utils.request_logger import RequestLoggerConfig

    cfg = RequestLoggerConfig()
    assert cfg.header_allowlist == {"content-type", "host"}


def test_log_levels_reads_a_QUOTED_LOG_LEVELS_correctly(monkeypatch):
    """The next variable that would have hit this (STABL-ataigkdk)."""
    monkeypatch.setenv("LOG_LEVELS", '"comfy.jobs=WARNING"')
    from server.log_levels import parse_log_levels
    import os

    assert parse_log_levels(os.environ["LOG_LEVELS"]) == {"comfy.jobs": "WARNING"}
```

> **`parse_log_levels` takes a raw string, not a name.** Either route the unquoting
> through `env_str` inside `apply_runtime_levels` before calling it, or have
> `parse_log_levels` unquote its own argument. Prefer the former: keep
> `parse_log_levels` a pure function of its input, and put env policy at the env
> boundary. Adjust this test to match whichever seam you choose, but do not make
> `parse_log_levels` reach into `os.environ` — its current purity is what makes it
> cheap to test.

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_env_accessor.py -q -k QUOTED`
Expected: both FAIL — quotes survive into the parsed values.

- [x] **Step 3: Migrate `utils/request_logger.py`**

Replace the local `_env_bool` and the four `os.environ.get` reads:

```python
from utils.env import env_bool, env_int, env_list

@dataclass
class RequestLoggerConfig:
    enabled: bool = field(default_factory=lambda: env_bool("LOG_REQUESTS", True))
    log_headers: bool = field(default_factory=lambda: env_bool("LOG_REQUEST_HEADERS", True))
    log_body: bool = field(default_factory=lambda: env_bool("LOG_REQUEST_BODY", True))

    body_max: int = field(default_factory=lambda: env_int("LOG_BODY_MAX", 8192))

    header_allowlist: Set[str] = field(
        default_factory=lambda: {
            h.lower() for h in env_list(
                "LOG_HEADER_ALLOWLIST",
                "content-type,content-length,x-forwarded-for,x-real-ip,user-agent,host",
            )
        }
    )

    path_prefix_allowlist: Optional[Set[str]] = field(
        default_factory=lambda: set(env_list("LOG_PATH_PREFIXES")) or None
    )

    path_prefix_denylist: Set[str] = field(
        default_factory=lambda: set(env_list("LOG_PATH_DENYLIST", "/docs,/openapi.json"))
    )
```

**`env_int` now takes an int default where the old code took a string** — the old
`int(os.environ.get("LOG_BODY_MAX", "8192"))` would raise on junk input; the new one
degrades. That is the intended change, not a slip.

**Why `env_int` may widen while `env_bool` may not**, since the two look
inconsistent otherwise:

| | old behaviour on the diverging input | effect of the change |
|---|---|---|
| `env_bool` | returns `True` for `""`, `off`, `FALSE`, `NO` | a **working** deployment silently flips a flag |
| `env_int` | **raises** at import | only the crash path changes; every value that parsed before parses identically |

The test is not "is the old behaviour good", it is "can a deployment that works
today start behaving differently". For `env_bool` the answer is yes, so it is
preserved verbatim. For `env_int` it is no.

- [x] **Step 4: Migrate `server/log_levels.py` and `server/log_format.py`**

In `log_levels.apply_runtime_levels`, read through the accessor rather than
`os.getenv`, so `LOG_LEVEL` and `LOG_LEVELS` get the same treatment as everything
else. In `log_format.resolve_log_format`, the same for `LOG_FORMAT`.

Keep `parse_log_levels(raw)` pure — pass it an already-unquoted string.

- [x] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_env_accessor.py tests/test_log_levels.py tests/test_log_format.py tests/test_logging_wiring.py -q`
Expected: all pass, including the existing logging suites unchanged.

- [x] **Step 6: Check for other readers of the migrated variables**

```bash
grep -rn 'LOG_HEADER_ALLOWLIST\|LOG_PATH_PREFIXES\|LOG_PATH_DENYLIST\|LOG_BODY_MAX\|LOG_REQUESTS\|LOG_LEVELS\|LOG_FORMAT' --include='*.py' server backends utils
```

Any site still using `os.environ` for these is a second source of truth. Migrate or
justify each one in the commit message.

- [x] **Step 7: Commit**

```bash
git add utils/request_logger.py server/log_levels.py server/log_format.py tests/test_env_accessor.py
git commit -m "refactor(env): route logging and request-logger reads through the accessor (STABL-voqsoicx) — next: Task 3, the export guard"
```

---

## Task 3: The `export ` guard, and document the policy

**Files:**
- Create: `tests/test_env_file_contract.py`
- Modify: `docs/observability-contract.md`

`docker run --env-file` **rejects the entire file** on an `export ` line — the run
fails, it does not degrade. `env.prod` has two such lines and is compose-only today,
so it is fine; it breaks the moment anyone passes it to `runner.sh`. The wrapper
cannot help with this one, because the file never reaches Python.

- [ ] **Step 1: Write the failing-if-broken guard**

```python
# tests/test_env_file_contract.py
"""STABL-voqsoicx: what an env file may contain, per loader.

Quoting is HANDLED — utils/env.py accepts it, deliberately. `export ` is not, and
cannot be: `docker run --env-file` rejects the whole file before Python sees it.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files reachable by `docker run --env-file`, which cannot tolerate `export `.
# runner.sh:9 and :12.
DOCKER_RUN_ENV_FILES = ["env.cuda", "env.custom", "env.dev", "env.rknn"]


@pytest.mark.parametrize("name", DOCKER_RUN_ENV_FILES)
def test_no_export_prefix_in_docker_run_env_files(name):
    """`docker run --env-file` fails the ENTIRE run on one `export` line:
    "invalid env file: variable 'export X' contains whitespaces". Measured against
    docker 29.6.2."""
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    offenders = [
        (i, line.rstrip())
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if line.startswith("export ")
    ]
    assert not offenders, f"{name} has export-prefixed lines: {offenders}"


def test_the_file_list_still_matches_runner_sh():
    """A file added to runner.sh but not to this list is unguarded, and the guard
    would look green while covering nothing."""
    runner = (ROOT / "runner.sh").read_text()
    referenced = set(re.findall(r"--env-file\s+(\S+)", runner))
    assert referenced <= set(DOCKER_RUN_ENV_FILES), (
        f"runner.sh loads env files this guard does not cover: "
        f"{sorted(referenced - set(DOCKER_RUN_ENV_FILES))}"
    )
```

Add `import re` at the top.

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_env_file_contract.py -q`
Expected: PASS today — `env.cuda`/`env.custom`/`env.dev`/`env.rknn` have no
`export` lines (`env.prod` does, and is deliberately not in the list). If the second
test fails, `runner.sh` has changed and the list needs updating.

- [ ] **Step 3: Document the policy**

Append to the **Log levels** section of `docs/observability-contract.md`:

````markdown
### Env file syntax

Values may be quoted or bare. `utils/env.py` strips a matching outer pair, because
the two loaders disagree: `docker compose env_file` strips quotes, `docker run
--env-file` passes them through literally, and `env.dev` is loaded both ways.

`export ` prefixes are a different matter and must not be used in any file
`runner.sh` passes to `docker run --env-file` — that loader **rejects the entire
file**, failing the run rather than degrading. `tests/test_env_file_contract.py`
guards this.
````

- [ ] **Step 4: Full suite**

```bash
python -m pytest -q 2>&1 | tail -3
```

Baseline is **1497 passed**, 9 skipped, 1 xfailed. Run it on the tree you are about
to summarise.

- [ ] **Step 5: Commit and close out**

```bash
git add tests/test_env_file_contract.py docs/observability-contract.md docs/superpowers/plans/2026-08-10-env-accessor.md
git commit -m "test(env): guard against export-prefixed lines in docker-run env files (STABL-voqsoicx) — next: PR"
fp issue assign STABL-voqsoicx --rev <sha1>,<sha2>,<sha3>
fp comment STABL-voqsoicx "STOP: ... NEXT: ... DECIDED: ..."
```

---

## Deferred, NOT done here

| Item | Why |
|---|---|
| Migrating all 68 env read sites | Only values that are quoted today or list-shaped need it. Churn nobody asked for still has to be reviewed. |
| Un-quoting the env files | The rejected alternative. The wrapper is the decision. |
| `env.prod`'s two `export` lines | Harmless under compose, which is its only loader. Guarded against, not removed. |
| Normalising `env_bool`'s truthiness | `STABL-cfyshjre`. Seven values (`""`, `"  "`, `" false "`, `off`, `OFF`, `FALSE`, `NO`) are currently TRUE. Changing that is a behaviour decision, not a migration detail, and gets its own evidence. |
