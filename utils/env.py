"""Typed environment access with an explicit quoting policy (STABL-voqsoicx).

The two loaders this repo uses disagree about quotes. Measured on a live daemon
(docker 29.6.2):

    value form          docker run --env-file      docker compose env_file
    BARE=a=1,b=2        works                      works
    QUOTED="a=1,b=2"    QUOTES KEPT LITERALLY      quotes stripped
    export X=y          WHOLE FILE REJECTED        works (export stripped)

`env.dev` is loaded BOTH ways — compose for the dev and test containers,
`docker run --env-file` from `runner.sh` — so its quoted values are correct under
one loader and corrupt under the other at the same time. Rather than forbid quotes
in the files, accept them here and be explicit about where.

Nothing in this module raises. It is called at import time and inside dataclass
field factories; a bad value degrades to the default and says so.

`export ` is a different problem and this module cannot help with it: that loader
rejects the file before Python ever sees it. See tests/test_env_file_contract.py.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import List

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
    """An integer, degrading to `default` on junk.

    This WIDENS the old behaviour — `int(os.environ.get("LOG_BODY_MAX", "8192"))`
    raised inside a dataclass field factory at import. Safe to widen because only
    the crash path changes: every value that parsed before parses identically.
    Contrast env_bool below, where a divergence would flip a WORKING deployment.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(unquote(raw, quotes, name).strip())
    except (TypeError, ValueError):
        logger.warning("[env] %s=%r is not an integer; using %r", name, raw, default)
        return default


# NORMALISED, case-insensitively and after stripping whitespace (STABL-cfyshjre).
#
# The original `_env_bool` in utils/request_logger.py compared against the literal
# tuple ("0", "false", "False", "no", "No"), so SEVEN values were true that no one
# writing them could have meant that way:
#
#     ''  '  '  ' false '  'off'  'OFF'  'FALSE'  'NO'
#
# `off` reading as ON is the worst of them: it turns a deliberate "off" into its
# opposite. STABL-voqsoicx preserved the old behaviour verbatim because a
# migration must not change which deployments have a flag on; this is the separate
# decision that changes it, on its own evidence.
#
# Checked before landing: every env_bool caller reads LOG_REQUESTS,
# LOG_REQUEST_HEADERS or LOG_REQUEST_BODY, all three are `1` in env.prod, env.dev
# and env.live-test, and no env file contains any of the seven values above for a
# boolean flag. No current deployment changes behaviour.
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def env_bool(name: str, default: bool = True, *, quotes: Quotes = Quotes.ALLOW) -> bool:
    """Everything except 0/false/no/off/empty is true, case-insensitively.

    Whitespace is stripped first, so ` false ` is false. Empty is FALSE: setting
    `FLAG=` reads as an operator turning something off, which is the only sane
    reading of it.

    This is the STABL-cfyshjre normalisation. If you are chasing a flag that used
    to be on and is now off, the value is one of: '', 'off', 'OFF', 'FALSE', 'NO',
    or a value with surrounding whitespace.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return unquote(raw, quotes, name).strip().lower() not in _FALSE_VALUES
