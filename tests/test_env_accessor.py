"""STABL-voqsoicx: one place that reads the environment."""
import logging

import pytest

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


# --- env_bool: the oracle, and the seven values STABL-cfyshjre deliberately broke
#
# HISTORY, because the shape of these tests is otherwise puzzling. In STABL-voqsoicx
# this compared against the LIVE utils/request_logger._env_bool — no hand-written
# truth table existed to drift. That function was deleted when its caller migrated,
# so the oracle froze here, provably faithful because the identical parametrisation
# had passed against the real function first (50063d5).
#
# STABL-cfyshjre then NORMALISED env_bool, which breaks the oracle for exactly
# seven values. The oracle is kept rather than deleted, and split in two:
#
#   - the canonical values must STILL match it — that is the parity guarantee,
#     and it is the majority of the table
#   - the seven that changed are listed explicitly, with the decision named
#
# Deleting the oracle would have thrown away the record of what the old behaviour
# was, which is the only thing that makes the break reviewable.


def _original_env_bool(name: str, default: str = "1") -> bool:
    """utils/request_logger.py's `_env_bool`, verbatim as of `50063d5`.

    Kept as the historical record. It is no longer what env_bool does for the
    seven values below.
    """
    import os

    v = os.environ.get(name, default)
    return v not in ("0", "false", "False", "no", "No")


# Everything a person would actually write on purpose. Behaviour here is
# UNCHANGED by the normalisation, and must stay that way.
@pytest.mark.parametrize("raw", ["1", "0", "true", "false", "False", "no", "No", "yes"])
def test_env_bool_still_matches_the_ORIGINAL_for_canonical_values(monkeypatch, raw):
    monkeypatch.setenv("ST_TEST", raw)
    assert env_bool("ST_TEST", True) is _original_env_bool("ST_TEST")


# The deliberate break. Each of these was TRUE under the original function and is
# FALSE now. `off` reading as ON is the one that justifies the change on its own:
# it turned a deliberate "off" into its opposite.
@pytest.mark.parametrize("raw", ["", "  ", " false ", "off", "OFF", "FALSE", "NO"])
def test_env_bool_NORMALISES_the_seven_values_the_original_got_wrong(monkeypatch, raw):
    monkeypatch.setenv("ST_TEST", raw)
    assert env_bool("ST_TEST", True) is False
    # ...and this is precisely where the two disagree. Asserting the difference,
    # rather than just the new value, keeps the break visible to a reader who has
    # not read STABL-cfyshjre.
    assert _original_env_bool("ST_TEST") is True


def test_the_oracle_split_covers_every_case_it_used_to():
    """Guards the guard. The original parametrisation had 15 values; splitting it
    into 8 canonical + 7 normalised must not quietly drop any."""
    canonical = {"1", "0", "true", "false", "False", "no", "No", "yes"}
    normalised = {"", "  ", " false ", "off", "OFF", "FALSE", "NO"}
    assert len(canonical | normalised) == 15
    assert not (canonical & normalised)


def test_the_shipped_constant_is_the_NORMALISED_set():
    """Ties the tests back to the code they protect. If someone edits
    utils.env._FALSE_VALUES, this fails immediately rather than leaving two truth
    tables quietly disagreeing."""
    from utils.env import _FALSE_VALUES

    assert _FALSE_VALUES == frozenset({"0", "false", "no", "off", ""})


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
    for raw in ['"', "''", '"""', " ", ",", "=", '"a', 'a"']:
        monkeypatch.setenv("ST_TEST", raw)
        env_str("ST_TEST")
        env_list("ST_TEST")
        env_int("ST_TEST", 1)
        env_bool("ST_TEST", True)


# ---------------------------------------------------------------------------
# Task 2 — the migrated readers
# ---------------------------------------------------------------------------

def test_request_logger_reads_a_QUOTED_allowlist_correctly(monkeypatch):
    """The live bug: env.dev quotes this value and runner.sh passes the quotes
    through, so the first entry was `"content-type` and the last `host"`."""
    monkeypatch.setenv("LOG_HEADER_ALLOWLIST", '"content-type,host"')
    from utils.request_logger import RequestLoggerConfig

    cfg = RequestLoggerConfig()
    assert cfg.header_allowlist == {"content-type", "host"}


def test_request_logger_reads_a_QUOTED_path_prefix_list_correctly(monkeypatch):
    monkeypatch.setenv("LOG_PATH_PREFIXES", '"/generate,/superres"')
    from utils.request_logger import RequestLoggerConfig

    assert RequestLoggerConfig().path_prefix_allowlist == {"/generate", "/superres"}


def test_request_logger_body_max_survives_junk(monkeypatch):
    """The old int(os.environ.get(...)) raised here, inside a dataclass field
    factory at import. Widening is safe: every value that parsed before still
    parses."""
    monkeypatch.setenv("LOG_BODY_MAX", "not-a-number")
    from utils.request_logger import RequestLoggerConfig

    assert RequestLoggerConfig().body_max == 8192


def test_apply_runtime_levels_reads_a_QUOTED_LOG_LEVELS(monkeypatch):
    """Asserted at apply_runtime_levels, NOT parse_log_levels.

    The seam is deliberate: unquoting is transport policy and belongs at the env
    boundary; parse_log_levels stays a pure function of its argument, which is
    what makes it cheap to test. A test that fed a quoted string straight to the
    parser would be asserting the wrong contract.
    """
    import logging

    from server.log_levels import apply_runtime_levels

    monkeypatch.setenv("LOG_LEVELS", '"st.quoted.probe=DEBUG"')
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    try:
        apply_runtime_levels()
        assert logging.getLogger("st.quoted.probe").level == logging.DEBUG
    finally:
        logging.getLogger("st.quoted.probe").setLevel(logging.NOTSET)


def test_apply_runtime_levels_reads_a_QUOTED_LOG_LEVEL(monkeypatch):
    import json
    import logging
    import logging.config

    from server.log_levels import apply_runtime_levels
    from server.logging_config import LOGGING_CONFIG

    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    monkeypatch.setenv("LOG_LEVEL", '"WARNING"')
    monkeypatch.delenv("LOG_LEVELS", raising=False)
    try:
        apply_runtime_levels()
        assert logging.getLogger("").level == logging.WARNING
    finally:
        logging.config.dictConfig(LOGGING_CONFIG)


def test_resolve_log_format_reads_a_QUOTED_LOG_FORMAT(monkeypatch):
    from server.log_format import JSON, resolve_log_format

    monkeypatch.setenv("LOG_FORMAT", '"json"')
    assert resolve_log_format() == JSON


def test_a_QUOTED_LOG_LEVEL_does_not_break_dictConfig(monkeypatch):
    """Found by Task 2 step 6, and the sharpest case in this issue.

    LOG_LEVEL is substituted into LOGGING_CONFIG at import. A quoted value used to
    land as the literal level name '"DEBUG"', and dictConfig raises
    `ValueError: Unable to configure logger ''` on it — the server does not start.
    LOG_LEVEL is also the variable most likely to be quoted once the project says
    quotes are acceptable.
    """
    import importlib
    import json
    import logging.config

    import server.logging_config as cfg

    monkeypatch.setenv("LOG_LEVEL", '"DEBUG"')
    try:
        importlib.reload(cfg)
        assert cfg.LOG_LEVEL == "DEBUG"
        logging.config.dictConfig(json.loads(json.dumps(cfg.LOGGING_CONFIG)))
    finally:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        importlib.reload(cfg)
        logging.config.dictConfig(cfg.LOGGING_CONFIG)
