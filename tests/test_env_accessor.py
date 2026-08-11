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


# --- env_bool: checked against the LIVE original, not a copy -------------------
#
# utils/request_logger._env_bool is the thing this must not change. While it still
# exists, compare against IT rather than a hand-written truth table — a copied
# table is a second source of truth that drifts silently, and the whole point of
# these cases is that they are the ones a "tidy-up" would flip.
#
# Task 2 deletes _env_bool. At that point the oracle is frozen into this file, and
# it is provably faithful because THIS test passed against the real function first.

@pytest.mark.parametrize("raw", [
    "1", "0", "true", "false", "False", "no", "No", "yes",
    "", "  ", " false ", "off", "OFF", "FALSE", "NO",
])
def test_env_bool_matches_the_LIVE_original_EXACTLY(monkeypatch, raw):
    """This migration moves the read; it does not change which deployments have a
    flag on. All three current flags are `1` everywhere, so a divergence would be
    inert TODAY and would bite the first time someone wrote `LOG_REQUESTS=` or
    `off` — and env.live-test:27 (`MODEL=`) shows empty values do get written into
    these files.
    """
    from utils.request_logger import _env_bool as original

    monkeypatch.setenv("ST_TEST", raw)
    assert env_bool("ST_TEST", True) is original("ST_TEST")


def test_the_oracle_is_the_REAL_function_not_a_reimplementation():
    """Guards the guard. If _env_bool is ever removed or renamed without moving
    the oracle deliberately, the test above would silently stop testing anything —
    it would fail at import, which is what we want, but this states the intent."""
    from utils import request_logger

    assert callable(getattr(request_logger, "_env_bool", None)), (
        "utils.request_logger._env_bool is gone. That is expected at Task 2 — "
        "freeze the oracle into this file and note that it was verified against "
        "the real function first."
    )


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
