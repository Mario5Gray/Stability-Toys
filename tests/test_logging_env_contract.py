from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_env_file(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (REPO_ROOT / name).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        values[key] = value
    return values


def test_only_prod_defaults_to_json_logs():
    prod = _read_env_file("env.prod")

    assert prod["LOG_FORMAT"] == "json"


def test_dev_and_live_test_default_to_text_logs():
    dev = _read_env_file("env.dev")
    live_test = _read_env_file("env.live-test")

    assert "LOG_FORMAT" not in dev
    assert "LOG_FORMAT" not in live_test


def test_prod_defaults_root_logging_to_info():
    prod = _read_env_file("env.prod")

    assert prod["LOG_LEVEL"] == "INFO"


def test_dev_and_live_test_default_root_logging_to_debug():
    dev = _read_env_file("env.dev")
    live_test = _read_env_file("env.live-test")

    assert dev["LOG_LEVEL"] == "DEBUG"
    assert live_test["LOG_LEVEL"] == "DEBUG"
