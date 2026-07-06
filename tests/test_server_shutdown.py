import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _srv():
    # Imported lazily (not at module scope): several test modules install a
    # MagicMock torch into sys.modules at import time, which breaks collection
    # if server.lcm_sr_server (which transitively imports torch) is imported
    # while that mock is live.
    import server.lcm_sr_server as srv
    return srv


def test_close_providers_closes_both(monkeypatch):
    srv = _srv()
    calls = []

    class Storage:
        def close(self):
            calls.append("storage")

    class App:
        class state:
            storage = Storage()

    monkeypatch.setattr(srv, "close_store", lambda: calls.append("store"))
    srv._close_providers(App)
    assert "storage" in calls
    assert "store" in calls


def test_close_providers_tolerates_storage_error(monkeypatch):
    srv = _srv()
    calls = []

    class BadStorage:
        def close(self):
            raise IOError("boom")

    class App:
        class state:
            storage = BadStorage()

    monkeypatch.setattr(srv, "close_store", lambda: calls.append("store"))
    srv._close_providers(App)  # must not raise
    assert "store" in calls  # asset store still closed despite storage error
