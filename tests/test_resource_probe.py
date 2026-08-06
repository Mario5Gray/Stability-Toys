"""OS resource counts (STABL-cxbwwgly).

Plan: docs/superpowers/plans/2026-08-06-leaked-resource-gauges.md Task 1.

The load-bearing test is test_missing_shm_root_reports_none_not_zero: a 0 for
"leaked semaphores" on a host with no /dev/shm is indistinguishable from a
healthy Linux box.
"""
import pytest

from server.resource_probe import ResourceCounts, probe_resources


def _make_shm(tmp_path, sems=0, segments=0):
    for i in range(sems):
        (tmp_path / f"sem.mp-abc{i}").write_text("")
    for i in range(segments):
        (tmp_path / f"psm_{i:08x}").write_text("")
    return str(tmp_path)


def test_counts_semaphores_by_the_sem_prefix(tmp_path):
    counts = probe_resources(shm_root=_make_shm(tmp_path, sems=3, segments=2))
    assert counts.leaked_semaphores == 3


def test_counts_shm_segments_as_everything_else(tmp_path):
    """/dev/shm holds both; the sem.* prefix is what separates them."""
    counts = probe_resources(shm_root=_make_shm(tmp_path, sems=3, segments=2))
    assert counts.shm_segments == 2


def test_an_empty_shm_root_is_zero_not_none(tmp_path):
    """Readable and empty is a real measurement - distinct from unreadable."""
    counts = probe_resources(shm_root=str(tmp_path))
    assert counts.leaked_semaphores == 0
    assert counts.shm_segments == 0


def test_missing_shm_root_reports_none_not_zero(tmp_path):
    """THE distinction this issue turns on. A host-side check once reported 0
    while the container held several (STABL-nstyyrhh); reporting 0 for an
    unreadable source repeats that mistake in metric form."""
    counts = probe_resources(shm_root=str(tmp_path / "does-not-exist"))
    assert counts.leaked_semaphores is None
    assert counts.shm_segments is None


def test_fds_are_counted_even_when_shm_is_unavailable(tmp_path):
    """Availability is PER-SOURCE. macOS has no /dev/shm but num_fds() works -
    one unavailable source must not suppress the others."""
    counts = probe_resources(shm_root=str(tmp_path / "does-not-exist"))
    assert counts.open_fds is not None and counts.open_fds > 0


def test_probe_never_raises(monkeypatch):
    """It runs on the sampler pass that also carries the device gauges."""
    import server.resource_probe as rp

    monkeypatch.setattr(rp, "_count_fds", lambda: (_ for _ in ()).throw(OSError("nope")))
    counts = probe_resources(shm_root="/definitely/not/here")
    assert counts == ResourceCounts(None, None, None)


def test_unexpected_probe_failures_are_debug_logged(monkeypatch, caplog):
    import logging
    import server.resource_probe as rp

    monkeypatch.setattr(rp, "_count_shm", lambda root: (_ for _ in ()).throw(OSError("shm nope")))
    monkeypatch.setattr(rp, "_count_fds", lambda: (_ for _ in ()).throw(OSError("fds nope")))

    with caplog.at_level(logging.DEBUG):
        counts = probe_resources(shm_root="/definitely/not/here")

    assert counts == ResourceCounts(None, None, None)
    assert any("[ResourceProbe] shm count failed" in r.message for r in caplog.records)
    assert any("[ResourceProbe] fd count failed" in r.message for r in caplog.records)
    assert all(r.exc_info is not None for r in caplog.records if "[ResourceProbe]" in r.message)


def test_an_unreadable_shm_root_reports_none(tmp_path, monkeypatch):
    """Permission denied is unreadable, not empty."""
    import server.resource_probe as rp

    monkeypatch.setattr(
        rp.os, "listdir", lambda p: (_ for _ in ()).throw(PermissionError("nope"))
    )
    counts = probe_resources(shm_root=str(tmp_path))
    assert counts.leaked_semaphores is None


def test_module_imports_nothing_from_backends_or_metrics():
    import importlib.util

    spec = importlib.util.find_spec("server.resource_probe")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as fh:
        head = [ln for ln in fh if ln.startswith(("import ", "from "))]
    assert not any("backends" in ln or "server.metrics" in ln for ln in head)
