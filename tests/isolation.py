"""Select test files before pytest imports them.

Each stub file gets its own interpreter. Stub files install conflicting
objects into sys.modules at import time. One file may need real Torch while
another installs a MagicMock Torch. Grouping them in one interpreter makes
the result depend on import order (STABL-rhjokuwd). The file is the smallest
safe boundary: separate files, separate processes.
"""

from pathlib import Path


STUB_FILES = frozenset({
    'test_cuda_worker_base.py',
    'test_cuda_worker_capabilities.py',
    'test_cuda_worker_controlnet.py',
    'test_hunyuandit_worker.py',
    'test_model_lifecycle.py',
    'test_worker_controlnet_metadata.py',
    'test_worker_handle.py',
    'test_worker_pool.py',
})
LIVE_FILES = frozenset({'test_hunyuandit_acceptance.py'})


def partition(selectors):
    """Expand directories without importing tests. Preserve node selectors."""
    groups = {}
    seen = set()
    for selector in selectors:
        path = Path(selector.split('::', 1)[0])
        if not path.exists():
            raise ValueError(f'Test path does not exist: {path}')
        files = sorted(path.rglob('test_*.py')) if path.is_dir() else [path]
        for file in files:
            if any(part in {'.git', '.worktrees', '__pycache__'} for part in file.parts):
                continue
            target = str(file) if path.is_dir() else selector
            if target in seen:
                continue
            seen.add(target)
            if file.name in LIVE_FILES:
                cohort = f'live:{file.name}'
            elif file.name in STUB_FILES:
                cohort = f'stub:{file.name}'
            else:
                cohort = 'real'
            groups.setdefault(cohort, []).append(target)
    return dict(sorted(groups.items(), key=lambda item: (item[0].startswith('live:'), item[0])))


def guard_collection(session):
    """Reject mixed cohorts before test modules execute."""
    import pytest

    try:
        groups = partition(session.config.args)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    if len(groups) > 1:
        raise pytest.UsageError(
            'Mixed test collection can corrupt Torch imports. '
            'Use python -m tests.run <paths> -- <pytest options>.'
        )
