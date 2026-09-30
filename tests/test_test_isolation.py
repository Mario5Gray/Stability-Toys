"""Process boundaries must precede test imports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def sandbox(tmp_path):
    suite = tmp_path / 'tests'
    suite.mkdir()
    (suite / '__init__.py').touch()
    for name in ('isolation.py', 'run.py'):
        source = Path(__file__).with_name(name)
        if source.exists():
            (suite / name).write_text(source.read_text())
    (suite / 'conftest.py').write_text(
        'from tests.isolation import guard_collection\n'
        'def pytest_collection(session):\n'
        '    guard_collection(session)\n'
    )
    (suite / 'test_hunyuandit_worker.py').write_text(
        'import sys\nfrom pathlib import Path\n'
        'Path("stub-imported").touch()\n'
        'sys.modules["_isolation_probe"] = object()\n'
        'def test_stub():\n    assert "_isolation_probe" in sys.modules\n'
    )
    (suite / 'test_real.py').write_text(
        'import sys\n'
        'assert "_isolation_probe" not in sys.modules\n'
        'def test_real():\n    assert "_isolation_probe" not in sys.modules\n'
    )
    return tmp_path


def invoke(sandbox, *args):
    env = dict(os.environ, PYTHONPATH=str(sandbox), PYTEST_ADDOPTS='',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    return subprocess.run([sys.executable, *args], cwd=sandbox, env=env,
                          text=True, capture_output=True, timeout=30)


def test_runner_separates_collection(sandbox):
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', '-q')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('1 passed') == 2


def test_direct_mixed_collection_rejects_before_import(sandbox):
    result = invoke(sandbox, '-m', 'pytest', 'tests', '-q')
    assert result.returncode == 4, result.stdout + result.stderr
    assert 'python -m tests.run' in result.stderr
    assert not (sandbox / 'stub-imported').exists()


def test_marker_filter_cannot_bypass_collection_guard(sandbox):
    result = invoke(sandbox, '-m', 'pytest', 'tests', '-k', 'real', '-q')
    assert result.returncode == 4, result.stdout + result.stderr
    assert 'Mixed test collection' in result.stderr
    assert not (sandbox / 'stub-imported').exists()


def test_runner_preserves_filter_and_empty_cohort(sandbox):
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', '-k', 'real', '-q')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout


def test_runner_preserves_no_tests_exit_code(sandbox):
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', '-k', 'absent', '-q')
    assert result.returncode == 5, result.stdout + result.stderr


def test_runner_preserves_failure_and_runs_other_cohort(sandbox):
    with (sandbox / 'tests/test_real.py').open('a') as stream:
        stream.write('def test_failure():\n    assert False\n')
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', '-q')
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.count('1 passed') >= 1
    assert (sandbox / 'stub-imported').exists()


def test_runner_exitfirst_stops_before_next_cohort(sandbox):
    with (sandbox / 'tests/test_real.py').open('a') as stream:
        stream.write('def test_failure():\n    assert False\n')
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', '-x', '-q')
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (sandbox / 'stub-imported').exists()


def test_runner_preserves_node_selector(sandbox):
    result = invoke(sandbox, '-m', 'tests.run', 'tests/test_real.py::test_real', '--', '-q')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (sandbox / 'stub-imported').exists()


@pytest.mark.parametrize('option', ['--ignore=x', '--deselect=tests/test_real.py::test_real'])
def test_runner_passes_path_options_to_pytest(sandbox, option):
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', option, '-q')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    if option.startswith('--deselect='):
        assert '1 deselected' in result.stdout


def test_runner_reports_unknown_pytest_option(sandbox):
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', '--unknown-path=x')
    assert result.returncode == 4, result.stdout + result.stderr
    assert 'unrecognized arguments: --unknown-path=x' in result.stderr
    assert not (sandbox / 'stub-imported').exists()


def test_runner_requires_paths_before_explicit_separator(sandbox):
    result = invoke(sandbox, '-m', 'tests.run', '--', '-q')
    assert result.returncode == 4, result.stdout + result.stderr
    assert 'Specify test paths before --.' in result.stderr
    assert 'Test cohort:' not in result.stdout
    assert not (sandbox / 'stub-imported').exists()


def test_runner_defaults_to_suite_without_arguments(sandbox):
    result = invoke(sandbox, '-m', 'tests.run')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('1 passed') == 2


def test_cuda_acceptance_cannot_collect_with_other_files(sandbox):
    (sandbox / 'tests/test_hunyuandit_acceptance.py').write_text('def test_gpu(): pass\n')
    result = invoke(sandbox, '-m', 'pytest', 'tests/test_real.py',
                    'tests/test_hunyuandit_acceptance.py', '-q')
    assert result.returncode == 4, result.stdout + result.stderr
    assert 'Mixed test collection' in result.stderr


def test_runner_combines_coverage_before_threshold(sandbox):
    import json

    (sandbox / 'covered.py').write_text('def real():\n    return 1\ndef stub():\n    return 2\n')
    for name, function in [('test_real.py', 'real'), ('test_hunyuandit_worker.py', 'stub')]:
        (sandbox / 'tests' / name).write_text(
            f'from covered import {function}\ndef test_{function}():\n    assert {function}()\n'
        )
    result = invoke(sandbox, '-m', 'tests.run', 'tests', '--', '-q', '-p', 'pytest_cov',
                    '--cov=covered', '--cov-report=json:coverage.json', '--cov-fail-under=100')
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((sandbox / 'coverage.json').read_text())
    assert report['totals']['percent_covered'] == 100


def test_compose_watch_invokes_runner(sandbox):
    import shlex
    import yaml
    from pytest_watcher.parse import parse_arguments

    compose = yaml.safe_load((Path(__file__).resolve().parents[1] / 'docker-compose.test.yml').read_text())
    command = shlex.split(compose['services']['test-watch']['command'])
    namespace, args = parse_arguments(command[1:])
    assert namespace.runner == 'python'
    result = invoke(sandbox, *args, '-o', 'addopts=')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('1 passed') == 2


def test_collection_module_mutations_have_stub_cohort():
    import ast
    from tests.isolation import STUB_FILES

    class CollectionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.mutates_modules = False

        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'sys' and node.attr == 'modules':
                self.mutates_modules = True
            self.generic_visit(node)

    unclassified = []
    for path in Path(__file__).parent.glob('test_*.py'):
        visitor = CollectionVisitor()
        visitor.visit(ast.parse(path.read_text()))
        if visitor.mutates_modules and path.name not in STUB_FILES:
            unclassified.append(path.name)
    assert not unclassified, f'Collection module access needs cohort review: {unclassified}'
