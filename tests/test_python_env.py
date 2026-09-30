"""Reject wrong interpreters before scientific imports or test collection."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def env_sandbox(tmp_path):
    suite = tmp_path / 'tests'
    suite.mkdir()
    (suite / '__init__.py').touch()
    for name in ('conftest.py', 'isolation.py', 'run.py', 'python_env.py'):
        source = REPO / 'tests' / name
        if source.exists():
            (suite / name).write_text(source.read_text())
    (suite / 'test_probe.py').write_text(
        'from pathlib import Path\n'
        'Path("collected").touch()\n'
        'def test_probe():\n    assert True\n'
    )
    (tmp_path / 'sitecustomize.py').write_text(
        'import sys\nfrom pathlib import Path\n'
        'class ImportProbe:\n'
        '    def find_spec(self, fullname, path=None, target=None):\n'
        '        if fullname in ("numpy", "PIL", "torch", "diffusers"):\n'
        '            Path("scientific-imported").touch()\n'
        'sys.meta_path.insert(0, ImportProbe())\n'
    )
    return tmp_path


def invoke(root, entrypoint, **settings):
    env = dict(os.environ, PYTHONPATH=str(root), PYTEST_ADDOPTS='',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.pop('CONDA_PREFIX', None)
    env.pop('ST_TEST_PYTHON_PREFIX', None)
    env.update(settings)
    command = [sys.executable, '-m', entrypoint, 'tests/test_probe.py']
    if entrypoint == 'tests.run':
        command.append('--')
    return subprocess.run(command + ['-q'], cwd=root, env=env,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('entrypoint', ['pytest', 'tests.run'])
@pytest.mark.parametrize('stale_activation', [False, True])
def test_wrong_prefix_fails_before_imports(env_sandbox, entrypoint, stale_activation):
    expected = str(env_sandbox / 'expected-env')
    settings = {'ST_TEST_PYTHON_PREFIX': expected}
    if stale_activation:
        settings['CONDA_PREFIX'] = expected
    result = invoke(env_sandbox, entrypoint, **settings)
    assert result.returncode == 4, result.stdout + result.stderr
    assert sys.executable in result.stderr
    assert str(Path(expected) / 'bin/python') in result.stderr
    assert 'conda activate' in result.stderr
    assert not (env_sandbox / 'scientific-imported').exists()
    assert not (env_sandbox / 'collected').exists()
    assert 'Test cohort:' not in result.stdout


@pytest.mark.parametrize('entrypoint', ['pytest', 'tests.run'])
def test_expected_interpreter_needs_no_activation(env_sandbox, entrypoint):
    result = invoke(env_sandbox, entrypoint, ST_TEST_PYTHON_PREFIX=sys.prefix)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout


@pytest.mark.parametrize('entrypoint', ['pytest', 'tests.run'])
def test_default_prefix_names_project_interpreter(env_sandbox, entrypoint):
    result = invoke(env_sandbox, entrypoint, HOME=str(env_sandbox))
    expected = env_sandbox / 'miniforge3/envs/stability-toys/bin/python'
    assert result.returncode == 4, result.stdout + result.stderr
    assert str(expected) in result.stderr
    assert 'conda activate stability-toys' in result.stderr
    assert not (env_sandbox / 'scientific-imported').exists()


def test_equivalent_prefix_symlink_is_allowed(env_sandbox):
    alias = env_sandbox / 'env-alias'
    alias.symlink_to(sys.prefix, target_is_directory=True)
    result = invoke(env_sandbox, 'tests.run', ST_TEST_PYTHON_PREFIX=str(alias))
    assert result.returncode == 0, result.stdout + result.stderr


def test_make_uses_interpreter_and_forwards_prefix_override(env_sandbox):
    env = dict(os.environ, PYTHONPATH=str(env_sandbox), PYTEST_ADDOPTS='',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', CONDA_PREFIX='/stale-shell-env',
               HOME=str(env_sandbox))
    result = subprocess.run(
        ['make', '-f', str(REPO / 'Makefile.test'), 'local-test',
         f'PYTHON={sys.executable}', f'EXPECTED_CONDA_PREFIX={sys.prefix}',
         'TEST=tests/test_probe.py', 'PYTEST_ARGS=-q'],
        cwd=env_sandbox, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout


def test_test_containers_declare_expected_python_prefix():
    import yaml

    compose = yaml.safe_load((REPO / 'docker-compose.test.yml').read_text())
    assert compose['services']['test']['environment']['ST_TEST_PYTHON_PREFIX'] == '/usr/local'
    assert 'ENV ST_TEST_PYTHON_PREFIX=/usr/local' in (REPO / 'Dockerfile.test').read_text()
