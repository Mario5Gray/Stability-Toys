"""Run collection cohorts in fresh Python interpreters."""

import subprocess
import sys
import importlib.util

from tests.isolation import partition


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--' in args:
        boundary = args.index('--')
        paths, options = args[:boundary], args[boundary + 1:]
        if not paths:
            print('Specify test paths before --.', file=sys.stderr)
            return 4
    else:
        paths, options = args, []
    if any(path.startswith('-') for path in paths):
        print('Put pytest options after --.', file=sys.stderr)
        return 4
    try:
        groups = partition(paths or ['tests'])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    status = 5
    coverage_plugin = importlib.util.find_spec('pytest_cov') is not None
    for index, (cohort, files) in enumerate(groups.items()):
        print(f'Test cohort: {cohort} ({len(files)} files)', flush=True)
        command = [sys.executable, '-m', 'pytest', *files, *options]
        # Defer threshold until aggregate exists. First process starts fresh.
        if coverage_plugin:
            command += ['-p', 'pytest_cov']
            if index:
                command.append('--cov-append')
            if index < len(groups) - 1:
                command += ['--cov-fail-under=0', '--cov-report=']
        code = subprocess.call(command)
        if code not in (0, 5):
            status = code
            if code != 1 or '-x' in options or '--exitfirst' in options:
                break
        elif code == 0 and status == 5:
            status = 0
    return status


if __name__ == '__main__':
    raise SystemExit(main())
