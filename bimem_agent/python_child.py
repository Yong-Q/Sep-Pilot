"""Native project Python launch plumbing, independent of task working directories.

Only host code constructs inline programs. No shell, package installation or
execution grant is provided here. Operator environment values remain intact;
the installed SDK root is prepended for its own explicit package imports.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys


def project_root():
    return Path(__file__).resolve().parents[1]


def project_python_environment(environment=None):
    result = dict(os.environ if environment is None else environment)
    root = str(project_root())
    previous = result.get('PYTHONPATH', '')
    if not isinstance(previous, str):
        raise ValueError('PYTHONPATH must be text')
    if not previous or previous.split(os.pathsep, 1)[0] != root:
        result['PYTHONPATH'] = root + (os.pathsep + previous if previous else '')
    return result


def resolve_child_executable(executable, environment=None):
    """Freeze executable lookup before switching into a task directory.

    Explicit relative installation paths belong to the SDK root. Bare command
    names retain the operator's PATH lookup from the launching process. Keep
    interpreter symlinks intact: resolving a virtualenv's python symlink would
    silently select its base interpreter and lose the virtualenv packages.
    """
    value = os.fsdecode(os.fspath(executable))
    if not value.strip():
        raise ValueError('native Python executable must be nonempty')
    path = Path(value).expanduser()
    if path.is_absolute():
        return os.path.abspath(path)
    if os.path.dirname(value):
        return os.path.abspath(project_root() / path)
    env = os.environ if environment is None else environment
    search = os.pathsep.join(os.path.abspath(entry or os.curdir)
                             for entry in os.get_exec_path(env))
    located = shutil.which(value, path=search)
    if located is None:
        raise FileNotFoundError('Native Python executable not found: ' + value)
    return os.path.abspath(located)


def validate_child_cwd(cwd, owned_root=None):
    selected = Path(cwd).expanduser() if cwd is not None else None
    if selected is not None:
        selected = (selected if selected.is_absolute() else project_root() / selected).resolve()
    if owned_root is not None:
        owner = Path(owned_root).expanduser().resolve()
        if selected is None or not selected.is_relative_to(owner):
            raise PermissionError('native Python child cwd must stay in its owned session')
    return selected


def run_project_python(python, code, *, cwd=None, timeout=900, environment=None,
                       owned_root=None, create_cwd=False):
    if not isinstance(code, str):
        raise TypeError('native Python source must be text')
    if len(code.encode('utf-8')) > 16 * 1024 * 1024:
        raise ValueError('native Python source exceeds its bounded input')
    selected = validate_child_cwd(cwd, owned_root)
    environment = project_python_environment(environment)
    interpreter = resolve_child_executable(python, environment)
    if create_cwd and selected is not None:
        selected.mkdir(parents=True, exist_ok=True)
    # An absolute trusted script, rather than -c, prevents Python's implicit
    # cwd import entry from resolving task-owned agents/sitecustomize modules.
    # Source travels over stdin, not a shell or the process argument list.
    return subprocess.run([interpreter, str(Path(__file__).resolve()), '--native-inline'],
        input=code, capture_output=True, text=True, timeout=timeout,
        cwd=str(selected if selected is not None else project_root()),
        env=environment)


def _main():
    if sys.argv[1:] != ['--native-inline']:
        raise SystemExit('This entry is reserved for native project Python programs')
    source = sys.stdin.read(16 * 1024 * 1024 + 1)
    if len(source.encode('utf-8')) > 16 * 1024 * 1024:
        raise ValueError('native Python source exceeds its bounded input')
    root = str(project_root())
    sys.path[:] = [root] + [value for value in sys.path if value and value != root]
    sys.argv = ['-c']
    scope = {'__name__': '__main__'}
    exec(compile(source, '<native-project-child>', 'exec'), scope, scope)


if __name__ == '__main__':
    _main()
