"""Bounded, host-declared import health; never solver validation or submission.

Profiles contain only reviewed SDK module names and adapter paths. Caller/model
arguments cannot inject Python, module names, environment values, or commands.
The audit guard is defense in depth for trusted imports, not a hostile-code sandbox.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess


PROFILES = {'run_cdft': {
    'kind': 'project_python_child',
    'modules': ('agents.placement_budget', 'agents.node_inventory'),
    'adapters': (('cdft_submit', 'tools/cdft/cDFT_Initialization/cdft_submit.py'),),
}}


def _source_facts(root):
    """Freeze the small trusted Python source closure before starting imports."""
    paths = []
    for relative in ('agents', 'bimem_agent', 'tools/cdft/cDFT_Initialization'):
        paths.extend((root / relative).glob('*.py'))
    if len(paths) > 1024:
        raise ValueError('source_bound')
    facts, size = {}, 0
    for path in sorted(paths):
        raw = path.read_bytes()
        size += len(raw)
        if size > 32 * 1024 * 1024:
            raise ValueError('source_bound')
        stat = path.stat()
        facts[str(path.resolve())] = {'sha256': hashlib.sha256(raw).hexdigest(),
            'size': len(raw), 'identity': [stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns]}
    return facts


_CHILD = r'''
import contextlib, hashlib, importlib, importlib.util, importlib.machinery, io, json, os, sys
from pathlib import Path
payload = json.loads(PAYLOAD)
root = Path(payload['sdk_root']).resolve()
answer = {'ok': False, 'reason': 'import_failed', 'module_facts': [],
          'sdk_root': str(root), 'python': os.path.abspath(sys.executable),
          'cwd': str(Path.cwd().resolve()), 'python_version': list(sys.version_info[:3])}
def guard(event, args):
    if (event.startswith(('socket.', 'subprocess.', 'os.exec', 'os.spawn'))
            or event in {'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                         'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod',
                         'os.chown', 'os.link', 'os.symlink', 'os.truncate', 'os.chdir'}):
        raise PermissionError('probe_side_effect_denied')
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise PermissionError('probe_write_denied')
sys.dont_write_bytecode = True
# Compile source directly: an existing timestamp-valid pyc is not source proof.
def source_code(loader, fullname):
    path = loader.get_filename(fullname)
    return compile(loader.get_data(path), path, 'exec', dont_inherit=True)
importlib.machinery.SourceFileLoader.get_code = source_code
# Discard inherited cwd/PYTHONPATH entries: only SDK and interpreter libraries.
sys.path[:] = [str(root)] + [p for p in sys.path if p and
    (Path(p).resolve().is_relative_to(Path(sys.prefix).resolve()) or
     Path(p).resolve().is_relative_to(Path(sys.base_prefix).resolve()))]
sys.addaudithook(guard)
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for name in payload['modules']:
            importlib.import_module(name)
        for name, relative in payload['adapters']:
            spec = importlib.util.spec_from_file_location(name, root / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            continue
        expected = payload['sources'].get(str(path))
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('source_identity')
        answer['module_facts'].append({'name': name, 'path': str(path),
            'sha256': expected['sha256'], 'loader': type(module.__loader__).__name__,
            'package': getattr(module, '__package__', None),
            'spec_origin': getattr(getattr(module, '__spec__', None), 'origin', None),
            'source_before': expected, 'source_after': expected})
    required = set(payload['modules']) | {item[0] for item in payload['adapters']}
    if not required.issubset({item['name'] for item in answer['module_facts']}):
        raise ValueError('module_origin')
    answer.update(ok=True, reason='imports_verified')
except BaseException as error:
    answer['reason'] = 'import_' + type(error).__name__
print(json.dumps(answer, sort_keys=True))
'''


def inspect_runtime_health(tool, arguments, config, *, cwd=None, owned_root=None,
                           environment=None, timeout=10):
    """Host API. Explicit owner/cwd are for native controllers/tests only.

    Ordinary inspect_installation calls derive ownership from watch_context;
    arguments provide only the existing job_work_dir selection within that owner.
    """
    from bimem_agent.python_child import (project_root, run_project_python,
                                          resolve_child_executable, validate_child_cwd)
    profile = PROFILES.get(tool)
    result = {'ok': False, 'status': 'unverifiable', 'reason': 'unsupported_profile',
        'profile': profile['kind'] if profile else None, 'read_only': True,
        'solver_verified': False, 'required_modules': [], 'module_facts': [],
        'sdk_root': str(project_root()), 'python': None, 'cwd': None, 'owned_root': None,
        'source_before': None, 'source_after': None, 'configured_asset_root': None}
    if profile is None:
        return result
    result['required_modules'] = list(profile['modules']) + [p[0] for p in profile['adapters']]
    try:
        root = Path(config.project_root).resolve()
        if root != project_root().resolve():
            result['reason'] = 'sdk_root_mismatch'
            return result
        if owned_root is None:
            from .watch_context import get_context
            from .workspace import validate_scope_component
            context = get_context()
            if not context.get('username') or not context.get('conv_id'):
                result['reason'] = 'missing_owner'
                return result
            user = validate_scope_component(context['username'])
            session = validate_scope_component(context['conv_id'])
            owned_root = root / 'runs' / user / session
        owner = Path(owned_root).absolute()
        if owner.resolve() != owner or not owner.is_dir():
            raise ValueError('owner_unavailable')
        selected = cwd if cwd is not None else arguments.get('job_work_dir', owner / 'cdft')
        selected = validate_child_cwd(selected, owner)
        if not selected.is_dir():
            raise ValueError('cwd_unavailable')
        result.update(cwd=str(selected), owned_root=str(owner))
        from .installations import cdft_installation_files
        layout = cdft_installation_files(root, {'action': 'expand'})
        if 'framework_forcefield' in layout:
            result['configured_asset_root'] = str(layout['framework_forcefield'][0].parent)
        env = dict(os.environ if environment is None else environment)
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root), PYTHONNOUSERSITE='1')
        # Python startup cannot consume an operator-supplied startup script.
        env.pop('PYTHONSTARTUP', None)
        env.pop('PYTHONINSPECT', None)
        interpreter = resolve_child_executable(layout['python'][0], env)
        result['python'] = interpreter
        before = _source_facts(root)
        digest = lambda facts: hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
        result['source_before'] = digest(before)
        payload = dict(sdk_root=str(root), modules=profile['modules'],
                       adapters=profile['adapters'], sources=before)
        code = 'PAYLOAD = ' + repr(json.dumps(payload)) + '\n' + _CHILD
        child = run_project_python(interpreter, code, cwd=selected, owned_root=owner,
            environment=env, timeout=min(10, max(0.1, float(timeout))))
        after = _source_facts(root)
        result['source_after'] = digest(after)
        if before != after:
            result.update(status='failed', reason='source_changed')
            return result
        if child.returncode != 0 or len(child.stdout) > 1024 * 1024:
            result.update(status='failed', reason='child_failed')
            return result
        observed = json.loads(child.stdout)
        result['module_facts'] = observed['module_facts']
        if (observed['sdk_root'] != str(root) or observed['cwd'] != str(selected)
                or observed['python'] != interpreter):
            result.update(status='failed', reason='child_identity_mismatch')
            return result
        result.update(ok=observed['ok'] is True,
            status='passed' if observed['ok'] is True else 'failed',
            reason=observed['reason'], python_version=observed['python_version'])
    except subprocess.TimeoutExpired:
        result.update(status='failed', reason='timeout')
    except (OSError, ValueError, TypeError, KeyError, ImportError, PermissionError) as error:
        result.update(status='failed', reason='probe_' + type(error).__name__)
    return result
