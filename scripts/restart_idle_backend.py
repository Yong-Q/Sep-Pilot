"""Bounded local deployment. Never kills jobs or unrelated servers.

The normal path remains strictly idle-only.  ``--maintenance-interrupted`` is
the explicit hand-off path for an operator-requested release: every live or
lifecycle-busy conversation must already be durably frozen, and no workflow
node may be executing.  The replacement process can then recover mailbox
claims from the dead PID without losing or duplicating submitted jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


RELEASE_SOURCE_FILES = (
    'api.py', 'auth.py', 'agents/session.py', 'agents/scientific_review.py',
    'agents/defns.py', 'agents/goal_contract.py', 'agents/charge_contract.py',
    'agents/unit_contract.py', 'agents/forcefield_catalog.py',
    'env/forcefield_sources.json', 'env/physical_units.json',
    'registry/catalog.json', 'agents/workspace.py', 'agents/workflow_patch.py',
    'agents/workflow_compiler.py', 'agents/parallel_workflow.py',
    'agents/orchestration_chain.py', 'agents/workflow_view.py',
    'agents/reflection_progress.py', 'agents/output_contract.py',
    'agents/project_validation.py', 'agents/registry.py', 'agents/recovery.py',
    'agents/state_io.py', 'agents/task_line.py', 'agents/job_watch.py',
    'agents/slurm.py', 'agents/node_inventory.py', 'agents/resource_review.py',
    'tools/cdft/cDFT_Initialization/cdft_submit.py',
    'tools/cdft/cDFT_Initialization/data_input.py',
    'frontend/build/index.html', 'frontend/public/avatars/avatars.json',
    *tuple(f'frontend/public/avatars/anime-{index:02d}.png' for index in range(1, 11)),
    'agents/job_control.py', 'agents/control_policy.py',
    'scripts/restart_idle_backend.py',
)


def release_source_hashes(root):
    """Hash every runtime source that must stay stable during deployment."""
    root = Path(root)
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in RELEASE_SOURCE_FILES
    }


def running_python(proc, env, fallback):
    """Reuse the executable that actually owns the server process.

    Shells can retain a stale CONDA_PREFIX after launching an executable from
    another environment. /proc/<pid>/exe is the authoritative interpreter.
    """
    try:
        executable = (Path(proc) / 'exe').resolve(strict=True)
        if executable.is_file():
            return executable
    except (FileNotFoundError, OSError, RuntimeError):
        pass
    candidate = Path(env.get('CONDA_PREFIX', '')) / 'bin' / 'python'
    return candidate if candidate.is_file() else Path(fallback)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('--maintenance-interrupted', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    release_sources = release_source_hashes(root)
    proc = Path('/proc') / str(args.pid)
    if (proc / 'cwd').resolve() != root or b'uvicorn\x00api:app' not in (proc / 'cmdline').read_bytes():
        raise RuntimeError('PID is not the api:app server in this exact project')
    env = dict(item.split('=', 1) for item in (proc / 'environ').read_bytes().decode().split('\0') if '=' in item)
    python = running_python(proc, env, sys.executable)
    def status(endpoint):
        with urllib.request.urlopen('http://127.0.0.1:8000/' + endpoint, timeout=3) as response:
            return json.load(response)
    def frozen_scopes():
        users = json.loads((root / 'users.json').read_text())
        frozen, active = set(), set()
        for username, user in users.items():
            for conv_id, conv in user.get('conversations', {}).items():
                scope = (username, conv_id)
                if conv.get('interrupt_requested'):
                    frozen.add(scope)
                if conv.get('is_processing'):
                    active.add(scope)
        return frozen, active

    system_state = status('api/system/status')
    frozen, persisted_active = frozen_scopes()
    if system_state['active_queries']:
        if not args.maintenance_interrupted:
            raise RuntimeError('active main turn; deployment deferred')
        if len(persisted_active) != system_state['active_queries'] or not persisted_active <= frozen:
            raise RuntimeError('active turn is not durably interrupted; deployment deferred')
    busy_scopes = set()
    for index, path in enumerate((root / 'runs').glob('*/*/lifecycle.json')):
        if index >= 10000: raise RuntimeError('scope audit exceeded bound; refusing restart')
        state = json.loads(path.read_text())
        if any(role.get('busy_event_id') for role in state.get('agents', {}).values()):
            relative = path.relative_to(root / 'runs')
            scope = (relative.parts[0], relative.parts[1])
            busy_scopes.add(scope)
            if not args.maintenance_interrupted or scope not in frozen:
                raise RuntimeError('busy lifecycle observer is not durably interrupted; deployment deferred')
    runtime_file = root / 'data/state/parallel_workflows.json'
    if runtime_file.exists():
        runtime = json.loads(runtime_file.read_text())
        if any(node.get('status') == 'running' for workflow in runtime.get('workflows', {}).values()
               for node in workflow.get('nodes', {}).values()):
            raise RuntimeError('active/unknown worker thread; deployment deferred')
    final_state = status('api/system/status')
    final_frozen, final_active = frozen_scopes()
    if final_state['active_queries']:
        if (not args.maintenance_interrupted or len(final_active) != final_state['active_queries']
                or not final_active <= final_frozen):
            raise RuntimeError('main became busy during audit; deployment deferred')
    if busy_scopes and not busy_scopes <= final_frozen:
        raise RuntimeError('lifecycle freeze changed during audit; deployment deferred')
    if release_source_hashes(root) != release_sources:
        raise RuntimeError('release sources changed during audit; deployment deferred')
    os.kill(args.pid, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while proc.exists() and time.monotonic() < deadline: time.sleep(.1)
    if proc.exists(): raise RuntimeError('server did not stop gracefully; no force kill or duplicate launch')
    log_path = root / 'logs/api_backend.log'
    with log_path.open('ab') as log:
        child = subprocess.Popen([str(python), '-u', '-m', 'uvicorn', 'api:app', '--host', '0.0.0.0',
            '--port', '8000', '--log-level', 'info', '--timeout-graceful-shutdown', '10'], cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True)
    # Restoring many durable conversation checkpoints is intentionally done
    # before readiness.  Production histories can need more than 20 seconds;
    # do not misreport a healthy unique replacement as a failed deployment.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if child.poll() is not None: raise RuntimeError(f'new backend exited; inspect {log_path}')
        try:
            health = status('health')
            if health['version'] == args.expected_version:
                sys.path.insert(0, str(root))
                from agents.state_io import write_checkpoint
                if release_source_hashes(root) != release_sources:
                    raise RuntimeError('release sources changed during startup; verification required')
                release_state = {'pid': child.pid, 'root': str(root), 'version': health['version'],
                    'log': str(log_path), 'updated_at': time.time(), 'source_sha256': release_sources}
                if args.maintenance_interrupted:
                    release_state['maintenance_interrupted_scopes'] = [list(scope) for scope in sorted(final_frozen)]
                write_checkpoint(root / 'logs/api_backend_state.json', release_state)
                release_dir = root / 'logs/releases'
                release_dir.mkdir(exist_ok=True)
                write_checkpoint(release_dir / f'{args.expected_version}_{child.pid}.json', release_state)
                print(json.dumps({'pid': child.pid, 'health': health, 'log': str(log_path)}, ensure_ascii=False))
                return
        except (OSError, ValueError):
            pass
        time.sleep(.2)
    raise RuntimeError(f'health verification timed out for PID {child.pid}; do not duplicate launch')


if __name__ == '__main__':
    main()
