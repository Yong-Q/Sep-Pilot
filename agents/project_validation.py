"""Fixed, bounded development diagnostics callable by the SDK maintainer."""
import os
import hashlib
from pathlib import Path
import subprocess
import sys
import uuid


def run_project_regressions():
    from .config import get_config
    from .workspace import session_dir
    root = Path(get_config().project_root).resolve()
    def source_snapshot():
        paths = [root / 'api.py', root / 'auth.py'] + list((root / 'agents').glob('*.py')) + list((root / 'tests').glob('*.py'))
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    tested_sources = source_snapshot()
    output = session_dir('validation')
    output.mkdir(parents=True, exist_ok=True)
    log = output / ('regressions_' + uuid.uuid4().hex + '.log')
    command = [sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider']
    env = {**os.environ, 'BIMEM_BACKGROUND_MONITOR':'0', 'PYTHONDONTWRITEBYTECODE':'1',
           'PYTHONPYCACHEPREFIX':str(output/('bytecode_'+uuid.uuid4().hex)),
           'BIMEM_JOB_WATCH_FILE':str(output/('validation_job_watch_'+uuid.uuid4().hex+'.json'))}
    try:
        with log.open('xb') as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=stream, timeout=240)
        text = log.read_text(errors='replace')
        current_sources = source_snapshot()
        changed = sorted(k for k in set(tested_sources) | set(current_sources) if tested_sources.get(k) != current_sources.get(k))
        return {'ok': result.returncode == 0 and not changed, 'exit_code': result.returncode, 'log_path':str(log.resolve()),
                'source_stable': not changed, 'changed_during_test': changed, 'source_sha256': tested_sources,
                'summary':text[-20000:], 'scope':'supplementary structural regressions; not a substitute for real SDK acceptance'}
    except subprocess.TimeoutExpired:
        return {'ok':False, 'error':'Supplementary regression budget exhausted', 'log_path':str(log.resolve())}


def build_project_frontend():
    from .config import get_config
    from .workspace import session_dir
    root=Path(get_config().project_root).resolve()
    log=session_dir('validation')/('frontend_build_'+uuid.uuid4().hex+'.log')
    try:
        with log.open('xb') as stream:
            result=subprocess.run(['npm','run','build'],cwd=root/'frontend',stdout=stream,stderr=stream,
                env={**os.environ,'GENERATE_SOURCEMAP':'false'},timeout=240)
        return {'ok':result.returncode==0,'exit_code':result.returncode,'log_path':str(log),'summary':log.read_text(errors='replace')[-16000:]}
    except subprocess.TimeoutExpired:return {'ok':False,'error':'frontend build budget exhausted','log_path':str(log)}
