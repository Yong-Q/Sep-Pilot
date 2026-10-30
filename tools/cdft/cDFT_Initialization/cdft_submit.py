"""
cDFT batch submission: set up job directory, generate PBS script, submit via qsub.

Directory layout created per submission:
  {data_dir}/{timestamp}/
  ├── input/          # original .dat files copied from input_dat_dir
  ├── input_dir/      # run directories, one per MOF (created by PBS script)
  │   ├── MOF-5/
  │   │   ├── input.dat
  │   │   ├── output.dat
  │   │   └── hr.dat
  │   └── ...
  └── results.csv     # collected after all calculations finish
"""

import os
import re
import sys
import shutil
import subprocess
import json
import shlex
from datetime import datetime

_INIT_DIR = os.path.dirname(os.path.abspath(__file__))
_COLLECT_SCRIPT = os.path.join(_INIT_DIR, 'collect_cdft_results.py')
if _INIT_DIR not in sys.path:sys.path.insert(0,_INIT_DIR)

# ── PBS script template ──────────────────────────────────────────────────────
_PBS_TEMPLATE = r"""#!/bin/bash
#SBATCH -J cdft_{job_name}
#SBATCH -p {partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_processes}
#SBATCH -t {walltime}
{nodelist_line}{memory_line}{dependency_line}#SBATCH -o {work_dir}/slurm_%j.log
#SBATCH -e {work_dir}/slurm_%j.err

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUM_PROCESSES="${{SLURM_CPUS_PER_TASK-{num_processes}}}"
if ! [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid CPU allocation for cDFT worker count" >&2
    exit 2
fi
{execution_command}
"""


def _load_cdft_config():
    # Project-LOCAL settings file ships with this folder (self-contained for
    # git release). If present it wins — no dependency on the external
    # utils/config.py tree. Otherwise fall back to (external utils.config →
    # default _INIT_DIR paths).
    local_cfg = os.path.join(_INIT_DIR, 'cdft_config.json')
    if os.path.isfile(local_cfg):
        try:
            import json as _json
            with open(local_cfg, 'r', encoding='utf-8') as _f:
                cfg = _json.load(_f)
            return {
                'executable':    cfg.get('executable') or os.path.join(_INIT_DIR, 'cDFT', 'DM_cdft'),
                'timeout':       cfg.get('timeout', 3000),
                'num_processes': cfg.get('num_processes', 40),
                'data_dir':      cfg.get('data_dir') or os.path.join(_INIT_DIR, 'cdft_data'),
                'walltime':      cfg.get('walltime', '72:00:00'),
                'nodelist':      cfg.get('nodelist', ''),
                'scratch_policy':cfg.get('scratch_policy'),
            }
        except Exception:
            pass
    try:
        root = os.path.join(_INIT_DIR, '..', '..', '..', '..')
        sys.path.insert(0, os.path.abspath(root))
        from utils.config import get
        return {
            'executable':    get('cdft', 'executable'),
            'timeout':       get('cdft', 'timeout', 3000),
            'num_processes': get('cdft', 'num_processes', 40),
            'data_dir':      get('cdft', 'data_dir'),
            'walltime':      get('cdft', 'walltime', '72:00:00'),
            'nodelist':      get('cdft', 'nodelist', ''),
            'scratch_policy':get('cdft', 'scratch_policy', None),
        }
    except Exception:
        return {
            'executable':    os.path.join(_INIT_DIR, 'cDFT', 'DM_cdft'),
            'timeout':       3000,
            'num_processes': 40,
            'data_dir':      os.path.join(_INIT_DIR, 'cdft_data'),
            'walltime':      '72:00:00',
            'nodelist':      '',
        }


# DM_cdft needs a recent glibc (2.28 / Rocky 8.x). Submitting to a CentOS 7
# node (glibc 2.17) makes it crash with GLIBC_* errors — exactly the failure
# seen in the field (job on node13). Default to high-glibc nodes listed in
# config/site_overrides.glibc_compatibility (the canonical copy lives in
# ~/.claude/nodes_glibc.md). The registry layer can override via `nodelist`.
_GLIBC_OK_NODES = [
    "node03", "node07", "node08", "node18", "node19", "node20", "node21",
    "node26", "node27", "node28", "node29", "node30",
]


def _default_nodelist():
    """Return comma-separated node list for high-glibc nodes (or '' if none).

    Reads config/runtime.json site_overrides.glibc_compatibility first (the
    live canonical list, maintained alongside ~/.claude/nodes_glibc.md), and
    falls back to the built-in _GLIBC_OK_NODES list below.
    """
    nodes = _GLIBC_OK_NODES[:]
    try:
        import json as _json
        rt = os.path.join(_INIT_DIR, '..', '..', '..', '..', 'config', 'runtime.json')
        rt = os.path.abspath(rt)
        if os.path.isfile(rt):
            with open(rt, 'r', encoding='utf-8') as _f:
                data = _json.load(_f)
            so = data.get("site_overrides", {}).get("glibc_compatibility", {})
            pref = (so.get("preferred_high_glibc_pbs_nodes")
                    or so.get("preferred_high_glibc_nodes"))
            if pref:
                nodes = list(pref)
    except Exception:
        pass
    return ",".join(nodes)


def submit_cdft_batch(
    input_dat_dir,
    gases,
    node=None,
    ppn=None,
    num_processes=None,
    timeout=None,
    walltime=None,
    executable=None,
    data_dir=None,
    dry_run=False,
    nodelist=None,
    partition=None,
    memory_mb=None,
    scratch_policy=None,
    wait_job_ids=None,
):
    """
    Set up job directory, generate PBS script, and submit via qsub.

    Parameters
    ----------
    input_dat_dir : str
        Directory containing .dat files generated by data_input.py.
    gases : list of str
        Gas names in order (e.g. ['CO2', 'CO']).  Used to label CSV columns.
    node : str, optional
        PBS node name (e.g. 'cpu').  None → auto-select.
    ppn : int, optional
        Processors per node for PBS header (default = num_processes).
    num_processes : int, optional
        xargs parallel jobs (overrides config).
    timeout : int, optional
        Per-MOF timeout in seconds (overrides config).
    executable : str, optional
        Path to DM_cdft (overrides config).
    data_dir : str, optional
        Root directory for job folders (overrides config).
    dry_run : bool
        If True, generate script but do not submit.
    nodelist : str, optional
        Comma-separated compute nodes for #SBATCH -w. DM_cdft requires
        glibc 2.28 (Rocky 8.x); default = high-glibc nodes from config.
        Pass '' to disable node pinning.

    Returns
    -------
    dict
        job_id, work_dir, script_path, status
    """
    cfg = _load_cdft_config()
    executable    = executable    or cfg['executable']
    timeout = cfg['timeout'] if timeout is None else timeout
    if not dry_run and (not isinstance(memory_mb,int) or isinstance(memory_mb,bool) or memory_mb<64):
        raise ValueError('finite memory_mb >=64 MiB is required before real cDFT submission; implicit whole-node RAM is forbidden')
    num_processes = num_processes or cfg['num_processes']
    data_dir      = data_dir      or cfg['data_dir']
    walltime      = walltime      or cfg['walltime']
    if type(timeout) is not int or timeout<1:
        raise ValueError('timeout must be a positive integer number of seconds')
    match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})',str(walltime))
    if not match or int(match[3])>=60 or int(match[4])>=60:
        raise ValueError('walltime must use [D-]HH:MM:SS')
    wall_seconds=int(match[1] or 0)*86400+int(match[2])*3600+int(match[3])*60+int(match[4])
    if timeout>wall_seconds:raise ValueError('per-case timeout cannot exceed allocated walltime')
    if wait_job_ids is None:wait_job_ids=[]
    if not isinstance(wait_job_ids,list) or len(wait_job_ids)>64:
        raise ValueError('wait_job_ids must be a bounded list of native-authorized job IDs')
    dependencies=[]
    for job in wait_job_ids:
        if isinstance(job,bool) or not isinstance(job,(str,int)) or not re.fullmatch(r'[1-9][0-9]*',str(job)):
            raise ValueError('wait_job_ids contains an invalid scheduler job ID')
        if str(job) not in dependencies:dependencies.append(str(job))
    from cdft_scratch import normalized_policy, file_fact
    # Only the operator's private installation configuration or a trusted host
    # caller supplies this policy; it is deliberately absent from model schema.
    policy=normalized_policy(cfg.get('scratch_policy') if scratch_policy is None else scratch_policy)

    # DM_cdft needs glibc 2.28. Restrict the batch to high-glibc compute nodes
    # unless the caller explicitly passes a nodelist ('' disables the filter).
    if nodelist is None:
        nodelist = cfg.get('nodelist') or _default_nodelist()

    if not dry_run:
        from agents.placement_budget import project_policy, validate_memory_budget
        from pathlib import Path
        validate_memory_budget(project_policy(Path(__file__).resolve().parents[3]),
            cpus=num_processes,memory_mb=memory_mb,nodelist=nodelist or '',partition=partition or 'compute')

    if not os.path.isfile(executable):
        raise FileNotFoundError(f"DM_cdft executable not found: {executable}")
    if not os.path.isdir(input_dat_dir):
        raise NotADirectoryError(f"input_dat_dir not found: {input_dat_dir}")

    import importlib.util
    from pathlib import Path
    guard_path = Path(__file__).resolve().parents[3] / 'agents/cdft_grid_guard.py'
    guard_spec = importlib.util.spec_from_file_location('_cdft_submit_grid_guard', guard_path)
    grid_guard = importlib.util.module_from_spec(guard_spec); guard_spec.loader.exec_module(grid_guard)
    grid_proof = grid_guard.inspect_grid_directory(input_dat_dir)
    dat_files = [item['path'] for item in grid_proof['items']]

    # Create job work directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    work_dir  = os.path.join(data_dir, timestamp)
    input_dir = os.path.join(work_dir, 'input')
    os.makedirs(input_dir, exist_ok=False)

    # Copy input .dat files
    for fname in dat_files:
        shutil.copy2(os.path.join(input_dat_dir, fname), os.path.join(input_dir, fname))
    if grid_guard.inspect_grid_directory(input_dir) != grid_proof:
        raise ValueError('native staged inputs changed after grid verification; submission refused')

    # Build SLURM node spec (all computations go through sbatch)
    ppn = ppn or num_processes
    partition = partition or "compute"

    job_name  = f"batch_{timestamp}"
    gases_str = ' '.join(gases)

    runner=os.path.join(work_dir,'cdft_scratch.py')
    collector=os.path.join(work_dir,'collect_cdft_results.py')
    shutil.copy2(os.path.join(_INIT_DIR,'cdft_scratch.py'),runner)
    shutil.copy2(_COLLECT_SCRIPT,collector)
    common=['--work-dir',work_dir,'--input-dat-dir',input_dir,'--executable',executable,
        '--collect-script',collector,'--gases',json.dumps(gases),'--timeout',str(timeout)]
    def command(action,extra=()):
        # Only Slurm's numeric job/CPU variables expand; every configured string is shell-quoted.
        args=[sys.executable,runner,action,*common,*extra]
        return shlex.join(args)+' --job-id "${SLURM_JOB_ID:?}" --workers "$NUM_PROCESSES"'
    shared=command('shared')
    execution_command=shared+'\nexit $?'
    if policy:
        expected_inputs={name:file_fact(os.path.join(input_dir,name),16*1024*1024) for name in sorted(dat_files)}
        extra=['--policy',json.dumps(policy,sort_keys=True),'--memory-mb',str(memory_mb or 0),
            '--expected-inputs',json.dumps(expected_inputs,sort_keys=True),
            '--expected-executable-sha256',file_fact(executable,256*1024*1024)['sha256']]
        execution_command=command('dispatch',extra)+'\nstatus=$?\nif [ "$status" -ne 78 ]; then exit "$status"; fi\n'+shared+'\nexit $?'
    script_content = _PBS_TEMPLATE.format(
        job_name=job_name,partition=partition,walltime=walltime,
        nodelist_line=f"#SBATCH -w {nodelist}\n" if nodelist else "",
        memory_line=f"#SBATCH --mem={int(memory_mb)}M\n" if memory_mb else "",
        dependency_line='#SBATCH --dependency=afterany:'+':'.join(dependencies)+'\n' if dependencies else '',
        work_dir=work_dir,num_processes=num_processes,execution_command=execution_command,
    )

    from agents.placement_budget import submission_flags, clean_submission_environment
    from pathlib import Path
    flags = submission_flags(script_content, project_root=Path(__file__).resolve().parents[3]) if not dry_run else []
    scoped = any(flag.startswith(('--mem=', '--mem-per-cpu=')) for flag in flags)
    script_path = os.path.join(work_dir, 'cdft_job.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    result = {
        'job_id':      None,
        'work_dir':    work_dir,
        'script_path': script_path,
        'n_mofs':      len(dat_files),
        'status':      'ready',
        'timeout_seconds':timeout,
        'wait_job_ids':dependencies,
        'storage_mode':'operator_opt_in_scratch' if policy else 'shared',
        'execution_runner_sha256':file_fact(runner,4*1024*1024)['sha256'],
    }

    if dry_run:
        result['status'] = 'dry_run'
        return result

    try:
        proc = subprocess.run(
            ['sbatch', *flags, script_path],
            capture_output=True, text=True, check=True,
            **({'env': clean_submission_environment()} if scoped else {})
        )
        # ⚠️ 必须提取**纯数字** job_id，不能把整行 "Submitted batch job 3697"
        # 当 job_id —— 否则 JobWatch 拿带前缀的字符串去 squeue 查不到，
        # 状态永远 UNKNOWN，且会被当成"成功完成"把 TaskLine 步骤误标 ✅，
        # 下游接管 agent 就"不知道继续干什么"。
        _m = re.search(r"Submitted batch job (\d+)", proc.stdout)
        job_id = _m.group(1) if _m else proc.stdout.strip()
        result['job_id'] = job_id
        result['status'] = 'submitted'
    except FileNotFoundError:
        result['status'] = 'sbatch_not_found'
        result['note']   = 'sbatch not available; run script manually'
    except subprocess.CalledProcessError as e:
        result['status'] = 'submit_failed'
        result['note']   = e.stderr.strip()

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Submit cDFT batch job to PBS')
    parser.add_argument('input_dat_dir', help='Directory with .dat input files')
    parser.add_argument('--gases', nargs='+', required=True, help='Gas names')
    parser.add_argument('--node',          default=None)
    parser.add_argument('--nodelist',      default=None, help="Comma-separated node list for #SBATCH -w (DM_cdft needs glibc 2.28). Default: high-glibc nodes from config.")
    parser.add_argument('--ppn',           type=int, default=None)
    parser.add_argument('--num_processes', type=int, default=None)
    parser.add_argument('--memory_mb',type=int,default=None,help='Finite Slurm RAM budget in MiB; required for real submission')
    parser.add_argument('--timeout',       type=int, default=None)
    parser.add_argument('--dry_run',       action='store_true')
    args = parser.parse_args()

    r = submit_cdft_batch(
        input_dat_dir=args.input_dat_dir,
        gases=args.gases,
        node=args.node,
        ppn=args.ppn,
        num_processes=args.num_processes,
        memory_mb=args.memory_mb,
        timeout=args.timeout,
        dry_run=args.dry_run,
        nodelist=args.nodelist,
    )
    print(f"Status   : {r['status']}")
    print(f"Job ID   : {r.get('job_id', 'N/A')}")
    print(f"Work dir : {r['work_dir']}")
    print(f"MOFs     : {r['n_mofs']}")
    if 'note' in r:
        print(f"Note     : {r['note']}")
