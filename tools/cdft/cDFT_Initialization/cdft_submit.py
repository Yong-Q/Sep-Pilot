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
from datetime import datetime

_INIT_DIR = os.path.dirname(os.path.abspath(__file__))
_COLLECT_SCRIPT = os.path.join(_INIT_DIR, 'collect_cdft_results.py')

# ── PBS script template ──────────────────────────────────────────────────────
_PBS_TEMPLATE = r"""#!/bin/bash
#SBATCH -J cdft_{job_name}
#SBATCH -p {partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_processes}
#SBATCH -t {walltime}
{nodelist_line}{memory_line}#SBATCH -o {work_dir}/slurm_%j.log
#SBATCH -e {work_dir}/slurm_%j.err

export CDFT_EXE="{executable}"
export CDFT_WORK_DIR="{work_dir}"
export CDFT_TIMEOUT={timeout}
NUM_PROCESSES={num_processes}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

process_input() {{
    local src="$1"
    local mof_name
    mof_name=$(basename "${{src%.dat}}")
    echo "[$(date '+%H:%M:%S')] Processing ${{mof_name}}..."

    if [ ! -f "$src" ]; then
        echo "  ERROR: $src not found, skipping."
        return 1
    fi

    local run_dir="${{CDFT_WORK_DIR}}/input_dir/${{mof_name}}"
    mkdir -p "$run_dir"
    cp "$src" "$run_dir/input.dat"

    pushd "$run_dir" > /dev/null || return 1
    "${{CDFT_EXE}}" &
    local pid=$!

    local start elapsed
    start=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        elapsed=$(( $(date +%s) - start ))
        if [ "$elapsed" -gt "${{CDFT_TIMEOUT}}" ]; then
            echo "  TIMEOUT: ${{mof_name}} exceeded ${{CDFT_TIMEOUT}}s, killing."
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            popd > /dev/null
            return 1
        fi
        sleep 5
    done
    wait "$pid" 2>/dev/null
    popd > /dev/null

    if grep -q "Molecule" "${{run_dir}}/output.dat" 2>/dev/null; then
        echo "  SUCCESS: ${{mof_name}}"
        return 0
    else
        echo "  FAILED: ${{mof_name}} (no valid output)"
        return 1
    fi
}}

export -f process_input
export CDFT_EXE CDFT_WORK_DIR CDFT_TIMEOUT

find "{work_dir}/input" -maxdepth 1 -type f -name '*.dat' -print0 | \
    xargs -0 -P "$NUM_PROCESSES" -I{{}} bash -c 'process_input "$@"' _ {{}}

echo "[$(date '+%H:%M:%S')] All cDFT calculations complete."

# Collect results into CSV
python3 "{collect_script}" \
    --work_dir "{work_dir}" \
    --gases {gases_str} \
    --output "{work_dir}/results.csv"

echo "Results saved to {work_dir}/results.csv"
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
    timeout       = timeout       or cfg['timeout']
    if not dry_run and (not isinstance(memory_mb,int) or isinstance(memory_mb,bool) or memory_mb<64):
        raise ValueError('finite memory_mb >=64 MiB is required before real cDFT submission; implicit whole-node RAM is forbidden')
    num_processes = num_processes or cfg['num_processes']
    data_dir      = data_dir      or cfg['data_dir']
    walltime      = walltime      or cfg['walltime']
    # DM_cdft needs glibc 2.28. Restrict the batch to high-glibc compute nodes
    # unless the caller explicitly passes a nodelist ('' disables the filter).
    if nodelist is None:
        nodelist = cfg.get('nodelist') or _default_nodelist()

    if not os.path.isfile(executable):
        raise FileNotFoundError(f"DM_cdft executable not found: {executable}")
    if not os.path.isdir(input_dat_dir):
        raise NotADirectoryError(f"input_dat_dir not found: {input_dat_dir}")

    dat_files = [f for f in os.listdir(input_dat_dir) if f.endswith('.dat')]
    if not dat_files:
        raise ValueError(f"No .dat files found in {input_dat_dir}")

    # Create job work directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir  = os.path.join(data_dir, timestamp)
    input_dir = os.path.join(work_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)

    # Copy input .dat files
    for fname in dat_files:
        shutil.copy2(os.path.join(input_dat_dir, fname), os.path.join(input_dir, fname))

    # Build SLURM node spec (all computations go through sbatch)
    ppn = ppn or num_processes
    partition = partition or "compute"

    job_name  = f"batch_{timestamp}"
    gases_str = ' '.join(gases)

    script_content = _PBS_TEMPLATE.format(
        job_name=job_name,
        partition=partition,
        walltime=walltime,
        nodelist_line=f"#SBATCH -w {nodelist}\n" if nodelist else "",
        memory_line=f"#SBATCH --mem={int(memory_mb)}M\n" if memory_mb else "",
        work_dir=work_dir,
        executable=executable,
        timeout=timeout,
        num_processes=num_processes,
        collect_script=_COLLECT_SCRIPT,
        gases_str=gases_str,
    )

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
    }

    if dry_run:
        result['status'] = 'dry_run'
        return result

    try:
        proc = subprocess.run(
            ['sbatch', script_path],
            capture_output=True, text=True, check=True
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
