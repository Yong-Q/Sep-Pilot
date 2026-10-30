"""Zeo++ 孔结构特征计算（SLURM 版，SDK-LOCAL）

收敛原则（用户明确要求）：
  · 前面只确认几个必要参数（cif_path XOR cif_dir、probe_radius、n_samples）
  · 一个调用 → 复用唯一一个 sbatch 提交，绝不在登录节点 subprocess 跑 zeo++
  · 提交后立即返回 job_id/work_dir/output_csv，不阻塞回读结果，不浪费 token

Zeo++ 特征命令（已对照 zeo++0.3 network 帮助逐项核验）：
    network -ha -res -sa {r} {r} {n} -volpo {r} {r} {n} {cif}
输出文件 {stem}.res / .sa / .volpo 写入 cwd。

zeo++0.3 选项约束（防回归，勿改动）：
  · -sa / -volpo 必须各带 3 个参数: chan_radius probe_radius num_samples
    （仅给 1 个参数会报 'option accepts 3 or 4 arguments but 1 were supplied'）
  · -ha / -res 为裸选项（0 或 1 参数），-ha 合法（见 arguments.cc）
  · -s / -pld / -r(半径简写) 均非法——-s 会报 'Invalid option -s'；历史上
    曾因测试脚本误用 -s / -sa 单参数导致该工具作业 FAILED（runs/pore_fix_test）

批量模式：同一个 sbatch 脚本内 find|xargs -P 并行跑所有 CIF，
每个 CIF 独立容错（一个失败不影响其余），结果统一收集成 CSV。
"""

import os
import sys
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_SDK_ROOT = Path(__file__).resolve().parents[1]


def resolve_zeopp() -> dict:
    """Resolve the operator installation deterministically, never run a solver.

    Canonical runtime software settings win. Legacy zeopp.network_bin is read
    directly from the SDK config or at most three ancestor legacy layouts
    (identified by utils/config.py); Python's import search path is irrelevant.
    Explicit broken settings are errors, never a reason to pick another binary.
    """
    import hashlib
    import json
    import stat
    from bimem_agent import paths
    paths.load_runtime_config.cache_clear()
    runtime = paths.load_runtime_config()
    software = runtime.get('software', {})
    if not isinstance(software, dict): raise ValueError('runtime software must be an object')
    checked = []
    def read_configuration(path):
        path = Path(path).absolute()
        if not path.is_file(): return None
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1024*1024:
                raise ValueError('Zeo configuration must be a bounded regular JSON file')
            raw = stream.read(1024*1024+1)
            after = os.fstat(stream.fileno())
        key = lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if len(raw)!=before.st_size or key(before)!=key(after) or key(after)!=key(path.stat()):
            raise ValueError('Zeo configuration changed during resolution')
        checked.append({'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)})
        data=json.loads(raw)
        if not isinstance(data,dict):raise ValueError('Zeo configuration must contain a JSON object')
        return data
    def selected(value, source, base):
        if not isinstance(value,str) or not value.strip() or any(c in value for c in ('\n','\r','\0')):
            raise ValueError('configured Zeo executable must be a nonempty safe path')
        path=Path(value).expanduser()
        if not path.is_absolute():path=Path(base)/path
        path=path.resolve()
        if not path.is_file() or not os.access(path,os.X_OK):
            raise ValueError('configured Zeo network executable is missing or not executable; correct its configuration')
        return {'executable':str(path),'source':source,'configuration_files':checked}
    runtime_path=Path(paths.RUNTIME_CONFIG_PATH)
    if runtime_path.is_file():read_configuration(runtime_path)
    if 'zeopp_executable' in software:
        return selected(str(paths.get_config_path('software','zeopp_executable')),'runtime:software.zeopp_executable',paths.PROJECT_ROOT)
    embedded=runtime.get('zeopp')
    if embedded is not None and not isinstance(embedded,dict):raise ValueError('runtime zeopp must be an object')
    if isinstance(embedded,dict) and 'network_bin' in embedded:
        return selected(embedded['network_bin'],'runtime:zeopp.network_bin',paths.PROJECT_ROOT)
    candidates=[_SDK_ROOT/'config.json']
    for ancestor in list(_SDK_ROOT.parents)[:3]:
        if (ancestor/'utils/config.py').is_file():candidates.append(ancestor/'config.json')
    for cfg_path in dict.fromkeys(candidates):
        data=read_configuration(cfg_path)
        if data is None:continue
        section=data.get('zeopp')
        if section is not None and not isinstance(section,dict):raise ValueError('legacy zeopp configuration must be an object')
        if isinstance(section,dict)and'network_bin'in section:
            return selected(section['network_bin'],'legacy:zeopp.network_bin',cfg_path.parent)
    for candidate in (Path.home()/'zeo++-0.3/network',Path('/opt/zeoplusplus/network'),
                      Path('/usr/local/bin/network'),Path.home()/'zeoplusplus/network'):
        if candidate.is_file()and os.access(candidate,os.X_OK):
            return selected(str(candidate),'standard-installation-location',candidate.parent)
    return {'executable':None,'source':'unconfigured','configuration_files':checked}


def find_zeopp() -> Optional[str]:
    """Return the same resolved binary used by native installation inspection."""
    return resolve_zeopp()['executable']



@dataclass
class StructuralProperties:
    cif_name: str = ""
    di: Optional[float] = None          # Å 最大内切球直径 (LCD)
    df: Optional[float] = None          # Å 最大自由球直径 (PLD)
    dif: Optional[float] = None         # Å 沿自由路径最大内切球
    density: Optional[float] = None     # g/cm³
    vsa: Optional[float] = None         # m²/cm³ 体积比表面积
    gsa: Optional[float] = None         # m²/g 质量比表面积
    pore_volume: Optional[float] = None # cm³/g
    void_fraction: Optional[float] = None
    error: Optional[str] = None

    @property
    def lcd(self):
        return self.di

    @property
    def pld(self):
        return self.df

    @property
    def surface_area_m2g(self):
        return self.gsa

    @property
    def pore_volume_cm3g(self):
        return self.pore_volume


def submit_pore_analysis(
    cif_path: Optional[str] = None,
    cif_dir: Optional[str] = None,
    probe_radius: float = 1.525,
    n_samples: int = 5000,
    output_csv: Optional[str] = None,
    n_threads: int = 4,
    work_dir: Optional[str] = None,
    partition: str = "compute",
    walltime: str = "01:00:00",
) -> dict:
    """提交 Zeo++ 孔特征计算 —— 一个调用 = 一个 SLURM job。

    Parameters
    ----------
    cif_path / cif_dir : 必须二选一（且只能给一个）
    probe_radius       : 探针半径 Å（默认 1.525 对应 He；CO2 用 1.65）
    n_samples          : 蒙特卡洛采样数
    output_csv         : 结果 CSV 路径；None → {work_dir}/pore_results.csv
    n_threads          : 批内并行进程数（xargs -P）
    work_dir           : 作业工作目录；None → 会话 runs/{user}/{conv}/pore

    Returns
    -------
    dict: submitted, job_id, work_dir, output_csv, n_mofs, status, message
    """
    if bool(cif_path) == bool(cif_dir):
        return {"submitted": False, "failed": True,
                "error": "必须且只能提供 cif_path 与 cif_dir 中的一个"}

    if isinstance(n_threads, bool) or not isinstance(n_threads, int) or n_threads < 1:
        raise ValueError('n_threads must be a positive integer')

    try:
        bin_path = find_zeopp()
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'submitted':False,'failed':True,'error_kind':'zeopp_configuration',
                'error':str(error),'next_action':'Correct the existing Zeo executable configuration; no installation or scientific execution was attempted.'}
    if not bin_path:
        return {"submitted": False, "failed": True,
                "error_kind": "zeopp_configuration",
                "error": "未解析到 Zeo++ network 可执行路径；请检查 software.zeopp_executable 或已有 zeopp.network_bin 配置，当前未尝试安装或执行计算。"}

    if cif_dir:
        cifs = sorted(str(p) for p in Path(cif_dir).glob("*.cif"))
        if not cifs:
            return {"submitted": False, "failed": True,
                    "error": f"cif_dir 下没有 .cif 文件: {cif_dir}"}
        n_mofs = len(cifs)
    else:
        if not Path(cif_path).is_file():
            return {"submitted": False, "failed": True,
                    "error": f"cif_path 不存在: {cif_path}"}
        cifs = [cif_path]
        n_mofs = 1

    # Batch parallelism is a process budget, not solver-internal threads.
    sys.path.insert(0, str(_SDK_ROOT))
    from agents.watch_context import get_context
    allocation = get_context().get('resource_allocation') or {}
    cpus = min(n_threads, n_mofs)
    if allocation.get('resource_review_id'):
        cpus = allocation.get('cpus')
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            raise ValueError('resource_allocation.cpus must be a positive integer')
    workers = min(n_threads, n_mofs, cpus)

    # 工作目录：优先显式指定，否则落在会话 runs/{user}/{conv}/pore
    if work_dir is None:
        try:
            sys.path.insert(0, str(_SDK_ROOT))
            from agents.workspace import session_dir
            work_dir = str(session_dir("pore"))
        except Exception:
            work_dir = str(_SDK_ROOT / "gcmc_output" / "pore")
    work_dir = str(Path(work_dir).expanduser())
    out_dir = os.path.join(work_dir, "output")
    csv_path = output_csv or os.path.join(work_dir, "pore_results.csv")
    os.makedirs(out_dir, exist_ok=True)

    # 产物 CSV 收集用的自包含 Python 解析（计算节点无需 SDK import 路径）
    collect_py = _COLLECT_CSV_PY.replace("__OUT__", out_dir).replace("__CSV__", csv_path)

    run_block = "\n".join(f"run_one '{c}'" for c in cifs)
    batch_line = (
        ""
        if cif_path else
        f'find "{cif_dir}" -maxdepth 1 -name "*.cif" -print0 | '
        f'xargs -0 -P "$WORKERS" -I{{}} bash -c \'run_one "$@"\' _ {{}}'
    )

    command = "\n".join([
        "#!/bin/bash",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        f'WORKERS={workers}',
        f'ALLOCATED_CPUS="${{SLURM_CPUS_PER_TASK:-{cpus}}}"',
        '[[ "$ALLOCATED_CPUS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid SLURM CPU allocation" >&2; exit 2; }',
        'if (( WORKERS > ALLOCATED_CPUS )); then WORKERS="$ALLOCATED_CPUS"; fi',
        f'ZEO={shlex.quote(bin_path)}',
        f'PROBE="{probe_radius}"',
        f'NSAMP="{n_samples}"',
        f'OUT="{out_dir}"',
        f'WORK="{work_dir}"',
        "mkdir -p \"$OUT\"",
        "run_one() {",
        "  local cif=\"$1\"; local stem",
        "  stem=$(basename \"$cif\" .cif)",
        "  local d=\"$OUT/$stem\"",
        "  mkdir -p \"$d\"",
        "  cp \"$cif\" \"$d/${stem}.cif\"",
        # 特征命令：-ha -res -sa -volpo，产物写入 cwd（$d）
        "  ( cd \"$d\" && \"$ZEO\" -ha -res -sa \"$PROBE\" \"$PROBE\" \"$NSAMP\" \\",
        "        -volpo \"$PROBE\" \"$PROBE\" \"$NSAMP\" \"${stem}.cif\" > zeopp.log 2>&1 )",
        "  local rc=$?",
        # 独立容错：单个 CIF 失败不杀整个批
        "  if [ $rc -ne 0 ]; then",
        "    echo \"ERROR $stem rc=$rc\" | tee -a \"$WORK/errors.log\"",
        "    return 0",
        "  fi",
        "  echo \"OK $stem\"",
        "  return 0",
        "}",
        "export -f run_one",
        "export ZEO PROBE NSAMP OUT WORK",
        batch_line or run_block,
        "",
        f"CSV='{csv_path}' python3 - << 'PYEOF'",
        collect_py,
        "PYEOF",
        "",
        "echo 'Pore analysis completed'",
        # 完成标记：SLURM 会计禁用时 check_job_status 也能识别
        f'echo "$SLURM_JOB_ID" > "{work_dir}/job.done"',
    ])

    try:
        sys.path.insert(0, str(_SDK_ROOT))
        from agents.slurm import submit_and_return
    except Exception as e:
        return {"submitted": False, "failed": True,
                "error": f"无法加载 slurm 提交模块: {e}"}

    result = submit_and_return(
        job_name="pore_batch" if n_mofs > 1 else f"pore_{Path(cifs[0]).stem}",
        command=command,
        work_dir=work_dir,
        partition=partition,
        cpus_per_task=cpus,
        walltime=walltime,
    )
    if result.get("submitted"):
        result["status"] = "SUBMITTED"
        result["work_dir"] = work_dir
        result["output_csv"] = csv_path
        result["n_mofs"] = n_mofs
        result["message"] = (
            f"✅ Zeo++ {'批处理' if n_mofs > 1 else '单结构'}已提交 (job {result.get('job_id','?')})："
            f"{n_mofs} 个 CIF，探针 {probe_radius} Å，采样 {n_samples}。"
            f"完成用 check_job 查看，产物 CSV: {csv_path}"
        )
    return result


# 自包含的 CSV 收集脚本（解析 .res/.sa/.volpo）
_COLLECT_CSV_PY = r'''
import os, sys, glob

OUT, CSV = "__OUT__", os.environ.get("CSV", "__CSV__")

def parse_res(path):
    try:
        parts = open(path).readline().split()
        if len(parts) >= 4:
            return parts[1], parts[2], parts[3]
    except Exception:
        pass
    return ("", "", "")

def parse_sa(path):
    try:
        parts = open(path).readline().split()
        if len(parts) > 11:
            return parts[5], parts[9], parts[11]
    except Exception:
        pass
    return ("", "", "")

def parse_volpo(path):
    try:
        parts = open(path).readline().split()
        if len(parts) > 11:
            return parts[9], parts[11]
    except Exception:
        pass
    return ("", "")

def f(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return ""

rows = []
for res in sorted(glob.glob(os.path.join(OUT, "*", "*.res"))):
    stem = os.path.splitext(os.path.basename(res))[0]
    di, df, dif = parse_res(res)
    dens, vsa, gsa = parse_sa(os.path.join(OUT, stem, stem + ".sa"))
    vf, pv = parse_volpo(os.path.join(OUT, stem, stem + ".volpo"))
    rows.append(",".join([stem, f(di), f(df), f(dif), f(dens), f(vsa), f(gsa), f(pv), f(vf)]))

header = "cif_name,di_A,df_A,dif_A,density_g_cm3,vsa_m2_cm3,gsa_m2_g,pore_vol_cm3_g,void_fraction"
os.makedirs(os.path.dirname(CSV), exist_ok=True)
with open(CSV, "w") as fh:
    fh.write(header + "\n" + "\n".join(rows) + "\n" if rows else header + "\n")
print(f"Wrote {len(rows)} rows -> {CSV}")
'''


# ── 解析辅助（供外部/测试直接读取已算好的输出）──────────────────────────
def _parse_res(path: str, result: StructuralProperties) -> StructuralProperties:
    if not os.path.exists(path):
        return result
    with open(path) as f:
        parts = f.readline().split()
    if len(parts) >= 4:
        try:
            result.di, result.df, result.dif = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            pass
    return result


def _parse_sa(path: str, result: StructuralProperties) -> StructuralProperties:
    if not os.path.exists(path):
        return result
    with open(path) as f:
        first_line = f.readline()
    parts = first_line.split()
    try:
        if len(parts) > 11:
            result.density = float(parts[5])
            result.vsa = float(parts[9])
            result.gsa = float(parts[11])
    except (ValueError, IndexError):
        pass
    return result


def _parse_volpo(path: str, result: StructuralProperties) -> StructuralProperties:
    if not os.path.exists(path):
        return result
    with open(path) as f:
        first_line = f.readline()
    parts = first_line.split()
    try:
        if len(parts) > 11:
            result.void_fraction = float(parts[9])
            result.pore_volume = float(parts[11])
    except (ValueError, IndexError):
        pass
    return result


def read_csv(path: str) -> List[StructuralProperties]:
    """读取批量结果 CSV → StructuralProperties 列表（供报告/下游直接使用）"""
    out: List[StructuralProperties] = []
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        header = fh.readline()
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) < 9:
                continue
            r = StructuralProperties(cif_name=p[0])
            r.di = _num(p[1]); r.df = _num(p[2]); r.dif = _num(p[3])
            r.density = _num(p[4]); r.vsa = _num(p[5]); r.gsa = _num(p[6])
            r.pore_volume = _num(p[7]); r.void_fraction = _num(p[8])
            out.append(r)
    return out


def _num(s: str) -> Optional[float]:
    try:
        return float(s) if s else None
    except ValueError:
        return None


# 供脚本 run_pore_analysis.py 向后兼容（但真正计算走 SLURM）
run_single = submit_pore_analysis
run_batch = submit_pore_analysis
