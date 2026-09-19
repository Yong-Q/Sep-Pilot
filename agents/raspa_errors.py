"""RASPA-aware fault diagnosis.

Real GCMC failures are almost always software-specific parameter problems:
wrong force field, missing molecule definition, bad CIF, invalid input file.
This module maps the actual RASPA error messages (from run.log / stderr)
to concrete causes + fixes, so the agent can recover properly instead of
reporting a successful run on an empty output.

Sources:
  - RASPA2 source error/warning strings (grep of /home/user/RASPA2/src)
  - RASPA2 Docs/TroubleShooting/troubleshooting.tex
  - Real run.log evidence from /home/user/gcmc_agent/gcmc_output
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Catalog: (substring, cause, fixes, severity) ────────────────────
# Each entry is matched case-insensitively against the concatenation of
# run.log tail + stderr tail.

_CATALOG: List[Dict[str, Any]] = [
    {
        "patterns": ["returnpseudoatomnumber: error", "no force field parameters found",
                     "forcefieldparameter", "cannot find the force field", "missing atom type",
                     "forcefield parameter"],
        "cause": "力场参数缺失：RASPA 在激活的力场中找不到某个伪原子的参数",
        "detail": ("最常见的 RASPA GCMC 失败原因。CIF 里的原子类型或客体分子定义引用了 "
                   "当前力场中没有的伪原子。典型：PACMAN 电荷输出会生成带后缀的原子 "
                   "(如 O_co2/C_co2), 而所用力场 (UFF/GenericMOFs) 的 pseudo_atoms.def 里没有。"),
        "fixes": [
            "检查 run.log 中 'ReturnPseudoAtomNumber: Error!!!! :<atom>' 提示的具体原子",
            "若原子来自 CIF → 用 _normalize_cif_for_raspa 清理后缀，或换一个包含该原子类型的力场 (GenericMOFs 通常最全)",
            "若原子来自客体分子 → 切换 MoleculeDefinition / Forcefield 组合，确保二者匹配",
            "查看 force_field_mixing_rules.def 是否包含该原子",
        ],
    },
    {
        "patterns": ["cannot open ", "no such file or directory", "unable to open",
                     "file not found", "does not exist"],
        "cause": "RASPA 找不到必需的文件（分子定义 / 力场 / CIF）",
        "detail": ("例如 'Cannot open .../molecules/TraPPE/CH4.def. Error: No such file or directory' "
                   "表示当前 RASPA 安装缺少 TraPPE 的 CH4 分子定义。"),
        "fixes": [
            "确认缺少的文件路径（.def 分子文件 或 力场文件）",
            "若缺少分子定义 → 换用已安装的分子（如 CO2/CO/N2/H2O），或安装对应 .def 到 molecules/<Forcefield>/ 目录",
            "若缺少力场 → 换用已安装力场 (UFF, GenericMOFs, TraPPEForceField...)",
        ],
    },
    {
        # NOTE: RASPA prints the CIF preamble ("P 1 found space group: 1",
        # "End reading cif-file", "_cell_length_a: ...") on EVERY successful
        # run, and this RASPA build only writes that preamble into run.log
        # (stdout fully buffered; full log goes to Output/System_0/*.data).
        # Patterns must therefore REQUIRE an explicit error token — broad
        # substrings like "space group"/"cell length"/"cif-file" matched the
        # benign preamble and produced endless fake "CIF parse failure"
        # diagnoses (job COMPLETED with valid data still flagged FAILED).
        "patterns": ["error in cif-file", "error in cif file", "error reading cif",
                     "error while reading cif", "cannot read cif", "unable to read cif",
                     "could not read cif", "failed to read cif", "error: unknown space group",
                     "error: space group", "invalid space group", "cannot determine space group",
                     "cannot determine cell", "invalid cell length", "invalid cell angle"],
        "cause": "CIF 文件解析失败或框架定义有误",
        "detail": ("RASPA 读 CIF 失败：空间群定义缺失、晶胞参数无法识别、或对称操作错误。"
                   "PACMAN 生成的 CIF 常用 _space_group_name_H-M_alt 而非 RASPA 期望的 _symmetry_space_group_name_H-M。"),
        "fixes": [
            "用 _normalize_cif_for_raspa 修正 CIF 的 space-group 关键字 (H-M_alt → H-M)",
            "确认晶胞参数 (a,b,c,alpha,beta,gamma) 完整且为正",
            "确认 UnitCells 乘积在内存允许范围内 (避免过大的 3x3x3)",
        ],
    },
    {
        "patterns": ["no space", "cannot fit", "more molecules than", "create number of molecules",
                     "system is full", "no free", "insertion failed", "overlap"],
        "cause": "体系过拥挤：尝试插入的分子数超过框架可容纳量",
        "detail": ("RASPA 文档明确建议：设置 CreateNumberOfMolecules 0 验证；若成功则说明初始分子数太多。"
                   "也可能是框架原子重叠 (UnitCells 太小 或 CIF 有重叠原子)。"),
        "fixes": [
            "设 CreateNumberOfMolecules 0 并重跑验证",
            "减小初始分子数 / 增大 UnitCells",
            "检查 CIF 是否有重叠原子",
        ],
    },
    {
        "patterns": ["energy drift", "internal consistency error", "energy.*different",
                     "current energy.*true energy", "simulation results are wrong"],
        "cause": "数值不稳定：模拟结果不可信（能量漂移 / 内部一致性错误）",
        "detail": ("RASPA 检测到能量不一致，结果作废。常见于截断半径 < 盒长一半、"
                   "电荷处理 (Ewald) 参数不当、或力场兼容问题。"),
        "fixes": [
            "检查 CutOffVDW 是否 < 盒长一半，必要时调大",
            "调整 EwaldPrecision 或改用不同 ChargeMethod",
            "换更稳的力场重新验证",
        ],
    },
    {
        "patterns": ["cutoff smaller than half", "cutoff", "boxlength"],
        "cause": "截断半径小于半个盒长",
        "detail": "RASPA 报 ERROR: Cutoff smaller than half of one of the perpendicular boxlengths。",
        "fixes": [
            "增大 CutOffVDW 或减小 UnitCells（使盒长变小则截断变大，矛盾时需折中）",
            "检查晶胞是否被错误放大",
        ],
    },
    {
        "patterns": ["lowenstein", "net charge", "charged ions", "reinsertion move used on charged ions"],
        "cause": "框架带电 / 电荷处理警告",
        "detail": ("框架有净电荷或 Lowenstein 规则不满足。带电离子的重插入移动可能导致数值问题。"),
        "fixes": [
            "检查 CIF 电荷是否平衡（PACMAN 输出的框架应整体电中性）",
            "对带电离子改用 Random Translation 移动",
        ],
    },
    {
        "patterns": ["omitted vdw interactions", "atom-pairs with no vdw", "no vdw interaction",
                     "no parameters", "vdw.*omit"],
        "cause": "部分原子对缺失 VDW 相互作用参数",
        "detail": "RASPA 警告有原子对没有 VDW 参数（可能被忽略或数值为 0）。",
        "fixes": [
            "确认力场覆盖所有原子类型",
            "补充 force_field_mixing_rules.def 中的混合规则",
        ],
    },
    {
        "patterns": ["segmentat", "core dumped", "segfault", "abort", "bus error", "sigsegv", "sigabrt"],
        "cause": "RASPA C++ 程序崩溃（段错误等）",
        "detail": ("通常是编译/环境问题（依赖库、编译器版本）或极端输入触发未捕获错误。"),
        "fixes": [
            "检查 stderr 中崩溃前的最后几行错误信息",
            "减少单元数/分子数，换更简单的体系验证是否复现",
            "确认 RASPA 可执行文件与本机库兼容 (LD_LIBRARY_PATH)",
        ],
    },
    {
        "patterns": ["mkl", "mkl service", "cannot load lib", "undefined symbol", "error while loading shared libraries"],
        "cause": "动态库 / MKL 链接问题",
        "detail": "RASPA 运行加载动态库失败（MKL 版本不匹配、LD_LIBRARY_PATH 缺库）。",
        "fixes": [
            "检查 LD_LIBRARY_PATH 是否指向 RASPA2 的 lib",
            "确认 MKL 版本与 RASPA 编译版本兼容",
        ],
    },
    {
        "patterns": ["out of memory", "killed", "cannot allocate", "bad_alloc", "oom"],
        "cause": "内存不足，作业被系统或 OOM killer 终止",
        "detail": "体系太大（UnitCells 过大 / 分子数过多 / 网格过密）。",
        "fixes": [
            "减小 UnitCells（如 2x2x2 → 1x1x1）",
            "减少分子数 / 简化体系",
            "申请更多内存的节点 (himem partition)",
        ],
    },
    {
        "patterns": ["timed out", "time limit", "maximum run time", "dndl", "cancelled", "cancel"],
        "cause": "作业超时或被取消 (SLURM 层)",
        "detail": ("作业超过 walltime 被取消 (TIMEOUT/CANCELLED)，或队列策略取消。"
                   "循环 10 个压力点 × 50000 cycles 常超 4h。"),
        "fixes": [
            "减少 NumberOfCycles (如 50000 → 20000)",
            "减少压力点数 / 缩短 walltime 需求与计算量匹配",
            "对每个压力点独立提交，避免一个长作业",
        ],
    },
    {
        "patterns": ["no convergence", "did not converge", "not converge", "nan", "inf", "nan in energy"],
        "cause": "计算不收敛 / 出现 NaN",
        "detail": "模拟发散：初始构型不合理、力场极端、或电荷处理不稳定。",
        "fixes": [
            "检查初始分子放置（避免重叠）",
            "调整 Ewald 精度 / 力场参数",
            "减小插入概率或增大初始化周期",
        ],
    },
]

# Substring → catalog index (precompiled for speed)
_MATCHERS: List[Dict[str, Any]] = []

def _compile():
    global _MATCHERS
    for entry in _CATALOG:
        _MATCHERS.append({
            "patterns": [p.lower() for p in entry["patterns"]],
            "entry": entry,
        })

_compile()


def match_raspa_error(text: str) -> Optional[Dict[str, Any]]:
    """Match a run.log/stderr text against the RASPA error catalog.

    Returns the best-matching catalog entry (with cause + fixes) or None.
    """
    if not text:
        return None
    low = text.lower()
    for m in _MATCHERS:
        for p in m["patterns"]:
            if p in low:
                return m["entry"]
    return None


def summarize_run_log(log_tail: str) -> Dict[str, Any]:
    """Produce a RASPA-aware summary of a run.log tail.

    Returns {raspa_error: bool, matched_pattern, cause, fixes, critical_lines}
    """
    critical = []
    for line in (log_tail or "").splitlines()[-30:]:
        ll = line.lower()
        if any(k in ll for k in ("error", "fatal", "warning", "cannot", "unable", "returnpseudoatom", "segmentation")):
            critical.append(line.strip()[:160])
    entry = match_raspa_error(log_tail)
    return {
        "raspa_error": entry is not None,
        "matched_pattern": entry["patterns"][0] if entry else None,
        "cause": entry["cause"] if entry else None,
        "fixes": entry["fixes"] if entry else [],
        "critical_lines": critical[-6:],
    }
