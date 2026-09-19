"""Durable, deterministic goal contract for a conversation.

Chat history is useful evidence, but it is not a safe source of truth for a
long-running scientific workflow.  This module extracts the small set of
constraints that must survive context trimming and agent handoffs, and guards
compute/delegation calls against unapproved goal drift.
"""
from __future__ import annotations
import json

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_GAS_ALIASES = {
    "co2": "CO2", "co₂": "CO2", "二氧化碳": "CO2",
    "ch4": "CH4", "ch₄": "CH4", "甲烷": "CH4",
    "n2": "N2", "n₂": "N2", "氮气": "N2",
    "h2": "H2", "h₂": "H2", "氢气": "H2",
    "o2": "O2", "o₂": "O2", "氧气": "O2",
    "h2o": "H2O", "h₂o": "H2O", "水蒸气": "H2O",
    "so2": "SO2", "so₂": "SO2", "二氧化硫": "SO2",
    "h2s": "H2S", "h₂s": "H2S", "硫化氢": "H2S",
    "nh3": "NH3", "nh₃": "NH3", "氨气": "NH3",
    "kr": "Kr", "氪": "Kr",
    "xe": "Xe", "氙": "Xe",
    "ar": "Ar", "氩": "Ar",
    "he": "He", "氦": "He",
    "co": "CO", "一氧化碳": "CO",
    "c2h4": "C2H4", "c₂h₄": "C2H4", "乙烯": "C2H4",
    "c2h6": "C2H6", "c₂h₆": "C2H6", "乙烷": "C2H6",
}

_METHOD_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("CDFT", ("cdft", "经典密度泛函")),
    ("GCMC", ("gcmc", "grand canonical", "巨正则", "蒙特卡洛")),
    ("HENRY", ("run_henry", "henry", "亨利系数")),
    ("VASP", ("vasp",)),
    ("DFT", ("dft", "量子化学", "电子结构")),
    ("MD", ("molecular dynamics", "分子动力学", " md ")),
    ("TST", ("string_tst", "tst", "过渡态")),
]

_TOOL_METHOD = {
    "submit_job": "",
    "generate_structure": "",
    "run_pacman_charge": "",
    "run_cdft": "CDFT",
    "run_gcmc_isotherm": "GCMC",
    "run_gcmc_batch": "GCMC",
    "run_isotherm_chain": "GCMC",
    "run_henry": "HENRY",
    "run_henry_chain": "HENRY",
    "run_vasp": "VASP",
    "run_md_optimize": "MD",
    "run_string_tst": "TST",
}

_COMPUTE_WORDS = (
    "计算", "模拟", "提交", "筛选", "吸附", "henry", "gcmc", "cdft",
    "run_", "simulate", "screen", "calculate",
)


def is_readonly_request(text: str) -> bool:
    """Explicit investigation-only scope overrides incidental compute nouns."""
    prohibited = bool(re.search(
        r'只查|仅查|只调查|仅调查|不(?:要)?(?:提交计算|提交作业|开算|执行计算)|不开算|read[- ]only|do not (?:run|submit)',
        text, re.IGNORECASE))
    preparing = bool(re.search(r'(?:生成|准备|写出|构建).{0,8}(?:输入|脚本|文件|结构|参数)|prepare|generate', text, re.IGNORECASE))
    return prohibited and not preparing


def prohibits_submission(text: str) -> bool:
    return bool(re.search(r'不(?:要)?(?:提交计算|提交作业|开算|执行计算)|不开算|do not (?:run|submit)', text, re.IGNORECASE))


def scientific_constraints(text: str) -> Dict[str, Any]:
    """Pin explicit scientific choices, without choosing a missing variant."""
    from pathlib import Path
    import json
    out = {}
    catalog = Path(__file__).resolve().parents[1] / 'forcefields/towhee/catalog.json'
    names = set()
    if catalog.exists():
        entries = json.loads(catalog.read_text())['entries']
        names = {e['name'] for e in entries} | {e.get('family', e['name']) for e in entries}
    selected = []
    for clause in re.split(r'[，,。；;\n]', text):
        if re.search(r'不要|不用|不能|不应|not\s', clause, re.I):
            continue
        for name in sorted(names, key=len, reverse=True):
            if re.search(r'(?<![a-z0-9])' + re.escape(name) + r'(?![a-z0-9_-])', clause, re.I):
                if not any(name.casefold() in other.casefold() for other in selected):
                    selected.append(name)
    if len(selected) == 1:
        out['force_field'] = selected[0]
    if selected:
        out['declared_force_fields'] = selected
    for scope, marker in (('framework', '框架'), ('guest', '客体')):
        for clause in re.split(r'[，,。；;\n]', text):
            if marker not in clause or re.search(r'不要|不用|不能|不应', clause):
                continue
            scoped = [name for name in selected if re.search(re.escape(name), clause, re.I)]
            if len(scoped) == 1:
                out[scope + '_force_field'] = scoped[0]
    source = re.search(r'(?:保留|沿用|使用|采用).{0,12}(?:原|已有|现有)?\s*CIF.{0,8}电荷', text, re.I)
    if source:
        out['framework_charge_source'] = 'CIF'
    elif re.search(r'(?:电荷|charge).{0,12}(?:PACMAN|PACMOF|DDEC6|REPEAT|CM5)|(?:PACMAN|PACMOF|DDEC6|REPEAT|CM5).{0,12}(?:电荷|charge)', text, re.I):
        out['framework_charge_source'] = re.search(r'PACMAN|PACMOF|DDEC6|REPEAT|CM5', text, re.I).group().upper()
    expected = re.search(r'(?:晶胞|cell).{0,12}(?:净电荷|net.?charge)\s*[:=：为]?\s*([+-]?\d+(?:\.\d+)?)', text, re.I)
    if expected:
        out['expected_cell_net_charge'] = float(expected.group(1))
    provenance = re.search(r'(?:参数来源|forcefield.source)\s*[:=：为]\s*([^\n，,。；;]+)', text, re.I)
    if provenance:
        out['forcefield_source'] = provenance.group(1).strip()
    return out

_DIRECT_EXECUTE = (
    "直接开始执行", "直接开始", "直接执行", "立即执行", "开始执行",
    "不用确认", "无需确认", "按此执行", "execute now", "start now",
)


def explicitly_authorizes_execution(text: str) -> bool:
    """Recognize direct authority expressed as a natural autonomous mandate.

    This is deliberately about authority, not scientific routing: methods and
    protected conditions remain typed elsewhere in the goal contract.
    """
    low = str(text or "").lower()
    if any(phrase in low for phrase in _DIRECT_EXECUTE):
        return True
    return bool(re.search(
        r'(?:自主|自动|agent自己).{0,12}(?:执行|推进|完成|修补)|'
        r'(?:完整|全程).{0,10}(?:推动|推进|执行).{0,12}(?:结果|报告|完成)',
        low,
    ))

_APPROVAL_WORDS = (
    "确认执行", "确认，执行", "确认，跑吧", "确认开始", "同意方案",
    "按方案执行", "可以开始", "approved", "approve", "go ahead",
)

_CHANGE_WORDS = (
    "改用", "改成", "改算", "换成", "切换", "替换", "不要", "停止", "取消",
    "加上", "再算", "另外", "方案a", "方案b", "switch", "replace", "instead",
)


def extract_gases(text: str) -> Set[str]:
    """Return canonical gas names found in free text without CO/CO2 overlap."""
    raw_text = str(text or "")
    # Metal-prefixed framework names are material identities, not gas formulae.
    # In particular, ``Co-MOF-74`` used to contribute a spurious ``CO`` gas
    # because the hyphen is a token boundary after case-folding.
    gas_text = re.sub(
        r"\b(?:Ni|Mg|Co|Cu|Fe|Zn|Mn|Cr|Al|Ti|Zr)-?MOF-?\d+\b",
        " ", raw_text, flags=re.IGNORECASE,
    )
    low = gas_text.lower()
    found: Set[str] = set()
    # Token-like formulas first.  Boundaries avoid matching CO inside CO2.
    for raw, canonical in _GAS_ALIASES.items():
        if raw.isascii() and raw.replace(" ", "").isalnum():
            if re.search(rf"(?<![a-z0-9]){re.escape(raw)}(?![a-z0-9])", low):
                found.add(canonical)
        elif raw in low:
            found.add(canonical)
    return found


def extract_negated_gases(text: str) -> Set[str]:
    """Gases explicitly rejected/cancelled by the user."""
    negated: Set[str] = set()
    clauses = re.split(r"[，,。；;\n]|\bbut\b|\bnot\b", str(text or ""), flags=re.IGNORECASE)
    for clause in clauses:
        low = clause.lower()
        if any(k in low for k in (
            "不是", "不要", "停止", "取消", "排除", "禁用", "无关", "跑偏",
            "stop", "cancel", "exclude", "instead of",
        )):
            negated |= extract_gases(clause)
    return negated


def extract_method(text: str) -> str:
    # Filesystem provenance is not a scientific method declaration.
    # Compatibility parser only; real execution uses typed tool/node contracts.
    raw=str(text or '')
    raw=re.sub(r'(?:\./|\.\./)[^\s\"\'<>]+|(?<![A-Za-z0-9])/(?:[^/\s\"\'<>]+/)+[^\s\"\'<>]*',' ',raw)
    low = f" {raw.lower()} "
    # cDFT is the computational method in phrases such as "用cDFT计算Henry系数";
    # Henry there is the requested observable, not permission to switch to RASPA.
    if "cdft" in low or "经典密度泛函" in low:
        return "CDFT"
    if "run_henry" in low or "henry" in low or "亨利系数" in low:
        return "HENRY"
    for method, patterns in _METHOD_PATTERNS:
        if any(p in low for p in patterns):
            return method
    return ""


def extract_parameters(text: str) -> Dict[str, Any]:
    """Extract durable scientific constraints; never invent omitted values."""
    raw = str(text or "")
    low = raw.lower()
    out: Dict[str, Any] = {}
    temps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*[kK](?![a-zA-Z])", raw)]
    if temps:
        out["temperature_K"] = temps[0] if len(temps) == 1 else temps
    pressure = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|~|–|到)\s*(\d+(?:\.\d+)?)\s*(bar|kpa|mpa|pa|atm)\b",
        low,
    )
    if pressure:
        out["pressure_range"] = [float(pressure.group(1)), float(pressure.group(2))]
        out["pressure_unit"] = pressure.group(3)
    n_structures = re.search(r"(?:生成|构建|准备|共)\s*(\d+)\s*个", raw)
    if n_structures:
        out["n_structures"] = int(n_structures.group(1))
    max_atoms = re.search(r"原子数(?:限制|上限)?\s*(\d+)\s*(?:以内|以下|上限)?", raw)
    if max_atoms:
        out["max_atoms"] = int(max_atoms.group(1))
    topology = re.search(r"拓扑(?:用|为|包括|[:：])\s*([^\n。；;]+)", raw, re.IGNORECASE)
    if topology:
        vals = [x.strip() for x in re.split(r"[,，/\s]+", topology.group(1)) if x.strip()]
        # Stop when prose resumes.
        out["topologies"] = vals[:30]
    material = re.search(
        r"(u-?HOF|HOF|(?:Ni|Mg|Co|Cu|Fe|Zn|Mn|Cr|Al|Ti|Zr)-?MOF-?\d+|"
        r"MOF-?\d+|COF-?\d+|Cu-?BTC|CuBTC|HKUST-?1|MFI|ZSM-?5|"
        r"ZIF-?\d+|UiO-?\d+|MIL-?\d+)",
        raw,
        re.IGNORECASE,
    )
    if material:
        out["material"] = material.group(1)
    return out


def _is_automated_message(text: str) -> bool:
    s = str(text or "").lstrip()
    return s.startswith((
        '[生命周期事件]',
        "[系统", "[JobWatch]", "✅ [作业完成提醒]", "⚠️ [作业故障提醒]",
        "⏳ [作业", "[TaskLine]", "[链条·",
    ))


@dataclass
class GoalContract:
    """The authoritative, versioned objective for one conversation."""

    schema_version: int = 4
    original_goal: str = ""
    active_goal: str = ""
    version: int = 0
    gases: List[str] = field(default_factory=list)
    method: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    directives: List[str] = field(default_factory=list)
    requires_plan_approval: bool = False
    execution_authorized: bool = False
    execution_mode: str = "workflow"
    approved_plan_version: int = 0
    pending_plan_version: int = 0
    drift_blocks: List[Dict[str, Any]] = field(default_factory=list)
    approved_nodes: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_user_message(cls, message: str) -> "GoalContract":
        text = str(message or "").strip()
        low = text.lower()
        phases = sum(bool(any(k in low for k in group)) for group in (
            ("生成", "构建", "准备", "generate"),
            ("电荷", "pacman", "pacmof", "charge"),
            ("计算", "模拟", "gcmc", "cdft", "henry", "vasp"),
            ("分析", "排序", "选择性", "报告", "analy", "report"),
        ))
        readonly = is_readonly_request(text)
        prepare_only = prohibits_submission(text) and not readonly
        requires_approval = not (readonly or prepare_only) and (phases >= 2 or any(
            k in low for k in ("完整流程", "工作流", "依次", "然后", "多步骤", "pipeline")
        ))
        direct = explicitly_authorizes_execution(text)
        return cls(
            original_goal=text,
            active_goal=text,
            version=1 if text else 0,
            gases=sorted(extract_gases(text) - extract_negated_gases(text)),
            method=extract_method(text),
            parameters={**extract_parameters(text), **scientific_constraints(text)},
            directives=[text] if text else [],
            requires_plan_approval=requires_approval,
            execution_authorized=not (readonly or prepare_only) and (direct or not requires_approval),
            execution_mode='read_only' if readonly else 'prepare_only' if prepare_only else 'workflow',
        )

    def apply_user_message(self, message: str) -> bool:
        """Merge a genuine user directive. Returns True when the contract changed."""
        text = str(message or "").strip()
        if not text or _is_automated_message(text):
            return False

        low = text.lower()
        is_approval = any(k in low for k in _APPROVAL_WORDS)
        is_direct = explicitly_authorizes_execution(text)
        explicit_change = any(k in low for k in _CHANGE_WORDS)
        explicit_task = any(k in low for k in (
            "计算", "模拟", "筛选", "分析", "生成", "构建", "查找", "调研",
            "calculate", "simulate", "screen", "analyze", "generate",
        ))
        phases = sum(bool(any(k in low for k in group)) for group in (
            ("生成", "构建", "准备", "generate"),
            ("电荷", "pacman", "pacmof", "charge"),
            ("计算", "模拟", "gcmc", "cdft", "henry", "vasp"),
            ("分析", "排序", "选择性", "报告", "analy", "report"),
        ))
        multi_step = phases >= 2 or any(
            k in low for k in ("完整流程", "工作流", "依次", "然后", "多步骤", "pipeline")
        )
        negated_gases = extract_negated_gases(text)
        new_gases = extract_gases(text) - negated_gases
        new_method = extract_method(text)
        new_parameters = {**extract_parameters(text), **scientific_constraints(text)}
        changed = False

        if prohibits_submission(text) or is_readonly_request(text):
            changed = self.execution_mode != 'read_only' or self.execution_authorized
            self.execution_mode = 'read_only' if is_readonly_request(text) else 'prepare_only'
            self.execution_authorized = False
            self.requires_plan_approval = False
        elif is_approval or is_direct:
            if not self.execution_authorized:
                changed = True
            if self.execution_mode in {'read_only', 'prepare_only'}:
                self.execution_mode = 'workflow'
                self.requires_plan_approval = True
            self.execution_authorized = True

        # A greeting or short first turn may later become a multi-stage task.
        # Re-evaluate this durable property on every real user message.
        if (self.execution_mode == 'workflow' and multi_step
                and not self.approved_nodes and not self.requires_plan_approval):
            self.requires_plan_approval = True
            changed = True

        # A concrete method selection ("用 cDFT" / "方案A用 run_henry_chain") is an
        # authoritative update. Merely discussing a method is kept as a directive
        # but does not silently replace an existing method.
        method_selection = bool(new_method) and (
            not self.method or explicit_change
            or bool(re.search(r"(?:用|使用|选|指定|run_).{0,24}(?:cdft|gcmc|henry|dft|md|tst)", low))
        )
        if method_selection and new_method != self.method:
            self.method = new_method
            changed = True

        if new_gases:
            current = set(self.gases)
            if not current:
                current = new_gases
            elif explicit_change or explicit_task:
                # "加上/另外" extends; "改用/换成/不要" replaces the active set.
                if any(k in low for k in ("加上", "另外", "再算")):
                    current |= new_gases
                else:
                    current = new_gases
            elif new_gases.issubset(current):
                pass
            # Without an explicit change verb, do not let incidental comparison
            # gases silently mutate the protected objective.
            new_list = sorted(current)
            if new_list != self.gases:
                self.gases = new_list
                changed = True

        if negated_gases:
            filtered = sorted(set(self.gases) - negated_gases)
            if filtered != self.gases:
                self.gases = filtered
                changed = True

        if new_parameters:
            for key, value in new_parameters.items():
                if self.parameters.get(key) != value:
                    self.parameters[key] = value
                    changed = True

        # Keep meaningful directives, but confirmation-only replies do not replace
        # the active scientific goal.
        if not is_approval or len(text) > 30:
            self.directives.append(text)
            self.directives = self.directives[-24:]
        if explicit_change or (explicit_task and (new_gases or new_method)):
            self.active_goal = text
            changed = True

        if changed:
            self.version = max(1, self.version + 1)
        return changed

    def propose_plan(self) -> int:
        self.pending_plan_version = max(
            self.pending_plan_version + 1,
            self.approved_plan_version + 1,
            1,
        )
        return self.pending_plan_version

    def approve_pending_plan(self) -> int:
        if self.pending_plan_version:
            self.approved_plan_version = self.pending_plan_version
        self.execution_authorized = True
        return self.approved_plan_version

    def guard_tool_call(self, tool_name: str, params: Optional[Dict[str, Any]] = None,
                        enforce_approved_node: bool = True) -> Tuple[bool, str]:
        """Reject compute/delegation calls that contradict protected constraints."""
        params = params or {}
        name = str(tool_name or "")
        blob = " ".join(str(v) for v in params.values())
        is_handoff = name.startswith("handoff_to_")
        delegated_node={}
        if is_handoff:
            try:
                context=params.get('context','')
                structured=json.loads(context) if isinstance(context,str) else context
                if isinstance(structured,dict):delegated_node=structured.get('node') or {}
            except (ValueError,TypeError):pass
            # A natural-language delegation is not a scientific tool call.
            # Main/expert models understand it; execution is checked when an
            # actual tool (or structured node contract) is chosen. In particular
            # paths such as gcmc_agent cannot imply a GCMC method switch.
            if not delegated_node.get('tool'): return True, ''
            delegated_args=delegated_node.get('arguments',{})
            blob=str(delegated_args)
        is_compute_handoff = is_handoff and delegated_node.get('tool') in _TOOL_METHOD
        is_compute = name in _TOOL_METHOD or is_compute_handoff
        from .recovery import is_submission
        is_compute = is_compute or is_submission(name, params)

        if name == "run_bash":
            cmd = str(params.get("command") or params.get("cmd") or "").lower()
            is_compute = is_compute or bool(re.search(r'\b(?:qsub|sbatch|srun|mpirun|simulate)\b', cmd))
        if name == 'validate_framework_charges':
            expected_charge = self.parameters.get('expected_cell_net_charge')
            if expected_charge is not None and params.get('expected_net_charge') != expected_charge:
                return self._block(name, '不能通过改变预期晶胞净电荷来掩盖电荷错误', params)
        if not is_compute:
            return True, ""

        if self.execution_mode in {'read_only', 'prepare_only'}:
            return self._block(name, '用户当前仅授权参数调查/只读查证，未授权计算或提交', params)

        if self.requires_plan_approval and not self.execution_authorized:
            return self._block(
                name,
                "多步骤方案尚未获得用户确认",
                params,
            )

        expected_ff = self.parameters.get('framework_force_field') or self.parameters.get('force_field')
        if expected_ff and name == 'run_md_optimize':
            actual_ff = params.get('force_field')
            if actual_ff is None or str(actual_ff).casefold() != expected_ff.casefold():
                return self._block(name, '必须显式传递用户指定力场；不能省略后走默认力场或静默替换', params)
        expected_provenance = self.parameters.get('forcefield_source')
        if expected_provenance and is_submission(name, params):
            provenance = params.get('forcefield_source') or params.get('forcefield_dir')
            if not provenance or str(provenance).casefold() != expected_provenance.casefold():
                return self._block(name, '参数来源未与用户指定来源一致验证；不能默默换原生库或未实现的转换', params)
        expected_source = self.parameters.get('framework_charge_source')
        if name == 'run_pacman_charge' and expected_source == 'CIF':
            return self._block(name, '用户要求保留已有CIF电荷，不能重新赋电荷', params)
        if name == 'run_pacman_charge' and expected_source in {'PACMAN','PACMOF'}:
            expected_method = 'pacman' if expected_source == 'PACMAN' else 'pacmof'
            if params.get('method') != expected_method:
                return self._block(name, '必须显式传入用户指定的PACMAN/PACMOF路线，不能走默认模型替代', params)
        if name == 'run_pacman_charge' and expected_source in {'DDEC6', 'CM5', 'REPEAT'}:
            actual_source = str(params.get('charge_type', '')).upper()
            if actual_source != expected_source or params.get('method') != 'pacman':
                return self._block(name, '电荷模型必须与用户明确指定的路线一致', params)
        from .recovery import is_submission
        exact_nodes = [n for n in self.approved_nodes if n.get('tool') == name]
        if enforce_approved_node and self.approved_nodes and is_submission(name, params):
            if any(n.get('arguments') == params for n in exact_nodes):
                return True, ''
            return self._block(name, '调用参数必须与用户批准的结构化节点一致；改变参数需编排补丁', params)

        science_args = delegated_node.get('arguments',{}) if is_handoff else params
        candidate_gases = extract_gases(' '.join(str(science_args.get(key,'')) for key in ('gas','gases')))
        protected_gases = set(self.gases)
        unauthorized = candidate_gases - protected_gases if protected_gases else set()
        if unauthorized:
            return self._block(
                name,
                f"未经用户批准的气体 {sorted(unauthorized)}；当前目标只允许 {sorted(protected_gases)}",
                params,
            )

        expected_method = self.method
        # Handoff background contains other tools/recipes, not a new method.
        # Concrete prerequisite/control tools have an intentionally empty method.
        candidate_method = _TOOL_METHOD.get(name,'')
        if delegated_node.get('tool') in _TOOL_METHOD:candidate_method=_TOOL_METHOD[delegated_node['tool']]
        if expected_method == 'MULTI':
            allowed_methods = {_TOOL_METHOD.get(n.get('tool')) for n in self.approved_nodes}
            candidate = candidate_method
            if candidate and candidate not in allowed_methods:
                return self._block(name, '计算方法不在用户批准的多方法节点中', params)
            expected_method = ''
        if expected_method and candidate_method:
            compatible = expected_method == candidate_method
            # Henry is commonly implemented by GCMC Widom insertion; allow the
            # pair only when the user selected Henry/GCMC, never when cDFT is pinned.
            if {expected_method, candidate_method} <= {"GCMC", "HENRY"}:
                compatible = True
            if not compatible:
                return self._block(
                    name,
                    f"未经用户批准的方法切换 {expected_method} → {candidate_method}",
                    params,
                )

        protected_temp = self.parameters.get("temperature_K")
        candidate_temp = params.get("temperature")
        if protected_temp is not None and candidate_temp is not None:
            allowed_temps = protected_temp if isinstance(protected_temp, list) else [protected_temp]
            try:
                if not any(abs(float(candidate_temp) - float(t)) < 1e-9 for t in allowed_temps):
                    return self._block(
                        name,
                        f"未经用户批准的温度变更 {allowed_temps} K → {candidate_temp} K",
                        params,
                    )
            except Exception:
                return self._block(name, "温度参数无法与用户约束核对", params)

        if name == "generate_structure":
            for contract_key, tool_key in (("n_structures", "n_structures"), ("max_atoms", "max_atoms")):
                expected = self.parameters.get(contract_key)
                actual = params.get(tool_key)
                if expected is not None and actual is not None and int(expected) != int(actual):
                    return self._block(
                        name,
                        f"未经用户批准的 {tool_key} 变更 {expected} → {actual}",
                        params,
                    )
            expected_topologies = self.parameters.get("topologies") or []
            actual_topologies = params.get("topologies") or []
            if expected_topologies and set(map(str, actual_topologies)) != set(map(str, expected_topologies)):
                return self._block(
                    name,
                    f"拓扑必须与用户约束完全一致: {expected_topologies}",
                    params,
                )
        return True, ""

    def _block(self, tool: str, reason: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        record = {
            "tool": tool,
            "reason": reason,
            "params": str(params)[:500],
            "goal_version": self.version,
        }
        self.drift_blocks.append(record)
        self.drift_blocks = self.drift_blocks[-20:]
        return False, (
            f"[GOAL_CONTRACT_BLOCK] {reason}。工具 {tool} 未执行。"
            "如果确实需要更改气体、方法或研究范围，必须先向用户展示变更并获得明确确认。"
        )

    def prompt_block(self) -> str:
        directives = "\n".join(f"  - {d[:500]}" for d in self.directives[-8:]) or "  - (无)"
        return (
            "[GOAL CONTRACT · 运行时硬约束]\n"
            f"目标版本: v{self.version}\n"
            f"原始目标: {self.original_goal[:1200]}\n"
            f"当前活动目标: {self.active_goal[:800]}\n"
            f"受保护气体: {', '.join(self.gases) if self.gases else '(未指定)'}\n"
            f"受保护方法: {self.method or '(未指定)'}\n"
            f"受保护参数: {self.parameters or '(未指定)'}\n"
            f"执行已授权: {self.execution_authorized}\n"
            f"执行范围: {self.execution_mode}（read_only只查证，不编译计算链、不提交计算）\n"
            f"需要方案确认: {self.requires_plan_approval}\n"
            f"已批准方案版本: {self.approved_plan_version or '(无)'}\n"
            "有效用户指令:\n" + directives + "\n"
            "不得用其他气体、其他方法或无关历史任务替换上述目标。"
            "失败时必须在同一目标内诊断/修复；需要改目标时先请用户确认。"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalContract":
        if not isinstance(data, dict):
            return cls()
        if int(data.get("schema_version", 0) or 0) < 2:
            migrated = cls.from_user_message(str(data.get("original_goal", "") or ""))
            directives = list(data.get("directives", []) or [])
            for directive in directives[1:]:
                migrated.apply_user_message(str(directive))
            migrated.approved_plan_version = int(data.get("approved_plan_version", 0) or 0)
            migrated.pending_plan_version = int(data.get("pending_plan_version", 0) or 0)
            migrated.execution_authorized = bool(data.get("execution_authorized", migrated.execution_authorized))
            migrated.drift_blocks = list(data.get("drift_blocks", []) or [])[-20:]
            return migrated
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        result = cls(**{k: v for k, v in data.items() if k in allowed})
        if int(data.get('schema_version', 0)) < 4:
            text = '\n'.join(result.directives) or result.active_goal or result.original_goal
            for key, value in scientific_constraints(text).items():
                result.parameters.setdefault(key, value)
            if 'execution_mode' not in data:
                inferred = cls.from_user_message(result.active_goal or result.original_goal)
                result.execution_mode = inferred.execution_mode
                if result.execution_mode != 'workflow' and not result.approved_nodes:
                    result.requires_plan_approval = False
                    result.execution_authorized = False
            result.schema_version = 4
        return result
