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
    ("MD", ("molecular dynamics", "分子动力学", " md ", "lammps")),
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


def _execution_scope_text(text: str) -> str:
    """Remove negated scope markers before interpreting execution limits.

    A reference to a read-only mode is not itself a restriction when the
    user explicitly rejects that mode. Other prohibitions in the message
    remain intact and continue to take precedence over execution requests.
    """
    value = str(text or '')
    value = re.sub(
        r'(?:(?:并)?不是|并非|不只是|不仅(?:仅)?是?|不要|别)\s*'
        r'(?:只调查|仅调查|只读|只查|仅查)|'
        r'\b(?:not|isn[\'’]t)\s+(?:(?:just|only)\s+)?read[- ]only\b|'
        r'\b(?:do\s+not|don[\'’]t)\s+(?:only|just)\s+(?:inspect|check|investigate|read)\b',
        ' ', value, flags=re.IGNORECASE)
    return re.sub(r'(?:不能不|不得不|不要不|不是不|并非不)(?:提交|执行|运行|启动|开算)', ' ', value)


_READONLY_SCOPE = (
    r'只调查|仅调查|只读|只查|仅查|\bread[- ]only\b|'
    r'\b(?:only|just)\s+(?:inspect|check|investigate|read)\b'
)
_NO_EXECUTION = (
    r'(?:不(?:要|再|允许|准|得|会)?|禁止|无需|不用)\s*'
    r'(?:再|直接|实际|立即|继续|自动|自主)?(?:启动|运行|提交|开算|执行|开始计算|计算|模拟)|'
    r'\b(?:do\s+not|don[\'’]t|must\s+not|never|not(?:\s+to)?)\s+'
    r'(?:(?:actually|directly|immediately|automatically)\s+)?(?:run|submit|start|launch|execute|calculate|compute|simulate)\b|'
    r'\bwithout\s+(?:running|submitting|starting|launching|executing)\b'
)


def is_readonly_request(text: str) -> bool:
    """Explicit investigation-only scope overrides incidental compute nouns."""
    scope = _execution_scope_text(text)
    prohibited = bool(re.search(_READONLY_SCOPE + '|' + _NO_EXECUTION, scope, re.IGNORECASE))
    preparation = re.sub(
        r'(?:不(?:要)?|禁止|无需|不用)(?:生成|准备|写出|构建)|'
        r'\b(?:do\s+not|don[\'’]t)\s+(?:prepare|generate)\b', ' ', scope, flags=re.IGNORECASE)
    preparing = bool(re.search(
        r'(?:生成|准备|写出|构建).{0,8}(?:输入|脚本|文件|结构|参数)|\b(?:prepare|generate)\b',
        preparation, re.IGNORECASE))
    return prohibited and not preparing


def prohibits_submission(text: str) -> bool:
    return bool(re.search(_READONLY_SCOPE + '|' + _NO_EXECUTION,
                          _execution_scope_text(text), re.IGNORECASE))


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

_ACTION_VERBS = r'分析|计算|模擬|模拟|运行|执行|启动|处理|生成|绘图|绘制|训练|预测|提取|收集|整理|筛选|做|跑'
_CONDITIONAL = re.compile(r'如果|假如|倘若|除非|一旦|(?:等|待).{0,16}(?:确认|批准|同意|准备|完成)|\b(?:if|unless|once|until|when)\b', re.I)


def _authority_text(text):
    value=_execution_scope_text(str(text or ''))
    value=re.sub(r'```.*?```|`[^`\n]+`|[“「『][^”」』]*[”」』]|"[^"\n]*"', ' ',value,flags=re.S)
    return re.sub(r'https?://\S+|(?:\./|\.\./)[^\s]+|(?<![A-Za-z0-9])/(?:[^/\s]+/)+[^\s]*',' ',value).lower()


def _authority_match_allowed(text,start):
    sentence=re.split(r'[。；;!?！？\n]',text[:start])[-1]
    clause=re.split(r'[，,]',sentence)[-1]
    if _CONDITIONAL.search(sentence):return False
    if re.search(r'(?:不要|不能|不应|禁止|别|尚未|没有|未曾|不是|并非|不准|未授权|不授权|没同意|不必|无需).{0,18}$|\b(?:not|never|without)\b.{0,24}$',clause,re.I):return False
    return not re.search(r'^\s*(?:请|帮我)?\s*(?:解释|说明|讨论|举例|介绍|为何|为什么|如何|是否|能否|这句话|提示词)|^\s*(?:explain|describe|discuss|why|how|whether)\b',clause,re.I)


def _natural_action_matches(text):
    patterns=(
        r'(?:直接|立即|马上|即刻|立刻|现在就)\s*(?:开始)?(?:'+_ACTION_VERBS+r')',
        r'(?:直接|立即|马上|现在就)\s*(?:对|针对|把|将)[^。；;!?！？\n]{1,2048}?(?:'+_ACTION_VERBS+r')',
        r'\b(?:directly|immediately)\s+(?:analy[sz]e|calculate|compute|simulate|run|execute|perform|process|train|predict|plot)\b',
    )
    return [match for pattern in patterns for match in re.finditer(pattern,text,re.I)]


def _deferred_action_request(text):
    value=_authority_text(text)
    matches=_natural_action_matches(value)+[match for phrase in _DIRECT_EXECUTE for match in re.finditer(re.escape(phrase),value)]
    return bool(matches)and not any(_authority_match_allowed(value,match.start())for match in matches)and any(
        _CONDITIONAL.search(re.split(r'[。；;!?！？\n]',value[:match.start()])[-1])for match in matches)


def explicitly_authorizes_execution(text: str) -> bool:
    """Recognize direct authority expressed as a natural autonomous mandate.

    This is deliberately about authority, not scientific routing: methods and
    protected conditions remain typed elsewhere in the goal contract.
    """
    low = _authority_text(text)
    if prohibits_submission(text):return False
    matches=_natural_action_matches(low)+[match for phrase in _DIRECT_EXECUTE for match in re.finditer(re.escape(phrase),low)]
    if any(_authority_match_allowed(low,match.start())for match in matches):
        return True
    return any(_authority_match_allowed(low,match.start())for match in re.finditer(
        r'(?:自主|自动|agent自己).{0,12}(?:执行|推进|完成|修补)|'
        r'(?:完整|全程).{0,10}(?:推动|推进|执行).{0,12}(?:结果|报告|完成)',
        low,
    ))

def _continuation_control_text(text):
    """A rejected proposed change is not a positive change instruction.

    Strip only whole negated proposal clauses ending in the plan noun. Any
    following physical change, new source path or unrecognized wording keeps
    the ordinary conservative checks. This parser never grants execution.
    """
    clauses = re.split(r'[，,。；;\n]|\b(?:but|then|and)\b|但是|然后|并且|同时', str(text), flags=re.I)
    kept = []
    for clause in clauses:
        rejected = re.fullmatch(
            r'\s*(?:先|暂时|目前)?\s*(?:不采用|不接受|不批准|不同意|拒绝|'
            r'do\s+not\s+accept|don[\'’]t\s+accept|reject)'
            r'[^。；;\n]*?(?:方案|提案|计划|\bproposal|\bplan)\s*[.!！]?\s*', clause, flags=re.I)
        # A newly named source remains significant even inside a rejection.
        scope_text = re.sub(r'第\s*\d+\s*版', '版本', clause)
        if rejected and not re.search(r'\d|\.(?:cif|csv|dat|json|pdf|html?)(?![A-Za-z0-9_])|改用|改为|改成|换用|换成|新增|增加|另加|另外', scope_text, re.I):
            continue
        if re.fullmatch(r'\s*(?:也)?(?:不接受|不同意|不批准|拒绝)删除[^。；;\n]*步骤\s*[.!！]?\s*', clause, re.I) and not re.search(r'\d|\.(?:cif|csv|dat|json|pdf|html?)(?![A-Za-z0-9_])|改|换|新增|增加|另加|另外', clause, re.I):
            continue
        kept.append(clause)
    return ' ; '.join(kept)


def scope_preserving_continuation(text, approved_nodes):
    """Conservative plain resume recognition, never an execution grant."""
    low=_continuation_control_text(text).lower()
    return (scope_preserving_status_review(text,approved_nodes) or (bool(re.search(r'继续|接着|resume|continue',low))
        and bool(re.search(r'原来|原先|原任务|原参数|原要求|既有|保持|按原|same|previous|existing|unchanged',low))
        and not re.search(r'改|换|新增|增加|减少|移除|删除|另加|另外|替换|force.?field|力场|结构|材料范围|输出范围|new input|different|change|\b(?:remove|delete|replace|add|reduce)\b',low)
        and not (set(re.findall(r'[A-Za-z0-9_./+\-]+\.(?:cif|csv|dat|json|pdf|html?)(?![A-Za-z0-9_])',str(text)))
            - set(re.findall(r'[A-Za-z0-9_./+\-]+\.(?:cif|csv|dat|json|pdf|html?)(?![A-Za-z0-9_])',json.dumps(approved_nodes,ensure_ascii=False))))))


def scope_preserving_status_review(text,approved_nodes):
    """Recognize discussion of existing execution status, never a new task."""
    if not approved_nodes:return False
    low=str(text or '').lower()
    if not re.search(r'复核|核实|核对|检查|更正|纠正|修正|澄清',low):return False
    if not re.search(r'任务状态|执行状态|运行状态|执行记录|运行记录|状态.{0,8}(?:说明|表述)|(?:说明|表述).{0,8}状态',low):return False
    # Explicitly negated recomputation is a preservation request. All other
    # computation, science, data, report-scope or write changes remain new scope.
    low=re.sub(r'(?:不要|无需|不必|不再|禁止)\s*(?:重新|再次|再)?\s*(?:计算|模拟|运行|重算)', '',low)
    # Accept only a bounded vocabulary of status/wording operations; unknown
    # clauses, numbers, paths and scientific changes fail closed.
    tokens=(r'任务状态|执行状态|运行状态|执行记录|运行记录|先前|之前|当前|原来|原先|现有|已有|原始|历史|'
            r'复核|核实|核对|检查|更正|纠正|修正|澄清|说明|表述|回复|答复|结果|数据|记录|状态|'
            r'根据|依据|按照|针对|关于|保留|保持|不变|不重算|请|把|将|的|及|和|与|并|后|再|下')
    remainder=re.sub(tokens,'',low)
    return not re.sub(r'[\s，,。；;：:！!？?、（）()]+','',remainder)


_APPROVAL_WORDS = (
    "确认执行", "确认，执行", "确认，跑吧", "确认开始", "同意方案",
    "按方案执行", "可以开始", "approved", "approve", "go ahead",
)

_CHANGE_WORDS = (
    "改用", "改成", "改为", "改算", "换成", "换用", "切换", "替换", "不要", "停止", "取消",
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
    # Excluding an algorithm is not selecting it. Keep positive later clauses.
    raw=re.sub(r'(?:不要|不用|不采用|不使用|不是|并非|而非)\s*(?:直接|用|使用|采用|改用)?\s*(?:gcmc|md|cdfT|henry|vasp|dft)(?![A-Za-z0-9_])',' ',raw,flags=re.I)
    low = f" {raw.lower()} "
    discussion=bool(re.search(r'比较|对比|区别|差异|优缺点|哪个更|哪种更|\b(?:compare|difference|versus|vs)\b',low,re.I))
    selection=bool(re.search(r'(?:用|使用|采用|选用|指定|改用)\s*(?:gcmc|md|cdft|henry|vasp|dft)|\b(?:use|using|run|execute)\s+(?:gcmc|md|cdft|henry|vasp|dft)\b',low,re.I))
    named_methods=set(re.findall(r'(?<![a-z0-9_])(?:gcmc|md|cdft|henry|vasp|dft)(?![a-z0-9_])',low,re.I))
    if discussion and len(named_methods)>1 and not selection:return ''
    # cDFT is the computational method in phrases such as "用cDFT计算Henry系数";
    # Henry there is the requested observable, not permission to switch to RASPA.
    if "cdft" in low or "经典密度泛函" in low:
        return "CDFT"
    if "run_henry" in low or "henry" in low or "亨利系数" in low:
        return "HENRY"
    if re.search(r'(?<![a-z0-9_])string\s*(?:[/_-]\s*tst|方法|method\b)', low):
        programming = re.search(r'\b(?:python|javascript|typescript|java)\b', low)
        physical_path = re.search(r'扩散|能垒|过渡态|diffusion|barrier|transition', low)
        if not programming or physical_path:
            return 'TST'
    for method, patterns in _METHOD_PATTERNS:
        if any(p in low for p in patterns):
            return method
    if re.search(r'(?<![a-z0-9_])md(?![a-z0-9_])',low):return 'MD'
    # A fixed-T/P adsorption request with an explicit MC production-cycle
    # budget names the sampling operation even without the GCMC abbreviation.
    if re.search(r'吸附|\badsorption\b|\buptake\b',low)and re.search(
            r'(?:生产段?|采样|production|sampling).{0,30}[+-]?\d+\s*(?:cycles?\b|(?:个)?循环)',low,re.I):
        return 'GCMC'
    return ""


def extract_parameters(text: str) -> Dict[str, Any]:
    """Extract durable scientific constraints; never invent omitted values."""
    raw = str(text or "")
    low = raw.lower()
    out: Dict[str, Any] = {}
    # An explicit replacement identifies the old value as history, not as an
    # additional target. Leave unrelated temperatures and ordinary ranges
    # untouched: they may represent a genuine multi-temperature request.
    replacement = re.compile(
        r"(?:从\s*|\bfrom\s+)(\d+(?:\.\d+)?\s*[kK](?![a-zA-Z]))"
        r"\s*(?:改为|改成|调整为|变为|替换为|\bto\b)\s*"
        r"(\d+(?:\.\d+)?\s*[kK](?![a-zA-Z]))", re.IGNORECASE,
    )
    def discard_old_temperature(match):
        prefix = re.split(r"[，,。；;\n]", raw[:match.start()])[-1]
        if re.search(r"(?:不要|不用|不应|不能|不可|禁止|别|并非|不是|\b(?:not|never|avoid)\b)", prefix, re.I):
            return " "
        return match.group(2)
    temperature_source = replacement.sub(discard_old_temperature, raw)
    temps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*[kK](?![a-zA-Z])", temperature_source)]
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
    out.update(extract_cycle_constraints(raw,extract_method(raw)))
    return out


def extract_cycle_constraints(text: str, method_hint: str = '') -> Dict[str, int]:
    """Only explicit Monte Carlo cycle counts; never infer MD/String steps."""
    raw = str(text or '')
    context = method_hint in {'GCMC', 'HENRY', 'MC'}
    selected = None
    number = r'([+-]?(?:\d{1,3}(?:,\d{3})+|\d+))'
    for clause in re.split(r'[，。；;\n]|,(?!\d)|(?<!\d),|\bbut\b', raw, flags=re.I):
        method = extract_method(clause)
        explicit_mc = bool(re.search(r'(?<![a-zA-Z0-9_])(?:gcmc|mc|henry|widom|monte[ -]?carlo)(?![a-zA-Z0-9_])|蒙特卡洛|亨利', clause, re.I))
        if explicit_mc or method in {'GCMC', 'HENRY'}:
            context = True
        elif method:
            context = False
        if (not context or re.search(r'不要|不用|并非|不是|\b(?:not|never)\b|don[\'’]t', clause, re.I)
                or re.search(r'什么意思|是什么|够不够|是否足够|\bhow many\b|\bwhat\b.{0,24}\bcycles?\b', clause, re.I)):
            continue
        suffix = re.search(r'(?<![a-zA-Z0-9_.+-])' + number +
            r'\s*(?:cycles?(?![a-zA-Z_])|(?:个)?(?:MC)?循环)', clause, re.I)
        prefix = re.search(r'(?<![a-zA-Z_])(?:cycles?(?![a-zA-Z_])|循环次数|MC循环数)'
            r'\s*(?:[:=：]|设为|设置为|设成|改为|改成|调整为|调整到|降到|为|用|使用)?\s*' + number + r'(?![a-zA-Z0-9_.+-])', clause, re.I)
        match = suffix or prefix
        if match:
            value = int(match.group(1).replace(',', ''))
            # Retain an explicit invalid integer as well: the dispatch guard
            # rejects it rather than silently falling back to a larger run.
            selected = value
    return {'cycles': selected} if selected is not None else {}


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
        requires_approval = requires_approval or _deferred_action_request(text)
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
        deferred_action = _deferred_action_request(text)
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
        implicit_mc=(new_method=='GCMC'and not re.search(r'(?<![a-z0-9_])gcmc(?![a-z0-9_])|grand canonical|巨正则|蒙特卡洛',low,re.I))
        if implicit_mc and self.method and self.method not in {'GCMC','MULTI'}:
            new_method=''
        cycle_hint = new_method or self.method
        if cycle_hint == 'MULTI' and any(_TOOL_METHOD.get(n.get('tool')) in {'GCMC', 'HENRY'} for n in self.approved_nodes):
            cycle_hint = 'MC'
        new_parameters = {**extract_parameters(text), **scientific_constraints(text),
                          **extract_cycle_constraints(text, cycle_hint)}
        if implicit_mc and self.method and self.method not in {'GCMC','HENRY','MULTI'}:
            new_parameters.pop('cycles',None)
        changed = False

        if prohibits_submission(text) or is_readonly_request(text):
            changed = self.execution_mode != 'read_only' or self.execution_authorized
            self.execution_mode = 'read_only' if is_readonly_request(text) else 'prepare_only'
            self.execution_authorized = False
            self.requires_plan_approval = False
        elif deferred_action:
            changed = self.execution_authorized or not self.requires_plan_approval
            self.execution_authorized=False
            self.requires_plan_approval=True
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
            # Repeating the existing method while explicitly continuing its
            # original task is chat metadata, not a new scientific version.
            # Any typed science/control change above or new file still advances.
            resume_metadata = not changed and bool(self.approved_nodes) and scope_preserving_continuation(text,self.approved_nodes)
            if not resume_metadata or not scope_preserving_status_review(text,self.approved_nodes):
                self.active_goal = text
            if not resume_metadata:
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
        if name in {'run_bash','write_file'}:
            from .sandbox_execution_policy import sandbox_violation
            violation=sandbox_violation(self,name,params)
            if violation:return False,violation
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

        science_tool = delegated_node.get('tool') if is_handoff else name
        science_args = delegated_node.get('arguments', {}) if is_handoff else params
        protected_cycles = self.parameters.get('cycles')
        if protected_cycles is not None and _TOOL_METHOD.get(science_tool) in {'GCMC', 'HENRY'}:
            if (isinstance(protected_cycles, bool) or not isinstance(protected_cycles, int)
                    or protected_cycles <= 0):
                return self._block(name, '用户 cycles 约束必须为正整数', params)
            if science_tool not in {'run_gcmc_batch', 'run_gcmc_isotherm'}:
                return self._block(name, '该路线未支持显式 cycles 参数，不能静默采用默认预算；请选择支持预算的原生GCMC工具', params)
            actual_cycles = science_args.get('cycles')
            if (isinstance(actual_cycles, bool) or not isinstance(actual_cycles, int)
                    or actual_cycles != protected_cycles):
                return self._block(name, f'cycles 必须显式匹配用户约束 {protected_cycles}，不能省略或沿用旧预算', params)

        expected_ff = self.parameters.get('framework_force_field') or self.parameters.get('force_field')
        if expected_ff and science_tool == 'run_md_optimize':
            actual_ff = science_args.get('force_field')
            if actual_ff is None or str(actual_ff).casefold() != expected_ff.casefold():
                return self._block(name, '必须显式传递用户指定力场；不能省略后走默认力场或静默替换', params)
        expected_provenance = self.parameters.get('forcefield_source')
        if expected_provenance and is_submission(science_tool, science_args):
            provenance = science_args.get('forcefield_source') or science_args.get('forcefield_dir')
            if not provenance or str(provenance).casefold() != expected_provenance.casefold():
                return self._block(name, '参数来源未与用户指定来源一致验证；不能默默换原生库或未实现的转换', params)
        expected_source = self.parameters.get('framework_charge_source')
        if science_tool == 'run_pacman_charge' and expected_source == 'CIF':
            return self._block(name, '用户要求保留已有CIF电荷，不能重新赋电荷', params)
        if science_tool == 'run_pacman_charge' and expected_source in {'PACMAN','PACMOF'}:
            expected_method = 'pacman' if expected_source == 'PACMAN' else 'pacmof'
            if science_args.get('method') != expected_method:
                return self._block(name, '必须显式传入用户指定的PACMAN/PACMOF路线，不能走默认模型替代', params)
        if science_tool == 'run_pacman_charge' and expected_source in {'DDEC6', 'CM5', 'REPEAT'}:
            actual_source = str(science_args.get('charge_type', '')).upper()
            if actual_source != expected_source or science_args.get('method') != 'pacman':
                return self._block(name, '电荷模型必须与用户明确指定的路线一致', params)

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
        candidate_method = _TOOL_METHOD.get(science_tool, '')
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
        candidate_temp = science_args.get("temperature")
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

        if science_tool == "generate_structure":
            for contract_key, tool_key in (("n_structures", "n_structures"), ("max_atoms", "max_atoms")):
                expected = self.parameters.get(contract_key)
                actual = science_args.get(tool_key)
                if expected is not None and actual is not None and int(expected) != int(actual):
                    return self._block(
                        name,
                        f"未经用户批准的 {tool_key} 变更 {expected} → {actual}",
                        params,
                    )
            expected_topologies = self.parameters.get("topologies") or []
            actual_topologies = science_args.get("topologies") or []
            if expected_topologies and set(map(str, actual_topologies)) != set(map(str, expected_topologies)):
                return self._block(
                    name,
                    f"拓扑必须与用户约束完全一致: {expected_topologies}",
                    params,
                )
        # A persisted approval identifies a node, but cannot supersede newer
        # user science. Check exact node identity only after every applicable
        # gas/method/temperature/generation/force-field/charge constraint above.
        exact_nodes = [n for n in self.approved_nodes if n.get('tool') == name]
        if enforce_approved_node and self.approved_nodes and is_submission(name, params):
            if any(n.get('arguments') == params for n in exact_nodes):
                return True, ''
            return self._block(name, '调用参数必须与用户批准的结构化节点一致；改变参数需编排补丁', params)
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
        # Older checkpoints kept only the primary method even after a full
        # multi-method graph was approved. Recover authority from that exact
        # approved graph, never from prose or an unapproved draft.
        methods = {_TOOL_METHOD.get(node.get('tool')) for node in result.approved_nodes}
        methods -= {None, ''}
        if result.approved_plan_version and len(methods) > 1 and result.method in methods:
            result.method = 'MULTI'
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
        if 'cycles' not in result.parameters:
            hint = result.method
            if hint == 'MULTI' and methods & {'GCMC', 'HENRY'}:
                hint = 'MC'
            for directive in result.directives or [result.original_goal]:
                if not _is_automated_message(str(directive)):
                    result.parameters.update(extract_cycle_constraints(str(directive), hint))
        return result
