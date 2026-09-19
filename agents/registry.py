"""Tool registry — maps tool names to callable functions and Claude schema.

Two layers:
1. Python functions (for execution)
2. Claude tool_use JSON schemas (for API)

The registry auto-generates schemas from function docstrings + annotations,
or you can provide explicit schemas.
"""
from __future__ import annotations

import json
import copy
import math
import os
import re
import sys
import traceback
import shutil
import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from . import slurm
from .config import AgentConfig, get_config
from typing import Any, Callable, Dict, List, Optional
from .output_contract import OUTPUT_SCHEMA

INPUT_BINDINGS_SCHEMA = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "step_id": {"type": "string", "minLength": 1},
            "output_index": {"type": "integer", "minimum": 0},
        },
        "required": ["step_id", "output_index"],
    },
}

# Legacy paths
_legacy_root = Path("/home/user/gcmc_agent/BiMemAgent")
_project_root = Path(__file__).resolve().parents[1]
for p in (str(_project_root), str(_legacy_root), str(_legacy_root.parent), str(_legacy_root.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Tool Definition ─────────────────────────────────────────────────

@dataclass
class ToolDef:
    """A single tool: schema + executor."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    execute: Callable[[Dict[str, Any]], Any]
    
    def to_claude_schema(self) -> Dict[str, Any]:
        """Convert to Claude API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ── Registry ────────────────────────────────────────────────────────

class ToolRegistry:
    """Central registry of all tools."""
    
    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}
    
    def register(self, tool: ToolDef) -> None:
        if tool.name in self._tools and self._tools[tool.name] is not tool:
            raise ValueError(f"tool {tool.name!r} already registered; conflicting definitions are forbidden")
        from jsonschema import Draft7Validator
        Draft7Validator.check_schema(tool.input_schema)
        self._tools[tool.name] = tool

    def validate_handoff_params(self, params) -> List[str]:
        if not isinstance(params, dict):
            return ['handoff arguments must be a JSON object']
        issues = []
        if not isinstance(params.get('task'), str) or not params['task'].strip():
            issues.append('handoff task must be a non-empty string')
        if 'context' in params and not isinstance(params['context'], str):
            issues.append('handoff context must be a string containing the structured context envelope')
        unknown = set(params) - {'task', 'context'}
        if unknown:
            issues.append(f'unknown handoff arguments: {sorted(unknown)}')
        return issues

    def validate_params(self, name: str, params: Dict[str, Any]) -> List[str]:
        """Validate schema *and* scientific red lines before execution.

        Claude tool schemas guide generation but do not replace server-side
        validation.  Direct registry calls and restored/replayed calls must pass
        the same gate.
        """
        tool = self._tools.get(name)
        if tool is None:
            return [f"Unknown tool: {name}"]
        if not isinstance(params, dict):
            return ['tool arguments must be a JSON object']
        issues: List[str] = []
        # ``charge_method`` was a misleading public name: Ewald/Wolf select
        # the RASPA electrostatics summation algorithm; they do not assign
        # framework charges.  Accept persisted legacy nodes by normalizing the
        # alias for validation, while exposing only the unambiguous canonical
        # field in the current tool schema.
        schema_params = params
        if name in {'run_gcmc_isotherm', 'run_gcmc_batch'} and 'charge_method' in params:
            if 'electrostatics_method' in params:
                issues.append('use electrostatics_method only; charge_method is a legacy alias')
            else:
                schema_params = dict(params)
                schema_params['electrostatics_method'] = schema_params.pop('charge_method')
        from .watch_context import get_context
        from .workspace import tool_scope_issues
        context = get_context()
        issues += tool_scope_issues(params, context.get('username', ''), context.get('conv_id', ''), get_config().project_root)
        try:
            from jsonschema import Draft7Validator
            validation_errors = list(Draft7Validator(tool.input_schema).iter_errors(schema_params))
            for err in validation_errors:
                loc = ".".join(str(x) for x in err.absolute_path)
                # ``anyOf`` normally reports only "not valid under any schema",
                # which gives an agent no actionable repair.  Select the branch
                # whose required fields already match best and name exactly what
                # is missing.  Other validation errors keep their precise path.
                if err.validator == 'anyOf' and isinstance(schema_params, dict):
                    alternatives = err.validator_value or []
                    ranked = []
                    for index, alternative in enumerate(alternatives, 1):
                        required = list(alternative.get('required', []))
                        missing = [key for key in required if key not in schema_params]
                        ranked.append((len(required) - len(missing), -len(missing), index, required, missing))
                    if ranked:
                        _matched, _negative_missing, index, required, missing = max(ranked)
                        if missing:
                            prefix = f"{loc}: " if loc else ""
                            issues.append(
                                f"{prefix}missing required parameter(s) for closest valid mode: "
                                f"{', '.join(missing)} (mode {index} requires together: {', '.join(required)})"
                            )
                            continue
                issues.append(f"{loc + ': ' if loc else ''}{err.message}")
        except Exception as e:
            issues.append(f"schema validation unavailable: {e}")
        for key in tool.input_schema.get("required", []):
            if key in params and isinstance(params.get(key), str) and not params[key].strip():
                issues.append(f"{key} cannot be empty")
        if name in {'run_gcmc_isotherm', 'run_gcmc_batch', 'run_henry', 'run_henry_chain',
                    'run_isotherm_chain', 'run_cdft', 'run_pacman_charge', 'submit_job'}:
            unknown = set(schema_params) - set(tool.input_schema.get('properties', {}))
            if unknown:
                issues.append(f'unknown/ignored arguments are forbidden: {sorted(unknown)}')

        def _num(key: str) -> Optional[float]:
            value = params.get(key)
            if value is None:
                return None
            try:
                value = float(value)
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError
                return value
            except Exception:
                issues.append(f"{key} must be a finite number")
                return None

        temperature = _num("temperature")
        if temperature is not None and not (0 < temperature <= 2000):
            issues.append(f"temperature must be in (0, 2000] K, got {temperature}")
        for key in ("pressure", "pressure_start", "pressure_end"):
            value = _num(key)
            if value is not None and value < 0:
                issues.append(f"{key} cannot be negative")
        p0, p1 = _num("pressure_start"), _num("pressure_end")
        if p0 is not None and p1 is not None and p0 >= p1:
            issues.append("pressure_start must be lower than pressure_end")
        cycles = params.get("cycles")
        if cycles is not None:
            try:
                if int(cycles) <= 0:
                    issues.append("cycles must be a positive integer")
            except Exception:
                issues.append("cycles must be an integer")
        for key in ("n_structures", "max_atoms", "structures_per_topology", "n_pressure_points", "n_samples", 'cpus_per_task'):
            if params.get(key) is not None:
                try:
                    if int(params[key]) <= 0:
                        issues.append(f"{key} must be a positive integer")
                except Exception:
                    issues.append(f"{key} must be an integer")
        if name == "run_pore_analysis" and bool(params.get("cif_path")) == bool(params.get("cif_dir")):
            issues.append("exactly one of cif_path or cif_dir is required")
        if name == "run_pacman_charge" and bool(params.get("cif_path")) == bool(params.get("cif_dir")):
            issues.append("exactly one of cif_path or cif_dir is required")
        if name == "extract_features" and bool(params.get("cif_path")) == bool(params.get("cif_dir")):
            issues.append("exactly one of cif_path or cif_dir is required")
        if name == "run_cdft":
            action = params.get("action")
            if action in {"pipeline", "inputs"}:
                if not (params.get("cif_path") or params.get("cif_dir")):
                    issues.append("run_cdft inputs/pipeline requires cif_path or cif_dir")
                if not (params.get("gas") or params.get("gases")):
                    issues.append("run_cdft inputs/pipeline requires gas or gases")
                if params.get("temperature") is None:
                    issues.append("run_cdft inputs/pipeline requires user-confirmed temperature")
            elif action == "submit" and not params.get("input_dir"):
                issues.append("run_cdft submit requires input_dir")
            elif action == "collect" and not (params.get("job_work_dir") or params.get("input_dir")):
                issues.append("run_cdft collect requires job_work_dir or input_dir")
        if name in {'run_cdft', 'ml_predict'} and params.get('cif_path') and params.get('cif_dir'):
            issues.append('cif_path and cif_dir are mutually exclusive; choose one input scope')
        if params.get('gas') and params.get('gases'):
            issues.append('gas and gases are mutually exclusive; do not specify two competing gas scopes')
        if name == 'run_md_optimize':
            if str(params.get('force_field', '')).upper().startswith('OPLS'):
                issues.append('OPLS Towhee assets are discoverable but no Towhee→lammps-interface OPLS adapter is implemented; resolve atom typing/charges and conversion before submission. Never substitute UFF silently.')
            if params.get('mode', 'single') == 'md' and params.get('temperature') is None:
                issues.append('NPT MD requires user-confirmed temperature')
            if params.get('mode', 'single') == 'single' and any(k in params for k in ('temperature', 'pressure', 'npt_ps', 'timestep')):
                issues.append('temperature/pressure/duration are MD-only; single mode performs minimization')
            unknown = set(params) - set(tool.input_schema.get('properties', {}))
            if unknown:
                issues.append(f'unknown/ignored arguments are forbidden: {sorted(unknown)}')
        if name == 'run_vasp':
            if params.get('action') != 'submit':
                issues.append('VASP setup/setup-batch/parse are not implemented; never substitute a compute submission')
            if not params.get('work_dir'):
                issues.append('VASP submit requires an existing prepared work_dir')
        if name == 'generate_structure' and params.get('topologies'):
            count, topologies = params.get('n_structures'), params['topologies']
            if isinstance(count, int) and count < len(topologies):
                issues.append('n_structures must cover every requested topology')
            per = params.get('structures_per_topology')
            if isinstance(count, int) and isinstance(per, int) and count != per * len(topologies):
                issues.append('n_structures must equal structures_per_topology × topology count')

        def finite_values(value, location='arguments'):
            if isinstance(value, float) and not math.isfinite(value):
                issues.append(f'{location} must be finite')
            elif isinstance(value, dict):
                for key, item in value.items():
                    finite_values(item, f'{location}.{key}')
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    finite_values(item, f'{location}[{index}]')
        finite_values(params)
        return issues
    
    def get(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def tool_schema(self, name):
        tool = self.get(name)
        if not tool: return {'error': 'unknown concrete tool', 'tool': name}
        return {'ok': True, 'read_only': True, **copy.deepcopy(tool.to_claude_schema()),
                'note': 'Schema discovery is not permission to execute; assigned agent capability and argument guards still apply.'}

    @staticmethod
    def _validation_receipt(name: str, issues: List[str], schema: Dict[str, Any]) -> Dict[str, Any]:
        missing = []
        for issue in issues:
            match = re.search(
                r"missing required parameter\(s\)(?: for closest valid mode)?: ([^()]+)",
                issue,
            )
            if match:
                missing.extend(item.strip() for item in match.group(1).split(',') if item.strip())
                continue
            match = re.search(r"^'([^']+)' is a required property$", issue)
            if match:
                missing.append(match.group(1))
        return {
            "error": "parameter validation failed",
            "error_kind": "schema_validation",
            "tool": name,
            "issues": issues,
            "missing_parameters": sorted(set(missing)),
            "input_schema": schema,
            "next_action": (
                "Correct only the listed arguments against input_schema and retry. "
                "Do not ask the user unless a scientific method or condition is genuinely unspecified."
            ),
            "submitted": False,
        }
    
    def execute(self, name: str, params: Dict[str, Any]) -> str:
        """Execute a tool by name, returning result string."""
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        issues = self.validate_params(name, params)
        if issues:
            return json.dumps(self._validation_receipt(name, issues, tool.input_schema),
                              ensure_ascii=False, indent=2)
        try:
            result = self.tool_schema(params['tool_name']) if name == 'get_tool_schema' else tool.execute(params)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
    
    def execute_dict(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name, returning result dict."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        issues = self.validate_params(name, params)
        if issues:
            return self._validation_receipt(name, issues, tool.input_schema)
        try:
            result = self.tool_schema(params['tool_name']) if name == 'get_tool_schema' else tool.execute(params)
            if isinstance(result, dict):
                return result
            # If result is a string, try to parse as JSON
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {"result": result}
            return {"result": result}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}
    def claude_tools(self, tool_names: List[str] | None = None) -> List[Dict[str, Any]]:
        """Get Claude API tool schemas, optionally filtered.

        Also includes handoff functions (handoff_to_*) that are defined
        in agent functions but not in the registry.
        """
        names = set(tool_names) if tool_names else set(self._tools.keys())
        result = [
            tool.to_claude_schema()
            for tool in self._tools.values()
            if tool.name in names
        ]
        # Add handoff tools that aren't in the registry
        if tool_names:
            for name in dict.fromkeys(tool_names):
                if name.startswith("handoff_to_") and name not in self._tools:
                    result.append({
                        "name": name,
                        "description": f"Hand off task to a specialist agent. Call with task description and context.",
                        "input_schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "task": {"type": "string", "minLength": 1, "description": "Task description for the specialist"},
                                "context": {"type": "string", "description": "Context information"},
                            },
                            "required": ["task"],
                        },
                    })
        return result
    
    def all_names(self) -> List[str]:
        return list(self._tools.keys())
    
    def all_tools(self) -> List[ToolDef]:
        return list(self._tools.values())


# ── Global registry singleton ───────────────────────────────────────

_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry


# ── Lazy imports from legacy ────────────────────────────────────────

def _lazy(mod_path: str, attr: str) -> Any:
    try:
        mod = __import__(mod_path, fromlist=[attr])
        return getattr(mod, attr)
    except ImportError as e:
        return None


# ── Tool executors ──────────────────────────────────────────────────

def _ws(kind: str) -> str:
    """Per-session workspace directory for a tool kind (runs/{user}/{conv}/{kind})."""
    try:
        from .workspace import session_dir
        return str(session_dir(kind))
    except Exception:
        from .config import get_config
        return str(get_config().project_root / "tmp" / kind)



def _normalize_cif_for_raspa(cif_path: str) -> str:
    """Normalize a PACMAN-generated CIF for RASPA compatibility.
    
    PACMAN uses _space_group_name_H-M_alt which RASPA doesn't recognize.
    This fixes it to _symmetry_space_group_name_H-M.
    Returns the path to a normalized copy in a temp directory.
    """
    import tempfile
    import shutil as _shutil
    
    path = Path(cif_path)
    content = path.read_text(encoding="utf-8")
    
    needs_fix = False
    if "_space_group_name_H-M_alt" in content:
        content = content.replace("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M")
        needs_fix = True
    if "_space_group_IT_number" in content:
        content = content.replace("_space_group_IT_number", "_symmetry_Int_Tables_number")
        needs_fix = True
    if "_space_group_symop_operation_xyz" in content:
        content = content.replace("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
        needs_fix = True
    
    if not needs_fix:
        return cif_path
    
    # Write normalized copy to temp dir
    tmp_dir = Path(tempfile.mkdtemp(prefix="raspa_cif_"))
    normalized = tmp_dir / path.name
    normalized.write_text(content, encoding="utf-8")
    return str(normalized)

def _exec_gcmc_isotherm(p: Dict[str, Any]) -> Any:
    """Submit GCMC isotherm via SLURM."""
    cif = p.get("cif", "")
    gas = p.get("gas", "CO2")
    T = p.get("temperature", 298.0)
    p_min = p.get("pressure_start", 0.1)
    p_max = p.get("pressure_end", 10.0)
    n_pts = p.get("n_pressure_points", 10)
    n_cycles = p.get("cycles", 50000)
    ff = p.get("force_field", "")  # empty → auto-selects via GAS_PRESETS
    guest_ff = p.get("guest_ff", "")  # empty → auto-selects via GAS_PRESETS
    unit_cells = p.get("unit_cells", "2 2 2")
    electrostatics_method = p.get("electrostatics_method", p.get("charge_method", "Ewald"))

    # ── Framework-charge contract ───────────────────────────────────
    # Ewald/Wolf only *sum* existing electrostatics. They never create atomic
    # charges.  Any electrostatic GCMC run therefore needs a charged CIF,
    # independently of whether the guest preset itself carries point charges.
    # Prefer an existing verified charged sibling; otherwise return a typed
    # prerequisite directing the workflow to PACMOF/PACMAN.
    if electrostatics_method != "None" and cif:
        cif_path = Path(cif)
        if cif_path.exists():
            if not slurm._cif_has_charges(cif):
                # 优先复用已验证的带电荷版本；没有时返回可编排的结构化前置条件。
                charged_cif = slurm._find_charged_cif(cif)
                if charged_cif != cif:
                    print(f"  ⚠️ CIF缺少电荷，自动选择带电荷版本: {charged_cif}", flush=True)
                    cif = charged_cif
                else:
                    return {
                        "ok": False,
                        "submitted": False,
                        "error_code": "FRAMEWORK_CHARGES_REQUIRED",
                        "error": f"CIF文件缺少电荷列(_atom_site_charge): {cif}",
                        "explanation": (
                            f"{electrostatics_method} 是静电求和算法，不是电荷赋值方法；"
                            "切换 Ewald/Wolf 不能修复缺失的框架电荷。"
                        ),
                        "next_action": {
                            "tool": "run_pacman_charge",
                            "arguments": {"cif_path": cif, "method": "pacmof"},
                            "alternatives": ["reuse_verified_charged_cif", "method=pacman"],
                            "then": "validate_framework_charges",
                        },
                    }

    # ── 去重检查：同一个CIF+gas+T的作业已在运行中则跳过 ──
    try:
        from .workspace import session_dir
        _work_base = str(session_dir("gcmc"))
    except Exception:
        from .config import get_config
        _work_base = str(get_config().project_root / "tmp" / "gcmc")
    _cif_stem = Path(cif).stem
    _dedup_dir = os.path.join(_work_base, gas, f"{int(T)}K", _cif_stem)
    _dedup_marker = os.path.join(_dedup_dir, "job.done")
    if os.path.exists(_dedup_dir):
        # 检查是否有正在运行的作业
        import subprocess as _sp
        try:
            _sq = _sp.run(["squeue", "-u", os.environ.get("USER", "user"),
                           "--noheader", "-o", "%j %T %R"],
                          capture_output=True, text=True, timeout=10)
            for _line in _sq.stdout.splitlines():
                _parts = _line.split()
                if _parts and f"gcmc_{_cif_stem}_{gas}" in _parts[0] and _parts[1] == "RUNNING":
                    return {
                        "submitted": False,
                        "skipped": True,
                        "reason": f"去重: {_cif_stem} + {gas} + {T}K 的GCMC作业已在运行中",
                        "existing_job": _parts[0],
                        "work_dir": _dedup_dir,
                        "hint": "使用 check_job 或 inspect_run 查看现有作业状态",
                    }
        except Exception:
            pass

    result = slurm.submit_gcmc_isotherm(
        cif_path=cif, gas=gas, temperature=T,
        p_min=p_min, p_max=p_max, n_points=n_pts,
        n_cycles=n_cycles, force_field=ff, guest_ff=guest_ff,
        unit_cells=unit_cells,
        charge_method=electrostatics_method,
    )
    # Auto-register the job with JobWatch so a later failure wakes the agent.
    if result.get("submitted") and result.get("job_id"):
        try:
            from .watch_context import get_context
            from .job_watch import get_watch
            ctx = get_context()
            get_watch().register(
                result["job_id"],
                work_dir=result.get("work_dir", ""),
                gas=gas, cif=cif, temperature=T,
                username=ctx.get("username", ""),
                conv_id=ctx.get("conv_id", ""),
                agent_name=ctx.get("agent_name", "adsorption"),
                tool="run_gcmc_isotherm",
            )
        except Exception as e:
            print(f"  [JobWatch] register failed: {e}", flush=True)
    return result

def _exec_gcmc_batch(p: Dict[str, Any]) -> Any:
    """Submit batch GCMC as ONE SLURM job (screens all CIFs × gases in a single job).

    Previously this looped and called submit_gcmc_isotherm per (cif, gas) —
    that submitted N separate jobs for one screening task. Now a single
    sbatch script runs every (cif, gas) pair inside one job.
    """
    cif_dir = p.get("cif_dir", "")
    gases = p.get("gases", ["CO2"])
    T = p.get("temperature", 298.0)
    pressure = p.get("pressure", 1.0)
    n_cycles = p.get("cycles", 10000)
    ff = p.get("force_field", "")  # empty → per-gas preset default (GenericMOFs/wbao)
    unit_cells = p.get("unit_cells", "2 2 2")
    electrostatics_method = p.get("electrostatics_method", p.get("charge_method", "Ewald"))

    if not cif_dir:
        return {"error": "cif_dir is required for batch GCMC screening"}

    # A batch branch owns one explicit, session-scoped output directory.  This
    # lets independent force-field/temperature branches run concurrently
    # without sharing the historical ``gcmc/batch`` folder.
    from .workspace import resolve_project_path, session_root
    root = session_root()
    output_dir = resolve_project_path(
        p.get("output_dir") or (root / "gcmc" / "batch"),
        get_config().project_root,
    )
    if not output_dir.is_relative_to(root):
        return {
            "ok": False,
            "submitted": False,
            "error_code": "OUTPUT_OUTSIDE_SESSION",
            "error": "run_gcmc_batch output_dir must be inside the current session workspace",
            "session_root": str(root),
        }

    if electrostatics_method != "None":
        directory = Path(cif_dir)
        if directory.is_dir():
            uncharged = [str(path) for path in sorted(directory.glob("*.cif"))
                         if not slurm._cif_has_charges(str(path))]
            if uncharged:
                return {
                    "ok": False,
                    "submitted": False,
                    "error_code": "FRAMEWORK_CHARGES_REQUIRED",
                    "error": f"批量目录中有 {len(uncharged)} 个 CIF 缺少 _atom_site_charge",
                    "uncharged_cifs": uncharged[:20],
                    "explanation": (
                        f"{electrostatics_method} 只是静电求和算法；"
                        "不能代替 PACMOF/PACMAN 电荷赋值。"
                    ),
                    "next_action": {
                        "tool": "run_pacman_charge",
                        "arguments": {"cif_dir": cif_dir, "method": "pacmof"},
                        "alternatives": ["reuse_verified_charged_cif_directory", "method=pacman"],
                        "then": "validate_framework_charges",
                    },
                }

    result = slurm.submit_gcmc_batch(
        cif_dir=cif_dir,
        gases=gases,
        temperature=T,
        pressure=pressure,
        n_cycles=n_cycles,
        forcefield=ff,
        mode="local",
        unit_cells=unit_cells,
        charge_method=electrostatics_method,
        output_dir=str(output_dir),
    )
    _register_jobwatch(result, tool="run_gcmc_batch")
    return result

def _exec_henry(p: Dict[str, Any]) -> Any:
    """Submit Henry coefficient calculation and wait for result."""
    cif = p.get("cif", "")
    gas = p.get("gas", "CO2")
    T = p.get("temperature", 298.0)

    from .config import get_config
    # ── 去重检查：同一个CIF+gas的Henry作业已在运行中则跳过 ──
    try:
        from .workspace import session_dir
        _henry_base = str(session_dir("gcmc"))
    except Exception:
        _henry_base = str(get_config().project_root / "tmp" / "gcmc")
    _cif_stem = Path(cif).stem
    _henry_dir = os.path.join(_henry_base, f"henry_{_cif_stem}_{gas}")
    if os.path.exists(_henry_dir):
        import subprocess as _sp
        try:
            _sq = _sp.run(["squeue", "-u", os.environ.get("USER", "user"),
                           "--noheader", "-o", "%j %T %R"],
                          capture_output=True, text=True, timeout=10)
            for _line in _sq.stdout.splitlines():
                _parts = _line.split()
                if _parts and f"henry_{_cif_stem}_{gas}" in _parts[0] and _parts[1] == "RUNNING":
                    return {
                        "submitted": False,
                        "skipped": True,
                        "reason": f"去重: {_cif_stem} + {gas} 的Henry作业已在运行中",
                        "existing_job": _parts[0],
                        "work_dir": _henry_dir,
                        "hint": "使用 check_job 或 inspect_run 查看现有作业状态",
                    }
        except Exception:
            pass

    # Auto-select force field and molecule definition from gas presets
    preset = slurm._gas_preset(gas)
    
    # Auto-select charged CIF if needed
    if preset.get("use_charges") and not slurm._cif_has_charges(cif):
        charged_cif = slurm._find_charged_cif(cif)
        if charged_cif != cif:
            print(f"  ⚠️ CIF缺少电荷，自动选择带电荷版本: {charged_cif}", flush=True)
            cif = charged_cif
    
    ff = preset["forcefield"]
    mol_name = preset["molecule_name"]
    mol_def = preset["molecule_definition"]
    use_charges = "yes" if preset["use_charges"] else "no"

    config = get_config()
    work_dir = _ws("gcmc") / f"henry_{Path(cif).stem}_{gas}"

    # Local-mode staging: copy FF + molecule defs into the run dir from the
    # project-local mirror (never modify the external RASPA2 share).
    local_raspa_share = slurm.ensure_local_raspa_share()
    _stage = "\n".join([
        f"mkdir -p {work_dir}",
        f"cp \"{local_raspa_share}/forcefield/{ff}/pseudo_atoms.def\" {work_dir}/ 2>/dev/null || true",
        f"cp \"{local_raspa_share}/forcefield/{ff}/force_field.def\" {work_dir}/ 2>/dev/null || true",
        f"cp \"{local_raspa_share}/forcefield/{ff}/force_field_mixing_rules.def\" {work_dir}/ 2>/dev/null || true",
        f"cp \"{local_raspa_share}/molecules/{mol_def}/{mol_name}.def\" {work_dir}/ 2>/dev/null || true",
    ])

    command = f"""set -euo pipefail
RASPA_DIR=/home/user/RASPA2/simulations
export LD_LIBRARY_PATH=$RASPA_DIR/lib:$LD_LIBRARY_PATH
run_sim() {{
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$RASPA_DIR/bin/simulate" "$@"
  else
    "$RASPA_DIR/bin/simulate" "$@"
  fi
}}
mkdir -p {work_dir}/Output/System_0
cp {cif} {work_dir}/
{_stage}
cd {work_dir}
cat > simulation.input << 'EOF'
SimulationType            MonteCarlo
NumberOfCycles            10000
NumberOfInitializationCycles 5000
PrintEvery                1000
Forcefield                {ff}
UseChargesFromCIFFile     {use_charges}
ChargeMethod              Ewald
Framework 0
FrameworkName {Path(cif).stem}
CutOffVDW                 12.0
UnitCells                 2 2 2
ExternalTemperature       {T}
ExternalPressure          1e-6
Component 0 MoleculeName             {mol_name}
            MoleculeDefinition       {mol_def}
            TranslationProbability   0.5
            RotationProbability      0.5
            ReinsertionProbability   0.5
            SwapProbability          1.0
            CreateNumberOfMolecules  0
EOF
run_sim simulation.input
echo "Henry calculation completed"
"""
    result = slurm.submit_and_return(
        job_name=f"henry_{Path(cif).stem}_{gas}",
        command=command,
        work_dir=work_dir,
        partition="compute",
        cpus_per_task=4,
        walltime="02:00:00",
    )
    if result.get("submitted"):
        result["status"] = "SUBMITTED"
        _henry_name = Path(cif).stem
        result["note"] = f"Henry coefficient job submitted for {_henry_name} + {gas}. Job ID: {result.get('job_id', '?')}. Use check_job to monitor."

    _register_jobwatch(result, tool="run_henry")

    # Parse actual results from RASPA output
    out_dir = Path(work_dir) / "Output" / "System_0"
    if out_dir.exists():
        for f in out_dir.glob("*.data"):
            content = f.read_text()
            # Extract Henry coefficient
            for line in content.split("\n"):
                if "Henry coefficient" in line or "K0" in line:
                    result["henry_coefficient"] = line.strip()
                if "Heat of adsorption" in line or "Qst" in line:
                    result["heat_of_adsorption"] = line.strip()

    return result

def _load_sdk_zeopp():
    """按文件路径确定性加载 SDK 本地的 tools/zeopp.py。

    不能用 `from tools import zeopp` —— 外层 gcmc_agent/tools/ 是带
    __init__.py 的常规包且常排在 sys.path 前面，会把它解析成外层旧版
    （subprocess 登录节点直跑，无 submit_pore_analysis）。按绝对路径加载
    可以绕开包名解析歧义。
    """
    import importlib.util
    _sdk_zeopp = Path(__file__).resolve().parents[1] / "tools" / "zeopp.py"
    _spec = importlib.util.spec_from_file_location("_sdk_zeopp", _sdk_zeopp)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


def _exec_pore_analysis(p: Dict[str, Any]) -> Any:
    """Zeo++ 孔结构特征计算 —— 确认必要参数后复用唯一一个 sbatch 提交。

    收敛原则（用户要求）：工具类不在提交环节浪费 token/时间 ——
    · 只确认几个必要参数：cif_path XOR cif_dir、probe_radius、n_samples
    · 一个调用 = 一个 SLURM job（10 个 MOF 也只有一个 job）
    · 提交后立即返回 job_id / 产物 CSV 路径，不阻塞回读结果
    · 脚本生成、批内逐 CIF 独立容错、结果落盘均在 tools/zeopp.py 完成
    """
    cif_path = p.get("cif_path", "")
    cif_dir = p.get("cif_dir", "")
    probe_radius = p.get("probe_radius", 1.525)
    n_samples = int(p.get("n_samples", 5000))
    output_csv = p.get("output_csv", "")

    if bool(cif_path) == bool(cif_dir):
        return {"error": "run_pore_analysis 必须提供 cif_path 或 cif_dir（二选一）"}

    result = _load_sdk_zeopp().submit_pore_analysis(
        cif_path=cif_path or None,
        cif_dir=cif_dir or None,
        probe_radius=probe_radius,
        n_samples=n_samples,
        output_csv=output_csv or None,
    )

    # 自动注册 JobWatch：作业完成/失败时唤醒对应会话，并把 job_id
    # 同步进 TaskLine（锁定到当前任务线，下游 agent 知道去哪取产物）。
    if result.get("submitted"):
        result["output_dir"] = result.get("work_dir", "")
    _register_jobwatch(result, tool="run_pore_analysis")
    return result

def _exec_charge(p: Dict[str, Any]) -> Any:
    """Assign framework partial charges to MOF CIFs.

    method:
      - "pacmof" (DEFAULT): fast CPU ML charge prediction (PACMOF random-forest
        trained on DDEC6). Runs via the dedicated `pacmof` conda env (sklearn
        1.3.2, compatible with the pickled models) on the compute partition —
        no GPU, no 12h DDEC6 wait. Output CIFs: <name>_pacmof.cif.
      - "pacman": GPU DDEC6/CM5 via PACMAN-charge (SLURM, slow, accurate).
        Output CIFs: <name>_pacman.cif.
    """
    cif_dir = p.get("cif_dir", "")
    cif_path = p.get("cif_path", "")
    charge_type = p.get("charge_type", "DDEC6")
    method = (p.get("method") or "pacmof").lower()
    config = get_config()
    # Session-isolated workspace: products live under runs/{user}/{conv}/charged
    try:
        from .workspace import session_dir
        _default_out = str(session_dir("charged"))
    except Exception:
        _default_out = str(config.project_root / "cifs" / "charged")
    output_dir = p.get("output_dir", _default_out)
    if not cif_dir and cif_path:
        src = Path(cif_path)
        if not src.is_file():
            return {"error": f"cif_path does not exist: {cif_path}", "submitted": False}
        try:
            from .workspace import session_dir
            import shutil
            input_dir = session_dir("charge_inputs")
            staged = input_dir / src.name
            if src.resolve() != staged.resolve():
                shutil.copy2(src, staged)
            cif_dir = str(input_dir)
        except Exception as e:
            return {"error": f"failed to stage cif_path: {e}", "submitted": False}
    if not cif_dir:
        return {"error": "cif_dir or cif_path is required", "submitted": False}
    # work_dir == output_dir: the real product directory. JobWatch registers
    # work_dir and the completion reminder tells the agent "go read results
    # here" — if work_dir points to a scratch tmp dir that has nothing to do
    # with the actual products, the agent has no clue where the charged CIFs
    # are and ends up grep-ing the whole tree (finding historical GCMC files
    # from other runs). Keep them identical so the path is meaningful.
    work_dir = output_dir

    if method == "pacman":
        pacman_dir = "/home/user/PACMAN-charge"
        pacman_python = "/home/user/.conda/envs/pacmof2/bin/python3.9"
        command = f"""set -euo pipefail
mkdir -p {output_dir}
{pacman_python} {pacman_dir}/pmcharge.py {cif_dir} --charge_type {charge_type}
cp {cif_dir}/*_pacman.cif {output_dir}/ 2>/dev/null || true
echo "Charge calculation completed"
"""
        result = slurm.submit_and_return(
            job_name="pacman_charge",
            command=command,
            work_dir=work_dir,
            partition="gpu",
            extra_directives=["--exclude=gpu3"],
            gres="gpu:1",
            cpus_per_task=4,
            walltime="12:00:00",
        )
        suffix = "_pacman.cif"
    else:
        # PACMOF (CPU, fast, default): sklearn RF prediction in the `pacmof` env.
        # No GPU, no DDEC6 SCF — a batch of ~100 MOFs finishes in minutes.
        py = "/home/user/.conda/envs/pacmof/bin/python"
        command = f"""set -euo pipefail
mkdir -p {output_dir}
{py} -c "
import sys, glob, os
sys.path.insert(0, '/home/user/pacmof')
from pacmof.pacmof import get_charges_single_serial
cifs = sorted(glob.glob('{cif_dir}/*.cif'))
os.makedirs('{output_dir}', exist_ok=True)
ok = 0
for c in cifs:
    try:
        get_charges_single_serial(c, create_cif=True,
            path_to_output_dir='{output_dir}', add_string='_pacmof')
        ok += 1
    except Exception as e:
        print('FAILED', os.path.basename(c), str(e)[:200])
print('PACMOF charges: %d/%d CIFs done' % (ok, len(cifs)))
"
echo "PACMOF charge calculation completed"
"""
        result = slurm.submit_and_return(
            job_name="pacmof_charge",
            command=command,
            work_dir=work_dir,
            partition="compute",
            cpus_per_task=8,
            walltime="02:00:00",
        )
        suffix = "_pacmof.cif"

    result["output_dir"] = output_dir
    result["work_dir"] = work_dir
    # Read charge results
    output_path = Path(output_dir)
    if output_path.exists():
        charged_cifs = list(output_path.glob(f"*{suffix}"))
        result["charged_cifs"] = [str(f) for f in charged_cifs]
        result["charge_method"] = method
        result["charge_suffix"] = suffix
        # Read charge data from first CIF if available
        if charged_cifs:
            with open(charged_cifs[0]) as f:
                result["charge_sample"] = f.read()[:500]
    # Register JobWatch AFTER enriching the result so the record carries the
    # real product paths (output_dir + charged_cifs), not just a scratch work_dir.
    _register_jobwatch(result, tool="run_pacman_charge")
    # Record the step on the pipeline TaskLine so downstream agents can query
    # "where are the charged CIFs" instead of blind-grepping the whole tree.
    try:
        from .watch_context import get_context as _gctx
        from .task_line import get_store as _taskline
        _ctx = _gctx()
        _tl = _taskline()
        _tl.begin_line(_ctx.get("line_id") or _ctx.get("conv_id", ""),
                       username=_ctx.get("username", ""),
                       conv_id=_ctx.get("conv_id", ""),
                       title="charge pipeline")
        _tl.upsert_step(
            _ctx.get("line_id") or _ctx.get("conv_id", ""),
            (_ctx.get('step_id') if _ctx.get('tool_name') == 'run_pacman_charge' else
             (_ctx['step_id'] + ':charge') if _ctx.get('step_id') else 'charge'),
            tool="run_pacman_charge",
            arguments=p,
            branch_parent=_ctx.get('step_id', '') if _ctx.get('tool_name') not in {'', 'run_pacman_charge'} else '',
            job_ids=[result.get("job_id", "")],
            input_dir=str(cif_dir),
            output_dir=str(output_dir),
            output_files=list(result.get("charged_cifs", [])),
            done=False,
            note="submitted",
        )
    except Exception as _te:
        print(f"  [TaskLine] charge step write failed: {_te}", flush=True)
    return result

def _exec_md_optimize(p: Dict[str, Any]) -> Any:
    """Submit LAMMPS MD optimization via SLURM (real backend: tools/lammps_optimize.py)."""
    cif_path = p.get("cif_path", "")
    mode = p.get("mode", "single")
    backend = 'run_md' if mode == 'md' else 'run_single'
    options = {'cutoff': p.get('cutoff', 6.0), 'force_field': p.get('force_field', 'UFF')}
    if mode == 'md':
        options.update(temperature=p['temperature'], pressure=p.get('pressure', 1.0),
                       npt_ps=p.get('npt_ps', 700.0), timestep=p.get('timestep', 1.0))
    config = get_config()
    work_dir = _ws("md")
    import shlex

    # Real LAMMPS backend: CIF → lammps-interface → LAMMPS minimize → opt CIF.
    # Runs on the login/slurm node via python module (lammps-interface + LAMMPS).
    command = f"""set -euo pipefail
mkdir -p {shlex.quote(work_dir)}
source /home/user/.conda/etc/profile.d/conda.sh && conda activate Agent
python3 - <<'BIMEM_MD_SCRIPT' 2>&1 | tee {shlex.quote(work_dir)}/md_result.json
import sys, json
sys.path.insert(0, '{config.project_root.parent}')
from tools.lammps_optimize import {backend}
r = {backend}({cif_path!r}, output_dir={work_dir!r}, n_procs=4, **{options!r})
out = {{'cif_name': r.cif_name, 'success': r.success,
       'energy': r.energy, 'output_cif': r.output_cif,
       'work_dir': r.work_dir, 'error': r.error}}
print(json.dumps(out, indent=2))
if not r.success: sys.exit(1)
BIMEM_MD_SCRIPT
echo 'LAMMPS calculation completed'
"""
    result = slurm.submit_and_return(
        job_name=f"md_{Path(cif_path).stem}",
        command=command,
        work_dir=work_dir,
        partition="himem",
        cpus_per_task=4,
        walltime="04:00:00",
        timeout_minutes=1440,
    )
    result.update({
        "scientific_capability": "framework geometry minimization/NPT relaxation only",
        "guest_diffusion_supported": False,
        "trajectory_path": None,
        "warning": "This wrapper does not insert guest molecules or emit an unwrapped trajectory; it cannot support a guest diffusion coefficient or MSD node.",
    })
    return result

def _exec_structure_gen(p: Dict[str, Any]) -> Any:
    """Submit structure generation via SLURM using pormake."""
    config = get_config()
    output_dir = p.get("output_dir", _ws("structures"))
    dataset = Path(output_dir)
    dataset = dataset if dataset.is_absolute() else config.project_root / dataset
    if dataset.is_dir():
        for index, path in enumerate(dataset.rglob('*')):
            if index >= 10000 or path.is_file() and path.suffix.lower() == '.cif':
                return {'blocked': True, 'executed': False, 'status': 'existing_dataset_requires_reuse',
                        'reason': 'This directory contains CIF results or exceeds the bounded audit. Reuse/verify existing data; new generation requires a new isolated output directory. Never implicitly overwrite or append to completed science.'}
    material_type = p.get("material_type", "MOF")
    # Normalize material type: u-HOF -> HOF, u-COF -> COF, etc.
    material_type_upper = material_type.upper()
    if "HOF" in material_type_upper:
        material_type = "HOF"
    elif "COF" in material_type_upper:
        material_type = "COF"
    elif "MOF" in material_type_upper:
        material_type = "MOF"
    n_structures = p.get("n_structures", 10)
    max_atoms = int(p.get("max_atoms", 1500))
    topologies = [str(x).strip() for x in (p.get("topologies") or []) if str(x).strip()]
    per_topology = int(p.get("structures_per_topology", 0) or 0)

    # Get conda path from config
    conda_path = config.conda_path

    # Project-local generator is the authoritative implementation and supports
    # topology filters/max-atoms. The old external script silently ignored both.
    pormake_script = str(config.project_root / "pormake_generate_topo.py")

    import shlex
    _py = shlex.quote(pormake_script)
    _out = shlex.quote(str(output_dir))
    if topologies:
        quotient, remainder = divmod(int(n_structures), len(topologies))
        commands = []
        for i, topo in enumerate(topologies):
            count = per_topology or (quotient + (1 if i < remainder else 0))
            commands.append(
                f"python3 {_py} generate --type {shlex.quote(material_type)} "
                f"--n {count} --max-atoms {max_atoms} --output-dir {_out} "
                f"--topo {shlex.quote(topo)}"
            )
        generate_cmd = "\n".join(commands)
    else:
        generate_cmd = (
            f"python3 {_py} generate --type {shlex.quote(material_type)} "
            f"--n {int(n_structures)} --max-atoms {max_atoms} --output-dir {_out}"
        )

    command = f"""set -euo pipefail
mkdir -p {_out}
source {conda_path} && conda activate pormake
{{
{generate_cmd}
}} 2>&1 | tee {_out}/generation.log
echo "Structure generation: {n_structures} {material_type} structures completed"
"""
    # Never let Slurm's default turn this into an implicit full-node request.
    # When the graph did not pin resources, select from a fresh scheduler
    # snapshot and budget from *schedulable* free RAM.
    from .node_inventory import node_inventory, profiled_resources
    if all(p.get(key) for key in ('nodelist','partition','memory_mb')):
        selected={'nodelist':p['nodelist'],'partition':p['partition'],'memory_mb':int(p['memory_mb'])}
    else:
        resource_snapshot=node_inventory(config.project_root,refresh=True,persist=False)
        selected=profiled_resources(resource_snapshot,'generate_structure',
            cpus=p.get('cpus_per_task'),nodelist=p.get('nodelist',''),partition=p.get('partition',''),
            memory_mb=p.get('memory_mb'))
    if not selected:
        return {"blocked":True,"executed":False,
                "error":"no fresh scheduler target satisfies the generate_structure CPU/RAM profile"}
    memory_mb=selected['memory_mb']
    result = slurm.submit_and_return(
        job_name=f"gen_{material_type.lower()}",
        command=command,
        work_dir=output_dir,
        partition=selected['partition'],
        cpus_per_task=p.get('cpus_per_task',4),
        walltime=p.get('walltime','02:00:00'),
        nodelist=selected['nodelist'],
        memory_mb=memory_mb,
    )
    result["output_dir"] = str(output_dir)
    result["topologies"] = topologies
    result["n_structures"] = int(n_structures)
    result["max_atoms"] = max_atoms
    _register_jobwatch(result, tool="generate_structure")
    return result

def _exec_xtb(p: Dict[str, Any]) -> Any:
    """Submit xTB optimization via SLURM."""
    input_path = p.get("input_path", "")
    output_dir = p.get("output_dir", "")
    xtb_exe = "/home/user/wbao/xtb/xtb-dist/bin/xtb"
    gfn = p.get("gfn", 2)
    import shlex
    opt_level = shlex.quote(p.get('opt_level', 'normal'))
    charge = int(p.get('charge', 0))

    command = f"""set -euo pipefail
mkdir -p {shlex.quote(output_dir)}
cd {shlex.quote(output_dir)}
{shlex.quote(xtb_exe)} {shlex.quote(input_path)} --opt {opt_level} --gfn {gfn} --chrg {charge}
"""
    return slurm.submit_and_return(
        job_name=f"xtb_{Path(input_path).stem}",
        command=command,
        work_dir=output_dir,
        partition="compute",
        cpus_per_task=1,
        walltime="00:30:00",
    )

def _exec_guest_ff(p: Dict[str, Any]) -> Any:
    """Submit guest forcefield build via SLURM."""
    config = get_config()
    name = p.get("name", "CO2")
    output_dir = p.get("output_dir", _ws("ff"))
    script = str(config.project_root / "scripts" / "build_guest_forcefield.py")
    import shlex
    input_path = p.get('input_path')
    input_option = f' --input {shlex.quote(input_path)}' if input_path else ''

    command = f"""set -euo pipefail
mkdir -p {shlex.quote(output_dir)}
python3 {shlex.quote(script)} --name {shlex.quote(name)} --outdir {shlex.quote(output_dir)}{input_option}
"""
    return slurm.submit_and_return(
        job_name=f"ff_{name}",
        command=command,
        work_dir=output_dir,
        partition="compute",
        cpus_per_task=1,
        walltime="00:30:00",
    )

def _register_jobwatch(result: Dict[str, Any], tool: str = "") -> None:
    """Auto-register a compute job with JobWatch under the current thread-local
    conversation, so the session's report gate can poll it before the final
    report is allowed."""
    if not result or not result.get("submitted") or not result.get("job_id"):
        return
    try:
        from .watch_context import get_context
        from .job_watch import get_watch
        ctx = get_context()
        get_watch().register(
            result["job_id"],
            work_dir=result.get("work_dir", ""),
            output_dir=result.get("output_dir", ""),
            charged_cifs=result.get("charged_cifs", []),
            username=ctx.get("username", ""),
            conv_id=ctx.get("conv_id", ""),
            agent_name=ctx.get("agent_name", "adsorption"),
            tool=tool or "run_cdft",
        )
    except Exception as e:
        print(f"  [JobWatch] register failed: {e}", flush=True)


def _exec_cdft(p: Dict[str, Any]) -> Any:
    """Run cDFT using the SDK-LOCAL cDFT installation — ALL compute goes through SLURM.

    Everything lives under tools/cdft/cDFT_Initialization/ in this project:
    the DM_cdft binary, the UFF / coarsening force fields (Molecule_FF/UFF,
    data_ff_UFF, data_ff_Coarsening_gas) and data_input.py / cdft_submit.py /
    collect_cdft_results.py all resolve from their own directory, so the
    EXTERNAL High_Batch_tool copy is never touched (local mode, same
    philosophy as GCMC force fields).

    Flow (agent chooses action):
      inputs   → generate <mof>_<gas>.dat into tmp/cdft/inputs/ (a short SLURM job).
      submit   → submit_cdft_batch() sbatch-submits ONE SLURM job that runs DM_cdft
                 per MOF (xargs parallel) and auto-collects results.csv.
      collect  → re-collect results.csv from a completed job work_dir.
      pipeline→ inputs then submit (one SLURM cDFT job).
    """
    import subprocess as _sp
    action = p.get("action", "pipeline")
    config = get_config()
    if action in {'pipeline','submit'} and (not isinstance(p.get('memory_mb'),int) or isinstance(p.get('memory_mb'),bool) or p['memory_mb']<64):
        return {'blocked':True,'executed':False,'submitted':False,'status':'needs_resource_review',
                'error':'finite memory_mb budget is required; partition defaults may request all node RAM',
                'next_action':'delegate resource monitor and establish workload-backed or user-approved MiB budget; do not guess small RAM to force scheduling'}

    cdft_dir = str(config.project_root / "tools" / "cdft" / "cDFT_Initialization")
    dm_cdft = os.path.join(cdft_dir, "cDFT", "DM_cdft")
    # Session-isolated workspace: cDFT inputs/results under runs/{user}/{conv}/cdft
    try:
        from .workspace import session_dir
        work_dir = str(session_dir("cdft"))
        data_dir = str(session_dir("cdft") / "data")
    except Exception:
        work_dir = str(config.project_root / "tmp" / "cdft")
        data_dir = str(config.project_root / "tmp" / "cdft_data")
    # A disjoint scheduler directory alone does not isolate pipeline preparation.
    # Scope inputs, auxiliary data and collection output to that same explicit
    # workspace; otherwise parallel Kr/Xe pipelines overwrite shared inputs.
    if p.get('job_work_dir'):
        explicit = Path(p['job_work_dir'])
        work_dir = str((explicit if explicit.is_absolute() else config.project_root / explicit).resolve())
        data_dir = os.path.join(work_dir, 'data')
    py = "/home/user/.conda/envs/pacmof2/bin/python3"

    cif_dir = p.get("cif_dir", p.get("cif_path", ""))
    gases = p.get("gases") or ([p["gas"]] if p.get("gas") else ["CO2"])
    T = p.get("temperature", 298.0)
    fc = p.get("framework_charge", None)
    if fc is not None:
        fc = bool(fc)
    # Even legacy serial preparation must not overwrite the other gas's .dat
    # files. Explicit input_dir remains authoritative for approved DAGs.
    import hashlib
    input_scope = hashlib.sha256(json.dumps({'gases': gases, 'temperature': T, 'cif_dir': cif_dir,
        'framework_charge': fc, 'bulk_densities': p.get('bulk_densities', [])}, sort_keys=True).encode()).hexdigest()[:16]
    input_dir = p.get("input_dir", os.path.join(work_dir, 'inputs_' + input_scope))
    job_work_dir = work_dir if p.get('job_work_dir') else ''
    output_csv = p.get("output", os.path.join(work_dir, "results.csv"))
    bulk = p.get("bulk_densities", [])
    density_unit=p.get('bulk_density_unit','mol/L')
    if density_unit=='mol/L':
        # SI Avogadro constant is exact; 1 L = 10^27 cubic angstrom.
        bulk=[float(value)*6.02214076e-4 for value in bulk]
    elif density_unit!='molecule/angstrom^3':
        return {'error':'unsupported bulk_density_unit; native cDFT uses molecule/angstrom^3','submitted':False}
    resources = {key: p[key] for key in ('nodelist', 'partition', 'num_processes', 'walltime','memory_mb') if key in p}
    if action in {'pipeline', 'submit'} and (config.project_root / 'env/node_inventory.json').exists():
        from .node_inventory import node_inventory, candidates
        inventory = node_inventory(config.project_root, refresh=True)
        profile = inventory['policy']['tool_profiles']['run_cdft']
        choices = candidates(inventory, cpus=resources.get('num_processes', profile['default_cpus']), min_glibc=profile['min_glibc'], memory_mb=p['memory_mb'])
        if resources.get('nodelist'):
            choices=[node for node in choices if node['name'] in set(resources['nodelist'].split(','))]
        allowed_partitions = [resources['partition']] if resources.get('partition') else profile['partition_preference']
        targets = [(partition, node) for partition in allowed_partitions for node in choices if partition in node['partitions']]
        if not targets:
            queued = candidates(inventory, cpus=resources.get('num_processes', profile['default_cpus']),
                                min_glibc=profile['min_glibc'], memory_mb=p['memory_mb'], for_queue=True)
            if resources.get('nodelist'):
                queued = [node for node in queued if node['name'] in set(resources['nodelist'].split(','))]
            targets = [(partition, node) for partition in allowed_partitions for node in queued if partition in node['partitions']]
        if not targets:
            return {'blocked': True, 'executed': False, 'submitted': False,
                    'error': 'No verified compatible cDFT node capacity; resource agent must refresh/probe inventory or revise its resource estimate within the scientific contract'}
        partition, selected = targets[0]
        resources.update(partition=partition, nodelist=selected['name'],
                         num_processes=resources.get('num_processes', profile['default_cpus']))
    def _project_path(value):
        path = Path(value)
        return str((path if path.is_absolute() else config.project_root / path).resolve())
    if cif_dir: cif_dir = _project_path(cif_dir)
    input_dir, output_csv = _project_path(input_dir), _project_path(output_csv)

    def _run_py(code: str, _work: str = work_dir) -> Dict[str, Any]:
        os.makedirs(_work, exist_ok=True)
        try:
            proc = _sp.run([py, "-c", code], capture_output=True, text=True, timeout=900, cwd=_work)
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if proc.returncode != 0:
                return {"submitted": False, "failed": True, "error": err[-1500:] or f"cDFT step exited {proc.returncode}", "stdout": out, "stderr": err, "exit_code": proc.returncode}
            result = {"submitted": True, "stdout": out}
            # Native helpers print one final JSON receipt. Input/collection
            # receipts have no job_id, so the submission-only parser cannot
            # decode them. Merge the typed helper receipt before interpreting it.
            try:
                inner = json.loads(out.splitlines()[-1])
                if isinstance(inner, dict) and any(key in inner for key in ('n_inputs', 'input_dir', 'rows', 'output_csv', 'job_id', 'submitted', 'failed', 'error')):
                    result.update(inner)
            except (ValueError, IndexError):
                pass
            return result
        except _sp.TimeoutExpired:
            return {"submitted": False, "failed": True, "error": "cDFT step timed out (900s)"}

    _BOOT = f"""import sys, os, json
sys.path.insert(0, {cdft_dir!r})
from cdft_submit import submit_cdft_batch
"""

    if action == "inputs":
        if not cif_dir:
            return {"error": "cif_dir/cif_path is required for cDFT input generation"}
        code = _BOOT + f"""from data_input import generate_cdft_inputs, _load_cg_ff, _CG_FF
cg = _load_cg_ff(_CG_FF)
bulk = {bulk!r}
if not bulk:
    bulk = [cg.get(g, 0.0000125) for g in {gases!r}]
generated = generate_cdft_inputs(cif_dir={cif_dir!r}, gases={gases!r}, temperature={T},
                     bulk_densities=bulk, output_dir={input_dir!r},
                     framework_charge={fc!r})
n = len(generated)
if n == 0:
    raise ValueError('No cDFT inputs generated: source must contain CIF files; inspect the source layout before submitting')
print(json.dumps({{'n_inputs': n, 'input_dir': {input_dir!r}}}))
"""
        result = _run_py(code)
        if not result.get('failed'):
            from .recovery import result_object
            result.update(result_object(result))
            result['submitted'] = False  # input generation is not a scheduler receipt
            if not result.get('n_inputs'):
                result.update(failed=True, error='No cDFT inputs generated; inspect CIF source layout')
        return result
    elif action == "submit":
        r = _run_py(_BOOT + f"""r = submit_cdft_batch(input_dat_dir={input_dir!r}, gases={gases!r}, data_dir={data_dir!r}, executable={dm_cdft!r}, **{resources!r})
print(json.dumps(r, ensure_ascii=False))
""")
        r.update(input_dir=input_dir, execution_resources=resources)
        if r.get("submitted") and r.get("stdout"):
            try:
                inner = json.loads(r["stdout"].strip().splitlines()[-1])
                r["job_id"] = inner.get("job_id")
                r["work_dir"] = inner.get("work_dir")
                r["n_mofs"] = inner.get("n_mofs")
                r["status"] = inner.get("status")
                r["note"] = (f"cDFT SLURM job {inner.get('job_id')} submitted (sbatch). "
                             f"Work dir: {inner.get('work_dir')}. Results auto-collected to "
                             f"{{work_dir}}/results.csv when it completes.")
            except Exception as e:
                r["parse_error"] = str(e)
        # JobWatch reminder must know where the cDFT products live (results.csv
        # + per-MOF input_dir/) — same product-path principle as charge steps.
        r["output_dir"] = str(data_dir)
        _register_jobwatch(r, tool="run_cdft")
        # Record cDFT step on the pipeline TaskLine (input_dir=charge products).
        try:
            from .watch_context import get_context as _gctx
            from .task_line import get_store as _taskline
            _ctx = _gctx()
            _tl = _taskline()
            _lid = _ctx.get("line_id") or _ctx.get("conv_id", "")
            _tl.upsert_step(
                _lid, _ctx.get('step_id') or 'cdft', tool="run_cdft", arguments=p,
                job_ids=[r.get("job_id", "")],
                input_dir=str(input_dir), output_dir=str(data_dir),
                done=False, note=f"gases={gases} T={T}K",
                username=_ctx.get("username", ""),
                conv_id=_ctx.get("conv_id", ""),
            )
        except Exception as _te:
            print(f"  [TaskLine] cdft step write failed: {_te}", flush=True)
        return r
    elif action == "collect":
        # Work dir = the cDFT job folder containing input_dir/. If not given,
        # fall back to the most recent job folder under data_dir.
        wd = job_work_dir
        if not wd and os.path.isdir(data_dir):
            _subs = sorted(
                (d for d in Path(data_dir).iterdir() if d.is_dir()),
                key=lambda d: d.name, reverse=True,
            )
            if _subs:
                wd = str(_subs[0])
        wd = wd or work_dir
        code = _BOOT + f"""from collect_cdft_results import collect_results
rows = collect_results({wd!r}, {gases!r}, output_csv={output_csv!r})
if not rows:
    raise ValueError('cDFT collection produced zero material rows; check the actual job work_dir')
for row in rows:
    for gas in {gases!r}:
        value = row.get(gas + '_henry_mol_L_atm', '')
        if value == '' or not __import__('math').isfinite(float(value)) or float(value) < 0:
            raise ValueError('cDFT Henry output missing/nonfinite/negative for ' + str(row.get('MOF')) + ' / ' + gas)
print(json.dumps({{'rows': rows, 'output_csv': {output_csv!r}}}, ensure_ascii=False))
"""
        result = _run_py(code)
        if not result.get('failed'):
            from .recovery import result_object
            result.update(result_object(result))
            result['submitted'] = False
            result['validation_passed'] = bool(result.get('rows'))
        return result
    else:  # pipeline → generate inputs then sbatch-submit ONE SLURM cDFT job
        if not cif_dir:
            return {"error": "cif_dir/cif_path is required for cDFT pipeline"}
        code = _BOOT + f"""from data_input import generate_cdft_inputs, _load_cg_ff, _CG_FF
cg = _load_cg_ff(_CG_FF)
bulk = {bulk!r}
if not bulk:
    bulk = [cg.get(g, 0.0000125) for g in {gases!r}]
generated = generate_cdft_inputs(cif_dir={cif_dir!r}, gases={gases!r}, temperature={T},
                     bulk_densities=bulk, output_dir={input_dir!r},
                     framework_charge={fc!r})
n = len(generated)
print(json.dumps({{'n_inputs': n}}))
if n == 0:
    raise ValueError('No cDFT inputs generated; scheduler submission refused')
r = submit_cdft_batch(input_dat_dir={input_dir!r}, gases={gases!r}, data_dir={data_dir!r}, executable={dm_cdft!r}, **{resources!r})
print(json.dumps(r, ensure_ascii=False))
"""
        r = _run_py(code)
        r.update(input_dir=input_dir, execution_resources=resources)
        if r.get("submitted") and r.get("stdout"):
            try:
                lines = [ln for ln in r["stdout"].strip().splitlines() if ln.startswith("{")]
                inner = json.loads(lines[-1])
                r["job_id"] = inner.get("job_id")
                r["work_dir"] = inner.get("work_dir")
                r["n_mofs"] = inner.get("n_mofs")
                r["status"] = inner.get("status")
                r["note"] = (f"cDFT inputs generated ({r['stdout'].split(chr(10))[0]}), SLURM job "
                             f"{inner.get('job_id')} submitted via sbatch. Work dir: {inner.get('work_dir')}. "
                             "Results auto-collected to <work_dir>/results.csv on completion.")
            except Exception as e:
                r["parse_error"] = str(e)
        # JobWatch reminder carries the real product dir (results.csv location).
        r["output_dir"] = str(data_dir)
        _register_jobwatch(r, tool="run_cdft")
        # Record cDFT step on the pipeline TaskLine.
        try:
            from .watch_context import get_context as _gctx
            from .task_line import get_store as _taskline
            _ctx = _gctx()
            _tl = _taskline()
            _lid = _ctx.get("line_id") or _ctx.get("conv_id", "")
            _tl.upsert_step(
                _lid, _ctx.get('step_id') or 'cdft', tool="run_cdft", arguments=p,
                job_ids=[r.get("job_id", "")],
                input_dir=str(input_dir), output_dir=str(data_dir),
                done=False, note=f"pipeline gases={gases} T={T}K",
                username=_ctx.get("username", ""),
                conv_id=_ctx.get("conv_id", ""),
            )
        except Exception as _te:
            print(f"  [TaskLine] cdft step write failed: {_te}", flush=True)
        return r

def _exec_expand_cell(p: Dict[str, Any]) -> Any:
    """Expand CIFs so the box satisfies the minimum-image rule min(a,b,c) > 2×cutoff.

    Pure text processing (no scientific binary → runs on the node, no SLURM).
    Writes expanded (or verbatim-copied) CIFs into output_dir and returns a
    report of which structures were expanded (old/new box, atom counts).
    Useful so the SAME expanded cell is fed to cDFT and RASPA GCMC (identical
    2×cutoff rule). The cDFT Input.dat generator also auto-expands inline.
    """
    config = get_config()
    cdft_dir = str(config.project_root / "tools" / "cdft" / "cDFT_Initialization")
    cif_dir = p.get("cif_dir", "")
    if not cif_dir:
        return {"error": "cif_dir is required"}
    cutoff = p.get("cutoff", 15.0)
    output_dir = p.get("output_dir", "")
    py = "/home/user/.conda/envs/pacmof2/bin/python3"
    _code = f"""import sys, json
sys.path.insert(0, {cdft_dir!r})
from data_input import expand_cif_dir
r = expand_cif_dir({cif_dir!r}, cutoff={float(cutoff)}, output_dir={output_dir!r} or None)
print(json.dumps(r, ensure_ascii=False))
"""
    try:
        import subprocess as _sp
        proc = _sp.run([py, "-c", _code], capture_output=True, text=True, timeout=900)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return {"expanded": False, "error": err[-1200:] or f"expand_cell exited {proc.returncode}", "stdout": out[-800:]}
        r = json.loads(out.splitlines()[-1])
        r["expanded"] = True
        return r
    except Exception as e:
        return {"expanded": False, "error": str(e)}


def _exec_task_line_query(p: Dict[str, Any]) -> Any:
    """Query the pipeline TaskLine: where each step's artifacts are + done(BOOL).
    Lets ANY agent/tool on the chain find the previous step's products without
    blind-grepping the whole tree (the bug that hit job 3669)."""
    line_id = p.get("line_id", "")
    conv_id = p.get("conv_id", "")
    try:
        from .watch_context import get_context
        from .task_line import get_store
        store = get_store()
        ctx = get_context()
        username, current_conv = ctx.get('username', ''), ctx.get('conv_id', '')
        if not username or not current_conv:
            return {'ok': False, 'error': 'task_line_query requires a user/conversation scope'}
        if conv_id and conv_id != current_conv:
            return {'ok': False, 'error': 'task line access outside the current conversation is forbidden'}
        if not line_id:
            line_id = ctx.get("line_id") or ctx.get("conv_id", "")
        if line_id:
            line = store.get_line(line_id, username=username, conv_id=current_conv)
            if not line:
                return {"ok": False, "error": 'task line not found in the current user/conversation'}
            return {"ok": True, "line_id": line['line_id'], "title": line.get("title", ""),
                    'plan_version': line.get('plan_version') or max((s.get('plan_version', 0) for s in line.get('steps', [])), default=0),
                    "steps": line.get("steps", [])}
        if conv_id:
            lines = store.get_by_conv(conv_id, username=username)
            return {"ok": bool(lines), "lines": lines}
        # No id given → list all lines as a digest
        return {"ok": True, "lines": store.get_by_conv(current_conv, username=username)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _exec_task_line_update(p: Dict[str, Any]) -> Any:
    """Evidence-backed status update for non-job workflow nodes."""
    try:
        from .watch_context import get_context
        from .task_line import get_store
        ctx = get_context()
        if not ctx.get('username') or not ctx.get('conv_id'):
            return {'ok': False, 'error': 'task_line_update requires a user/conversation scope'}
        line_id = p.get("line_id") or ctx.get("line_id") or ctx.get("conv_id", "")
        step_id = str(p.get("step_id", ""))
        status = str(p.get("status", ""))
        evidence = str(p.get("evidence", "") or "")
        output_files = list(p.get("output_files") or [])
        job_ids = [str(x) for x in (p.get("job_ids") or []) if x]
        if not line_id or not step_id:
            return {"ok": False, "error": "line_id and step_id are required"}
        store = get_store()
        line = store.get_line(line_id)
        if not line or not any(s.get("step_id") == step_id for s in line.get("steps", [])):
            return {"ok": False, "error": f"unknown workflow step: {line_id}/{step_id}"}
        line_id = line['line_id']
        if status == "completed" and not (evidence.strip() or output_files or job_ids):
            return {"ok": False, "error": "completed status requires evidence, output_files, or job_ids"}
        store.upsert_step(
            line_id, step_id, status=status, done=status == "completed",
            output_files=output_files, job_ids=job_ids, note=evidence[:1000],
            username=ctx.get("username", ""), conv_id=ctx.get("conv_id", ""),
        )
        return {"ok": True, "line_id": line_id, "step_id": step_id, "status": status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _exec_restart_backend(p: Dict[str, Any]) -> Any:
    """Request a backend (uvicorn api:app) restart after the current turn ends.

    A framework-patching agent (patcher) changes api.py / agents/*.py — those
    only take effect after a backend restart. But killing uvicorn from inside
    the turn would kill the agent's own answer. So this only *requests* a
    restart via agents._restart; api.py consumes the flag in the turn's finally
    and performs the detached restart strictly after the response is written.
    """
    from .watch_context import get_context
    from auth import is_admin
    username = get_context().get('username', '')
    if not username or not is_admin(username):
        return {'blocked': True, 'error': 'backend restart affects all users and requires administrator authority', 'restart_scheduled': False}
    try:
        from ._restart import request
        request()
        return {
            "restart_scheduled": True,
            "note": "后端重启已安排，将在本回合结束后自动生效（先写完当前回复并保存，再重启）。",
        }
    except Exception as e:
        return {"restart_scheduled": False, "error": str(e)}


def _exec_binding(p: Dict[str, Any]) -> Any:
    """Submit binding energy calculation via SLURM using LAMMPS."""
    cif_path = p.get("cif_path", "")
    gas = p.get("gas", "CO2")
    config = get_config()
    work_dir = _ws("binding")

    # Use legacy analysis/interaction module via Python
    python_cmd = f"""import sys; sys.path.insert(0, '{config.project_root.parent}'); \\
from analysis.interaction import calc_binding_energy; \\
from dataclasses import asdict; \\
import json; \\
result = calc_binding_energy(cif_path='{cif_path}', gas='{gas}'); \\
print(json.dumps(asdict(result), indent=2))
"""
    command = f"""set -euo pipefail
mkdir -p {work_dir}
cd {work_dir}
source /home/user/.conda/etc/profile.d/conda.sh && conda activate Agent
python3 -c "{python_cmd}" 2>&1 | tee {work_dir}/result.json
echo "Binding energy calculation for {gas} on {cif_path} completed"
"""
    return slurm.submit_and_return(
        job_name=f"binding_{Path(cif_path).stem}",
        command=command,
        work_dir=work_dir,
        partition="compute",
        cpus_per_task=4,
        walltime="04:00:00",
    )

def _exec_vasp(p: Dict[str, Any]) -> Any:
    """Submit VASP DFT calculation via SLURM."""
    action = p.get("action", "setup")
    if action != 'submit':
        return {'error': 'VASP action not implemented; prepare inputs or parse outputs explicitly', 'submitted': False}
    config = get_config()
    
    if action == "setup":
        work_dir = p.get("work_dir", _ws("vasp"))
        return {"work_dir": work_dir, "mode": action, "note": "VASP setup - create INCAR/KPOINTS/POSCAR manually"}
    
    work_dir = p.get("work_dir", "")
    command = f"""set -euo pipefail
cd {work_dir}
module load vasp 2>/dev/null || true
mpirun vasp_std
"""
    return slurm.submit_and_return(
        job_name="vasp_calc",
        command=command,
        work_dir=work_dir,
        partition="compute",
        cpus_per_task=10,
        walltime="04:00:00",
        timeout_minutes=240,
    )

def _exec_string_tst(p: Dict[str, Any]) -> Any:
    """Submit String TST calculation via SLURM."""
    config = get_config()
    work_dir = p.get("output_dir", _ws("tst"))
    gas = p.get("gas", "CO2")
    cif_dir = p.get("cif_dir", "")
    temperature = p.get("temperature", 298.0)

    generate_script = "/home/user/gcmc_agent/High_Batch_tool/TST/String_method/generate_string_inputs.py"
    string_exe = "/home/user/TuTraSt_String/String/GPU_string_polyatmoic"
    input_dir = f"{work_dir}/string_input_{gas}"

    command = f"""set -euo pipefail
mkdir -p {work_dir} {input_dir}
python3 {generate_script} --cif-dir {cif_dir} --output-dir {input_dir} --gas {gas} --temperature {temperature}
if [ -f "{input_dir}/manifest.csv" ]; then
  cd {input_dir}
  while IFS=, read -r cif_name n_string rest; do
    [ "$cif_name" = "cif_name" ] && continue
    mkdir -p {work_dir}/$cif_name
    {string_exe} {input_dir}/$cif_name 2>&1 | tee {work_dir}/$cif_name/run.log || true
  done < manifest.csv
fi
echo "String TST calculation for {gas} completed"
"""
    return slurm.submit_and_return(
        job_name=f"string_tst_{gas}",
        command=command,
        work_dir=work_dir,
        partition="gpu",
        extra_directives=["--exclude=gpu3"],
        gres="gpu:1",
        cpus_per_task=8,
        walltime="48:00:00",
    )

def _exec_vext(p: Dict[str, Any]) -> Any:
    """Submit external potential (TuSraST) calculation via SLURM."""
    config = get_config()
    work_dir = p.get("output_dir", _ws("vext"))
    gas = p.get("gas", "CO2")
    cif_dir = p.get("cif_dir", "")
    temperature = p.get("temperature", 298.0)

    generate_script = "/home/user/gcmc_agent/High_Batch_tool/TST/TuTraSt/generate_vext_input.py"
    vext_exe = "/home/user/gcmc_agent/High_Batch_tool/TST/TuTraSt/bin/ewald_0719_cell_v"
    input_param = "/home/user/gcmc_agent/High_Batch_tool/TST/TuTraSt/TuTraSttest/input.param"
    input_dir = f"{work_dir}/vext_inputs"
    run_dir = f"{work_dir}/vext_runs"

    command = f"""set -euo pipefail
mkdir -p {work_dir} {input_dir} {run_dir}
python3 {generate_script} --cif-dir {cif_dir} --output-dir {input_dir} --gas {gas} --temperature {temperature}
for cif_dir_path in {input_dir}/*/; do
  [ -d "$cif_dir_path" ] || continue
  cd "$cif_dir_path"
  cp {input_param} . 2>/dev/null || true
  {vext_exe} 2>&1 | tee {run_dir}/$(basename "$cif_dir_path")_run.log || true
done
echo "External potential calculation for {gas} completed"
"""
    return slurm.submit_and_return(
        job_name=f"vext_{gas}",
        command=command,
        work_dir=work_dir,
        partition="compute",
        cpus_per_task=40,
        walltime="72:00:00",
        timeout_minutes=1440,
    )

def _exec_check_job(p: Dict[str, Any]) -> Any:
    """Check SLURM job status."""
    job_id = p.get("job_id", "")
    host = p.get("host", "")
    work_dir = p.get("work_dir", "")
    return slurm.check_job_status(job_id, host=host, work_dir=work_dir)


def _exec_list_my_jobs(p: Dict[str, Any]) -> Any:
    """List SLURM jobs submitted by THIS conversation (thread-local conv_id).

    This is the ONLY authoritative source for "which jobs belong to my current
    task". It reads JobWatch, which auto-registers every compute tool result
    under the current conversation when the job is submitted. Use this instead
    of a bare `squeue` (which shows ALL users' / all conversations' jobs and
    caused the bug where an agent reported another conversation's GCMC jobs as
    its own).
    """
    try:
        from .watch_context import get_context
        from .job_watch import get_watch
        ctx = get_context()
        conv = ctx.get("conv_id", "")
        user = ctx.get("username", "")
        if not conv or not user:
            return {'error': 'list_my_jobs requires an authenticated conversation scope', 'count': 0, 'jobs': []}
        jobs = [j for j in get_watch().list()
                if (not conv or j.get("conv_id") == conv)
                and (not user or j.get("username") == user)]
        if not jobs:
            return {"count": 0, "conv_id": conv,
                    "note": f"本对话（conv={conv}）尚未通过计算工具提交任何 SLURM 作业。",
                    "jobs": []}
        return {
            "count": len(jobs),
            "conv_id": conv,
            "note": f"本对话已提交 {len(jobs)} 个 SLURM 作业。这些才是你当前任务的作业；"
                    "squeue 里其他 job 属于别的对话，不要把它们当作自己的任务汇报。",
            "jobs": [
                {
                    "job_id": j.get("job_id"),
                    "state": j.get("state", "UNKNOWN"),
                    "tool": j.get("tool", ""),
                    "work_dir": j.get("work_dir", ""),
                    "gas": j.get("gas", ""),
                    "cif": j.get("cif", ""),
                    "agent_name": j.get("agent_name", ""),
                }
                for j in sorted(jobs, key=lambda x: str(x.get("job_id", "")))
            ],
        }
    except Exception as e:
        return {"error": f"list_my_jobs failed: {e}", "count": 0, "jobs": []}

def _exec_diagnose_job(p: Dict[str, Any]) -> Any:
    """Diagnose a failed SLURM job — returns state, stderr tail, likely causes.

    A successful diagnose_job call also marks the job as *diagnosed* in
    JobWatch, so the session report gate stops treating it as an unhandled
    failure (the agent has now formally analysed it and will report the
    conclusion to the user). Without this, an already-diagnosed failed job
    (e.g. a defect-repro sample) permanently blocks every final report of
    the conversation.
    """
    job_id = p.get("job_id", "")
    work_dir = p.get("work_dir", "")
    host = p.get("host", "")
    result = slurm.diagnose_job(job_id, work_dir=work_dir, host=host)
    if job_id and result and not result.get("error"):
        try:
            from .job_watch import get_watch
            get_watch().mark_diagnosed(job_id)
        except Exception as e:
            print(f"  [JobWatch] mark_diagnosed failed for {job_id}: {e}", flush=True)
    return result


def _exec_discover_forcefield(p):
    from .forcefield_catalog import discover_forcefield
    return discover_forcefield(get_config().project_root, **p)


def _exec_inspect_forcefield(p):
    from .forcefield_catalog import inspect_forcefield
    return inspect_forcefield(get_config().project_root, **p)


def _exec_validate_framework_charges(p):
    from .charge_contract import validate_framework_charges
    return validate_framework_charges(**p)


def _exec_run_project_regressions(p):
    from .project_validation import run_project_regressions
    return run_project_regressions()


def _exec_convert_physical_units(p):
    from .unit_contract import convert_physical_units
    return convert_physical_units(**p)


def _exec_resource_health(p: Dict[str, Any]) -> Any:
    """Fixed, bounded read-only resource probes; no arbitrary shell/submitter."""
    import shutil
    import subprocess
    facts = {'cpu_count': os.cpu_count(), 'load_average': list(os.getloadavg())}
    try:
        mem = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in {'MemTotal', 'MemAvailable'}:
                mem[key] = int(value.strip().split()[0])
        facts['memory_kib'] = mem
    except OSError:
        facts['memory_kib'] = None
    probes = {}
    for program, args in [('sinfo', ['--noheader', '--format=%P %a %l %D %t']),
                          ('squeue', ['--noheader', '--format=%T']), ('qstat', []), ('pbsnodes', ['-a'])]:
        executable = shutil.which(program)
        if not executable:
            continue
        try:
            result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=5)
            probes[program] = {'exit_code': result.returncode, 'verified': result.returncode == 0,
                               'output': result.stdout[:6000], 'error': result.stderr[:1000]}
            if program in {'qstat', 'pbsnodes'}:
                probes[program]['output'] = 'Structured node facts are supplied in node_inventory; job identities are only available through scoped job tools.'
        except (OSError, subprocess.TimeoutExpired) as error:
            probes[program] = {'verified': False, 'error': str(error)}
    from .node_inventory import node_inventory, candidates
    inventory = node_inventory(get_config().project_root, refresh=True, persist=False)
    inventory['candidate_nodes'] = [n['name'] for n in candidates(inventory)]
    return {'read_only': True, 'local_resources': facts, 'scheduler_probes': probes, 'node_inventory': inventory,
            'cluster_verified': any(x.get('verified') for x in probes.values()),
            'note': 'Resource availability is not calculation convergence. Unavailable probes mean UNKNOWN, not healthy.'}

def _exec_assess_job_resources(p):
    from .resource_review import assess_job_resources
    return assess_job_resources(**p)


def _exec_resource_review_decision(p):
    return {'ok':True,'read_only':True,'proposal_only':True,**p}


def _exec_build_project_frontend(p):
    from .project_validation import build_project_frontend
    return build_project_frontend()


def _exec_retarget_queued_job(p):
    return {'blocked':True,'executed':False,'reason':'retarget requires the main Session user-authorization binding; no direct registry/shell scheduling write'}


def _exec_submit_job(p: Dict[str, Any]) -> Any:
    """Submit a generic job via SLURM."""
    job_name = p.get("job_name", "generic_job")
    command = p.get("command", "echo 'No command specified'")
    nested = _scheduler_launch(command) or _script_scheduler_launch(command, p.get('work_dir', ''))
    if nested:
        return {'error': f'nested scheduler launch {nested} is forbidden; pass the actual compute command, not sbatch/qsub of another submit.sh',
                'blocked': True, 'submitted': False}
    work_dir = p.get("work_dir", str(get_config().project_root / "tmp"))
    mode = p.get("mode", "auto")

    # 防止递归调用 sbatch submit.sh 导致无限提交
    if "sbatch submit.sh" in command or "sbatch ./submit.sh" in command:
        return {
            "success": False,
            "error": "递归调用错误: command 包含 'sbatch submit.sh'，这会导致无限递归提交。"
                    "请使用正确的模拟命令，例如: run_sim simulation.input",
            "hint": "Henry系数计算应使用: run_henry tool 或 run_sim simulation.input",
            "work_dir": work_dir,
        }

    resources = {k: p[k] for k in ('partition', 'cpus_per_task', 'walltime', 'nodelist') if k in p}
    result = slurm.submit_and_return(job_name=job_name, command=command, work_dir=work_dir, mode=mode, **resources)
    result.setdefault('work_dir', str(Path(work_dir).resolve()))
    # Auto-register generic jobs too (e.g. cDFT / PACMAN / custom commands).
    if result.get("submitted") and result.get("job_id"):
        try:
            from .watch_context import get_context
            from .job_watch import get_watch
            ctx = get_context()
            get_watch().register(
                result["job_id"],
                work_dir=work_dir,
                username=ctx.get("username", ""),
                conv_id=ctx.get("conv_id", ""),
                agent_name=ctx.get("agent_name", "harness"),
                tool="submit_job",
            )
        except Exception as e:
            print(f"  [JobWatch] register failed: {e}", flush=True)
    return result

def _exec_rag(p: Dict[str, Any]) -> Any:
    """Literature search — LOCAL RAG corpus first, then WEB (Crossref + arXiv)
    when the local corpus misses the topic. Local has only ~33 preset PDFs, so
    topics like SO2 force-field parameters / SO2-N2 selectivity benchmarks are
    often absent — the web fallback closes that gap so the agent isn't left
    "information missing".
    """
    query = p.get("query", "")
    n_results = p.get("n_results", 5)
    require_both = bool(p.get("require_both", False))
    if not query:
        return {"error": "query is required"}
    # 1) Local RAG first
    local = None
    try:
        from bimem_agent.legacy_bridge import run_literature_rag
        local = run_literature_rag(query=query, n_results=n_results,
                                   query_type=p.get("query_type", "auto"))
    except Exception as e:
        local = {"ok": False, "error": str(e)}
    local_hits = None
    if isinstance(local, dict):
        local_hits = local.get("results") or local.get("matches") or []
    if local_hits:
        if require_both:
            web = _web_literature(query, n_results)
            web_hits = web.get("results") or []
            return {
                "ok": True,
                "source": "rag+web" if web_hits else "rag",
                "coverage": ["rag", "web"] if web_hits else ["rag"],
                "evidence_complete": bool(web_hits),
                "results": local_hits + web_hits,
                "local": local_hits,
                "web": web_hits,
                "note": ("已同时查询本地 RAG 与 Crossref/arXiv Web"
                         if web_hits else "已查询本地 RAG，但 Web 未返回结果；专业结论须降低强度并披露证据缺口"),
            }
        # Local corpus has relevant docs — enough.
        if p.get("web_fallback", True) and len(local_hits) < max(1, n_results // 2):
            web = _web_literature(query, n_results)
            if web.get("results"):
                return {"ok": True, "source": "rag+web", "results": local_hits + web["results"],
                        "local": local_hits, "web": web["results"],
                        "note": "本地语料不足，已用 Crossref/arXiv 补充"}
        return {"ok": True, "source": "rag", "results": local_hits}
    # 2) Local missed → web
    web = _web_literature(query, n_results)
    if web.get("results"):
        web["source"] = "web"
        web["coverage"] = ["web"]
        web["evidence_complete"] = not require_both
        web["note"] = "本地文献库未命中，结果来自 Crossref/arXiv web 检索"
        return web
    # 3) Both empty → honest failure (no fabrication)
    return {"ok": False, "error": "本地 RAG 与 web 检索均无结果", "query": query}


def _web_literature(query: str, n_results: int = 5) -> Dict[str, Any]:
    """Web literature search via Crossref (DOI metadata) + arXiv (preprints).
    Both are free, keyless HTTP APIs. Returns normalized results.
    """
    import re as _re
    import requests as _req
    results = []
    seen = set()
    # ── Crossref ──
    try:
        r = _req.get("https://api.crossref.org/works",
                     params={"query": query, "rows": n_results, "select": "title,DOI,author,container-title,published-print,published-online,abstract"},
                     timeout=15, headers={"User-Agent": "bimem-agent/3.1"})
        if r.status_code == 200:
            for it in r.json().get("message", {}).get("items", []):
                doi = it.get("DOI", "")
                # Skip supplemental/SI entries (*.s001, -s001, etc.) — these are
                # per-article supplements, not distinct papers.
                if _re.search(r"\.s\d{2,4}$|[-_/]s\d{2,4}$", doi or "", _re.I):
                    continue
                if doi in seen:
                    continue
                seen.add(doi)
                title = (it.get("title") or [""])[0]
                if not title:
                    continue
                authors = ", ".join(
                    f"{a.get('given','')} {a.get('family','')}".strip()
                    for a in (it.get("author") or [])[:3])
                jr = (it.get("container-title") or [""])
                jr = jr[0] if isinstance(jr, list) else str(jr)
                year = (it.get("published-print") or it.get("published-online") or {}).get("date-parts", [[None]])[0][0]
                results.append({
                    "title": title, "doi": doi, "authors": authors,
                    "journal": jr or "Crossref", "year": year or "",
                    "url": f"https://doi.org/{doi}",
                    "abstract": (it.get("abstract") or "")[:300],
                    "source": "crossref",
                })
    except Exception:
        pass
    # ── arXiv (http fallback to https; free preprint search) ──
    for base in ("https://export.arxiv.org/api/query", "http://export.arxiv.org/api/query"):
        try:
            r = _req.get(base,
                         params={"search_query": f'all:"{query[:120]}"',
                                 "max_results": max(0, n_results - len(results)),
                                 "sortBy": "relevance"},
                         timeout=15, headers={"User-Agent": "bimem-agent/3.1"})
            if r.status_code != 200:
                continue
            entries = _re.findall(r"<entry>.*?</entry>", r.text, _re.S)
            for e in entries[: max(0, n_results - len(results))]:
                t = (_re.search(r"<title>([^<]+)</title>", e, _re.S) or [None, ""])[1].strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                _idm = _re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", e)
                arx = _idm.group(1) if _idm else ""
                _sum = (_re.search(r"<summary>([^<]+)</summary>", e, _re.S) or [None, ""])[1].strip()
                _pub = (_re.search(r"<published>([^<]+)</published>", e) or [None, ""])[1][:4]
                results.append({
                    "title": t, "arxiv_id": arx, "authors": "",
                    "journal": "arXiv", "year": _pub or "",
                    "url": f"https://arxiv.org/abs/{arx}" if arx else "",
                    "abstract": _sum[:300], "source": "arxiv",
                })
            break  # success on first reachable base
        except Exception:
            continue
    return {"ok": bool(results), "results": results[:n_results]}

def _resolve_fs_path(path: str) -> str:
    """Resolve a (possibly relative) path against project root and its parent."""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    config = get_config()
    candidates = [
        os.path.join(config.project_root, path),
        os.path.join(config.project_root.parent, path),
        os.path.join(os.path.expanduser("~"), path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def _is_within(path: str, root: str) -> bool:
    """True if path (resolving symlinks) is inside root, or equals root."""
    try:
        rp = os.path.realpath(path)
        rr = os.path.realpath(root)
        return rp == rr or rp.startswith(rr + os.sep)
    except Exception:
        return False


# Files that must NEVER be overwritten by write_file — backend source, config,
# credentials, and the users store. An agent may legitimately write runs/tmp/
# data or scripts, but clobbering these would break the platform itself.
_WRITE_PROTECTED_FILES = frozenset({
    "env/node_inventory.json", "env/node_policy.json", "env/forcefield_sources.json",
    "api.py", "auth.py", "config.py", "config.json", "users.json",
    "agents/session.py", "agents/defns.py", "agents/registry.py",
    "agents/agent.py", "agents/raspa_errors.py", "agents/watch_context.py",
    "requirements.txt", "frontend/src/App.js", "frontend/src/App.css",
})

def _write_protected_error(path: str, root: str) -> str:
    rel = os.path.relpath(path, root) if _is_within(path, root) else path
    if not _is_within(path, root):
        return (
            f"write_file 仅允许在项目根目录内写入。目标路径 '{path}' 在项目外，已拒绝。\n"
            f"项目根目录: {root}\n"
            "如需写入项目外位置，请改用 run_bash（并自行承担风险）。"
        )
    return (
        f"write_file 拒绝覆盖受保护的系统文件: '{rel}'。\n"
        "该文件属于平台核心代码/配置，禁止修改。\n"
        "如需创建数据/脚本文件，请写到 runs/、tmp/、output/ 等数据目录。"
    )


def _exec_inspect(p: Dict[str, Any]) -> Any:
    try:
        from bimem_agent.result_inspector import inspect_path
        path = _resolve_fs_path(p["path"])
        return inspect_path(path, max_entries=p.get("max_entries", 20))
    except Exception as e:
        return {"error": str(e)}


# ── General-purpose file/shell tools (borrowed from Claude Code) ─────

def _exec_read_file(p: Dict[str, Any]) -> Any:
    """Read a file — Claude Code style: offset/limit line windows, truncation."""
    try:
        path = _resolve_fs_path(p["path"])
        fp = Path(path)
        if not fp.exists():
            return {"error": f"File not found: {path}"}
        if fp.is_dir():
            return {"error": f"Path is a directory: {path}. Use inspect_path to list a directory."}
        text = fp.read_text(errors="replace")
        lines = text.splitlines()
        total = len(lines)
        offset = max(0, int(p.get("offset", 0)))
        limit = int(p.get("limit", 0))
        if offset > total:
            return {"error": f"offset {offset} > total lines {total}"}
        snippet = "\n".join(lines[offset:offset + limit]) if limit > 0 else text
        max_chars = int(p.get("max_chars", 20000))
        truncated = False
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars]
            truncated = True
        note = ""
        if truncated:
            # Explicitly signal "this is NOT an error" so the agent doesn't burn
            # rounds diagnosing an expected truncation.
            note = ("\n\n⚠️ 输出已截断（超过 max_chars={}），这是正常行为，不是错误。"
                    "如需完整内容，请用 offset/limit 分页读取，或用 grep_search 定位关键行。"
                    "不要为此重试或诊断。").format(max_chars)
        return {
            "path": str(fp),
            "total_lines": total,
            "offset": offset,
            "limit": limit or total,
            "chars": len(snippet),
            "truncated": truncated,
            "content": snippet + note,
        }
    except Exception as e:
        return {"error": str(e)}


def _exec_write_file(p: Dict[str, Any]) -> Any:
    """Write a file (create or overwrite). Parent dirs are created as needed.

    SAFETY: writes are restricted to the project root — project-external paths
    (../, ~, /tmp, absolute system paths) and platform-critical source/config
    files are rejected, so an agent can't clobber the platform or the host.
    """
    try:
        path = _resolve_fs_path(p["path"])
        root = get_config().project_root
        # 1) Path must be inside the project root
        if not _is_within(path, root):
            return {"error": _write_protected_error(path, root)}
        # 2) Platform-critical files are read-only for the agent
        fp = Path(path)
        rel = os.path.relpath(fp, root)
        if rel in _WRITE_PROTECTED_FILES or any(
            rel == pr or rel.startswith(pr + os.sep)
            for pr in ("frontend/src", "frontend/build", ".claude", "agents", "forcefields/towhee")
        ):
            return {"error": _write_protected_error(path, root)}
        content = p.get("content", "")
        mode = p.get("mode", "write")  # write | append
        fp.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with fp.open("a", encoding="utf-8") as f:
                f.write(content if content.endswith("\n") else content + "\n")
            action = "appended to"
        else:
            with fp.open("w", encoding="utf-8") as f:
                f.write(content)
            action = "wrote"
        return {"status": "ok", "action": action, "path": str(fp), "bytes": len(content.encode("utf-8"))}
    except Exception as e:
        return {"error": str(e)}


# Heavy compute binaries that must NEVER be executed directly on the LOGIN
# node. The platform rule is: ALL scientific computation goes through SLURM
# (sbatch → compute node). An agent that needs to run one of these must submit
# a SLURM job (run_gcmc_* / run_cdft / submit_job / run_henry / ...), NOT
# invoke the binary here. This guard closes the loophole where an agent runs
# "./DM_cdft --help" (or a full DM_cdft/raspa/zeo++/xtb run) directly on the
# login node — which burns login-node CPU/RAM and is exactly the incident the
# user reported ("让这个进程在登陆节点跑了").
_LOGIN_NODE_BLOCKED_BINARIES = {
    "DM_cdft", "dm_cdft", "raspa", "zeo++", "zeo++-0.3", "xtb",
    "pacman", "pacman_charge", "GCMC_bin", "simulate",
}


def _scheduler_launch(command: str) -> Optional[str]:
    match = re.search(
        r'(?:(?:^|[;&|()\n])\s*|\b(?:exec|command|nohup|env|sudo)\s+)(?:[\w/.-]+/)?(sbatch|qsub|bsub|srun)\b',
        command,
    )
    return match.group(1) if match else None


def _script_scheduler_launch(command: str, cwd: str = '') -> Optional[str]:
    """Inspect directly referenced local scripts, without executing them.

    This closes the observed python batch_henry.py/subprocess.sbatch bypass.
    It is conservative static detection, not a sandbox for arbitrary programs.
    """
    import shlex
    try:
        tokens = shlex.split(command)
    except ValueError:
        return 'unparseable command'
    for i, token in enumerate(tokens):
        if token in {'-c', '--command'} and i + 1 < len(tokens):
            launch = _scheduler_launch(tokens[i + 1])
            if launch:
                return launch
        if not token.endswith(('.py', '.sh', '.pbs')):
            continue
        path = Path(token)
        roots = [Path(cwd)] if cwd else []
        roots.append(get_config().project_root)
        for root in roots:
            candidate = path if path.is_absolute() else root / path
            if not candidate.is_file() or candidate.stat().st_size > 1024 * 1024:
                continue
            source = '\n'.join(line for line in candidate.read_text(errors='replace').splitlines()
                               if not line.lstrip().startswith('#'))
            if (_scheduler_launch(source)
                    or re.search(r'[\"\'](?:[^\"\']*/)?(?:sbatch|qsub|bsub)[\"\']', source)):
                return f'scheduler invocation in script {candidate}'
    return None

def _detect_login_node_compute(command: str) -> Optional[str]:
    """Return a blocking reason if `command` would run a compute binary directly
    on the login node (not via sbatch/srun/ssh-to-node)."""
    low = command.lower()
    # Commands that are legitimate on the login node (job submission, queue
    # inspection, file ops) are always allowed.
    if any(k in low for k in ("sbatch", "srun", "scancel", "squeue", "sacct",
                               "ssh ", "bsub", "qsub", "pbsnodes", "sinfo",
                               "mkdir", "cp ", "ls ", "cat ", "grep", "tail",
                               "find ", "sed", "python3 -c", "ldd ", "file ",
                               "env ", "which ")):
        return None
    # Look for a compute binary invoked as a program (word boundary). "python3"
    # is in the blocked set but we already allowed "python3 -c" (inline script)
    # above; running a python script that itself calls DM_cdft is the agent's
    # choice and stays on SLURM submission only via the *_cdft tools.
    for name in _LOGIN_NODE_BLOCKED_BINARIES:
        if re.search(r"(^|[^a-zA-Z0-9_/.-])" + re.escape(name) + r"([^a-zA-Z0-9_/.-]|$)", low):
            return (
                f"命令 `{command}` 直接调用了计算二进制 `{name}`。"
                "**禁止在登录节点直接运行科学计算程序**（会占满登录节点 CPU/内存，且不满足"
                "“所有计算必须提交到计算节点”的硬规则）。"
                "请改用平台的计算工具提交 SLURM 作业："
                "GCMC→run_gcmc_isotherm / run_gcmc_batch；cDFT→run_cdft（action=submit 或 pipeline）；"
                "其他→submit_job。若只是诊断二进制依赖（ldd/file），请保留在这些只读命令内。"
            )
    return None


def _exec_run_bash(p: Dict[str, Any]) -> Any:
    """Run a shell command — Claude Code style: returns stdout/stderr/exit code.

    Guard: heavy compute binaries (DM_cdft/raspa/zeo++/xtb/...) may NOT run
    directly on the login node — all computation must go through SLURM. If the
    command would launch one of them on this (login) node, it is refused with
    an explanation instead of executed.
    """
    import subprocess
    command = p.get("command", "").strip()
    if not command:
        return {"error": "No command provided"}
    from .watch_context import get_context
    from .workspace import scoped_shell_contract, canonical_session_shell
    context = get_context()
    shell_cwd, scope_issues = scoped_shell_contract(command, p.get('cwd', ''),
        context.get('username', ''), context.get('conv_id', ''), get_config().project_root)
    if scope_issues:
        return {'blocked': True, 'executed': False, 'error': 'session shell scope violation',
                'issues': scope_issues, 'exit_code': None,
                'next_action': 'use native scientific/read tools for shared inputs; relative shell paths resolve inside your own session'}
    if context.get('username') and context.get('conv_id'):
        command = canonical_session_shell(command, get_config().project_root)
    nested = _scheduler_launch(command) or _script_scheduler_launch(command, p.get('cwd', ''))
    if nested:
        return {'blocked': True, 'error': f'{nested} must go through a formal submission tool; run_bash cannot bypass recovery/JobWatch', 'exit_code': None}
    blocked = _detect_login_node_compute(command)
    if blocked:
        return {
            "error": "登录节点计算被禁止",
            "blocked": True,
            "message": blocked,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
        }
    timeout = max(1, int(p.get("timeout", 60)))
    cwd = shell_cwd
    if context.get('username') and context.get('conv_id'):
        Path(cwd).mkdir(parents=True, exist_ok=True)
    max_out = int(p.get("max_output", 20000))
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=_resolve_fs_path(cwd) if cwd else None,
            env={**os.environ, 'PATH': '/usr/bin:/bin'} if context.get('username') and context.get('conv_id') else None,
        )
        out, err = proc.stdout, proc.stderr
        result = {
            "exit_code": proc.returncode,
            "stdout": out[:max_out],
            "stderr": err[:max_out],
            "cwd": cwd or "",
        }
        if len(out) > max_out:
            result["stdout_truncated"] = True
            # Explicitly signal: exit_code is authoritative; truncation is NOT an
            # error, so the agent doesn't burn rounds diagnosing it.
            result["stdout"] += ("\n\n⚠️ stdout 已截断（max_output={}），这是正常行为，不是错误。"
                                 "命令 exit_code={}。如需完整输出请用更精确的命令（grep/tail）过滤。"
                                 ).format(max_out, proc.returncode)
        if len(err) > max_out:
            result["stderr_truncated"] = True
        return result
    except subprocess.TimeoutExpired as e:
        return {
            "error": f"Command timed out after {timeout}s",
            "partial_stdout": (e.stdout or "")[:2000] if isinstance(e.stdout, str) else "",
            "exit_code": None,
        }
    except Exception as e:
        return {"error": str(e)}


def _exec_stage_cif_subset(p: Dict[str, Any]) -> Any:
    """Create an immutable, deterministic CIF subset with a hash manifest."""
    from .workspace import resolve_project_path, session_root
    source = resolve_project_path(p.get("cif_dir", ""), get_config().project_root)
    root = session_root()
    output = resolve_project_path(p.get("output_dir", ""), get_config().project_root)
    if not source.is_dir():
        return {"ok": False, "error": f"CIF directory not found: {source}"}
    if not output.is_relative_to(root):
        return {"ok": False, "error_code": "OUTPUT_OUTSIDE_SESSION",
                "error": "output_dir must be inside the current session workspace"}
    limit, offset = int(p.get("limit", 0)), int(p.get("offset", 0))
    if limit < 1 or offset < 0:
        return {"ok": False, "error": "limit must be >=1 and offset must be >=0"}
    files = sorted(source.glob(p.get("pattern", "*.cif")), key=lambda item: item.name)
    selected = files[offset:offset + limit]
    if len(selected) != limit:
        return {"ok": False, "error_code": "INSUFFICIENT_CIFS",
                "error": f"requested {limit} CIFs at offset {offset}, found {len(selected)}"}
    manifest_path = output / "subset_manifest.json"
    wanted = []
    for path in selected:
        wanted.append({"name": path.name, "source": str(path.resolve()),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if output.exists() and any(output.iterdir()):
        if manifest_path.is_file():
            old = json.loads(manifest_path.read_text())
            copied_match = all(
                (output / row["name"]).is_file()
                and hashlib.sha256((output / row["name"]).read_bytes()).hexdigest() == row["sha256"]
                for row in wanted
            )
            if old.get("files") == wanted and copied_match:
                return {"ok": True, "status": "reused", "output_dir": str(output),
                        "manifest": str(manifest_path), "count": len(wanted), "files": wanted}
        return {"ok": False, "error_code": "OUTPUT_NOT_EMPTY",
                "error": "subset output already contains different content; choose a new output_dir"}
    output.mkdir(parents=True, exist_ok=True)
    for source_path, row in zip(selected, wanted):
        shutil.copy2(source_path, output / row["name"])
    manifest = {"selection": "filename_ascending", "source_dir": str(source),
                "pattern": p.get("pattern", "*.cif"), "offset": offset,
                "limit": limit, "files": wanted}
    part = manifest_path.with_suffix(".json.part")
    part.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    os.replace(part, manifest_path)
    return {"ok": True, "status": "created", "output_dir": str(output),
            "manifest": str(manifest_path), "count": len(wanted), "files": wanted}


def _exec_analyze_gcmc_screening(p: Dict[str, Any]) -> Any:
    """Extract real mol/kg loadings and uptake ratios from a batch GCMC tree."""
    from .workspace import resolve_project_path, session_root
    work = resolve_project_path(p.get("work_dir", ""), get_config().project_root)
    root = session_root()
    output_csv = resolve_project_path(p.get("output_csv", ""), get_config().project_root)
    output_md = resolve_project_path(p.get("output_markdown", ""), get_config().project_root)
    if not work.is_dir(): return {"ok": False, "error": f"work_dir not found: {work}"}
    if not output_csv.is_relative_to(root) or not output_md.is_relative_to(root):
        return {"ok": False, "error_code": "OUTPUT_OUTSIDE_SESSION",
                "error": "analysis outputs must be inside the current session workspace"}
    gases = p.get("gases") or []
    if len(gases) < 2: return {"ok": False, "error": "at least two gases are required"}
    pattern = re.compile(r"Average loading absolute \[mol/kg framework\]\s+([-+0-9.eE]+)\s+\+/-\s+([-+0-9.eE]+)")
    values, evidence, failures = {}, {}, []
    for gas in gases:
        gas_root = work / str(gas)
        for data_file in sorted(gas_root.glob("**/Output/System_0/*.data")):
            material = data_file.parents[2].name
            text = data_file.read_text(errors="replace")
            matches = pattern.findall(text)
            key = (material, str(gas))
            if len(matches) != 1:
                failures.append({"material": material, "gas": gas, "file": str(data_file),
                                 "reason": f"expected one mol/kg loading, found {len(matches)}"})
                continue
            if key in values:
                failures.append({"material": material, "gas": gas, "file": str(data_file),
                                 "reason": "ambiguous duplicate result"})
                continue
            values[key] = (float(matches[0][0]), float(matches[0][1]))
            evidence[key] = str(data_file)
    materials = sorted({material for material, _ in values})
    rows = []
    numerator, denominator = map(str, gases[:2])
    for material in materials:
        missing = [gas for gas in map(str, gases) if (material, gas) not in values]
        if missing:
            failures.append({"material": material, "reason": "missing gas results", "gases": missing})
            continue
        num, num_sd = values[(material, numerator)]; den, den_sd = values[(material, denominator)]
        ratio = None if den <= 0 else num / den
        row = {"material": material, "loading_unit": "mol/kg_framework",
               **{f"{gas}_loading_mol_kg": values[(material, str(gas))][0] for gas in gases},
               **{f"{gas}_std_mol_kg": values[(material, str(gas))][1] for gas in gases},
               f"{numerator}_{denominator}_uptake_ratio": ratio,
               "ratio_definition": f"single-component uptake ratio {numerator}/{denominator}; not IAST selectivity",
               "zero_or_negative_denominator": den <= 0}
        rows.append(row)
    if failures or not rows:
        return {"ok": False, "error_code": "INCOMPLETE_GCMC_EVIDENCE", "failures": failures,
                "parsed_materials": len(rows), "work_dir": str(work)}
    rows.sort(key=lambda row: (row[f"{numerator}_{denominator}_uptake_ratio"] is not None,
                               row[f"{numerator}_{denominator}_uptake_ratio"] or float("-inf")), reverse=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    part_csv = output_csv.with_suffix(output_csv.suffix + ".part")
    with part_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    os.replace(part_csv, output_csv)
    lines = [f"# GCMC screening: {numerator}/{denominator}", "",
             f"Ratio definition: single-component uptake ratio at the simulated condition; **not IAST selectivity**.", "",
             "| Rank | Material | " + " | ".join(f"{g} (mol/kg)" for g in gases) + f" | {numerator}/{denominator} ratio |",
             "|---:|---|" + "---:|" * (len(gases) + 1)]
    for rank, row in enumerate(rows, 1):
        lines.append("| " + " | ".join([str(rank), row["material"]] +
                     [f"{row[f'{g}_loading_mol_kg']:.8g}" for g in gases] +
                     [f"{row[f'{numerator}_{denominator}_uptake_ratio']:.8g}"]) + " |")
    part_md = output_md.with_suffix(output_md.suffix + ".part")
    output_md.parent.mkdir(parents=True, exist_ok=True); part_md.write_text("\n".join(lines) + "\n"); os.replace(part_md, output_md)
    return {"ok": True, "status": "success", "rows": rows, "row_count": len(rows),
            "output_csv": str(output_csv), "output_markdown": str(output_md),
            "source_files": sorted(evidence.values()), "synthetic_data_used": False}
def _exec_grep_search(p: Dict[str, Any]) -> Any:
    """Search file contents for a regex pattern — Claude Code style grep."""
    import subprocess
    pattern = p.get("pattern", "")
    from .watch_context import get_context
    from .workspace import session_root
    context = get_context()
    scoped = bool(context.get('username') and context.get('conv_id'))
    default = str(session_root(context['conv_id'], context['username'])) if scoped else '.'
    path = _resolve_fs_path(p.get("path", default))
    include = p.get("include", "")
    max_results = int(p.get("max_results", 50))
    if not pattern:
        return {"error": "No pattern provided"}
    try:
        cmd = ["grep", "-rn", "--color=never", "-E", pattern, path]
        if scoped and not Path(path).resolve().is_relative_to(session_root(context['conv_id'], context['username'])):
            exclusions = ['--exclude-dir=runs', '--exclude-dir=state', '--exclude-dir=logs', '--exclude-dir=.ssh',
                '--exclude=users.json', '--exclude=tokens.json', '--exclude=conversation_logs.jsonl', '--exclude=.env']
            cmd[1:1] = exclusions
        if include:
            cmd = cmd[:3] + [f"--include={include}"] + cmd[3:]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = proc.stdout.splitlines()
        shown = lines[:max_results]
        return {
            "total_matches": len(lines),
            "results": shown,
            "truncated": len(lines) > max_results,
            "exit_code": proc.returncode,  # 0 = found, 1 = none, 2 = error
        }
    except subprocess.TimeoutExpired:
        return {"error": "grep timed out after 30s"}
    except Exception as e:
        return {"error": str(e)}

def _exec_inspect_run(p: Dict[str, Any]) -> Any:
    try:
        run_id = p.get("run_id", "")
        # If run_id looks like a SLURM job id (numeric), fall back to job status
        if str(run_id).strip().isdigit():
            from . import slurm
            return slurm.check_job_status(str(run_id).strip(), work_dir=p.get("work_dir", ""))
        from bimem_agent.result_inspector import inspect_run
        return inspect_run(run_id, project=p.get("project"))
    except Exception as e:
        return {"error": str(e)}

def _exec_find_cif(p: Dict[str, Any]) -> Any:
    try:
        func = _lazy("helper", "_resolve_cif")
        if func is None:
            return {"error": "helper._resolve_cif not available"}
        path = func(p["name"])
        if path is None:
            return {"error": f"CIF not found: {p['name']}"}
        return {"cif_path": str(path)}
    except Exception as e:
        return {"error": str(e)}

def _exec_features(p: Dict[str, Any]) -> Any:
    try:
        import hashlib
        import os
        from dataclasses import asdict
        from tools import mof_features
        if bool(p.get("cif_path")) == bool(p.get("cif_dir")):
            return {"error": "exactly one of cif_path or cif_dir is required"}
        if p.get("cif_path"):
            source = Path(_resolve_fs_path(p["cif_path"])).resolve()
            if not source.is_file():
                return {"error": f"cif_path is not a file: {source}"}
            result = asdict(mof_features.extract_features(str(source)))
            result.update(input_path=str(source), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            return result
        source = Path(_resolve_fs_path(p["cif_dir"])).resolve()
        if not source.is_dir():
            return {"error": f"cif_dir is not a directory: {source}"}
        threads = int(p.get("n_threads", 4))
        if not 1 <= threads <= 32:
            return {"error": "n_threads must be in [1, 32]"}
        output = _ml_path(p.get("output_csv", ""), str(Path(_ws("ml")) / "features.csv"), output=True)
        if output.exists():
            return {"error": "feature output already exists; choose a fresh output_csv", "output_csv": str(output)}
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".part")
        results = mof_features.extract_features_batch(
            cif_dir=str(source), output_csv=str(partial), n_threads=threads)
        os.replace(partial, output)
        rows = [asdict(result) for result in results]
        failures = [row for row in rows if row.get("error")]
        return {"status": "success" if rows and not failures else "partial" if rows else "failed",
                "input_dir": str(source), "n_inputs": len(list(source.glob("*.cif"))),
                "n_features": len(rows), "n_failures": len(failures), "failures": failures,
                "features": rows, "output_csv": str(output), "output_files": [str(output)]}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── ML Tool executors ────────────────────────────────────────────────

def _legacy_exec_ml_train(p: Dict[str, Any]) -> Any:
    """Train a surrogate model (GBR/RF/DNN) on MOF features + simulation data."""
    config = get_config()
    data_csv = p.get("data_csv", "")
    model_type = p.get("model_type", "GBR")
    target = p.get("target", "uptake")
    output_dir = p.get("output_dir", _ws("ml"))
    features = p.get("features", ["volume_A3", "metallic_pct", "total_unsaturation", "en_ratio"])

    # Use absolute path to avoid SLURM work_dir nesting issue
    abs_output_dir = str(Path(config.project_root) / output_dir) if not Path(output_dir).is_absolute() else output_dir

    # Get conda path from config
    conda_path = config.conda_path

    # Train in Agent environment but save with protocol=2 for cross-env compatibility
    script = f"""set -euo pipefail
mkdir -p {abs_output_dir}
source {conda_path} && conda activate Agent
python3 -c "
import pandas as pd, numpy as np, json, os, joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

data_csv = '{data_csv}'
features = {features}
target = '{target}'
model_type = '{model_type}'

# 如果没有提供数据，自动生成合成数据
if not data_csv or not os.path.exists(data_csv):
    print('未提供数据，自动生成合成MOF数据集...')
    np.random.seed(42)
    n_samples = 50
    data = {{}}
    data['volume_A3'] = np.random.uniform(500, 5000, n_samples)
    data['metallic_pct'] = np.random.uniform(0.1, 0.5, n_samples)
    data['total_unsaturation'] = np.random.uniform(10, 200, n_samples)
    data['en_ratio'] = np.random.uniform(0.2, 0.5, n_samples)
    data[target] = (
        0.001 * data['volume_A3'] +
        5.0 * data['metallic_pct'] +
        0.01 * data['total_unsaturation'] +
        2.0 * data['en_ratio'] +
        np.random.normal(0, 0.5, n_samples)
    )
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(data_csv)

available_features = [f for f in features if f in df.columns]
if not available_features:
    available_features = [c for c in df.columns if c != target and df[c].dtype in ['float64', 'int64']][:4]

X = df[available_features].fillna(0).values
y = df[target].fillna(0).values

n_samples = len(y)
cv_folds = min(5, n_samples)
if cv_folds < 2:
    cv_folds = 2

# Use Ridge regression for cross-env compatibility (saves only coefficients, no numpy arrays)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
model.fit(X_scaled, y)

# Save as simple dict for cross-environment compatibility (no sklearn object pickling)
model_data = {{
    'type': 'linear',
    'coefficients': model.coef_.tolist(),
    'intercept': float(model.intercept_),
    'feature_names': available_features,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
}}

import json as json_mod
with open(os.path.join('{abs_output_dir}', 'model.json'), 'w') as f:
    json_mod.dump(model_data, f)
# Also save as pkl for backward compatibility
import pickle
with open(os.path.join('{abs_output_dir}', 'model.pkl'), 'wb') as f:
    pickle.dump(model_data, f, protocol=2)
with open(os.path.join('{abs_output_dir}', 'scaler.pkl'), 'wb') as f:
    pickle.dump({{'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}}, f, protocol=2)
with open(os.path.join('{abs_output_dir}', 'features.pkl'), 'wb') as f:
    pickle.dump(available_features, f, protocol=2)

result = {{
    'model_type': 'Ridge',
    'cv_r2_mean': float(np.mean(scores)),
    'cv_r2_std': float(np.std(scores)),
    'n_samples': n_samples,
    'n_features': len(available_features),
    'features_used': available_features,
    'cv_folds': cv_folds,
    'model_path': os.path.join('{abs_output_dir}', 'model.json'),
    'feature_importance': sorted(zip(available_features, [abs(c) for c in model.coef_]), key=lambda x: -x[1])
}}
print(json.dumps(result, indent=2))
"
"""
    return slurm.submit_and_wait(job_name=f"ml_train_{model_type}", command=script, work_dir=abs_output_dir, timeout_minutes=10, partition="compute", cpus_per_task=4, walltime="02:00:00")


def _legacy_exec_ml_predict(p: Dict[str, Any]) -> Any:
    """Predict gas adsorption properties using trained ML model."""
    config = get_config()
    model_dir = p.get("model_dir", _ws("ml"))
    cif_path = p.get("cif_path", "")
    cif_dir = p.get("cif_dir", "")
    output_csv = p.get("output_csv", "")

    # Use absolute paths to avoid SLURM work_dir nesting issue
    abs_model_dir = str(Path(config.project_root) / model_dir) if not Path(model_dir).is_absolute() else model_dir

    # Get conda path from config
    conda_path = config.conda_path

    script = f"""set -euo pipefail
source {conda_path} && conda activate Agent
python3 -c "
import pandas as pd, numpy as np, json, os, joblib, glob, sys
sys.path.insert(0, '{config.project_root}')
from tools.mof_features import extract_features
model = joblib.load(os.path.join('{abs_model_dir}', 'model.pkl'))
scaler = joblib.load(os.path.join('{abs_model_dir}', 'scaler.pkl'))
features = joblib.load(os.path.join('{abs_model_dir}', 'features.pkl'))
results = []
cif_path = '{cif_path}'
cif_dir = '{cif_dir}'
if cif_path:
    feat = extract_features(cif_path)
    X = np.array([[getattr(feat, f, 0) for f in features]])
    pred = model.predict(scaler.transform(X))[0]
    results.append({{'cif': os.path.basename(cif_path), 'predicted': float(pred)}})
elif cif_dir:
    for cif in sorted(glob.glob(os.path.join(cif_dir, '*.cif'))):
        try:
            feat = extract_features(cif)
            X = np.array([[getattr(feat, f, 0) for f in features]])
            pred = model.predict(scaler.transform(X))[0]
            results.append({{'cif': os.path.basename(cif), 'predicted': float(pred)}})
        except: pass
output = '{output_csv}' or os.path.join('{abs_model_dir}', 'predictions.csv')
pd.DataFrame(results).to_csv(output, index=False)
print(json.dumps({{'n_predicted': len(results), 'predictions': results[:10]}}, indent=2))
"
"""
    return slurm.submit_and_wait(job_name="ml_predict", command=script, work_dir=abs_model_dir, timeout_minutes=10, partition="compute", cpus_per_task=2, walltime="00:30:00")


def _legacy_exec_ml_feature_importance(p: Dict[str, Any]) -> Any:
    """Get feature importance from trained ML model."""
    config = get_config()
    model_dir = p.get("model_dir", _ws("ml"))

    # Use absolute paths to avoid SLURM work_dir nesting issue
    abs_model_dir = str(Path(config.project_root) / model_dir) if not Path(model_dir).is_absolute() else model_dir

    # Get conda path from config
    conda_path = config.conda_path

    script = f"""set -euo pipefail
source {conda_path} && conda activate Agent
python3 -c "
import json, os, joblib, numpy as np
model = joblib.load(os.path.join('{abs_model_dir}', 'model.pkl'))
features = joblib.load(os.path.join('{abs_model_dir}', 'features.pkl'))
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
elif hasattr(model, 'coef_'):
    importances = np.abs(model.coef_)
else:
    importances = np.zeros(len(features))
ranked = sorted(zip(features, importances.tolist()), key=lambda x: -x[1])
print(json.dumps({{'feature_importance': ranked}}, indent=2))
"
"""
    return slurm.submit_and_wait(job_name="ml_importance", command=script, work_dir=abs_model_dir, timeout_minutes=5, partition="compute", cpus_per_task=1, walltime="00:10:00")


def _legacy_exec_ml_active_learning(p: Dict[str, Any]) -> Any:
    """Active learning: predict candidates, select most uncertain for simulation."""
    config = get_config()
    cif_dir = p.get("cif_dir", "")
    model_dir = p.get("model_dir", str(config.project_root / "tmp" / "ml_models"))
    n_select = p.get("n_select", 5)
    strategy = p.get("strategy", "uncertainty")

    work_dir = _ws("ml")
    script = f"""set -euo pipefail
source /home/user/.conda/etc/profile.d/conda.sh && conda activate Agent
python3 -c "
import numpy as np, json, os, joblib, glob, random, sys
sys.path.insert(0, '{config.project_root}')
from tools.mof_features import extract_features
model = joblib.load(os.path.join('{model_dir}', 'model.pkl'))
scaler = joblib.load(os.path.join('{model_dir}', 'scaler.pkl'))
features = joblib.load(os.path.join('{model_dir}', 'features.pkl'))
candidates = []
for cif in sorted(glob.glob(os.path.join('{cif_dir}', '*.cif'))):
    try:
        feat = extract_features(cif)
        X = np.array([[getattr(feat, f, 0) for f in features]])
        pred = model.predict(scaler.transform(X))[0]
        candidates.append({{'cif': cif, 'predicted': float(pred), 'name': os.path.basename(cif)}})
    except: pass
strategy = '{strategy}'
n_select = {n_select}
if strategy == 'uncertainty' and hasattr(model, 'estimators_'):
    for c in candidates:
        preds = [t.predict(scaler.transform(np.array([[getattr(extract_features(c['cif']), f, 0) for f in features]])))[0] for t in model.estimators_[:10]]
        c['uncertainty'] = float(np.std(preds))
    selected = sorted(candidates, key=lambda x: -x.get('uncertainty', 0))[:n_select]
elif strategy == 'max_prediction':
    selected = sorted(candidates, key=lambda x: -x['predicted'])[:n_select]
else:
    selected = random.sample(candidates, min(n_select, len(candidates)))
print(json.dumps({{'strategy': strategy, 'n_candidates': len(candidates), 'selected': [s['name'] for s in selected]}}, indent=2))
"
"""
    return slurm.submit_and_wait(job_name="ml_active_learning", command=script, work_dir=work_dir, partition="compute", cpus_per_task=2, walltime="00:30:00")


# Evidence-only ML implementation.  The legacy submitters above are retained
# only so old persisted source references remain readable; they are deliberately
# not registered.  These definitions are the sole live executors below.

def _ml_path(value: str, default: str, *, output: bool = False) -> Path:
    from .watch_context import get_context
    from .workspace import resolve_project_path, session_root
    config = get_config()
    path = resolve_project_path(value or default, config.project_root)
    if output:
        context = get_context()
        root = session_root(context.get("conv_id", ""), context.get("username", "")).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"ML output must be inside the current session root: {root}")
    return path


def _exec_ml_train(p: Dict[str, Any]) -> Any:
    """Train only from an explicit real labelled table; never synthesize data."""
    try:
        import hashlib
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold, cross_validate, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from .state_io import write_checkpoint

        data_path = _ml_path(p.get("data_csv", ""), "")
        if not data_path.is_file():
            return {"error": "training data_csv does not exist; synthetic labels are forbidden",
                    "data_csv": str(data_path)}
        model_type = str(p.get("model_type", "GBR")).upper()
        if model_type not in {"GBR", "RF", "RIDGE"}:
            return {"error": "model_type must be GBR, RF, or Ridge"}
        target = str(p.get("target", "uptake"))
        frame = pd.read_csv(data_path)
        if target not in frame.columns:
            return {"error": f"target column {target!r} is absent", "columns": list(frame.columns)}
        features = list(p.get("features") or
                        [name for name in frame.select_dtypes(include="number").columns if name != target])
        missing = [name for name in features if name not in frame.columns]
        if missing:
            return {"error": "requested feature columns are absent", "missing_features": missing}
        non_numeric = [name for name in features if not pd.api.types.is_numeric_dtype(frame[name])]
        if non_numeric:
            return {"error": "ML features must be numeric", "non_numeric_features": non_numeric}
        clean = frame[features + [target]].replace([np.inf, -np.inf], np.nan)
        clean[target] = pd.to_numeric(clean[target], errors="coerce")
        clean = clean.dropna(subset=[target])
        n_samples = len(clean)
        if n_samples < 8:
            return {"error": "at least 8 real labelled rows are required", "n_samples": n_samples}
        if clean[target].nunique(dropna=True) < 2:
            return {"error": "target has no variance; regression is undefined", "n_samples": n_samples}

        output_dir = _ml_path(p.get("output_dir", ""), _ws("ml"), output=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.joblib"
        metadata_path = output_dir / "model_metadata.json"
        predictions_path = output_dir / "test_predictions.csv"
        outputs = [model_path, metadata_path, predictions_path]
        if any(path.exists() for path in outputs):
            return {"error": "model outputs already exist; choose a fresh output_dir to preserve evidence",
                    "existing": [str(path) for path in outputs if path.exists()]}

        estimator = {
            "GBR": GradientBoostingRegressor(random_state=42),
            "RF": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
            "RIDGE": Ridge(alpha=1.0),
        }[model_type]
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_type == "RIDGE":
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)
        X, y = clean[features], clean[target].astype(float)
        test_count = max(2, int(round(n_samples * .2)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_count, random_state=42)
        folds = min(5, max(2, len(X_train) // 2))
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"})
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        baseline = np.repeat(float(y_train.mean()), len(y_test))
        fitted = pipeline.named_steps["model"]
        raw_importance = (np.abs(fitted.coef_) if model_type == "RIDGE"
                          else fitted.feature_importances_)
        importance = sorted(
            [{"feature": name, "importance": float(value)} for name, value in zip(features, raw_importance)],
            key=lambda row: -row["importance"])
        bundle = {"pipeline": pipeline, "feature_names": features, "target": target,
                  "model_type": model_type, "format_version": 1}
        partial_model = output_dir / "model.joblib.part"
        joblib.dump(bundle, partial_model)
        os.replace(partial_model, model_path)
        prediction_frame = pd.DataFrame({"row_index": y_test.index, "observed": y_test.values,
                                         "predicted": predicted, "baseline_mean": baseline})
        partial_csv = output_dir / "test_predictions.csv.part"
        prediction_frame.to_csv(partial_csv, index=False)
        os.replace(partial_csv, predictions_path)
        metrics = {
            "status": "success", "synthetic_data_used": False, "random_seed": 42,
            "data_csv": str(data_path), "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "model_type": model_type, "target": target, "features_used": features,
            "n_source_rows": len(frame), "n_samples": n_samples,
            "n_train": len(X_train), "n_test": len(X_test),
            "rows_dropped_missing_target": len(frame) - n_samples,
            "duplicate_feature_target_rows": int(clean.duplicated(subset=features + [target]).sum()),
            "cv_folds": folds, "cv_mae_mean": float(-scores["test_mae"].mean()),
            "cv_rmse_mean": float(-scores["test_rmse"].mean()),
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "test_mae": float(mean_absolute_error(y_test, predicted)),
            "test_rmse": float(mean_squared_error(y_test, predicted) ** .5),
            "test_r2": float(r2_score(y_test, predicted)),
            "baseline_test_mae": float(mean_absolute_error(y_test, baseline)),
            "baseline_test_rmse": float(mean_squared_error(y_test, baseline) ** .5),
            "feature_importance": importance, "model_path": str(model_path),
            "output_files": [str(path) for path in outputs],
        }
        write_checkpoint(metadata_path, metrics)
        return metrics
    except Exception as error:
        return {"error": str(error), "traceback": traceback.format_exc()}


def _cif_feature_frame(cif: Path, features: List[str]):
    import pandas as pd
    from dataclasses import asdict, is_dataclass
    from tools.mof_features import extract_features
    value = extract_features(str(cif))
    values = asdict(value) if is_dataclass(value) else dict(value)
    missing = [name for name in features if name not in values or values[name] is None]
    if missing:
        raise ValueError(f"extractor lacks trained features: {missing}")
    return pd.DataFrame([{name: values[name] for name in features}])


def _exec_ml_predict(p: Dict[str, Any]) -> Any:
    """Predict CIFs using the exact persisted training pipeline."""
    try:
        import hashlib
        import joblib
        import pandas as pd
        model_dir = _ml_path(p.get("model_dir", ""), _ws("ml"), output=True)
        model_path = model_dir / "model.joblib"
        if not model_path.is_file():
            return {"error": "model.joblib is missing", "model_dir": str(model_dir)}
        cif_path, cif_dir = p.get("cif_path", ""), p.get("cif_dir", "")
        if bool(cif_path) == bool(cif_dir):
            return {"error": "exactly one of cif_path or cif_dir is required"}
        source = _ml_path(cif_path or cif_dir, "")
        cifs = [source] if cif_path else sorted(source.glob("*.cif"))
        if not cifs:
            return {"error": "no CIF inputs found", "input": str(source)}
        bundle = joblib.load(model_path)
        pipeline, features = bundle["pipeline"], bundle["feature_names"]
        rows, failures = [], []
        for cif in cifs:
            try:
                sample = _cif_feature_frame(cif, features)
                rows.append({"cif": cif.name, "cif_path": str(cif),
                             "predicted": float(pipeline.predict(sample)[0])})
            except Exception as error:
                failures.append({"cif": str(cif), "error": str(error)})
        output = _ml_path(p.get("output_csv", ""), str(model_dir / "predictions.csv"), output=True)
        if output.exists():
            return {"error": "prediction output already exists; choose a fresh output_csv", "output_csv": str(output)}
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".part")
        pd.DataFrame(rows).to_csv(partial, index=False)
        os.replace(partial, output)
        return {"status": "success" if rows else "failed", "n_inputs": len(cifs),
                "n_predicted": len(rows), "predictions": rows, "failures": failures,
                "model_path": str(model_path),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "output_csv": str(output), "output_files": [str(output)]}
    except Exception as error:
        return {"error": str(error), "traceback": traceback.format_exc()}


def _exec_ml_feature_importance(p: Dict[str, Any]) -> Any:
    try:
        model_dir = _ml_path(p.get("model_dir", ""), _ws("ml"), output=True)
        path = model_dir / "model_metadata.json"
        if not path.is_file():
            return {"error": "model_metadata.json is missing", "model_dir": str(model_dir)}
        metadata = json.loads(path.read_text())
        return {"model_type": metadata.get("model_type"), "target": metadata.get("target"),
                "feature_importance": metadata.get("feature_importance", []), "evidence_path": str(path)}
    except Exception as error:
        return {"error": str(error)}


def _exec_ml_active_learning(p: Dict[str, Any]) -> Any:
    """Select candidates without random fallback or fabricated uncertainty."""
    try:
        import joblib
        import numpy as np
        model_dir = _ml_path(p.get("model_dir", ""), _ws("ml"), output=True)
        cif_dir = _ml_path(p.get("cif_dir", ""), "")
        strategy = p.get("strategy", "uncertainty")
        if strategy not in {"uncertainty", "max_prediction"}:
            return {"error": "strategy must be uncertainty or max_prediction; random fallback is forbidden"}
        bundle = joblib.load(model_dir / "model.joblib")
        pipeline, features = bundle["pipeline"], bundle["feature_names"]
        estimator = pipeline.named_steps["model"]
        rows, failures = [], []
        for cif in sorted(cif_dir.glob("*.cif")):
            try:
                sample = _cif_feature_frame(cif, features)
                row = {"cif": cif.name, "cif_path": str(cif),
                       "predicted": float(pipeline.predict(sample)[0])}
                if strategy == "uncertainty":
                    if bundle.get("model_type") != "RF" or not hasattr(estimator, "estimators_"):
                        return {"error": "uncertainty selection requires an RF ensemble; no uncertainty is fabricated"}
                    transformed = pipeline[:-1].transform(sample)
                    member_predictions = [float(tree.predict(transformed)[0]) for tree in estimator.estimators_]
                    row["uncertainty"] = float(np.std(member_predictions, ddof=1))
                rows.append(row)
            except Exception as error:
                failures.append({"cif": str(cif), "error": str(error)})
        key = "uncertainty" if strategy == "uncertainty" else "predicted"
        ranked = sorted(rows, key=lambda row: -row[key])
        selected = ranked[:min(int(p.get("n_select", 5)), len(ranked))]
        result = {"status": "success" if selected else "failed", "strategy": strategy,
                  "n_candidates": len(rows), "selected": selected, "failures": failures,
                  "model_path": str(model_dir / "model.joblib")}
        output = model_dir / "active_learning_selection.json"
        if output.exists():
            return {"error": "selection evidence already exists; use a fresh model_dir", "output_path": str(output)}
        from .state_io import write_checkpoint
        write_checkpoint(output, result)
        result.update(output_path=str(output), output_files=[str(output)])
        return result
    except Exception as error:
        return {"error": str(error), "traceback": traceback.format_exc()}


# ── Schema definitions ──────────────────────────────────────────────

_SCHEMAS = {
    "run_gcmc_isotherm": {
        "description": "Run GCMC adsorption isotherm for a gas in a MOF using RASPA in LOCAL mode: every run copies the force field (force_field.def / pseudo_atoms.def / force_field_mixing_rules.def), the molecule .def and the CIF into the job working folder so RASPA resolves them from ./ (local first). The project-local mirror lives in forcefields/raspa/ — the external RASPA2 share is never modified. If a force field is missing an atom type, fix a COPY inside the job folder, not the mirror.",
        "schema": {
            "type": "object",
            "properties": {
                "cif": {"type": "string", "description": "CIF file path or short name"},
                "gas": {"type": "string", "description": "Gas molecule (CO2, CH4, etc.)"},
                "temperature": {"type": "number", "description": "Temperature in K (e.g. 298 for room temp, 77 for liquid N2)"},
                "pressure_start": {"type": "number", "description": "Start pressure in bar (converted to Pa automatically for RASPA)"},
                "pressure_end": {"type": "number", "description": "End pressure in bar (converted to Pa automatically for RASPA)"},
                "n_pressure_points": {"type": "integer", "description": "Number of pressure points"},
                "unit_cells": {"type": "string", "description": "Unit cell multiplication (2x2x2)"},
                "electrostatics_method": {"type": "string", "description": "RASPA electrostatics summation for charges already present in the framework/guest model: Ewald (default) or Wolf. This does NOT assign framework charges. If the CIF lacks _atom_site_charge, reuse a verified charged CIF or call run_pacman_charge (PACMOF default; PACMAN for higher-accuracy DDEC6/CM5). Do not switch this field to repair missing charges.", "enum": ["Ewald", "Wolf"], "default": "Ewald"},
                "force_field": {"type": "string", "description": "Framework (MOF) force field, selected from forcefields/raspa/forcefield/. Options: GenericMOFs (most gases), UFF (CO2/CH4 only), TraPPEForceField, DREIDING, wbao. If omitted, auto-selected per gas."},
                "guest_ff": {"type": "string", "description": "Guest molecule definition source, from forcefields/raspa/molecules/. Options: TraPPE, ExampleDefinitions. If omitted, auto-selected per gas (e.g. CO2→TraPPE, CH4→ExampleDefinitions)."},
                "cycles": {"type": "integer", "description": "MC cycles (5000-10000 for screening, 50000-100000 for production)"},
            },
            "required": ["cif", "gas", "temperature", "pressure_start", "pressure_end"],
        },
    },
    "run_gcmc_batch": {
        "description": "Run GCMC batch screening for multiple CIFs as ONE SLURM job in LOCAL mode: every (CIF × gas) pair is processed INSIDE the single sbatch script (serial loop). 10 MOFs = exactly 1 job — NEVER loop this tool per-CIF (that would create N separate jobs); always pass the whole cif_dir. Each pair gets the force field, molecule .def and CIF copied into its own working folder (RASPA resolves them locally). Project-local FF mirror: forcefields/raspa/. External RASPA2 share is never modified.",
        "schema": {
            "type": "object",
            "properties": {
                "cycles": {"type": "integer", "minimum": 1, "description": "MC cycles passed to the batch submitter"},
                "cif_dir": {"type": "string", "description": "Directory with CIF files"},
                "gases": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Gas list"},
                "temperature": {"type": "number", "description": "Temperature in K"},
                "pressure": {"type": "number", "description": "Pressure in bar"},
                "force_field": {"type": "string", "description": "Optional force field applied to all (CIF, gas) pairs, from forcefields/raspa/forcefield/ (GenericMOFs, UFF, TraPPEForceField, wbao...). If omitted, per-gas preset default is used."},
                "unit_cells": {"type": "string", "description": "Unit cell multiplication applied to every (CIF, gas) pair, e.g. '2 2 2'. Use smaller (e.g. '1 1 1') for very large unit cells / memory limits; the actual value is written into each simulation.input UnitCells line."},
                "electrostatics_method": {"type": "string", "description": "RASPA electrostatics summation for charges already present in every framework/guest model: Ewald (default) or Wolf. It is not a charge assignment method. Missing framework charges require charged CIF inputs or run_pacman_charge, not a solver switch.", "enum": ["Ewald", "Wolf"], "default": "Ewald"},
                "output_dir": {"type": "string", "description": "Session-scoped branch directory. Give every independent force-field/temperature branch a distinct directory so those branches may run in parallel. Existing scientific results are never overwritten implicitly."},
            },
            "required": ["cif_dir", "gases", "temperature", "pressure"],
        },
    },
    "run_henry": {
        "description": "Calculate Henry coefficient and heat of adsorption in LOCAL mode (force field + molecule .def + CIF copied into the job folder; external RASPA share untouched).",
        "schema": {
            "type": "object",
            "properties": {
                "cif": {"type": "string", "description": "CIF file path"},
                "gas": {"type": "string", "description": "Gas molecule"},
                "temperature": {"type": "number", "description": "Temperature in K"},
            },
            "required": ["cif", "gas", "temperature"],
        },
    },
    "run_pore_analysis": {
        "description": "Zeo++ 孔结构特征计算（比表面积/孔体积/孔径/空隙率）。必填：cif_path 或 cif_dir（二选一）。可选：probe_radius（默认1.525 Å=He，CO2用1.65）、n_samples（默认5000）、output_csv。一个调用=一个SLURM job，提交后立即返回job_id与产物CSV路径（{work_dir}/pore_results.csv），用 check_job 查状态、用 inspect_run 读CSV。批量CIF时仍是单job（内部xargs并行）。",
        "schema": {
            "type": "object",
            "properties": {
                "cif_path": {"type": "string", "description": "单个 CIF 路径（与 cif_dir 二选一，只给一个）"},
                "cif_dir": {"type": "string", "description": "含 .cif 的目录，批量计算（内部一个 job）"},
                "probe_radius": {"type": "number", "description": "探针半径 Å（默认1.525=He；CO2 用 1.65）"},
                "n_samples": {"type": "integer", "description": "MC 采样数（默认 5000）"},
                "output_csv": {"type": "string", "description": "结果 CSV 路径（默认 {work_dir}/pore_results.csv）"},
            },
        },
    },
    "run_pacman_charge": {
        "description": (
            "Assign framework partial charges to MOF CIFs. "
            "method='pacmof' (DEFAULT): fast CPU ML charges (PACMOF RF, DDEC6-trained) "
            "via SLURM compute partition — minutes, no GPU; outputs <name>_pacmof.cif. "
            "method='pacman': accurate but slow GPU DDEC6/CM5 (PACMAN-charge, 12h); "
            "outputs <name>_pacman.cif. Both produce CIFs with a real _atom_site_charge "
            "column that cDFT/GCMC consume."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "cif_dir": {"type": "string", "description": "CIF directory"},
                "cif_path": {"type": "string", "description": "Single CIF path (alternative to cif_dir)"},
                "method": {"type": "string", "enum": ["pacmof", "pacman"], "default": "pacmof", "description": "'pacmof' (default, fast CPU) or 'pacman' (GPU DDEC6/CM5)"},
                "charge_type": {"type": "string", "description": "DDEC6 or CM5 (only used when method='pacman')"},
                "scheduler": {"type": "string", "description": "local, pbs, remote_slurm"},
                "output_dir": {"type": "string", "description": "Output directory"},
            },
        },
    },
    "run_md_optimize": {
        "description": "Submit SLURM LAMMPS framework geometry minimization (single) or framework NPT relaxation (md). This wrapper does NOT insert guest molecules and does NOT emit an unwrapped trajectory, so it cannot be used for guest diffusion/MSD. MD requires explicit temperature; unknown arguments such as gas are rejected.",
        "schema": {
            "type": "object",
            "properties": {
                "cif_path": {"type": "string", "description": "CIF file path"},
                "mode": {"type": "string", "enum": ["single", "md"]},
                "force_field": {"type": "string", "description": "Force field name (auto-selected per gas if omitted). Only specify if you need to override."},
                "cutoff": {"type": "number", "exclusiveMinimum": 0, "description": "Cutoff in Å"},
                "temperature": {"type": "number", "description": "NPT temperature"},
                "pressure": {"type": "number", "minimum": 0, "description": "NPT pressure in atm (not bar)"},
                "npt_ps": {"type": "number", "exclusiveMinimum": 0, "description": "NPT production time in ps"},
                "timestep": {"type": "number", "exclusiveMinimum": 0, "description": "MD timestep in fs"},
            },
            "required": ["cif_path"],
            "additionalProperties": False,
        },
    },
    "generate_structure": {
        "description": "Generate MOF/COF/HOF structures using the project-local pormake generator. Supports exact topology lists and max-atom constraints; one call submits one SLURM job.",
        "schema": {
            "type": "object",
            "properties": {
                "material_type": {"type": "string", "description": "MOF, COF, etc."},
                "n_structures": {"type": "integer", "description": "Number to generate"},
                "output_dir": {"type": "string", "description": "Output directory"},
                "max_atoms": {"type": "integer", "description": "Max atoms per cell"},
                "topologies": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Optional exact topology list, e.g. [utk,hxg,bto,cds,cdt,srs,eta]"},
                "structures_per_topology": {"type": "integer", "description": "Optional count per topology; otherwise distributed from n_structures"},
                "nodelist":{"type":"string","pattern":"^[A-Za-z0-9_.-]+$","description":"Actual scheduler node from fresh resource facts."},
                "partition":{"type":"string","pattern":"^[A-Za-z0-9_.-]+$"},
                "cpus_per_task":{"type":"integer","minimum":1},
                "walltime":{"type":"string","pattern":"^(?:[0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$"},
                "memory_mb":{"type":"integer","minimum":64,"maximum":2147483647},
            },
            "required": ["material_type", "n_structures", "output_dir"],
        },
    },
    "run_xtb_optimize": {
        "description": "Run xTB semiempirical optimization on molecules/clusters.",
        "schema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Input XYZ/MOL file"},
                "output_dir": {"type": "string", "description": "Output directory"},
                "gfn": {"type": "integer", "enum": [0, 1, 2]},
                "opt_level": {"type": "string", "enum": ["loose", "normal", "tight"]},
                "charge": {"type": "integer", "description": "System charge"},
            },
            "required": ["input_path", "output_dir"],
        },
    },
    "build_guest_forcefield": {
        "description": "Build a RASPA guest molecule force field package.",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Molecule name"},
                "output_dir": {"type": "string", "description": "Output directory"},
                "input_path": {"type": "string", "description": "Optional input file"},
            },
            "required": ["name", "output_dir"],
        },
    },
    "run_cdft": {
        "description": (
            "Run classical DFT (cDFT) fluid-in-pore calculation using the SDK-LOCAL "
            "cDFT installation (tools/cdft/cDFT_Initialization) — DM_cdft binary, UFF / "
            "coarsening force fields and molecule defs all resolve locally; the external "
            "High_Batch_tool copy is never modified. ALL cDFT compute is submitted to SLURM "
            "(ONE sbatch job runs DM_cdft per MOF in parallel; 10 MOFs = exactly 1 job — "
            "NEVER call this tool once per MOF, pass the whole cif_dir). Actions: inputs "
            "(generate .dat inputs from a cif_dir), submit (sbatch-submit the DM_cdft batch "
            "on an input_dir), collect (parse results.csv from job_work_dir), pipeline "
            "(inputs + submit).\n"
            "Input generation AUTOMATICALLY: ① audits the box (min(a,b,c) > 2×cutoff) and "
            "supercell-expands too-small cells; ② audits framework charges GAS-AWARE — "
            "polar gases (CO2/SO2/CO/N2/…) require charged frameworks (structures must have "
            "real _atom_site_charge from run_pacman_charge; uncharged CIFs are rejected with "
            "a clear error), while apolar gases (CH4/H2/…) pass with neutral frameworks. "
            "Pass framework_charge=True/False explicitly to override the gas-based decision."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["pipeline", "submit", "collect", "inputs"]},
                "cif_dir": {"type": "string", "description": "Directory of CIF files for cDFT input generation (or cif_path)"},
                "cif_path": {"type": "string"},
                "gas": {"type": "string"},
                "gases": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "temperature": {"type": "number"},
                "bulk_densities": {"type": "array", "items": {"type": "number"}, "description": "Optional per-component densities, order matching gases. Default unit mol/L; SDK converts to native molecule/angstrom^3 before input generation. gases with multiple entries are SIMULTANEOUS mixture components, not separate pure-gas jobs."},
                "bulk_density_unit":{"type":"string","enum":["mol/L","molecule/angstrom^3"],"description":"Explicit density unit. Default mol/L for supplied values; native tables/density defaults are molecule/angstrom^3. Never mix these units."},
                "framework_charge": {"type": "boolean", "description": "Override framework-charge requirement: True=require charged CIFs (polar gases CO2/SO2/CO/N2), False=allow neutral frameworks (apolar gases CH4/H2). Omitted → auto by gas."},
                "input_dir": {"type": "string", "description": "Directory of generated .dat inputs (submit/collect)"},
                "job_work_dir": {"type": "string", "description": "Explicit isolated preparation workspace for inputs/pipeline/submit; for collect use the actual job work_dir returned by submit. Independent gas branches MUST use disjoint workspaces and match input_dir exactly across inputs and submit."},
                "output": {"type": "string", "description": "Output CSV path for collect"},
                "nodelist": {"type": "string", "minLength": 1, "pattern": "^[A-Za-z0-9_.,\\[\\]-]+$", "description": "Optional actual compatible compute node from resource_health.node_inventory, with matching partition; otherwise SDK selects one verified compatible node via env/node_policy.json."},
                "memory_mb": {"type":"integer","minimum":64,"maximum":2147483647,"description":"Finite job RAM budget in Slurm MiB, required for real pipeline/submit. Use workload evidence or user approval; do not guess small RAM to eliminate queueing."},
                "resource_review_id": {"type":"string","minLength":1,"description":"SDK-assigned evidence-backed resource review receipt."},
                "partition": {"type": "string"},
                "num_processes": {"type": "integer", "minimum": 1},
                "walltime": {"type": "string", "pattern": "^[0-9]+:[0-5][0-9]:[0-5][0-9]$"},
            },
            "required": ["action"],
        },
    },
    "calc_binding_energy": {
        "description": "Calculate adsorbate-framework binding energy at specific sites.",
        "schema": {
            "type": "object",
            "properties": {
                "cif_path": {"type": "string", "description": "CIF file path"},
                "gas": {"type": "string", "description": "Gas molecule"},
                "site_atoms": {"type": "array", "items": {"type": "string"}},
                "distance": {"type": "number", "description": "Distance in Å"},
            },
            "required": ["cif_path", "gas"],
        },
    },
    "run_vasp": {
        "description": "Submit VASP from an existing prepared work_dir. Setup/setup-batch/parse are not implemented and will not launch computations.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["submit"]},
                "work_dir": {"type": "string"},
            },
            "required": ["action", "work_dir"],
        },
    },
    "run_string_tst": {
        "description": "Run String method transition state search for diffusion barriers.",
        "schema": {
            "type": "object",
            "properties": {
                "cif_dir": {"type": "string"},
                "output_dir": {"type": "string"},
                "gas": {"type": "string"},
                "gases": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "temperature": {"type": "number"},
            },
            "required": ["cif_dir", "output_dir", "gas"],
        },
    },
    "run_external_potential": {
        "description": "Run TuSraST external potential calculations for diffusion.",
        "schema": {
            "type": "object",
            "properties": {
                "cif_dir": {"type": "string"},
                "output_dir": {"type": "string"},
                "gas": {"type": "string"},
                "temperature": {"type": "number"},
            },
            "required": ["cif_dir", "output_dir"],
        },
    },
    "check_job": {
        "description": "Check PBS/Slurm job status. Returns status, whether it terminated, whether it FAILED, and stderr tail if it failed.",
        "schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "scheduler": {"type": "string", "description": "pbs, slurm, remote_slurm"},
                "work_dir": {"type": "string", "description": "Work directory of the job (to read stderr logs)"},
                "host": {"type": "string", "description": "SLURM host"},
            },
            "required": ["job_id"],
        },
    },
    "list_my_jobs": {
        "description": "List SLURM jobs submitted by THIS conversation (JobWatch, conv-scoped). Use this (NOT bare squeue) whenever you need to report 'my submitted jobs' or check their status — squeue shows ALL users'/conversations' jobs and would make you report OTHER tasks' jobs as your own.",
        "schema": {
            "type": "object",
            "properties": {},
        },
    },
    "diagnose_job": {
        "description": "Diagnose a FAILED/CANCELLED/TIMEOUT SLURM job. Returns structured report: state, exit code, stderr/stdout tails, and suspected failure causes. Call this BEFORE retrying any failed computation job.",
        "schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "SLURM job ID to diagnose"},
                "work_dir": {"type": "string", "description": "Job work directory (to read .err/.out logs)"},
                "host": {"type": "string", "description": "SLURM host (default from config)"},
            },
            "required": ["job_id"],
        },
    },
    "submit_job": {
        "description": "Submit a raw SLURM job with an arbitrary shell command. For GCMC/adsorption jobs use run_gcmc_isotherm instead. Returns job_id and the work_dir used.",
        "schema": {
            "type": "object",
            "properties": {
                "job_name": {"type": "string", "description": "SLURM job name (default 'generic_job')"},
                "command": {"type": "string", "description": "Shell command the job runs. IMPORTANT: pass the FULL command as ONE string, including any semicolons/redirects. Example: 'cd /scratch && python run.py > out.log 2>&1'"},
                "work_dir": {"type": "string", "description": "Directory for the submit.sh + job logs (created if missing). Default: project tmp/"},
                "mode": {"type": "string", "enum": ["local", "remote", "auto"], "description": "local = sbatch on this node; remote = ssh to SLURM host; auto = local first then remote. Default 'local'."},
                "partition": {"type": "string", "description": "SLURM partition (default 'compute')"},
                "cpus_per_task": {"type": "integer", "description": "CPUs per task (default 4)"},
                "nodelist": {"type": "string", "minLength": 1, "pattern": "^[A-Za-z0-9_.,\\[\\]-]+$", "description": "Optional scheduler node constraint; select only from fresh resource_health.node_inventory facts and matching partition. Never use the login node for compute."},
                "walltime": {"type": "string", "description": "Walltime, e.g. '04:00:00'"},
            },
            "required": ["command"],
        },
    },
    "query_literature": {
        "description": "查文献/参数/基准值：先查本地文献库，本地没命中时自动用 Crossref+arXiv web 检索补充"
                       "（返回 source=rag/web/rag+web）。适用于 SO2 力场参数、选择性基准值等本地库缺的题目。",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "n_results": {"type": "integer", "description": "Number of results (default 5)"},
                "query_type": {"type": "string", "description": "auto, parameter, mechanism"},
                "web_fallback": {"type": "boolean", "description": "Use web fallback if local hits are insufficient (default true)"},
                "require_both": {"type": "boolean", "description": "Always query both local RAG and Crossref/arXiv Web. Returns coverage and evidence_complete so professional conclusions can disclose partial evidence."},
            },
            "required": ["query"],
        },
    },
    "stage_cif_subset": {
        "description": "Deterministically stage an exact filename-sorted CIF subset into the current session and write a SHA-256 manifest. Use this before batch tools when the scientific scope is first/next N structures; never pass the larger source directory downstream.",
        "schema": {"type": "object", "properties": {
            "cif_dir": {"type": "string", "minLength": 1},
            "output_dir": {"type": "string", "minLength": 1, "description": "New session-owned subset directory"},
            "limit": {"type": "integer", "minimum": 1},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "pattern": {"type": "string", "default": "*.cif"}},
            "required": ["cif_dir", "output_dir", "limit"]},
    },
    "analyze_gcmc_screening": {
        "description": "Parse completed real run_gcmc_batch RASPA .data evidence, extract absolute loading in mol/kg-framework with uncertainty, calculate a declared single-component uptake ratio for the first two gases, rank materials, and atomically write CSV+Markdown. This ratio is explicitly not IAST selectivity. Fails closed on missing/duplicate/unparseable results; never emits placeholders.",
        "schema": {"type": "object", "properties": {
            "work_dir": {"type": "string", "minLength": 1},
            "gases": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 2, "uniqueItems": True},
            "output_csv": {"type": "string", "minLength": 1},
            "output_markdown": {"type": "string", "minLength": 1}},
            "required": ["work_dir", "gases", "output_csv", "output_markdown"]},
    },
    "inspect_path": {
        "description": "Inspect a file or directory to understand its contents.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_entries": {"type": "integer", "description": "Max entries (default 20)"},
            },
            "required": ["path"],
        },
    },
    "inspect_run": {
        "description": "Inspect a recorded workflow run by run_id. If run_id is a numeric SLURM job id, returns job status instead.",
        "schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "project": {"type": "string"},
                "work_dir": {"type": "string", "description": "Work directory for numeric SLURM job ids"},
            },
            "required": ["run_id"],
        },
    },
    "find_cif": {
        "description": "Resolve a CIF file path by name or pattern.",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "CIF name to search"},
            },
            "required": ["name"],
        },
    },
    "download_scientific_file": {
        "description": "Download one public scientific artifact over HTTPS into the current private session. Enforces redirect-by-redirect SSRF checks, size limits, optional SHA256 verification, atomic publication, and returns provenance metadata. Never use run_bash/curl for downloads.",
        "schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "pattern": "^https://", "description": "Direct public HTTPS artifact URL"},
                "output_path": {"type": "string", "description": "Destination inside runs/<user>/<session>/"},
                "expected_sha256": {"type": "string", "pattern": "^[0-9A-Fa-f]{64}$"},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 104857600, "default": 20971520},
                "timeout_seconds": {"type": "number", "minimum": 1, "maximum": 120, "default": 30},
            },
            "required": ["url", "output_path"],
        },
    },
    "analyze_diffusion_msd": {
        "description": "Compute ensemble mean-square displacement and a diffusion coefficient from a CSV or LAMMPS dump trajectory. Requires unwrapped coordinates (xu/yu/zu or x/y/z plus image flags), fits an explicit time window, converts A^2/ps to m^2/s, and reports R2/block uncertainty.",
        "schema": {
            "type": "object",
            "properties": {
                "trajectory_path": {"type": "string"},
                "format": {"type": "string", "enum": ["auto", "csv", "lammps_dump"], "default": "auto"},
                "timestep_ps": {"type": "number", "exclusiveMinimum": 0, "description": "LAMMPS timestep multiplier in ps; CSV already supplies time_ps but a positive value is still required for an explicit unit contract"},
                "dimensions": {"type": "integer", "enum": [1, 2, 3], "default": 3},
                "atom_ids": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
                "fit_start_fraction": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2},
                "fit_end_fraction": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
                "output_csv": {"type": "string", "description": "MSD table inside runs/<user>/<session>/"},
            },
            "required": ["trajectory_path", "timestep_ps"],
        },
    },
    "extract_features": {
        "description": "Extract deterministic composition/cell features from exactly one CIF or one CIF directory. Batch CSV defaults to the current session and is atomically published with per-file failures and provenance.",
        "schema": {
            "type": "object",
            "properties": {
                "cif_path": {"type": "string"},
                "cif_dir": {"type": "string"},
                "output_csv": {"type": "string"},
                "n_threads": {"type": "integer", "description": "Parallel threads (default 4)"},
            },
        },
    },
    "ml_train": {
        "description": "Train a reproducible GBR/RF/Ridge surrogate only from an explicit real labelled CSV. Missing data, missing targets and insufficient samples fail closed; synthetic/random labels are never generated. Returns held-out and cross-validation metrics plus provenance.",
        "schema": {
            "type": "object",
            "properties": {
                "data_csv": {"type": "string", "description": "Training data CSV with features + target"},
                "model_type": {"type": "string", "enum": ["GBR", "RF", "Ridge"], "description": "GBR, RF, or Ridge"},
                "target": {"type": "string", "description": "Target column name (e.g. uptake,Henry)"},
                "features": {"type": "array", "items": {"type": "string"}, "description": "Feature columns"},
                "output_dir": {"type": "string", "description": "Output directory for model"},
            },
            "required": ["data_csv"],
        },
    },
    "ml_predict": {
        "description": "Predict from CIF-derived features using the exact persisted validated training pipeline; missing trained features fail per material and are reported.",
        "schema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Directory with trained model"},
                "cif_path": {"type": "string", "description": "Single CIF to predict"},
                "cif_dir": {"type": "string", "description": "Directory of CIFs to predict"},
                "output_csv": {"type": "string", "description": "Output CSV"},
            },
            "required": ["model_dir"],
        },
    },
    "ml_feature_importance": {
        "description": "Get feature importance ranking from trained ML model.",
        "schema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Directory with trained model"},
            },
            "required": ["model_dir"],
        },
    },
    "ml_active_learning": {
        "description": "Select candidates by RF ensemble uncertainty or maximum prediction. Random fallback and fabricated uncertainty are forbidden.",
        "schema": {
            "type": "object",
            "properties": {
                "cif_dir": {"type": "string", "description": "Directory of candidate CIFs"},
                "model_dir": {"type": "string", "description": "Trained model directory"},
                "n_select": {"type": "integer", "description": "Number of candidates to select"},
                "strategy": {"type": "string", "enum": ["uncertainty", "max_prediction"], "description": "uncertainty requires an RF ensemble"},
                "gas": {"type": "string"},
                "temperature": {"type": "number"},
            },
            "required": ["cif_dir", "model_dir"],
        },
    },
    "run_ga_optimization": {
        "description": "DISABLED legacy GA: historical implementation used estimated/random features and a hand-coded selectivity formula. Use real-label ml_train + RF ml_active_learning + batched simulation validation instead.",
        "schema": {
            "type": "object",
            "properties": {
                "target_selectivity": {"type": "number", "description": "Target CO2/N2 selectivity (required, must have scientific basis)"},
                "target_uptake": {"type": "number", "description": "Target CO2 uptake in mmol/g (required, must have scientific basis)"},
                "pop_size": {"type": "integer", "description": "GA population size (default 20)"},
                "n_generations": {"type": "integer", "description": "Number of GA generations (default 10)"},
                "output_dir": {"type": "string", "description": "Output directory for results"},
                "ml_model_dir": {"type": "string", "description": "Directory containing trained ML model"},
            },
            "required": ["target_selectivity", "target_uptake"],
        },
    },
    "validate_ga_result": {
        "description": "Validate GA optimization results - check if ML model was used, prediction rate, and result quality.",
        "schema": {
            "type": "object",
            "properties": {
                "ga_result_dir": {"type": "string", "description": "Directory containing ga_result.json"},
            },
            "required": ["ga_result_dir"],
        },
    },
    "build_mof_database": {
        "description": "Build a real, session-scoped CIF inventory from a supplied directory. Parses composition/cell/charge metadata, records SHA256 provenance, detects byte-identical duplicates, reports failures, and atomically writes JSON+CSV. It never injects hard-coded example materials or properties.",
        "schema": {
            "type": "object",
            "properties": {
                "cif_dir": {"type": "string", "description": "Source directory containing real CIF files"},
                "output_dir": {"type": "string", "description": "Destination inside runs/<user>/<session>/"},
                "recursive": {"type": "boolean", "default": False},
                "max_files": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 1000},
            },
            "required": ["cif_dir"],
        },
    },
    "validate_method": {
        "description": "Validate computational method correctness - check parameters, convergence, and physics.",
        "schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "description": "Method name (GCMC, ML, GA, cDFT, etc.)"},
                "parameters": {"type": "object", "description": "Method parameters to validate"},
            },
            "required": ["method"],
        },
    },
    "check_conclusion_reliability": {
        "description": "Check if conclusion has statistical support and is reliable.",
        "schema": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "number"}, "description": "Numerical data supporting the conclusion"},
                "conclusion": {"type": "string", "description": "The conclusion to verify"},
            },
            "required": ["data", "conclusion"],
        },
    },
    "verify_data_authenticity": {
        "description": "Verify that data comes from real calculations, not random generation.",
        "schema": {
            "type": "object",
            "properties": {
                "data_source": {"type": "string", "description": "Source of the data (file path, calculation method)"},
                "feature_values": {"type": "array", "items": {"type": "number"}, "description": "Feature values to check for randomness"},
            },
            "required": [],
        },
    },
    "read_file": {
        "description": "Read the contents of a file. Use offset/limit to page through long files.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or project-relative file path"},
                "offset": {"type": "integer", "description": "Line number to start reading from (0-based)"},
                "limit": {"type": "integer", "description": "Max lines to read (0 = all)"},
                "max_chars": {"type": "integer", "description": "Max characters to return (default 20000)"},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "description": "Write content to a file, creating parent directories if needed.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or project-relative file path"},
                "content": {"type": "string", "description": "Full content to write"},
                "mode": {"type": "string", "enum": ["write", "append"], "description": "write=overwrite (default), append=append to end"},
            },
            "required": ["path", "content"],
        },
    },
    "run_bash": {
        "description": "Execute a shell command and return stdout/stderr/exit code. Use for file inspection, data prep, quick scripts. ⚠️ 禁止在登录节点直接运行科学计算二进制（DM_cdft/raspa/zeo++/xtb/pacman 等）——此类命令会被拦截，必须改用 run_gcmc_*/run_cdft/submit_job 提交 SLURM 作业到计算节点。诊断二进制依赖请用 ldd/file 等只读命令。",
        "schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)"},
                "cwd": {"type": "string", "description": "Working directory (default: project root)"},
                "max_output": {"type": "integer", "description": "Max output chars per stream (default 20000)"},
            },
            "required": ["command"],
        },
    },
    "restart_backend": {
        "description": (
            "Schedule a backend (uvicorn api:app) restart so framework code "
            "edits take effect. It runs AFTER the current agent turn finishes "
            "writing its reply (the current turn is never interrupted). Use it "
            "after patching api.py / agents/*.py / registry.py / defns.py. "
            "Returns immediately with 'restart_scheduled': True."
        ),
        "schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "expand_cell": {
        "description": (
            "Supercell-expand CIFs so the box satisfies the minimum-image rule "
            "min(a,b,c) > 2×cutoff. Pure text processing (no SLURM). Writes "
            "expanded (or verbatim-copied) CIFs into output_dir and reports "
            "which structures were expanded. Use it BEFORE cDFT / RASPA GCMC "
            "when you want the SAME expanded cell fed to both (both have the "
            "identical 2×cutoff truncation rule); the cDFT Input.dat generator "
            "also auto-expands inline."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "cif_dir": {"type": "string", "description": "Directory of CIF files to expand"},
                "cutoff": {"type": "number", "description": "Interaction cutoff in Å; box must be > 2×cutoff (default 15.0 to match cDFT cut_coul)"},
                "output_dir": {"type": "string", "description": "Where expanded CIFs are written (default: <cif_dir>/expanded)"},
            },
            "required": ["cif_dir"],
        },
    },
    "grep_search": {
        "description": "Search file contents for a regex pattern, returning matching lines with file:line.",
        "schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search (default: project root)"},
                "include": {"type": "string", "description": "Only search matching filenames, e.g. '*.py'"},
                "max_results": {"type": "integer", "description": "Max matches to return (default 50)"},
            },
            "required": ["pattern"],
        },
    },
    "record_workflow_draft": {
        "description": "Persist the complete research DAG from inputs through validation/analysis and the user's final deliverable before presenting or compiling it. Never save only the next runnable step. Drafts never execute; use propose_workflow_patch once to compile every draft node, then use versioned patches only for real runtime repairs.",
        "schema": {"type":"object","properties":{"completion_criteria":{"type":"string","minLength":1,"description":"Concrete final artifact/conclusion that makes the whole user request complete, not merely the next stage."},"nodes":{"type":"array","minItems":1,"items":{
            "type":"object","properties":{"step_id":{"type":"string","minLength":1},"description":{"type":"string","minLength":1},
            "agent":{"type":"string","minLength":1},"tool":{"type":"string","minLength":1},"depends_on":{"type":"array","items":{"type":"string"}},
            "arguments":{"type":"object"},"missing_parameters":{"type":"array","items":{"type":"string"}},
            "input_bindings":INPUT_BINDINGS_SCHEMA,
            "expected_outputs":{"type":"array","items":OUTPUT_SCHEMA},"resource_locks":{"type":"array","items":{"type":"string"}}},
            "required":["step_id","description","agent","depends_on"],"additionalProperties":False}}},"required":["completion_criteria","nodes"],"additionalProperties":False}},
    "task_line_query": {
        "description": "查询流水线任务线（TaskLine）：当前会话各步骤的工具、job_ids、输入/输出产物目录、done(BOOL)。"
                       "下游 agent 在做下一步前用它拿到上一步产物路径（如赋电荷的 *_pacmof.cif 目录），"
                       "只允许当前用户当前会话；不传ID时查询当前会话，不列出其他用户/会话。",
        "schema": {
            "type": "object",
            "properties": {
                "line_id": {"type": "string", "description": "任务线ID（默认当前会话线）"},
                "conv_id": {"type": "string", "description": "按会话查所有线"},
            },
        },
    },
    "task_line_update": {
        "description": "Update one existing TaskLine workflow node with evidence. Use for non-job steps (literature synthesis, analysis, report) or explicit failure/user-wait state. completed requires evidence/output_files/job_ids.",
        "schema": {
            "type": "object",
            "properties": {
                "line_id": {"type": "string", "description": "Defaults to current conversation workflow"},
                "step_id": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "ready", "running", "submitted", "completed", "failed", "needs_user", "superseded", "unconfirmed"]},
                "evidence": {"type": "string", "description": "Concrete result, file path, numbers, or reason"},
                "output_files": {"type": "array", "items": {"type": "string"}},
                "job_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["step_id", "status"],
        },
    },
    "recovery_state": {
        "description": "Read durable submission attempts and recent diagnostic/verification call IDs. A failed calculation cannot be resubmitted until prepare_retry grants a one-use permit.",
        "schema": {"type": "object", "properties": {}},
    },
    "prepare_retry": {
        "description": "Record structured failure reflection and request ONE resubmission permit. Requires real diagnostic and successful verification call IDs acquired AFTER failure, a concrete fix, and exact corrected tool arguments. Does not submit a job. UNKNOWN/job submission success alone are not evidence.",
        "schema": {
            "type": "object",
            "properties": {
                "recovery_key": {"type": "string"},
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"},
                "diagnosis": {"type": "string", "minLength": 10},
                "diagnosis_call_id": {"type": "string"},
                "fix_summary": {"type": "string", "minLength": 10},
                "verification_call_id": {"type": "string"},
            },
            "required": ["recovery_key", "tool_name", "arguments", "diagnosis", "diagnosis_call_id", "fix_summary", "verification_call_id"],
        },
    },
    "accept_recovered_result": {
        "description": "Close a recovery only AFTER scheduler completion and fresh result validation. Reference the actual successful output verification call ID; submission receipts are not result evidence.",
        "schema": {"type": "object", "properties": {
            "recovery_key": {"type": "string"}, "verification_call_id": {"type": "string"},
            "conclusion": {"type": "string", "minLength": 10},
        }, "required": ["recovery_key", "verification_call_id", "conclusion"]},
    },
    "lifecycle_state": {
        "description": "Read authoritative current workflow/runtime, pending patch, lifetime roles and a bounded recent-event index. Historical events are not applied plans; do not reconstruct the current chain from old proposals.",
        "schema": {"type": "object", "properties": {}},
    },
    "assess_job_resources": {
        "description":"Read this session's actual scheduler constraints/pending reason and node CPU/memory facts; never cancel/update/resubmit.",
        "schema":{"type":"object","properties":{"job_id":{"type":"string","minLength":1}},"required":["job_id"],"additionalProperties":False}},
    "resource_review_decision": {
        "description":"Deliver an evidence-backed resource proposal to main chat, not submission or mutation authorization.",
        "schema":{"type":"object","properties":{"status":{"type":"string","enum":["ready","wait_existing","needs_user","propose_resource_patch"]},
            "reason":{"type":"string","minLength":1},"evidence_call_ids":{"type":"array","items":{"type":"string"},"minItems":1},
            "suggested_resources":{"type":"object","properties":{"memory_mb":{"type":"integer","minimum":64},"num_processes":{"type":"integer","minimum":1},"nodelist":{"type":"string"},"partition":{"type":"string"},"walltime":{"type":"string"}},"additionalProperties":False}},
            "required":["status","reason","evidence_call_ids"],"additionalProperties":False}},
    "build_project_frontend": {"description":"Fixed maintenance action compiling this project's frontend; no arbitrary command, job submission or file target.","schema":{"type":"object","properties":{},"additionalProperties":False}},
    "cancel_watched_job": {"description": "Main chat only: explicitly user-authorized cancellation of this session's tracked job. Bypasses data-path lease only, serializes scheduler control, verifies termination before marking CANCELLED. Node-change requests are not cancellation authorization.",
        "schema": {"type": "object", "properties": {"job_id": {"type": "string", "pattern": "^[0-9]+$"}}, "required": ["job_id"], "additionalProperties": False}},
    "apply_workflow_patch": {"description":"Main chat chooses to apply an existing structured patch after understanding real user instructions/decision, not a bool classifier. Checks user-turn provenance and exact pending/base versions, preserves audit history, activates compiled nodes. Never invent user science choices.",
        "schema":{"type":"object","properties":{"plan_version":{"type":"integer","minimum":1}},"required":["plan_version"],"additionalProperties":False}},
    "discard_workflow_patch": {"description": "Main chat withdraws the exact current candidate proposal after understanding the user or finding it invalid. Does not cancel jobs, reset nodes or change the approved executable chain. Retains the proposal and reason as audit history; this is an operation, not an intent classifier.",
        "schema": {"type": "object", "properties": {"plan_version": {"type": "integer", "minimum": 1}, "reason": {"type": "string", "minLength": 5}}, "required": ["plan_version", "reason"], "additionalProperties": False}},
    "revalidate_workflow_node_outputs":{"description":"Repair stopped native output contracts without resubmission: generator uses owned completed job and full fresh CIF coverage; cDFT uses the actual scheduler receipt CSV within its approved calculation root, never an arbitrary output. Restoring a historical generator result uses completed_job_id. Does not mark scientific success; prefinish still requires finish_workflow_node evidence.",
        "schema":{"type":"object","properties":{"step_id":{"type":"string","minLength":1},"expected_outputs":{"type":"array","items":OUTPUT_SCHEMA,"minItems":1},"completed_job_id":{"type":"string","pattern":"^[0-9]+$","description":"Optional restore of an archived verified generator result after all newer attempts are confirmed stopped. Original file identities are frozen from actual historical proof; no replay or inclusion of extra files."}},"required":["step_id","expected_outputs"],"additionalProperties":False}},
    "finish_workflow_node":{"description":"Complete a prefinish node only after the scientific owner has checked actual results/conditions with fresh real tools. Requires scoped evidence IDs and a factual validation conclusion; code rechecks artifacts. This is a completion operation, not a classifier bool. Never treat job exit alone as scientific success.",
        "schema":{"type":"object","properties":{"step_id":{"type":"string","minLength":1},"evidence_call_ids":{"type":"array","items":{"type":"string"},"minItems":1},"conclusion":{"type":"string","minLength":10}},"required":["step_id","evidence_call_ids","conclusion"],"additionalProperties":False}},
    "retarget_queued_job":{"description":"Main chat scheduling control for an authorized node/resource move: update the same owned PENDING job, verify CPU/RAM/partition and journal receipt. memory_mb may only equal the operator-owned profile for that tool. Never cancel/resubmit or change science parameters; no data-path lock bypass.",
        "schema":{"type":"object","properties":{"job_id":{"type":"string","pattern":"^[0-9]+$"},"nodelist":{"type":"string","pattern":"^[A-Za-z0-9_.-]+$"},"partition":{"type":"string","pattern":"^[A-Za-z0-9_.-]+$"},"memory_mb":{"type":"integer","minimum":64,"description":"Optional finite RAM profile for this tool; kernel verifies it against env/node_policy.json."}},"required":["job_id","nodelist","partition"],"additionalProperties":False}},
    "resource_health": {
        "description": "Read overall CPU/load/memory and bounded scheduler/node health probes. Read-only; no arbitrary command, submission or plan modification. UNKNOWN is not healthy.",
        "schema": {"type": "object", "properties": {}},
    },
    "get_tool_schema": {
        "description": "Read a registered tool's real parameter schema before compiling a delegation. No source inspection, no execution, no capability grant.",
        "schema": {"type": "object", "properties": {"tool_name": {"type": "string", "minLength": 1}}, "required": ["tool_name"]},
    },
    "validate_framework_charges": {
        "description": "Read-only approved framework CIF charge validation, expanding symmetry and weighting occupancy. Declared expected net charge/source required. Never modifies charges, fills zeros or forces neutrality; mismatch needs diagnosis/user-approved model repair.",
        "schema": {"type":"object","properties": {
            "cif_path":{"type":"string","minLength":1}, "expected_net_charge":{"type":"number"},
            "charge_source":{"type":"string","minLength":1},
            "tolerance":{"type":"number","minimum":0,"maximum":0.0001},
            "output_path":{"type":"string","minLength":1,"description":"Required for directory batch validation; writes one persistent JSON evidence report without modifying CIF inputs."}
        },"required":["cif_path","expected_net_charge","charge_source"]},
    },
    "run_project_regressions": {
        "description":"SDK maintainer-only bounded supplementary structural regressions. Fixed project pytest suite, no arbitrary commands or scheduler jobs. Preserves actual diagnostic logs; NOT scientific or end-to-end acceptance.",
        "schema":{"type":"object","properties":{},"additionalProperties":False},
    },
    "convert_physical_units": {
        "description":"Dimension-checked physical unit conversion with source/value provenance. energy epsilon/kB uses kelvin_energy, NOT temperature K. Supports energy, length, pressure, time, temperature. Does not convert potential styles/sigma conventions/harmonic prefactors/mixing/1-4 scaling or infer reduced-LJ scales.",
        "schema":{"type":"object","properties":{
            "value":{"type":"number","description":"A JSON numeric scalar, not a quoted numeric string. Multiple values use multiple tool calls."},
            "quantity":{"type":"string","enum":["energy","length","pressure","time","temperature"]},
            "from_unit":{"type":"string","enum":["kelvin_energy","kcal/mol","kJ/mol","eV","J/particle","angstrom","nm","m","s","ns","ps","fs","Pa","kPa","MPa","bar","atm","K","C"]},
            "to_unit":{"type":"string","enum":["kelvin_energy","kcal/mol","kJ/mol","eV","J/particle","angstrom","nm","m","s","ns","ps","fs","Pa","kPa","MPa","bar","atm","K","C"]}
        },"required":["value","quantity","from_unit","to_unit"],"additionalProperties":False},
    },
    "discover_forcefield": {
        "description": "Discover project-owned scientific parameter assets by name/family (e.g. OPLS). Returns variants, potential, mixrule, source/hash and actual engine readiness. Finding a file is NOT molecular assignment or conversion. Main chat confirms ambiguity before computing.",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 1},
            "target_engine": {"type": "string", "enum": ["towhee", "raspa", "lammps", "cdft"]},
            "required_atom_types": {"type": "array", "maxItems": 64, "items": {"type": "string"}},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 32}
        }, "required": ["query"]},
    },
    "inspect_forcefield": {
        "description": "Inspect exact atom names/raw coefficients and reviewed knowledge. For metal name discovery use element=Fe instead of guessing atom labels. Paginate with offset/max_types<=32. Distinguishes parsed sections from preserved bonded data, and supplies charge source/known water site facts. Legacy supported=false is a terminal capability result, not a retryable parser failure.",
        "schema": {"type": "object", "properties": {
            "forcefield_name": {"type": "string", "minLength": 1},
            "atom_type_names": {"type": "array", "maxItems": 64, "items": {"type": "string"}},
            "element": {"type": "string", "minLength": 1, "maxLength": 8},
            "offset": {"type": "integer", "minimum": 0},
            "name_namespace": {"type": "string", "enum": ["any", "nonbonded", "bonded", "angle", "torsion"]},
            "max_types": {"type": "integer", "minimum": 1, "maximum": 32}
        }, "required": ["forcefield_name"]},
    },
    "request_user_decision": {
        "description": "Main chat only: ask for an external/plan/dispatch decision with structured related-node and recovery identity. Stores the genuine user's answer; does not submit or rewrite the plan.",
        "schema": {"type": "object", "properties": {
            "question": {"type": "string", "minLength": 10}, "reason": {"type": "string", "minLength": 5},
            "related_nodes": {"type": "array", "items": {"type": "string"}},
            "recovery_key": {"type": "string"}, "candidate_job_id": {"type": "string"},
        }, "required": ["question", "reason", "related_nodes"]},
    },
    "reconcile_watched_job": {
        "description": "Main chat only: bind an unresolved dispatch to an EXISTING, same-user/same-conversation watched job. Requires genuine user identity decision ID plus an actual matching diagnostic call. Never submits or cancels jobs.",
        "schema": {"type": "object", "properties": {
            "recovery_key": {"type": "string"}, "job_id": {"type": "string"},
            "decision_id": {"type": "string"}, "diagnosis_call_id": {"type": "string"},
        }, "required": ["recovery_key", "job_id", "decision_id", "diagnosis_call_id"]},
    },
    "supervisor_decision": {
        "description": "Supervisor's structured hand-back to main chat. Does not submit jobs or apply changes. User negotiation/final delivery always belong to main chat.",
        "schema": {"type": "object", "properties": {
            "next_action": {"type": "string", "enum": ["diagnose_and_fix", "wait_existing", "advance_dependencies", "propose_plan_patch", "ask_user", "verified_complete"]},
            "reason": {"type": "string"}, "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "suggested_changes": {"type": "array", "items": {"type": "object"}},
            "scientific_review": {"type": "object", "additionalProperties": False, "properties": {
                "passed": {"type": "boolean"}, "issues": {"type": "array", "items": {"type": "string"}}
            }, "required": ["passed", "issues"]},
        }, "required": ["next_action", "reason", "evidence_refs"]},
    },
    "propose_workflow_patch": {
        "description": "Change executable DAG contracts, NOT node execution state. Identical upsert preserves existing failed/completed state; it cannot retry, restore, or mark complete. Restore archived results with revalidate_workflow_node_outputs(completed_job_id=...), verify prefinish with finish_workflow_node. Changed science/budgets require negotiation; plans are candidates, never mandatory commands.",
        "schema": {"type": "object", "properties": {
            "base_version": {"type": "integer", "minimum": 0},
            "reason": {"type": "string", "minLength": 10},
            "changes": {"type": "array", "minItems": 1, "items": {
                "type": "object", "properties": {
                    "operation": {"type": "string", "enum": ["upsert", "remove"]},
                    "step_id": {"type": "string"},
                    "node": {"type": "object", "additionalProperties": False, "properties": {
                        "step_id": {"type": "string"}, "description": {"type": "string"},
                        "agent": {"type": "string"}, "tool": {"type": "string"}, "arguments": {"type": "object"},
                        "input_bindings": INPUT_BINDINGS_SCHEMA,
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "expected_outputs": {"type": "array", "items": OUTPUT_SCHEMA,
                            "description": "Actual file paths or typed {kind:directory,path:...,pattern:*.dat,min_count:10} artifacts, NEVER prose success claims. Use distinct explicit job_work_dir/input_dir for each cDFT gas branch."},
                        "resource_locks": {"type": "array", "items": {"type": "string"},
                            "description": "Actual shared mutable resources (path:/absolute/resource or explicit global:name), not bare tool names like submit_job. Independent workspaces need no common tool mutex. Leases persist through scheduler completion; dispatch concurrency is separately limited."},
                    }, "required": ["agent", "tool", "arguments", "depends_on", "expected_outputs"]},
                }, "required": ["operation", "step_id"],
            }},
        }, "required": ["base_version", "reason", "changes"]},
    },
    "execute_workflow": {
        "description": "Main chat only: schedule the exact user-approved DAG. Independent nodes use isolated Sessions and global resource leases; no legacy-job redispatch.",
        "schema": {"type": "object", "properties": {"plan_version": {"type": "integer", "minimum": 1}}, "required": ["plan_version"]},
    },
    "message_workflow_node": {
        "description": "Persist a node message and notify main chat/supervisor. Changes pause dispatch for negotiation, never mutate running parameters; comments arrive at tool boundaries.",
        "schema": {"type": "object", "properties": {
            "step_id": {"type": "string", "minLength": 1}, "text": {"type": "string", "minLength": 1},
            "kind": {"type": "string", "enum": ["status", "comment", "change"]}, "message_id": {"type": "string", "minLength": 1},
        }, "required": ["step_id", "text", "kind"]},
    },
    "resolve_local_workflow_write": {
        "description": "Main chat only: resolve an uncertain, stopped write_file node as failed after a genuine node-related user decision and fresh original-path read_file inspection. Releases its lease for negotiated repair; never resubmits or declares output success.",
        "schema": {"type": "object", "properties": {
            "step_id": {"type": "string"}, "decision_id": {"type": "string"}, "verification_call_id": {"type": "string"},
        }, "required": ["step_id", "decision_id", "verification_call_id"]},
    },
    "run_henry_chain": {
        "description": "运行Henry系数计算链条（程序化执行，自动处理依赖和错误诊断）。"
                       "这个工具会自动按顺序执行：查找CIF → 检查/赋电荷 → 计算Henry系数。"
                       "每一步的输出会自动传递给下一步，错误会自动诊断并重试。",
        "schema": {
            "type": "object",
            "properties": {
                "material": {"type": "string", "description": "材料名称（如 Ni-MOF-74）"},
                "gas": {"type": "string", "description": "气体分子（如 CO2, CH4）", "default": "CO2"},
                "temperature": {"type": "number", "description": "温度(K)", "default": 298.0},
            },
            "required": ["material", "gas", "temperature"],
        },
    },
    "run_isotherm_chain": {
        "description": "运行GCMC等温线计算链条（程序化执行）。"
                       "自动按顺序执行：查找CIF → 检查/赋电荷 → GCMC计算 → 等待完成。"
                       "适用于需要完整等温线数据的场景。",
        "schema": {
            "type": "object",
            "properties": {
                "material": {"type": "string", "description": "材料名称"},
                "gas": {"type": "string", "description": "气体分子", "default": "CO2"},
                "pressures": {
                    "type": "array",
                    "description": "压力点列表(bar)",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "default": [0.1, 0.5, 1.0, 5.0, 10.0]
                },
                "temperature": {"type": "number", "description": "温度(K)", "default": 298.0},
            },
            "required": ["material", "gas", "pressures", "temperature"],
        },
    },
    "generate_scientific_report": {
        "description": "Generate a Markdown report. Generic DAG mode reads persisted evidence from completed direct dependency source_steps and writes one auditable report to output_path. GCMC work_dir mode remains supported. Use this for dynamic reports instead of write_file with empty or pre-invented content.",
        "schema": {
            "type": "object",
            "properties": {
                "work_dir": {"type": "string", "description": "GCMC计算工作目录"},
                "material": {"type": "string", "description": "材料名称"},
                "gas": {"type": "string", "description": "气体分子", "default": "CO2"},
                "temperature": {"type": "number", "description": "温度(K)", "default": 298.0},
                "source_steps": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "uniqueItems": True, "description": "Completed direct dependency step IDs"},
                "output_path": {"type": "string", "minLength": 1, "description": "Session-owned Markdown output path"},
                "title": {"type": "string", "minLength": 1},
                "instructions": {"type": "string", "description": "Reporting scope; cannot override or invent tool evidence"},
            },
            "anyOf": [
                {"required": ["work_dir"]},
                {"required": ["source_steps", "output_path", "title"]},
            ],
        },
    },
    "validate_gcmc_results": {
        "description": "验证GCMC计算结果的正确性。检查电荷定义、loading数据、CIF文件等。",
        "schema": {
            "type": "object",
            "properties": {
                "work_dir": {"type": "string", "description": "GCMC计算工作目录"},
            },
            "required": ["work_dir"],
        },
    },
}


# ── Chain Tool executors ──────────────────────────────────────────────

def _exec_henry_chain(p: Dict[str, Any]) -> Any:
    """运行Henry系数计算链条（程序化执行）"""
    try:
        from .chain import create_henry_chain
        import json

        material = p.get("material", "")
        gas = p.get("gas", "CO2")
        temperature = p.get("temperature", 298.0)

        if not material:
            return {"error": "material is required"}

        chain = create_henry_chain(material, gas, temperature)
        context = chain.run({
            "material": material,
            "gas": gas,
            "temperature": temperature,
        })

        statuses = [sr.status.value for sr in context.step_results.values()]
        chain_status = 'waiting' if 'running' in statuses else "failed" if any(s in ("failed", "skipped") for s in statuses) else (
            "submitted" if context.get("henry_job_id") else "completed"
        )

        return {
            "chain_status": chain_status,
            "context": context.data,
            "errors": context.errors,
            "step_results": {
                name: {
                    "status": sr.status.value,
                    "output": sr.output,
                    "error": sr.error,
                    "diagnosis": sr.diagnosis,
                    "retry_count": sr.retry_count,
                    "duration_seconds": sr.duration_seconds,
                }
                for name, sr in context.step_results.items()
            }
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def _exec_isotherm_chain(p: Dict[str, Any]) -> Any:
    """运行GCMC等温线计算链条（程序化执行）"""
    try:
        from .chain import create_isotherm_chain
        import json

        material = p.get("material", "")
        gas = p.get("gas", "CO2")
        pressures = p.get("pressures", [0.1, 0.5, 1.0, 5.0, 10.0])
        temperature = p.get("temperature", 298.0)

        if not material:
            return {"error": "material is required"}

        chain = create_isotherm_chain(material, gas, pressures, temperature)
        context = chain.run({
            "material": material,
            "gas": gas,
            "pressures": pressures,
            "temperature": temperature,
        })

        statuses = [sr.status.value for sr in context.step_results.values()]
        chain_status = 'waiting' if 'running' in statuses else "failed" if any(s in ("failed", "skipped") for s in statuses) else (
            "submitted" if context.get("gcmc_job_ids") else "completed"
        )

        return {
            "chain_status": chain_status,
            "context": context.data,
            "errors": context.errors,
            "step_results": {
                name: {
                    "status": sr.status.value,
                    "output": sr.output,
                    "error": sr.error,
                    "diagnosis": sr.diagnosis,
                    "retry_count": sr.retry_count,
                    "duration_seconds": sr.duration_seconds,
                }
                for name, sr in context.step_results.items()
            }
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def _exec_generate_report(p: Dict[str, Any]) -> Any:
    """自动生成科学验证报告"""
    import time
    from pathlib import Path

    if p.get("source_steps"):
        import json
        import os
        import tempfile
        from .watch_context import get_context
        from .parallel_workflow import WorkflowStore
        context = get_context()
        username, conv_id = context.get('username', ''), context.get('conv_id', '')
        current_step = context.get('step_id', '')
        if not username or not conv_id or not current_step:
            return {"error": "generic evidence report requires an owned workflow node context"}
        store = WorkflowStore(get_config().project_root / 'data/state/parallel_workflows.json')
        state = store.snapshot(store.identity(username, conv_id))
        current = state.get('nodes', {}).get(current_step, {})
        dependencies = set(current.get('contract', {}).get('depends_on', []))
        source_steps = list(p['source_steps'])
        if not set(source_steps) <= dependencies:
            return {"error": "source_steps must be completed direct dependencies of this report node"}
        evidence_rows = []
        own_root = (get_config().project_root / 'runs' / username / conv_id).resolve()
        for step_id in source_steps:
            node = state.get('nodes', {}).get(step_id, {})
            if node.get('status') != 'succeeded':
                return {"error": f"source step is not succeeded: {step_id}"}
            reference = node.get('result_ref') or {}
            evidence_path = Path(reference.get('evidence_path') or '')
            try:
                resolved = evidence_path.resolve()
            except Exception:
                return {"error": f"invalid evidence path for {step_id}"}
            if not evidence_path.is_file() or not resolved.is_relative_to(own_root):
                return {"error": f"owned evidence file missing for {step_id}"}
            evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
            raw_result = evidence.get('result', evidence.get('result_preview', ''))
            try:
                parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                rendered = json.dumps(parsed, ensure_ascii=False, indent=2, default=str)
            except (ValueError, TypeError):
                rendered = str(raw_result)
            evidence_rows.append({
                'step_id': step_id,
                'tool': evidence.get('tool') or node.get('contract', {}).get('tool', ''),
                'agent': evidence.get('agent') or node.get('contract', {}).get('agent', ''),
                'call_id': evidence.get('call_id') or reference.get('call_id', ''),
                'params': evidence.get('params', {}),
                'result': rendered[:12000],
                'evidence_path': str(resolved),
            })
        title = p['title'].strip()
        lines = [f"# {title}", "", "## 验收范围", "",
                 p.get('instructions', '依据本会话已完成 DAG 节点的持久化工具证据生成；未添加证据外结论。'), "",
                 "## DAG 证据索引", "", "| 节点 | Agent | Tool | Call ID |", "|---|---|---|---|"]
        for row in evidence_rows:
            lines.append(f"| {row['step_id']} | {row['agent']} | {row['tool']} | {row['call_id']} |")
        for row in evidence_rows:
            lines.extend(["", f"## {row['step_id']} · {row['tool']}", "",
                          f"- 证据：`{row['evidence_path']}`", "- 参数：", "```json",
                          json.dumps(row['params'], ensure_ascii=False, indent=2, default=str), "```",
                          "- 工具结果：", "```json", row['result'], "```"])
        lines.extend(["", "## 结论边界", "",
                      f"已汇总 {len(evidence_rows)} 个成功节点的原始证据。科学解释与最终结论须与上述工具结果一致；无证据项不视为通过。", ""])
        output = Path(p['output_path']).resolve()
        if not output.is_relative_to(own_root):
            return {"error": "report output_path must stay inside the current user/session root",
                    "session_root": str(own_root)}
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=output.name + '.', dir=str(output.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write('\n'.join(lines))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {"status": "success", "report_file": str(output), "output_files": [str(output)],
                "source_steps": source_steps, "evidence_count": len(evidence_rows)}

    work_dir = p.get("work_dir", "")
    material = p.get("material", "Unknown")
    gas = p.get("gas", "CO2")
    temperature = p.get("temperature", 298.0)

    if not work_dir:
        return {"error": "work_dir is required"}

    work_path = Path(work_dir)
    if not work_path.exists():
        return {"error": f"Work directory not found: {work_dir}"}

    # 收集GCMC结果
    results = {
        "material": material,
        "gas": gas,
        "temperature": temperature,
        "pressure_points": [],
        "completed_points": 0,
        "total_points": 0,
    }

    for p_dir in sorted(work_path.glob("P_*")):
        results["total_points"] += 1
        data_files = list(p_dir.glob("Output/System_0/*.data"))
        if data_files:
            content = data_files[0].read_text(errors="ignore")
            if "Average loading absolute" in content:
                for line in content.splitlines():
                    if "Average loading absolute" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                loading = float(parts[3])
                                pressure = float(p_dir.name.replace("P_", ""))
                                results["pressure_points"].append({
                                    "pressure_Pa": pressure,
                                    "pressure_bar": pressure / 1e5,
                                    "loading_mol_kg": loading,
                                })
                                results["completed_points"] += 1
                            except ValueError:
                                pass
                        break

    # 验证物理正确性
    if results["pressure_points"]:
        sorted_points = sorted(results["pressure_points"], key=lambda x: x["pressure_Pa"])
        loadings = [p["loading_mol_kg"] for p in sorted_points]
        is_monotonic = all(loadings[i] <= loadings[i+1] for i in range(len(loadings)-1))
        results["isotherm_shape"] = "Langmuir (monotonic)" if is_monotonic else "Non-monotonic"
        results["physical_validity"] = "PASS" if is_monotonic else "FAIL"
        results["loading_range"] = f"{min(loadings):.3f} - {max(loadings):.3f} mol/kg"

    # 生成报告
    report = f"""# 科学验证报告 - {material} + {gas} 吸附计算

## 1. 计算参数
- **材料**: {material}
- **气体**: {gas}
- **温度**: {temperature} K
- **方法**: GCMC (Grand Canonical Monte Carlo)

## 2. 计算结果

### 等温线数据
| 压力 (bar) | 吸附量 (mol/kg) |
|------------|-----------------|
"""

    for p in sorted(results["pressure_points"], key=lambda x: x["pressure_Pa"]):
        report += f"| {p['pressure_bar']:.3f} | {p['loading_mol_kg']:.3f} |\n"

    report += f"""
### 物理验证
| 验证项 | 结果 |
|--------|------|
| 完成进度 | {results['completed_points']}/{results['total_points']} |
| 等温线形状 | {results.get('isotherm_shape', 'N/A')} |
| 吸附量范围 | {results.get('loading_range', 'N/A')} |
| 物理有效性 | {results.get('physical_validity', 'N/A')} |

## 3. 结论
"""

    if results.get("physical_validity") == "PASS":
        report += "✅ 计算结果物理正确，等温线呈Langmuir型（单调递增）。\n"
    else:
        report += "⚠️ 计算结果需要进一步验证。\n"

    # 保存报告
    output_dir = work_path / "reports"
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / f"scientific_report_{material}_{gas}_{temperature}K.md"
    report_file.write_text(report, encoding="utf-8")

    return {
        "status": "success",
        "report_file": str(report_file),
        "report_content": report,
        "results": results,
    }


def _exec_validate_gcmc(p: Dict[str, Any]) -> Any:
    """验证GCMC计算结果的正确性"""
    from pathlib import Path

    work_dir = p.get("work_dir", "")
    if not work_dir:
        return {"error": "work_dir is required"}

    work_path = Path(work_dir)
    if not work_path.exists():
        return {"error": f"Work directory not found: {work_dir}"}

    # 检查RASPA输出
    issues = []
    warnings = []

    for p_dir in sorted(work_path.glob("P_*")):
        data_files = list(p_dir.glob("Output/System_0/*.data"))
        if not data_files:
            issues.append(f"{p_dir.name}: No data files found")
            continue

        content = data_files[0].read_text(errors="ignore")

        # 检查电荷警告
        if "charge definition not found" in content:
            issues.append(f"{p_dir.name}: Charge definitions missing - atoms have q=0")

        # 检查loading
        if "Average loading absolute" not in content:
            warnings.append(f"{p_dir.name}: No loading data (still running?)")
        else:
            for line in content.splitlines():
                if "Average loading absolute" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            loading = float(parts[3])
                            if loading == 0:
                                warnings.append(f"{p_dir.name}: Zero loading (possible silent failure)")
                        except ValueError:
                            pass
                    break

    # 检查CIF电荷
    cif_files = list(work_path.glob("*.cif"))
    for cif in cif_files:
        content = cif.read_text(errors="ignore")
        if "_atom_site_charge" not in content:
            issues.append(f"CIF {cif.name}: Missing charge column")

    return {
        "status": "success" if not issues else "failed",
        "issues": issues,
        "warnings": warnings,
        "total_pressure_points": len(list(work_path.glob("P_*"))),
        "validation_passed": len(issues) == 0,
    }


def _exec_ga_optimization(p: Dict[str, Any]) -> Any:
    """Run genetic algorithm optimization for MOF material design using PORMAKE and ML model."""
    return {
        "blocked": True,
        "executed": False,
        "error": "legacy GA is disabled: it can substitute estimated/random structure features and a hand-coded selectivity formula",
        "safe_alternative": "train ml_train on real labels, use RF ml_active_learning uncertainty, then validate selected CIFs with one real batched simulation node",
    }

    # Historical implementation retained below only so old run manifests can
    # be audited against their cited source.  It is intentionally unreachable.
    config = get_config()

    # 必填参数：目标选择性和吸附量（必须有科学依据）
    target_selectivity = p.get("target_selectivity")
    target_uptake = p.get("target_uptake")

    if target_selectivity is None or target_uptake is None:
        return {
            "error": "target_selectivity and target_uptake are required",
            "hint": "请根据应用场景设定目标值：\n"
                   "- 燃烧后CO2捕获: selectivity>20, uptake>2 mmol/g\n"
                   "- 天然气净化: selectivity>50, uptake>3 mmol/g\n"
                   "- 需要有文献或实验依据",
            "literature_reference": "Typical ranges: selectivity 3-200, uptake 1-15 mmol/g"
        }

    pop_size = p.get("pop_size", 20)
    n_generations = p.get("n_generations", 10)
    output_dir = p.get("output_dir", _ws("ga"))
    ml_model_dir = p.get("ml_model_dir", _ws("ml"))

    # Use absolute paths to avoid SLURM work_dir nesting issue
    abs_output_dir = str(Path(config.project_root) / output_dir) if not Path(output_dir).is_absolute() else output_dir
    abs_ml_model_dir = str(Path(config.project_root) / ml_model_dir) if not Path(ml_model_dir).is_absolute() else ml_model_dir

    # Get tool paths from config
    zeo_path = config.zeopp_path
    conda_path = config.conda_path

    script = f"""set -euo pipefail
mkdir -p {abs_output_dir}
export ZEO_PATH={zeo_path}
source {conda_path} && conda activate pormake
python3 /home/user/gcmc_agent/BiMemAgent-claude-sdk/tools/pormake_ga_v3.py --target-selectivity {target_selectivity} --target-uptake {target_uptake} --pop-size {pop_size} --n-generations {n_generations} --output-dir {abs_output_dir} --ml-model-dir {abs_ml_model_dir}
"""
    return slurm.submit_and_wait(job_name="ga_optimization", command=script, work_dir=abs_output_dir, timeout_minutes=15, partition="compute", cpus_per_task=16, walltime="02:00:00")


def _exec_validate_ga_result(p: Dict[str, Any]) -> Any:
    """Validate GA results by checking if ML model was used and results are reasonable."""
    config = get_config()
    ga_result_dir = p.get("ga_result_dir", "")

    if not ga_result_dir:
        return {"error": "ga_result_dir is required"}

    result_path = Path(config.project_root) / ga_result_dir / "ga_result.json"
    if not result_path.exists():
        return {"error": f"GA result not found: {result_path}"}

    import json
    with open(result_path) as f:
        ga_result = json.load(f)

    total_mofs = ga_result.get("generated_mofs", 1)
    ml_predicted = ga_result.get("ml_predicted_count", 0)
    features_real = ga_result.get("features_real_count", 0)

    validation = {
        "ga_result_dir": ga_result_dir,
        "ml_model_used": ga_result.get("optimization_params", {}).get("ml_model_used", False),
        "ml_predicted_count": ml_predicted,
        "features_real_count": features_real,
        "total_generated": total_mofs,
        "best_fitness": ga_result.get("predicted_performance", {}).get("fitness_score", 0),
        "best_selectivity": ga_result.get("best_material", {}).get("CO2_N2_selectivity", 0),
        "best_uptake": ga_result.get("best_material", {}).get("CO2_uptake_mmolg", 0),
        "ml_prediction_rate": ml_predicted / max(1, total_mofs),
        "features_real_rate": features_real / max(1, total_mofs),
        "validation_passed": True,
        "issues": [],
        "recommendations": []
    }

    # 检查ML模型使用
    if not validation["ml_model_used"]:
        validation["issues"].append("❌ 严重问题: GA未使用ML模型，使用了硬编码公式")
        validation["recommendations"].append("重新训练ML模型并传入ml_model_dir参数")
        validation["validation_passed"] = False

    # 检查ML预测率
    if validation["ml_prediction_rate"] < 1.0:
        validation["issues"].append(f"⚠️ ML预测率不足: {validation['ml_prediction_rate']:.1%}")

    # 检查特征是否来自真实计算
    if validation["features_real_rate"] == 0:
        validation["issues"].append("⚠️ 特征未从CIF文件计算，可能使用了随机数或估算值")
        validation["recommendations"].append("安装zeo++并确保从CIF文件计算真实结构特征")

    # 检查结果合理性
    if validation["best_uptake"] > 15:
        validation["issues"].append(f"⚠️ 吸附量异常高: {validation['best_uptake']:.2f} mmol/g")
    if validation["best_selectivity"] > 200:
        validation["issues"].append(f"⚠️ 选择性异常高: {validation['best_selectivity']:.1f}")

    return validation


def _exec_build_mof_database(p: Dict[str, Any]) -> Any:
    """Build an evidence-backed CIF inventory; never inject known/example values."""
    import csv
    import hashlib
    import shlex
    from collections import Counter, defaultdict
    from datetime import datetime, timezone

    from .watch_context import get_context
    from .workspace import session_root

    source = Path(_resolve_fs_path(str(p.get("cif_dir") or ""))).resolve()
    if not source.is_dir():
        return {"error": f"cif_dir is not a directory: {source}"}
    max_files = int(p.get("max_files", 1000))
    recursive = bool(p.get("recursive", False))
    candidates = sorted(source.rglob("*.cif") if recursive else source.glob("*.cif"))[:max_files]
    if not candidates:
        return {"error": f"no CIF files found in {source}"}
    ctx = get_context()
    if not ctx.get("username") or not ctx.get("conv_id"):
        return {"error": "database construction requires an authenticated user/session context"}
    root = session_root(ctx["conv_id"], ctx["username"]).resolve()
    requested = p.get("output_dir") or str(root / "database")
    output_dir = Path(str(requested)).expanduser()
    output_dir = (output_dir if output_dir.is_absolute() else get_config().project_root / output_dir).resolve()
    if not output_dir.is_relative_to(root):
        return {"error": "output_dir must be inside the current session root", "session_root": str(root)}

    def number(text: str):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", text)
        return float(match.group()) if match else None

    def atom_rows(lines):
        for index, line in enumerate(lines):
            if line.strip().lower() != "loop_":
                continue
            headers = []
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].lstrip().startswith("_"):
                headers.append(lines[cursor].strip().split()[0])
                cursor += 1
            lowered = [header.lower() for header in headers]
            if "_atom_site_label" not in lowered or not any(
                    name in lowered for name in ("_atom_site_type_symbol", "_atom_type_symbol")):
                continue
            rows = []
            while cursor < len(lines):
                stripped = lines[cursor].strip()
                if not stripped or stripped.startswith("#"):
                    cursor += 1
                    continue
                if stripped.lower() == "loop_" or stripped.startswith("_") or stripped.lower().startswith("data_"):
                    break
                try:
                    values = shlex.split(stripped, comments=False, posix=True)
                except ValueError:
                    values = stripped.split()
                if len(values) < len(headers):
                    break
                rows.append(dict(zip(lowered, values)))
                cursor += 1
            return rows, lowered
        return [], []

    records, failures = [], []
    hashes = defaultdict(list)
    for cif in candidates:
        try:
            raw = cif.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            sha = hashlib.sha256(raw).hexdigest()
            hashes[sha].append(str(cif))
            fields = {}
            for key in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
                        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"):
                match = re.search(rf"(?im)^\s*{re.escape(key)}\s+([^\s#]+)", text)
                fields[key] = number(match.group(1)) if match else None
            a, b, c = (fields[key] for key in ("_cell_length_a", "_cell_length_b", "_cell_length_c"))
            alpha, beta, gamma = (fields[key] for key in ("_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"))
            volume = None
            if all(value is not None for value in (a, b, c, alpha, beta, gamma)):
                ar, br, gr = (math.radians(value) for value in (alpha, beta, gamma))
                factor = 1 + 2 * math.cos(ar) * math.cos(br) * math.cos(gr) \
                    - math.cos(ar) ** 2 - math.cos(br) ** 2 - math.cos(gr) ** 2
                volume = a * b * c * math.sqrt(max(0.0, factor))
            rows, headers = atom_rows(lines)
            composition = Counter()
            charges = []
            for row in rows:
                symbol = row.get("_atom_site_type_symbol") or row.get("_atom_type_symbol") or row.get("_atom_site_label", "")
                element = (re.match(r"[A-Z][a-z]?", symbol) or re.match(r"[A-Za-z]+", symbol))
                if element:
                    composition[element.group().capitalize()] += 1
                for charge_key in ("_atom_site_charge", "_atom_type_partial_charge"):
                    if charge_key in row:
                        value = number(row[charge_key])
                        if value is not None:
                            charges.append(value)
                        break
            data_match = re.search(r"(?im)^\s*data_([^\s#]+)", text)
            records.append({
                "cif_name": cif.name, "relative_path": str(cif.relative_to(source)), "absolute_path": str(cif),
                "sha256": sha, "bytes": len(raw), "data_block": data_match.group(1) if data_match else "",
                "atom_count": len(rows), "composition": dict(sorted(composition.items())),
                "cell_a_A": a, "cell_b_A": b, "cell_c_A": c,
                "alpha_deg": alpha, "beta_deg": beta, "gamma_deg": gamma, "cell_volume_A3": volume,
                "has_charge_column": any(key in headers for key in ("_atom_site_charge", "_atom_type_partial_charge")),
                "net_charge_e": sum(charges) if charges else None,
                "parse_warnings": (["atom loop not parsed"] if not rows else [])
                    + (["cell parameters incomplete"] if volume is None else []),
            })
        except Exception as error:
            failures.append({"path": str(cif), "error": str(error)})

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    database = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source), "recursive": recursive, "max_files": max_files,
        "records": records, "duplicate_sha256_groups": duplicates, "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / "mof_database.json", output_dir / "mof_database.csv"
    json_partial = json_path.with_name(json_path.name + ".part")
    json_partial.write_text(json.dumps(database, indent=2, ensure_ascii=False), encoding="utf-8")
    json_partial.replace(json_path)
    csv_partial = csv_path.with_name(csv_path.name + ".part")
    columns = ["cif_name", "relative_path", "absolute_path", "sha256", "bytes", "data_block", "atom_count",
               "composition", "cell_a_A", "cell_b_A", "cell_c_A", "alpha_deg", "beta_deg", "gamma_deg",
               "cell_volume_A3", "has_charge_column", "net_charge_e", "parse_warnings"]
    with csv_partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["composition"] = json.dumps(row["composition"], ensure_ascii=False, sort_keys=True)
            row["parse_warnings"] = json.dumps(row["parse_warnings"], ensure_ascii=False)
            writer.writerow(row)
    csv_partial.replace(csv_path)
    return {
        "status": "success", "source_dir": str(source), "n_cifs": len(records),
        "n_failures": len(failures), "duplicate_groups": duplicates,
        "database_json": str(json_path), "database_csv": str(csv_path),
        "output_files": [str(json_path), str(csv_path)],
    }


def _exec_validate_method(p: Dict[str, Any]) -> Any:
    """Validate computational method correctness - check parameters, convergence, and physics."""
    method = p.get("method", "")
    params = p.get("parameters", {})
    issues = []

    # GCMC验证
    if method.upper() == "GCMC":
        if params.get("cycles", 0) < 10000:
            issues.append("⚠️ MC cycles不足，建议>=50000用于生产计算")
        if params.get("unit_cells", "") == "1x1x1":
            issues.append("⚠️ 超胞太小，建议至少2x2x2")
        if params.get("pressure", 0) > 100:
            issues.append("⚠️ 压力过高，需要检查力场适用范围")

    # ML验证
    elif method.upper() == "ML":
        cv_r2 = params.get("cv_r2", 0)
        if cv_r2 < 0.5:
            issues.append(f"⚠️ ML模型R²={cv_r2:.3f}，预测能力不足（建议>0.7）")
        n_samples = params.get("n_samples", 0)
        if n_samples < 30:
            issues.append(f"⚠️ 训练样本不足: n={n_samples}（建议>=50）")

    # GA验证
    elif method.upper() == "GA":
        if not params.get("ml_model_used", False):
            issues.append("❌ 严重问题: GA未使用ML模型，使用了硬编码公式")
        if params.get("features_random", False):
            issues.append("❌ 严重问题: 特征是随机数，不是从CIF计算")

    return {
        "method": method,
        "valid": len(issues) == 0,
        "issues": issues,
        "recommendation": "方法正确" if len(issues) == 0 else "需要修正参数"
    }


def _exec_check_conclusion_reliability(p: Dict[str, Any]) -> Any:
    """Check if conclusion has statistical support and is reliable."""
    data = p.get("data", [])
    conclusion = p.get("conclusion", "")
    import numpy as np

    issues = []
    n = len(data) if data else 0

    if n == 0:
        return {"valid": False, "issues": ["❌ 无数据支持结论"]}

    if n < 3:
        issues.append(f"⚠️ 样本量不足: n={n}，结论不稳定")

    if n > 0:
        mean = np.mean(data)
        std = np.std(data, ddof=1) if n > 1 else 0
        cv = std / mean if mean != 0 else float('inf')

        if cv > 0.5:
            issues.append(f"⚠️ 变异系数过高: CV={cv:.2f}，数据波动大")

        # 检查是否有异常值
        if n > 4:
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            outliers = [x for x in data if x < q1 - 1.5*iqr or x > q3 + 1.5*iqr]
            if outliers:
                issues.append(f"⚠️ 存在异常值: {outliers[:3]}")

    return {
        "valid": len(issues) == 0,
        "n_samples": n,
        "mean": float(np.mean(data)) if data else 0,
        "cv": float(cv) if data and mean != 0 else 0,
        "issues": issues,
        "recommendation": "结论可靠" if len(issues) == 0 else "需要更多数据或修正结论"
    }


def _exec_verify_data_authenticity(p: Dict[str, Any]) -> Any:
    """Verify that data comes from real calculations, not random generation."""
    data_source = p.get("data_source", "")
    feature_values = p.get("feature_values", [])
    import numpy as np

    issues = []

    if not data_source:
        issues.append("❌ 未指定数据来源")

    # 检查特征值是否为随机生成
    if feature_values:
        data = np.array(feature_values)
        n = len(data)

        # 检查分布是否过于均匀（随机数特征）
        if n >= 10:
            # 计算变异系数
            cv = np.std(data) / np.mean(data) if np.mean(data) != 0 else 0
            if cv < 0.1:
                issues.append("⚠️ 数据变异系数过低，可能是随机生成")

            # 检查是否有重复值
            unique_ratio = len(np.unique(data)) / n
            if unique_ratio < 0.3:
                issues.append(f"⚠️ 唯一值比例过低: {unique_ratio:.1%}，可能是随机生成")

    return {
        "valid": len(issues) == 0,
        "data_source": data_source,
        "issues": issues,
        "recommendation": "数据来源可信" if len(issues) == 0 else "需要验证数据来源"
    }


def _exec_download_scientific_file(p: Dict[str, Any]) -> Any:
    """Download one HTTPS scientific artifact into the current session safely."""
    import hashlib
    import ipaddress
    import socket
    from urllib.parse import urljoin, urlparse

    import requests

    from .watch_context import get_context
    from .workspace import session_root

    url = str(p.get("url") or "").strip()
    output_path = str(p.get("output_path") or "").strip()
    expected_sha256 = str(p.get("expected_sha256") or "").strip().lower()
    max_bytes = int(p.get("max_bytes", 20 * 1024 * 1024))
    timeout = float(p.get("timeout_seconds", 30))
    ctx = get_context()
    if not ctx.get("username") or not ctx.get("conv_id"):
        return {"error": "download requires an authenticated user/session context"}
    root = session_root(ctx["conv_id"], ctx["username"]).resolve()
    target = Path(output_path).expanduser()
    target = (target if target.is_absolute() else get_config().project_root / target).resolve()
    if not target.is_relative_to(root):
        return {"error": "output_path must be inside the current session root", "session_root": str(root)}
    if expected_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        return {"error": "expected_sha256 must contain exactly 64 lowercase/uppercase hex characters"}

    def validate_address(address: str) -> None:
        ip = ipaddress.ip_address(address)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            raise ValueError(f"URL resolves/connects to a forbidden network address: {address}")

    def validate_remote(candidate: str) -> None:
        parsed = urlparse(candidate)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("only credential-free HTTPS URLs are allowed")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
        except socket.gaierror as error:
            raise ValueError(f"hostname resolution failed: {parsed.hostname}") from error
        if not addresses:
            raise ValueError("hostname resolved to no address")
        for address in addresses:
            validate_address(address)

    def validate_connected_peer(download_response) -> None:
        raw = getattr(download_response, "raw", None)
        connection = getattr(raw, "_connection", None) or getattr(raw, "connection", None)
        sock = getattr(connection, "sock", None)
        if sock is None:
            raise ValueError("unable to verify the download TCP peer address")
        validate_address(sock.getpeername()[0])

    current = url
    response = None
    redirects = []
    try:
        for _ in range(6):
            validate_remote(current)
            response = requests.get(current, stream=True, allow_redirects=False, timeout=timeout,
                                    headers={"User-Agent": "BiMemAgent-scientific-downloader/1.0"})
            validate_connected_peer(response)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ValueError("redirect response omitted Location")
                current = urljoin(current, location)
                redirects.append(current)
                continue
            break
        else:
            raise ValueError("too many redirects")
        if response is None or response.status_code != 200:
            status = response.status_code if response is not None else None
            return {"error": f"download failed with HTTP {status}", "url": current, "redirects": redirects}
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            response.close()
            return {"error": "declared content length exceeds max_bytes", "content_length": int(declared), "max_bytes": max_bytes}
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("download exceeded max_bytes")
                    digest.update(chunk)
                    handle.write(chunk)
            actual = digest.hexdigest()
            if expected_sha256 and actual != expected_sha256:
                partial.unlink(missing_ok=True)
                return {"error": "SHA256 mismatch", "expected_sha256": expected_sha256,
                        "actual_sha256": actual, "bytes": size}
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return {
            "status": "success", "source_url": url, "final_url": current,
            "redirects": redirects, "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "bytes": size, "sha256": digest.hexdigest(),
            "output_path": str(target), "output_files": [str(target)],
        }
    except (requests.RequestException, ValueError, OSError) as error:
        return {"error": str(error), "url": current, "redirects": redirects}
    finally:
        if response is not None:
            response.close()


def _exec_analyze_diffusion_msd(p: Dict[str, Any]) -> Any:
    """Compute ensemble MSD and a diffusion coefficient from CSV/LAMMPS trajectories."""
    import csv
    import numpy as np

    from .watch_context import get_context
    from .workspace import session_root

    trajectory = Path(_resolve_fs_path(str(p.get("trajectory_path") or ""))).resolve()
    if not trajectory.is_file():
        return {"error": f"trajectory file not found: {trajectory}"}
    timestep_ps = float(p.get("timestep_ps", 0))
    dimensions = int(p.get("dimensions", 3))
    fit_start = float(p.get("fit_start_fraction", 0.2))
    fit_end = float(p.get("fit_end_fraction", 0.8))
    if timestep_ps <= 0 or dimensions not in {1, 2, 3} or not (0 <= fit_start < fit_end <= 1):
        return {"error": "require timestep_ps>0, dimensions in {1,2,3}, and 0<=fit_start<fit_end<=1"}
    atom_filter = {int(value) for value in p.get("atom_ids", [])}
    fmt = p.get("format", "auto")
    if fmt == "auto":
        fmt = "csv" if trajectory.suffix.lower() == ".csv" else "lammps_dump"

    frames = []
    if fmt == "csv":
        with trajectory.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"time_ps", "particle_id", "x_A", "y_A", "z_A"}
        if not rows or not required.issubset(rows[0]):
            return {"error": "CSV requires time_ps,particle_id,x_A,y_A,z_A columns"}
        grouped = {}
        for row in rows:
            particle = int(row["particle_id"])
            if atom_filter and particle not in atom_filter:
                continue
            grouped.setdefault(float(row["time_ps"]), {})[particle] = np.array(
                [float(row["x_A"]), float(row["y_A"]), float(row["z_A"])], dtype=float)
        frames = sorted(grouped.items())
    elif fmt == "lammps_dump":
        lines = trajectory.read_text(errors="replace").splitlines()
        index = 0
        while index < len(lines):
            if lines[index].strip() != "ITEM: TIMESTEP":
                index += 1
                continue
            step = int(lines[index + 1].strip())
            if lines[index + 2].strip() != "ITEM: NUMBER OF ATOMS":
                return {"error": f"invalid LAMMPS dump near line {index + 3}"}
            count = int(lines[index + 3].strip())
            box_header = lines[index + 4].strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                return {"error": f"missing BOX BOUNDS near line {index + 5}"}
            bounds = []
            for offset in range(3):
                values = [float(value) for value in lines[index + 5 + offset].split()[:2]]
                bounds.append(values)
            atom_header_index = index + 8
            header = lines[atom_header_index].split()[2:]
            if not lines[atom_header_index].startswith("ITEM: ATOMS ") or "id" not in header:
                return {"error": f"LAMMPS dump requires an atom id column near line {atom_header_index + 1}"}
            column = {name: pos for pos, name in enumerate(header)}
            unwrapped = all(name in column for name in ("xu", "yu", "zu"))
            wrapped_with_images = all(name in column for name in ("x", "y", "z", "ix", "iy", "iz"))
            if not unwrapped and not wrapped_with_images:
                return {"error": "LAMMPS dump needs xu,yu,zu or x,y,z plus ix,iy,iz; wrapped coordinates alone are invalid for MSD"}
            positions = {}
            for line in lines[atom_header_index + 1: atom_header_index + 1 + count]:
                values = line.split()
                particle = int(values[column["id"]])
                if atom_filter and particle not in atom_filter:
                    continue
                if unwrapped:
                    xyz = [float(values[column[name]]) for name in ("xu", "yu", "zu")]
                else:
                    xyz = []
                    for axis, image, bound in zip(("x", "y", "z"), ("ix", "iy", "iz"), bounds):
                        xyz.append(float(values[column[axis]]) + int(values[column[image]]) * (bound[1] - bound[0]))
                positions[particle] = np.asarray(xyz, dtype=float)
            frames.append((step * timestep_ps, positions))
            index = atom_header_index + 1 + count
    else:
        return {"error": f"unsupported trajectory format: {fmt}"}

    if len(frames) < 5:
        return {"error": f"at least 5 frames are required, found {len(frames)}"}
    initial_ids = set(frames[0][1])
    common_ids = initial_ids.intersection(*(set(frame) for _, frame in frames[1:]))
    if not common_ids:
        return {"error": "no particle ids occur in every frame"}
    origin = frames[0][1]
    times = np.asarray([time_value - frames[0][0] for time_value, _ in frames], dtype=float)
    msd = np.asarray([
        np.mean([np.sum((positions[particle][:dimensions] - origin[particle][:dimensions]) ** 2)
                 for particle in common_ids])
        for _, positions in frames
    ], dtype=float)
    first = max(1, int(len(times) * fit_start))
    last = min(len(times), max(first + 3, int(math.ceil(len(times) * fit_end))))
    if last - first < 3 or np.ptp(times[first:last]) <= 0:
        return {"error": "selected fit window has fewer than 3 distinct time points"}
    slope, intercept = np.polyfit(times[first:last], msd[first:last], 1)
    predicted = slope * times[first:last] + intercept
    residual = float(np.sum((msd[first:last] - predicted) ** 2))
    total = float(np.sum((msd[first:last] - np.mean(msd[first:last])) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 0.0
    diffusion = float(slope / (2 * dimensions) * 1e-8)
    block_values = []
    for indices in np.array_split(np.arange(first, last), min(3, last - first)):
        if len(indices) >= 3 and np.ptp(times[indices]) > 0:
            block_slope = np.polyfit(times[indices], msd[indices], 1)[0]
            block_values.append(float(block_slope / (2 * dimensions) * 1e-8))
    stderr = float(np.std(block_values, ddof=1) / math.sqrt(len(block_values))) if len(block_values) > 1 else None
    warnings = []
    if slope <= 0:
        warnings.append("non-positive MSD slope; diffusion coefficient is not physically usable")
    if r2 < 0.8:
        warnings.append("fit R2 below 0.8; trajectory may not contain a clear diffusive regime")

    ctx = get_context()
    if not ctx.get("username") or not ctx.get("conv_id"):
        return {"error": "MSD analysis requires an authenticated user/session context"}
    root = session_root(ctx["conv_id"], ctx["username"]).resolve()
    output = Path(str(p.get("output_csv") or root / "analysis" / "msd.csv")).expanduser()
    output = (output if output.is_absolute() else get_config().project_root / output).resolve()
    if not output.is_relative_to(root):
        return {"error": "output_csv must be inside the current session root", "session_root": str(root)}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_ps", "msd_A2"])
        writer.writerows(zip(times.tolist(), msd.tolist()))
    partial.replace(output)
    return {
        "status": "success", "trajectory_path": str(trajectory), "format": fmt,
        "n_frames": len(frames), "n_particles": len(common_ids), "dimensions": dimensions,
        "fit_time_ps": [float(times[first]), float(times[last - 1])],
        "slope_A2_per_ps": float(slope), "fit_intercept_A2": float(intercept), "fit_r2": float(r2),
        "diffusion_coefficient_m2_s": diffusion, "diffusion_stderr_m2_s": stderr,
        "block_estimates_m2_s": block_values, "warnings": warnings,
        "output_csv": str(output), "output_files": [str(output)],
    }


_EXECUTORS = {
    "run_gcmc_isotherm": _exec_gcmc_isotherm,
    "run_gcmc_batch": _exec_gcmc_batch,
    "run_henry": _exec_henry,
    "run_pore_analysis": _exec_pore_analysis,
    "run_pacman_charge": _exec_charge,
    "run_md_optimize": _exec_md_optimize,
    "generate_structure": _exec_structure_gen,
    "run_xtb_optimize": _exec_xtb,
    "build_guest_forcefield": _exec_guest_ff,
    "run_cdft": _exec_cdft,
    "expand_cell": _exec_expand_cell,
    "restart_backend": _exec_restart_backend,
    "calc_binding_energy": _exec_binding,
    "run_vasp": _exec_vasp,
    "run_string_tst": _exec_string_tst,
    "run_external_potential": _exec_vext,
    "check_job": _exec_check_job,
    "list_my_jobs": _exec_list_my_jobs,
    "diagnose_job": _exec_diagnose_job,
    "submit_job": _exec_submit_job,
    "query_literature": _exec_rag,
    "inspect_path": _exec_inspect,
    "stage_cif_subset": _exec_stage_cif_subset,
    "analyze_gcmc_screening": _exec_analyze_gcmc_screening,
    "inspect_run": _exec_inspect_run,
    "find_cif": _exec_find_cif,
    "extract_features": _exec_features,
    "ml_train": _exec_ml_train,
    "ml_predict": _exec_ml_predict,
    "ml_feature_importance": _exec_ml_feature_importance,
    "ml_active_learning": _exec_ml_active_learning,
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "run_bash": _exec_run_bash,
    "grep_search": _exec_grep_search,
    "task_line_query": _exec_task_line_query,
    "task_line_update": _exec_task_line_update,
    "recovery_state": lambda p: {"error": "recovery_state must run inside agents.Session"},
    "prepare_retry": lambda p: {"error": "prepare_retry must run inside agents.Session"},
    "accept_recovered_result": lambda p: {"error": "accept_recovered_result must run inside agents.Session"},
    "lifecycle_state": lambda p: {"error": "lifecycle_state must run inside agents.Session"},
    "resource_health": _exec_resource_health,
    "assess_job_resources": _exec_assess_job_resources,
    "resource_review_decision": _exec_resource_review_decision,
    "build_project_frontend": _exec_build_project_frontend,
    "retarget_queued_job": _exec_retarget_queued_job,
    "cancel_watched_job": lambda p: {"blocked": True, "executed": False, "error": "owned main chat Session must handle cancellation"},
    "apply_workflow_patch": lambda p: {"blocked":True,"executed":False,"error":"owned main chat Session must handle patch application"},
    "discard_workflow_patch": lambda p: {"blocked": True, "executed": False, "error": "owned main chat Session must withdraw the proposal"},
    "revalidate_workflow_node_outputs": lambda p: {"blocked":True,"executed":False,"error":"owned main chat Session must verify runtime artifacts"},
    "finish_workflow_node": lambda p: {"blocked":True,"executed":False,"error":"owned runtime must verify node completion"},
    "request_user_decision": lambda p: {"error": "main chat Session must handle user negotiation"},
    "reconcile_watched_job": lambda p: {"error": "main chat Session must handle dispatch reconciliation"},
    "supervisor_decision": lambda p: p,
    "record_workflow_draft": lambda p: {"error": "record_workflow_draft must run inside agents.Session"},
    "propose_workflow_patch": lambda p: {"error": "propose_workflow_patch must run inside agents.Session"},
    "execute_workflow": lambda p: {"error": "execute_workflow requires the main chat runtime"},
    "message_workflow_node": lambda p: {"error": "message_workflow_node requires the main chat runtime"},
    "get_tool_schema": lambda p: {"error": "schema discovery must bind to the current registry"},
    "discover_forcefield": _exec_discover_forcefield,
    "inspect_forcefield": _exec_inspect_forcefield,
    "validate_framework_charges": _exec_validate_framework_charges,
    "run_project_regressions": _exec_run_project_regressions,
    "convert_physical_units": _exec_convert_physical_units,
    "resolve_local_workflow_write": lambda p: {"error": "resolve_local_workflow_write requires the main chat runtime"},
    "run_henry_chain": _exec_henry_chain,
    "run_isotherm_chain": _exec_isotherm_chain,
    "generate_scientific_report": _exec_generate_report,
    "validate_gcmc_results": _exec_validate_gcmc,
    "run_ga_optimization": _exec_ga_optimization,
    "validate_ga_result": _exec_validate_ga_result,
    "build_mof_database": _exec_build_mof_database,
    "validate_method": _exec_validate_method,
    "check_conclusion_reliability": _exec_check_conclusion_reliability,
    "verify_data_authenticity": _exec_verify_data_authenticity,
    "download_scientific_file": _exec_download_scientific_file,
    "analyze_diffusion_msd": _exec_analyze_diffusion_msd,
}


def _build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    # These legacy schema fields were accepted but never applied by their
    # executors. Do not advertise fictitious capabilities or silently ignore them.
    unsupported = {'run_pacman_charge': {'scheduler'},
                   'check_job': {'scheduler'}, 'run_string_tst': {'gases'},
                   'calc_binding_energy': {'site_atoms', 'distance'},
                   'ml_active_learning': {'gas', 'temperature'}}
    for name, schema_info in _SCHEMAS.items():
        executor = _EXECUTORS.get(name)
        if executor is None:
            continue
        schema = copy.deepcopy(schema_info['schema'])
        schema['additionalProperties'] = False
        for field in unsupported.get(name, set()):
            schema.get('properties', {}).pop(field, None)
        registry.register(ToolDef(
            name=name,
            description=schema_info["description"],
            input_schema=schema,
            execute=executor,
        ))
    return registry
