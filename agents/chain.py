"""Scientific Chain — 程序化科学计算链条

核心理念：
1. 每个步骤有明确的输入/输出契约（不靠提示词，靠代码）
2. 实例内线程锁确保固定链步骤顺序；跨进程依赖由 TaskLine 门禁负责
3. 上下文（路径、结果、错误）在步骤间显式传递
4. 错误诊断结果直接注入下一步的上下文，而不是靠agent自己去"反思"

使用方式：
    chain = ScientificChain("mof_analysis")
    chain.add_step("find_cif", FindCifStep())
    chain.add_step("charge", ChargeStep())
    chain.add_step("gcmc", GcmcStep())
    chain.add_step("analysis", AnalysisStep())
    
    # 自动按顺序执行，每步自动获得前序步骤的上下文
    result = chain.run(initial_context={"material": "Ni-MOF-74", "gas": "CO2"})
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """每一步的执行结果"""
    step_name: str
    status: StepStatus
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    diagnosis: Optional[str] = None  # 错误诊断结果
    retry_count: int = 0
    duration_seconds: float = 0.0
    
    @property
    def success(self) -> bool:
        return self.status == StepStatus.SUCCESS
    
    def to_context(self) -> Dict[str, Any]:
        """转换为可注入下一步的上下文"""
        ctx = {
            f"{self.step_name}_status": self.status.value,
            f"{self.step_name}_output": self.output,
        }
        if self.error:
            ctx[f"{self.step_name}_error"] = self.error
        if self.diagnosis:
            ctx[f"{self.step_name}_diagnosis"] = self.diagnosis
        return ctx


@dataclass
class ChainContext:
    """链条上下文 - 在步骤间传递的状态"""
    data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, key: str, value: Any):
        """线程安全地更新上下文"""
        with self._lock:
            self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def record_step(self, result: StepResult):
        """记录步骤结果"""
        with self._lock:
            self.step_results[result.step_name] = result
            # Store step output with step_name prefix (for history)
            self.data.update(result.to_context())
            # Also flatten output fields into context for downstream steps
            # This allows downstream steps to access output fields directly
            # e.g., FindCifStep.output["cif_path"] becomes context["cif_path"]
            if result.output:
                self.data.update(result.output)
            if result.error:
                self.errors.append({
                    "step": result.step_name,
                    "error": result.error,
                    "diagnosis": result.diagnosis,
                    "timestamp": time.time(),
                })
    
    def get_previous_results(self, before_step: str) -> Dict[str, Any]:
        """获取指定步骤之前的所有步骤结果"""
        result = {}
        for name, sr in self.step_results.items():
            if name == before_step:
                break
            result.update(sr.to_context())
        return result
    
    def get_error_summary(self) -> str:
        """获取所有错误的摘要"""
        if not self.errors:
            return ""
        lines = ["=== 历史错误摘要 ==="]
        for err in self.errors:
            lines.append(f"[{err['step']}] {err['error'][:200]}")
            if err.get('diagnosis'):
                lines.append(f"  诊断: {err['diagnosis'][:200]}")
        return "\n".join(lines)
    
    def to_prompt_context(self) -> str:
        """将上下文转换为可注入prompt的文本"""
        parts = ["=== 链条上下文 ==="]
        
        # 前序步骤的结果
        for name, sr in self.step_results.items():
            if sr.success:
                parts.append(f"✅ {name}: 成功")
                for k, v in sr.output.items():
                    parts.append(f"   {k}: {v}")
            else:
                parts.append(f"❌ {name}: 失败 - {sr.error[:100] if sr.error else '未知'}")
        
        # 关键数据
        parts.append("\n=== 可用数据 ===")
        for k, v in self.data.items():
            if not k.endswith(("_status", "_output", "_error", "_diagnosis")):
                parts.append(f"  {k}: {v}")
        
        # 错误历史
        err_summary = self.get_error_summary()
        if err_summary:
            parts.append(f"\n{err_summary}")
        
        return "\n".join(parts)


class ChainStep(ABC):
    """链条步骤的基类"""
    
    # 子类定义的输入契约 - 必须有哪些字段
    required_inputs: List[str] = []
    # 子类定义的输出契约 - 会产出哪些字段
    output_fields: List[str] = []
    # 最大重试次数
    max_retries: int = 3
    
    @abstractmethod
    def execute(self, context: ChainContext) -> StepResult:
        """执行步骤，返回结果"""
        pass
    
    def validate_inputs(self, context: ChainContext) -> Optional[str]:
        """验证输入是否满足契约，返回错误信息或None"""
        missing = [f for f in self.required_inputs if context.get(f) is None]
        if missing:
            return f"缺少必要输入: {missing}"
        return None
    
    def diagnose_error(self, error: str, context: ChainContext) -> str:
        """诊断错误原因，返回诊断结果"""
        # 默认实现：检查常见错误模式
        diagnosis_parts = []
        
        if "No such file" in error or "not found" in error:
            diagnosis_parts.append("文件不存在 - 检查路径是否正确")
        if "sbatch" in error and "submit.sh" in error:
            diagnosis_parts.append("递归调用错误 - 应使用 run_sim 而非 sbatch submit.sh")
        if "permission denied" in error.lower():
            diagnosis_parts.append("权限问题 - 检查文件权限")
        if "timeout" in error.lower():
            diagnosis_parts.append("超时 - 可能需要增加时间限制或优化计算")
        if "memory" in error.lower() or "oom" in error.lower():
            diagnosis_parts.append("内存不足 - 减小计算规模或增加资源")
        
        # 检查上下文中的历史错误
        err_summary = context.get_error_summary()
        if err_summary:
            diagnosis_parts.append(f"历史错误:\n{err_summary[:500]}")
        
        return "\n".join(diagnosis_parts) if diagnosis_parts else "无法自动诊断，请检查日志"


class Chain:
    """科学计算链条 - 程序化驱动agent执行"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[tuple[str, ChainStep]] = []
        self._lock = threading.Lock()  # instance-local thread lock
        self._callbacks: Dict[str, List[Callable]] = {
            "on_step_start": [],
            "on_step_complete": [],
            "on_step_fail": [],
            "on_chain_complete": [],
        }
    
    def add_step(self, name: str, step: ChainStep) -> "Chain":
        """添加步骤"""
        self.steps.append((name, step))
        return self
    
    def on(self, event: str, callback: Callable) -> "Chain":
        """注册回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        return self
    
    def _emit(self, event: str, **kwargs):
        """触发回调"""
        for cb in self._callbacks.get(event, []):
            try:
                cb(**kwargs)
            except Exception:
                pass
    
    def run(self, initial_context: Optional[Dict[str, Any]] = None, 
            session=None) -> ChainContext:
        """执行整个链条
        
        Args:
            initial_context: 初始上下文数据
            session: 可选的Session对象，用于注入消息
        """
        context = ChainContext(data=initial_context or {})
        
        for step_name, step in self.steps:
            # 实例内线程锁：确保本条固定链顺序执行
            with self._lock:
                print(f"\n{'='*60}", flush=True)
                print(f"🔗 Chain [{self.name}] Step: {step_name}", flush=True)
                print(f"{'='*60}", flush=True)
                
                # 验证输入
                validation_error = step.validate_inputs(context)
                if validation_error:
                    result = StepResult(
                        step_name=step_name,
                        status=StepStatus.SKIPPED,
                        error=f"输入验证失败: {validation_error}",
                    )
                    context.record_step(result)
                    print(f"  ⏭️ 跳过 {step_name}: {validation_error}", flush=True)
                    continue
                
                # 重试循环
                last_result = None
                for attempt in range(step.max_retries):
                    self._emit("on_step_start", step_name=step_name, attempt=attempt)
                    
                    start_time = time.time()
                    try:
                        # 执行步骤
                        result = step.execute(context)
                        result.duration_seconds = time.time() - start_time
                        result.retry_count = attempt
                        if result.status == StepStatus.RUNNING:
                            context.record_step(result)
                            return context  # scheduler owns the wait; never repeat submission here
                        
                        if result.success:
                            context.record_step(result)
                            self._emit("on_step_complete", step_name=step_name, result=result)
                            print(f"  ✅ {step_name} 完成 ({result.duration_seconds:.1f}s)", flush=True)
                            break
                        else:
                            # 失败 - 自动诊断
                            if result.error:
                                diagnosis = step.diagnose_error(result.error, context)
                                result.diagnosis = diagnosis
                            
                            context.record_step(result)
                            last_result = result
                            
                            print(f"  ❌ {step_name} 失败 (尝试 {attempt+1}/{step.max_retries})", flush=True)
                            print(f"     错误: {result.error[:200] if result.error else '未知'}", flush=True)
                            if result.diagnosis:
                                print(f"     诊断: {result.diagnosis[:200]}", flush=True)
                            
                            # 如果还有重试机会，注入诊断信息到上下文
                            if attempt < step.max_retries - 1:
                                context.update(f"{step_name}_last_error", result.error)
                                context.update(f"{step_name}_diagnosis", result.diagnosis)
                                print(f"     → 诊断结果已注入上下文，准备重试...", flush=True)
                    
                    except Exception as e:
                        result = StepResult(
                            step_name=step_name,
                            status=StepStatus.FAILED,
                            error=str(e),
                            duration_seconds=time.time() - start_time,
                            retry_count=attempt,
                        )
                        result.diagnosis = step.diagnose_error(str(e), context)
                        context.record_step(result)
                        last_result = result
                        print(f"  💥 {step_name} 异常: {e}", flush=True)
                
                # 所有重试都失败
                if last_result and not last_result.success:
                    self._emit("on_step_fail", step_name=step_name, result=last_result)
                    print(f"\n⚠️ 步骤 {step_name} 在 {step.max_retries} 次尝试后仍然失败", flush=True)
                    print(f"   最终错误: {last_result.error[:300]}", flush=True)
                    print(f"   诊断: {last_result.diagnosis[:300] if last_result.diagnosis else '无'}", flush=True)
                    # A failed dependency terminates this deterministic chain.
                    # Continuing only creates misleading SKIPPED records and can
                    # never satisfy downstream input contracts.
                    break
        
        self._emit("on_chain_complete", context=context)
        return context


# ── 具体步骤实现 ─────────────────────────────────────────────────────

class FindCifStep(ChainStep):
    """查找CIF文件的步骤"""
    required_inputs = ["material"]
    output_fields = ["cif_path", "cif_name"]
    
    def execute(self, context: ChainContext) -> StepResult:
        from .registry import get_registry
        registry = get_registry()

        material = context.get("material")
        result = registry.execute_dict("find_cif", {"name": material})

        if result.get("error"):
            return StepResult(
                step_name="find_cif",
                status=StepStatus.FAILED,
                error=result["error"],
            )

        cif_path = result.get("cif_path", "")
        if not cif_path:
            return StepResult(
                step_name="find_cif",
                status=StepStatus.FAILED,
                error=f"未找到材料 {material} 的CIF文件",
            )

        return StepResult(
            step_name="find_cif",
            status=StepStatus.SUCCESS,
            output={"cif_path": cif_path, "cif_name": material},
        )


class ChargeStep(ChainStep):
    """PACMAN赋电荷的步骤"""
    required_inputs = ["cif_path"]
    output_fields = ["charged_cif_path"]
    max_retries = 1  # submission is not safe to duplicate automatically
    
    def execute(self, context: ChainContext) -> StepResult:
        from .registry import get_registry
        from .slurm import _cif_has_charges, _find_charged_cif
        
        registry = get_registry()
        cif_path = context.get("cif_path")
        from pathlib import Path
        try:
            from .workspace import session_dir
            for suffix in ('_pacmof.cif', '_pacman.cif'):
                candidate = session_dir('charged') / (Path(cif_path).stem + suffix)
                if candidate.is_file() and _cif_has_charges(str(candidate)):
                    return StepResult(step_name='charge', status=StepStatus.SUCCESS,
                                      output={'charged_cif_path': str(candidate), 'charge_method': 'verified_session_output'})
        except (OSError, ValueError):
            pass
        
        # 检查是否已有电荷
        if _cif_has_charges(cif_path):
            return StepResult(
                step_name="charge",
                status=StepStatus.SUCCESS,
                output={"charged_cif_path": cif_path, "charge_method": "existing"},
            )
        
        # 尝试找已有的带电荷版本
        charged_cif = _find_charged_cif(cif_path)
        if charged_cif != cif_path:
            return StepResult(
                step_name="charge",
                status=StepStatus.SUCCESS,
                output={"charged_cif_path": charged_cif, "charge_method": "pre_existing"},
            )
        
        # 需要运行PACMAN赋电荷
        result = registry.execute_dict("run_pacman_charge", {"cif_path": cif_path})
        
        if result.get("error"):
            return StepResult(
                step_name="charge",
                status=StepStatus.FAILED,
                error=result["error"],
            )
        
        # 等待PACMAN完成
        job_id = result.get("job_id")
        work_dir = result.get("work_dir", "")
        
        if job_id:
            from .slurm import check_job_status
            import time
            
            terminal = False
            for _ in range(30):  # 最多等150秒，保持在Session工具超时内
                time.sleep(5)
                st = check_job_status(job_id, work_dir=work_dir)
                if st.get("terminal"):
                    terminal = True
                    if st.get("failed"):
                        return StepResult(
                            step_name="charge",
                            status=StepStatus.FAILED,
                            error=f"PACMAN作业失败: {st.get('error', '未知')}",
                        )
                    break
            if not terminal:
                return StepResult(
                    step_name="charge",
                    status=StepStatus.RUNNING,
                    error=f"PACMAN作业仍在运行: job_id={job_id}; 等待JobWatch完成提醒后重跑链条",
                    output={"charge_job_id": job_id, "charge_work_dir": work_dir},
                )
        
        # 查找生成的带电荷CIF
        import os
        from pathlib import Path
        
        work_path = Path(work_dir)
        charged_files = [work_path / (Path(cif_path).stem + suffix) for suffix in ('_pacmof.cif', '_pacman.cif')]
        charged_files = [p for p in charged_files if p.is_file() and _cif_has_charges(str(p))]
        if charged_files:
            return StepResult(
                step_name="charge",
                status=StepStatus.SUCCESS,
                output={"charged_cif_path": str(charged_files[0]), "charge_method": "pacman"},
            )
        
        return StepResult(
            step_name="charge",
            status=StepStatus.FAILED,
            error="PACMAN完成但未找到带电荷的CIF文件",
        )


class HenryStep(ChainStep):
    """Henry系数计算步骤"""
    required_inputs = ["charged_cif_path", "gas"]
    output_fields = ["henry_coefficient", "henry_job_id"]
    max_retries = 1  # every resubmission must return through Session's recovery gate
    
    def execute(self, context: ChainContext) -> StepResult:
        from .registry import get_registry
        registry = get_registry()
        
        cif_path = context.get("charged_cif_path")
        gas = context.get("gas", "CO2")
        temperature = context.get("temperature", 298.0)
        
        result = registry.execute_dict("run_henry", {
            "cif": cif_path,
            "gas": gas,
            "temperature": temperature,
        })
        
        if result.get("error"):
            return StepResult(
                step_name="henry",
                status=StepStatus.FAILED,
                error=result["error"],
            )
        
        return StepResult(
            step_name="henry",
            status=StepStatus.SUCCESS,
            output={
                "henry_job_id": result.get("job_id"),
                "henry_work_dir": result.get("work_dir"),
                "henry_status": result.get("status"),
            },
        )
    
    def diagnose_error(self, error: str, context: ChainContext) -> str:
        """Henry特定的错误诊断"""
        diagnosis_parts = []
        
        if "sbatch submit.sh" in error:
            diagnosis_parts.append(
                "🚨 递归调用错误！submit.sh 中包含了 'sbatch submit.sh'，"
                "这会导致无限递归。应使用 run_sim simulation.input 作为命令。"
            )
        
        if "UseChargesFromCIFFile" in error:
            diagnosis_parts.append(
                "CIF文件缺少电荷信息。需要先运行PACMAN赋电荷步骤。"
            )
        
        # 检查前序步骤
        charge_status = context.get("charge_status")
        if charge_status == "failed":
            diagnosis_parts.append(
                "前置步骤（赋电荷）失败，无法继续Henry计算。"
                "需要先解决电荷问题。"
            )
        
        base = super().diagnose_error(error, context)
        if base:
            diagnosis_parts.append(base)
        
        return "\n".join(diagnosis_parts)


class GcmcStep(ChainStep):
    """GCMC等温线计算步骤"""
    required_inputs = ["charged_cif_path", "gas", "pressures"]
    output_fields = ["gcmc_job_ids", "gcmc_work_dirs"]
    max_retries = 1

    def execute(self, context: ChainContext) -> StepResult:
        from .registry import get_registry
        registry = get_registry()

        cif_path = context.get("charged_cif_path")
        gas = context.get("gas", "CO2")
        pressures = context.get("pressures", [0.1, 0.5, 1.0, 5.0, 10.0])
        temperature = context.get("temperature", 298.0)

        result = registry.execute_dict("run_gcmc_isotherm", {
            "cif": cif_path,
            "gas": gas,
            "pressure_start": min(pressures),
            "pressure_end": max(pressures),
            "n_pressure_points": len(pressures),
            "temperature": temperature,
        })

        if result.get("error"):
            return StepResult(
                step_name="gcmc",
                status=StepStatus.FAILED,
                error=result["error"],
            )

        # 处理job_id: submit_gcmc_isotherm返回单个job_id，需要转换为列表
        job_id = result.get("job_id", "")
        job_ids = [job_id] if job_id else result.get("job_ids", [])

        return StepResult(
            step_name="gcmc",
            status=StepStatus.SUCCESS,
            output={
                "gcmc_job_ids": job_ids,
                "gcmc_work_dirs": [result.get("work_dir", "")],
                "gcmc_status": result.get("status"),
            },
        )


class WaitJobStep(ChainStep):
    """等待作业完成的步骤（通用）"""
    required_inputs = ["job_ids"]  # 或者具体的 job_id 字段
    output_fields = ["job_results"]
    
    def __init__(self, job_id_field: str = "job_ids", timeout_minutes: int = 30):
        self.job_id_field = job_id_field
        self.timeout_minutes = timeout_minutes
        self.required_inputs = [job_id_field]
    
    def execute(self, context: ChainContext) -> StepResult:
        from .slurm import check_job_status
        
        job_ids = context.get(self.job_id_field, [])
        if isinstance(job_ids, str):
            job_ids = [job_ids]
        
        if not job_ids:
            return StepResult(
                step_name="wait_job",
                status=StepStatus.FAILED,
                error=f"没有找到作业ID ({self.job_id_field})",
            )
        
        results = {}
        start_time = time.time()
        max_wait = self.timeout_minutes * 60
        
        while time.time() - start_time < max_wait:
            all_done = True
            for jid in job_ids:
                if jid in results and results[jid].get("terminal"):
                    continue
                st = check_job_status(jid)
                results[jid] = st
                if not st.get("terminal"):
                    all_done = False
            
            if all_done:
                break
            
            time.sleep(10)
        
        # 检查是否有失败的
        failed = [jid for jid, st in results.items() if st.get("failed")]
        
        if failed:
            return StepResult(
                step_name="wait_job",
                status=StepStatus.FAILED,
                error=f"作业失败: {failed}",
                output={"job_results": results},
            )

        unfinished = [jid for jid in job_ids if not results.get(jid, {}).get("terminal")]
        if unfinished:
            return StepResult(
                step_name="wait_job",
                status=StepStatus.FAILED,
                error=f"等待作业超时，仍未终止: {unfinished}",
                output={"job_results": results},
            )
        
        return StepResult(
            step_name="wait_job",
            status=StepStatus.SUCCESS,
            output={"job_results": results},
        )


# ── 预定义的科学计算链条 ──────────────────────────────────────────────

def create_henry_chain(material: str, gas: str, temperature: float = 298.0) -> Chain:
    """创建Henry系数计算链条"""
    chain = Chain("henry_calculation", f"{material} + {gas} Henry系数计算")

    chain.add_step("find_cif", FindCifStep())
    chain.add_step("charge", ChargeStep())
    chain.add_step("henry", HenryStep())
    # Submission is asynchronous. JobWatch owns monitoring and will resume the
    # conversation on completion/failure; blocking a tool thread for 30 minutes
    # defeats interruption and per-tool timeout semantics.

    return chain


def create_isotherm_chain(material: str, gas: str, pressures: List[float],
                          temperature: float = 298.0) -> Chain:
    """创建GCMC等温线计算链条"""
    chain = Chain("isotherm_calculation", f"{material} + {gas} 等温线计算")
    
    chain.add_step("find_cif", FindCifStep())
    chain.add_step("charge", ChargeStep())
    chain.add_step("gcmc", GcmcStep())
    
    return chain


# ── Chain-aware Session 集成 ──────────────────────────────────────────

class ChainRunner:
    """在Session中运行Chain的适配器
    
    将Chain的执行过程注入Session的消息流中，
    使得agent能看到链条的每一步进展。
    """
    
    def __init__(self, session, chain: Chain):
        self.session = session
        self.chain = chain
    
    def run(self, initial_context: Dict[str, Any]) -> ChainContext:
        """运行链条，并将每步结果注入Session"""
        
        # 注册回调，将进度注入Session
        def on_step_start(step_name, attempt):
            self.session.messages.append({
                "role": "user",
                "content": f"[链条·{self.chain.name}] 开始执行步骤: {step_name} (尝试 {attempt+1})"
            })
        
        def on_step_complete(step_name, result):
            output_summary = json.dumps(result.output, ensure_ascii=False, indent=2)[:500]
            self.session.messages.append({
                "role": "user",
                "content": f"[链条·{self.chain.name}] ✅ {step_name} 完成\n输出: {output_summary}"
            })
        
        def on_step_fail(step_name, result):
            error_msg = f"[链条·{self.chain.name}] ❌ {step_name} 失败\n错误: {result.error[:300]}"
            if result.diagnosis:
                error_msg += f"\n诊断: {result.diagnosis[:300]}"
            self.session.messages.append({
                "role": "user",
                "content": error_msg
            })
        
        self.chain.on("on_step_start", on_step_start)
        self.chain.on("on_step_complete", on_step_complete)
        self.chain.on("on_step_fail", on_step_fail)
        
        # 运行链条
        context = self.chain.run(initial_context)
        
        # 最终报告
        summary = context.to_prompt_context()
        self.session.messages.append({
            "role": "user",
            "content": f"[链条·{self.chain.name}] 执行完成\n{summary}"
        })
        
        return context


# ── 工具注册 ──────────────────────────────────────────────────────────

def register_chain_tools(registry):
    """将链条相关工具注册到registry"""
    
    def run_henry_chain(params):
        """运行Henry系数计算链条（程序化，非提示词驱动）"""
        material = params.get("material", "")
        gas = params.get("gas", "CO2")
        temperature = params.get("temperature", 298.0)
        
        chain = create_henry_chain(material, gas, temperature)
        context = chain.run({
            "material": material,
            "gas": gas,
            "temperature": temperature,
        })
        
        return {
            "chain_status": "completed",
            "context": context.data,
            "errors": context.errors,
            "step_results": {
                name: {
                    "status": sr.status.value,
                    "output": sr.output,
                    "error": sr.error,
                    "diagnosis": sr.diagnosis,
                }
                for name, sr in context.step_results.items()
            }
        }
    
    registry.register("run_henry_chain", run_henry_chain, {
        "description": "运行Henry系数计算链条（程序化执行，自动处理依赖和错误诊断）",
        "parameters": {
            "material": {"type": "string", "description": "材料名称"},
            "gas": {"type": "string", "description": "气体分子", "default": "CO2"},
            "temperature": {"type": "number", "description": "温度(K)", "default": 298.0},
        }
    })
    
    def run_isotherm_chain(params):
        """运行GCMC等温线计算链条"""
        material = params.get("material", "")
        gas = params.get("gas", "CO2")
        pressures = params.get("pressures", [0.1, 0.5, 1.0, 5.0, 10.0])
        temperature = params.get("temperature", 298.0)
        
        chain = create_isotherm_chain(material, gas, pressures, temperature)
        context = chain.run({
            "material": material,
            "gas": gas,
            "pressures": pressures,
            "temperature": temperature,
        })
        
        return {
            "chain_status": "completed",
            "context": context.data,
            "errors": context.errors,
            "step_results": {
                name: {
                    "status": sr.status.value,
                    "output": sr.output,
                    "error": sr.error,
                    "diagnosis": sr.diagnosis,
                }
                for name, sr in context.step_results.items()
            }
        }
    
    registry.register("run_isotherm_chain", run_isotherm_chain, {
        "description": "运行GCMC等温线计算链条（程序化执行）",
        "parameters": {
            "material": {"type": "string", "description": "材料名称"},
            "gas": {"type": "string", "description": "气体分子", "default": "CO2"},
            "pressures": {"type": "array", "description": "压力点列表(bar)", "items": {"type": "number"}},
            "temperature": {"type": "number", "description": "温度(K)", "default": 298.0},
        }
    })
