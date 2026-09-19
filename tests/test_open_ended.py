#!/usr/bin/env python3
"""开放式科学任务测试 - 不提示步骤，评估智能体规划能力"""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.session import Session
from agents.config import get_config
from agents.defns import ORCHESTRATOR, ADSORPTION, ANALYST, COMMUNICATOR, HARNESS

# ── Mock数据 ──────────────────────────────────────────────────────
MOCK = {
    "henry": {"henry_coefficient": "K0=45.2 mol/(kg*Pa)", "heat_of_adsorption": "Qst=-28.5 kJ/mol", "status": "COMPLETED"},
    "gcmc": {"isotherm_data": [{"pressure": "1.0", "loading": "3.2 molecules/unit cell"}], "status": "COMPLETED"},
    "gcmc_batch": {"completed_count": 5, "results": [{"status": "COMPLETED"}]*5, "status": "COMPLETED"},
    "pore": {"pore_analysis": "Pore diameter: 11.2 A, Surface area: 2850 m2/g, Volume: 0.85 cm3/g", "status": "COMPLETED"},
    "charge": {
        "charged_cifs": ["/tmp/charged_MOF-5.cif"],
        "charge_type": "DDEC6",
        "status": "COMPLETED",
        "charges": {"Zn1": -0.82, "Zn2": -0.79, "O1": -0.55, "C1": 0.12},
        "charge_summary": "Zn: -0.81e, O: -0.54e, C: 0.07e"
    },
    "cdft": {
        "status": "COMPLETED",
        "density_profile": [
            {"distance_A": 0.0, "density_g_cm3": 0.000},
            {"distance_A": 3.0, "density_g_cm3": 0.182},
            {"distance_A": 6.0, "density_g_cm3": 0.278},
        ],
        "electrostatic_potential": {"min": -45.2, "max": 38.7, "avg": -2.3}
    },
    "binding": {
        "binding_energy": "-32.5 kJ/mol",
        "interaction_breakdown": {"electrostatic": "-18.3 kJ/mol", "vdW_LJ": "-14.2 kJ/mol"},
        "status": "COMPLETED"
    },
    "tst": {"energy_barrier": "15.2 kJ/mol", "diffusion_coefficient": "2.3e-9 m2/s", "status": "COMPLETED"},
    "md": {"msd_slope": 1.2e-8, "diffusion_coefficient": "3.0e-9 m2/s", "status": "COMPLETED"},
    "ml_train": {"model_type": "GBR", "r2_score": 0.87, "mae": 0.12, "status": "COMPLETED"},
    "ml_feature_importance": {"features": {"pore_volume": 0.35, "surface_area": 0.28, "pore_diameter": 0.22}},
    "generate_structure": {"cif_path": "/tmp/generated_MOF.cif", "status": "COMPLETED"},
    "external_potential": {"barrier_height": "12.5 kJ/mol", "status": "COMPLETED"},
    "literature": {
        "results": [
            {"title": "MOF for CO2 capture", "source": "Nature Chemistry 2020", "relevance": 0.95},
            {"title": "GCMC simulation of gas adsorption", "source": "JACS 2019", "relevance": 0.88},
        ],
        "summary": "MOF材料在CO2捕获中表现出优异性能，比表面积可达7000 m2/g。"
    }
}

def mock_submit(job_type, params, **kwargs):
    return MOCK.get(job_type, {"status": "COMPLETED"})

def mock_gcmc(params, **kwargs):
    return MOCK["gcmc"]

# ── 评分标准 ──────────────────────────────────────────────────────
SCORING_RUBRIC = {
    "planning": {
        "description": "规划能力 - 是否正确理解任务并规划步骤",
        "max_score": 20,
        "criteria": {
            "understands_goal": 5,      # 理解研究目标
            "identifies_steps": 5,      # 识别必要步骤
            "logical_order": 5,         # 逻辑顺序
            "considers_constraints": 5  # 考虑约束条件
        }
    },
    "tool_selection": {
        "description": "工具选择 - 是否选择合适的工具",
        "max_score": 20,
        "criteria": {
            "correct_tools": 8,         # 选择正确工具
            "no_unnecessary": 4,        # 无多余工具
            "right_sequence": 4,        # 正确顺序
            "handles_missing": 4        # 处理缺失工具
        }
    },
    "agent_delegation": {
        "description": "智能体委派 - 是否委派给正确的智能体",
        "max_score": 20,
        "criteria": {
            "right_agents": 8,          # 委派给正确智能体
            "appropriate_split": 4,     # 任务拆分合理
            "avoids_over_delegation": 4,# 避免过度委派
            "handles_failure": 4        # 处理委派失败
        }
    },
    "execution": {
        "description": "执行效率 - 工具调用和时间",
        "max_score": 20,
        "criteria": {
            "completes_tasks": 8,       # 完成任务
            "efficient_calls": 4,       # 高效调用
            "handles_errors": 4,        # 错误处理
            "iterates_if_needed": 4     # 必要时迭代
        }
    },
    "output_quality": {
        "description": "输出质量 - 报告完整性",
        "max_score": 20,
        "criteria": {
            "has_structure": 5,         # 结构完整(背景/方法/结果/结论)
            "has_data": 5,              # 包含数据
            "addresses_question": 5,    # 回答了问题
            "scientific_rigor": 5       # 科学严谨性
        }
    }
}

# ── 开放式任务定义 ──────────────────────────────────────────────
# 关键: 只给研究问题，不给步骤！

OPEN_TASKS = [
    # ══════════════════════════════════════════════════════════════
    # 复杂科学任务 (5个) - 测试完整研究流程
    # ══════════════════════════════════════════════════════════════
    {
        "id": "S01",
        "name": "CO2捕获MOF筛选",
        "user_message": "我想找一种能高效捕获CO2的MOF材料，用于烟道气处理。请帮我完成这个研究。",
        "expected_agents": ["lead-orchestrator", "communicator", "adsorption", "analyst"],
        "expected_tools": ["query_literature", "find_cif", "run_gcmc_isotherm", "run_henry", "extract_features"],
        "scoring_key": {
            "planning": "应该规划：文献调研→材料筛选→吸附计算→结果分析",
            "tool_selection": "应该使用：query_literature, find_cif, run_gcmc_isotherm",
            "agent_delegation": "应该委派：communicator(文献), adsorption(计算), analyst(分析)",
            "execution": "应该完成至少3个工具调用",
            "output": "应该包含：候选材料、吸附数据、推荐理由"
        }
    },
    {
        "id": "S02",
        "name": "MOF电荷与吸附关系",
        "user_message": "MOF的电荷分布如何影响气体吸附？我想理解这个机制。",
        "expected_agents": ["lead-orchestrator", "analyst", "communicator"],
        "expected_tools": ["query_literature", "find_cif", "run_pacman_charge", "calc_binding_energy"],
        "scoring_key": {
            "planning": "应该规划：文献调研→电荷计算→结合能分析→机制讨论",
            "tool_selection": "应该使用：run_pacman_charge, calc_binding_energy；若研究平衡吸附分布才使用经典cDFT",
            "agent_delegation": "应该委派：analyst(框架电荷/结合能), communicator(文献)",
            "execution": "应该完成电荷和结合能计算",
            "output": "应该包含：电荷数据、结合能、机制解释"
        }
    },
    {
        "id": "S03",
        "name": "MOF扩散性能研究",
        "user_message": "CO2在Ni-MOF-74中的扩散行为是怎样的？这对膜分离设计有什么启示？",
        "expected_agents": ["lead-orchestrator", "adsorption", "analyst", "communicator"],
        "expected_tools": ["query_literature", "find_cif", "run_pore_analysis", "run_string_tst", "run_md_optimize"],
        "scoring_key": {
            "planning": "应该规划：结构分析→扩散计算→机制研究→膜设计建议",
            "tool_selection": "应该使用：run_pore_analysis, run_string_tst或run_md_optimize",
            "agent_delegation": "应该委派：adsorption(扩散), analyst(分析), communicator(文献)",
            "execution": "应该完成扩散系数计算",
            "output": "应该包含：扩散系数、能垒、膜设计建议"
        }
    },
    {
        "id": "S04",
        "name": "MOF ML预测模型",
        "user_message": "我想用机器学习预测MOF的吸附性能，应该怎么做？",
        "expected_agents": ["lead-orchestrator", "analyst", "harness-maintainer", "communicator"],
        "expected_tools": ["query_literature", "find_cif", "extract_features", "ml_train", "ml_feature_importance"],
        "scoring_key": {
            "planning": "应该规划：文献调研→数据获取→特征提取→模型训练→评估",
            "tool_selection": "应该使用：extract_features, ml_train, ml_feature_importance",
            "agent_delegation": "应该委派：analyst(ML), harness(数据), communicator(文献)",
            "execution": "应该完成模型训练和特征重要性分析",
            "output": "应该包含：模型性能、特征重要性、预测结果"
        }
    },
    {
        "id": "S05",
        "name": "MOF结构设计",
        "user_message": "如何设计一种新型MOF用于CH4/H2分离？",
        "expected_agents": ["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        "expected_tools": ["query_literature", "find_cif", "generate_structure", "run_gcmc_batch", "extract_features"],
        "scoring_key": {
            "planning": "应该规划：文献调研→设计原则→结构生成→性能验证→优化",
            "tool_selection": "应该使用：generate_structure, run_gcmc_batch",
            "agent_delegation": "应该委派：harness(生成), adsorption(验证), communicator(文献)",
            "execution": "应该完成结构生成和吸附验证",
            "output": "应该包含：设计原则、候选结构、性能数据"
        }
    },

    # ══════════════════════════════════════════════════════════════
    # 文献调研任务 (5个) - 测试文献检索和综合能力
    # ══════════════════════════════════════════════════════════════
    {
        "id": "L01",
        "name": "MOF吸附综述",
        "user_message": "请综述MOF材料在气体吸附领域的最新研究进展。",
        "expected_agents": ["lead-orchestrator", "communicator"],
        "expected_tools": ["query_literature"],
        "scoring_key": {
            "planning": "应该规划：分类检索→整理→综合分析",
            "tool_selection": "应该使用：query_literature",
            "agent_delegation": "应该委派：communicator",
            "execution": "应该完成多次文献检索",
            "output": "应该包含：分类综述、关键发现、研究趋势"
        }
    },
    {
        "id": "L02",
        "name": "cDFT方法对比",
        "user_message": "DDEC6和CM5电荷计算方法有什么区别？各自适用什么场景？",
        "expected_agents": ["lead-orchestrator", "communicator"],
        "expected_tools": ["query_literature"],
        "scoring_key": {
            "planning": "应该规划：分别检索两种方法→对比→总结适用场景",
            "tool_selection": "应该使用：query_literature",
            "agent_delegation": "应该委派：communicator",
            "execution": "应该检索DDEC6和CM5相关文献",
            "output": "应该包含：方法原理、精度对比、适用场景"
        }
    },
    {
        "id": "L03",
        "name": "扩散理论综述",
        "user_message": "MOF中气体扩散的主要理论模型有哪些？各自优缺点是什么？",
        "expected_agents": ["lead-orchestrator", "communicator"],
        "expected_tools": ["query_literature"],
        "scoring_key": {
            "planning": "应该规划：检索扩散理论→分类→对比优缺点",
            "tool_selection": "应该使用：query_literature",
            "agent_delegation": "应该委派：communicator",
            "execution": "应该检索TST、MD、kMC等方法",
            "output": "应该包含：理论模型列表、对比表格、推荐场景"
        }
    },
    {
        "id": "L04",
        "name": "ML在MOF中的应用",
        "user_message": "机器学习在MOF研究中有哪些应用？目前的挑战是什么？",
        "expected_agents": ["lead-orchestrator", "communicator"],
        "expected_tools": ["query_literature"],
        "scoring_key": {
            "planning": "应该规划：检索ML应用→分类→分析挑战",
            "tool_selection": "应该使用：query_literature",
            "agent_delegation": "应该委派：communicator",
            "execution": "应该检索多个ML应用领域",
            "output": "应该包含：应用分类、典型案例、挑战分析"
        }
    },
    {
        "id": "L05",
        "name": "膜分离技术对比",
        "user_message": "MOF膜和聚合物膜在气体分离中的性能对比如何？",
        "expected_agents": ["lead-orchestrator", "communicator"],
        "expected_tools": ["query_literature"],
        "scoring_key": {
            "planning": "应该规划：分别检索→性能对比→优缺点分析",
            "tool_selection": "应该使用：query_literature",
            "agent_delegation": "应该委派：communicator",
            "execution": "应该检索两种膜材料的文献",
            "output": "应该包含：性能数据对比、优缺点、应用前景"
        }
    },

    # ══════════════════════════════════════════════════════════════
    # 计算任务 (5个) - 测试工具调用能力
    # ══════════════════════════════════════════════════════════════
    {
        "id": "C01",
        "name": "吸附等温线计算",
        "user_message": "计算MOF-5对CO2的吸附等温线，温度298K，压力0.01-10 bar。",
        "expected_agents": ["lead-orchestrator", "adsorption"],
        "expected_tools": ["find_cif", "run_gcmc_isotherm"],
        "scoring_key": {
            "planning": "应该规划：查找CIF→设置参数→运行GCMC→分析结果",
            "tool_selection": "应该使用：find_cif, run_gcmc_isotherm",
            "agent_delegation": "应该委派：adsorption",
            "execution": "应该完成等温线计算",
            "output": "应该包含：吸附数据、等温线图描述"
        }
    },
    {
        "id": "C02",
        "name": "Henry系数计算",
        "user_message": "计算Ni-MOF-74对CH4的Henry系数和吸附热。",
        "expected_agents": ["lead-orchestrator", "adsorption"],
        "expected_tools": ["find_cif", "run_henry"],
        "scoring_key": {
            "planning": "应该规划：查找CIF→运行Henry计算→分析",
            "tool_selection": "应该使用：find_cif, run_henry",
            "agent_delegation": "应该委派：adsorption",
            "execution": "应该完成Henry系数计算",
            "output": "应该包含：Henry系数、吸附热数值"
        }
    },
    {
        "id": "C03",
        "name": "孔径分析",
        "user_message": "分析MOF-5的孔隙结构特征。",
        "expected_agents": ["lead-orchestrator", "adsorption"],
        "expected_tools": ["find_cif", "run_pore_analysis"],
        "scoring_key": {
            "planning": "应该规划：查找CIF→孔隙分析→解读结果",
            "tool_selection": "应该使用：find_cif, run_pore_analysis",
            "agent_delegation": "应该委派：adsorption",
            "execution": "应该完成孔隙分析",
            "output": "应该包含：孔径、比表面积、孔体积"
        }
    },
    {
        "id": "C04",
        "name": "电荷计算",
        "user_message": "计算MOF-5的DDEC6部分原子电荷。",
        "expected_agents": ["lead-orchestrator", "analyst"],
        "expected_tools": ["find_cif", "run_pacman_charge"],
        "scoring_key": {
            "planning": "应该规划：查找CIF→设置电荷计算参数→运行",
            "tool_selection": "应该使用：find_cif, run_pacman_charge",
            "agent_delegation": "应该委派：analyst",
            "execution": "应该完成电荷计算",
            "output": "应该包含：原子电荷数据"
        }
    },
    {
        "id": "C05",
        "name": "结合能计算",
        "user_message": "计算CO2与MOF-5的结合能。",
        "expected_agents": ["lead-orchestrator", "analyst"],
        "expected_tools": ["find_cif", "calc_binding_energy"],
        "scoring_key": {
            "planning": "应该规划：查找CIF→设置结合能计算→分析能量分解",
            "tool_selection": "应该使用：find_cif, calc_binding_energy",
            "agent_delegation": "应该委派：analyst",
            "execution": "应该完成结合能计算",
            "output": "应该包含：结合能、能量分解"
        }
    },
]

# ── 测试运行器 ──────────────────────────────────────────────────
class OpenTaskRunner:
    def __init__(self, task: dict, verbose: bool = False):
        self.task = task
        self.verbose = verbose
        self.config = get_config()
        self.session = None
        self.result = {}

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def run(self) -> Dict[str, Any]:
        self.log(f"═══ {self.task['name']} ═══")
        t0 = time.time()
        patches = []
        try:
            self.session = Session(config=self.config)
            import agents.slurm as sm
            patches = [
                patch.object(sm, 'submit_and_wait', side_effect=mock_submit),
                patch.object(sm, 'submit_gcmc_isotherm', side_effect=mock_gcmc),
            ]
            for p in patches:
                p.start()

            resp = self.session.start(self.task['user_message'], agent=ORCHESTRATOR,
                                      max_rounds=12, verbose=self.verbose)

            full_resp = resp or ""
            self.result = {
                "task_id": self.task['id'],
                "task_name": self.task['name'],
                "status": "completed" if self.session.task_complete else "incomplete",
                "elapsed": round(time.time()-t0, 1),
                "response": full_resp[:10000],
                "full_response_length": len(full_resp),
                "agent_flow": self.session.memory.agent_history,
                "tool_calls": [{"agent": t["agent"], "tool": t["tool"],
                               "result_preview": t.get("result_preview", "")}
                               for t in self.session.memory.tool_call_log],
                "stats": self.session.memory.get_stats(),
            }
        except Exception as e:
            self.result = {"task_id": self.task['id'], "status": "error",
                           "elapsed": round(time.time()-t0, 1), "error": str(e)}
        finally:
            for p in patches:
                p.stop()

        Path(f"test_result_{self.task['id']}.json").write_text(
            json.dumps(self.result, indent=2, default=str))
        return self.result

    def score(self) -> Dict[str, Any]:
        """按评分标准打分"""
        r = self.result
        resp = r.get('response', '')
        agent_flow = r.get('agent_flow', [])
        tool_calls = r.get('tool_calls', [])
        stats = r.get('stats', {})

        scores = {}

        # 1. 规划能力 (0-20)
        planning = 0
        # 检查是否理解目标
        if any(kw in resp for kw in ['研究', '分析', '计算', '调研', '筛选']):
            planning += 5
        # 检查是否识别步骤
        if any(kw in resp for kw in ['第一步', '首先', '然后', '最后', '步骤']):
            planning += 5
        # 检查逻辑顺序
        if len(resp) > 500:
            planning += 5
        # 检查考虑约束
        if any(kw in resp for kw in ['温度', '压力', '条件', '参数']):
            planning += 5
        scores['planning'] = min(20, planning)

        # 2. 工具选择 (0-20)
        actual_tools = set(tc['tool'] for tc in tool_calls if not tc['tool'].startswith('handoff_to_'))
        expected_tools = set(self.task.get('expected_tools', []))
        
        tool_score = 0
        if expected_tools:
            # 正确工具比例
            correct = len(actual_tools & expected_tools)
            tool_score += min(8, correct * 2)
        # 无多余工具
        unnecessary = len(actual_tools - expected_tools - {'inspect_path', 'return_to_orchestrator'})
        tool_score += max(0, 4 - unnecessary)
        # 正确顺序
        if len(actual_tools) >= 2:
            tool_score += 4
        # 处理缺失
        if 'find_cif' in actual_tools:
            tool_score += 4
        scores['tool_selection'] = min(20, tool_score)

        # 3. 智能体委派 (0-20)
        unique_agents = set()
        for step in agent_flow:
            parts = step.split(' → ')
            if len(parts) == 2:
                unique_agents.add(parts[1])
        unique_agents.discard('lead-orchestrator')
        
        expected_agents = set(self.task.get('expected_agents', [])) - {'lead-orchestrator'}
        
        delegation = 0
        if expected_agents:
            correct = len(unique_agents & expected_agents)
            delegation += min(8, correct * 3)
        # 任务拆分
        if len(unique_agents) >= 2:
            delegation += 4
        # 避免过度委派
        if len(unique_agents) <= 4:
            delegation += 4
        # 处理失败
        delegation += 4  # 默认给分
        scores['agent_delegation'] = min(20, delegation)

        # 4. 执行效率 (0-20)
        execution = 0
        if r['status'] == 'completed':
            execution += 8
        # 高效调用
        total_calls = stats.get('tool_calls', 0)
        if 2 <= total_calls <= 10:
            execution += 4
        # 错误处理
        if stats.get('errors', 0) == 0:
            execution += 4
        # 迭代
        if stats.get('agent_handoffs', 0) >= 2:
            execution += 4
        scores['execution'] = min(20, execution)

        # 5. 输出质量 (0-20)
        output = 0
        # 结构完整
        if '##' in resp or '# ' in resp:
            output += 5
        # 包含数据
        if any(c.isdigit() for c in resp):
            output += 5
        # 回答问题
        if len(resp) > 200:
            output += 5
        # 科学严谨
        if any(kw in resp for kw in ['kcal', 'kJ', 'eV', 'Å', 'K', 'Pa', 'm2/g', 'cm3/g']):
            output += 5
        scores['output_quality'] = min(20, output)

        scores['total'] = sum(scores.values())
        scores['max_total'] = 100
        return scores


def run_all_open_tasks():
    """运行所有开放式任务"""
    all_results = []
    
    for task in OPEN_TASKS:
        print(f"\n{'='*60}")
        print(f"运行: {task['id']} - {task['name']}")
        print(f"{'='*60}")
        
        runner = OpenTaskRunner(task, verbose=True)
        result = runner.run()
        scores = runner.score()
        
        result['scores'] = scores
        all_results.append(result)
        
        # 输出结果
        s = '✅' if result['status'] == 'completed' else '❌'
        print(f"\n{s} 状态: {result['status']}")
        print(f"得分: {scores['total']}/100")
        print(f"  规划: {scores['planning']}/20")
        print(f"  工具: {scores['tool_selection']}/20")
        print(f"  委派: {scores['agent_delegation']}/20")
        print(f"  执行: {scores['execution']}/20")
        print(f"  输出: {scores['output_quality']}/20")
    
    # 汇总
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    
    total_scores = [r['scores']['total'] for r in all_results]
    print(f"总任务数: {len(all_results)}")
    print(f"平均分: {sum(total_scores)/len(total_scores):.1f}")
    print(f"最高分: {max(total_scores)}")
    print(f"最低分: {min(total_scores)}")
    
    # 各维度平均
    for dim in ['planning', 'tool_selection', 'agent_delegation', 'execution', 'output_quality']:
        avg = sum(r['scores'][dim] for r in all_results) / len(all_results)
        print(f"{dim}: {avg:.1f}/20")
    
    # 保存
    with open('test_open_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\n结果已保存: test_open_results.json")


if __name__ == "__main__":
    run_all_open_tasks()
