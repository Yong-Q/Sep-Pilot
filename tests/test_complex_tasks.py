#!/usr/bin/env python3
"""5 complex scientific tasks for BiMemAgent multi-agent system.

Each task is a real research problem requiring all 5 agents.
Uses mock SLURM for fast testing.
"""
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

# ── Mock SLURM ──────────────────────────────────────────────────────
MOCK = {
    "henry": {"henry_coefficient": "K0=45.2 mol/(kg*Pa)", "heat_of_adsorption": "Qst=-28.5 kJ/mol", "status": "COMPLETED"},
    "gcmc": {"isotherm_data": [{"pressure": "1.0", "loading": "3.2 molecules/unit cell"}], "status": "COMPLETED"},
    "gcmc_batch": {"completed_count": 5, "results": [{"status": "COMPLETED"}]*5, "status": "COMPLETED"},
    "pore": {"pore_analysis": "Pore diameter: 11.2 A, Surface area: 2850 m2/g, Volume: 0.85 cm3/g", "status": "COMPLETED"},
    "charge": {
        "charged_cifs": ["/tmp/charged_MOF-5.cif"],
        "charge_type": "DDEC6",
        "status": "COMPLETED",
        "charges": {
            "Zn1": -0.82, "Zn2": -0.79, "O1": -0.55, "O2": -0.53,
            "C1": 0.12, "C2": 0.08, "C3": 0.15, "C4": -0.06,
            "H1": 0.09, "H2": 0.11, "H3": 0.07
        },
        "charge_summary": "Zn: -0.81e (avg), O: -0.54e (avg), C: 0.07e (avg), H: 0.09e (avg). Net charge: 0.00e (charge neutral).",
        "dipole_moment": "12.3 Debye",
        "electronegativity_difference": "Zn-O: 1.65, C-O: 1.12"
    },
    "cdft": {
        "status": "COMPLETED",
        "output_dir": "/tmp/cdft_MOF-5",
        "density_profile": [
            {"distance_A": 0.0, "density_g_cm3": 0.000},
            {"distance_A": 1.5, "density_g_cm3": 0.045},
            {"distance_A": 3.0, "density_g_cm3": 0.182},
            {"distance_A": 4.5, "density_g_cm3": 0.312},
            {"distance_A": 6.0, "density_g_cm3": 0.278},
            {"distance_A": 7.5, "density_g_cm3": 0.156},
            {"distance_A": 9.0, "density_g_cm3": 0.089},
        ],
        "electrostatic_potential": {"min": -45.2, "max": 38.7, "avg": -2.3, "unit": "kJ/mol"},
        "charge_equilibration": "Qeq converged in 15 iterations, delta=2.1e-6"
    },
    "vasp": {"status": "COMPLETED", "output_dir": "/tmp/vasp"},
    "md": {"status": "COMPLETED", "output_dir": "/tmp/md", "diffusion_coefficient": "1.2e-9 m2/s"},
    "tst": {"status": "COMPLETED", "diffusion_coefficient": "8.5e-10 m2/s", "energy_barrier": "15.2 kJ/mol"},
    "vext": {"status": "COMPLETED", "output_dir": "/tmp/vext"},
    "binding": {
        "status": "COMPLETED",
        "binding_energy": "-32.5 kJ/mol",
        "binding_site": "Zn-O cluster (paddle-wheel)",
        "interaction_breakdown": {
            "electrostatic": "-18.3 kJ/mol",
            "vdW_LJ": "-14.2 kJ/mol",
            "total": "-32.5 kJ/mol"
        },
        "equilibrium_distance": "2.35 A",
        "contributing_atoms": ["Zn1", "O1", "O2", "C1"]
    },
    "xtb": {"status": "COMPLETED", "output_dir": "/tmp/xtb"},
    "structure": {"status": "COMPLETED", "structures_generated": 3},
    "guest_ff": {"status": "COMPLETED", "forcefield_dir": "/tmp/ff"},
    "features": {"status": "COMPLETED", "features": {"pore_diameter": 11.2, "surface_area": 2850, "void_fraction": 0.85}},
    "ml_train": {"status": "COMPLETED", "model_path": "/tmp/model.pkl", "r2_score": 0.85},
    "ml_predict": {"status": "COMPLETED", "predictions": [{"cif": "MOF-5", "predicted": 3.2}]},
    "ml_importance": {"status": "COMPLETED", "importance": {"surface_area": 0.45, "pore_diameter": 0.32}},
    "ml_active": {"status": "COMPLETED", "selected": ["MOF-5", "Ni-MOF-74"]},
    "find_cif": {"files": ["MOF-5_pacman.cif", "Ni-MOF-74_pacman.cif"], "count": 2},
    "inspect": {"exists": True, "type": "directory", "contents": ["MOF-5_pacman.cif"]},
    "literature": {"results": [{"title": "CO2 separation MOFs", "source": "Nature"}]},
}

def mock_submit(job_name, command, work_dir, **kw):
    for k, r in MOCK.items():
        if k in job_name.lower():
            return {"submitted": True, "job_id": f"mock_{job_name}", "status": "COMPLETED", **r}
    return {"submitted": True, "job_id": f"mock_{job_name}", "status": "COMPLETED"}

def mock_gcmc(**kw):
    return {"submitted": True, "job_id": "mock_gcmc", "status": "COMPLETED",
            "isotherm_data": [{"pressure": p, "loading": f"{float(p)*3.2:.1f} molecules/uc"}
                              for p in ["0.1","1.0","5.0","10.0"]]}

# ── Task definitions ────────────────────────────────────────────────
@dataclass
class TestTask:
    id: str
    name: str
    user_message: str
    expected_agents: List[str]
    expected_tools: List[str]
    expected_output: str = ""  # What the final output should look like
    interactive_responses: Dict[int, str] = field(default_factory=dict)
    max_rounds: int = 60

TASKS = [
    # ── T01: CO2/N2膜分离MOF材料设计 ──
    TestTask(
        id="T01",
        name="CO2/N2膜分离MOF材料设计",
        expected_output="完整Markdown报告，含：文献综述、198个MOF筛选结果表格、Top10候选材料、结构-性能关系分析、设计原则",
        user_message="""研究问题：如何设计高效的CO2/N2膜分离MOF材料？

背景：CO2/N2分离是碳捕获的关键技术。膜分离相比传统吸收法具有能耗低、设备紧凑的优势。MOF材料因其可调的孔隙结构和化学功能化潜力，是理想的膜分离材料候选。

请完成以下研究：

第一步：文献调研
- 检索CO2/N2膜分离的最新研究进展
- 了解当前最好的MOF材料及其性能指标
- 确定关键性能指标（选择性、渗透性、稳定性）

第二步：建立候选材料库
- 数据库路径：data/mof_database/ (包含198个MOF结构)
- 对候选结构进行初步筛选

第三步：高通量计算筛选
- 批量计算CO2和N2的吸附等温线
- 计算CO2/N2吸附选择性
- 计算Henry系数和吸附热

第四步：结构-性能关系分析
- 提取结构特征（孔径、比表面积、孔体积、拓扑）
- 训练ML模型预测选择性
- 分析关键结构参数

第五步：机制分析
- 分析为什么某些结构选择性高
- 讨论吸附位点和相互作用机制

第六步：生成报告
- 包含文献对比、性能数据、结构-性能关系、设计策略
- 报告格式符合学术论文标准""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["query_literature", "find_cif", "run_gcmc_batch", "run_henry", "run_pore_analysis",
                        "extract_features", "ml_train", "ml_feature_importance"],
    ),
    # ── T02: MOF结构-吸附性能关系的ML建模 ──
    TestTask(
        id="T02",
        name="MOF结构-吸附性能关系的ML建模",
        expected_output="完整报告，含：ML模型R²评分、特征重要性排序、预测vs实际散点图描述、关键结构参数结论",
        user_message="""研究问题：MOF的哪些结构特征决定CO2吸附性能？

请完成以下研究：
1. 查找可用的CIF结构
2. 对多个MOF计算CO2吸附量（批量GCMC）
3. 提取所有MOF的结构特征（孔径、比表面积、孔体积、拓扑）
4. 训练GBR模型预测CO2吸附量
5. 分析特征重要性，找出关键结构参数
6. 检索文献中关于结构-吸附关系的研究
7. 生成报告，包含ML模型性能、特征重要性排序、结构-性能关系图""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "extract_features", "ml_train",
                        "ml_feature_importance", "query_literature"],
    ),
    # ── T03: CO2在MOF中的扩散机制研究(TST/MD选择) ──
    TestTask(
        id="T03",
        name="CO2在MOF中的扩散机制研究",
        expected_output="完整报告，含：Ni-MOF-74结构特征、扩散系数数据、能垒分析、hopping vs knickering机制讨论",
        user_message="""研究问题：CO2在Ni-MOF-74中的扩散机制是什么？

请分析扩散行为：
1. 查找Ni-MOF-74的CIF结构
2. 进行孔隙分析了解孔道结构
3. 计算CO2的扩散系数
4. 分析扩散能垒和扩散路径
5. 检索文献中关于MOF扩散机制的研究
6. 生成报告，讨论扩散机制（knickering vs hopping）""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_pore_analysis", "run_string_tst", "run_md_optimize",
                        "query_literature"],
        interactive_responses={1: "用TST方法分析扩散机制"},
    ),
    # ── T04: cDFT电荷分析与CO2吸附位点识别 ──
    TestTask(
        id="T04",
        name="cDFT电荷分析与CO2吸附位点识别",
        expected_output="完整报告，含：MOF-5原子电荷分布、DDEC6电荷数据、CO2结合能、吸附位点分析图描述",
        user_message="""研究问题：MOF-5中哪些位点对CO2吸附起关键作用？

请完成以下分析：
1. 查找MOF-5的CIF结构
2. 生成cDFT输入文件进行电荷分析
3. 计算DDEC6部分原子电荷
4. 计算MOF-5与CO2的结合能
5. 同时进行GCMC吸附计算验证
6. 检索文献中关于MOF电荷与吸附关系的研究
7. 生成报告，包含电荷分布、结合能、吸附位点分析""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_cdft", "run_pacman_charge", "calc_binding_energy",
                        "run_gcmc_isotherm", "query_literature"],
    ),
    # ── T05: 多气体分离MOF膜的综合评估 ──
    TestTask(
        id="T05",
        name="多气体分离MOF膜的综合评估",
        expected_output="完整报告，含：MOF-5 vs Ni-MOF-74对比表格、选择性数据、吸附等温线、推荐结论",
        user_message="""研究问题：MOF-5和Ni-MOF-74哪个更适合CO2/N2膜分离？

请进行全面对比评估：
1. 查找两个MOF的CIF结构
2. 分别计算CO2和N2的吸附等温线和Henry系数
3. 计算CO2/N2选择性
4. 进行孔隙分析对比孔道结构
5. 提取结构特征并用ML预测吸附性能
6. 检索文献中关于这两个MOF的实验数据
7. 生成综合评估报告，包含性能对比表、推荐结论，确保报告与文献数据对齐""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm", "run_henry", "run_pore_analysis",
                        "extract_features", "ml_train", "ml_predict", "query_literature"],
    ),
    # ── T06: MOF结构生成与性质预测闭环 ──
    TestTask(
        id="T06",
        name="MOF结构生成与性质预测闭环",
        user_message="""研究问题：如何设计新型MOF结构用于CO2捕获？

请完成以下研究流程：
1. 用pormake生成3个新的MOF结构
2. 检查生成任务的状态
3. 对生成的结构进行孔隙分析
4. 提取特征并用ML预测CO2吸附量
5. 选出最有潜力的1-2个结构
6. 对选出的结构进行详细的GCMC计算
7. 生成设计报告，包含结构对比和性能预测""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["generate_structure", "check_job", "run_pore_analysis", "extract_features",
                        "ml_predict", "run_gcmc_isotherm", "query_literature"],
    ),
    # ── T07: VASP DFT计算与电子结构分析 ──
    TestTask(
        id="T07",
        name="VASP DFT计算与电子结构分析",
        user_message="""研究问题：MOF-5的电子结构如何影响CO2吸附？

请完成以下DFT计算分析：
1. 查找MOF-5的CIF结构
2. 生成cDFT输入文件进行电荷分析
3. 设置VASP输入文件（INCAR, KPOINTS, POSCAR）
4. 执行结构优化
5. 计算MOF-5与CO2的结合能
6. 分析电子态密度
7. 检索文献中关于MOF电子结构的研究
8. 生成报告，包含电子结构分析和吸附机理""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_cdft", "run_vasp", "calc_binding_energy", "inspect_path",
                        "run_gcmc_isotherm", "query_literature"],
    ),
    # ── T08: 分子力场构建与GCMC验证 ──
    TestTask(
        id="T08",
        name="分子力场构建与GCMC验证",
        user_message="""研究问题：如何为乙醇分子构建准确的RASPA力场？

请完成以下力场构建和验证：
1. 用xTB优化乙醇分子结构
2. 为乙醇构建guest forcefield（使用ligpargen参数）
3. 用构建的力场对MOF-5进行GCMC计算
4. 验证力场参数的合理性
5. 如果参数不合理，建议修正方案
6. 检索文献中关于力场参数的研究
7. 生成报告，包含力场参数和验证结果""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["run_xtb_optimize", "build_guest_forcefield", "run_gcmc_isotherm", "inspect_path",
                        "extract_features", "query_literature"],
    ),
    # ── T09: Active Learning高通量筛选 ──
    TestTask(
        id="T09",
        name="Active Learning高通量筛选",
        user_message="""研究问题：如何用Active Learning高效筛选高性能MOF？

请完成以下Active Learning流程：
1. 查找可用的CIF结构
2. 检查之前运行的任务状态
3. 对多个MOF计算CO2吸附量（批量GCMC）
4. 训练ML模型预测吸附量
5. 用Active Learning选择最有信息量的候选结构
6. 对选出的结构进行详细计算
7. 检索文献中关于Active Learning的研究
8. 生成报告，包含筛选策略和结果""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "inspect_run", "run_gcmc_batch", "ml_train", "ml_active_learning",
                        "run_gcmc_isotherm", "query_literature"],
    ),
    # ── T10: 外部势场与扩散路径分析 ──
    TestTask(
        id="T10",
        name="外部势场与扩散路径分析",
        user_message="""研究问题：CO2在Ni-MOF-74中的扩散路径和能垒是什么？

请完成以下扩散分析：
1. 查找Ni-MOF-74的CIF结构
2. 计算Ni-MOF-74对CO2的外部势场分布
3. 用TST方法计算CO2在孔道中的扩散系数
4. 用MD方法验证扩散系数
5. 分析扩散能垒和过渡态结构
6. 讨论扩散机制（knickering vs hopping）
7. 检索文献中关于MOF扩散机制的研究
8. 生成报告，包含扩散系数和机制分析""",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_external_potential", "run_string_tst", "run_md_optimize",
                        "run_pore_analysis", "query_literature"],
    ),
]

# ════════════════════════════════════════════════════════════════════
# PART 2: 50 Literature-Driven Scientific Problems (L01-L50)
# Tests RAG query + agent planning + tool execution
# ════════════════════════════════════════════════════════════════════

LITERATURE_TASKS = [
    # ── Category A: GCMC/Adsorption (L01-L12) ──
    TestTask(id="L01", name="RASPA力场选择对CO2吸附的影响",
        user_message="文献调研：RASPA中不同力场（UFF, DREIDING, TraPPE）对MOF中CO2吸附模拟精度的影响。对比实验数据，给出推荐的力场选择方案。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["query_literature", "find_cif"],
        max_rounds=12),
    TestTask(id="L02", name="MOF-5 CO2/N2选择性文献对比",
        user_message="查找文献中MOF-5对CO2/N2的IAST选择性数据，与我们的GCMC模拟结果对比。数据来源：数据库中MOF-5结构。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm", "query_literature"],
        max_rounds=12),
    TestTask(id="L03", name="Henry系数与吸附热的关系",
        user_message="通过文献调研Henry系数和等量吸附热(Qst)的物理关系。用Ni-MOF-74的CO2数据验证这个关系。计算Henry系数和吸附热并对比文献值。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_henry", "query_literature"],
        max_rounds=12),
    TestTask(id="L04", name="GCMC模拟参数优化策略",
        user_message="文献调研GCMC模拟中equilibration cycles和production cycles的选择策略。对MOF-5进行CO2吸附模拟，讨论参数对结果收敛性的影响。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm", "query_literature"],
        max_rounds=12),
    TestTask(id="L05", name="高压CO2吸附等温线行为",
        user_message="调研文献中MOF在高压(>10 bar)下CO2吸附等温线的饱和行为。对Ni-MOF-74计算0.01-50 bar范围的吸附等温线，分析饱和吸附量。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm", "query_literature"],
        max_rounds=12),
    TestTask(id="L06", name="MOF拓扑结构对吸附选择性的影响",
        user_message="文献调研不同拓扑结构(pcu, fcu, rht, soc)对MOF气体分离选择性的影响。对数据库中不同拓扑的MOF进行批量CO2吸附计算，分析拓扑-性能关系。",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "extract_features", "query_literature"],
        max_rounds=15),
    TestTask(id="L07", name="混合气体吸附的IAST方法",
        user_message="调研IAST(Ideal Adsorbed Solution Theory)在MOF混合气体吸附中的应用和局限性。用MOF-5的数据进行IAST计算，讨论与GCMC结果的差异。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm", "run_henry", "query_literature"],
        max_rounds=12),
    TestTask(id="L08", name="MOF含水量对CO2吸附的影响",
        user_message="文献调研水蒸气对MOF材料CO2吸附性能的影响（竞争吸附、水稳定性）。讨论如何在GCMC中模拟含水条件。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "query_literature"],
        max_rounds=10),
    TestTask(id="L09", name="柔性MOF的门效应与吸附",
        user_message="调研柔性MOF(flexible MOF)的门效应(gate-opening)对气体吸附的影响。对比刚性和柔性模型在GCMC中的差异。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "query_literature"],
        max_rounds=10),
    TestTask(id="L10", name="MOF再生能耗与吸附热的关系",
        user_message="文献调研MOF材料再生能耗与吸附热的定量关系。根据Ni-MOF-74的CO2吸附热数据，估算工业再生能耗。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_henry", "query_literature"],
        max_rounds=10),
    TestTask(id="L11", name="CO2在MOF中的扩散系数文献值",
        user_message="汇总文献中CO2在不同MOF中的扩散系数实验值和模拟值。对Ni-MOF-74进行MD计算，与文献对比讨论。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_md_optimize", "query_literature"],
        max_rounds=12),
    TestTask(id="L12", name="批次GCMC的统计误差分析",
        user_message="调研GCMC模拟中统计误差的来源和估计方法。对MOF-5进行多次GCMC计算，分析结果的统计显著性。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm", "query_literature"],
        max_rounds=10),

    # ── Category B: 经典流体cDFT / 电荷 / 电子结构（严格区分）(L13-L24) ──
    TestTask(id="L13", name="DDEC6 vs CM5电荷分配方法对比",
        user_message="文献调研DDEC6和CM5两种电荷分配方法的原理和精度差异。对MOF-5分别计算DDEC6和CM5电荷，对比结果。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_pacman_charge", "query_literature"],
        max_rounds=12),
    TestTask(id="L14", name="MOF静电势与气体吸附亲和力",
        user_message="调研MOF框架部分电荷与气体吸附亲和力的关系。先为MOF-5赋框架电荷，再用经典cDFT计算CO2平衡流体密度，分析静电模型对吸附分布的影响。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_cdft", "run_pacman_charge", "query_literature"],
        max_rounds=12),
    TestTask(id="L15", name="PACMAN电荷预测的精度验证",
        user_message="文献调研PACMAN(PACMOF)电荷预测模型的精度和适用范围。对数据库中的MOF验证PACMAN预测电荷与DFT计算电荷的一致性。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_pacman_charge", "query_literature"],
        max_rounds=12),
    TestTask(id="L16", name="金属团簇与CO2的结合能",
        user_message="调研MOF中金属团簇(如Zn4O, Cu2(COO)4)与CO2分子的结合能文献值。对MOF-5的Zn4O团簇计算结合能，与文献对比。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "calc_binding_energy", "query_literature"],
        max_rounds=12),
    TestTask(id="L17", name="cDFT在MOF吸附模拟中的应用",
        user_message="调研cDFT(classical DFT)方法在MOF气体吸附模拟中的应用案例和精度。对Ni-MOF-74进行cDFT计算，讨论与GCMC结果的一致性。",
        expected_agents=["lead-orchestrator", "analyst", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_cdft", "run_gcmc_isotherm", "query_literature"],
        max_rounds=15),
    TestTask(id="L18", name="MOF电荷分布的周期性边界效应",
        user_message="讨论经典cDFT中周期性边界和超胞尺寸对孔内流体密度及吸附量的影响。对带既定框架电荷的MOF-5进行不同超胞尺寸计算，分析收敛性。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_cdft", "run_pacman_charge", "query_literature"],
        max_rounds=12),
    TestTask(id="L19", name="MOF功能化对电荷分布的影响",
        user_message="调研MOF配体功能化(-NH2, -OH, -NO2)对框架电荷分布的影响。对比功能化前后MOF的DDEC6电荷变化。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_pacman_charge", "query_literature"],
        max_rounds=10),
    TestTask(id="L20", name="CO2在MOF中的吸附位点识别",
        user_message="文献调研CO2在MOF中主要吸附位点的类型（开放金属位点、π-π堆积、静电作用位点）。结合结合能计算和GCMC验证。",
        expected_agents=["lead-orchestrator", "analyst", "adsorption", "communicator"],
        expected_tools=["find_cif", "calc_binding_energy", "run_gcmc_isotherm", "query_literature"],
        max_rounds=15),
    TestTask(id="L21", name="VASP与cDFT在MOF研究中的比较",
        user_message="对比VASP量子DFT与经典流体cDFT在MOF研究中的对象、输出、成本和适用场景；分别计算MOF-5电子结构与孔内CO2平衡流体密度，禁止把两者视为同一种电子结构方法。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_vasp", "run_cdft", "query_literature"],
        max_rounds=15),
    TestTask(id="L22", name="MOF范德华修正对吸附能的影响",
        user_message="调研DFT计算中范德华修正(DFT-D3, vdW-DF)对MOF吸附能精度的影响。讨论不同修正方法的优缺点。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_vasp", "query_literature"],
        max_rounds=10),
    TestTask(id="L23", name="MOF电子结构与催化活性的关系",
        user_message="调研MOF电子结构(能带结构、态密度)与其催化活性的构效关系。分析MOF-5的电子结构特征。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_vasp", "query_literature"],
        max_rounds=12),
    TestTask(id="L24", name="机器学习力场vs DFT精度对比",
        user_message="调研ML力场(如MACE, NequIP)与DFT在MOF体系中的精度和效率对比。讨论ML力场的训练数据需求。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "query_literature"],
        max_rounds=10),

    # ── Category C: ML/筛选 (L25-L36) ──
    TestTask(id="L25", name="MOF吸附性能的ML预测模型",
        user_message="调研MOF气体吸附性能ML预测的常用模型(GBR, RF, XGBoost, GNN)和特征描述符。训练GBR模型预测CO2吸附量。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "extract_features", "ml_train", "ml_feature_importance", "query_literature"],
        max_rounds=15),
    TestTask(id="L26", name="特征工程对ML预测精度的影响",
        user_message="调研MOF ML建模中特征工程的方法（几何特征、化学特征、拓扑特征）。对比不同特征组合对预测精度的影响。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "extract_features", "ml_train", "query_literature"],
        max_rounds=12),
    TestTask(id="L27", name="高通量筛选的最优策略",
        user_message="文献调研MOF高通量筛选的最优流程（粗筛→细筛→验证）。设计并执行一个3级筛选流程。",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "extract_features", "ml_train", "ml_feature_importance", "query_literature"],
        max_rounds=15),
    TestTask(id="L28", name="迁移学习在MOF筛选中的应用",
        user_message="调研迁移学习在MOF吸附预测中的应用（如从CO2到CH4的模型迁移）。讨论数据需求和精度损失。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "ml_train", "ml_predict", "query_literature"],
        max_rounds=12),
    TestTask(id="L29", name="图神经网络在MOF中的应用",
        user_message="调研GNN(图神经网络)在MOF性质预测中的应用现状。讨论图表示方法(CGCNN, MOFNet)的优缺点。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "query_literature"],
        max_rounds=10),
    TestTask(id="L30", name="主动学习加速MOF筛选",
        user_message="调研Active Learning在MOF筛选中的应用策略（不确定性采样、查询合成、预期梯度变化）。执行主动学习流程。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "ml_train", "ml_active_learning", "query_literature"],
        max_rounds=12),
    TestTask(id="L31", name="多目标优化MOF设计",
        user_message="调研MOF多目标优化设计的方法（Pareto最优、加权目标）。同时优化CO2吸附量和CO2/N2选择性。",
        expected_agents=["lead-orchestrator", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "ml_train", "query_literature"],
        max_rounds=15),
    TestTask(id="L32", name="数据增强在MOF ML中的应用",
        user_message="调研MOF ML建模中数据增强的方法（SMOTE, 物理约束增强）。讨论小样本情况下的模型泛化能力。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "extract_features", "ml_train", "query_literature"],
        max_rounds=12),
    TestTask(id="L33", name="ML模型可解释性分析",
        user_message="调研ML模型可解释性方法(SHAP, LIME, 特征重要性)在MOF研究中的应用。训练模型并分析预测结果的可解释性。",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "extract_features", "ml_train", "ml_feature_importance", "query_literature"],
        max_rounds=12),
    TestTask(id="L34", name="MOF数据库质量评估",
        user_message="调研MOF数据库(CoRE MOF, hMOF, ToBaCCo)的数据质量和覆盖范围。评估我们198个MOF数据库的代表性。",
        expected_agents=["lead-orchestrator", "harness-maintainer", "analyst", "communicator"],
        expected_tools=["find_cif", "extract_features", "query_literature"],
        max_rounds=10),
    TestTask(id="L35", name="生成模型设计新型MOF",
        user_message="调研生成模型(VAE, GAN, Diffusion Model)在MOF结构生成中的应用。用pormake生成新型MOF结构。",
        expected_agents=["lead-orchestrator", "harness-maintainer", "communicator"],
        expected_tools=["generate_structure", "check_job", "query_literature"],
        max_rounds=12),
    TestTask(id="L36", name="GCMC+ML联合筛选框架",
        user_message="设计GCMC+ML联合筛选框架：先用少量GCMC计算训练ML模型，再用ML模型筛选整个数据库。验证框架的有效性。",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "extract_features", "ml_train", "ml_predict", "ml_feature_importance", "query_literature"],
        max_rounds=15),

    # ── Category D: 扩散/动力学 (L37-L44) ──
    TestTask(id="L37", name="MD模拟中力场选择对扩散的影响",
        user_message="调研MD模拟中不同力场(UFF, DREIDING, BOF)对MOF中气体扩散系数的影响。对Ni-MOF-74进行MD计算。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_md_optimize", "query_literature"],
        max_rounds=12),
    TestTask(id="L38", name="TST方法计算MOF扩散能垒",
        user_message="调研过渡态理论(TST)在MOF扩散计算中的应用。用string method计算CO2在Ni-MOF-74中的扩散能垒。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_string_tst", "query_literature"],
        max_rounds=12),
    TestTask(id="L39", name="外部势场与扩散路径",
        user_message="调研外部势场(external potential)方法在MOF扩散研究中的应用。计算Ni-MOF-74对CO2的势能面，分析扩散路径。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_external_potential", "run_string_tst", "query_literature"],
        max_rounds=12),
    TestTask(id="L40", name="MOF孔径与扩散机制的关系",
        user_message="调研MOF孔径大小对扩散机制(Knudsen, 表面扩散, 活化扩散)的影响。对比不同孔径MOF的扩散系数。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_pore_analysis", "run_md_optimize", "query_literature"],
        max_rounds=12),
    TestTask(id="L41", name="温度对MOF扩散系数的影响",
        user_message="调研温度对MOF中气体扩散系数的影响（Arrhenius行为）。在不同温度下计算扩散系数，拟合活化能。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_md_optimize", "query_literature"],
        max_rounds=10),
    TestTask(id="L42", name="混合气体在MOF中的扩散",
        user_message="调研MOF中混合气体扩散的竞争效应和选择性扩散。讨论Maxwell-Stefan模型在MOF中的应用。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_md_optimize", "query_literature"],
        max_rounds=10),
    TestTask(id="L43", name="MOF缺陷对扩散的影响",
        user_message="调研MOF晶体缺陷(缺失连接体、缺失节点)对气体扩散行为的影响。讨论缺陷工程在膜分离中的应用。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "query_literature"],
        max_rounds=10),
    TestTask(id="L44", name="从MD轨迹分析扩散机制",
        user_message="调研从MD轨迹分析扩散机制的方法(MSD, VACF, 停留时间分布)。分析Ni-MOF-74中CO2的扩散轨迹。",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_md_optimize", "query_literature"],
        max_rounds=10),

    # ── Category E: 综合/跨方法 (L45-L50) ──
    TestTask(id="L45", name="cDFT+GCMC联合研究CO2吸附机理",
        user_message="综合运用经典cDFT和GCMC研究MOF-5中CO2吸附：先用PACMOF/PACMAN获得框架部分电荷，再用经典cDFT计算平衡流体密度、用GCMC计算吸附等温线，最后比较两种统计力学方法。",
        expected_agents=["lead-orchestrator", "analyst", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_cdft", "run_pacman_charge", "run_gcmc_isotherm", "query_literature"],
        max_rounds=15),
    TestTask(id="L46", name="多尺度模拟框架设计",
        user_message="设计MOF气体分离的多尺度模拟框架：量子DFT(电子结构/参考电荷)→经典cDFT或GCMC(平衡吸附)→真实客体轨迹/TST(扩散)→ML(筛选)。讨论各尺度的衔接并严格区分方法输出。",
        expected_agents=["lead-orchestrator", "analyst", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_vasp", "run_cdft", "run_gcmc_isotherm", "run_md_optimize", "extract_features", "query_literature"],
        max_rounds=15),
    TestTask(id="L47", name="MOF稳定性评估框架",
        user_message="设计MOF热力学和化学稳定性评估框架：结合DFT能量计算、GCMC吸附模拟、文献数据对比。评估数据库中MOF的稳定性。",
        expected_agents=["lead-orchestrator", "analyst", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_vasp", "run_gcmc_isotherm", "extract_features", "query_literature"],
        max_rounds=15),
    TestTask(id="L48", name="MOF碳捕获全流程评估",
        user_message="评估MOF材料在碳捕获中的全流程性能：吸附容量(高通量GCMC)→选择性(IAST)→再生能耗(吸附热)→扩散速率(MD)→成本(ML预测)。",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch", "run_henry", "run_md_optimize", "extract_features", "ml_train", "query_literature"],
        max_rounds=15),
    TestTask(id="L49", name="MOF膜分离性能预测",
        user_message="预测MOF膜的CO2/N2分离性能：计算吸附选择性、扩散选择性、渗透性Robeson上界。对比文献中的实验膜性能数据。",
        expected_agents=["lead-orchestrator", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm", "run_henry", "run_md_optimize", "extract_features", "query_literature"],
        max_rounds=15),
    TestTask(id="L50", name="文献驱动的MOF设计策略",
        user_message="基于文献调研结果，提出3种MOF结构设计策略来提高CO2/N2分离性能。验证策略的可行性：批量计算+ML预测。",
        expected_agents=["lead-orchestrator", "harness-maintainer", "adsorption", "analyst", "communicator"],
        expected_tools=["find_cif", "generate_structure", "run_gcmc_batch", "ml_predict", "query_literature"],
        max_rounds=15),
]

# ════════════════════════════════════════════════════════════════════
# PART 3: 50 Parameter-Missing Tasks (V01-V50)
# Tests interactive feedback: will the agent ask for missing params?
# ════════════════════════════════════════════════════════════════════

VAGUE_TASKS = [
    # ── 缺少气体种类 (V01-V10) ──
    TestTask(id="V01", name="缺少气体: 吸附计算",
        user_message="帮我计算MOF-5的吸附等温线",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V02", name="缺少气体: 选择性",
        user_message="计算MOF-5和Ni-MOF-74的选择性",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V03", name="缺少气体: 批量筛选",
        user_message="对数据库中的MOF进行批量吸附计算筛选",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V04", name="缺少气体: Henry系数",
        user_message="计算Ni-MOF-74的Henry系数和吸附热",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V05", name="缺少气体: 结合能",
        user_message="计算MOF-5中不同位点的结合能",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V06", name="缺少MOF: 扩散",
        user_message="研究CO2在MOF中的扩散机制",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V07", name="缺少气体: ML预测",
        user_message="训练ML模型预测MOF的吸附性能",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V08", name="缺少气体: cDFT",
        user_message="用cDFT分析MOF-5的电荷分布",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V09", name="缺少气体: VASP",
        user_message="对Ni-MOF-74进行VASP结构优化",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V10", name="缺少气体: 力场构建",
        user_message="为分子构建RASPA力场",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),

    # ── 缺少材料名称 (V11-V20) ──
    TestTask(id="V11", name="缺少材料: 等温线",
        user_message="计算CO2的吸附等温线",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V12", name="缺少材料: 选择性",
        user_message="计算CO2/N2的选择性",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V13", name="缺少材料: 电荷分析",
        user_message="计算DDEC6部分原子电荷",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V14", name="缺少材料: 结合能",
        user_message="计算CO2与框架的结合能",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V15", name="缺少材料: MD扩散",
        user_message="用MD计算气体扩散系数",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V16", name="缺少材料: TST",
        user_message="用TST方法计算扩散能垒",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V17", name="缺少材料: 孔隙分析",
        user_message="分析MOF的孔隙结构",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V18", name="缺少材料: VASP优化",
        user_message="进行VASP结构优化计算",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V19", name="缺少材料: ML训练",
        user_message="训练GBR模型预测吸附量",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V20", name="缺少材料: 结构生成",
        user_message="用pormake生成新的MOF结构",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),

    # ── 缺少压力/温度 (V21-V30) ──
    TestTask(id="V21", name="缺少压力: 等温线",
        user_message="计算MOF-5对CO2的吸附等温线",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm"],
        max_rounds=10),
    TestTask(id="V22", name="缺少温度: Henry系数",
        user_message="计算Ni-MOF-74对CO2的Henry系数",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_henry"],
        max_rounds=10),
    TestTask(id="V23", name="缺少压力范围: 批量筛选",
        user_message="对MOF-5和Ni-MOF-74进行CO2吸附批量计算",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_batch"],
        max_rounds=10),
    TestTask(id="V24", name="缺少温度: MD扩散",
        user_message="用MD计算Ni-MOF-74中CO2的扩散系数",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_md_optimize"],
        max_rounds=10),
    TestTask(id="V25", name="缺少温度: cDFT计算",
        user_message="对MOF-5进行cDFT电荷计算",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_cdft"],
        max_rounds=10),
    TestTask(id="V26", name="缺少压力: 吸附验证",
        user_message="用GCMC验证MOF-5的CO2吸附性能",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm"],
        max_rounds=10),
    TestTask(id="V27", name="缺少温度: 结合能",
        user_message="计算MOF-5与CO2的结合能",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "calc_binding_energy"],
        max_rounds=10),
    TestTask(id="V28", name="缺少压力: IAST",
        user_message="计算MOF-5的IAST选择性",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm"],
        max_rounds=10),
    TestTask(id="V29", name="缺少温度: 活化能",
        user_message="计算CO2在Ni-MOF-74中的扩散活化能",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_md_optimize"],
        max_rounds=10),
    TestTask(id="V30", name="缺少温度: VASP",
        user_message="对MOF-5进行VASP单点能计算",
        expected_agents=["lead-orchestrator", "analyst", "communicator"],
        expected_tools=["find_cif", "run_vasp"],
        max_rounds=10),

    # ── 多参数缺失 (V31-V40) ──
    TestTask(id="V31", name="多缺: 筛选",
        user_message="帮我筛选好的MOF材料",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V32", name="多缺: 计算",
        user_message="做一个计算",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V33", name="多缺: 分析",
        user_message="分析一下这个材料",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V34", name="多缺: 设置",
        user_message="帮我设置GCMC计算",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V35", name="多缺: 查询",
        user_message="查一下文献",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V36", name="多缺: 做",
        user_message="做一下吸附计算",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V37", name="多缺: 设计",
        user_message="设计一个MOF用于气体分离",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V38", name="多缺: 对比",
        user_message="对比两个MOF的性能",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V39", name="多缺: 优化",
        user_message="优化MOF的吸附性能",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V40", name="多缺: 预测",
        user_message="预测MOF的性质",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),

    # ── 模糊但有部分参数 (V41-V50) ──
    TestTask(id="V41", name="有气体无材料无条件",
        user_message="CO2的吸附量是多少",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V42", name="有材料无气体无条件",
        user_message="MOF-5的吸附性能怎么样",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V43", name="有气体有材料无条件",
        user_message="MOF-5对CO2的吸附量",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm"],
        max_rounds=10),
    TestTask(id="V44", name="有气体无材料有温度",
        user_message="在298K下CO2的吸附量",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V45", name="有材料有温度无气体",
        user_message="在300K下MOF-5的吸附性能",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V46", name="有气体有压力无材料",
        user_message="在1bar下CO2的吸附量",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V47", name="有材料有气体无温度无压力",
        user_message="MOF-5对CO2的吸附等温线",
        expected_agents=["lead-orchestrator", "adsorption", "communicator"],
        expected_tools=["find_cif", "run_gcmc_isotherm"],
        max_rounds=10),
    TestTask(id="V48", name="有任务无对象",
        user_message="计算选择性",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V49", name="有方法无目标",
        user_message="用GCMC计算一下",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
    TestTask(id="V50", name="完全模糊",
        user_message="帮我做点什么",
        expected_agents=["lead-orchestrator"],
        expected_tools=[],
        max_rounds=5),
]

# ── TaskRunner ──────────────────────────────────────────────────────
class TaskRunner:
    def __init__(self, task: TestTask, verbose=True, use_mock=True):
        self.task = task
        self.verbose = verbose
        self.use_mock = use_mock
        self.config = get_config()
        self.session: Optional[Session] = None
        self.result: Dict[str, Any] = {}

    def log(self, msg):
        if self.verbose:
            print(f"  [{self.task.id}] {msg}", flush=True)

    def run(self) -> Dict[str, Any]:
        self.log(f"═══ {self.task.name} ═══")
        t0 = time.time()
        patches = []
        try:
            self.session = Session(config=self.config)
            if self.use_mock:
                import agents.slurm as sm
                patches = [
                    patch.object(sm, 'submit_and_wait', side_effect=mock_submit),
                    patch.object(sm, 'submit_gcmc_isotherm', side_effect=mock_gcmc),
                ]
                for p in patches:
                    p.start()

            resp = self.session.start(self.task.user_message, agent=ORCHESTRATOR,
                                      max_rounds=self.task.max_rounds, verbose=self.verbose)
            # Handle interactive replies
            rc = 0
            while not self.session.task_complete and rc < 5:
                rc += 1
                if rc in self.task.interactive_responses:
                    self.log(f"  → User: {self.task.interactive_responses[rc]}")
                    resp = self.session.reply(self.task.interactive_responses[rc],
                                              max_rounds=self.task.max_rounds, verbose=self.verbose)

            # Save full response to separate file
            full_resp = resp or ""
            resp_file = Path(f"test_response_{self.task.id}.md")
            resp_file.write_text(full_resp)

            self.result = {
                "task_id": self.task.id, "task_name": self.task.name,
                "status": "completed" if self.session.task_complete else "incomplete",
                "elapsed": round(time.time()-t0, 1),
                "response": full_resp[:10000],
                "full_response_file": str(resp_file),
                "full_response_length": len(full_resp),
                "agent_flow": self.session.memory.agent_history,
                "tool_calls": [{"agent": t["agent"], "tool": t["tool"],
                               "result_preview": t.get("result_preview", ""),
                               "params": t.get("params", "")}
                               for t in self.session.memory.tool_call_log],
                "critical_results": self.session.memory.critical_results,
                "stats": self.session.memory.get_stats(),
                "errors": self.session.memory.errors,
            }
        except Exception as e:
            self.result = {"task_id": self.task.id, "status": "error",
                           "elapsed": round(time.time()-t0, 1), "error": str(e),
                           "traceback": traceback.format_exc()}
        finally:
            for p in patches:
                p.stop()
        # Save
        Path(f"test_result_{self.task.id}.json").write_text(
            json.dumps(self.result, indent=2, default=str))
        return self.result

    def verify(self) -> Dict[str, Any]:
        r = self.result
        agents_in = set()
        for step in r.get("agent_flow", []):
            for a in step.split(" → "):
                agents_in.add(a)
        agents_in.add("lead-orchestrator")
        called = set(t["tool"] for t in r.get("tool_calls", []))

        # Check report completeness
        resp = r.get("response", "")
        has_report = bool(resp and len(resp) > 200)
        has_markdown = "##" in resp or "# " in resp
        has_table = "|" in resp
        report_quality = "full" if (has_report and has_markdown and has_table) else \
                         "partial" if has_report else "missing"

        return {
            "completed": r.get("status") == "completed",
            "all_agents": self.task.expected_agents == sorted(agents_in),
            "missing_agents": sorted(set(self.task.expected_agents) - agents_in),
            "all_tools": set(self.task.expected_tools) <= called,
            "missing_tools": sorted(set(self.task.expected_tools) - called),
            "handoffs": r.get("stats", {}).get("agent_handoffs", 0) > 0,
            "tool_count": r.get("stats", {}).get("tool_calls", 0),
            "report_quality": report_quality,
            "report_length": len(resp),
            "agents_used": sorted(agents_in),
            "tools_called": sorted(called),
        }

def run_all(task_ids=None, verbose=True, task_set="original"):
    """Run tasks and collect statistics.
    task_set: "original" | "literature" | "vague" | "all"
    """
    # Select task set
    if task_set == "literature":
        all_tasks = LITERATURE_TASKS
    elif task_set == "vague":
        all_tasks = VAGUE_TASKS
    elif task_set == "all":
        all_tasks = TASKS + LITERATURE_TASKS + VAGUE_TASKS
    else:
        all_tasks = TASKS

    if task_ids:
        all_tasks = [t for t in all_tasks if t.id in task_ids]

    results, checks = [], []
    for i, task in enumerate(all_tasks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(all_tasks)}] {task.id}: {task.name}")
        print(f"{'='*60}")
        runner = TaskRunner(task, verbose=verbose)
        res = runner.run()
        ck = runner.verify()
        results.append(res)
        checks.append(ck)

        s = "✅" if res["status"]=="completed" else "❌"
        a = "✅" if ck["all_agents"] else f"⚠️ missing:{ck['missing_agents']}"
        t_str = "✅" if ck["all_tools"] else f"⚠️ missing:{ck['missing_tools']}"
        rq = ck.get("report_quality", "unknown")
        rl = ck.get("report_length", 0)
        print(f"\n{s} {task.id}: {task.name} | {res.get('elapsed','?')}s")
        print(f"  Agents: {a}")
        print(f"  Tools: {t_str} ({ck['tool_count']} calls)")
        print(f"  Handoffs: {'✅' if ck['handoffs'] else '❌'}")
        print(f"  Report: {rq} ({rl} chars)")
        if res.get("errors"):
            print(f"  Errors: {res['errors']}")

    # ── Comprehensive Summary ──
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*70)

    # Basic stats
    c = sum(1 for r in results if r["status"]=="completed")
    aa = sum(1 for ck in checks if ck["all_agents"])
    at = sum(1 for ck in checks if ck["all_tools"])
    h = sum(1 for ck in checks if ck["handoffs"])
    total_time = sum(r.get("elapsed", 0) for r in results)
    total_tools = sum(ck.get("tool_count", 0) for ck in checks)

    print(f"\nTask Set: {task_set} ({len(results)} tasks)")
    print(f"Completed: {c}/{len(results)} ({100*c/max(len(results),1):.0f}%)")
    print(f"All agents used: {aa}/{len(checks)} ({100*aa/max(len(checks),1):.0f}%)")
    print(f"All expected tools called: {at}/{len(checks)} ({100*at/max(len(checks),1):.0f}%)")
    print(f"Agent handoffs: {h}/{len(checks)} ({100*h/max(len(checks),1):.0f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Total tool calls: {total_tools}")

    # Tool usage frequency
    tool_freq = {}
    for ck in checks:
        for tool in ck.get("tools_called", []):
            tool_freq[tool] = tool_freq.get(tool, 0) + 1
    if tool_freq:
        print(f"\nTool usage frequency:")
        for tool, count in sorted(tool_freq.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")

    # Agent routing frequency
    agent_freq = {}
    for ck in checks:
        for agent in ck.get("agents_used", []):
            agent_freq[agent] = agent_freq.get(agent, 0) + 1
    if agent_freq:
        print(f"\nAgent routing frequency:")
        for agent, count in sorted(agent_freq.items(), key=lambda x: -x[1]):
            print(f"  {agent}: {count}")

    # Report quality distribution
    rq_dist = {}
    for ck in checks:
        rq = ck.get("report_quality", "unknown")
        rq_dist[rq] = rq_dist.get(rq, 0) + 1
    print(f"\nReport quality: {rq_dist}")

    # Vague task specific stats (interactive feedback)
    if task_set == "vague":
        asked_user = sum(1 for r in results
                        if "需要确认" in r.get("response", "") or
                           "缺失" in r.get("response", "") or
                           "请提供" in r.get("response", "") or
                           "which" in r.get("response", "").lower() or
                           "what" in r.get("response", "").lower())
        print(f"\nInteractive feedback (asked user for params): {asked_user}/{len(results)}")
        no_tool = sum(1 for ck in checks if ck.get("tool_count", 0) == 0)
        print(f"No tools called (correct for vague): {no_tool}/{len(results)}")

    # Save results
    output = {
        "task_set": task_set,
        "summary": {
            "total": len(results),
            "completed": c,
            "all_agents": aa,
            "all_tools": at,
            "handoffs": h,
            "total_time": round(total_time, 1),
            "total_tool_calls": total_tools,
        },
        "tool_frequency": tool_freq,
        "agent_frequency": agent_freq,
        "report_quality": rq_dist,
        "results": results,
        "checks": checks,
    }
    out_path = Path(f"test_results_{task_set}.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved: {out_path}")
    return results, checks


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="BiMemAgent 100-task test suite")
    p.add_argument("--tasks", nargs="*", help="Specific task IDs to run (e.g. L01 V05 T01)")
    p.add_argument("--set", default="original",
                    choices=["original", "literature", "vague", "all"],
                    help="Task set to run")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    run_all(task_ids=a.tasks, verbose=not a.quiet, task_set=a.set)
