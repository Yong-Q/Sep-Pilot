from __future__ import annotations

import re
from typing import Any, Dict, List


HARNESS_HINTS = (
    "harness",
    "agent",
    "skill",
    "catalog",
    "claude",
    "同步",
    "重建",
    "命名",
    "生成 md",
    "trigger",
    "触发",
)

LITERATURE_HINTS = (
    "why",
    "mechanism",
    "explain",
    "explains",
    "benchmark",
    "literature",
    "reported",
    "parameter",
    "force field",
    "rationale",
    "文献",
    "机理",
    "为什么",
    "原因",
    "对比",
    "参数",
    "力场",
)

RESULT_HINTS = (
    ".csv",
    ".json",
    ".out",
    "result",
    "results",
    "output",
    "analyze",
    "analysis",
    "interpret",
    "rank",
    "解释结果",
    "分析结果",
    "结果文件",
)

COMMUNICATION_HINTS = (
    "figure",
    "plot",
    "panel",
    "svg",
    "pdf",
    "tiff",
    "citation",
    "reference",
    "reader",
    "translate paper",
    "manuscript",
    "abstract",
    "discussion",
    "写作",
    "作图",
    "画图",
    "绘图",
    "引文",
    "引用",
    "参考文献",
    "全文翻译",
    "中英文对照",
    "论文解读",
)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _route_agent(question: str) -> str:
    if _contains_any(question, HARNESS_HINTS):
        return "harness-maintainer"
    if _contains_any(question, COMMUNICATION_HINTS):
        return "scientific-communicator"
    if _contains_any(question, RESULT_HINTS):
        return "analyst"
    return "lead-orchestrator"


def _skills_for_lead(question: str) -> List[str]:
    skills = ["overview"]
    lowered = question.lower()
    if _contains_any(question, LITERATURE_HINTS):
        skills.append("literature-rag")
    if any(token in lowered for token in ("doi", "arxiv", "pdf", "paper", "论文")):
        skills.append("paper")
    if any(token in lowered for token in ("cif", "structure", "material", "mof-", "cof-", "zif-")):
        skills.append("cif")
    if any(token in lowered for token in ("gcmc", "isotherm", "henry", "uptake", "adsorption")):
        skills.append("adsorption")
    if any(token in lowered for token in ("cdft", "density functional")):
        skills.append("cdft")
    if any(token in lowered for token in ("job", "pbs", "qsub", "submit")):
        skills.append("jobs")
    return skills


def _skills_for_analyst(question: str) -> List[str]:
    skills = ["results", "analysis"]
    lowered = question.lower()
    if _contains_any(question, LITERATURE_HINTS):
        skills.append("literature-rag")
    if any(token in lowered for token in ("binding", "interaction", "吸附位点", "结合能")):
        skills.append("interaction")
    if any(token in lowered for token in ("descriptor", "property", "feature", "材料性质")):
        skills.append("material-props")
    return skills


def _skills_for_scientific_communicator(question: str) -> List[str]:
    skills: List[str] = []
    lowered = question.lower()
    if any(token in lowered for token in ("figure", "plot", "panel", "svg", "pdf", "tiff", "画图", "作图", "绘图")):
        skills.append("nature-figure")
    if any(token in lowered for token in ("citation", "reference", "引文", "引用", "参考文献")):
        skills.append("nature-citation")
    if any(token in lowered for token in ("reader", "translate paper", "全文翻译", "中英文对照", "论文解读")):
        skills.append("nature-reader")
    if any(token in lowered for token in ("manuscript", "abstract", "discussion", "写作")):
        skills.append("nature-writing")
    if not skills:
        skills.append("nature-figure")
    if _contains_any(question, LITERATURE_HINTS):
        skills.append("literature-rag")
    if _contains_any(question, RESULT_HINTS):
        skills.extend(["results", "analysis"])
    return skills


def _skills_for_harness() -> List[str]:
    return ["harness", "overview", "results"]


def route_question(question: str) -> Dict[str, Any]:
    agent = _route_agent(question)
    if agent == "lead-orchestrator":
        skills = _skills_for_lead(question)
    elif agent == "scientific-communicator":
        skills = _skills_for_scientific_communicator(question)
    elif agent == "analyst":
        skills = _skills_for_analyst(question)
    else:
        skills = _skills_for_harness()
    return {
        "question": question,
        "agent": agent,
        "skills": skills,
    }


def run_trigger_suite() -> Dict[str, Any]:
    cases = [
        {
            "label": "lead mechanism",
            "question": "Why does ZIF-8 separate C3H6/C3H8? Give a literature-backed mechanism explanation.",
            "expected_agent": "lead-orchestrator",
            "expected_skills": ["overview", "literature-rag"],
        },
        {
            "label": "lead adsorption setup",
            "question": "Run a GCMC isotherm for CO2 in MOF-5 at 298 K and 1 bar.",
            "expected_agent": "lead-orchestrator",
            "expected_skills": ["overview", "cif", "adsorption"],
        },
        {
            "label": "analyst mechanism on result",
            "question": "Interpret this result.csv and explain the mechanism compared with literature benchmarks.",
            "expected_agent": "analyst",
            "expected_skills": ["results", "analysis", "literature-rag"],
        },
        {
            "label": "analyst quantitative only",
            "question": "Analyze this output/result.csv and rank the top materials by selectivity.",
            "expected_agent": "analyst",
            "expected_skills": ["results", "analysis"],
        },
        {
            "label": "scientific communicator figure",
            "question": "Use these result.csv files to make a publication-ready multi-panel figure in SVG and PDF.",
            "expected_agent": "scientific-communicator",
            "expected_skills": ["nature-figure", "results", "analysis"],
        },
        {
            "label": "scientific communicator citation",
            "question": "Give me Nature-style supporting references for this discussion paragraph and export citations.",
            "expected_agent": "scientific-communicator",
            "expected_skills": ["nature-citation"],
        },
        {
            "label": "scientific communicator reader",
            "question": "Read this PDF and make a Chinese-English side-by-side paper reader with figure placement.",
            "expected_agent": "scientific-communicator",
            "expected_skills": ["nature-reader"],
        },
        {
            "label": "lead paper retrieval",
            "question": "Download this arXiv paper into the workflow and summarize whether it is useful for CH4/H2 separation.",
            "expected_agent": "lead-orchestrator",
            "expected_skills": ["overview", "paper"],
        },
        {
            "label": "lead jobs submit",
            "question": "Submit a PBS job for a batch GCMC screening on these CIFs.",
            "expected_agent": "lead-orchestrator",
            "expected_skills": ["overview", "jobs"],
        },
        {
            "label": "analyst interaction followup",
            "question": "Analyze this result.json and estimate whether the binding interaction explains the selectivity trend.",
            "expected_agent": "analyst",
            "expected_skills": ["results", "analysis", "interaction", "literature-rag"],
        },
        {
            "label": "harness maintenance",
            "question": "Audit the harness and regenerate agent markdown after catalog changes.",
            "expected_agent": "harness-maintainer",
            "expected_skills": ["harness", "overview", "results"],
        },
    ]
    results = []
    for case in cases:
        routed = route_question(case["question"])
        ok_agent = routed["agent"] == case["expected_agent"]
        ok_skills = all(skill in routed["skills"] for skill in case["expected_skills"])
        results.append(
            {
                **case,
                "actual_agent": routed["agent"],
                "actual_skills": routed["skills"],
                "status": "passed" if ok_agent and ok_skills else "failed",
            }
        )
    return {
        "results": results,
        "passed": all(item["status"] == "passed" for item in results),
    }
