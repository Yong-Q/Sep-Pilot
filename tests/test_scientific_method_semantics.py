"""Regression tests for scientific method semantics and evidence coverage."""
from unittest.mock import patch

from agents import registry
from agents.defns import ORCHESTRATOR
from agents.registry import get_registry
from agents.session import Session
from agents.config import get_config
from agents.scientific_review import SCIENCE_TOOLS


def test_prompts_separate_classical_cdft_from_electronic_dft_and_binding_energy():
    instructions = ORCHESTRATOR.get_instructions({})
    assert "classical density functional theory（经典密度泛函理论）" in instructions
    assert "不计算电子密度、能带或量子电子结构" in instructions
    assert "单构型结合能用 calc_binding_energy" in instructions
    assert "不得宣称 cDFT 天然比 GCMC“更精确”" in instructions


def test_method_question_describes_classical_cdft_without_electronic_claims():
    session = Session(config=get_config())
    text = session._build_param_question("计算方法(GCMC 或 cDFT)", "预测COF分离性能")
    assert "经典 cDFT" in text
    assert "不是电子结构 DFT" in text
    assert "不直接计算框架电荷或单构型结合能" in text


def test_query_literature_require_both_queries_rag_and_web():
    local = {"ok": True, "results": [{"title": "local evidence"}]}
    web = {"ok": True, "results": [{"title": "web evidence"}]}
    with patch("bimem_agent.legacy_bridge.run_literature_rag", return_value=local) as rag_call, \
            patch.object(registry, "_web_literature", return_value=web) as web_call:
        result = registry._exec_rag({"query": "classical DFT adsorption", "require_both": True})
    assert rag_call.called and web_call.called
    assert result["source"] == "rag+web"
    assert result["coverage"] == ["rag", "web"]
    assert result["evidence_complete"] is True


def test_query_literature_schema_exposes_dual_source_mode():
    schema = get_registry().get("query_literature").input_schema
    assert "require_both" in schema["properties"]


def test_literature_evidence_enters_independent_scientific_review():
    assert "query_literature" in SCIENCE_TOOLS
