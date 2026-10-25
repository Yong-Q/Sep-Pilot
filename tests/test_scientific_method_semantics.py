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
        result = registry._exec_rag({"query": "sentinel polygon 98765", "require_both": True})
    assert rag_call.called and web_call.called
    assert result["source"] == "rag+web"
    assert result["coverage"] == ["rag", "web"]
    assert result["evidence_complete"] is True


def test_query_literature_reads_actual_faiss_hits_field():
    local = {"ok": True, "hits": [{"title": "local scientific passage", "text": "cDFT adsorption", "score": 0.8}]}
    with patch("bimem_agent.legacy_bridge.run_literature_rag", return_value=local), \
            patch.object(registry, "_web_literature", return_value={"ok": False, "results": []}) as web_call:
        result = registry._exec_rag({"query": "sentinel polygon 98765", "web_fallback": False})
    assert result["source"] == "rag"
    assert result["results"][0]["title"] == "local scientific passage"
    assert not web_call.called


def test_curated_knowledge_answers_relevant_method_query_without_slow_legacy_index():
    with patch("bimem_agent.legacy_bridge.run_literature_rag", side_effect=AssertionError("legacy index should not load")), \
            patch.object(registry, "_web_literature", return_value={"ok": True, "results": [{"title": "web paper"}]}) as web_call:
        result = registry._exec_rag({"query": "classical cDFT pore adsorption", "require_both": True})
    assert result["coverage"] == ["rag", "web"]
    assert any(hit.get("id") == "classical_cdft" for hit in result["local"])
    assert web_call.called


def test_query_literature_surfaces_paper_metadata_without_claiming_full_text_evidence():
    paper = {"title": "verified paper title", "doi": "10.1000/example", "source": "crossref_metadata",
             "evidence_level": "bibliographic_lead_only", "source_url": "https://doi.org/10.1000/example"}
    with patch("bimem_agent.knowledge.search_knowledge", return_value=[]), \
            patch("bimem_agent.knowledge.search_papers", return_value=[paper]), \
            patch("bimem_agent.legacy_bridge.run_literature_rag", side_effect=AssertionError("metadata hit should avoid cold FAISS")), \
            patch.object(registry, "_web_literature", return_value={"ok": False, "results": []}):
        result = registry._exec_rag({"query": "ZIF-8 ethene membrane", "require_both": True})
    assert result["local"][0]["doi"] == "10.1000/example"
    assert result["evidence_complete"] is False
    assert result["bibliographic_only"] is True


def test_query_literature_schema_exposes_dual_source_mode():
    schema = get_registry().get("query_literature").input_schema
    assert "require_both" in schema["properties"]


def test_literature_evidence_enters_independent_scientific_review():
    assert "query_literature" in SCIENCE_TOOLS
