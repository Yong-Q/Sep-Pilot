import json
from pathlib import Path

from bimem_agent.knowledge import search_knowledge, search_papers


def test_curated_index_imports_without_private_runtime_config(tmp_path):
    import os
    import shutil
    import subprocess
    from pathlib import Path
    package = Path(__file__).resolve().parents[1] / "bimem_agent"
    isolated = tmp_path / "bimem_agent"
    (isolated / "knowledge").mkdir(parents=True)
    shutil.copy2(package / "__init__.py", isolated / "__init__.py")
    for name in ("__init__.py", "retriever.py", "curated.json"):
        shutil.copy2(package / "knowledge" / name, isolated / "knowledge" / name)
    result = subprocess.run(["python", "-c", "from bimem_agent.knowledge import search_knowledge; assert search_knowledge('classical cDFT pore adsorption')"],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)},
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr[-350:]


def test_classical_cdft_is_not_electronic_dft():
    hits = search_knowledge("经典 cDFT 气体吸附 Vext 密度泛函", limit=2)
    assert hits
    assert hits[0]["id"] == "classical_cdft"
    assert "电子" in hits[0]["caveat"]
    assert hits[0]["source_url"].startswith("https://")


def test_adsorption_does_not_claim_membrane_permeance():
    hits = search_knowledge("GCMC uptake adsorption selectivity membrane permeance", limit=3)
    assert any(hit["id"] == "adsorption_vs_membrane" for hit in hits)
    assert any("diffusion" in hit["caveat"].lower() for hit in hits)


def test_active_learning_split_evidence():
    hits = search_knowledge("主动学习 未标注候选 测试集泄漏 acquisition", limit=3)
    ids = {hit["id"] for hit in hits}
    assert "active_learning" in ids or "data_leakage" in ids


def test_irrelevant_question_has_no_curated_match():
    assert search_knowledge("weather forecast for tomorrow", limit=3) == []


def test_result_limit_and_citations():
    hits = search_knowledge("membrane gas adsorption permeability diffusion", limit=2)
    assert len(hits) <= 2
    assert all(hit["source_url"] and hit["citation"] for hit in hits)


def test_paper_corpus_size_doi_dedup_and_metadata_provenance():
    path = Path(__file__).resolve().parents[1] / "bimem_agent" / "knowledge" / "papers.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    assert 45 <= len(records) <= 65
    assert len({record["doi"] for record in records}) == len(records)
    assert all(record["title"] and record["source_url"] == "https://doi.org/" + record["doi"] for record in records)
    assert all(record["metadata_source"] == "Crossref REST API" for record in records)


def test_paper_retrieval_across_scientific_topics():
    cases = [
        ("Uni-MOF transformer gas adsorption", "10.1038/s41467-024-46276-x"),
        ("classical density functional theory methane adsorption", "10.1002/aic.14877"),
        ("active learning adsorption simulations metal organic frameworks", "10.1021/acs.iecr.3c01589"),
        ("ZIF-8 ethene ethane membrane diffusion", "10.1016/j.memsci.2010.12.001"),
    ]
    for question, expected_doi in cases:
        hits = search_papers(question, limit=5)
        assert expected_doi in {hit["doi"] for hit in hits}, question
        assert all(hit["evidence_level"] == "bibliographic_lead_only" for hit in hits)


def test_paper_retrieval_rejects_unrelated_query():
    assert search_papers("weather forecast for tomorrow", limit=5) == []
