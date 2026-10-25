"""Dependency-free sparse TF-IDF retrieval over a small reviewed science corpus.

This supplements, rather than replaces, the paper FAISS index. Entries are
paraphrased method guardrails linked to primary publications, not paper text.
The index is rebuilt in memory at startup, so no opaque pickle or account data
is stored or loaded by this route.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any


_CORPUS_PATH = Path(__file__).with_name("curated.json")
_PAPERS_PATH = Path(__file__).with_name("papers.json")
_WORD = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u3400-\u9fff]+", re.I)
_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "under", "with"}
_TOPIC_CONTEXT = {
    "membrane_transport": "MOF membrane transport permeation permeability permeance diffusion selectivity 气体 膜分离 渗透 扩散",
    "adsorption_separation": "MOF gas adsorption separation selectivity breakthrough uptake 吸附 分离 选择性",
    "classical_cdft": "classical density functional theory cDFT pore fluid gas adsorption 经典密度泛函 吸附",
    "molecular_simulation": "GCMC molecular dynamics Monte Carlo adsorption diffusion MOF 分子模拟 扩散",
    "machine_learning": "machine learning neural network representation learning prediction MOF 模型 机器学习",
    "active_learning": "active learning acquisition sequential design adsorption simulation 主动学习 采集",
}


def _terms(text: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for match in _WORD.finditer(text.casefold()):
        token = match.group()
        if "\u3400" <= token[0] <= "\u9fff":
            # Bigrams permit Chinese queries to match short paraphrases while
            # avoiding a separate tokenizer/model installation.
            if len(token) == 1:
                out[token] += 1
            else:
                out.update(token[i : i + 2] for i in range(len(token) - 1))
        elif len(token) > 1 and token not in _STOPWORDS:
            out[token] += 1
    return out


def _vector(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    weights = {term: (1.0 + math.log(freq)) * idf[term]
               for term, freq in counts.items() if term in idf}
    norm = math.sqrt(sum(value * value for value in weights.values()))
    return {term: value / norm for term, value in weights.items()} if norm else {}


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, float]]]:
    with _CORPUS_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Curated science corpus must be a list")
    ids: set[str] = set()
    counts: list[Counter[str]] = []
    for record in records:
        if not isinstance(record, dict) or not {"id", "title", "aliases", "claim", "scope", "caveat", "citation", "source_url"} <= record.keys():
            raise ValueError("Malformed curated science record")
        if record["id"] in ids or not str(record["source_url"]).startswith("https://doi.org/"):
            raise ValueError("Duplicate record or missing primary DOI link")
        ids.add(record["id"])
        counts.append(_terms(" ".join(str(record[key]) for key in ("title", "aliases", "claim", "scope", "caveat"))))
    document_frequency = Counter(term for row in counts for term in row)
    size = len(records)
    idf = {term: math.log((size + 1) / (frequency + 1)) + 1
           for term, frequency in document_frequency.items()}
    return records, idf, [_vector(row, idf) for row in counts]


def search_knowledge(query: str, *, limit: int = 3, min_score: float = 0.13) -> list[dict[str, Any]]:
    """Return relevant reviewed claims with citations, or an honest empty list.

    Scores are sparse-vector cosine similarities for ranking within this small
    corpus. They are not calibrated probabilities or scientific confidence.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(limit, int) or limit < 1 or limit > 10:
        raise ValueError("limit must be an integer between 1 and 10")
    records, idf, vectors = _index()
    query_vector = _vector(_terms(query), idf)
    if not query_vector:
        return []
    ranked: list[dict[str, Any]] = []
    for record, vector in zip(records, vectors):
        score = sum(weight * vector.get(term, 0.0)
                    for term, weight in query_vector.items())
        if score >= min_score:
            ranked.append({
                **{key: record[key] for key in ("id", "title", "claim", "scope", "caveat", "citation", "source_url")},
                "score": round(score, 4),
                "source": "curated_science",
            })
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return ranked[:limit]


@lru_cache(maxsize=1)
def _paper_index() -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, float]]]:
    with _PAPERS_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Paper metadata corpus must be a list")
    seen: set[str] = set()
    counts: list[Counter[str]] = []
    required = {"id", "title", "doi", "year", "journal", "first_author", "topics", "source_url", "metadata_source"}
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("Malformed paper metadata record")
        doi = str(record["doi"]).lower()
        if doi in seen or record["id"] != "doi:" + doi or record["source_url"] != "https://doi.org/" + doi:
            raise ValueError("Duplicate paper DOI or invalid source URL")
        if not record["title"] or not record["topics"] or any(topic not in _TOPIC_CONTEXT for topic in record["topics"]):
            raise ValueError("Paper metadata lacks title or recognized topic")
        seen.add(doi)
        # Titles carry more discriminating information than broad topic tags.
        context = " ".join(_TOPIC_CONTEXT[topic] for topic in record["topics"])
        counts.append(_terms(record["title"] + " " + record["title"] + " " + record.get("aliases", "") + " " + context))
    frequencies = Counter(term for row in counts for term in row)
    size = len(records)
    idf = {term: math.log((size + 1) / (frequency + 1)) + 1
           for term, frequency in frequencies.items()}
    return records, idf, [_vector(row, idf) for row in counts]


def search_papers(query: str, *, limit: int = 5, min_score: float = 0.12) -> list[dict[str, Any]]:
    """Search ~50 verified paper metadata records; never infer paper findings.

    Each title and DOI was retrieved from the Crossref works API. A metadata
    match is a lead for paper reading, not claim-level scientific evidence.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(limit, int) or limit < 1 or limit > 20:
        raise ValueError("limit must be an integer between 1 and 20")
    records, idf, vectors = _paper_index()
    query_vector = _vector(_terms(query), idf)
    if not query_vector:
        return []
    hits: list[dict[str, Any]] = []
    for record, vector in zip(records, vectors):
        score = sum(weight * vector.get(term, 0.0)
                    for term, weight in query_vector.items())
        if score >= min_score:
            hits.append({
                **record,
                "score": round(score, 4),
                "source": "crossref_metadata",
                "evidence_level": "bibliographic_lead_only",
                "note": "Read the paper before citing it for a scientific claim.",
            })
    hits.sort(key=lambda item: (-item["score"], item["doi"]))
    return hits[:limit]
