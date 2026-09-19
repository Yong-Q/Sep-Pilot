from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bimem_agent.paths import LEGACY_ROOT, ensure_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the legacy scientific literature RAG index.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument(
        "--query-type",
        default="auto",
        choices=["auto", "overview", "methods", "mechanism", "benchmark", "parameter", "material_lookup"],
    )
    args = parser.parse_args()

    ensure_project_path()
    from rag.retriever import KnowledgeBase

    t0 = time.time()
    kb = KnowledgeBase()
    result = kb.query_structured(args.query, n_results=args.n_results, query_type=args.query_type)
    payload = {
        **result,
        "elapsed_s": round(time.time() - t0, 2),
        "formatted": kb.format_search_result(result),
        "python": sys.executable,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
