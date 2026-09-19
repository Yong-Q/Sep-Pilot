from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict

from .paths import CATALOG_PATH


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def skill_index() -> Dict[str, Dict[str, Any]]:
    catalog = load_catalog()
    return {item["id"]: item for item in catalog["skills"]}


def agent_index() -> Dict[str, Dict[str, Any]]:
    catalog = load_catalog()
    return {item["id"]: item for item in catalog["agents"]}
