from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "registry" / "catalog.json"
CONFIG_ROOT = PROJECT_ROOT / "config"
RUNTIME_CONFIG_PATH = CONFIG_ROOT / "runtime.json"


@lru_cache(maxsize=1)
def load_runtime_config() -> dict:
    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_config_path(section: str, key: str) -> Path:
    config = load_runtime_config()
    try:
        value = config[section][key]
    except KeyError as exc:
        raise KeyError(f"Missing config entry: {section}.{key}") from exc
    return _resolve_path(value)


def get_scheduler_defaults() -> dict:
    return dict(load_runtime_config().get("scheduler_defaults", {}))


def get_runtime_section(section: str) -> dict:
    return dict(load_runtime_config().get(section, {}))


LEGACY_ROOT = get_config_path("paths", "legacy_root")
LEGACY_PARENT = LEGACY_ROOT.parent
AGENT_ENV_PYTHON = get_config_path("paths", "agent_env_python")


def ensure_project_path() -> None:
    import sys

    for path in (str(PROJECT_ROOT), str(LEGACY_PARENT), str(LEGACY_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
