from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "registry" / "catalog.json"
CONFIG_ROOT = PROJECT_ROOT / "config"
_RUNTIME_CONFIG_OVERRIDE = os.environ.get("BIMEM_RUNTIME_CONFIG")
RUNTIME_CONFIG_PATH = (
    Path(_RUNTIME_CONFIG_OVERRIDE).expanduser()
    if _RUNTIME_CONFIG_OVERRIDE else CONFIG_ROOT / "runtime.json"
)


@lru_cache(maxsize=1)
def load_runtime_config() -> dict:
    """Load optional site settings without requiring them to import the core.

    Only the project directory and running interpreter have portable defaults.
    Scientific executables, datasets and scheduler settings remain user supplied.
    An explicitly selected config must exist; broken configs never fall back.
    """
    try:
        with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        if _RUNTIME_CONFIG_OVERRIDE:
            raise
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Runtime config must be a JSON object")
    configured_paths = config.get("paths", {})
    if not isinstance(configured_paths, dict):
        raise ValueError("Runtime config paths must be a JSON object")
    return {
        **config,
        "paths": {
            "legacy_root": str(PROJECT_ROOT),
            "agent_env_python": sys.executable,
            **configured_paths,
        },
    }


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
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config entry {section}.{key} must be a nonempty path string")
    return _resolve_path(value)


def get_scheduler_defaults() -> dict:
    return dict(load_runtime_config().get("scheduler_defaults", {}))


def get_runtime_section(section: str) -> dict:
    return dict(load_runtime_config().get(section, {}))


LEGACY_ROOT = get_config_path("paths", "legacy_root")
LEGACY_PARENT = LEGACY_ROOT.parent
AGENT_ENV_PYTHON = get_config_path("paths", "agent_env_python")
PORMAKE_DATABASE_DIR = (
    get_config_path('software', 'pormake_database_dir')
    if 'pormake_database_dir' in get_runtime_section('software')
    else LEGACY_PARENT / 'pormake_db'
)


def ensure_project_path() -> None:
    import sys

    for path in (str(PROJECT_ROOT), str(LEGACY_PARENT), str(LEGACY_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
