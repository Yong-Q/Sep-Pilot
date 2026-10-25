"""Portable core paths must not require a workstation's private config."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "bimem_agent" / "paths.py"


def load_isolated_paths(tmp_path, monkeypatch, config=None):
    monkeypatch.delenv("BIMEM_RUNTIME_CONFIG", raising=False)
    package = tmp_path / "bimem_agent"
    package.mkdir()
    module_file = package / "paths.py"
    shutil.copyfile(SOURCE, module_file)
    if config is not None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "runtime.json").write_text(json.dumps(config))
    spec = importlib.util.spec_from_file_location("portable_paths_under_test", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_private_config_still_allows_core_paths(tmp_path, monkeypatch):
    module = load_isolated_paths(tmp_path, monkeypatch)
    assert module.PROJECT_ROOT == tmp_path
    assert module.LEGACY_ROOT == tmp_path
    assert module.AGENT_ENV_PYTHON == Path(sys.executable)
    assert module.get_scheduler_defaults() == {}
    assert module.get_runtime_section("remote_slurm") == {}
    assert not (tmp_path / "config").exists(), "Import must not write a config"


def test_unconfigured_scientific_software_fails_explicitly(tmp_path, monkeypatch):
    module = load_isolated_paths(tmp_path, monkeypatch)
    with pytest.raises(KeyError, match="software.pacman_dir"):
        module.get_config_path("software", "pacman_dir")


def test_partial_site_config_keeps_core_defaults(tmp_path, monkeypatch):
    module = load_isolated_paths(tmp_path, monkeypatch, {
        "software": {"pacman_dir": "tools/charge"},
        "scheduler_defaults": {"charge": {"queue": "user-queue"}},
    })
    assert module.LEGACY_ROOT == tmp_path
    assert module.get_config_path("software", "pacman_dir") == tmp_path / "tools/charge"
    assert module.get_scheduler_defaults()["charge"]["queue"] == "user-queue"


def test_existing_config_paths_remain_authoritative(tmp_path, monkeypatch):
    module = load_isolated_paths(tmp_path, monkeypatch, {
        "paths": {"legacy_root": "external", "agent_env_python": "/opt/python/bin/python"},
    })
    assert module.LEGACY_ROOT == tmp_path / "external"
    assert module.AGENT_ENV_PYTHON == Path("/opt/python/bin/python")


def test_explicit_config_override_is_used(tmp_path, monkeypatch):
    module = load_isolated_paths(tmp_path, monkeypatch)
    private_config = tmp_path / "site.json"
    private_config.write_text(json.dumps({"paths": {"legacy_root": "other"}}))
    monkeypatch.setenv("BIMEM_RUNTIME_CONFIG", str(private_config))
    spec = importlib.util.spec_from_file_location("override_paths", module.__file__)
    override = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(override)
    assert override.LEGACY_ROOT == tmp_path / "other"


def test_explicit_missing_config_is_not_silently_ignored(tmp_path, monkeypatch):
    module = load_isolated_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("BIMEM_RUNTIME_CONFIG", str(tmp_path / "missing.json"))
    spec = importlib.util.spec_from_file_location("missing_paths", module.__file__)
    override = importlib.util.module_from_spec(spec)
    with pytest.raises(FileNotFoundError):
        spec.loader.exec_module(override)


@pytest.mark.parametrize("config", [[], {"paths": []}, {"paths": {"legacy_root": None}}, {"paths": {"legacy_root": ""}}])
def test_invalid_config_is_not_replaced_with_defaults(tmp_path, monkeypatch, config):
    with pytest.raises((ValueError, TypeError)):
        load_isolated_paths(tmp_path, monkeypatch, config)
