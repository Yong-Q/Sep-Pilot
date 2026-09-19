"""Agent configuration and API settings."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/home/user/gcmc_agent/BiMemAgent")
ENV_DIR = PROJECT_ROOT / "env"


@dataclass
class AgentConfig:
    """Central configuration for the agent system."""

    # API settings
    api_key: str = ""
    base_url: str = ""
    model: str = "mimo-v2.5-pro"  # Use mimo-v2.5-pro as default
    max_tokens: int = 16384  # Increase for longer responses
    temperature: float = 0.7  # Add temperature for more creative responses

    # Orchestrator settings
    orchestrator_model: str = ""
    specialist_model: str = ""
    # A user-supplied connection is a complete endpoint/model choice.  It must
    # override Agent definition defaults instead of only filling blank models.
    force_model_override: bool = False
    max_orchestrator_turns: int = 50
    max_specialist_turns: int = 20

    # Paths
    project_root: Path = PROJECT_ROOT
    legacy_root: Path = LEGACY_ROOT

    # Tool execution
    execute_by_default: bool = False

    # Interaction settings
    ask_for_missing_params: bool = True  # Ask user for missing parameters
    require_all_agents: bool = True  # Require all agents to participate
    require_literature_alignment: bool = True  # Require literature alignment

    # Tool paths (loaded from env/tools.json)
    _tools_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.orchestrator_model:
            self.orchestrator_model = self.model
        if not self.specialist_model:
            self.specialist_model = self.model
        # Load tools config
        self._load_tools_config()

    def _load_tools_config(self):
        """Load tool paths from env/tools.json"""
        tools_path = self.project_root / "env" / "tools.json"
        if tools_path.exists():
            with open(tools_path, "r") as f:
                self._tools_config = json.load(f)

    @property
    def zeopp_path(self) -> str:
        return self._tools_config.get("zeopp", {}).get("path", "/home/user/zeo++-0.3/network")

    @property
    def raspa_path(self) -> str:
        return self._tools_config.get("raspa", {}).get("path", "/home/user/RASPA2/simulations")

    @property
    def pormake_path(self) -> str:
        return self._tools_config.get("pormake", {}).get("path", "/opt/pormake")

    @property
    def pormake_python(self) -> str:
        configured = self._tools_config.get("pormake", {}).get("python")
        return configured or str(Path(self.pormake_path) / "bin" / "python")

    @property
    def conda_path(self) -> str:
        return self._tools_config.get("conda", {}).get("path", "/opt/conda/miniconda/3-python3.9.13/etc/profile.d/conda.sh")

    @classmethod
    def from_env_dir(cls, env_dir: Path | str | None = None, **overrides: Any) -> AgentConfig:
        """Load config from env/settings.json (third-party proxy format)."""
        env_dir = Path(env_dir) if env_dir else ENV_DIR
        settings_path = env_dir / "settings.json"

        if not settings_path.exists():
            raise FileNotFoundError(f"No settings.json found at {settings_path}")

        with open(settings_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        env = raw.get("env", {})
        config = cls(
            api_key=env.get("ANTHROPIC_AUTH_TOKEN", os.environ.get("ANTHROPIC_API_KEY", "")),
            base_url=env.get("ANTHROPIC_BASE_URL", ""),
            model=env.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_env(cls, **overrides: Any) -> AgentConfig:
        """Create config from environment variables with overrides."""
        # Try env dir first
        if ENV_DIR.exists() and (ENV_DIR / "settings.json").exists():
            return cls.from_env_dir(**overrides)

        config = cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        )
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = AgentConfig.from_env()
    return _config


def set_config(config: AgentConfig) -> None:
    """Set the global config singleton."""
    global _config
    _config = config
