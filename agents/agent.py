"""Agent definition — inspired by OpenAI Swarm's elegant pattern.

Core idea (from swarm):
    Agent = name + instructions + functions
    A function can return an Agent → triggers handoff
    A function can return Result(value, agent, context_variables) → rich handoff

This is the simplest, most composable pattern. No inheritance, no complex config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


# Forward reference for type hints
AgentFunc = Callable[..., Union[str, "Agent", dict, None]]


@dataclass
class Agent:
    """A lightweight agent definition.
    
    Modeled after swarm.Agent but extended for our domain:
    - name: unique identifier
    - instructions: system prompt (str or callable)
    - functions: tool functions this agent can call
    - handoff_to: optional list of agent names this agent can delegate to
    - model: optional model override
    - max_turns: max conversation turns before forced stop
    """
    name: str = "Agent"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunc] = field(default_factory=list)
    handoff_to: List[str] = field(default_factory=list)
    model: Optional[str] = None
    max_turns: int = 30

    def function_map(self) -> Dict[str, AgentFunc]:
        """Resolve exposed tools without silent last-writer-wins routing."""
        result = {}
        for function in self.functions:
            name = function.__name__
            if name in result and result[name] is not function:
                raise ValueError(f"ambiguous tool name {name!r} in agent {self.name!r}")
            result[name] = function
        return result
    
    def get_instructions(self, context: Dict[str, Any] | None = None) -> str:
        """Resolve instructions (supports callable)."""
        if callable(self.instructions):
            return self.instructions(context or {})
        return self.instructions


@dataclass
class AgentResult:
    """Rich return from a tool function — controls agent handoff.
    
    Inspired by swarm.Result:
    - value: string result to show
    - agent: handoff to this agent (triggers routing)
    - context_variables: shared state to pass along
    """
    value: str = ""
    agent: Optional[Agent] = None
    context_variables: Dict[str, Any] = field(default_factory=dict)


# Type alias for what a tool function can return
ToolResult = Union[str, AgentResult, Agent, dict, None]
