"""Router — the core execution loop.

Inspired by swarm.Swarm but using Anthropic API:
1. Start with an agent
2. Call Claude API with agent's instructions + tools
3. If tool calls → execute, feed results back
4. If tool returns Agent → handoff to that agent
5. If tool returns AgentResult with agent → handoff + inject context
6. Repeat until agent returns plain text (no tool calls)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import anthropic

from .agent import Agent, AgentResult
from .registry import ToolRegistry, get_registry
from .config import AgentConfig, get_config


class Router:
    """Swarm-inspired router that manages agent execution and handoff.
    
    Key features:
    - Agent handoff: tool functions can return Agent to transfer control
    - Context variables: shared state between agents
    - Model routing: different agents can use different models
    - Max turn limits: prevent infinite loops
    """
    
    def __init__(
        self,
        config: AgentConfig | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.config = config or get_config()
        self.registry = registry or get_registry()
        self.client = anthropic.Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url if self.config.base_url else None,
        )
    
    def run(
        self,
        agent: Agent,
        user_message: str,
        context_variables: Dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> tuple[str, Dict[str, Any]]:
        """Run an agent loop until completion.
        
        Returns:
            (final_text_response, final_context_variables)
        """
        context = dict(context_variables or {})
        current_agent = agent
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message}]
        
        for turn in range(current_agent.max_turns):
            model = self.config.model if self.config.force_model_override else (current_agent.model or self.config.model)
            instructions = current_agent.get_instructions(context)
            tools = self.registry.claude_tools(
                [f.__name__ for f in current_agent.functions]
            ) if current_agent.functions else []
            
            if verbose:
                print(f"\n[{current_agent.name}] Turn {turn + 1}...", flush=True)
            
            # Call Claude API
            response = self._call_api(messages, instructions, tools, model)
            
            # Print text content
            if verbose:
                for block in response.content:
                    if hasattr(block, "text"):
                        print(block.text, flush=True)
            
            # Extract tool calls
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            
            if not tool_calls:
                # No tools → final response
                text = "\n".join(
                    b.text for b in response.content if hasattr(b, "text")
                ) or "(no text response)"
                return text, context
            
            # Process tool calls
            # Add assistant message
            messages.append({"role": "assistant", "content": response.content})
            
            # Build function map
            func_map = {f.__name__: f for f in current_agent.functions}
            
            tool_results = []
            handoff_agent: Optional[Agent] = None
            
            for tc in tool_calls:
                if verbose:
                    print(f"  🔧 {tc.name}", flush=True)
                
                func = func_map.get(tc.name)
                if func is None:
                    result_str = f"Error: tool '{tc.name}' not available for agent '{current_agent.name}'"
                else:
                    try:
                        raw_result = func(**tc.input)
                        
                        # Handle AgentResult (swarm pattern)
                        if isinstance(raw_result, AgentResult):
                            if raw_result.agent:
                                handoff_agent = raw_result.agent
                            context.update(raw_result.context_variables)
                            result_str = raw_result.value
                        elif isinstance(raw_result, Agent):
                            handoff_agent = raw_result
                            result_str = json.dumps({"handed_off_to": raw_result.name})
                        else:
                            result_str = (
                                raw_result if isinstance(raw_result, str)
                                else json.dumps(raw_result, ensure_ascii=False, default=str)
                            )
                    except Exception as e:
                        result_str = f"Error: {e}"
                
                # Truncate long results
                if len(result_str) > 50000:
                    result_str = result_str[:50000] + "\n... (truncated)"
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_str,
                })
            
            messages.append({"role": "user", "content": tool_results})
            
            # Handle handoff
            if handoff_agent:
                if verbose:
                    print(f"  ✈️  Handoff → {handoff_agent.name}", flush=True)
                current_agent = handoff_agent
                # Don't reset messages — carry conversation forward
        
        return f"(max turns ({agent.max_turns}) reached)", context
    
    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        tools: List[Dict[str, Any]],
        model: str,
    ) -> anthropic.types.Message:
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.messages.create(**kwargs)
    
    def run_named(
        self,
        agent_name: str,
        user_message: str,
        agents: Dict[str, Agent] | None = None,
        context_variables: Dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> tuple[str, Dict[str, Any]]:
        """Run by agent name. agents dict must map name → Agent."""
        if agents is None:
            from . import agents as default_agents
            agents = default_agents.AGENT_REGISTRY
        
        agent = agents.get(agent_name)
        if agent is None:
            raise KeyError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")
        return self.run(agent, user_message, context_variables, verbose)
