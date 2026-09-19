"""CLI entry point for BiMemAgent agent system."""
from __future__ import annotations

import argparse
import sys

from .config import AgentConfig, set_config
from .router import Router
from .defns import AGENT_REGISTRY, resolve_agent


def main():
    parser = argparse.ArgumentParser(
        description="BiMemAgent - Claude SDK multi-agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agents "Run GCMC isotherm for CO2 in MOF-5"
  python -m agents --agent adsorption "Run isotherm for CH4 in ZIF-8"
  python -m agents -i                    # interactive mode
  python -m agents --list-agents
""",
    )
    parser.add_argument("query", nargs="?", help="User query")
    parser.add_argument("--agent", "-a", default="lead-orchestrator")
    parser.add_argument("--list-agents", action="store_true")
    parser.add_argument("--model", "-m", help="Override model")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if args.list_agents:
        print("Agents:")
        for name, agent in AGENT_REGISTRY.items():
            tools = [f.__name__ for f in agent.functions]
            handoffs = agent.handoff_to or []
            print(f"\n  {name}")
            print(f"    Tools: {', '.join(tools)}")
            if handoffs:
                print(f"    Delegates to: {', '.join(handoffs)}")
        return

    config = AgentConfig.from_env()
    if args.model:
        config.model = args.model
    set_config(config)

    router = Router(config)

    if args.interactive:
        print("BiMemAgent Interactive (quit to exit)")
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            agent = resolve_agent(args.agent)
            result, ctx = router.run(agent, user_input, verbose=not args.quiet)
            print(f"\n{result}")
        return

    if not args.query:
        parser.print_help()
        return

    agent = resolve_agent(args.agent)
    result, ctx = router.run(agent, args.query, verbose=not args.quiet)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
