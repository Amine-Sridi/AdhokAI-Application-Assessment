"""CLI entry point for the weather tool-calling demo.

    python main.py
    Ask a question: What's the weather in Hyderabad?

This file only wires components together and handles the console
interaction; all the interesting logic lives in `src/`.
"""

from __future__ import annotations

import logging
import sys

from src.config import ConfigurationError, load_config
from src.llm.client import GroqLLMClient
from src.orchestrator import Orchestrator, OrchestratorError
from src.tools.registry import TOOL_SCHEMAS, execute_tool

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_orchestrator() -> Orchestrator:
    config = load_config()
    llm_client = GroqLLMClient(api_key=config.llm_api_key, model=config.llm_model)
    return Orchestrator(
        llm_client=llm_client,
        tool_schemas=TOOL_SCHEMAS,
        execute_tool=execute_tool,
    )


def main() -> int:
    try:
        orchestrator = build_orchestrator()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        question = input("Ask a question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0

    if not question:
        print("Please enter a non-empty question.")
        return 1

    try:
        answer = orchestrator.ask(question)
    except OrchestratorError as exc:
        print(f"\nSomething went wrong while talking to the LLM: {exc}")
        return 1

    print(f"\nAssistant: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
