"""Tool-calling orchestration loop.

This module contains the actual tool-calling *lifecycle*:

    1. Send the user's question + tool schemas to the LLM.
    2. Inspect the response: does it want to call a tool?
    3. If yes: extract the tool name + arguments, execute it, send the
       result back to the LLM, and repeat.
    4. If no: the LLM's text is the final answer.

Nothing here decides "does this question need weather" by inspecting text --
that decision is made entirely by the LLM, based on the tool schemas it was
given. This module only reacts to what the LLM decides.

Message format follows the OpenAI-compatible "tools" / "tool_calls"
convention used by Groq: the LLM's turn may include a `tool_calls` list, and
each executed tool's result is appended as its own `{"role": "tool", ...}`
message referencing the corresponding `tool_call_id`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from src.llm.client import LLMClientError
from src.tools.registry import ToolArgumentError, UnknownToolError

logger = logging.getLogger(__name__)

# Safety valve: caps how many times we'll go back to the LLM with tool
# results before giving up, so a misbehaving model can't loop forever.
MAX_TOOL_ROUNDS = 4


class OrchestratorError(Exception):
    """Raised when the conversation cannot be completed at all (e.g. LLM unreachable)."""


class Orchestrator:
    """Drives a single user question through the LLM tool-calling loop."""

    def __init__(
        self,
        llm_client: Any,
        tool_schemas: list[dict[str, Any]],
        execute_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._llm_client = llm_client
        self._tool_schemas = tool_schemas
        self._execute_tool = execute_tool

    def ask(self, question: str) -> str:
        """Run the full tool-calling loop for one user question and return the final answer."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        for round_number in range(1, MAX_TOOL_ROUNDS + 1):
            try:
                response = self._llm_client.create_message(messages, self._tool_schemas)
            except LLMClientError as exc:
                raise OrchestratorError(f"LLM request failed: {exc}") from exc

            choice = response.choices[0]
            message = choice.message
            tool_calls = list(message.tool_calls or [])

            if choice.finish_reason != "tool_calls" or not tool_calls:
                return _extract_text(message)

            logger.info(
                "Round %d: LLM requested %d tool call(s)", round_number, len(tool_calls)
            )
            messages.append(message.model_dump(exclude_none=True))
            messages.extend(self._run_tools(tool_calls))

        logger.warning("Reached MAX_TOOL_ROUNDS (%d) without a final answer", MAX_TOOL_ROUNDS)
        return (
            "I wasn't able to finish answering that after several tool calls. "
            "Please try rephrasing your question."
        )

    def _run_tools(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Execute each requested tool call and build the tool-result messages."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            logger.info("Calling tool: %s(%s)", tool_name, raw_arguments)

            payload = self._call_one_tool(tool_name, raw_arguments)

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(payload),
                }
            )
        return results

    def _call_one_tool(self, tool_name: str, raw_arguments: str) -> dict[str, Any]:
        """Run a single tool call, translating any failure into a tool-result payload.

        Returning the error as a tool result (rather than raising) lets the
        LLM see what went wrong and respond sensibly (e.g. apologize, ask for
        clarification) instead of crashing the whole conversation.
        """
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            logger.warning("Tool %s received malformed JSON arguments: %s", tool_name, exc)
            return {"error": f"Arguments for tool '{tool_name}' were not valid JSON"}

        try:
            result = self._execute_tool(tool_name, arguments)
            logger.info("Tool %s succeeded", tool_name)
            return result
        except UnknownToolError as exc:
            logger.warning("Unknown tool requested: %s", exc)
            return {"error": f"Unknown tool: {exc}"}
        except ToolArgumentError as exc:
            logger.warning("Invalid arguments for tool %s: %s", tool_name, exc)
            return {"error": f"Invalid arguments: {exc}"}
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any tool failure
            # becomes a structured, recoverable tool result rather than a crash.
            logger.exception("Unexpected error while executing tool '%s'", tool_name)
            return {"error": f"Tool execution failed: {exc}"}


def _extract_text(message: Any) -> str:
    """Pull the plain-text answer out of the LLM's final message."""
    text = (message.content or "").strip()
    return text or "(The model returned an empty response.)"
