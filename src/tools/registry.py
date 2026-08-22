"""Tool registry: schema definitions + dispatch.

This is the single place that knows which tools exist, what their JSON
schemas look like (as advertised to the LLM), and which Python callable each
tool name maps to. The orchestrator never talks to `weather.py` directly --
it always goes through `execute_tool`, which keeps tool implementations easy
to add/remove/swap.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from src.tools.weather import get_weather

# --- Tool schemas -----------------------------------------------------------
# These are sent to the LLM verbatim, in the OpenAI-compatible "function"
# tool format used by Groq (and most other providers). The LLM decides, based
# on the user's question and these descriptions, whether and how to call each
# tool. Nothing here inspects the user's text.

WEATHER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather conditions (temperature, wind speed, and "
            "general conditions) for a given city or location. Use this whenever "
            "the user asks about current/live weather for a specific place."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "City name, optionally with region/country, e.g. "
                        "'Hyderabad' or 'Paris, France'."
                    ),
                }
            },
            "required": ["location"],
        },
    },
}

TOOL_SCHEMAS: list[dict[str, Any]] = [WEATHER_TOOL_SCHEMA]

# --- Dispatch table ----------------------------------------------------------

_TOOL_IMPLEMENTATIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_weather": get_weather,
}


class UnknownToolError(Exception):
    """Raised when the LLM requests a tool that isn't registered."""


class ToolArgumentError(Exception):
    """Raised when the LLM supplies arguments that don't match a tool's signature."""


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up and run the tool implementation for `name` with `arguments`.

    Args:
        name: Tool name, as emitted by the LLM in a tool_use block.
        arguments: Parsed JSON arguments, as emitted by the LLM.

    Returns:
        A JSON-serializable dict with the tool's result.

    Raises:
        UnknownToolError: If `name` is not a registered tool.
        ToolArgumentError: If `arguments` don't match the tool's parameters.
        Exception: Any exception the underlying tool implementation raises
            (e.g. WeatherServiceError) propagates unchanged, so callers can
            handle domain-specific failures distinctly from dispatch failures.
    """
    if not isinstance(arguments, dict):
        raise ToolArgumentError(f"Arguments for tool '{name}' must be a JSON object")

    implementation = _TOOL_IMPLEMENTATIONS.get(name)
    if implementation is None:
        raise UnknownToolError(f"No tool registered with name '{name}'")

    _validate_arguments(name, implementation, arguments)

    return implementation(**arguments)


def _validate_arguments(
    name: str, implementation: Callable[..., Any], arguments: dict[str, Any]
) -> None:
    """Check `arguments` against the implementation's signature before calling it.

    This catches missing required parameters or unexpected extra keys with a
    clear error, rather than letting a raw TypeError bubble up from deep
    inside the call.
    """
    signature = inspect.signature(implementation)
    valid_params = set(signature.parameters)
    required_params = {
        param_name
        for param_name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
    }

    unexpected = set(arguments) - valid_params
    if unexpected:
        raise ToolArgumentError(
            f"Tool '{name}' received unexpected argument(s): {sorted(unexpected)}"
        )

    missing = required_params - set(arguments)
    if missing:
        raise ToolArgumentError(
            f"Tool '{name}' is missing required argument(s): {sorted(missing)}"
        )
