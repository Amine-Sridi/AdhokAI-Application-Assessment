"""Tests for src/tools/registry.py: dispatch, validation, and error handling."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.tools.registry import (
    TOOL_SCHEMAS,
    ToolArgumentError,
    UnknownToolError,
    execute_tool,
)


def test_get_weather_is_routed_to_the_weather_implementation():
    # execute_tool looks up the implementation via _TOOL_IMPLEMENTATIONS, which
    # was populated with a reference to get_weather at import time. Patch the
    # module-level dict entry directly to intercept and verify the routing.
    with patch.dict(
        "src.tools.registry._TOOL_IMPLEMENTATIONS",
        {"get_weather": lambda location: {"location": location, "ok": True}},
    ):
        result = execute_tool("get_weather", {"location": "Hyderabad"})

    assert result == {"location": "Hyderabad", "ok": True}


def test_unknown_tool_raises_unknown_tool_error():
    with pytest.raises(UnknownToolError):
        execute_tool("get_stock_price", {"ticker": "ANTH"})


def test_missing_required_argument_raises_tool_argument_error():
    with pytest.raises(ToolArgumentError):
        execute_tool("get_weather", {})


def test_unexpected_argument_raises_tool_argument_error():
    with pytest.raises(ToolArgumentError):
        execute_tool("get_weather", {"location": "Hyderabad", "unit": "celsius"})


def test_non_dict_arguments_raise_tool_argument_error():
    with pytest.raises(ToolArgumentError):
        execute_tool("get_weather", ["Hyderabad"])  # type: ignore[arg-type]


def test_weather_tool_schema_is_registered_and_well_formed():
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert "get_weather" in names

    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "get_weather")
    assert schema["type"] == "function"
    assert schema["function"]["parameters"]["required"] == ["location"]
    assert "location" in schema["function"]["parameters"]["properties"]
