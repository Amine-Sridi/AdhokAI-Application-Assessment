"""Tests for src/orchestrator.py.

A fake LLM client stands in for the real Groq client so these tests exercise
the full tool-calling *lifecycle* -- tool_calls detection, argument
extraction, tool execution, tool-result round-trip, final answer -- without
any network access or API key. The fake response objects mirror the
OpenAI-compatible shape Groq returns (choices[0].message.tool_calls /
.content, choices[0].finish_reason).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.orchestrator import MAX_TOOL_ROUNDS, Orchestrator
from src.tools.registry import TOOL_SCHEMAS, UnknownToolError


@dataclass
class FakeFunction:
    name: str
    arguments: str  # JSON-encoded string, as the real API sends it


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] = field(default_factory=list)
    role: str = "assistant"

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        dumped: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            dumped["content"] = self.content
        if self.tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return dumped


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


class FakeLLMClient:
    """Returns a pre-scripted sequence of responses, one per call."""

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def create_message(self, messages, tools):
        self.calls.append(messages)
        assert tools == TOOL_SCHEMAS  # orchestrator must always pass the tool schemas through
        return self._responses.pop(0)


def _text_response(text: str) -> FakeResponse:
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=text), finish_reason="stop")])


def _tool_use_response(name: str, arguments: dict[str, Any], call_id: str = "call_fake") -> FakeResponse:
    tool_call = FakeToolCall(id=call_id, function=FakeFunction(name=name, arguments=json.dumps(arguments)))
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(tool_calls=[tool_call]), finish_reason="tool_calls")]
    )


def test_no_tool_needed_returns_direct_text_answer():
    llm = FakeLLMClient([_text_response("Paris is the capital of France.")])
    orchestrator = Orchestrator(llm, TOOL_SCHEMAS, execute_tool=lambda name, args: {})

    answer = orchestrator.ask("What is the capital of France?")

    assert answer == "Paris is the capital of France."
    assert len(llm.calls) == 1  # tool was never invoked


def test_weather_question_triggers_full_tool_calling_lifecycle():
    llm = FakeLLMClient(
        [
            _tool_use_response("get_weather", {"location": "Hyderabad"}),
            _text_response("It's 28°C and partly cloudy in Hyderabad."),
        ]
    )
    executed_calls = []

    def fake_execute_tool(name, arguments):
        executed_calls.append((name, arguments))
        return {"temperature_c": 28.0, "condition": "partly cloudy"}

    orchestrator = Orchestrator(llm, TOOL_SCHEMAS, execute_tool=fake_execute_tool)

    answer = orchestrator.ask("What's the weather in Hyderabad?")

    assert answer == "It's 28°C and partly cloudy in Hyderabad."
    assert executed_calls == [("get_weather", {"location": "Hyderabad"})]
    assert len(llm.calls) == 2  # one round trip for the tool call, one for the final answer

    # The second call to the LLM must include the tool result in the conversation.
    second_call_messages = llm.calls[1]
    roles = [m["role"] for m in second_call_messages]
    assert roles == ["user", "assistant", "tool"]

    tool_result_message = second_call_messages[-1]
    assert tool_result_message["tool_call_id"] == "call_fake"
    assert json.loads(tool_result_message["content"]) == {
        "temperature_c": 28.0,
        "condition": "partly cloudy",
    }

    # The assistant turn we send back must carry the original tool_calls too.
    assistant_message = second_call_messages[1]
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"


def test_unknown_tool_request_is_reported_back_to_llm_instead_of_crashing():
    llm = FakeLLMClient(
        [
            _tool_use_response("get_stock_price", {"ticker": "ANTH"}),
            _text_response("I don't have a way to check stock prices."),
        ]
    )

    def fake_execute_tool(name, arguments):
        raise UnknownToolError(f"No tool registered with name '{name}'")

    orchestrator = Orchestrator(llm, TOOL_SCHEMAS, execute_tool=fake_execute_tool)

    answer = orchestrator.ask("What's Anthropic's stock price?")

    assert answer == "I don't have a way to check stock prices."
    tool_result_message = llm.calls[1][-1]
    assert "error" in json.loads(tool_result_message["content"])


def test_malformed_tool_call_arguments_are_reported_back_instead_of_crashing():
    bad_call = FakeToolCall(id="call_bad", function=FakeFunction(name="get_weather", arguments="{not valid json"))
    llm = FakeLLMClient(
        [
            FakeResponse(
                choices=[FakeChoice(message=FakeMessage(tool_calls=[bad_call]), finish_reason="tool_calls")]
            ),
            _text_response("Sorry, I couldn't parse that request."),
        ]
    )
    orchestrator = Orchestrator(llm, TOOL_SCHEMAS, execute_tool=lambda name, args: {})

    answer = orchestrator.ask("weather??")

    assert answer == "Sorry, I couldn't parse that request."
    tool_result_message = llm.calls[1][-1]
    assert "error" in json.loads(tool_result_message["content"])


def test_gives_up_gracefully_after_max_tool_rounds():
    # The LLM (mis)behaves by requesting a tool call every single round.
    responses = [
        _tool_use_response("get_weather", {"location": "Hyderabad"}) for _ in range(MAX_TOOL_ROUNDS)
    ]
    llm = FakeLLMClient(responses)
    orchestrator = Orchestrator(llm, TOOL_SCHEMAS, execute_tool=lambda name, args: {"temperature_c": 20.0})

    answer = orchestrator.ask("What's the weather in Hyderabad?")

    assert "couldn't" in answer.lower() or "wasn't able" in answer.lower()
    assert len(llm.calls) == MAX_TOOL_ROUNDS
