"""LLM client: thin wrapper around the Groq Chat Completions API.

Groq was chosen as the LLM provider because it has a genuinely free tier (no
credit card required) with fast, reliable tool/function calling, and its API
is OpenAI-compatible -- so this client speaks the same "tools" / "tool_calls"
message format used by the OpenAI SDK and many other providers.

Isolating the provider call here means the orchestrator never imports the
`groq` package directly -- it only depends on `LLMClient.create_message` and
the shape of the response. Swapping providers later means writing a new class
with the same interface, not touching the orchestration logic.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import groq

logger = logging.getLogger(__name__)

# llama-3.3-70b-versatile is Groq's general-purpose model with solid tool-use
# support, available on the free tier.
DEFAULT_MODEL = "qwen/qwen3.6-27b"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to a get_weather tool that "
    "returns live weather conditions for a named location. Call it whenever "
    "the user's question depends on current weather data. For anything else, "
    "answer directly without calling a tool."
)


class LLMClientError(Exception):
    """Raised when the LLM request fails (network, auth, rate limit, etc.)."""


class LLMResponse(Protocol):
    """Structural type describing the parts of the SDK response we rely on.

    Documented here so the orchestrator's expectations are explicit, even
    though at runtime this is just whatever `groq.Groq().chat.completions
    .create(...)` returns (an OpenAI-compatible ChatCompletion object).
    """

    choices: list[Any]


class GroqLLMClient:
    """Wraps the Groq Chat Completions API for tool-calling conversations."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = groq.Groq(api_key=api_key)
        self._model = model

    def create_message(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Send the conversation so far (+ available tools) to the LLM.

        Args:
            messages: OpenAI-format message list (roles: user/assistant/tool).
            tools: Tool schemas, in OpenAI "function" format, the LLM may call.

        Returns:
            The raw SDK ChatCompletion response. `.choices[0].message` holds
            `.content` (text) and `.tool_calls` (a list of requested calls);
            `.choices[0].finish_reason` is `"tool_calls"` when the model wants
            to call a tool, or `"stop"` for a normal final answer.
        """
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        try:
            return self._client.chat.completions.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                messages=full_messages,
                tools=tools,
                tool_choice="auto",
            )
        except groq.APIConnectionError as exc:
            raise LLMClientError(f"Could not reach the LLM provider: {exc}") from exc
        except groq.RateLimitError as exc:
            raise LLMClientError(f"LLM rate limit exceeded: {exc}") from exc
        except groq.APIStatusError as exc:
            raise LLMClientError(
                f"LLM provider returned an error (status {exc.status_code}): {exc}"
            ) from exc
        except groq.APIError as exc:
            raise LLMClientError(f"LLM request failed: {exc}") from exc
