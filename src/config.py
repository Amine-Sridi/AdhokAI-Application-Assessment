"""Environment-based configuration.

Centralizes reading required/optional environment variables so the rest of
the app never touches `os.environ` directly, and so missing configuration
produces one clear error instead of a confusing failure deep in the call
stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.llm.client import DEFAULT_MODEL


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    llm_api_key: str
    llm_model: str


def load_config() -> AppConfig:
    """Load configuration from environment variables (and a local .env file).

    Required:
        GROQ_API_KEY: Free API key for the Groq Chat Completions API
            (https://console.groq.com/keys -- no credit card required).

    Optional:
        LLM_MODEL: Overrides the default model string.
    """
    load_dotenv()  # no-op if there is no .env file; never overwrites real env vars set by the shell

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "Missing required environment variable 'GROQ_API_KEY'.\n"
            "Groq's free tier needs no credit card -- grab a key at "
            "https://console.groq.com/keys, then:\n"
            "  cp .env.example .env\n"
            "  echo 'GROQ_API_KEY=gsk_...' >> .env"
        )

    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL

    return AppConfig(llm_api_key=api_key, llm_model=model)
