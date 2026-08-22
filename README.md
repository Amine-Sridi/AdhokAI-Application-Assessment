# Weather Tool-Calling Demo

A small CLI application that demonstrates genuine **LLM tool calling / function calling**:
the user asks a question in natural language, and an LLM (served by [Groq](https://groq.com/),
free tier, no credit card required) decides for itself — based on a tool schema, not
keyword matching — whether it needs live weather data to answer, calls a `get_weather`
tool if so, and uses the result to produce its final answer.

## Overview

This project is intentionally small. It is not a weather app with an LLM bolted on;
it's a demonstration of the tool-calling *lifecycle* itself:

1. The user asks a question.
2. The question and a `get_weather` tool schema are sent to the LLM.
3. The LLM decides, on its own, whether answering requires calling the tool.
4. If it does, the LLM emits a structured tool call (name + arguments).
5. The application executes the corresponding Python function, which calls a
   real weather API.
6. The tool's result is sent back to the LLM.
7. The LLM produces the final natural-language answer, which is shown to the user.

Nothing in the code inspects the user's text for words like "weather". The decision
to call the tool is made entirely by the LLM.

## Architecture

```text
User → LLM (Groq) → [decides] → get_weather tool → Weather API → LLM → User
```

```text
weather-tool-app/
├── main.py                 # CLI entry point: wires everything together
├── src/
│   ├── config.py             # Loads/validates environment variables
│   ├── orchestrator.py       # The tool-calling loop (the core lifecycle)
│   ├── llm/
│   │   └── client.py          # Thin wrapper around the Groq Chat Completions API
│   └── tools/
│       ├── registry.py       # Tool schemas + name -> function dispatch
│       └── weather.py        # get_weather(): geocoding + Open-Meteo forecast
└── tests/
    ├── test_weather_tool.py     # Weather tool: valid input, API failures, bad data
    ├── test_tool_dispatch.py    # Registry: routing, unknown tools, bad arguments
    └── test_orchestrator.py     # Full loop, using a fake LLM client (no API key needed)
```

Each layer only depends on the one below it:

- `main.py` depends on `orchestrator`, `llm.client`, and `tools.registry`.
- `orchestrator.py` depends on an LLM client interface and a tool-dispatch function —
  it does not import `groq` or `weather.py` directly, so either could be swapped.
- `tools/registry.py` depends on `tools/weather.py`, but nothing about the LLM.
- `tools/weather.py` depends on nothing in this project — it's a plain function that
  calls a weather API.

## Setup

```bash
git clone <this-repo>
cd weather-tool-app
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set your GROQ_API_KEY
```

## Environment Variables

| Variable       | Required | Description                                                                 |
|----------------|----------|-------------------------------------------------------------------------------|
| `GROQ_API_KEY` | Yes      | Free API key for the Groq Chat Completions API (https://console.groq.com/keys) |
| `LLM_MODEL`    | No       | Overrides the default model string (see `src/llm/client.py`)                  |

Groq's free tier requires no credit card — sign up, generate a key, and you're
making requests within minutes.

The weather tool uses [Open-Meteo](https://open-meteo.com/), which also requires **no
API key** for geocoding or forecast lookups, so there's nothing else to configure.

If `GROQ_API_KEY` is missing, the app exits immediately with a clear message
explaining how to fix it — it does not crash with a stack trace.

## Running

```bash
python main.py
```

```text
Ask a question: What's the weather in Hyderabad?
```

## Example

```text
Ask a question: What's the weather in Hyderabad?
Round 1: LLM requested 1 tool call(s)
Calling tool: get_weather({"location": "Hyderabad"})
Tool get_weather succeeded

Assistant: The current weather in Hyderabad is about 28°C with partly cloudy
conditions and light wind.
```

```text
Ask a question: What is the capital of France?

Assistant: The capital of France is Paris.
```

Note how the second question never triggers a tool call — the LLM decides the tool
is unnecessary and answers directly, without any code checking for the word "weather".

## How Tool Calling Works

The application does **not** do this:

```python
if "weather" in user_question:
    call_weather_api()
```

Instead, it defines a tool schema (`src/tools/registry.py`, in the OpenAI-compatible
`{"type": "function", "function": {...}}` format Groq uses) that describes what the
`get_weather` tool does and what arguments it takes, and sends that schema to the LLM
alongside every user message:

```text
User question
      ↓
LLM receives the question + get_weather tool schema
      ↓
LLM decides, on its own, whether the tool is necessary
      ↓
  ┌─── if yes ──────────────────────────────┐    if no
  │ LLM emits a tool_calls entry:             │      ↓
  │   { "function": { "name": "get_weather", │  LLM answers directly
  │       "arguments": "{\"location\":\"..\"}" }} │
  │      ↓                                   │
  │ Application extracts name + arguments    │
  │      ↓                                   │
  │ Application validates & executes the     │
  │ matching Python function                 │
  │      ↓                                   │
  │ Function geocodes the location and       │
  │ calls the Open-Meteo forecast API        │
  │      ↓                                   │
  │ Structured result sent back to the LLM   │
  │ as a {"role": "tool", ...} message        │
  │      ↓                                   │
  │ LLM reads the result and generates the   │
  │ final natural-language answer            │
  └───────────────────────────────────────────┘
      ↓
Answer shown to the user
```

This lifecycle is implemented in `src/orchestrator.py::Orchestrator.ask()`, which loops
between the LLM and the tool registry until the LLM stops requesting tools (or a small
round limit is hit, as a safety valve against a misbehaving model).

## Error Handling

The app handles, without crashing:

- **Missing `GROQ_API_KEY`** — clear message at startup, exits before making any request.
- **Empty user input** — prompts the user again rather than sending a blank question.
- **Weather API / geocoding failures** (network errors, HTTP errors, unknown locations,
  malformed responses) — raised as `WeatherServiceError` / `LocationNotFoundError`,
  caught by the orchestrator, and reported back to the LLM as a tool-result error so it
  can respond sensibly (e.g. "I couldn't retrieve the weather right now").
- **Unknown tool requested by the LLM** — caught as `UnknownToolError`, reported back
  to the LLM as a tool-result error rather than raising an unhandled exception.
- **Malformed/incomplete tool arguments** (including non-JSON `arguments` strings from
  the model) — validated before the tool is called; mismatches raise `ToolArgumentError`
  and are reported back the same way.
- **LLM/network failures** — wrapped as `LLMClientError`, surfaced to the user as a
  clear message instead of a raw stack trace.

## Testing

Run the full suite with:

```bash
pytest
```

**18 tests, all passing** (mocked HTTP/LLM — no network access or API key required to run them):

- `tests/test_weather_tool.py` (7 tests) — valid location, unresolvable location,
  geocoding network failure, forecast HTTP error, malformed forecast payload, empty
  input, unrecognized weather code.
- `tests/test_tool_dispatch.py` (6 tests) — `get_weather` is correctly routed by name,
  unknown tool names raise `UnknownToolError`, missing/unexpected arguments raise
  `ToolArgumentError`, non-dict arguments are rejected, and the schema itself is
  checked for the expected OpenAI-style shape.
- `tests/test_orchestrator.py` (5 tests) — a fake LLM client is used to drive the full
  loop end-to-end: a plain question never triggers a tool call; a weather question
  triggers a `tool_calls` → tool execution → tool-result round trip; an unknown tool
  request and malformed tool-call arguments are both reported back to the LLM instead
  of crashing; and the loop gives up gracefully if the model keeps requesting tools
  past `MAX_TOOL_ROUNDS`.

## Design Decisions

- **LLM provider — Groq.** Chosen specifically because it's genuinely free (no credit
  card, no trial period to expire), fast, and its Chat Completions API is
  OpenAI-compatible with solid native tool/function-calling support
  (`qwen/qwen3.6-27b` is used by default). The client is isolated in
  `src/llm/client.py` behind a single `create_message()` method, so swapping providers
  later means writing a new class with the same method, not touching `orchestrator.py`.
- **Weather API — Open-Meteo.** No API key is needed for geocoding or forecasts,
  which keeps setup friction at zero for a demo whose point is tool calling, not
  weather-data sourcing. Geocoding (city name → coordinates) is handled entirely
  inside `tools/weather.py`, so the LLM only ever needs to supply a plain location
  string.
- **Project structure** — a flat `src/llm` / `src/tools` split with a single
  `orchestrator.py` tying them together. No classes or abstractions beyond what's
  needed to keep the LLM, the tool registry, and the tool implementation
  independently testable and swappable.
- **Tool schema** — a single `get_weather(location: str)` tool with one required
  string parameter, in the OpenAI-compatible `{"type": "function", "function": {...}}`
  format Groq expects. This keeps the "LLM decides to call a tool" mechanism easy to
  follow, while still requiring the LLM to do real extraction/normalization work
  (e.g. turning "Hyderabad, India" or "the weather back home" into a location string).
- **Error handling** — tool-level failures (bad location, API errors, malformed
  arguments) are turned into `{"role": "tool", ...}` messages containing a structured
  `{"error": ...}` payload and handed back to the LLM, so the model can react in
  natural language, rather than being treated as fatal exceptions. Only genuinely
  unrecoverable failures (missing config, LLM unreachable) stop the program, and
  always with a clear, specific message.

## A Note on Verification in This Environment

This app was built and tested in a sandboxed environment whose network egress
allowlist does **not** include `api.groq.com` or `open-meteo.com`, and which has no
real `GROQ_API_KEY` configured. Concretely, that means:

- All 18 automated tests pass using mocked HTTP/LLM calls (no network dependency).
- `python main.py` was run end-to-end with a placeholder key to confirm the app
  correctly builds a request to `https://api.groq.com/openai/v1/chat/completions` and
  that request/response failures (including this sandbox's own egress block, which
  surfaces as an HTTP error) are caught by `LLMClientError`/`OrchestratorError` and
  reported cleanly instead of crashing.
- A real, successful LLM tool-calling round trip against Groq, and a real Open-Meteo
  lookup, could not be exercised from inside this sandbox, since outbound requests to
  both hosts are blocked here and no valid Groq key is available.

Once you add your own free `GROQ_API_KEY` to `.env` and run `python main.py` on a
machine with normal internet access, the full live flow (LLM → tool call → Open-Meteo
→ final answer) will work end-to-end as described above.
