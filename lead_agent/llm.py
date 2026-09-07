"""
LLM provider for the Lead Response Agent.

Uses Llama through Groq for tasks that require language understanding
or generation. Deterministic logic remains outside this file.

If GROQ_API_KEY is not available, or the live call fails, the supplied
mock_fn is used so the application can still run locally/offline.

call_structured(prompt, schema, mock_fn, tool_name=..., max_tokens=800)
Every node in graph.py builds one big self-contained prompt string
(lead + history + qualification + rules all inlined) and calls this
with a JSON schema the model must fill via Groq's function-calling.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PROVIDER = "groq"

# Treat an unset OR placeholder key (e.g. left over from a .env.example)
# as "no key" so the app falls back to mock mode instead of trying a
# doomed live call.
_PLACEHOLDER_VALUES = {"", "your_actual_groq_key", "changeme", "sk-..."}
_HAS_KEY = bool(GROQ_API_KEY) and GROQ_API_KEY not in _PLACEHOLDER_VALUES


def _call_groq(prompt, schema, tool_name, max_tokens=800):
    """Call Llama through Groq and return structured JSON."""
    if not _HAS_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (or is a placeholder)")

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a structured-output component of a real-estate "
                "lead-response system. Respond ONLY via the provided tool call, "
                "filling every required field. Do not include any text outside "
                "the tool call."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Return structured output for {tool_name}.",
            "parameters": schema,
        },
    }]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": tool_name}},
        max_tokens=max_tokens,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError("Groq did not return a structured tool call")

    return json.loads(tool_calls[0].function.arguments)


def call_structured(prompt, schema, mock_fn, tool_name="structured_output", max_tokens=800):
    """
    Call the LLM for structured output.

    If no API key exists, or the live call fails, use mock_fn() so the
    application remains runnable. Every result gets a "_mode" field so
    the evaluator panel can show whether a turn was live or mocked.
    """
    if not _HAS_KEY:
        result = mock_fn()
        result.setdefault("_mode", "mock (GROQ_API_KEY not set)")
        return result

    try:
        result = _call_groq(prompt, schema, tool_name, max_tokens)
        result.setdefault("_mode", f"live ({PROVIDER})")
        return result
    except Exception as exc:
        result = mock_fn()
        result.setdefault("_mode", f"mock (fallback: {type(exc).__name__})")
        return result


def current_mode_label():
    """Return the current LLM execution mode, for the sidebar."""
    if _HAS_KEY:
        return f"{PROVIDER} (live)"
    return f"{PROVIDER} (mock -- no valid GROQ_API_KEY set)"
