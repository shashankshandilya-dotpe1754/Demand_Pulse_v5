"""
ml/llm.py
=========
Thin, provider-agnostic LLM client for the two places DemandPulse uses one:

  * ml/event_ingest.py — reads messy public sources and proposes structured events
  * ml/chat_agent.py   — answers merchant questions by CALLING the forecast model

Both are *language* jobs. Neither is allowed to produce a demand number: the
LightGBM models do that, and the LLM only routes, extracts and narrates. That
separation is the whole design — an LLM asked to guess a forecast will happily
invent a confident one.

Keys
----
Read from the environment or Streamlit secrets, never from the repo:

    ANTHROPIC_API_KEY=sk-ant-...        # provider "anthropic" (default)
    OPENAI_API_KEY=sk-...               # provider "openai"

On Streamlit Community Cloud put them in Settings → Secrets, e.g.

    ANTHROPIC_API_KEY = "sk-ant-..."

`available()` returns False when no key is configured; every caller degrades
gracefully rather than crashing the dashboard.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ml.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o",
}


# ── Key discovery ─────────────────────────────────────────────────────────────
def _secret(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v:
        return v
    try:                                        # Streamlit secrets, if running in it
        import streamlit as st
        return st.secrets.get(name)             # type: ignore[attr-defined]
    except Exception:
        return None


def provider() -> str:
    p = (LLM_PROVIDER or "anthropic").lower()
    if p == "anthropic" and not _secret("ANTHROPIC_API_KEY") and _secret("OPENAI_API_KEY"):
        return "openai"                         # fall back to whichever key exists
    return p


def available() -> bool:
    key = "ANTHROPIC_API_KEY" if provider() == "anthropic" else "OPENAI_API_KEY"
    return bool(_secret(key))


def status() -> dict:
    return {
        "available": available(),
        "provider": provider(),
        "model": LLM_MODEL or DEFAULT_MODELS.get(provider()),
        "hint": None if available() else
                f"Set {'ANTHROPIC_API_KEY' if provider() == 'anthropic' else 'OPENAI_API_KEY'} "
                f"in the environment or Streamlit secrets to enable LLM features.",
    }


class LLMUnavailable(RuntimeError):
    pass


# ── Unified call ──────────────────────────────────────────────────────────────
def chat(messages: list, system: str = "", tools: list = None,
         max_tokens: int = LLM_MAX_TOKENS, temperature: float = 0.0) -> dict:
    """One turn of conversation, normalised across providers.

    `messages`: [{"role": "user"|"assistant", "content": str | list}]
    `tools`:    provider-neutral specs — [{"name", "description", "input_schema"}]

    Returns {"text": str, "tool_calls": [{"id", "name", "input"}], "raw": ...}
    so callers never branch on the provider.
    """
    if not available():
        raise LLMUnavailable(status()["hint"])
    p = provider()
    model = LLM_MODEL or DEFAULT_MODELS[p]
    if p == "anthropic":
        return _anthropic(messages, system, tools, model, max_tokens, temperature)
    return _openai(messages, system, tools, model, max_tokens, temperature)


def _anthropic(messages, system, tools, model, max_tokens, temperature) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                  messages=messages)
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [
            {"name": t["name"], "description": t["description"],
             "input_schema": t["input_schema"]} for t in tools
        ]
    resp = client.messages.create(**kwargs)
    text, calls = "", []
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            calls.append({"id": block.id, "name": block.name, "input": block.input})
    return {"text": text, "tool_calls": calls, "raw": resp,
            "stop_reason": resp.stop_reason}


def _openai(messages, system, tools, model, max_tokens, temperature) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=_secret("OPENAI_API_KEY"))
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    kwargs = dict(model=model, messages=msgs, max_tokens=max_tokens,
                  temperature=temperature)
    if tools:
        kwargs["tools"] = [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}} for t in tools
        ]
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    calls = [
        {"id": tc.id, "name": tc.function.name,
         "input": json.loads(tc.function.arguments or "{}")}
        for tc in (msg.tool_calls or [])
    ]
    return {"text": msg.content or "", "tool_calls": calls, "raw": resp,
            "stop_reason": resp.choices[0].finish_reason}


def complete_json(prompt: str, system: str = "", max_tokens: int = LLM_MAX_TOKENS):
    """Ask for JSON and parse it, tolerating ```json fences."""
    out = chat([{"role": "user", "content": prompt}], system=system,
               max_tokens=max_tokens)
    txt = out["text"].strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        txt = txt[4:] if txt.startswith("json") else txt
    start = min([i for i in (txt.find("["), txt.find("{")) if i != -1] or [0])
    return json.loads(txt[start:])
