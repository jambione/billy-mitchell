"""Thin client for the local LM Studio server (OpenAI-compatible).

Chat completions drive Billy and the Coach; embeddings power the knowledge base. Kept to
`requests` + stdlib so it runs on Python 3.14 with no compiled deps. JSON helpers are
defensive because small local models love to wrap JSON in prose or code fences.
"""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from . import config

_session = requests.Session()


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict[str, str]], *, model: str | None = None,
         temperature: float = 0.6, max_tokens: int = 512,
         response_json: bool = False) -> str:
    """Return the assistant message text for a chat completion."""
    body: dict[str, Any] = {
        "model": model or config.CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_json:
        # LM Studio's current API wants json_schema (not json_object). A loose object schema
        # nudges the model toward valid JSON without constraining Billy's fields; the
        # _extract_json parser is the backstop for models that still add prose.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "strict": False, "schema": {"type": "object"}},
        }
    try:
        return _post_chat(body)
    except requests.HTTPError as e:
        # Some runtimes (notably the MLX engine) reject response_format with a 400. Drop it
        # and retry on the prompt + _extract_json alone so Billy keeps thinking.
        status = getattr(e.response, "status_code", None)
        if response_json and status == 400 and "response_format" in body:
            body.pop("response_format")
            try:
                return _post_chat(body)
            except (requests.RequestException, KeyError, ValueError) as e2:
                raise LLMError(f"chat completion failed (no structured output): {e2}") from e2
        raise LLMError(f"chat completion failed: {e}") from e
    except (requests.RequestException, KeyError, ValueError) as e:
        raise LLMError(f"chat completion failed: {e}") from e


def _post_chat(body: dict[str, Any]) -> str:
    r = _session.post(f"{config.LMSTUDIO_BASE_URL}/chat/completions",
                      json=body, timeout=config.LLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_json(messages: list[dict[str, str]], *, model: str | None = None,
              temperature: float = 0.4, max_tokens: int = 512) -> dict[str, Any]:
    """Chat and parse a JSON object out of the reply (tolerant of fences/prose)."""
    text = chat(messages, model=model, temperature=temperature,
                max_tokens=max_tokens, response_json=True)
    return _extract_json(text)


def embed(text: str, *, model: str | None = None) -> list[float]:
    """Return an embedding vector for a string."""
    try:
        r = _session.post(f"{config.LMSTUDIO_BASE_URL}/embeddings",
                          json={"model": model or config.EMBED_MODEL, "input": text},
                          timeout=config.LLM_TIMEOUT_S)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except (requests.RequestException, KeyError, ValueError, IndexError) as e:
        raise LLMError(f"embedding failed: {e}") from e


def health() -> bool:
    """Quick check that LM Studio is reachable (used by run.py preflight)."""
    try:
        r = _session.get(f"{config.LMSTUDIO_BASE_URL}/models", timeout=5)
        return r.ok
    except requests.RequestException:
        return False


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"could not parse JSON from model reply: {e}\n---\n{text}") from e
    raise LLMError(f"no JSON object in model reply:\n{text}")
